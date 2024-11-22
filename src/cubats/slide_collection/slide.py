# Standard Library
import concurrent.futures
import logging
import os
import pickle
from time import time

# Third Party
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from pyvips import BandFormat
from pyvips import Image as VipsImage
from tqdm import tqdm

# CuBATS
import cubats.logging_config as log_config
from cubats.slide_collection import tile_processing


class Slide(object):
    """ Slide Class.

    'Slide' class instatiates a slide object containing all relevant information and results for a single slide. All
     slide specific operations that can be performed on a single slide rather than on a collection of slides are
     implemented in this class. This includes quantification of staining intensities and reconstruction of a slide. The
     class is initialized with the name of the slide, the path to the slide file, as well as information on whether the
     slide is a mask or reference slide. The class contains a dictionary of detailed quantification results for each
     tile, as well as a dictionary of summarized quantification results for the entire slide. The slide object also
     contains information on the OpenSlide object, the tiles, the level count, the level dimensions, as well as the
     tile count. The slide object also contains a directory to save the tiles after color deconvolution, which is
     necessary for reconstruction of the slide later on.

    Attributes:
        name (str): The name of the slide.

        openslide_object (openslide.OpenSlide): The openslide object of the slide.

        tiles (openslide.deepzoom.DeepZoomGenerator): DeepZoom Generator containing the tiles of the slide.

        level_count (int): The number of DeepZoom levels of the slide.

        level_dimensions (list):  List of tuples (pixels_x, pixels_y) for each Deep Zoom level.

        tile_count (int): The number of tiles in the slide.

        dab_tile_dir (str): Directory to save the tiles after color deconvolution is applied. Necessary for
            reconstruction of the slide. If save_img is False no tiles are saved and this attribute is None.

        is_mask (bool): Whether the slide is the slide_collections mask slide.

        is_reference (bool): Whether the slide is the slide_collections reference slide.

        detailed_quantification_results (dict): Dictionary containing detailed quantification results for each tile of
            the slide. The dictionary is structured as follows:

            - key (int): Index of the tile.
            - value (dict): Dictionary containing the following:

                - Flag (int): Flag indicating whether the tile was processed (1) or not (0).
                - Histogram (array): Array containing the histogram of the tile.
                - Hist_centers (array): Array containing the centers of the histogram bins.
                - Zones (array): Array containing the number of pixels in each zone sorted by index according to the
                  following attribution High Positive, Positive, Low Positive, Negative, Background).
                - Score (str): Score of the tile based on the zones.

    """

    def __init__(self, name, path, is_mask=False, is_reference=False):
        """
        Initialize a Slide object.

        Args:
            name (str): The name of the slide.
            path (str): The path to the slide file.
            is_mask (bool, optional): Whether the slide is a mask. Defaults to False.
            is_reference (bool, optional): Whether the slide is a reference slide. Defaults to False.
        """
        # Initialize logger
        logging.config.dictConfig(log_config.LOGGING)
        self.logger = logging.getLogger(__name__)

        self.name = name
        self.openslide_object = openslide.OpenSlide(path)
        self.tiles = DeepZoomGenerator(
            self.openslide_object, tile_size=1024, overlap=0, limit_bounds=True)
        self.level_count = self.tiles.level_count
        self.level_dimensions = self.tiles.level_dimensions
        self.tile_count = self.tiles.tile_count
        self.masked_tiles = None

        self.dab_tile_dir = None

        self.is_mask = is_mask
        self.is_reference = is_reference

        self.detailed_quantification_results = {}
        self.quantification_summary = {}

        self.properties = {
            "name": self.name,
            "reference": self.is_reference,
            "mask": self.is_mask,
            "openslide_object": self.openslide_object,
            "tiles": self.tiles,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "tile_count": self.tile_count,
        }
        self.logger.debug(f"Slide {self.name} initialized.")

    def quantify_slide(self, mask_coordinates, save_dir, save_img=False, img_dir=None, detailed_mask=None):
        """ Quantifies staining intensities for all tiles of this slide.

        This function uses multiprocessing to quantify staining intensities of all tiles for the slide.

        Each tile undergoes color deconvolution followed by staining intensity quantification based on the IHC
        Profiler's algorithm. If `save_img` is True, tiles are saved in `img_dir` after deconvolution for later
        reconstruction. After color deconvolution, each tile is processed as a grayscale image, and each pixel's
        staining intensity (0-255) is quantified and categorized into zones:

            - Zone1: High Positive (0-60)
            - Zone2: Positive (61-120)
            - Zone3: Low Positive (121-180)
            - Zone4: Negative (181-235)
            - (Zone5: White Space or Fatty Tissues (236-255), irrelevant for quantification)

        Results are stored in `self.detailed_quantification_results` and summarized in `self.quantification_summary`.
        Both are saved as PICKLE files in `save_dir`.

        Args:
            mask_coordinates (list): List of xy-coordinates from the maskslide where the mask is positive.
            save_dir (str): Directory to save the results. Usually the slides pickle directory.
            save_img (bool, optional): Whether to save the tiles after color deconvolution. Defaults to False.
                Necessary for reconstruction of the slide.
            img_dir (str, optional): Directory to save the tiles. Must be provided if tiles shall be saved. Defaults to
                None.
            detailed_mask (openslide.deepzoom.DeepZoomGenerator, optional): DeepZoomGenerator containing the detailed
                mask. Defaults to None. Provides a more detailed mask for the quantification of the slide, however,
                might result in larger inaccuracies for WSI with low congruence.

        """
        start_time = time()
        self.logger.debug(
            f"Quantifying slide: {self.name}, save_img: {save_img}, detailed_mode: {detailed_mask is not None}")
        if self.is_mask:
            self.logger.error("Cannot quantify mask slide.")
            raise ValueError("Cannot quantify mask slide.")
        elif self.is_reference:
            self.logger.error("Cannot quantify reference slide.")
            raise ValueError("Cannot quantify reference slide.")

        # Create directory to save tiles if save_img is True
        if save_img:
            if img_dir is None:
                self.logger.error(
                    "img_dir must be provided if save_img is True.")
                raise ValueError(
                    "img_dir must be provided if save_img is True.")
            self.dab_tile_dir = img_dir
            os.makedirs(self.dab_tile_dir, exist_ok=True)

        # Creates an iterable containing xy-Tuples for each tile, DeepZoomGenerator, and directory.
        iterable = [
            (
                x,
                y,
                tile_processing.mask_tile(self.tiles.get_tile(
                    self.level_count - 1, (x, y)), detailed_mask.get_tile(self.level_count - 1, (x, y)))
                if detailed_mask is not None
                else
                self.tiles.get_tile(self.level_count - 1, (x, y)),
                self.dab_tile_dir,
                save_img,
            )
            for x, y in tqdm(
                mask_coordinates,
                desc="Pre-processing image: ",
                total=len(mask_coordinates),
            )
        ]

        # Multiprocessing using concurrent.futures, gathering results and adding them to dictionary in linear manner.
        with concurrent.futures.ProcessPoolExecutor() as exe:
            results = tqdm(
                exe.map(tile_processing.quantify_tile, iterable),
                total=len(iterable),
                desc="Processing image: " + self.name,
            )
            for idx, res in enumerate(results):
                # if result is not None:
                self.detailed_quantification_results[idx] = res
        end_time = time()
        self.logger.info(
            f"Finished quantifying slide: {self.name} in {round((end_time - start_time)/60,2)} minutes.")

        # Retrieve Quantification results and save to disk
        self.summarize_quantification_results()
        self.logger.info(
            f"Saving quantification results for {self.name} to {save_dir}")
        # Save dictionary as pickle
        start_time_save = time()
        f_out = os.path.join(
            save_dir, f"{self.name}_processing_info.pickle")
        pickle.dump(self.detailed_quantification_results, open(f_out, "wb"))
        end_time_save = time()
        self.logger.debug(
            f"Saved quantification results for {self.name} to {f_out} in \
                {round((end_time_save - start_time_save)/60,2)} minutes.")
        self.logger.info(f"Finished processing slide: {self.name}")

    def summarize_quantification_results(self):
        """
        Summarizes quantification results.

        Summarizes quantification results for a given slide and appends them to `self.quantification_summary`. This
        includes the total number of pixels in each zone, the percentage of pixels in each zone, and a score for each
        zone.

        The summary contains the following keys:
            - Slide (str): Name of the slide.
            - High Positive (float): Percentage of pixels in the high positive zone.
            - Positive (float): Percentage of pixels in the positive zone.
            - Low Positive (float): Percentage of pixels in the low positive zone.
            - Negative (float): Percentage of pixels in the negative zone.
            - Background (float): Percentage of pixels in the white space background or fatty tissues.
            - Score (str): Overall score of the slide based on the zones. However, the score for the entire slide
              may be misleading since much negative tissue may lead to a negative score even though the slide may
              contain a lot of positive tissue as well. Therefore, the score for the entire slide should be
              interpreted with caution.

        Raises:
            ValueError: If the slide is a mask slide or a reference slide.
        """
        start_time_summarize = time()
        self.logger.info(
            f"Summarizing quantification results for slide: {self.name}")
        if self.is_mask:
            self.logger.error(
                "Cannot summarize quantification results for mask slide.")
            raise ValueError(
                "Cannot summarize quantification results for mask slide.")
        elif self.is_reference:
            self.logger.error(
                "Cannot summarize quantification results for reference slide.")
            raise ValueError(
                "Cannot summarize quantification results for reference slide.")

        # Init variables:
        # sum_z1: sum of pixels in high positive zone
        # sum_z2: sum of pixels in positive zone
        # sum_z3: sum of pixels in low positive zone
        # sum_z4: sum of pixels in negative zone
        # white: sum of pixels in white space or fatty tissues zone
        # count: total number of pixels in the slide
        sum_z1, sum_z2, sum_z3, sum_z4, background, count = (
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0,
        )

        # Score counts
        score_counts = {
            "High Positive": 0,
            "Positive": 0,
            "Low Positive": 0,
            "Negative": 0,
            "Background": 0
        }
        # Iterate through tiles_dict and sum up number of pixels in each zone as well as the total number of pixels
        for i in self.detailed_quantification_results:
            if self.detailed_quantification_results[i]["Flag"] == 1:
                sum_z1 += self.detailed_quantification_results[i]["Zones"][0]
                sum_z2 += self.detailed_quantification_results[i]["Zones"][1]
                sum_z3 += self.detailed_quantification_results[i]["Zones"][2]
                sum_z4 += self.detailed_quantification_results[i]["Zones"][3]
                background += self.detailed_quantification_results[i]["Zones"][4]
                count += self.detailed_quantification_results[i]["Px_count"]
                score_name = self.detailed_quantification_results[i]["Score"]
                score_counts[score_name] += 1

        # Calculate percentages and scores
        zones = [sum_z1, sum_z2, sum_z3, sum_z4, background]
        percentages = [0] * 5
        for i in range(len(zones)):
            percentages[i] = (zones[i] / count) * 100

        # Calculate the summary score for the slide using weights
        weights = {
            "High Positive": 4,
            "Positive": 3,
            "Low Positive": 2,
            "Negative": 1,
            "Background": 0
        }

        total_processed_tiles = sum(score_counts.values())
        weighted_scores = [score_counts[score] * weights[score]
                           / total_processed_tiles for score in score_counts]
        max_score_index = np.argmax(weighted_scores[:4])  # Exclude Background
        zone_names = ["High Positive", "Positive",
                      "Low Positive", "Negative", "Background"]
        slide_score = zone_names[max_score_index]

        # Create dictionary and append to quantification_results
        self.quantification_summary["Name"] = self.name
        self.quantification_summary["High Positive (%)"] = round(
            percentages[0], 2)
        self.quantification_summary["Positive (%)"] = round(percentages[1], 2)
        self.quantification_summary["Low Positive (%)"] = round(
            percentages[2], 2)
        self.quantification_summary["Negative (%)"] = round(percentages[3], 2)
        self.quantification_summary["Background (%)"] = round(
            percentages[4], 2)
        self.quantification_summary["Score"] = slide_score
        end_time_summarize = time()
        self.logger.debug(
            f"Finished summarizing quantification results for slide: {self.name} in \
                {round((end_time_summarize - start_time_summarize)/60,2)} minutes.")

    def reconstruct_slide(self, in_path, out_path):
        """
        Reconstructs a slide into a Whole Slide Image (WSI) based on saved tiles. This is only possible if tiles have
        been saved during processing. The WSI is then saved as .tif in the specified `out_path`.

        Args:
            in_path (str): Path to saved tiles
            out_path (str): Path where to save the reconstructed slide.

        """
        start_time = time()
        # Init variables
        counter = 0
        cols, rows = self.tiles.level_tiles[self.level_count - 1]
        row_array = []
        # append tiles for each column and row. Previously not processed tiles are replaced by white tiles.
        for row in tqdm(range(rows), desc="Reconstructing slide: " + self.name):
            column_array = []
            for col in range(cols):
                tile_name = str(col) + "_" + str(row)
                file = os.path.join(in_path, tile_name + ".tif")
                if os.path.exists(file):
                    img = Image.open(file)
                    counter += 1
                else:
                    img = Image.new("RGB", (1024, 1024), (192, 192, 192))

                column_array.append(img)

            segmented_row = np.concatenate(column_array, axis=1)
            row_array.append(segmented_row)

        # Create WSI and save as pyramidal TIF in self.reconstruct_dir
        logging.getLogger('pyvips').setLevel(logging.WARNING)
        segmented_wsi = np.concatenate(row_array, axis=0)
        segmented_wsi = VipsImage.new_from_array(
            segmented_wsi).cast(BandFormat.INT)
        end_time = time()
        self.logger.info(
            f"Finished reconstructing slide: {self.name} in {round((end_time - start_time)/60,2)} minutes.")

        start_time_save = time()
        out = os.path.join(out_path, self.name + "_reconst.tif")
        self.logger.info(f"Saving reconstructed slide to {out}")
        segmented_wsi.crop(0, 0, self.openslide_object.dimensions[0], self.openslide_object.dimensions[1]).tiffsave(
            out,
            tile=True,
            compression="jpeg",
            bigtiff=True,
            pyramid=True,
            tile_width=256,
            tile_height=256,
        )
        end_time_save = time()
        self.logger.debug(
            f"Saved reconstructed slide to {out} in {round((end_time_save - start_time_save)/60,2)} minutes.")
