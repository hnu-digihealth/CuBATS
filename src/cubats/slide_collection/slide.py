# Standard Library
import concurrent.futures
import logging
import os
import pickle
from time import time

# Third Party
import cupy as cp
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
    """Slide Class.

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
            self.openslide_object, tile_size=1024, overlap=0, limit_bounds=True
        )
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

    def quantify_slide(
        self,
        mask_coordinates,
        save_dir,
        save_img=False,
        img_dir=None,
        detailed_mask=None,
        gpu_acceleration=False,
    ):
        """Quantifies staining intensities for all tiles of this slide.

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
            gpu_acceleration (bool): Boolean determining whether GPU acceleration is applied. If the device is capable
                of GPU acceleration this will automatically be set by the SlideCollection class and passed by the
                calling function.

        """
        self.logger.debug(
            f"Quantifying slide: {self.name}, save_img: {save_img}, \
                detailed_mode: {detailed_mask is not None}, gpu_acceleration {gpu_acceleration}"
        )
        if self.is_mask:
            self.logger.error("Cannot quantify mask slide.")
            raise ValueError("Cannot quantify mask slide.")
        elif self.is_reference:
            self.logger.error("Cannot quantify reference slide.")
            raise ValueError("Cannot quantify reference slide.")

        # Create directory to save tiles if save_img is True
        if save_img:
            if img_dir is None:
                self.logger.error("img_dir must be provided if save_img is True.")
                raise ValueError("img_dir must be provided if save_img is True.")
            self.dab_tile_dir = img_dir
            os.makedirs(self.dab_tile_dir, exist_ok=True)

        start_time_preprocessing = time()
        # Creates an iterable containing xy-Tuples for each tile, DeepZoomGenerator, and directory.
        iterable = [
            (
                x,
                y,
                (
                    tile_processing.mask_tile(
                        self.tiles.get_tile(self.level_count - 1, (x, y)),
                        detailed_mask.get_tile(self.level_count - 1, (x, y)),
                    )
                    if detailed_mask is not None
                    else self.tiles.get_tile(self.level_count - 1, (x, y))
                ),
                self.dab_tile_dir,
                save_img,
                gpu_acceleration,
            )
            for x, y in tqdm(
                mask_coordinates,
                desc="Pre-processing slide: " + self.name,
                total=len(mask_coordinates),
            )
        ]
        # iterable = [
        #     (x, y, tile, mask_px, self.dab_tile_dir, save_img, gpu_acceleration)
        #     for x, y in tqdm(
        #         mask_coordinates,
        #         desc="Pre-processing  Slide",
        #         total=len(mask_coordinates),
        #     )
        #     for tile, mask_px in [
        #         (
        #             tile_processing.mask_tile(
        #                 self.tiles.get_tile(self.level_count - 1, (x, y)),
        #                 detailed_mask.get_tile(self.level_count - 1, (x, y)),
        #             )
        #             if detailed_mask is not None
        #             else (self.tiles.get_tile(self.level_count - 1, (x, y)), 1048576)
        #         )
        #     ]
        # ]
        end_time_preprocessing = time()
        if end_time_preprocessing - start_time_preprocessing >= 60:
            self.logger.info(
                f"Finished pre-processing slide: {self.name} in \
                    {round((end_time_preprocessing - start_time_preprocessing)/60,2)} minutes."
            )
        else:
            self.logger.info(
                f"Finished pre-processing slide: {self.name} in \
                    {round((end_time_preprocessing - start_time_preprocessing),2)} seconds."
            )

        start_time_quantification = time()
        # Multiprocessing using concurrent.futures, gathering results and adding them to dictionary in linear manner.
        with concurrent.futures.ProcessPoolExecutor() as exe:
            results = tqdm(
                exe.map(tile_processing.quantify_tile, iterable),
                total=len(iterable),
                desc="Processing slide: " + self.name,
            )
            for idx, res in enumerate(results):
                # if result is not None:
                self.detailed_quantification_results[idx] = res
        end_time_quantification = time()
        if end_time_quantification - start_time_quantification >= 60:
            self.logger.info(
                f"Finished quantifying slide: {self.name} in \
                    {round((end_time_quantification - start_time_quantification)/60,2)} minutes."
            )
        else:
            self.logger.info(
                f"Finished quantifying slide: {self.name} in \
                    {round((end_time_quantification - start_time_quantification),2)} seconds."
            )

        # Retrieve Quantification results and save to disk
        self.summarize_quantification_results(gpu_acceleration)

        # Save dictionary as pickle
        start_time_save = time()
        f_out = os.path.join(save_dir, f"{self.name}_processing_info.pickle")
        self.logger.info(f"Saving quantification results for {self.name} to {f_out}")
        with open(f_out, "wb") as f:
            pickle.dump(
                self.detailed_quantification_results,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        end_time_save = time()
        self.logger.debug(
            f"Saved quantification results for {self.name} to {f_out} in \
                {round((end_time_save - start_time_save)/60,2)} minutes."
        )
        self.logger.info(f"Finished processing slide: {self.name}")

    def summarize_quantification_results(self, gpu_acceleration=False):
        """
        Summarizes quantification results.

        Summarizes quantification results for a given slide and appends them to `self.quantification_summary`. This
        includes the total number of pixels in each zone, the percentage of pixels in each zone, and a score for each
        zone.

        The summary contains the following keys:
            - Slide (str): Name of the slide.
            - Coverage (float): Tumor coverage of the antigen in the slide.
            - High Positive (float): Percentage of pixels in the high positive zone.
            - Positive (float): Percentage of pixels in the positive zone.
            - Low Positive (float): Percentage of pixels in the low positive zone.
            - Negative (float): Percentage of pixels in the negative zone.
            - Total Tissue (float): Total amount of tissue in the slide.
            - Background (float): Percentage of pixels in the white space background or fatty tissues.
            - H-Score (float): H-score calculation based on positive pixels.
            - Score (str): Overall score of the slide based on the zones. However, the score for the entire slide
              may be misleading since much negative tissue may lead to a negative score even though the slide may
              contain a lot of positive tissue as well. Therefore, the score for the entire slide should be
              interpreted with caution.
            - Total Processed Tiles (float): Percentage of total processed tiles.
            - Error (float): Percentage of tiles that were not processed as they did not contain sufficient tissue.

        Raises:
            ValueError: If the slide is a mask slide or a reference slide.
        """
        start_time_summarize = time()
        self.logger.info(f"Summarizing quantification results for slide: {self.name}")
        if self.is_mask:
            self.logger.error("Cannot summarize quantification results for mask slide.")
            raise ValueError("Cannot summarize quantification results for mask slide.")
        elif self.is_reference:
            self.logger.error(
                "Cannot summarize quantification results for reference slide."
            )
            raise ValueError(
                "Cannot summarize quantification results for reference slide."
            )

        # Init variables:
        # sum_z1: sum of pixels in high positive zone
        # sum_z2: sum of pixels in positive zone
        # sum_z3: sum of pixels in low positive zone
        # sum_z4: sum of pixels in negative zone
        # white: sum of pixels in white space or fatty tissues zone
        # count: total number of pixels in the slide
        (
            sum_high_positive,
            sum_positive,
            sum_low_positive,
            sum_negative,
            sum_background,
            sum_tissue_count,
            error,
            processed_tiles,
        ) = (
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0,
            0,
        )

        # Score counts
        score_counts = {
            "High Positive": 0,
            "Positive": 0,
            "Low Positive": 0,
            "Negative": 0,
            "Background": 0,
        }
        # Iterate through tiles_dict and sum up number of pixels in each zone as well as the total number of pixels
        for i in self.detailed_quantification_results:
            if self.detailed_quantification_results[i]["Flag"] == 1:
                processed_tiles += 1
                sum_high_positive += self.detailed_quantification_results[i]["Zones"][0]
                sum_positive += self.detailed_quantification_results[i]["Zones"][1]
                sum_low_positive += self.detailed_quantification_results[i]["Zones"][2]
                sum_negative += self.detailed_quantification_results[i]["Zones"][3]
                sum_background += self.detailed_quantification_results[i]["Zones"][4]
                sum_tissue_count += self.detailed_quantification_results[i][
                    "Px_count"
                ]  # Can be removed later
                score_name = self.detailed_quantification_results[i]["Score"]
                score_counts[score_name] += 1
            elif self.detailed_quantification_results[i]["Flag"] == 0:
                error += 1

        # Calculate percentages and scores
        if gpu_acceleration:
            zones = cp.array(
                [
                    sum_high_positive,
                    sum_positive,
                    sum_low_positive,
                    sum_negative,
                    sum_background,
                ]
            )
            total_pixel_count = cp.sum(zones)
            perc_high_positive = (zones[0] / sum_tissue_count) * 100
            perc_positive = (zones[1] / sum_tissue_count) * 100
            perc_low_positive = (zones[2] / sum_tissue_count) * 100
            perc_negative = (zones[3] / sum_tissue_count) * 100
            perc_background = (zones[4] / total_pixel_count) * 100
            perc_coverage = perc_high_positive + perc_positive + perc_low_positive
            perc_tissue = (sum_tissue_count / total_pixel_count) * 100
            h_score = (
                (perc_high_positive * 3) + (perc_positive * 2) + (perc_low_positive * 1)
            )
            # Calculate the summary score for the slide using weights
            weights_score = cp.array([4, 3, 2, 1, 0])
            total_processed_tiles = sum(score_counts.values())
            percentages = cp.array(
                [perc_high_positive, perc_positive, perc_low_positive, perc_negative]
            )
            if cp.any(percentages > 66.6):
                max_score_index = int(cp.argmax(percentages))  # Exclude Background
            else:
                scores = zones * weights_score / total_processed_tiles
                max_score_index = int(cp.argmax(scores[:4]))  # Exclude Background
        else:
            zones = np.array(
                [
                    sum_high_positive,
                    sum_positive,
                    sum_low_positive,
                    sum_negative,
                    sum_background,
                ]
            )
            total_pixel_count = np.sum(zones)
            perc_high_positive = (zones[0] / sum_tissue_count) * 100
            perc_positive = (zones[1] / sum_tissue_count) * 100
            perc_low_positive = (zones[2] / sum_tissue_count) * 100
            perc_negative = (zones[3] / sum_tissue_count) * 100
            perc_background = (zones[4] / total_pixel_count) * 100
            perc_coverage = perc_high_positive + perc_positive + perc_low_positive
            perc_tissue = (sum_tissue_count / total_pixel_count) * 100
            h_score = (
                (perc_high_positive * 3) + (perc_positive * 2) + (perc_low_positive * 1)
            )
            weights_h_score = [3, 2, 1, 0, 0]
            h_score = np.sum(percentages[:4] * weights_h_score[:4]) / 100
            # Calculate the summary score for the slide using weights
            weights_score = np.array([4, 3, 2, 1, 0])
            total_processed_tiles = sum(score_counts.values())
            if np.any(percentages > 66.6):
                max_score_index = int(np.argmax(percentages[:4]))  # Exclude Background
            else:
                scores = zones * weights_score / total_processed_tiles
                max_score_index = int(np.argmax(scores[:4]))  # Exclude Background

        # Calculate percentage of processed tiles and error
        perc_processed_tiles = (
            processed_tiles / len(self.detailed_quantification_results)
        ) * 100
        perc_error = (error / len(self.detailed_quantification_results)) * 100

        # Get Score of entire slide
        zone_names = [
            "High Positive",
            "Positive",
            "Low Positive",
            "Negative",
            "Background",
        ]
        slide_score = zone_names[max_score_index]

        # Update the dictionary
        self.quantification_summary = {
            "Name": self.name,
            "Coverage (%)": round(float(perc_coverage), 4),
            "High Positive (%)": round(float(perc_high_positive), 4),
            "Positive (%)": round(float(perc_positive), 4),
            "Low Positive (%)": round(float(perc_low_positive), 4),
            "Negative (%)": round(float(perc_negative), 4),
            "Total Tissue (%)": round(float(perc_tissue), 4),
            "Background / No Tissue (%)": round(float(perc_background), 4),
            "H-Score": round(float(h_score), 2),
            "Score": slide_score,
            "Total Processed Tiles (%)": round(float(perc_processed_tiles), 4),
            "Error (%)": round(float(perc_error), 4),
        }
        end_time_summarize = time()
        self.logger.debug(
            f"Finished summarizing quantification results for slide: {self.name} in \
                {round((end_time_summarize - start_time_summarize)/60,2)} minutes."
        )

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
        logging.getLogger("pyvips").setLevel(logging.WARNING)
        segmented_wsi = np.concatenate(row_array, axis=0)
        segmented_wsi = VipsImage.new_from_array(segmented_wsi).cast(BandFormat.INT)
        end_time = time()
        self.logger.info(
            f"Finished reconstructing slide: {self.name} in {round((end_time - start_time)/60,2)} minutes."
        )

        start_time_save = time()
        out = os.path.join(out_path, self.name + "_reconst.tif")
        self.logger.info(f"Saving reconstructed slide to {out}")
        segmented_wsi.crop(
            0,
            0,
            self.openslide_object.dimensions[0],
            self.openslide_object.dimensions[1],
        ).tiffsave(
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
            f"Saved reconstructed slide to {out} in {round((end_time_save - start_time_save)/60,2)} minutes."
        )
