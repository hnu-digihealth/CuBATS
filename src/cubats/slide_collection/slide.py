# Standard Library
import os
import concurrent.futures
import pickle

# Third Party
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
from tqdm import tqdm
from PIL import Image
from pyvips import BandFormat
from pyvips import Image as VipsImage

# CuBATS
from cubats.slide_collection import tile_processing
from cubats import Utils as utils


class Slide(object):
    def __init__(self, name, path, is_mask=False, is_reference=False):
        """
        Initialize a Slide object.

        Args:
            name (str): The name of the slide.
            path (str): The path to the slide file.
            is_mask (bool, optional): Whether the slide is a mask. Defaults to False.
            is_reference (bool, optional): Whether the slide is a reference slide. Defaults to False.
        """
        self.name = name
        self.openslide_object = openslide.OpenSlide(path)
        self.tiles = DeepZoomGenerator(
            self.openslide_object, tile_size=1024, overlap=0, limit_bounds=True)
        self.level_count = self.tiles.level_count
        self.level_dimensions = self.tiles.level_dimensions
        self.tile_count = self.tiles.tile_count

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

    def quantify_slide(self, mask_coordinates, save_dir, save_img=False, img_dir=None):
        """ Quantifies staining intensities for all tiles of this slide.

        This function uses multiprocessing to quantify staining intensities of all tiles for the slide. First Each tile
        is subjected to color deconvolution algotithm and quantification of staining intensities. If save_img = True,
        each tile is save in img_dir after color deconvolution allowing for reconstruction of the slide later on. After
        color deconvolution, each tile is processed as a grayscale image and each pixels staining intensity (0-255) is
        quantified and attributed to one of four zones according to the IHC Profilers algorithm:
            - Zone1: High Positive (0-60)
            - Zone2: Positive (61-120)
            - Zone3: Low Positive (121-180)
            - Zone4: Negative (181-235)
            (Zone5: White Space or Fatty Tissues (236-255), irrelevant for quantification)
        The results for each tile are stored within a dictionary and all dictionaries are accumulated in the nested
        dictionary self.detailed_quantification_results. Ultimately, the results are summarized and stored in self.
        quantification_summary. Both the detailed and summarized results are stored as PICKLE in the save_dir.

        Args:
            mask_coordinates (list): List of xy-coordinates from the maskslide where the mask is positive.
            save_dir (str): Directory to save the results. Usually the slides pickle directory.
            save_img (bool, optional): Whether to save the tiles after color deconvolution. Defaults to False.
                Necessary for reconstruction of the slide.
            img_dir (str, optional): Directory to save the tiles. Must be provided if tiles shall be saved. Defaults to
                None.

        """
        if self.is_mask:
            raise ValueError("Cannot quantify mask slide.")
        elif self.is_reference:
            raise ValueError("Cannot quantify reference slide.")

        # Create directory to save tiles if save_img is True
        if save_img:
            if img_dir is None:
                raise ValueError(
                    "img_dir must be provided if save_img is True.")
            self.dab_tile_dir = img_dir
            os.makedirs(self.dab_tile_dir, exist_ok=True)

        # Creates an iterable containing xy-Tuples for each created tile, the DeepZoomGenerator itself, as well as DAB_tile directory.
        iterable = [
            (
                x,
                y,
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

        # Multiprocessing of tiles using concurrent.futures, gathering results and adding them to dictionary in linear manner.
        with concurrent.futures.ProcessPoolExecutor() as exe:
            results = tqdm(
                exe.map(tile_processing.quantify_tile, iterable),
                total=len(iterable),
                desc="Processing image: " + self.name,
            )
            for idx, res in enumerate(results):
                # if result is not None:
                self.detailed_quantification_results[idx] = res

        # Retrieve Quantification results and save to disk
        print("\n==== Saving results\n")

        self.summarize_quantification_results()

        # Save dictionary as pickle
        f_out = os.path.join(
            save_dir, f"{self.name}_processing_info.pickle")
        pickle.dump(self.detailed_quantification_results, open(f_out, "wb"))
        print("\n==== Finished processing image: " + self.name + "\n")

    def summarize_quantification_results(self):
        """
            Summarizes quantification results for a given slide and appends them to self.quantification_results. This includes the sums of number
            if pixels in for zone, percentage of pixels in each zone, as well as a score for each zone. The score is calculated as follows:

            _dict:
                - Slide (str): Name of the slide
                - High Positive (float): Percentage of pixels in the high positive zone
                - Positive (float): Percentage of pixels in the positive zone
                - Low Positive (float): Percentage of pixels in the low positive zone
                - Negative (float): Percentage of pixels in the negative zone
                - White Space or Fatty Tissues (float): Percentage of pixels in the white space or fatty tissues zone
                - Unit (str): Unit of the percentages (%)
                - Score (str): Overall score of the slide based on the zones. However, the score for the entire slide may be misleading since
                    much negative tissue may lead to a negative score even though the slide may contain a lot of positive tissue as well.
                    Therefore, the score for the entire slide should be interpreted with caution.
        """
        if self.is_mask:
            raise ValueError(
                "Cannot summarize quantification results for mask slide.")
        elif self.is_reference:
            raise ValueError(
                "Cannot summarize quantification results for reference slide.")

        # Init variables:
        # sum_z1: sum of pixels in high positive zone
        # sum_z2: sum of pixels in positive zone
        # sum_z3: sum of pixels in low positive zone
        # sum_z4: sum of pixels in negative zone
        # white: sum of pixels in white space or fatty tissues zone
        # count: total number of pixels in the slide
        sum_z1, sum_z2, sum_z3, sum_z4, white, count = (
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0,
        )

        # Iterate through tiles_dict and sum up number of pixels in each zone as well as the total number of pixels
        for i in self.detailed_quantification_results:
            if self.detailed_quantification_results[i]["Flag"] == 1:
                sum_z1 += self.detailed_quantification_results[i]["Zones"][0]
                sum_z2 += self.detailed_quantification_results[i]["Zones"][1]
                sum_z3 += self.detailed_quantification_results[i]["Zones"][2]
                sum_z4 += self.detailed_quantification_results[i]["Zones"][3]
                white += self.detailed_quantification_results[i]["Zones"][4]
                count += self.detailed_quantification_results[i]["Px_count"]

        # Calculate percentages and scores
        zones = [sum_z1, sum_z2, sum_z3, sum_z4, white]
        percentages = [0] * 5
        scores = [0] * 5
        for i in range(len(zones)):
            percentages[i] = (zones[i] / count) * 100
            scores[i] = (zones[i] * (len(zones) - i)) / count

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
        self.quantification_summary["Score"] = utils.get_score_name(
            scores)  # TODO necessary?

    def reconstruct_slide(self, in_path, out_path):
        """
        Reconstructs a slide into a WSI based on saved tiles. This is only possible if tiles have been saved during processing. The WSI is then saved as .tif in the directory passed by out_path.

        Args:
            in_path (str): Path to saved tiles
            out_path (str): Path where to save the reconstructed slide.

        """
        # Init variables
        counter = 0
        cols, rows = self.tiles.level_tiles[self.level_count - 1]
        row_array = []
        # Iterate through tiles and append tiles for each column and row. Previously not processed tiles are replaced by white tiles.
        for row in tqdm(range(rows), desc="Reconstructing image: " + self.name):
            column_array = []
            for col in range(cols):
                tile_name = str(col) + "_" + str(row)
                file = os.path.join(in_path, tile_name + ".tif")
                if os.path.exists(file):
                    img = Image.open(file)
                    counter += 1
                else:
                    img = Image.new("RGB", (1024, 1024), (255, 255, 255))

                column_array.append(img)

            segmented_row = np.concatenate(column_array, axis=1)
            row_array.append(segmented_row)

        # Create WSI and save as pyramidal TIF in self.reconstruct_dir
        segmented_wsi = np.concatenate(row_array, axis=0)
        segmented_wsi = VipsImage.new_from_array(
            segmented_wsi).cast(BandFormat.INT)
        out = os.path.join(out_path, self.name + "_reconst.tif")
        segmented_wsi.crop(0, 0, self.openslide_object.dimensions[0], self.openslide_object.dimensions[1]).tiffsave(
            out,
            tile=True,
            compression="jpeg",
            bigtiff=True,
            pyramid=True,
            tile_width=256,
            tile_height=256,
        )
