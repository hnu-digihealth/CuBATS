# Standard Library
import concurrent.futures
import os
import pickle
import re
from itertools import combinations

# Third Party
import numpy as np
import pandas as pd
import skimage
import cubats.Utils as utils
from matplotlib import pyplot as plt
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from pyvips import BandFormat
from pyvips import Image as VipsImage
from tqdm import tqdm

OPENSLIDE_PATH = r"c:\Users\mlnot\anaconda3\envs\DoctorThesis\Lib\site-packages\openslide\openslide-win64-20231011\openslide-win64-20231011\bin"

if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        # Third Party
        import openslide
else:
    # Third Party
    import openslide


# Destination directories
RESULT_DATA_DIR = "data"
TILES_DIR = "tiles"
COLOCALIZATION = "colocalization"
ORIGINAL_TILES_DIR = "original"  # TODO: remove if no further use
DAB_TILE_DIR = "dab"
E_TILE_DIR = "eosin"  # TODO: remove if no further use
H_TILE_DIR = "hematoxylin"  # TODO: remove if no further use
PICKLE_DIR = "pickle"
RECONSTRUCT_DIR = "reconstructed_slides"


DEFAULT_TILE_SIZE = 1024


class Slide(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, src_dir, dest_dir, ref_slide=None):
        """
        Initializes the class. The class contains

        Args:
            src_dir (String): Path to src directory containing the WSIs
            dest_dir (String): Path to destination directory for results
            ref_slide (String, optional): Path to reference slide. If ref_slide is None it will be automatically set. Defaults to None.

        Class Variables:

        - name (String): name of parent directory (should be name of tumorset )

        - src_dir (String): Path to source directory containing the WSIs
        - dest_dir (String): Path to destination directory for results
        - data_dir (String): Path to data directory. Inside the data directory quantification summary, dual overlap summary and triplet
            overlap summary are stored. In addition it contains the pickle subdirectory.
        - pickle_dir (String): Path to pickle directory. Inside the pickle directory pickled copies are stored which can later be reloaded for future re-/processing.
        - tiles_dir (String): Path to tiles directory. Inside the tiles directory the tile directories for each slide are stored.
        - colocalization_dir (String): Path to colocalization directory. Inside the colocalization directory results of dual and triplet overlap analyses are stored.
        - reconstruct_dir (String): Path to reconstruct directory. Inside the reconstruct directory reconstructed slides are stored.

        - tile_dir_list (List): List containing the paths to the tile directories for each slide. Inside these directories tile images of the respective slide are stored.
        - dab_tile_dir_list (List): List containing the paths to the DAB tile directories for each slide. Inside these directories DAB tile images of the respective slide are stored.

        - quantification_results_list (List): List containing the results of the quantification for each slide. The list is nested and contains a dictionary for each slide.
            This dictionary contains another dictionary containing the results for each tile of the slide.

        - orig_img_list (List): List containing the original file names of the WSIs
        - img_names (List): List containing the clear names of the WSIs

        - mask_coordinates (List): List containing the tile coordinates for tiles that are covered by the mask. Coordinates are tuples (column, row).

        - quantification_summary (List): List containing a summary of the quantification results pf all slides.
            - Slide (String): Name of the slide
            - High Positive (float): Percentage of pixels in the high positive zone
            - Positive (float): Percentage of pixels in the positive zone
            - Low Positive (float): Percentage of pixels in the low positive zone
            - Negative (float): Percentage of pixels in the negative zone
            - White Space or Fatty Tissues (float): Percentage of pixels in the white space or fatty tissues zone
            - Unit (String): Unit of the percentages (%)
            - Score (String): Overall score of the slide based on the zones. However, the score for the entire slide may be misleading since

        - dual_overlap_summary (List): List containing a summary of the dual overlap results for all processed analyses.
            - Slide 1 (String): Name of the first slide
            - Slide 2 (String): Name of the second slide
            - Total Coverage (float): Combined coverage of the two slides
            - Total Overlap (float): Overlap of antigen expression in the two slides
            - Total Complement (float): Complementary antigen expressions in the two slides
            - Total Negative (float): Total of Negative in the two slides
            - Error (float): Percentage of tiles that were not processed due to insufficient tissue coverage
            - Unit (String): Unit of the percentages (%)

        - triplet_overlap_summary (List): List containing a summary of the triplet overlap results for all processed analyses.
            - Slide 1 (String): Name of the first slide
            - Slide 2 (String): Name of the second slide
            - Slide 3 (String): Name of the third slide
            - Total Coverage (float): Combined coverage of the three slides
            - Total Overlap (float): Overlap of antigen expression in the three slides
            - Total Complement (float): Complementary antigen expressions in the three slides
            - Total Negative (float): Total of Negative in the three slides
            - Error (float): Percentage of tiles that were not processed due to insufficient tissue coverage
            - Unit (String): Unit of the percentages (%)

        - slide_info_dict (Dict): Dictionary containing information on all slides.
            - name (String): Name of the slide
            - openslide_object (OpenSlide Object): OpenSlide Object of the slide
            - tiles (DeepZoomGenerator): DeepZoomGenerator wrapping the OpenSlide Object
            - tiles_count (int): Total number of tiles
            - level_dimensions (list): Dimensions of each DeepZoom level
            - total_count_tiles (int): Total number of tiles
        - ref_slide (Dict): Dictionary containing information on the reference slide
        - mask (Dict): Dictionary containing information on the mask slide

        """
        # Name of the tumorset
        self.name = os.path.split(os.path.split(
            os.path.split(src_dir)[0])[0])[1]

        # Directories
        self.src_dir = src_dir
        self.dest_dir = (
            # os.path.join(dest_dir, self.name) TODO how to handle this?
            dest_dir
        )
        self.data_dir = None
        self.pickle_dir = None
        self.tiles_dir = None
        self.colocalization_dir = None
        self.reconstruct_dir = None
        self.tile_dir_list = []
        self.dab_tile_dir_list = []

        self.orig_img_list = []
        self.img_names = []

        # Slide informations
        self.slide_info_dict = {}

        # Mask Variables
        self.mask_coordinates = []
        # Quantification Variables
        self.quantification_results_list = []
        self.quantification_summary = []

        # Antigen Expression Variables
        self.dual_overlap_summary = []
        self.triplet_overlap_summary = []

        # Init images
        self.init_img_list()

        # Set destination directories
        self.set_dst_dir()

        # Load previous results if exist
        self.load_previous_results()

        # Set slide information, reference slide and mask
        self.set_slide_info_dict()

        if ref_slide:
            self.ref_slide = ref_slide
        else:
            self.set_ref_slide()

        self.set_mask()

    def init_img_list(self):
        """
        Initializes a list containing the names of all images in the source directory. Original file names are stored in self.original_img_list.
        Clear names are appended to self.img_names.
        """
        full_path_list = [
            os.path.join(self.src_dir, f) for f in os.listdir(self.src_dir)
        ]

        for f in full_path_list:
            if os.path.isfile(f):
                filename = utils.get_name(f)
                # Necessary since server has hidden files
                if not filename.startswith(".") and f.endswith(".tiff"):
                    self.img_names.append(utils.get_name(f))
                    self.orig_img_list.append(f)

    def set_dst_dir(self):
        """Assign and initiate needed directories if they do not exist yet."""
        # Data dir
        self.data_dir = os.path.join(self.dest_dir, RESULT_DATA_DIR)
        os.makedirs(self.data_dir, exist_ok=True)
        # pickle dir
        self.pickle_dir = os.path.join(self.data_dir, PICKLE_DIR)
        os.makedirs(self.pickle_dir, exist_ok=True)
        # Tiles dir
        self.tiles_dir = os.path.join(self.dest_dir, TILES_DIR)
        os.makedirs(self.tiles_dir, exist_ok=True)
        # Colocalization dir
        self.colocalization_dir = os.path.join(self.dest_dir, COLOCALIZATION)
        os.makedirs(self.colocalization_dir, exist_ok=True)
        # Reconstructed dir
        self.reconstruct_dir = os.path.join(self.dest_dir, RECONSTRUCT_DIR)
        os.makedirs(self.reconstruct_dir, exist_ok=True)

        # Create subdirectories in tiles_dir for each slide except for the reference slide and the mask
        for f in self.orig_img_list:
            fname = utils.get_name(f)
            if re.search("HE", fname) or re.search("segmented", fname):
                slide_dir = os.path.join(self.tiles_dir, fname)
                self.tile_dir_list.append(slide_dir)
            else:
                slide_dir = os.path.join(self.tiles_dir, fname)
                self.tile_dir_list.append(slide_dir)
                os.makedirs(slide_dir, exist_ok=True)
                dab_dir = os.path.join(slide_dir, DAB_TILE_DIR)
                self.dab_tile_dir_list.append(dab_dir)

    def get_slide_dict(self):
        """
        Returns a nested dictionary containing information on all slides.

        Returns:
            Nested dict: Dict containing dicts for each slide
        """
        return self.slide_info_dict

    def set_ref_slide(self):
        """
        Sets the reference slide. The reference slide is used to generate the tumor mask.
        """
        key = "HE"
        for i, e in enumerate(self.img_names):
            if re.search(key, e):
                self.ref_slide = self.slide_info_dict[i]
                break

    def set_mask(self):
        """
        Sets the mask slide. The mask slide is used to define the region of interest for analysis.
        """
        search_key = "segmented"
        for i, e in enumerate(self.img_names):
            if re.search(search_key, e):
                self.mask = self.slide_info_dict[i]
                break

    def get_ref_slide(self):
        """
        Returns the reference slide.

        Returns:
            dict: dict containing information on the reference slide
        """
        return self.ref_slide

    def set_slide_info_dict(self):
        """
        Creates a dictionary containing information on all slides. The Information is stored in self.all_slides_dict.

        __dict__:
            - name (String): Name of the slide
            - openslide_object (OpenSlide Object): OpenSlide Object of the slide
            - tiles (DeepZoomGenerator): DeepZoomGenerator wrapping the OpenSlide Object
            - tiles_count (int): Total number of tiles
            - level_dimensions (list): Dimensions of each DeepZoom level
            - total_count_tiles (int): Total number of tiles
        """
        # Iterate through list of image names and summarized information for each slide into dict.
        for idx, f in enumerate(self.orig_img_list):
            openslide_obj = openslide.OpenSlide(f)
            tiles = self.generate_tiles(openslide_obj)
            __dict__ = {}
            __dict__["name"] = utils.get_name(f)
            __dict__["openslide_object"] = openslide_obj
            __dict__["tiles"] = tiles
            __dict__["tiles_count"] = tiles.level_count
            __dict__["level_dimensions"] = tiles.level_dimensions
            __dict__["total_count_tiles"] = tiles.tile_count
            self.slide_info_dict[idx] = __dict__

    def load_previous_results(self, path=None):
        """
        Loads results from previous processing. If no path is given, the objects pickle directory is used.
        OpenSlide objects cannot be saved as pickle as they are C-types. Therefore, they are initiatited separately in the set_all_slides_dict function.

        The following files are tried to be loaded if they exist in the given directory:
            - mask_coordinates.pickle: Load mask coordinates from previous mask generation
            - quantification_results.pickle: Load quantification results from previous processing
            - dual_overlap_results.pickle: Load dual antigen overlap results from previous processing
            - triplet_overlap_results.pickle: Load triplet antigen overlap results from previous processing
            - processing_info.pickle: Load processing information for each slide from previous processing

        Args:
            path (String): Path to directory containing pickle files.
        """
        print("\n==== Searching for previous results\n")
        if self.quantification_results_list.__len__() == 0:
            if path is None:
                path = self.pickle_dir
            mask = os.path.join(path, "mask_coordinates.pickle")
            quant_res = os.path.join(path, "quantification_results.pickle")
            dual_overlap_res = os.path.join(
                path, "dual_overlap_results.pickle")
            triplet_overlap_res = os.path.join(
                path, "triplet_overlap_results.pickle")

            # load mask coordinates
            if os.path.exists(mask):
                print("\n==== Loading mask coordinates\n")
                self.mask_coordinates = pickle.load(open(mask, "rb"))

            # load quantification results
            if os.path.exists(quant_res):
                print("\n==== Loading quantification results\n")
                self.quantification_summary = pickle.load(
                    open(quant_res, "rb"))

            # load dual overlap results
            if os.path.exists(dual_overlap_res):
                print("\n==== Loading dual overlap results\n")
                self.dual_overlap_summary = pickle.load(
                    open(dual_overlap_res, "rb"))

            # load triplet overlap results
            if os.path.exists(triplet_overlap_res):
                print("\n==== Loading triplet overlap results\n")
                self.triplet_overlap_summary = pickle.load(
                    open(triplet_overlap_res, "rb")
                )

            # Load processing info for each slide
            for file in sorted(os.listdir(path)):
                if re.search("processing_info", os.path.basename(file)):
                    print("\n==== Loading", file)
                    file = os.path.join(path, file)
                    tiles = pickle.load(open(file, "rb"))
                    filename = os.path.splitext(os.path.basename(file))[0].replace(
                        "_processing_info", ""
                    )
                    self.quantification_results_list.append((filename, tiles))
                else:
                    pass

    def generate_mask(self):
        """
        Generates a list containing coordinates of tiles that are part of the mask. This allows to only process
        tiles that are part of the mask and thus are relevant for analysis. Previous mask coordinates are cleared.
        """
        mask = self.mask["tiles"]
        self.mask_coordinates.clear()
        cols, rows = mask.level_tiles[mask.level_count - 1]
        for col in tqdm(range(cols), desc="Initializing Mask"):
            for row in range(rows):
                temp = mask.get_tile(mask.level_count - 1, (col, row))
                if not (temp.mode == "RGB"):
                    temp_rgb = temp.convert("RBG")
                    temp_np = np.array(temp_rgb)
                else:
                    temp_np = np.array(temp)

                # tilename = str(col) + "_" + str(row)
                if temp_np.mean() < 230:
                    self.mask_coordinates.append((col, row))
                    """ TODO: Think about saving mask tiles, only for display purposes
                    img = Image.fromarray(temp_np)
                    dir = r"C:/Users/mlnot/Desktop/Images/saved _tiles/N_2016_001260/tiles/mask"
                    path  = f"{dir}/{tilename}.tif"
                    img.save(path)"""
                else:
                    pass

        # Save mask coordinates as pickle
        out = os.path.join(self.pickle_dir, "mask_coordinates.pickle")
        pickle.dump(self.mask_coordinates, open(out, "wb"))

    def generate_tiles(
        self, slide, tile_size=DEFAULT_TILE_SIZE, overlap=0, limit_bounds=True
    ):
        """
        Creates a DeepZoomGenerator from an OpenSlide object. All created tiles are stored in a list, the list ist returned.

        Args:
            slide (OpenSlide Object): OpenSlide Object to be converted to DeepZoom Object
            tile_size (int, optional): Width and height of single tile. Defaults to DEFAULT_TILE_SIZE (=1024).
            overlap (int, optional): The number of pixels added to each interior edge of a tile Defaults to 0.
            limit_bounds (bool, optional): to render only non-empty slide region. Defaults to True.

        Returns:
            DeepZoomGenerator: DeepZoomGenerator wrapping the OpenSlide Object
        """
        tiles = DeepZoomGenerator(
            slide, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds
        )
        return tiles

    def quantify_all_slides(self, save_images=False):
        """
        Quantifies all slides that were instantiated successively with the exception of the reference slide and the mask. Results are stored as CSV to DATADIR.
        If slides have previously been quantified results stored in the object are reset and overwritten.
        The functio

        """

        # Reset previous results if exist.
        if self.quantification_results_list.__len__() != 0:
            self.quantification_results_list.clear()
        if self.quantification_summary.__len__() != 0:
            self.quantification_summary.clear()

        # iterate through list of image names and process each slide except for the reference slide and the mask
        for i, img in enumerate(self.img_names):
            if img == self.ref_slide["name"]:
                print("\n==== Not processing reference slide: " + img + "\n")
                pass
            elif img == self.mask["name"]:
                print("\n==== Not processing mask: " + img + "\n")
                pass
            else:
                print(
                    "Analyzing Slide: "
                    + img
                    + "("
                    + str(i + 1)
                    + "/"
                    + str(len(self.img_names))
                    + ")\n"
                )
                self.quantify_single_slide(img, save_images)
        # Get quantification results and save them at the end of quantification
        if not self.quantification_summary:
            self.gather_quantification_results()

        self.save_quantification_results()

    def quantify_single_slide(self, slide_name, save_img=False):
        """
        This function processes all tiles for a given slide. It retrieves the DeepZoomGenerator for the passed
        slide_name and quantifies all tiles using multiprocessing. Results are stored as PICKLE.

        Processed tiles undergo color separation and are stored in in the DAB_tile_dir of the respective slide directory.
        Quantification results are appended tiles_dict_list. The results for the entire slide are saved as PICKLE in the pickle_dir
        (<slide_name>_processing_info.pickle) to allow reloading the results for further processing in the future.
        The results include the entire quantification dictionary for the slide, whereas quantication results of the process_all_slides
        function only contain a summary for all quantified slides.

        Args:
            - slide_name (String): Name of the slide to be processed.

        tiles_dict:
            - Tilename (String): col_row of tile
            - Histogram (array): actual histogram of the tile
            - Hist_centers (array): center of bins of histogram
            - Zones (array): Number of pixels for each zone. During processing pixels in the tile are assigned to one of four
                     zones based on pixel intensity. For more information see utils.calculate_pixel_intensity
            - Percentage (array): Percentage of pixels in each zone
            - Score (array): Score for each of the ties
            - Px_count: Total number of pixels in the tile
            - Image Array (array): Pixelvalues of for positive pixels. Pixels with values ranging from 0 to 121 are considered positive.
            - Flag (int): 1: if Tile is processed, 0: if Tile is not processed because it does not contain sufficient tissue

        """

        # Retrieves slide with given name from slides_dict
        tiles = None
        idx = 0
        for i in self.slide_info_dict:
            if self.slide_info_dict[i]["name"] == slide_name:
                tiles = self.slide_info_dict[i]["tiles"]
                idx = i
                break

        # Create directories for DAB stain if save_img is True
        if save_img:
            dab_tile_dir = os.path.join(self.tile_dir_list[idx], DAB_TILE_DIR)
            self.dab_tile_dir_list.append(dab_tile_dir)
            os.makedirs(dab_tile_dir, exist_ok=True)
        else:
            dab_tile_dir = None

        # Init dictionary containing information on each tile. The dictionary will be nested, since each processed tile will return a dictionary with results
        tiles_dict = {}

        # If mask coordinates are not yet generated, generate them.
        if not self.mask_coordinates:
            self.generate_mask()

        # Creates an iterable containing xy-Tuples for each created tile, the DeepZoomGenerator itself, as well as DAB_tile directory.
        iterable = [
            (
                x,
                y,
                tiles.get_tile(tiles.level_count - 1, (x, y)),
                dab_tile_dir,
                save_img,
            )
            for x, y in tqdm(
                self.mask_coordinates,
                desc="Pre-processing image: ",
                total=len(self.mask_coordinates),
            )
        ]

        # Multiprocessing of tiles using concurrent.futures, gathering results and adding them to dictionary in linear manner.
        with concurrent.futures.ProcessPoolExecutor() as exe:
            results = tqdm(
                exe.map(utils.quantify_single_tile, iterable),
                total=len(iterable),
                desc="Processing image: " + slide_name,
            )
            for idx, res in enumerate(results):
                # if result is not None:
                tiles_dict[idx] = res

        # Retrieve Quantification results and save to disk
        print("\n==== Saving results\n")

        self.get_slide_results(slide_name, tiles_dict)

        # Save dictionary as pickle
        self.quantification_results_list.append((slide_name, tiles_dict))
        f_out = os.path.join(
            self.pickle_dir, f"{slide_name}_processing_info.pickle")
        pickle.dump(tiles_dict, open(f_out, "wb"))
        print("\n==== Finished processing image: " + slide_name + "\n")

    def gather_quantification_results(self):
        """
        Iterates through self.tiles_dict_list and gathers quantification results for each slide.
        """

        for ele in self.quantification_results_list:
            self.get_slide_results(ele[0], ele[1])

    def save_quantification_results(self):
        """
        Saves summay of quantification results as CSV in self.data_dir for analysis, as well as PICKLE for reloading results in the future.
        """
        if self.quantification_summary:
            df = pd.DataFrame.from_dict(self.quantification_summary)
            df.to_csv(
                self.data_dir + "/quantification_results.csv",
                sep=",",
                index=False,
                encoding="utf-8",
            )
            out = os.path.join(
                self.pickle_dir, "quantification_results.pickle")
            pickle.dump(self.quantification_summary, open(out, "wb"))

    def get_slide_results(self, name, tiles_dict):
        """
        Summarizes quantification results for a given slide and appends them to self.quantification_results. This includes the sums of number
        if pixels in for zone, percentage of pixels in each zone, as well as a score for each zone. The score is calculated as follows:

        Args:
            name (_type_): _description_
            tiles_dict (_type_): _description_

        _dict:
            - Slide (String): Name of the slide
            - High Positive (float): Percentage of pixels in the high positive zone
            - Positive (float): Percentage of pixels in the positive zone
            - Low Positive (float): Percentage of pixels in the low positive zone
            - Negative (float): Percentage of pixels in the negative zone
            - White Space or Fatty Tissues (float): Percentage of pixels in the white space or fatty tissues zone
            - Unit (String): Unit of the percentages (%)
            - Score (String): Overall score of the slide based on the zones. However, the score for the entire slide may be misleading since
                much negative tissue may lead to a negative score even though the slide may contain a lot of positive tissue as well.
                Therefore, the score for the entire slide should be interpreted with caution.
        """
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
        for i in tiles_dict:
            if tiles_dict[i]["Flag"] == 1:
                sum_z1 += tiles_dict[i]["Zones"][0]
                sum_z2 += tiles_dict[i]["Zones"][1]
                sum_z3 += tiles_dict[i]["Zones"][2]
                sum_z4 += tiles_dict[i]["Zones"][3]
                white += tiles_dict[i]["Zones"][4]
                count += tiles_dict[i]["Px_count"]

        # Calculate percentages and scores
        zones = [sum_z1, sum_z2, sum_z3, sum_z4, white]
        percentages = [0] * 5
        scores = [0] * 5
        for i in range(len(zones)):
            percentages[i] = (zones[i] / count) * 100
            scores[i] = (zones[i] * (len(zones) - i)) / count

        # Create dictionary and append to quantification_results
        _dict = {}
        _dict["Slide"] = name
        _dict["High Positive"] = round(percentages[0], 2)
        _dict["Positive"] = round(percentages[1], 2)
        _dict["Low Positive"] = round(percentages[2], 2)
        _dict["Negative"] = round(percentages[3], 2)
        _dict["Background/ No Tissue"] = round(percentages[4], 2)
        _dict["Unit"] = "%"
        _dict["Score"] = utils.get_score_name(scores)  # TODO necessary?

        self.quantification_summary.append(_dict)

    def get_dual_antigen_combinations(self):
        """
        Creates all possible combinations of pairs amongst all quantified slides and analyzes antigen expressions for each pair, including antigen overlap.
        Results are stored in self.dual_overlap_results.
        """
        self.dual_overlap_summary.clear()
        antigen_combinations = list(combinations(
            self.quantification_results_list, 2))
        for ele in antigen_combinations:
            self.compute_dual_antigen_combinations(ele[0], ele[1])

    def get_triple_antigen_combinations(self):
        """
        Creates all possible combinations of triplets amongst all quantified slides and analyzes antigen expressions for each triplet, including antigen overlap.
        Results are stored in self.triplet_overlap_results.
        """
        self.triplet_overlap_summary.clear()
        antigen_combinations = list(combinations(
            self.quantification_results_list, 3))
        for ele in antigen_combinations:
            self.compute_triplet_antigen_combinations(ele[0], ele[1], ele[2])

    def compute_dual_antigen_combinations(self, slide1, slide2, save_img=False):
        """
        Analyzes antigenexpressions for each of tiles of the given pair of slides using Multiprocesing. Results from each of the tiles are summarized,
        stored in self.dual_overlap_results and saved as CSV in self.data_dir as well as PICKLE in self.pickle_dir.

        Args:
            slide1 (dict): Quantification results for slide 1
            slide2 (dict): Quantification results for slide 2

        _overlap_dict:
            - Slide 1 (String): Name of the first slide
            - Slide 2 (String): Name of the second slide
            - Total Coverage (float): Combined coverage of the two slides
            - Total Overlap (float): Overlap of antigen expression in the two slides
            - Total Complement (float): Complementary antigen expressions in the two slides
            - Total Negative (float): Total of Negative in the two slides
            - Error (float): Percentage of tiles that were not processed due to insufficient tissue coverage
            - Unit (String): Unit of the percentages (%)


            TODO: saving image optional
        """
        name1 = slide1[0]
        name2 = slide2[0]
        slide1 = slide1[1]
        slide2 = slide2[1]

        # Create directory for pair of slides
        if save_img:
            dirname = os.path.join(
                self.colocalization_dir, (name1 + " and " + name2))
            os.makedirs(dirname, exist_ok=True)
        else:
            dirname = None

        # Create iterable for multiprocessing
        iterable = []
        for i in slide1:
            if slide1[i]["Tilename"] == slide2[i]["Tilename"]:
                img1 = slide1[i]
                img2 = slide2[i]
                iterable.append((img1, img2, dirname, save_img))

        # Init dict for results of each tile
        comparison_dict = {}

        with concurrent.futures.ProcessPoolExecutor() as exe:
            results = tqdm(
                exe.map(utils.compute_dual_antigen_colocalization, iterable),
                total=len(iterable),
                desc="Calculating Coverage of Slide " + name1 + " & " + name2,
            )
            for idx, result in enumerate(results):
                comparison_dict[idx] = result

        # Summarize results
        overlap_dict = {}
        counter = 0
        sum_total_coverage = 0.00
        sum_total_overlap = 0.00
        sum_total_complement = 0.00
        sum_high_overlap = 0.00
        sum_high_complement = 0.00
        sum_pos_overlap = 0.00
        sum_pos_complement = 0.00
        sum_low_overlap = 0.00
        sum_low_complement = 0.00
        sum_negative = 0.00
        sum_background = 0.00
        error1 = 0
        error2 = 0
        error3 = 0

        for i in comparison_dict:
            if comparison_dict[i].get("Flag") == 1:
                counter += 1
                sum_total_coverage += comparison_dict[i]["Total Coverage"]
                sum_total_overlap += comparison_dict[i]["Total Overlap"]
                sum_total_complement += comparison_dict[i]["Total Complement"]
                sum_high_overlap += comparison_dict[i]["High Positive Overlap"]
                sum_high_complement += comparison_dict[i]["High Positive Complement"]
                sum_pos_overlap += comparison_dict[i]["Positive Overlap"]
                sum_pos_complement += comparison_dict[i]["Positive Complement"]
                sum_low_overlap += comparison_dict[i]["Low Positive Overlap"]
                sum_low_complement += comparison_dict[i]["Low Positive Complement"]
                sum_negative += comparison_dict[i]["Negative"]
                sum_background += comparison_dict[i]["Background/No Tissue"]
            elif comparison_dict[i].get("Flag") == 0:
                error1 += 1
            elif comparison_dict[i].get("Flag") == -1:
                error2 += 1
            elif comparison_dict[i].get("Flag") == -2:
                error3 += 1

        sum_total_coverage = sum_total_coverage / counter
        sum_total_overlap = sum_total_overlap / counter
        sum_total_complement = sum_total_complement / counter
        sum_high_overlap = sum_high_overlap / counter
        sum_high_complement = sum_high_complement / counter
        sum_pos_overlap = sum_pos_overlap / counter
        sum_pos_complement = sum_pos_complement / counter
        sum_low_overlap = sum_low_overlap / counter
        sum_low_complement = sum_low_complement / counter
        sum_negative = sum_negative / counter
        sum_background = sum_background / counter
        total_error = error1 + error2 + error3

        overlap_dict["Slide 1"] = name1
        overlap_dict["Slide 2"] = name2
        overlap_dict["Total Coverage"] = round(sum_total_coverage, 2)
        overlap_dict["Total Overlap"] = round(sum_total_overlap, 2)
        overlap_dict["Total Complement"] = round(sum_total_complement, 2)
        overlap_dict["High Positive Overlap"] = round(sum_high_overlap, 2)
        overlap_dict["High Positive Complement"] = round(
            sum_high_complement, 2)
        overlap_dict["Positive Overlap"] = round(sum_pos_overlap, 2)
        overlap_dict["Positive Complement"] = round(sum_pos_complement, 2)
        overlap_dict["Low Positive Overlap"] = round(sum_low_overlap, 2)
        overlap_dict["Low Positive Complement"] = round(sum_low_complement, 2)
        overlap_dict["Negative Tissue"] = round(sum_negative, 2)
        overlap_dict["Background / No Tissue"] = round(sum_background, 2)
        overlap_dict["Total Error"] = round(
            (total_error / comparison_dict.__len__()) * 100, 2)
        overlap_dict["Error1"] = round(
            (error1 / comparison_dict.__len__()) * 100, 2)
        overlap_dict["Error2"] = round(
            (error2 / comparison_dict.__len__()) * 100, 2)
        overlap_dict["Error3"] = round(
            (error3 / comparison_dict.__len__()) * 100, 2)
        overlap_dict["Unit"] = "%"
        self.dual_overlap_summary.append(overlap_dict)

        # Save results as CSV and PICKLE
        overlap_df = pd.DataFrame.from_dict(self.dual_overlap_summary)
        overlap_df.to_csv(
            self.data_dir + "/dual_overlap_results.csv",
            sep=",",
            index=False,
            header=True,
            encoding="utf-8",
        )
        out = os.path.join(self.pickle_dir, "dual_overlap_results.pickle")
        pickle.dump(self.dual_overlap_summary, open(out, "wb"))

    def compute_triplet_antigen_combinations(
        self, slide1, slide2, slide3, save_img=False
    ):
        """
        Analyzes antigenexpressions for each of tiles of the given triplet of slides using Multiprocesing. Results from each of the tiles are summarized,
        stored in self.triplet_overlap_results and saved as CSV in self.data_dir as well as PICKLE in self.pickle_dir.

        Args:
            slide1 (dict): Quantification results for slide 1
            slide2 (dict): Quantification results for slide 2
            slide3 (dict): Quantification results for slide 3

        overlap_dict:
            - Slide 1 (String): Name of the first slide
            - Slide 2 (String): Name of the second slide
            - Slide 3 (String): Name of the third slide
            - Total Coverage (float): Combined coverage of the three slides
            - Total Overlap (float): Overlap of antigen expression in the three slides
            - Total Complement (float): Complementary antigen expressions in the three slides
            - Total Negative (float): Total of Negative in the three slides
            - Error (float): Percentage of tiles that were not processed due to insufficient tissue coverage
            - Unit (String): Unit of the percentages (%)

            TODO: saving image optional
        """
        name1 = slide1[0]
        name2 = slide2[0]
        name3 = slide3[0]
        slide1 = slide1[1]
        slide2 = slide2[1]
        slide3 = slide3[1]

        # Create directory for triplet of slides
        if save_img:
            dirname = os.path.join(
                self.colocalization_dir, (f"{name1} and {name2} and {name3}")
            )
            os.makedirs(dirname, exist_ok=True)
        else:
            dirname = None

        # Create iterable for multiprocessing
        iterable = []
        for i in slide1:
            if (
                slide1[i]["Tilename"] == slide2[i]["Tilename"]
                and slide1[i]["Tilename"] == slide3[i]["Tilename"]
            ):
                img1 = slide1[i]
                img2 = slide2[i]
                img3 = slide3[i]
                iterable.append((img1, img2, img3, dirname, save_img))

        # Init dict for results of each tile
        comparison_dict = {}

        # Process tiles using multiprocessing
        with concurrent.futures.ProcessPoolExecutor() as exe:
            results = tqdm(
                exe.map(utils.compute_triplet_antigen_colocalization, iterable),
                total=len(iterable),
                desc="Calculating Coverage of Slides "
                + name1
                + " & "
                + name2
                + " & "
                + name3,
            )
            for idx, result in enumerate(results):
                comparison_dict[idx] = result

        # Summarize results TODO add amount of low positive
        overlap_dict = {}
        counter = 0
        sum_total_coverage = 0.00
        sum_total_overlap = 0.00
        sum_total_complement = 0.00
        sum_high_overlap = 0.00
        sum_high_complement = 0.00
        sum_pos_overlap = 0.00
        sum_pos_complement = 0.00
        sum_low_overlap = 0.00
        sum_low_complement = 0.00
        sum_negative = 0.00
        sum_background = 0.00
        error1 = 0
        error2 = 0
        error3 = 0

        for i in comparison_dict:
            if comparison_dict[i].get("Flag") == 1:
                counter += 1
                sum_total_coverage += comparison_dict[i]["Total Coverage"]
                sum_total_overlap += comparison_dict[i]["Total Overlap"]
                sum_total_complement += comparison_dict[i]["Total Complement"]
                sum_high_overlap += comparison_dict[i]["High Positive Overlap"]
                sum_high_complement += comparison_dict[i]["High Positive Complement"]
                sum_pos_overlap += comparison_dict[i]["Positive Overlap"]
                sum_pos_complement += comparison_dict[i]["Positive Complement"]
                sum_low_overlap += comparison_dict[i]["Low Positive Overlap"]
                sum_low_complement += comparison_dict[i]["Low Positive Complement"]
                sum_negative += comparison_dict[i]["Negative"]
                sum_background += comparison_dict[i]["Background/No Tissue"]
            elif comparison_dict[i].get("Flag") == 0:
                error1 += 1
            elif comparison_dict[i].get("Flag") == -1:
                error2 += 1
            elif comparison_dict[i].get("Flag") == -2:
                error3 += 1

        sum_total_coverage = sum_total_coverage / counter
        sum_total_overlap = sum_total_overlap / counter
        sum_total_complement = sum_total_complement / counter
        sum_high_overlap = sum_high_overlap / counter
        sum_high_complement = sum_high_complement / counter
        sum_pos_overlap = sum_pos_overlap / counter
        sum_pos_complement = sum_pos_complement / counter
        sum_low_overlap = sum_low_overlap / counter
        sum_low_complement = sum_low_complement / counter
        sum_negative = sum_negative / counter
        sum_background = sum_background / counter
        total_error = error1 + error2 + error3

        overlap_dict["Slide 1"] = name1
        overlap_dict["Slide 2"] = name2
        overlap_dict["Slide 3"] = name3
        overlap_dict["Total Coverage"] = round(sum_total_coverage, 2)
        overlap_dict["Total Overlap"] = round(sum_total_overlap, 2)
        overlap_dict["Total Complement"] = round(sum_total_complement, 2)
        overlap_dict["High Positive Overlap"] = round(sum_high_overlap, 2)
        overlap_dict["High Positive Complement"] = round(
            sum_high_complement, 2)
        overlap_dict["Positive Overlap"] = round(sum_pos_overlap, 2)
        overlap_dict["Positive Complement"] = round(sum_pos_complement, 2)
        overlap_dict["Low Positive Overlap"] = round(sum_low_overlap, 2)
        overlap_dict["Low Positive Complement"] = round(sum_low_complement, 2)
        overlap_dict["Negative Tissue"] = round(sum_negative, 2)
        overlap_dict["Background / No Tissue"] = round(sum_background, 2)
        overlap_dict["Total Error"] = round(
            (total_error / comparison_dict.__len__()) * 100, 2)
        overlap_dict["Error1"] = round(
            (error1 / comparison_dict.__len__()) * 100, 2)
        overlap_dict["Error2"] = round(
            (error2 / comparison_dict.__len__()) * 100, 2)
        overlap_dict["Error3"] = round(
            (error3 / comparison_dict.__len__()) * 100, 2)
        overlap_dict["Unit"] = "%"
        self.triplet_overlap_summary.append(overlap_dict)
        # Save results as CSV and PICKLE
        overlap_df = pd.DataFrame.from_dict(self.triplet_overlap_summary)
        overlap_df.to_csv(
            self.data_dir + "/triplet_overlap_results.csv",
            sep=",",
            index=False,
            header=True,
            encoding="utf-8",
        )
        out = os.path.join(self.pickle_dir, "triplet_overlap_results.pickle")
        pickle.dump(self.triplet_overlap_summary, open(out, "wb"))

    def reconstruct_slide(self, slide_name, input_path):
        """
        Reconstructs a slide into a WSI based on saved tiles. This is only possible if tiles have been saved during processing. The WSI is then stored in the self.reconstruct_dir.

        Args:
            slide_name (String): Name of slide
            input_path (String): Path to saved tiles

        """

        slide = None
        tiles = None

        slide = self.ref_slide["openslide_object"]
        tiles = self.ref_slide["tiles"]

        # Init variables
        counter = 0
        cols, rows = tiles.level_tiles[tiles.level_count - 1]
        row_array = []

        # Iterate through tiles and append tiles for each column and row. Previously not processed tiles are replaced by white tiles.
        for row in tqdm(range(rows), desc="Reconstructing image: " + slide_name):
            column_array = []
            for col in range(cols):
                tile_name = str(col) + "_" + str(row)
                file = os.path.join(input_path, tile_name + ".tif")
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
        out = os.path.join(self.reconstruct_dir, slide_name + "_reconst.tif")
        segmented_wsi.crop(0, 0, slide.dimensions[0], slide.dimensions[1]).tiffsave(
            out,
            tile=True,
            compression="jpeg",
            bigtiff=True,
            pyramid=True,
            tile_width=256,
            tile_height=256,
        )

    def plot_tile_quantification_results(self, tilename, img_true=True, numeric=True):
        """ This function plots quantification results of a given tilename. It plots the DAB-image, the histogram of the intensity distribution,
            a bar plot containing the amount of pixels attributed to each zone and numeric results. Display of the image and numeric results are
            optional, default is set to True. Images can only be display if they have been saved during quantification in functions:
            - quantify_all_slides
            - quantify_single_slide

        Args:
            tilename (_type_): _description_
            img_true (bool, optional): _description_. Defaults to True.
            numeric (bool, optional): _description_. Defaults to True.
        """
        names = []
        images = []
        hists = []
        hist_centers = []
        zones = []
        percentages = []
        scores = []
        px_count = []
        img_idx = 0
        hist_idx = 1
        bar_idx = 2
        numeric_idx = 3

        if img_true and numeric:
            for dir in self.dab_tile_dir_list:
                file = os.path.join(dir, (tilename + "_DAB.tif"))
                if os.path.exists(file):
                    img = skimage.io.imread(file)
                    images.append(img)
            fig, ax = plt.subplots(
                4,
                self.quantification_results_list.__len__(),
                figsize=(self.quantification_results_list.__len__() * 5, 18),
            )
        elif img_true and not numeric:
            for dir in self.dab_tile_dir_list:
                file = os.path.join(dir, (tilename + "_DAB.tif"))
                if os.path.exists(file):
                    img = skimage.io.imread(file)
                    images.append(img)
            fig, ax = plt.subplots(
                3,
                self.quantification_results_list.__len__(),
                figsize=(self.quantification_results_list.__len__() * 5, 13),
            )
        elif not img_true and numeric:
            fig, ax = plt.subplots(
                3,
                self.quantification_results_list.__len__(),
                figsize=(self.quantification_results_list.__len__() * 5, 13),
            )
            hist_idx = 0
            bar_idx = 1
            numeric_idx = 2

        else:
            fig, ax = plt.subplots(
                2,
                self.quantification_results_list.__len__(),
                figsize=(self.quantification_results_list.__len__() * 5, 9),
            )
            hist_idx = 0
            bar_idx = 1

        tile_exists = False

        for i in range(self.quantification_results_list.__len__()):
            for j in range(self.quantification_results_list[i][1].__len__()):
                if self.quantification_results_list[i][1][j]["Tilename"] == tilename:
                    names.append(self.quantification_results_list[i][0])
                    hists.append(
                        self.quantification_results_list[i][1][j]["Histogram"])
                    hist_centers.append(
                        self.quantification_results_list[i][1][j]["Hist_centers"]
                    )
                    zones.append(
                        self.quantification_results_list[i][1][j]["Zones"])
                    percentages.append(
                        self.quantification_results_list[i][1][j]["Percentage"]
                    )
                    scores.append(
                        utils.get_score_name(
                            self.quantification_results_list[i][1][j]["Score"].tolist(
                            )
                        )
                    )
                    px_count.append(
                        self.quantification_results_list[i][1][j]["Px_count"])
                    tile_exists = True
                    break

        assert tile_exists, "The given tilename does not exist for one or more of the slides. Please make sure to select an existing tilename."

        max_y_hist = round(max([max(hist) for hist in hists]), -4) + 10000
        max_y_zone = round(max([max(zone) for zone in zones[:4]]), -4) + 20000

        for i in range(self.quantification_results_list.__len__()):
            if img_true:
                ax[img_idx, i].imshow(images[i])
                ax[img_idx, i].set_title("Slide: " + names[i])
                ax[img_idx, i].axis("off")
            # Histogram
            ax[hist_idx, i].plot(hist_centers[i], hists[i], lw=2)
            ax[hist_idx, i].set_title("Quantification: " + names[i])
            ax[hist_idx, i].set_xlabel("Pixel Intensity")
            ax[hist_idx, i].set_ylabel("Number of Pixels")
            ax[hist_idx, i].set_xlim([0, 255])
            ax[hist_idx, i].set_ylim([0, max_y_hist])
            ax[hist_idx, i].axvline(x=60, color="r", linestyle="--")
            ax[hist_idx, i].axvline(x=120, color="r", linestyle="--")
            ax[hist_idx, i].axvline(x=180, color="r", linestyle="--")
            ax[hist_idx, i].axvline(x=235, color="r", linestyle="--")
            # Bar Plot
            ax[bar_idx, i].bar(
                ["High Positive", "Positive", "Low Positive", "Negative"], zones[i][:4]
            )
            ax[bar_idx, i].set_title("Scoring: " + names[i])
            ax[bar_idx, i].set_ylim([0, max_y_zone])
            ax[bar_idx, i].set_ylabel("Number of Pixels")
            # Numeric Results
            if numeric:
                rows = [
                    "High Positive",
                    "Positive",
                    "Low Positive",
                    "Negative",
                    "Total Pixel Count",
                    "Score",
                ]
                cell_text = [
                    [str(round(percentages[i][0], 2)) + "%"],
                    [str(round(percentages[i][1], 2)) + "%"],
                    [str(round(percentages[i][2], 2)) + "%"],
                    [str(round(percentages[i][3], 2)) + "%"],
                    [px_count[i]],
                    [scores[i]],
                ]
                ax[numeric_idx, i].axis("off")
                ax[numeric_idx, i].table(
                    cellText=cell_text,
                    colWidths=[0.5] * 3,
                    rowLabels=rows,
                    loc="best",
                )
                ax[numeric_idx, i].set_title("Numeric Results: " + names[i])

        fig.suptitle(f"Quantification Results for Tile: {tilename}\n", fontsize=16)
        fig.tight_layout()
        plt.show()
