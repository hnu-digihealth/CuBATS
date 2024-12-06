# Standard Library
import concurrent.futures
import logging.config
import os
import pickle
import re
from itertools import combinations
from time import time

# Third Party
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# CuBATS
import cubats.cutils as cutils
import cubats.logging_config as log_config
import cubats.slide_collection.colocalization as colocalization
from cubats.slide_collection.slide import Slide

# Constants
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
SLIDE_COLLECTION_COLUMN_NAMES = ['Name', 'Reference', 'Mask',
                                 'Openslide Object', 'Tiles', 'Level Count', 'Level Dimensions', 'Tile Count']
# Define column names and data types
QUANTIFICATION_RESULTS_COLUMN_NAMES = {
    'Name': str,
    'High Positive (%)': float,
    'Positive (%)': float,
    'Low Positive (%)': float,
    'Negative (%)': float,
    'Background (%)': float,
    'Score': str
}
DEFAULT_TILE_SIZE = 1024


class SlideCollection(object):
    """Initializes a slide collection, stores slide info and performs slide processing.

    'Slide' is a class that initializes a slide collection and stores all relevant information so that processed
    information can be reloaded at a later time.

    Attributes:
        name (str): Name of parent directory (i.e. name of tumorset).

        src_dir (str): Path to source directory containing the WSIs.

        dest_dir (str): Path to destination directory for results.

        data_dir (str): Path to data directory, a subdirectory of dest_dir. The directory will be initiaded upon class
            creation inside the dest_dir. Inside the data directory summaries of quantification results, dual overlap
            results and triplet overlap results are stored as .CSV file. The data_dir also contains the pickle_dir.

        pickle_dir (str): Path to pickle directory, a subdirectory of data_dir. Inside the pickle directory pickled
            copies of the slide_collection, quantification results and antigen analysis are stored which will be
            automatically reloaded if a slide collection is (re-)initialized with the same output_dir.

        tiles_dir (str): Path to tiles directory, a subdirectory of dest_dir. Inside the tiles directory the tile
            directories for each slide are stored.

        colocalization_dir (str): Path to colocalization directory, a subdirectory of dest_dir. Inside the
            colocalization directory results of dual and triplet overlap analyses are stored.

        reconstruct_dir (str): Path to reconstruct directory, a subdirectory of dest_dir. Inside the reconstruct
            directory reconstructed slides are stored.

        collection_list (list of Slide): list containing all slide objects.

        collection_info_df (Dataframe): Dataframe containing relevant information on all the slides. The colums are:

            - Name (str): Name of slide.
            - Reference (bool): True if slide is reference slide.
            - Mask (bool): True if slide is mask slide.
            - OpenSlide Object (OpenSlide): OpenSlide object of the slide.
            - Tiles (DeepZoomGenerator): DeepZoom tiles of the slide.
            - Level Count (int): Number of Deep Zoom levels in the image.
            - Level Dimensions (list): List of tuples (pixels_x, pixels_y) for each Deep Zoom level.
            - Tile Count (int): Number of total tiles in the image.

        mask (Slide): Mask slide of the collection. Is set during initialization.

        mask_coordinates (list): List containing the tile coordinates for tiles that are covered by the mask
            Coordinates are tuples (column, row). TODO eliminate by using internal methods for applying mask to Image.

        quant_res_df (Dataframe): Dataframe containing the quantification results for all processed slides. The columns
            are:

            - Name (str): Name of the slide.

            - High Positive (float): Percentage of pixels in the high positive zone.

            - Positive (float): Percentage of pixels in the positive zone.

            - Low Positive (float): Percentage of pixels in the low positive zone.

            - Negative (float): Percentage of pixels in the negative zone.

            - Background (float): Percentage of pixels in the white space or fatty tissues zone.

            - Score (str): Overall score of the slide calculated from the average of scores for all tiles. However,
              this score may be misleading, as it is an average over the entire slide. TODO necessary?

        dual_overlap_summary (list): List containing a summary of the dual overlap results for all processed slides:

            - Slide 1 (str): Name of the first slide.

            - Slide 2 (str): Name of the second slide.

            - Total Coverage (float): Combined coverage of the two slides.

            - Total Overlap (float): Overlap of antigen expression in the two slides.

            - Total Complement (float): Complementary antigen expressions in the two slides.

            - Total Negative (float): Total of Negative in the two slides.

            - Error (float): Percentage of tiles that were not processed due to insufficient tissue coverage.

            - Unit (str): Unit of the percentages (%).

        triplet_overlap_summary (list): List containing a summary of the triplet overlap results for all processed
            analyses.

            - Slide 1 (str): Name of the first slide.

            - Slide 2 (str): Name of the second slide.

            - Slide 3 (str): Name of the third slide.

            - Total Coverage (float): Combined coverage of the three slides.

            - Total Overlap (float): Overlap of antigen expression in the three slides.

            - Total Complement (float): Complementary antigen expressions in the three slides.

            - Total Negative (float): Total of Negative in the three slides.

            - Error (float): Percentage of tiles that were not processed due to insufficient tissue coverage.

            - Unit (str): Unit of the percentages (%).

    """

    def __init__(self, collection_name, src_dir, dest_dir, ref_slide=None):
        """Initializes the class. The class contains information on the slide collection.

        Args:
            collection_name (str): Name of the collection (i.e. Name of tumor set or patient ID)

            src_dir (str): Path to src directory containing the WSIs.

            dest_dir (str): Path to destination directory for results.

            ref_slide (str, optional): Path to reference slide. If 'ref_slide' is None it will be automatically set to
                the HE slide based on the filename of input files. Defaults to None.

        """
        # Logging
        logging.config.dictConfig(log_config.LOGGING)
        self.logger = logging.getLogger(__name__)

        # Name of the tumorset
        self.collection_name = collection_name

        # Validate directories
        if not os.path.isdir(src_dir):
            raise ValueError(
                f"Source directory {src_dir} does not exist or is not accessible.")
        if not os.path.isdir(dest_dir):
            raise ValueError(
                f"Destination directory {dest_dir} does not exist or is not accessible.")

        # Directories
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.data_dir = None
        self.pickle_dir = None
        self.tiles_dir = None
        self.colocalization_dir = None
        self.reconstruct_dir = None

        # List containing all slide objects
        self.collection_list = []

        # Slide informations
        self.collection_info_df = pd.DataFrame(
            columns=SLIDE_COLLECTION_COLUMN_NAMES)

        # Mask Variables
        self.mask = None
        self.mask_coordinates = []  # TODO remove and replace

        # Reference Slide
        self.reference_slide = ref_slide

        # Quantification Variables
        self.quant_res_df = pd.DataFrame(
            columns=QUANTIFICATION_RESULTS_COLUMN_NAMES)

        # Antigen Expression Variables
        self.dual_overlap_summary = []
        self.triplet_overlap_summary = []

        # Set destination directories
        self.set_dst_dir()

        # Initialize the slide collection
        self.init_slide_collection()

        # Load previous results if exist
        self.load_previous_results()
        if not self.mask_coordinates:
            self.generate_mask()

    def set_dst_dir(self):
        """Assign and initiate needed directories if they do not exist yet."""
        # Data dir
        self.data_dir = os.path.join(self.dest_dir, RESULT_DATA_DIR)
        os.makedirs(self.data_dir, exist_ok=True)
        # Pickle dir
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
        for slide in self.collection_list:
            if slide.is_mask:
                pass
            elif slide.is_reference:
                pass
            else:
                slide_dir = os.path.join(self.tiles_dir, slide.name)
                os.makedirs(slide_dir, exist_ok=True)

        self.logger.debug("Destination directories set or already exist")

    def init_slide_collection(self):
        """
        Initializes the slide collection by iterating over the files in the source directory. Only files with the
        extensions '.tiff' or '.tif' are considered. For each valid file, a 'Slide' object is created. Each Slide's
        information is added to the 'collection_info_df'.

        Returns:
            None

            TODO: Check indexing of collection_info_df
        """
        self.logger.info("Initializing Slide Collection")
        init_start_time = time()
        for file in os.listdir(self.src_dir):
            if os.path.isfile(os.path.join(self.src_dir, file)):
                if not file.startswith(".") and (file.endswith(".tiff") or file.endswith(".tif")):
                    filename = cutils.get_name(file)
                    mask = False
                    ref = False
                    # Look for mask and reference slide. If no reference selected HE slide will be selected
                    if re.search("_mask", filename):
                        mask = True
                    elif re.search("HE", filename) or filename == self.reference_slide:
                        ref = True

                    slide = Slide(filename, os.path.join(
                        self.src_dir, file), is_mask=mask, is_reference=ref)
                    self.collection_list.append(slide)
                    self.collection_info_df.loc[len(
                        self.collection_info_df)] = slide.properties.values()
                    if ref and not self.reference_slide:
                        self.reference_slide = slide
                    elif mask:
                        self.mask = slide

        self.collection_info_df.to_csv(
            os.path.join(self.data_dir, "collection_info.csv"),
            sep=",",
            index=False,
            header=True,
            encoding="utf-8",
        )
        init_end_time = time()
        self.logger.debug(
            f"Slide collection initialized in {round((init_end_time - init_start_time),2)} seconds")

    def load_previous_results(self, path=None):
        """Loads results from previous processing if they exist.

        Tries to load results from previous processing. If no path is passed, the slide_collections pickle_dir is used.
        Slide objects are based on OpenSlide which are C-type objects and cannot be stored as pickle. Therefore, each
        Slide is re-initialized in the init_slide_collection function. The function will try to load the following
        files:

            - mask_coordinates.pickle: Load mask coordinates from previous mask processing.

            - quantification_results.pickle: Load quantification results from previous processing.

            - dual_overlap_results.pickle: Load dual antigen overlap results from previous processing.

            - triplet_overlap_results.pickle: Load triplet antigen overlap results from previous processing.

            - slidename_processing_info.pickle: Load processing information for each slide in the collection from
              previous processing.

        Args:
            path (str, optional): Path to directory containing pickle files. Defaults to pickle_dir of the slide
                collection.

        """
        prev_res_start_time = time()
        self.logger.info("Searching for previous results")
        if self.quant_res_df.__len__() == 0:
            if path is None:
                path = self.pickle_dir
            path_mask_coord = os.path.join(path, "mask_coordinates.pickle")
            path_quant_res = os.path.join(
                path, "quantification_results.pickle")
            path_dual_overlap_res = os.path.join(
                path, "dual_overlap_results.pickle")
            path_triplet_overlap_res = os.path.join(
                path, "triplet_overlap_results.pickle")

            # load mask coordinates
            if os.path.exists(path_mask_coord):
                self.mask_coordinates = pickle.load(
                    open(path_mask_coord, "rb"))
                self.logger.debug(
                    f"Successfully loaded mask for {self.collection_name}")
            else:
                self.logger.debug(
                    f"No mask coordinates found for {self.collection_name}")

            # load quantification results
            if os.path.exists(path_quant_res):
                self.quant_res_df = pickle.load(
                    open(path_quant_res, "rb"))
                self.logger.debug(
                    f"Sucessfully loaded quantification results for {self.collection_name}")
            else:
                self.logger.debug(
                    f"No previous quantification results found for {self.collection_name}")

            # load dual overlap results
            if os.path.exists(path_dual_overlap_res):
                self.dual_overlap_summary = pickle.load(
                    open(path_dual_overlap_res, "rb"))
                self.logger.debug(
                    f"Successfully loaded dual overlap results for {self.collection_name}")
            else:
                self.logger.debug(
                    f"No previous dual overlap results found for {self.collection_name}")

            # load triplet overlap results
            if os.path.exists(path_triplet_overlap_res):
                self.triplet_overlap_summary = pickle.load(
                    open(path_triplet_overlap_res, "rb"))
                self.logger.debug(
                    f"Successfully loaded triplet overlap results for {self.collection_name}")
            else:
                self.logger.debug(
                    f"No previous triplet overlap results found for {self.collection_name}")

            # Load processing info for each slide TODO load into slide object
            for slide in self.collection_list:
                path_slide = os.path.join(
                    path, f"{slide.name}_processing_info.pickle")
                if os.path.exists(path_slide):
                    slide.detailed_quantification_results = pickle.load(
                        open(path_slide, "rb"))
                    self.logger.debug(
                        f"Successfully loaded processing info for slide {slide.name}")

                    # If quantification results for loaded slide exist, load them into the slide object
                    if not self.quant_res_df[self.quant_res_df['Name'] == slide.name].empty:
                        slide.quantification_summary = self.quant_res_df.loc[
                            self.quant_res_df['Name'] == slide.name]
                        self.logger.debug(
                            f"Successfully loaded detailed quantification results for slide {slide.name}")
                    else:
                        self.logger.debug(
                            f"No quantification results found for slide {slide.name}")
                else:
                    self.logger.debug(
                        f"No previous processing info found for slide {slide.name}")
                    pass
        prev_res_end_time = time()
        self.logger.info(
            f"Finished loading previous results for {self.collection_name} in \
                {round((prev_res_end_time - prev_res_start_time), 2 )} seconds")

    def generate_mask(self, save_img=False):
        """Generates mask coordinates based on the mask slide.

        Generates a list containing of tiles coordinates that are part of the mask. This allows to only process tiles
        that are part of the mask and thus contain tumor tissue. Previous mask coordinates will be overwritten and the
        results will be stored as pickle in pickle_dir.

        Args:
            save_img (bool): Boolean to determine if mask tiles shall be saved as image. Necessary if mask shall be
                reconstructed later on. Note: Storing tiles will require addition storage. Defaults to False.

        """
        if self.mask is None:
            raise ValueError(
                "Slide Collection does not have a mask slide. Please check if src_dir contains a mask slide. \
                    If not,please run 'run_segmentation_pipeline' to generate a mask slide.")

        mask_start_time = time()
        self.logger.debug("Generating mask coordinates")
        mask_tiles = self.mask.tiles
        self.mask_coordinates.clear()
        cols, rows = mask_tiles.level_tiles[mask_tiles.level_count - 1]
        for col in tqdm(range(cols), desc="Initializing Mask"):
            for row in range(rows):
                temp = mask_tiles.get_tile(
                    mask_tiles.level_count - 1, (col, row))
                if not (temp.mode == "RGB"):
                    temp_rgb = temp.convert("RBG")
                    temp_np = np.array(temp_rgb)
                else:
                    temp_np = np.array(temp)

                # If tile is mostly white, drop tile coordinate
                if temp_np.mean() < 230:
                    self.mask_coordinates.append((col, row))
                    if save_img:
                        tile_name = str(col) + "_" + str(row)
                        img = Image.fromarray(temp_np)
                        dir = os.path.join(self.tiles_dir, "mask")
                        os.makedirs(dir, exist_ok=True)
                        out_path = os.path.join(dir, tile_name + ".tif")
                        img.save(out_path)
                else:
                    pass
        mask_end_time = time()
        self.logger.debug(
            f"Mask coordinates generated in {round((mask_end_time - mask_start_time)/60,2)} minutes")

        # Save mask coordinates as pickle
        out = os.path.join(self.pickle_dir, "mask_coordinates.pickle")
        pickle.dump(self.mask_coordinates, open(out, "wb"))
        self.logger.debug(f"Successfully saved mask coordinates to {out}")
        self.logger.info("Finished Mask Generation")

    def quantify_all_slides(self, save_imgs=False, detailed_mode=False):
        """Quantifies all registered slides sequentially and stores results.

        Quantifies all slides that were instantiated sequentially with the exception of the reference_slide and the
        mask_slide. Results are stored as .CSV into the data_dir. All previous quantification results in the
        'quant_res_df' will be reset and the .CSV file overwritten.

        Args:
            save_imgs (bool): Boolean determining if tiles shall be saved as image during processing. This is necessary
                if slides shall be reconstructed after processing. Note: storing tiles will require additional
                storage. Defaults to False.

            detailed_mode (bool): Boolean determining if detailed mask shall be used for quantification. Defaults to
                False.

        """
        if self.quant_res_df.__len__() != 0:
            self.quant_res_df = self.quant_res_df.iloc[0:0]
        # Counter variable for progress tracking
        c = 1
        for slide in self.collection_list:
            if not slide.is_mask and not slide.is_reference:
                self.logger.info(
                    f"Analyzing Slide: {slide.name}({c}/{len(self.collection_list) - 2})")
                self.quantify_single_slide(
                    slide.name, save_imgs, detailed_mode)
                c += 1

    def quantify_single_slide(self, slide_name, save_img=False, detailed_mode=False):
        """ Calls quantify_slide for given slide_name and appends results to self.quant_res_df.

        This function quantifies staining intensities for all tiles of the given slide using multiprocessing. The slide
        matching the passed slide_name is retrieved from the collection_list and quantified using the quantify_slide
        function of the Slide class. Results are appended to self.quant_res_df, which is then stored as .CSV in self.
        data_dir and as .PICKLE in self.pickle_dir. Existing .CSV/.PICKLE files are overwritten.
        For more information on quantification checkout Slide.quantify_slide() function in the slide.py.

        Args:
            slide_name (str): Name of the slide to be processed.

            save_img (bool): Boolean determining if tiles shall be saved during processing. Necessary if slide shall be
                reconstructed later on. However, storing images will require addition storage. Defaults to False.

            detailed_mode (bool): Boolean determining if detailed mask shall be used for quantification. Defaults to
                False.

        """
        slide = [
            slide for slide in self.collection_list if slide.name == slide_name][0]

        # Create directories for images if they are to be saved.
        if save_img:
            dab_tile_dir = os.path.join(
                self.tiles_dir, slide_name, DAB_TILE_DIR)
            if detailed_mode:
                slide.quantify_slide(self.mask_coordinates,
                                     self.pickle_dir, save_img, dab_tile_dir, detailed_mask=self.mask.tiles)
            else:
                slide.quantify_slide(self.mask_coordinates,
                                     self.pickle_dir, save_img, dab_tile_dir)
        else:
            if detailed_mode:
                slide.quantify_slide(
                    self.mask_coordinates, self.pickle_dir, detailed_mask=self.mask.tiles)
            else:
                slide.quantify_slide(self.mask_coordinates, self.pickle_dir)

        # Check if a row already exists
        existing_row_index = self.quant_res_df[self.quant_res_df['Name']
                                               == slide.quantification_summary['Name']].index

        if not existing_row_index.empty:
            # Update existing row
            self.quant_res_df.loc[existing_row_index[0]
                                  ] = slide.quantification_summary
        else:
            # Append the new row TODO change to pd.concat, append deprecated
            self.quant_res_df = pd.concat([self.quant_res_df, pd.DataFrame(
                slide.quantification_summary)], ignore_index=True)
            # self.quant_res_df = self.quant_res_df.append(
            #    slide.quantification_summary, ignore_index=True)

        # Sort the DataFrame by the 'Name' column
        self.quant_res_df = self.quant_res_df.sort_values(
            by='Name').reset_index(drop=True)

        self.save_quantification_results()

    def save_quantification_results(self):
        """
        Stores quant_res_df as .CSV for analysis and .PICKLE for reloading in data_dir and pickle_dir, respectively.
        """
        if self.quant_res_df.__len__() != 0:
            save_start_time = time()
            self.quant_res_df.to_csv(
                self.data_dir + "/quantification_results.csv",
                sep=",",
                index=False,
                encoding="utf-8",
            )
            out = os.path.join(
                self.pickle_dir, "quantification_results.pickle")
            pickle.dump(self.quant_res_df, open(out, "wb"))
            save_end_time = time()
            self.logger.debug(
                f"Successfully saved quantification results to {out} in \
                    {round((save_end_time - save_start_time),2)} seconds")
        else:
            self.logger.warning(
                "No quantification results were found. Please call quantify_all_slides() to quantify all slides \
                    in this slide collection or call quantify_single_slide() to quantify a single slide.")

    def get_dual_antigen_combinations(self):
        """ Creates antigen pairs and calls compute_antigen_combinations for each pair.

        Creates all possible combinations of pairs amongst all quantified slides and analyzes antigen expressions for
        each pair, including antigen overlap. Results are stored in self.dual_overlap_results.

        """
        self.dual_overlap_summary.clear()
        # Filter out mask and reference slides
        filtered_slides = [
            slide for slide in self.collection_list if not slide.is_mask and not slide.is_reference]

        # Generate all possible pairs of the filtered slides
        slide_combinations = list(combinations(filtered_slides, 2))

        # Pass each combination to the compute_dual_antigen_combination method
        for combo in slide_combinations:
            self.compute_dual_antigen_combination(combo[0], combo[1])

    def get_triplet_antigen_combinations(self):
        """ Creates antigen triplets and calls compute_antigen_combinations for each triplet.

        Creates all possible combinations of triplets amongst all quantified slides and analyzes antigen expressions
        for each triplet, including antigen overlap. Results are stored in self.triplet_overlap_results.

        """
        self.triplet_overlap_summary.clear()
        antigen_combinations = list(combinations(
            self.quantification_results_list, 3))
        for ele in antigen_combinations:
            self.compute_triplet_antigen_combinations(ele[0], ele[1], ele[2])

    # def compute_antigen_combinations(self, save_img=False):

    #     if save_img:
    #         comb2 = list(combinations(self.collection_list, 2))
    #         comb3 = list(combinations(self.collection_list, 3))
    #         comb2_dir = os.path.join(
    #             self.colocalization_dir, "dual_combinations")
    #         os.makedirs(comb2_dir, exist_ok=True)
    #         comb3_dir = os.path.join(
    #             self.colocalization_dir, "triplet_combinations")
    #         os.makedirs(comb3_dir, exist_ok=True)
    #         for combs in comb2:
    #             os.makedirs(os.path.join(
    #                 comb2_dir, combs[0].name + "_and_" + combs[1].name), exist_ok=True)
    #         for combs in comb3:
    #             os.makedirs(os.path.join(
    #                 comb3_dir, combs[0].name + "_and_" + combs[1].name + "_and_" + combs[2].name), exist_ok=True)
    #     else:
    #         dir = None

    #     iterable = []
    #     for i in self.collection_list[0].detailed_quantification_results:
    #         tiles = []
    #         for slide in self.collection_list:
    #             tiles.append(slide.detailed_quantification_results[i])
    #         iterable.append((tiles, dir, save_img))
    # TODO Work in pogress for optimizating antigen computation

    def compute_dual_antigen_combination(self, slide1, slide2, save_img=False):
        """
        Analyzes antigenexpressions for each of tiles of the given pair of slides using Multiprocesing. Results from
        each of the tiles are summarized, stored in self.dual_overlap_results and saved as CSV in self.data_dir as well
        as PICKLE in self.pickle_dir.

        Args:
            slide1 (dict): Quantification results for slide 1
            slide2 (dict): Quantification results for slide 2

        _overlap_dict:
            - Slide 1 (str): Name of the first slide
            - Slide 2 (str): Name of the second slide
            - Total Coverage (float): Combined coverage of the two slides
            - Total Overlap (float): Overlap of antigen expression in the two slides
            - Total Complement (float): Complementary antigen expressions in the two slides
            - Total Negative (float): Total of Negative in the two slides
            - Error (float): Percentage of tiles that were not processed due to insufficient tissue coverage
            - Unit (str): Unit of the percentages (%)



        """
        # name1 = slide1[0]
        # name2 = slide2[0]
        # slide1 = slide1[1]
        # slide2 = slide2[1]

        # Create directory for pair of slides
        if save_img:
            dir = os.path.join(
                self.colocalization_dir, (slide1.name + "_and_" + slide2.name))
            os.makedirs(dir, exist_ok=True)
        else:
            dir = None

        # Create iterable for multiprocessing
        iterable = []
        for i in slide1.detailed_quantification_results:
            if (
                slide1.detailed_quantification_results[i]["Tilename"]
                == slide2.detailed_quantification_results[i]["Tilename"]
            ):
                _dict1 = slide1.detailed_quantification_results[i]
                _dict2 = slide2.detailed_quantification_results[i]
                iterable.append((
                    _dict1, _dict2, dir, save_img)
                )

        # for i in slide1:
        #     if slide1[i]["Tilename"] == slide2[i]["Tilename"]:
        #         img1 = slide1[i]
        #         img2 = slide2[i]
        #         iterable.append((img1, img2, dirname, save_img))

        # Init dict for results of each tile
        comparison_dict = {}

        with concurrent.futures.ProcessPoolExecutor() as exe:
            results = tqdm(
                exe.map(
                    colocalization.analyze_dual_antigen_colocalization, iterable),
                total=len(iterable),
                desc="Calculating Coverage of Slide " + slide1.name + " & " + slide2.name,
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
        sum_tissue = 0.00
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
                sum_tissue += comparison_dict[i]["Tissue"]
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
        sum_tissue = sum_tissue / counter
        sum_background = sum_background / counter
        total_error = error1 + error2 + error3

        overlap_dict["Slide 1"] = slide1.name
        overlap_dict["Slide 2"] = slide2.name
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
        overlap_dict["Tissue"] = round(sum_tissue, 2)
        overlap_dict["Background / No Tissue"] = round(sum_background, 2)
        overlap_dict["Total Error"] = round(
            (total_error / comparison_dict.__len__()) * 100, 2
        )
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
            self.data_dir + "/dual_overlap_results.csv",  # TODO better naming
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
        Analyzes antigenexpressions for each of tiles of the given triplet of slides using Multiprocesing. Results from
        each of the tiles are summarized,stored in self.triplet_overlap_results and saved as CSV in self.data_dir as
        well as PICKLE in self.pickle_dir.

        Args:
            slide1 (dict): Quantification results for slide 1
            slide2 (dict): Quantification results for slide 2
            slide3 (dict): Quantification results for slide 3

        overlap_dict:
            - Slide 1 (str): Name of the first slide
            - Slide 2 (str): Name of the second slide
            - Slide 3 (str): Name of the third slide
            - Total Coverage (float): Combined coverage of the three slides
            - Total Overlap (float): Overlap of antigen expression in the three slides
            - Total Complement (float): Complementary antigen expressions in the three slides
            - Total Negative (float): Total of Negative in the three slides
            - Error (float): Percentage of tiles that were not processed due to insufficient tissue coverage
            - Unit (str): Unit of the percentages (%)

            TODO: saving image optional
        """
        # name1 = slide1[0]
        # name2 = slide2[0]
        # name3 = slide3[0]
        # slide1 = slide1[1]
        # slide2 = slide2[1]
        # slide3 = slide3[1]

        # Create directory for triplet of slides
        if save_img:
            dirname = os.path.join(
                self.colocalization_dir, (slide1.name + "_and_"
                                          + slide2.name + "_and_" + slide3.name)
            )
            os.makedirs(dirname, exist_ok=True)
        else:
            dirname = None

        # Create iterable for multiprocessing
        iterable = []
        for i in slide1.detailed_quantification_results:
            if (
                slide1.detailed_quantification_results[i]["Tilename"]
                == slide2.detailed_quantification_results[i]["Tilename"]
                == slide3.detailed_quantification_results[i]["Tilename"]
            ):
                _dict1 = slide1.detailed_quantification_results[i]
                _dict2 = slide2.detailed_quantification_results[i]
                _dict3 = slide3.detailed_quantification_results[i]
                iterable.append(
                    (_dict1, _dict2, _dict3, dirname, save_img)
                )
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
                exe.map(
                    colocalization.analyze_triplet_antigen_colocalization, iterable),
                total=len(iterable),
                desc="Calculating Coverage of Slides "
                + slide1.name
                + " & "
                + slide2.name
                + " & "
                + slide3.name,
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
        sum_tissue = 0.00
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
                sum_tissue += comparison_dict[i]["Tissue"]
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
        sum_tissue = sum_tissue / counter
        sum_background = sum_background / counter
        total_error = error1 + error2 + error3

        overlap_dict["Slide 1"] = slide1.name
        overlap_dict["Slide 2"] = slide2.name
        overlap_dict["Slide 3"] = slide3.name
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
        overlap_dict["Tissue"] = round(sum_tissue, 2)
        overlap_dict["Background / No Tissue"] = round(sum_background, 2)
        overlap_dict["Total Error"] = round(
            (total_error / comparison_dict.__len__()) * 100, 2
        )
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
            self.data_dir + "/triplet_overlap_results.csv",  # TODO better naming
            sep=",",
            index=False,
            header=True,
            encoding="utf-8",
        )
        out = os.path.join(self.pickle_dir, "triplet_overlap_results.pickle")
        pickle.dump(self.triplet_overlap_summary, open(out, "wb"))
