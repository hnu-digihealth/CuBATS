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
from matplotlib import pyplot as plt
from PIL import Image
from pyvips import BandFormat
from pyvips import Image as VipsImage
from tqdm import tqdm

# CuBATS
import cubats.Utils as utils
import cubats.slide_collection.colocalization as colocalization
from cubats.slide_collection.slide import Slide

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
QUANTIFICATION_RESULTS_COLUMN_NAMES = [
    'Name', 'High Positive (%)', 'Positive (%)', 'Low Positive (%)', 'Negative (%)', 'Background (%)', 'Score']


DEFAULT_TILE_SIZE = 1024


class SlideCollection(object):
    """Initializes a slide collection, stores slide info and performs slide
    processing.

    'Slide' is a class that initializes a slide collection and stores all
    relevant information so that processed information can be reloaded at a
    later time.

    Attributes:
        name (str): name of parent directory (should be name of tumorset )

        src_dir (str): Path to source directory containing the WSIs

        dest_dir (str): Path to destination directory for results

        data_dir (str): Path to data directory. Inside the data directory
            quantification summary, dual overlap summary and triplet overlap
            summary are stored. In addition it contains the pickle subdirectory

        pickle_dir (str): Path to pickle directory. Inside the pickle directory
            pickled copies are stored which can later be reloaded for future
            re-/processing.

        tiles_dir (str): Path to tiles directory. Inside the tiles directory
            the tile directories for each slide are stored.

        colocalization_dir (str): Path to colocalization directory. Inside the
            colocalization directory results of dual and triplet overlap
            analyses are stored.

        reconstruct_dir (str): Path to reconstruct directory. Inside the
            reconstruct directory reconstructed slides are stored.

        tile_dir_list (list): list containing the paths to the tile directories
            for each slide. Inside these directories tile images of the
            respective slide are stored.

        dab_tile_dir_list (list): List containing the paths to the DAB tile
            directories for each slide. Inside these directories DAB tile
            images of the respective slide are stored.

        quantification_results_list (list): List containing the results of the
            quantification for each slide. The list is nested and contains a
            dictionary for each slide. This dictionary contains another
            dictionary containing the results for each tile of the slide.

        orig_img_list (list): List containing the original file names of the
            WSIs

        img_names (list): List containing the clear names of the WSIs

        mask_coordinates (list): List containing the tile coordinates for tiles
            that are covered by the mask. Coordinates are tuples (column, row).

        quantification_summary (list): List containing a summary of the
            quantification results pf all slides:
            Slide (str): Name of the slide

            High Positive (float): Percentage of pixels in the high positive
                zone

            Positive (float): Percentage of pixels in the positive zone

            Low Positive (float): Percentage of pixels in the low positive zone

            Negative (float): Percentage of pixels in the negative zone

            White Space or Fatty Tissues (float): Percentage of pixels in
                the white space or fatty tissues zone

            Unit (str): Unit of the percentages (%)

            Score (str): Overall score of the slide based on the zones.
                However, the score for the entire slide may be misleading since

        dual_overlap_summary (list): List containing a summary of the dual
            overlap results for all processed analyses:
            Slide 1 (str): Name of the first slide

            Slide 2 (str): Name of the second slide

            Total Coverage (float): Combined coverage of the two slides

            Total Overlap (float): Overlap of antigen expression in the two
                slides

            Total Complement (float): Complementary antigen expressions in
                the two slides

            Total Negative (float): Total of Negative in the two slides

            Error (float): Percentage of tiles that were not processed due
                to insufficient tissue coverage

            Unit (str): Unit of the percentages (%)

        triplet_overlap_summary (list): List containing a summary of the
            triplet overlap results for all processed analyses.
            Slide 1 (str): Name of the first slide

            Slide 2 (str): Name of the second slide

            Slide 3 (str): Name of the third slide

            Total Coverage (float): Combined coverage of the three slides

            Total Overlap (float): Overlap of antigen expression in the
                three slides

            Total Complement (float): Complementary antigen expressions in
                the three slides

            Total Negative (float): Total of Negative in the three slides

            Error (float): Percentage of tiles that were not processed due
                to insufficient tissue coverage

            Unit (str): Unit of the percentages (%)

        slide_info_dict (Dict): Dictionary containing information on all slides
            name (str): Name of the slide

            openslide_object (OpenSlide Object): OpenSlide Object of the slide

            tiles (DeepZoomGenerator): DeepZoomGenerator wrapping the OpenSlide
                Object

            tiles_count (int): Total number of tiles

            level_dimensions (list): Dimensions of each DeepZoom level

            total_count_tiles (int): Total number of tiles

        ref_slide (Dict): Dictionary containing information on the reference
            slide

        mask (Dict): Dictionary containing information on the mask slide
    """

    def __init__(self, collection_name, src_dir, dest_dir, ref_slide=None):
        """Initializes the class. The class contains

        Args:
            src_dir (str): Path to src directory containing the WSIs

            dest_dir (str): Path to destination directory for results

            ref_slide (str, optional): Path to reference slide. If 'ref_slide'
                is None it will be automatically set to the HE slide based on
                the filename of input files. Defaults to None.

        """
        # Name of the tumorset
        self.collection_name = collection_name

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
        self.mask_coordinates = []

        # Reference Slide
        self.reference_slide = ref_slide

        # Quantification Variables
        self.quant_res_df = pd.DataFrame(
            columns=QUANTIFICATION_RESULTS_COLUMN_NAMES)

        # Antigen Expression Variables
        self.dual_overlap_summary = []
        self.triplet_overlap_summary = []

        # Initialize the slide collection
        self.set_slide_collection()

        # Set destination directories
        self.set_dst_dir()

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

        # Create subdirectories in tiles_dir for each slide except for the
        # reference slide and the mask
        for slide in self.collection_list:
            if slide.is_mask:
                pass
            elif slide.is_reference:
                pass
            else:
                slide_dir = os.path.join(self.tiles_dir, slide.name)
                # self.tile_dir_list.append(slide_dir)
                os.makedirs(slide_dir, exist_ok=True)
                # self.dab_tile_dir_list.append(dab_dir)

            # fname = utils.get_name(f)
            # if re.search("HE", fname) or re.search("mask", fname):
            #     slide_dir = os.path.join(self.tiles_dir, fname)
            #     self.tile_dir_list.append(slide_dir)
            # else:
            #     slide_dir = os.path.join(self.tiles_dir, fname)
            #     self.tile_dir_list.append(slide_dir)
            #     os.makedirs(slide_dir, exist_ok=True)
            #     dab_dir = os.path.join(slide_dir, DAB_TILE_DIR)
            #     self.dab_tile_dir_list.append(dab_dir)

    def set_slide_collection(self):
        """
        Sets the slide collection by iterating over the files in the source directory.
        Only files with the extensions '.tiff' or '.tif' are considered.
        For each valid file, a Slide object is created and added to the slide collection DataFrame.

        Returns:
            None

            TODO: Check indexing of collection_info_df
        """
        for file in os.listdir(self.src_dir):
            if os.path.isfile(os.path.join(self.src_dir, file)):
                if not file.startswith(".") and (file.endswith(".tiff") or file.endswith(".tif")):
                    filename = utils.get_name(file)
                    mask = False
                    ref = False
                    # if string contains _mask, it is the mask slide, if it does't contain mask but HE it is the reference slide
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

    def load_previous_results(self, path=None):
        """Loads results from previous processing if they exist.

        Tries to load results from previous processing. If no path is passed,
        the objects pickle directory is used. OpenSlide objects cannot be
        saved as pickle as they are C-types. Therefore, they are initiatited
        separately in the set_all_slides_dict function.
        The following files are tried to be loaded if they exist in the given
        directory:
            mask_coordinates.pickle: Load mask coordinates from previous mask
                generation
            quantification_results.pickle: Load quantification results from
                previous processing
            dual_overlap_results.pickle: Load dual antigen overlap results
                from previous processing
            triplet_overlap_results.pickle: Load triplet antigen overlap
                results from previous processing
            processing_info.pickle: Load processing information for each slide
                from previous processing

        Args:
            path (str): Path to directory containing pickle files.

        """
        print("\n==== Searching for previous results\n")
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
                print("\n==== Loading mask coordinates\n")
                self.mask_coordinates = pickle.load(
                    open(path_mask_coord, "rb"))

            # load quantification results
            if os.path.exists(path_quant_res):
                print("\n==== Loading quantification results\n")
                self.quant_res_df = pickle.load(
                    open(path_quant_res, "rb"))

            # load dual overlap results
            if os.path.exists(path_dual_overlap_res):
                print("\n==== Loading dual overlap results\n")
                self.dual_overlap_summary = pickle.load(
                    open(path_dual_overlap_res, "rb"))

            # load triplet overlap results
            if os.path.exists(path_triplet_overlap_res):
                print("\n==== Loading triplet overlap results\n")
                self.triplet_overlap_summary = pickle.load(
                    open(path_triplet_overlap_res, "rb")
                )

            # Load processing info for each slide TODO load into slide object
            for slide in self.collection_list:
                path_slide = os.path.join(
                    path, f"{slide.name}_processing_info.pickle")
                if os.path.exists(path_slide):
                    print("\n==== Loading processing info for slide", slide.name)
                    slide.detailed_quantification_results = pickle.load(
                        open(path_slide, "rb")
                    )
                    # If quantification results for loaded slide exist, load them into the slide object
                    if not self.quant_res_df[self.quant_res_df['Name'] == slide.name].empty:
                        slide.quantification_summary = self.quant_res_df.loc[
                            self.quant_res_df['Name'] == slide.name]
                else:
                    pass

    def generate_mask(self, save_img=False):
        """Generates mask coordinates.

        Generates a list containing of tiles coordinates that are part of the
        mask. This allows to only process tiles that are part of the mask and
        thus are relevant for analysis. Previous mask coordinates are cleared.

        """
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

        # Save mask coordinates as pickle
        out = os.path.join(self.pickle_dir, "mask_coordinates.pickle")
        pickle.dump(self.mask_coordinates, open(out, "wb"))

    def quantify_all_slides(self, save_images=False):
        """Quantifies all registered slides sequentially and stores results.

        Quantifies all slides that were instantiated sequentially with the
        exception of the reference slide and the mask. Results are stored as
        .CSV into the DATADIR.
        Previous quantification results stored in the slide collection will be
        reset and and the .CSV file overwritten.

        Args:
            save_images (bool): Boolean determining if tiles shall be saved as image during processing. This is
            necessary if slides shall be reconstructed after processing. However, storing tiles may require additional
            storage.

        """
        if self.quant_res_df.__len__() != 0:
            self.quant_res_df.iloc[0:0]
        # Counter variable for progress tracking
        c = 1
        for slide in self.collection_list:
            if not slide.is_mask and not slide.is_reference:
                print(
                    "Analyzing Slide: "
                    + slide.name
                    + "("
                    + str(c)
                    + "/"
                    + str(len(self.collection_list) - 2)
                    + ")\n"
                )
                self.quantify_single_slide(slide.name, save_images)
                c += 1

    def quantify_single_slide(self, slide_name, save_img=False):
        """ Calls quantify_slide for given slide_name and appends results to self.quant_res_df.

        This function quantifies staining intensities for all tiles of the given slide using multiprocessing. The slide
        matching the passed slide_name is retrieved from the collection_list and quantified using the quantify_slide
        function of the Slide class. Results are appended to self.quant_res_df, which is then stored as .CSV in self.
        data_dir and as .PICKLE in self.pickle_dir. Existing .CSV/.PICKLE files are overwritten.
        For more information on quantification checkout Slide.quantify_slide() function of the slide.py.

        Args:
            - slide_name (str): Name of the slide to be processed.
            - save_img (bool): Boolean determining if tiles shall be saved during processing.
        """
        slide = [
            slide for slide in self.collection_list if slide.name == slide_name][0]

        # if slide.is_mask:
        #     raise ValueError("Cannot quantify mask slide.")
        # elif slide.is_reference:
        #     raise ValueError("Cannot quantify reference slide.")

        if save_img:
            dab_tile_dir = os.path.join(
                self.tiles_dir, slide_name, DAB_TILE_DIR)
            slide.quantify_slide(self.mask_coordinates,
                                 self.pickle_dir, save_img, dab_tile_dir)
        else:
            slide.quantify_slide(self.mask_coordinates, self.pickle_dir)

        self.quant_res_df.loc[len(self.quant_res_df)
                              ] = slide.quantification_summary

        self.save_quantification_results()

    def save_quantification_results(self):
        """ Stores quant_res_df as .CSV for analysis and .PICKLE for reloading in self.data_dir and self.pickle_dir,
        respectively.
        """
        if self.quant_res_df.__len__() != 0:
            self.quant_res_df.to_csv(
                self.data_dir + "/quantification_results.csv",
                sep=",",
                index=False,
                encoding="utf-8",
            )
            out = os.path.join(
                self.pickle_dir, "quantification_results.pickle")
            pickle.dump(self.quant_res_df, open(out, "wb"))

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

    def get_triplet_antigen_combinations(self):
        """
        Creates all possible combinations of triplets amongst all quantified slides and analyzes antigen expressions for each triplet, including antigen overlap.
        Results are stored in self.triplet_overlap_results.
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

    def compute_dual_antigen_combinations(self, slide1, slide2, save_img=False):
        """
        Analyzes antigenexpressions for each of tiles of the given pair of slides using Multiprocesing. Results from each of the tiles are summarized,
        stored in self.dual_overlap_results and saved as CSV in self.data_dir as well as PICKLE in self.pickle_dir.

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
            if slide1.detailed_quantification_results[i]["Tilename"] == slide2.detailed_quantification_results[i]["Tilename"]:
                _dict1 = slide1.detailed_quantification_results[i]
                _dict2 = slide2.detailed_quantification_results[i]
                iterable.append((_dict1, _dict2, dir, save_img))

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
                    colocalization.compute_dual_antigen_colocalization, iterable),
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
            if (slide1.detailed_quantification_results[i]["Tilename"] == slide2.detailed_quantification_results[i]["Tilename"] == slide3.detailed_quantification_results[i]["Tilename"]):
                _dict1 = slide1.detailed_quantification_results[i]
                _dict2 = slide2.detailed_quantification_results[i]
                _dict3 = slide3.detailed_quantification_results[i]
                iterable.append((_dict1, _dict2, _dict3, dirname, save_img))
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
                    colocalization.compute_triplet_antigen_colocalization, iterable),
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
            slide_name (str): Name of slide
            input_path (str): Path to saved tiles

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
        """This function plots quantification results of a given tilename. It plots the DAB-image, the histogram of the intensity distribution,
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
                        self.quantification_results_list[i][1][j]["Px_count"]
                    )
                    tile_exists = True
                    break

        assert (
            tile_exists
        ), "The given tilename does not exist for one or more of the slides. Please make sure to select an existing tilename."

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

        fig.suptitle(
            f"Quantification Results for Tile: {tilename}\n", fontsize=16)
        fig.tight_layout()
        plt.show()
