# Standard Library
import math
import os
import re

# Third Party
import numpy as np
import tifffile as tiff
from PIL import Image
from skimage import img_as_ubyte
from skimage.color import hed2rgb, rgb2gray, rgb2hed
from skimage.exposure import histogram
from tqdm import tqdm


def ihc_stain_separation(
    ihc_rgb,
    hematoxylin=False,
    eosin=False,
):
    """
    Function receives IHC image, separates individual stains (Hematoxylin, Eosin, DAB)
        from image and returns an image for each of the individual stains

    Credits:
    Credits for this function are owed to A. C. Ruifrok and D. A. Johnston with there paper
    “Quantification of histochemical staining by color deconvolution,”
    Analytical and quantitative cytology and histology / the International Academy of Cytology
    [and] American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001. PMID: 11531144
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_ihc_color_separation.html#sphx-glr-auto-examples-color-exposure-plot-ihc-color-separation-py


    Args:
        ihc_rgb : IHC image in RGB
        hematoxylin : Boolean, if True returns Hematoxylin image
        eosin : Boolean, if True returns Eosin image

    Returns:
        ihc_h   : Hematoxylin staining of image
        ihc_e   : Eosin staining of image
        ihc_d   : DAB (3',3'-Diaminobenzidine)
    """
    # convert RGB image to HED using prebuild skimage method
    ihc_hed = rgb2hed(ihc_rgb)

    # Create RGB image for each seperate stain
    # Convert to ubyte for easier saving to drive as image
    null = np.zeros_like(ihc_hed[:, :, 0])
    if hematoxylin:
        ihc_h = img_as_ubyte(hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
    else:
        ihc_h = None

    if eosin:
        ihc_e = img_as_ubyte(hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1)))
    else:
        ihc_e = None

    ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))

    return (ihc_d, ihc_h, ihc_e)


def quantify_single_tile(iterable):
    """This function processes a single input tile and returns a dictionary.
            - If the tile is mostly white and thus shows no- or only little tissue, the tile will not be processed further.
                The returned dict will contain the Tilename and a Flag = -1
            - Else the tile will be processed, including stain_separation and pixel_intensity calculations. The DAB Image
                will then be saved in passed directory and results of pixelintensity will be returned inside the dictionary.

    Args:
        input (Iterable): Iterable containing Information on passed tile for further processing:
            - index 0: Column, necessary for naming
            - index 1: Row, necessary for naming
            - index 2: Tile itself, necessary since processes cannot access shared memory
            - DAB_TILE_DIR: Directory, for saving Image, since single processes cannot access shared memory

    Returns:
        Dictionary: Returns dictionary for input tile containing results of calculation of pixel intensity:
            - Tilename
            - Histogram
            - Hist_centers
            - Zones
            - Percentage
            - Score
            - Px_count
            - Flag
            - Image Array

            TODO Add modes for optional histogram etc save

    """
    # Assign local variables for better readability
    col = iterable[0]
    row = iterable[1]
    tile = iterable[2]
    DAB_TILE_DIR = iterable[3]
    save_img = iterable[4]
    # Tilename
    tile_name = str(col) + "_" + str(row)

    # Initialize Dictionary for single tile
    single_tile_dict = {}
    single_tile_dict["Tilename"] = tile_name

    # Convert tile to numpy array
    temp = tile  # DEEPZOOM_OBJECT.get_tile(DEEPZOOM_LEVEL - 1, (row, col))
    temp_rgb = temp.convert("RGB")
    temp_np = np.array(temp_rgb)

    # Only process tiles that are mostly covered and not blank to save runtime and space
    if temp_np.mean() < 230 and temp_np.std() > 15:

        # Separate stains
        DAB, H, E = ihc_stain_separation(temp_np)

        # Calculate pixel intensity
        (hist, hist_centers, zones, percentage, score, pixelcount, img_analysis) = (
            calculate_pixel_intensity(DAB)
        )

        # Save image as tif in passed directory if wanted.
        if save_img:
            img = Image.fromarray(DAB)
            DAB_TILE_DIR = f"{DAB_TILE_DIR}/{tile_name}.tif"
            print(DAB_TILE_DIR)
            img.save(DAB_TILE_DIR)

        # Complete dictionary
        single_tile_dict["Histogram"] = hist
        single_tile_dict["Hist_centers"] = hist_centers
        single_tile_dict["Zones"] = zones
        single_tile_dict["Percentage"] = percentage
        single_tile_dict["Score"] = score
        single_tile_dict["Px_count"] = pixelcount
        single_tile_dict["Image Array"] = img_analysis
        single_tile_dict["Flag"] = (
            1  # Flag = 1: Tile processed Flag necessary for analyzing by index
        )
    else:
        single_tile_dict["Flag"] = 0  # Flag = 0 : Tile not processed

    return single_tile_dict


def calculate_pixel_intensity(image):
    """Calculates pixel intensity of each pixel in the input image and separates them into 4 different zones based on their intensity. Intensity of each pixel lies between 0 and 255.
        Intensities above 235 are
        predominantly fatty tissues but dont contribute to pathological scoring:
        - Zone 1 = High positive (intensity: 0-60)
        - Zone 2 = Positive (intensity: 61-120)
        - Zone 3 = Low positive (intensity: 121-180)
        - Zone 4 = negative (intensity: 181-235)
        After calculating pixel intensities this function calculates percentage contribution of each of the zones
        as well as the a pathology score

        Credits Varghese et al. (2014) "IHC Profiler: An Open Source Plugin for the Quantitative Evaluation and Automated Scoring of Immunihistochemistry Images of Human Tissue Samples"

    Args:
        image (_type_): Input image

    Returns:
        - Histogram (array): actual histogram of the tile
        - Hist_centers (array): center of bins of histogram
        - Zones (array): Number of pixels for each zone. During processing pixels in the tile are assigned to one of four zones based on pixel intensity.
        For more information see utils.calculate_pixel_intensity
        - Percentage (array): Percentage of pixels in each zone
        - Score (array): Score for each of the ties
        - Px_count: Total number of pixels in the tile
        - Image Array (array): Pixelvalues of for positive pixels. Pixels with values ranging from 0 to 121 are considered positive.
    """

    # Conversion to gray-scale-ubyte image
    gray_scale_image = rgb2gray(image)
    gray_scale_ubyte = img_as_ubyte(gray_scale_image)
    # Calculates a histogram of the input image
    hist, hist_centers = histogram(image)

    w, h = gray_scale_ubyte.shape

    # array containg only high-/ & positive pixels
    img_analysis = np.full((w, h), 255, dtype="uint8")

    # Array for Zones of pixel intensity
    zones = np.zeros(5)
    # pixelcount
    pixelcount = 0
    # iterates through each pixel in the image and assigns it to one of the intensity zones
    for y in range(h):
        for x in range(w):
            if gray_scale_ubyte[x, y] < 61:  # High positive tissue
                zones[0] += 1
                img_analysis[x, y] = gray_scale_ubyte[x, y]
                pixelcount += 1
            elif gray_scale_ubyte[x, y] < 121:  # Positive tissue
                zones[1] += 1
                img_analysis[x, y] = gray_scale_ubyte[x, y]
                pixelcount += 1
            elif gray_scale_ubyte[x, y] < 181:  # Low positive tissue
                zones[2] += 1
                pixelcount += 1
                img_analysis[x, y] = gray_scale_ubyte[x, y]
            elif gray_scale_ubyte[x, y] < 236:  # Negative tissue
                zones[3] += 1
                pixelcount += 1
                img_analysis[x, y] = gray_scale_ubyte[x, y]
            else:  # White space aka. fatty tissue needed for calculation in respect to actual tissue
                zones[4] += 1
                pixelcount += 1

    percentage, score = calculate_score(zones, pixelcount)

    return hist, hist_centers, zones, percentage, score, pixelcount, img_analysis


def calculate_score(zones, count):
    """
    Calculates Percentage of amount of pixels in each zone in respect to pixelcount
    TODO Check for Score and >66,6% px intensity

    Args:
        zones (array): Array containing amount of pixels from each zone
        pixelcount (int): total pixelcount

    Returns:
        percentage (array): Array containing percentage of pixels in each zone
        Score (array): Array containing calculation of score for each zone
        TODO add scorename
    """
    percentage = np.zeros(zones.size)
    score = np.zeros(zones.size)

    for i in range(zones.size):
        percentage[i] = (zones[i] / count) * 100
        score[i] = (zones[i] * (zones.size - i)) / count

    return percentage, score


def get_score_name(score):
    """
    Returns name of Zone with the highest score.

    Args:
        score (list): List containing score for each zone

    Returns:
        String: Name of highest zone with highest score
    """
    list = ["High Positive", "Positive", "Low Positive", "Negative"]
    max = np.max(score[:4])
    score = list[score.index(max)]

    return score


def compute_dual_antigen_colocalization(iterable):
    """
    This function processes the information for the same tile from 2 different WSIs, analyzes these images for antigen coverage
    and calculates how much tissue is covered by these different antigen combinations, as well as how much of the tissue is covered
    by more than one image and how much tissure is only covered by one of the 2 images. The results are then saved in a dictionary
    and returned. Additionally an image containing the results can be saved in the passed directory.

    Args:
        iterable (iterable): Contains the 2 Tiles(iterable[0:2]) that are to be compared and analyzed for antigen coverage as well as the output directory
            - iterable[0]: img1
            - iterable[1]: img2
            - iterable[2]: output path

    Returns:
        colocal_dict (Dict): returns a dictionary containing the results of the analysis
            - Tilename: Name of the tile
            - Flag: Flag indicating if the tile was processed or not. ((1): processed; (0): not processed due to one or two tiles not being processable; (-1): all tiles not containing tissue)
            - Coverage: Amount of pixels (in %) that both tiles cover combined in respect to actual tissue in images. (Only if Flag == 1)
            - Overlap: Overlap of postitive pixels: Amount of the same pixels (in %) that are covered by both images. (Only if Flag == 1)
            - Complement: Amount of positive pixels (in %) that are only covered by one of the 2 images. (Only if Flag == 1)
            - Negative: Amount of negative pixels (in %) that are not covered either image. (Only if Flag == 1)
    """

    img1 = iterable[0]
    img2 = iterable[1]
    dir = iterable[2]
    save_img = iterable[3]

    tilename = img1["Tilename"]
    colocal_dict = {}
    colocal_dict["Tilename"] = tilename
    # Check if both Images contain tissue. If one doesn't: Flag = 0, if both dont: Flag = -1
    if img1["Flag"] == 0 or img2["Flag"] == 0:
        if img1["Flag"] == 0 and img2["Flag"] == 0:
            colocal_dict["Flag"] = -1
        else:
            colocal_dict["Flag"] = 0
    # Check if Images have the correct shape
    elif img1["Image Array"].shape != (1024, 1024) or img2["Image Array"].shape != (
        1024,
        1024,
    ):
        colocal_dict["Flag"] = -2
    else:
        colocal_dict["Flag"] = 1
        # init
        (
            high_overlap,
            pos_overlap,
            low_overlap,
            high_complement,
            pos_complement,
            low_complement,
            negative,
            background,
        ) = (0, 0, 0, 0, 0, 0, 0, 0)
        h, w = img1["Image Array"].shape  # TODO check if shape is the same for both images
        # output img
        img = None
        if save_img:
            """
            If save_img is True, in addition to the numerical analysis an image containing the results of the analysis will be created and saved in passed directory.
            Else only numerical analysis will be performed.
            """
            img = np.full((h, w, 3), 255, dtype="uint8")
            for y in range(h):
                for x in range(w):
                    if (
                        img1["Image Array"][x, y] < 61
                        and img2["Image Array"][x, y] < 61
                    ):
                        high_overlap += 1
                        img[x, y] = [254, 0, 0]  # Color red where both img Positive
                    elif (
                        img1["Image Array"][x, y] < 61
                        and not img2["Image Array"][x, y] < 61
                    ):
                        high_complement += 1
                        img[x, y] = [
                            116,
                            238,
                            21,
                        ]  # Color green where only img1 positive
                    elif (
                        not img1["Image Array"][x, y] < 61
                        and img2["Image Array"][x, y] < 61
                    ):
                        high_complement += 1
                        img[x, y] = [0, 30, 255]  # color blue where only img2 positive
                    elif (
                        img1["Image Array"][x, y] < 121
                        and img2["Image Array"][x, y] < 121
                    ):
                        pos_overlap += 1
                        img[x, y] = [254, 0, 0]  # Color red where both img Positive
                    elif (
                        img1["Image Array"][x, y] < 121
                        and not img2["Image Array"][x, y] < 121
                    ):
                        pos_complement += 1
                        img[x, y] = [116, 238, 21]
                    elif (
                        not img1["Image Array"][x, y] < 121
                        and img2["Image Array"][x, y] < 121
                    ):
                        pos_complement += 1
                        img[x, y] = [0, 30, 255]
                    elif (
                        img1["Image Array"][x, y] < 181
                        and img2["Image Array"][x, y] < 181
                    ):
                        low_overlap += 1
                        img[x, y] = [254, 0, 0]  # Color red where both img Positive
                    elif (
                        img1["Image Array"][x, y] < 181
                        and not img2["Image Array"][x, y] < 181
                    ):
                        low_complement += 1
                        img[x, y] = [116, 238, 21]  # TODO pick color
                    elif (
                        not img1["Image Array"][x, y] < 181
                        and img2["Image Array"][x, y] < 181
                    ):
                        low_complement += 1
                        img[x, y] = [0, 30, 255]  # TODO pick color
                    elif (
                        img1["Image Array"][x, y] < 235
                        or img2["Image Array"][x, y] < 235
                    ):
                        negative += 1
                        img[x, y] = [255, 255, 255]  # color white where both negative
                    else:
                        background += 1

            img = Image.fromarray(img.astype("uint8"))
            out = f"{dir}/{tilename}.tif"
            img.save(out)
        else:
            for y in range(h):
                for x in range(w):
                    pixel_values = [img1["Image Array"][x, y], img2["Image Array"][x, y]]
                    sum_high = sum(1 for value in pixel_values if value < 61)
                    sum_pos = sum(1 for value in pixel_values if value < 121)
                    sum_low = sum(1 for value in pixel_values if value < 181)

                    if sum_high >= 2:
                        high_overlap += 1
                    elif sum_high == 1:
                        high_complement += 1
                    elif sum_pos >= 2:
                        pos_overlap += 1
                    elif sum_pos == 1:
                        pos_complement += 1
                    elif sum_low >= 2:
                        low_overlap += 1
                    elif sum_low == 1:
                        low_complement += 1
                    elif any(value < 235 for value in pixel_values):
                        negative += 1
                    else:
                        background += 1

        coverage = (
            high_overlap
            + pos_overlap
            + low_overlap
            + high_complement
            + pos_complement
            + low_complement
        )
        total_overlap = high_overlap + pos_overlap + low_overlap
        total_complement = high_complement + pos_complement + low_complement
        tissue_count = coverage + negative

        # Vals in % for overlap, complement, negative in respect to entire image
        coverage = round((coverage / tissue_count) * 100, 4)
        total_overlap = round((total_overlap / tissue_count) * 100, 4)
        total_complement = round((total_complement / tissue_count) * 100, 4)
        high_overlap = round((high_overlap / tissue_count) * 100, 4)
        high_complement = round((high_complement / tissue_count) * 100, 4)
        pos_overlap = round((pos_overlap / tissue_count) * 100, 4)
        pos_complement = round((pos_complement / tissue_count) * 100, 4)
        low_overlap = round((low_overlap / tissue_count) * 100, 4)
        low_complement = round((low_complement / tissue_count) * 100, 4)
        negative = round((negative / tissue_count) * 100, 4)
        background = round((background / tissue_count) * 100, 4)

        # set dict
        colocal_dict["Total Coverage"] = coverage
        colocal_dict["Total Overlap"] = total_overlap
        colocal_dict["Total Complement"] = total_complement
        colocal_dict["High Positive Overlap"] = high_overlap
        colocal_dict["High Positive Complement"] = high_complement
        colocal_dict["Positive Overlap"] = pos_overlap
        colocal_dict["Positive Complement"] = pos_complement
        colocal_dict["Low Positive Overlap"] = low_overlap
        colocal_dict["Low Positive Complement"] = low_complement
        colocal_dict["Negative"] = negative
        colocal_dict["Background/No Tissue"] = background

    return colocal_dict


def compute_triplet_antigen_colocalization(iterable):
    """
    This function receives the same tile from 3 different WSIs, analyzes these images for antigen coverage and calculates how much
    tissue is covered by these different antigen combinations, as well as how much of the tissue is covered by more than one image
    and how much tissure is only covered by one of the 3 images.The results are then saved in a dictionary and returned. Additionally
    an image containing the results is saved in the passed directory.


    Args:
        iterable (iterable): Contains the 3 Tiles(iterable[0:3]) that are to be compared and analyzed for antigen coverage as well as the output directory
        - iterable[0]: img1
        - iterable[1]: img2
        - iterable[2]: img3
        - iterable[3]: output path

    Returns:

        colocal_dict (Dictionary): returns a dictionary containing the results of the analysis
         - Tilename: Name of the tile
         - Flag: Flag indicating if the tile was processed or not. ((1): processed; (0): not processed due to one or two tiles not being processable;
                (-1): all tiles not containing tissue)
         - Coverage: Amount of pixels (in %) that all 3 tiles cover combined in respect to actual tissue in images.  (Only if Flag == 1)
         - Overlap: Overlap of postitive pixels: Amount of positive pixels (in %) that are covered by at least 2 or all 3 images combined  (Only if Flag == 1)
         - Complement: Amount of positive pixels (in %) that are only covered by one of the 3 images (Only if Flag == 1)
         - Negative: Amount of negative pixels (in %) that are not covered by any of the 3 images  (Only if Flag == 1)

    """

    img1 = iterable[0]
    img2 = iterable[1]
    img3 = iterable[2]
    dir = iterable[3]
    save_img = iterable[4]

    tilename = img1["Tilename"]
    colocal_dict = {}
    colocal_dict["Tilename"] = tilename

    # Check if all Images contain tissue. If all dont: Flag = -1, if one or two dont: Flag = 0
    if img1["Flag"] == 0 or img2["Flag"] == 0 or img3["Flag"] == 0:
        if img1["Flag"] == 0 and img2["Flag"] == 0 and img3["Flag"] == 0:
            colocal_dict["Flag"] = -1
        else:
            colocal_dict["Flag"] = 0
    # Check for Error 3: Flag = -1: One of images doesn't have the correct shape # TODO Think about padding images to correct shape
    elif (
        img1["Image Array"].shape != (1024, 1024)
        or img2["Image Array"].shape != (1024, 1024)
        or img3["Image Array"].shape != (1024, 1024)
    ):
        colocal_dict["Flag"] = -2
    # Otherwise calculate overlap: Flag = 1
    else:
        colocal_dict["Flag"] = 1
        # init
        (
            high_overlap,
            pos_overlap,
            low_overlap,
            high_complement,
            pos_complement,
            low_complement,
            negative,
            background,
        ) = (0, 0, 0, 0, 0, 0, 0, 0)

        # output img
        h, w = img1["Image Array"].shape
        img = None

        if save_img:
            """
            If save_img is True, in addition to the numerical analysis an image containing the results of the analysis will be created and saved in passed directory.
            Else only numerical analysis will be performed.
            """
            img = np.full((h, w, 3), 255, dtype="uint8")
            for y in range(h):
                for x in range(w):
                    pixel_values = [img["Image Array"][x, y] for img in [img1, img2, img3]]
                    sum_high = sum(1 for val in pixel_values if val < 61)
                    sum_pos = sum(1 for val in pixel_values if val < 121)
                    sum_low = sum(1 for val in pixel_values if val < 181)

                    if sum_high >= 2:
                        high_overlap += 1
                        img[x, y] = [254, 0, 0]
                    elif sum_high == 1:
                        idx = pixel_values.index(min(pixel_values))
                        high_complement += 1
                        img[x, y] = [[116, 238, 21], [0, 30, 255], [255, 231, 0]][idx]
                    elif sum_pos >= 2:
                        pos_overlap += 1
                        img[x, y] = [254, 0, 0]
                    elif sum_pos == 1:
                        idx = pixel_values.index(min(pixel_values))
                        pos_complement += 1
                        img[x, y] = [[116, 238, 21], [0, 30, 255], [255, 231, 0]][idx]
                    elif sum_low >= 2:
                        low_overlap += 1
                    elif sum_low == 1:
                        idx = pixel_values.index(min(pixel_values))
                        low_complement += 1
                        # img[x, y] = [252, 140, 132]
                    elif any(val < 235 for val in pixel_values):
                        negative += 1
                        # img[x, y] = [255, 255, 255]
                    else:
                        background += 1

            img = Image.fromarray(img.astype("uint8"))
            out = f"{dir}/{tilename}.tif"
            img.save(out)
        else:
            # TODO simplify. However simpler version before somehow didn't return correct results
            for y in range(h):
                for x in range(w):
                    pixel_values = [img["Image Array"][x, y] for img in [img1, img2, img3]]

                    if sum(1 for val in pixel_values if val < 61) >= 2:
                        high_overlap += 1
                    elif any(val < 61 for val in pixel_values):
                        high_complement += 1
                    elif sum(1 for val in pixel_values if val < 121) >= 2:
                        pos_overlap += 1
                    elif any(val < 121 for val in pixel_values):
                        pos_complement += 1
                    elif sum(1 for val in pixel_values if val < 181) >= 2:
                        low_overlap += 1
                    elif any(val < 181 for val in pixel_values):
                        low_complement += 1
                    elif any(val < 235 for val in pixel_values):
                        negative += 1
                    else:
                        background += 1

        coverage = (
            high_overlap
            + pos_overlap
            + low_overlap
            + high_complement
            + pos_complement
            + low_complement
        )
        total_overlap = high_overlap + pos_overlap + low_overlap
        total_complement = high_complement + pos_complement + low_complement
        tissue_count = coverage + negative

        # Vals in % for overlap, complement, negative in respect to entire image
        coverage = round((coverage / tissue_count) * 100, 4)
        total_overlap = round((total_overlap / tissue_count) * 100, 4)
        total_complement = round((total_complement / tissue_count) * 100, 4)
        high_overlap = round((high_overlap / tissue_count) * 100, 4)
        high_complement = round((high_complement / tissue_count) * 100, 4)
        pos_overlap = round((pos_overlap / tissue_count) * 100, 4)
        pos_complement = round((pos_complement / tissue_count) * 100, 4)
        low_overlap = round((low_overlap / tissue_count) * 100, 4)
        low_complement = round((low_complement / tissue_count) * 100, 4)
        negative = round((negative / tissue_count) * 100, 4)
        background = round((background / tissue_count) * 100, 4)

        # set dict
        colocal_dict["Total Coverage"] = coverage
        colocal_dict["Total Overlap"] = total_overlap
        colocal_dict["Total Complement"] = total_complement
        colocal_dict["High Positive Overlap"] = high_overlap
        colocal_dict["High Positive Complement"] = high_complement
        colocal_dict["Positive Overlap"] = pos_overlap
        colocal_dict["Positive Complement"] = pos_complement
        colocal_dict["Low Positive Overlap"] = low_overlap
        colocal_dict["Low Positive Complement"] = low_complement
        colocal_dict["Negative"] = negative
        colocal_dict["Background/No Tissue"] = background

    return colocal_dict


def get_name(f):
    """Returns Object name, and removes image type extension from filename

    Credits: valtils.py, line 55-72

    Args:
        f (String): Path to file

    Returns:
        String: Image Filename without extension
    """
    if re.search(r"\.", f) is None:
        return f
    f = os.path.split(f)[-1]

    if f.endswith(".ome.tiff") or f.endswith(".ome.tif"):
        back_slice_idx = 2
    else:
        back_slice_idx = 1
    img_name = "".join([".".join(f.split(".")[:-back_slice_idx])])

    return img_name


def downsample_Openslide_to_PIL(Openslide_object, SCALEFACTOR: int):
    """This function takes an Openslide Object as input and downscales it based on the Slides
        optimal level for downsampling and the given Scalefactor. IT returns a PIL Image as well
        as downsample parameters.

    Args:
        Openslide_object (Openslide Object): The Object that needs to be downscaling
        SCALEFACTOR (int): Factor for downscaling

    Returns:
        img (PIL.Image): rescaled Image
        old_w (int): width of input Openslide Object
        old_h (int): height of input Openslide Object
        new_w (int): width of output Image
        new_h (int): height of output Image
    """
    # current width and height od Openslide Oject
    old_w, old_h = Openslide_object.dimensions
    # rescaled width and height of Image
    new_w = math.floor(old_w / SCALEFACTOR)
    new_h = math.floor(old_h / SCALEFACTOR)
    # Find optimal level for downsampling
    level = Openslide_object.get_best_level_for_downsample(SCALEFACTOR)
    # Conversion to PIL image
    wsi = Openslide_object.read_region(
        (0, 0), level, Openslide_object.level_dimensions[level]
    )
    wsi = wsi.convert("RGB")
    img = wsi.resize((new_w, new_h), Image.BILINEAR)

    return img, old_w, old_h, new_w, new_h


def separate_stains_and_save__tiles_as_tif(
    deepzoom_object, deepzoom_level, target_directory
):
    """
    This function iterates through OpenSlide Deep Zoom Object at the given Deep Zoom level and saves each individual
    slide in the given Directory named as "col_row".tif just as "save_all_tiles_as_tif" does. However, additionally this function
    calls the "color_separation" function, separates H-,E-,DAB stains from each tile and saves an additional tif for each
    stain in a newly created subdirectory respectively.

    Args:
        deepzoom_object (DeepZoomGenerator): DeepZoomGenerator
        deepzoom_level (Deep Zoom Level): Wanted Deep Zoom level through which shall be iterated
        target_directory (Str): Target directory in which tiles are stored
    """
    # Create directories for original tiles, hematoxylin stain, eosin stain and DAB stain
    ORIGINAL_TILES_DIR = target_directory + "/original_tiles"
    os.makedirs(ORIGINAL_TILES_DIR, exist_ok=True)
    DAB_TILE_DIR = target_directory + "/DAB_tiles"
    os.makedirs(DAB_TILE_DIR, exist_ok=True)
    H_TILE_DIR = target_directory + "/H_tiles"
    os.makedirs(H_TILE_DIR, exist_ok=True)
    E_TILE_DIR = target_directory + "/E_tiles"
    os.makedirs(E_TILE_DIR, exist_ok=True)

    cols, rows = deepzoom_object.level_tiles[deepzoom_level - 1]
    for row in tqdm(range(rows)):
        for col in range(cols):
            tile_name = str(col) + "_" + str(row)

            temp = deepzoom_object.get_tile(deepzoom_level - 1, (col, row))
            temp_rgb = temp.convert("RGB")
            temp_np = np.array(temp_rgb)

            tiff.imsave(ORIGINAL_TILES_DIR + "/" + tile_name + "_original.tif", temp_np)
            # print("Saving tile:" + tile_name)

            # Now only process tiles that are mostly covered and not blank to save runtime and space
            if temp_np.mean() < 230 and temp_np.std() > 15:
                # print("Separating color for tile: ", tile_name)
                H, E, DAB = ihc_stain_separation(temp_np)

                # saving DAB,H,E in subdirectories
                tiff.imsave(DAB_TILE_DIR + "/" + tile_name + "_DAB.tif", DAB)
                tiff.imsave(H_TILE_DIR + "/" + tile_name + "_H.tif", H)
                tiff.imsave(E_TILE_DIR + "/" + tile_name + "_E.tif", E)
            else:
                pass
                # print("NOT PROCESSING TILE", tile_name)
