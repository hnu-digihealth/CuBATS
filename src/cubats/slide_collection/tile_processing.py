"""
This module contains functions for processing and analyzing tiles from whole slide images (WSIs), including color
deconvolution and quantification.

Credits:

- The `ihc_stain_separation` function is adapted from the work of A. C. Ruifrok and D. A. Johnston in their paper
  “Quantification of histochemical staining by color deconvolution,” Analytical and quantitative cytology and
  histology / the International Academy of Cytology [and] American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug.
  2001. PMID: 11531144. Source: \
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_ihc_color_separation.html

- The `calculate_pixel_intensity` and `calculate_percentage_and_score` function are inspired by the work of Varghese et
  al. (2014) "IHC Profiler: An Open Source Plugin for the Quantitative Evaluation and Automated Scoring of
  Immunohistochemistry Images of Human Tissue Samples."


Last Modified: 2023-10-05
"""

# Standard Library
import os

# Third Party
import cv2
import numpy as np
import tifffile as tiff
from PIL import Image
from skimage import img_as_ubyte
from skimage.color import hed2rgb, rgb2gray, rgb2hed
from skimage.exposure import histogram
from tqdm import tqdm

# CuBATS
from cubats.config import xp
from cubats.cutils import to_numpy


def quantify_tile(iterable):
    """This function quantifies a single input tile and returns a dictionary.

    The function offers two masking modes: tile-level masking and pixel-level masking. In tile-level masking tiles that
    contain sufficient tissue (mean pixel value < 230 and standard deviation > 15), will undergo stain separation and
    quantification. The results will be returned in the dictionary. The 'flag' will be set to 1. If 'save_img' is True
    the DAB image will additionally be saved in the specified directory. If the tile does not contain sufficient tissue
    it will not be processed and the returned dictionary will only contain the tile name and a 'flag' set to 0.
    In pixel-level masking the mask will be applied to the tile prior to quantification to receive a merged tile which
    will then be quantified. Non-mask pixels will be set to 255.


    Args:
        iterable (iterable): Iterable containing the following Information on passed tile:

            - index 0: Column, necessary for naming.
            - index 1: Row, necessary for naming.
            - index 2: Tile itself, necessary since processes cannot access shared memory.
            - DAB_TILE_DIR: Directory, for saving Image, since single processes cannot access shared memory.
            - save_img: Boolean, if True, DAB image will be saved in specified directory.
            - antigen_profile: Antigen-specific thresholds used during quantification.
            - masking_mode: Masking mode for quantification.

    Returns:
        dict: Dictionary containing tile results:

            - Tilename (str): Name of the tile.
            - Histogram (ndarray): Histogram of the tile.
            - Hist_centers (ndarray): Centers of histogram bins.
            - Zones (ndarray): Number of pixels in each intensity zone.
            - Percentage (ndarray): Percentage of pixels in each zone.
            - Score (ndarray): Score for the tile.
            - Tissue Count (int): Total number of tissue pixels in the tile.
            - Flag (int): Processing flag (1 if processed, 0 if not).
            - Image Array (ndarray): Array of pixel values for positive pixels.

        TODO Add modes for optional histogram etc save

    """
    # Assign local variables for better readability
    col = iterable[0]
    row = iterable[1]
    tile = iterable[2][0]
    mask = iterable[2][1]
    DAB_TILE_DIR = iterable[3]
    save_img = iterable[4]
    antigen_profile = iterable[5]
    # masking_mode = iterable[6]

    tile_name = str(col) + "_" + str(row)

    # Initialize Dictionary for single tile
    single_tile_dict = {}
    single_tile_dict["Tilename"] = tile_name

    # Convert tile to numpy array
    temp = tile  # DEEPZOOM_OBJECT.get_tile(DEEPZOOM_LEVEL - 1, (row, col))
    temp_rgb = temp.convert("RGB")

    temp = xp.array(temp_rgb)
    mean = xp.mean(temp)
    std = xp.std(temp)

    process_tile = False
    # Masking mode
    if mask is None:
        if mean < 230 and std > 15:
            process_tile = True
        else:
            single_tile_dict["Flag"] = 0
    else:
        process_tile = True

    if process_tile:
        # Separate stains
        DAB, H, E = ihc_stain_separation(temp)

        # Calculate pixel intensity
        (hist, hist_centers, zones, percentage, score, mask_count, img_analysis) = (
            calculate_pixel_intensity(DAB, antigen_profile, mask)
        )

        # Save image as tif in passed directory if wanted.
        if save_img:
            if not DAB_TILE_DIR:
                raise ValueError(
                    "Target directory must be specified if save_img is True"
                )
            img = Image.fromarray(to_numpy(DAB))
            DAB_TILE_DIR = f"{DAB_TILE_DIR}/{tile_name}.tif"
            # print(DAB_TILE_DIR)
            img.save(DAB_TILE_DIR)

        # Complete dictionary
        single_tile_dict["Histogram"] = hist
        single_tile_dict["Hist_centers"] = hist_centers
        single_tile_dict["Zones"] = zones
        single_tile_dict["Percentage"] = percentage
        single_tile_dict["Score"] = score
        single_tile_dict["Mask Count"] = mask_count
        single_tile_dict["Image Array"] = img_analysis
        single_tile_dict["Flag"] = 1

    return single_tile_dict


def ihc_stain_separation(
    ihc_rgb,
    hematoxylin=False,
    eosin=False,
):
    """
    Separates individual stains (Hematoxylin, Eosin, DAB) from an IHC image and returns an image for each stain.

    Args:
        ihc_rgb (Image): IHC image in RGB format.
        hematoxylin (bool): If True, returns Hematoxylin image as well. Defaults to False.
        eosin (bool): If True, returns Eosin image. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - ihc_d (Image): DAB (3',3'-Diaminobenzidine) stain of the image.
            - ihc_h (Image): Hematoxylin stain of the image if hematoxylin=True, otherwise None.
            - ihc_e (Image): Eosin stain of the image if eosin=True, otherwise None.
    """
    # Convert RGB image to HED using prebuilt skimage method
    # Convert to numpy array if transformation
    ihc_hed = rgb2hed(to_numpy(ihc_rgb))
    ihc_hed = xp.array(ihc_hed)
    # Create RGB image for each seperate stain
    null = xp.zeros_like(ihc_hed[:, :, 0])
    # Separate Hematoxylin stain
    ihc_h = (
        img_as_ubyte(
            hed2rgb(to_numpy(xp.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
        )
        if hematoxylin
        else None
    )
    # Separate Eosin stain
    ihc_e = (
        img_as_ubyte(
            hed2rgb(to_numpy(xp.stack((null, ihc_hed[:, :, 1], null), axis=-1)))
        )
        if eosin
        else None
    )
    # Separate DAB stain
    ihc_d = img_as_ubyte(
        hed2rgb(to_numpy(xp.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))
    )

    return ihc_d, ihc_h, ihc_e


def calculate_pixel_intensity(image, antigen_profile, tumor_mask=None):
    """
    Calculates pixel intensity of each pixel in the input image and separates them into 5 different zones based on
    their intensity. The image is converted to grayscale format, resulting in a distribution of intensity values
    between 0-255. Intensities above 235 are predominantly background or fatty tissues and do not contribute to
    pathological scoring. Thresholds for high-positive, medium-positive, low-positive and negative pixels are defined
    by the passed antigen profile. If 'masking_mode' is 'pixel-level', pixels with an intensity value of 255 will be
    excluded as they mark non-mask areas.
    After calculating pixel intensities this function calculates percentage contribution of each of the zones with
    respect to the mask tissue in the tile, as well as the a pathology score.

    Args:
        image (numpy.ndarray): Input image.
        antigen_profile (dict): Dictionary with threshold values.
        masking_mode (String). Masking mode: tile-level or pixel-level

    Returns:
        tuple: A tuple containing:
            - hist (ndarray): Histogram of the image.
            - hist_centers (ndarray): Centers of histogram bins.
            - zones (xp.ndarray): Number of pixels in each intensity zone with respect to the tumor mask.
            - percentage (xp.ndarray): Percentage of pixels in each intensity zone with respect to the tumor mask.
            - score (str): Overall score of the tile.
            - tissuecount (int): Tissue count of the tile with respect to the tumor mask.
            - img_analysis (xp.ndarray): Array of pixel values for multi-antigen evaluation.
    """

    # Conversion to gray-scale-ubyte image
    gray_scale_image = rgb2gray(image)
    gray_scale_ubyte = img_as_ubyte(gray_scale_image)

    # Calculate histogram
    hist, hist_centers = histogram(gray_scale_image)

    # Convert to xp array for processing
    gray_scale_ubyte = xp.array(gray_scale_ubyte)
    w, h = gray_scale_ubyte.shape

    # Initilize arrays for analysis
    img_analysis = xp.full((w, h), 255, dtype="float32")

    zones = xp.zeros(5, dtype=xp.int32)

    # Get thresholds from antigen_profile
    high_thresh = antigen_profile["high_positive_threshold"]
    medium_thresh = antigen_profile["medium_positive_threshold"]
    low_thresh = antigen_profile["low_positive_threshold"]

    # Define intensity masks
    high_positive_mask = gray_scale_ubyte < high_thresh
    positive_mask = (gray_scale_ubyte >= high_thresh) & (
        gray_scale_ubyte < medium_thresh
    )
    low_positive_mask = (gray_scale_ubyte >= medium_thresh) & (
        gray_scale_ubyte < low_thresh
    )
    negative_mask = (gray_scale_ubyte >= low_thresh) & (gray_scale_ubyte < 235)
    background_mask = gray_scale_ubyte >= 235

    if tumor_mask is not None:
        tumor_mask = xp.array(tumor_mask, dtype=bool)  # ensure boolean
        high_positive_mask &= tumor_mask
        positive_mask &= tumor_mask
        low_positive_mask &= tumor_mask
        negative_mask &= tumor_mask
        background_mask &= tumor_mask
    # if mask == "pixel-level":
    #     tumor_mask = gray_scale_ubyte < 255
    #     # Apply tumor mask to all masks including background
    #     high_positive_mask &= tumor_mask
    #     positive_mask &= tumor_mask
    #     low_positive_mask &= tumor_mask
    #     negative_mask &= tumor_mask
    #     background_mask = (
    #         (gray_scale_ubyte >= 235) & (gray_scale_ubyte < 255) & tumor_mask
    #     )
    # else:
    #     tumor_mask = None
    #     background_mask = gray_scale_ubyte >= 235

    # Update img_analysis with pixel values based on intensity masks also containing background
    img_analysis[high_positive_mask] = gray_scale_ubyte[high_positive_mask]
    img_analysis[positive_mask] = gray_scale_ubyte[positive_mask]
    img_analysis[low_positive_mask] = gray_scale_ubyte[low_positive_mask]
    img_analysis[negative_mask] = gray_scale_ubyte[negative_mask]
    img_analysis[background_mask] = gray_scale_ubyte[background_mask]

    if tumor_mask is not None:
        img_analysis[~tumor_mask] = xp.nan
    # Calculate zones except non-mask
    zones[0] = xp.sum(high_positive_mask)
    zones[1] = xp.sum(positive_mask)
    zones[2] = xp.sum(low_positive_mask)
    zones[3] = xp.sum(negative_mask)
    zones[4] = xp.sum(background_mask)

    tissue_count = xp.sum(zones[:4])
    if tumor_mask is not None:
        mask_count = xp.sum(tumor_mask)
    else:
        mask_count = tissue_count
    # Calculate pixel count and percentage

    if xp.sum(zones) == 0:
        percentage = xp.zeros(5, dtype=xp.int32)
        percentage[4] = 100
        score = "Background"
    else:
        percentage, score = calculate_percentage_and_score(zones)

    return (
        hist,
        hist_centers,
        zones,
        percentage,
        score,
        int(mask_count),
        to_numpy(img_analysis),
    )


def calculate_percentage_and_score(zones):
    """
    Calculates the percentage of pixels in each zone relative to the tissue count (Positive tissues) and total mask
    count (Background) and computes a score for each zone. If more than 66.6% of the total pixels are attributed to
    a single zone, that zone's score is assigned. Else, the score for each zone is calculated using this formula:

    .. math::

        \\text{Score} = \\frac{(\\text{number of pixels in zone} \\times \\text{weight of zone})}{\\text{total
        pixels in image}}

    with weights 4 for the high positive zone, 3 for the positive zone, 2 for the low positive zone, 1 for the negative
    zone, and 0 for the background. The final score is the maximum score among all zones.

    Args:
        zones (xp.ndarray): Array containing amount of pixels from each zone

    Returns:
        tuple: A tuple containing:
            - percentage (xp.ndarray): Array containing the percentage of pixels in each zone.
            - score (str): Name of the zone if it exceeds 66.6%, otherwise the name of the zone with the highest score.

    Raises:
        ZeroDivisionError: If pixel count is zero.
        ValueError: If all zones have zero pixels.
    """

    if xp.sum(zones) == 0:
        raise ValueError("All zones have zero pixels")

    tissue_count = xp.sum(zones[:4])
    total_pixels = xp.sum(zones)

    # Calculate percentage of pixels in each zone
    percentage_tissue = (zones[:4] / tissue_count) * 100
    percentage_background = (zones[4:] / total_pixels) * 100
    percentage = xp.concatenate([percentage_tissue, percentage_background])
    zone_names = [
        "High Positive",
        "Positive",
        "Low Positive",
        "Negative",
        "Background",
    ]

    # Check if any zone exceeds 66.6% of the total pixels
    if xp.any(percentage > 66.6):
        max_score_index = int(xp.argmax(percentage))
        return percentage, zone_names[max_score_index]

    # Else calculate wheighted score for each zone
    weights = xp.array([4, 3, 2, 1, 0])  # Weights for each zone
    scores = (zones * weights) / tissue_count
    max_score_index = int(xp.argmax(scores[:4]))  # ignore background zone

    return percentage, zone_names[max_score_index]


def mask_tile(tile, mask):
    """
    Masks the tile with the given mask. The mask is a binary image with the same dimensions as the tile. The function
    returns the masked tile as an Image, containing the tile where the mask is positive and white where it is negative.

    Args:
        tile (Image): Tile to be masked
        mask (Image): Mask to be applied to the tile

    Returns:
        Image: Masked tile
    """
    # Convert tile to numpy array
    tile_np = np.array(tile)
    mask_np = np.array(mask)

    if len(mask_np.shape) == 3:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
    # Resize mask to tile size if needed
    if mask_np.shape != tile_np.shape[:2]:
        mask_np = cv2.resize(
            mask_np,
            (tile_np.shape[1], tile_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    tumor_mask = binary_mask == 0

    binary_mask_inv = cv2.bitwise_not(binary_mask)
    binary_mask_inv_3ch = cv2.merge((binary_mask_inv, binary_mask_inv, binary_mask_inv))

    masked_tile = cv2.bitwise_and(tile_np, binary_mask_inv_3ch)
    white_bg = np.ones_like(tile_np) * 255
    masked_tile = np.where(binary_mask_inv_3ch == 0, white_bg, masked_tile)

    return Image.fromarray(masked_tile.astype(np.uint8)), tumor_mask


def separate_stains_and_save__tiles_as_tif(openslide_deepzoom, deepzoom_level, out_dir):
    """
    Iterates over an `openslide.DeepZoomGenerator` at the given `deepzoom_level` and separates hematoxylin, eosin and
    DAB stains for the from the original tile. The original tile and separated stains are saved as tif images in the
    specified `out_dir`.

    Args:
        openslide_deepzoom (DeepZoomGenerator): DeepZoomGenerator.
        deepzoom_level (int): Desired Deep Zoom level to iterate through.
        out_dir (str): Target directory in which tiles and separated stains are stored.
    """
    # Create directories for original tiles, hematoxylin stain, eosin stain
    # and DAB stain
    ORIGINAL_TILES_DIR = out_dir + "/original_tiles"
    os.makedirs(ORIGINAL_TILES_DIR, exist_ok=True)
    DAB_TILE_DIR = out_dir + "/DAB_tiles"
    os.makedirs(DAB_TILE_DIR, exist_ok=True)
    H_TILE_DIR = out_dir + "/H_tiles"
    os.makedirs(H_TILE_DIR, exist_ok=True)
    E_TILE_DIR = out_dir + "/E_tiles"
    os.makedirs(E_TILE_DIR, exist_ok=True)

    cols, rows = openslide_deepzoom.level_tiles[deepzoom_level - 1]
    for row in tqdm(range(rows)):
        for col in range(cols):
            tile_name = str(col) + "_" + str(row)

            temp = openslide_deepzoom.get_tile(deepzoom_level - 1, (col, row))
            temp_rgb = temp.convert("RGB")
            temp_np = np.array(temp_rgb)

            tiff.imsave(ORIGINAL_TILES_DIR + "/" + tile_name + "_original.tif", temp_np)

            # Now only process tiles that are mostly covered tiles
            if temp_np.mean() < 230 and temp_np.std() > 15:
                # print("Separating color for tile: ", tile_name)
                DAB, H, E = ihc_stain_separation(temp_np, True, True)

                # saving DAB,H,E in subdirectories
                tiff.imsave(DAB_TILE_DIR + "/" + tile_name + "_DAB.tif", DAB)
                tiff.imsave(H_TILE_DIR + "/" + tile_name + "_H.tif", H)
                tiff.imsave(E_TILE_DIR + "/" + tile_name + "_E.tif", E)
            else:
                pass
