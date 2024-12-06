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


def quantify_tile(iterable):
    """
    This function processes a single input tile and returns a dictionary.
    If the tile contains tissue (mean pixel value < 230 and standard deviation > 15), it will undergo stain separation
    and pixel intensity calculations. The results will be returned in the dictionary. The flag will be set to 1. If
    save_img is True the DAB image will additionally be saved in the specified directory.
    If the tile is mostly white (mean pixel value >= 230 or standard deviation <= 15), it will not be processed
    further. The returned dictionary will contain the tile name and a flag set to 0.

    Args:
        iterable (iterable): Iterable containing the following Information on passed tile:

            - index 0: Column, necessary for naming
            - index 1: Row, necessary for naming
            - index 2: Tile itself, necessary since processes cannot access shared memory
            - DAB_TILE_DIR: Directory, for saving Image, since single processes cannot access shared memory
            - save_img: Boolean, if True, DAB image will be saved in specified directory


    Returns:
        dict: Dictionary containing tile results:

            - Tilename (str): Name of the tile.
            - Histogram (ndarray): Histogram of the tile.
            - Hist_centers (ndarray): Centers of histogram bins.
            - Zones (ndarray): Number of pixels in each intensity zone.
            - Percentage (ndarray): Percentage of pixels in each zone.
            - Score (ndarray): Score for the tile.
            - Px_count (int): Total number of pixels in the tile.
            - Flag (int): Processing flag (1 if processed, 0 if not).
            - Image Array (ndarray): Array of pixel values for positive pixels.

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

    # Only process tiles that are mostly covered and not blank to save runtime
    # and space
    if temp_np.mean() < 230 and temp_np.std() > 15:

        # Separate stains
        DAB, H, E = ihc_stain_separation(temp_np)

        # Calculate pixel intensity
        (hist,
         hist_centers,
         zones,
         percentage,
         score,
         pixelcount,
         img_analysis) = (
            calculate_pixel_intensity(DAB)
        )

        # Save image as tif in passed directory if wanted.
        if save_img:
            if not DAB_TILE_DIR:
                raise ValueError(
                    "Target directory must be specified if save_img is True")
            img = Image.fromarray(DAB)
            DAB_TILE_DIR = f"{DAB_TILE_DIR}/{tile_name}.tif"
            # print(DAB_TILE_DIR)
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
    # convert RGB image to HED using prebuild skimage method
    ihc_hed = rgb2hed(ihc_rgb)

    # Create RGB image for each seperate stain
    null = np.zeros_like(ihc_hed[:, :, 0])
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = img_as_ubyte(hed2rgb(
        np.stack((ihc_hed[:, :, 0], null, null), axis=-1))) if hematoxylin else None
    ihc_e = img_as_ubyte(
        hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))) if eosin else None
    ihc_d = img_as_ubyte(
        hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))

    return ihc_d, ihc_h, ihc_e


def calculate_pixel_intensity(image):
    """
    Calculates pixel intensity of each pixel in the input image and separates them into 5 different zones based on
    their intensity. The image is converted to grayscale format, resulting in a distribution of intensity values
    between 0-255. Intensities above 235 are predominantly background or fatty tissues and do not contribute to
    pathological scoring:

        - Zone 1: High positive (intensity: 0-60)
        - Zone 2: Positive (intensity: 61-120)
        - Zone 3: Low positive (intensity: 121-180)
        - Zone 4: Negative (intensity: 181-235)
        - Zone 5: Background or fatty tissues (intensity: 236-255)

    After calculating pixel intensities this function calculates percentage contribution of each of the zones as well
    as the a pathology score.

    Args:
        image (Image): Input image.

    Returns:
        tuple: A tuple containing:
            - hist (ndarray): Histogram of the image.
            - hist_centers (ndarray): Centers of histogram bins.
            - zones (ndarray): Number of pixels in each intensity zone.
            - percentage (ndarray): Percentage of pixels in each intensity zone.
            - score (float): Calculated score for the image.
            - pixelcount (int): Total number of pixels in the image.
            - img_analysis (ndarray): Array of pixel values colocalization.
    """

    # Conversion to gray-scale-ubyte image
    gray_scale_image = rgb2gray(image)
    gray_scale_ubyte = img_as_ubyte(gray_scale_image)
    # Calculates a histogram of the input image
    hist, hist_centers = histogram(gray_scale_image)

    w, h = gray_scale_ubyte.shape

    # array containg only high-, positive & low positive pixels
    img_analysis = np.full((w, h), 255, dtype="uint8")

    # Array for Zones of pixel intensity
    zones = np.zeros(5)
    # pixelcount
    pixelcount = 0
    # Assigns each pixel to an intensity zone
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
            else:
                # Background or fatty tissue needed for calculation with respect to actual tissue
                zones[4] += 1
                pixelcount += 1

    percentage, score = calculate_percentage_and_score(zones, pixelcount)

    return hist, hist_centers, zones, percentage, score, pixelcount, img_analysis


def calculate_percentage_and_score(zones, pixel_count):
    """
    Calculates the percentage of pixels in each zone relative to the total pixel count and computes a score for each
    zone. If more than 66.6% of the total pixels are attributed to a single zone, that zone's score is assigned. Else,
    the score for each zone is calculated using this formula:

    .. math::

        \\text{Score} = \\frac{(\\text{number of pixels in zone} \\times \\text{weight of zone})}{\\text{total
        pixels in image}}

    with weights 4 for the high positive zone, 3 for the positive zone, 2 for the low positive zone, 1 for the negative
    zone, and 0 for the background. The final score is the maximum score among all zones.

    Args:
        zones (ndarray): Array containing amount of pixels from each zone
        pixelcount (int): Total number of pixels.

    Returns:
        tuple: A tuple containing:
            - percentage (ndarray): Array containing the percentage of pixels in each zone.
            - score (str): Name of the zone if it exceeds 66.6%, otherwise the name of the zone with the highest score.

    Raises:
        ZeroDivisionError: If pixel count is zero.
        ValueError: If all zones have zero pixels.
    """
    if pixel_count == 0:
        raise ZeroDivisionError("Count cannot be zero")

    if np.sum(zones) == 0:
        raise ValueError("All zones have zero pixels")

    percentage = np.zeros(zones.size)
    zone_names = ["High Positive", "Positive",
                  "Low Positive", "Negative", "Background"]
    weights = [4, 3, 2, 1, 0]  # Weights for each zone

    for i in range(zones.size):
        percentage[i] = (zones[i] / pixel_count) * 100

    # Check if any zone exceeds 66.6%
    for i in range(zones.size):
        if percentage[i] > 66.6:
            return percentage, zone_names[i]

    # Else calculate the score for each zone and determine highest score
    scores = [(zones[i] * weights[i]) / pixel_count for i in range(zones.size)]
    max_score_index = np.argmax(scores[:4])

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
    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    binary_mask_inv = cv2.bitwise_not(binary_mask)
    binary_mask_inv_3ch = cv2.merge(
        (binary_mask_inv, binary_mask_inv, binary_mask_inv))

    masked_tile = cv2.bitwise_and(tile_np, binary_mask_inv_3ch)
    white_bg = np.ones_like(tile_np) * 255
    masked_tile = np.where(binary_mask_inv_3ch == 0, white_bg, masked_tile)

    return Image.fromarray(masked_tile.astype(np.uint8))


def separate_stains_and_save__tiles_as_tif(
    openslide_deepzoom, deepzoom_level, out_dir
):
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

            tiff.imsave(ORIGINAL_TILES_DIR + "/"
                        + tile_name + "_original.tif", temp_np)

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
