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
        - If the tile is mostly white and thus shows no- or only little tissue, the tile will not be processed further
          The returned dict will contain the Tilename and a Flag = -1

        - Else the tile will be processed, including stain_separation and pixel_intensity calculations. The DAB Image
          will then be saved in passed directory and results of pixel intensity will be returned inside the dictionary.

    Args:
        iterable (iterable): Iterable containing the following Information on passed tile:

         - index 0: Column, necessary for naming
         - index 1: Row, necessary for naming
         - index 2: Tile itself, necessary since processes cannot access shared memory
         - DAB_TILE_DIR: Directory, for saving Image, since single processes cannot access shared memory


    Returns:
        dict: Dictionary containing tile results:
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
    Function receives IHC image, separates individual stains (Hematoxylin, Eosin, DAB) from image and returns an image
    for each of the individual stains.
    Credits: Credits for this function are owed to A. C. Ruifrok and D. A. Johnston with there paper “Quantification of
    histochemical staining by color deconvolution,” Analytical and quantitative cytology and histology / the
    International Academy of Cytology [and] American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001. PMID:
    11531144: https://scikit-image.orgdocs/stable/auto_examples/color_exposure/plot_ihc_color_separationhtml#sphx-glr-auto-examples-color-exposure-plot-ihc-color-separation-py

    Args:
        ihc_rgb (Image): IHC image in RGB

        hematoxylin(bool): Boolean, if True returns Hematoxylin image

        eosin (bool): Boolean, if True returns Eosin image

    Returns:
        Tuple:

        - ihc_h (Image): Hematoxylin staining of image if hematoxylin=True

        - ihc_e (Image): Eosin staining of image if eosin=True

        - ihc_d (Image): DAB (3',3'-Diaminobenzidine)

    """
    # convert RGB image to HED using prebuild skimage method
    ihc_hed = rgb2hed(ihc_rgb)

    # Create RGB image for each seperate stain
    # Convert to ubyte for easier saving to drive as image
    null = np.zeros_like(ihc_hed[:, :, 0])
    if hematoxylin:
        ihc_h = img_as_ubyte(
            hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
    else:
        ihc_h = None

    if eosin:
        ihc_e = img_as_ubyte(
            hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1)))
    else:
        ihc_e = None

    ihc_d = img_as_ubyte(
        hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))

    return (ihc_d, ihc_h, ihc_e)


def calculate_pixel_intensity(image):
    """
    Calculates pixel intensity of each pixel in the input image and separates them into 4 different zones based on
    their intensity. Intensity of each pixel lies between 0 and 255. Intensities above 235 are predominantly fatty
    tissues but dont contribute to pathological scoring:

        - Zone 1 = High positive (intensity: 0-60)
        - Zone 2 = Positive (intensity: 61-120)
        - Zone 3 = Low positive (intensity: 121-180)
        - Zone 4 = negative (intensity: 181-235)

    After calculating pixel intensities this function calculates percentage contribution of each of the zones as well
    as the a pathology score.

    Credits Varghese et al. (2014) "IHC Profiler: An Open Source Plugin for the Quantitative Evaluation and Automated
    Scoring of Immunihistochemistry Images of Human Tissue Samples"

    Args:
        image (Image): Input image

    Returns:
        Tuple:
        - hist (ndarray): histogram of the tile.
        - hist_centers(ndarray): center of bins of histogram
        - zones (ndarray): Number of pixels for each zone. During processing pixels in the tile are assigned to one of
          four zones based on pixel intensity. For more information see 'tile calculate_pixel_intensity'
        - percentage (ndarray): Percentage of pixels in each zone
        - score (ndarray): Score for each of the ties
        - pixel_count (int): Total number of pixels in the tile
        - image_analysis (ndarray): Pixelvalues of for positive pixels. Pixels with values ranging from 0 to 121 are
          considered positive.

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
    # iterates through each pixel in the image and assigns it to one of the
    # intensity zones
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
                # White space aka. fatty tissue needed for calculation in
                # respect to actual tissue
                zones[4] += 1
                pixelcount += 1

    percentage, score = calculate_score(zones, pixelcount)

    return hist, hist_centers, zones, percentage, score, pixelcount, img_analysis


def calculate_score(zones, count):
    """
    Calculates Percentage of amount of pixels in each zone in respect to
    pixelcount.
    TODO Check for Score and >66,6% px intensity

    Args:
        zones (ndarray): Array containing amount of pixels from each zone

        pixelcount (int): total pixelcount

    Returns:
        Tuple:
        - percentage (ndarray): Array containing percentage of pixels in each zone.
        - score (ndarray): Array containing calculation of score for each zone

    """
    if count == 0:
        raise ZeroDivisionError("Count cannot be zero")
    percentage = np.zeros(zones.size)
    score = np.zeros(zones.size)
    for i in range(zones.size):
        percentage[i] = (zones[i] / count) * 100
        score[i] = (zones[i] * (zones.size - (i + 1))) / count

    return percentage, score


def mask_tile(tile, mask):
    """
    This function takes a tile and a mask and masks the tile with the mask. The mask is a binary image with the same
    dimensions as the tile. The function returns the masked tile as Image, containing the tile where the mask is positive and white where it is negative.

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
    deepzoom_object, deepzoom_level, target_directory
):
    """
    This function iterates through OpenSlide Deep Zoom Object at the given
    Deep Zoom level and saves each individual slide in the given Directory
    named as "col_row".tif just as "save_all_tiles_as_tif" does. However,
    additionally this function calls the "color_separation" function,
    separates H-,E-,DAB stains from each tile and saves an additional tif for
    each stain in a newly created subdirectory respectively.

    Args:
        deepzoom_object (DeepZoomGenerator): DeepZoomGenerator
        deepzoom_level (int): Wanted Deep Zoom level through which
            shall be iterated
        target_directory (str): Target directory in which tiles are stored

    """
    # Create directories for original tiles, hematoxylin stain, eosin stain
    # and DAB stain
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

            tiff.imsave(ORIGINAL_TILES_DIR + "/"
                        + tile_name + "_original.tif", temp_np)
            # print("Saving tile:" + tile_name)

            # Now only process tiles that are mostly covered and not blank to
            # save runtime and space
            if temp_np.mean() < 230 and temp_np.std() > 15:
                # print("Separating color for tile: ", tile_name)
                DAB, H, E = ihc_stain_separation(temp_np, True, True)

                # saving DAB,H,E in subdirectories
                tiff.imsave(DAB_TILE_DIR + "/" + tile_name + "_DAB.tif", DAB)
                tiff.imsave(H_TILE_DIR + "/" + tile_name + "_H.tif", H)
                tiff.imsave(E_TILE_DIR + "/" + tile_name + "_E.tif", E)
            else:
                pass
                # print("NOT PROCESSING TILE", tile_name)
