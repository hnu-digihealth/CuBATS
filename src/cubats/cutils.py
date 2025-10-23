# Standard Library
import os
import re

# Third Party
import numpy as np
from PIL import Image


def get_name(f):
    """
    Returns Object name, and removes image type extension from filename.

    Args:
        f (String): Path to file

    Returns:
        String: Image Filename without extension
    """
    if not isinstance(f, str):
        raise ValueError("Input must be a string representing the file path.")

    if not f:
        raise ValueError("Input string is empty.")

    if re.search(r"\.", f) is None:
        return f
    f = os.path.split(f)[-1]

    if f.endswith(".ome.tiff") or f.endswith(".ome.tif"):
        back_slice_idx = 2
    else:
        back_slice_idx = 1
    img_name = "".join([".".join(f.split(".")[:-back_slice_idx])])

    return img_name


def get_score_name(score):
    """
    Returns name of Zone with the highest score.

    Args:
        score (list): List containing score for each zone

    Returns:
        String: Name of highest zone with highest score
    """
    if not isinstance(score, list):
        raise ValueError("Input must be a list of scores.")

    if not score:
        raise ValueError("Input list is empty.")

    if not all(isinstance(s, (int, float)) for s in score):
        raise ValueError("All elements in the score list must be numeric.")

    if any(s < 0 for s in score):
        raise ValueError("All elements in the score list must be positive.")

    zone_names = ["High Positive", "Positive", "Low Positive", "Negative"]
    if len(score) < 4:
        raise ValueError(
            f"Score list must contain exactly {len(zone_names)} elements.")

    max = np.max(score[:4])
    score = zone_names[score.index(max)]

    return score


def to_numpy(array):
    """
    Converts an array to NumPy format if necessary.

    Args:
        array: The input array (NumPy or CuPy).

    Returns:
        np.ndarray: The converted NumPy array.
    """
    # Check if the input is a CuPy array and convert it to NumPy
    # Try CuPy import, if it fails, assume array is not a CuPy array
    try:
        # Third Party
        import cupy as cp

        return cp.asnumpy(array)
    except ImportError:
        pass

    # Not CuPy — assume NumPy or NumPy-compatible
    return np.asarray(array)


def downsample_Openslide_to_PIL(openslide_object, SCALEFACTOR: int):
    """
    This function takes an Openslide Object as input and downscales it based on the Slides optimal level for
    downsampling and the given Scalefactor. IT returns a PIL Image as well as downsample parameters.

    Args:
        openslide_object (openslide.OpenSlide): The Object that needs to be downscaling
        SCALEFACTOR (int): Factor for downscaling

    Returns:
        img (PIL.Image): rescaled Image
        old_w (int): width of input Openslide Object
        old_h (int): height of input Openslide Object
        new_w (int): width of output Image
        new_h (int): height of output Image
    """
    if not hasattr(openslide_object, "dimensions") or not hasattr(
        openslide_object, "read_region"
    ):
        raise ValueError("Invalid Openslide object.")

    if not isinstance(SCALEFACTOR, int) or SCALEFACTOR <= 0:
        raise ValueError("SCALEFACTOR must be a positive integer.")

    old_w, old_h = openslide_object.dimensions
    # rescaled width and height of Image
    new_w, new_h = old_w // SCALEFACTOR, old_h // SCALEFACTOR

    if new_w < 1 or new_h < 1:
        new_w, new_h = 1, 1

    # Find optimal level for downsampling
    level = openslide_object.get_best_level_for_downsample(SCALEFACTOR)
    # Conversion to PIL image
    wsi = openslide_object.read_region(
        (0, 0), level, openslide_object.level_dimensions[level]
    )
    wsi = wsi.convert("RGB")
    img = wsi.resize((new_w, new_h), Image.LANCZOS)

    return img, old_w, old_h, new_w, new_h
