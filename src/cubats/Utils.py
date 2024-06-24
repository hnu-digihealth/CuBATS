# Standard Library
import math
import os
import re

# Third Party
import numpy as np
from PIL import Image


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
