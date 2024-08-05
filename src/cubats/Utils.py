# Standard Library
import math
import os
import re

# Third Party
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
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


def plot_tile_quantification_results(self, tilename, img_true=True, numeric=True):
    """This function plots quantification results of a given tilename. It plots the DAB-image, the histogram of the intensity distribution,
        a bar plot containing the amount of pixels attributed to each zone and numeric results. Display of the image and numeric results are
        optional, default is set to True. Images can only be display if they have been saved during quantification in functions:
        - quantify_all_slides
        - quantify_single_slide

    Args:
        tilename (str): name of tile (col_row)
        img_true (bool, optional): Plot DAB image. Defaults to True.
        numeric (bool, optional): include numeric information. Defaults to True.
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
                    self.get_score_name(
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
