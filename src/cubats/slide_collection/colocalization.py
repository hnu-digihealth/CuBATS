# Standard Library

# Third Party
import numpy as np
from PIL import Image

# CuBATS


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
        # TODO check if shape is the same for both images
        h, w = img1["Image Array"].shape
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
                        # Color red where both img Positive
                        img[x, y] = [254, 0, 0]
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
                        # color blue where only img2 positive
                        img[x, y] = [0, 30, 255]
                    elif (
                        img1["Image Array"][x, y] < 121
                        and img2["Image Array"][x, y] < 121
                    ):
                        pos_overlap += 1
                        # Color red where both img Positive
                        img[x, y] = [254, 0, 0]
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
                        # Color red where both img Positive
                        img[x, y] = [254, 0, 0]
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
                        # color white where both negative
                        img[x, y] = [255, 255, 255]
                    else:
                        background += 1

            img = Image.fromarray(img.astype("uint8"))
            out = f"{dir}/{tilename}.tif"
            img.save(out)
        else:
            for y in range(h):
                for x in range(w):
                    pixel_values = [img1["Image Array"]
                                    [x, y], img2["Image Array"][x, y]]
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
                    pixel_values = [img["Image Array"][x, y]
                                    for img in [img1, img2, img3]]
                    sum_high = sum(1 for val in pixel_values if val < 61)
                    sum_pos = sum(1 for val in pixel_values if val < 121)
                    sum_low = sum(1 for val in pixel_values if val < 181)

                    if sum_high >= 2:
                        high_overlap += 1
                        img[x, y] = [254, 0, 0]
                    elif sum_high == 1:
                        idx = pixel_values.index(min(pixel_values))
                        high_complement += 1
                        img[x, y] = [[116, 238, 21], [
                            0, 30, 255], [255, 231, 0]][idx]
                    elif sum_pos >= 2:
                        pos_overlap += 1
                        img[x, y] = [254, 0, 0]
                    elif sum_pos == 1:
                        idx = pixel_values.index(min(pixel_values))
                        pos_complement += 1
                        img[x, y] = [[116, 238, 21], [
                            0, 30, 255], [255, 231, 0]][idx]
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
                    pixel_values = [img["Image Array"][x, y]
                                    for img in [img1, img2, img3]]

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
