# Third Party
import numpy as np
from PIL import Image


def compute_dual_antigen_colocalization(iterable):
    """ Analyzes antigen colocalization in two tiles and returns the results in a dictionary.

    This function analyzes the same tile from 2 different WSIs. Using pixel-wise comparison it iterates over the two
    images and calculates a total coverage, overlapping & complementary coverage, as well as the contribution of the
    different intensity levels to both. The intensity levels are divided into 5 categories based on their intensity
    values. Overall positivity is defined as intensity value < 181, with further subdivisions into highly positive
    (intensity value 0-60), positive (intensity value 61-120), and low positive (intensity value 121-180). Negative
    pixels are defined as intensity value 181-234, and background pixels are defined as 235-255. The results are then
    saved in a dictionary and returned. Additionally, an image containing the colored results can be saved in the
    passed directory.

    Args:
        iterable (iterable): Contains the 2 Tiles(iterable[0:2]) that are to be compared and analyzed for antigen
        coverage as well as the output directory:
            - iterable[0]: tile1
            - iterable[1]: tile2
            - iterable[2]: output path
            - iterable[3]: save_img

    Returns:
        dict: returns a dictionary containing the results of the analysis:
            - Tilename: Name of the tile

            - Flag: Flag indicating if the tile was processed or not. ((1): processed; (-1): not processed due both
              tiles not being processable; (-2): one of the tiles was not processable due to wrong shape)

            - Total Coverage: Total amount of positive pixels (intensity values < 181)(in %) that both tiles cover combined in respect to actual
              tissue in tiles, regardless whether they are overlapping or complemetary (Only if Flag == 1)
            - Total Overlap: Total amount of positive pixels (in %) that are positive in both tiles and thus overlapping. (Only if Flag == 1)
            - Total Complement: Total amount of positive pixels (in %) that are only covered by one of the 2 tiles (Only if Flag == 1)
            - High Positive Overlap: Amount of highly positive pixels (in %) that are covered by both tiles combined (Only if Flag == 1)
            - High Positive Complement: Amount of highly positive pixels (in %) that are only covered by one of the 2 tiles (Only if Flag == 1)
            - Positive Overlap: Amount of positive pixels (in %) that are covered by both tiles combined (Only if Flag == 1)
            - Positive Complement: Amount of positive pixels (in %) that are only covered by one of the 2 tiles (Only if Flag == 1)
            - Low Positive Overlap: Amount of low positive pixels (in %) that are covered by both tiles combined (Only if Flag == 1)
            - Low Positive Complement: Amount of low positive pixels (in %) that are only covered by one of the 2 tiles (Only if Flag == 1)
            - Negative: Amount of negative pixels (in %) that are not covered by any of the 2 tiles (Only if Flag == 1)
            - Tissue: Amount of the tissue (in %) that both tiles cover combined. (Only if Flag == 1)
            - Background / No Tissue: Amount of the tile that is not covered by tissue in either tile (Only if Flag == 1)

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory. The coloring scheme is as follows:
            - Overlapping pixels (positive in both tiles):
                - Highly positive: Red ([255, 0, 0])
                - Positive: Red ([255, 0, 0])
                - Low positive: Red ([255, 0, 0])
            - Complementary pixels (positive in only one tile):
                - Tile1 highly positive: Green ([0, 255, 0])
                - Tile1 positive: Light Green ([153, 238, 153])
                - Tile1 low positive: Very Light Green ([204, 255, 204])
                - Tile2 highly positive: Blue ([0, 0, 204])
                - Tile2 positive: Light Blue ([102, 204, 255])
                - Tile2 low positive: Very Light Blue ([153, 204, 255])
            - Negative pixels (both tiles negative): White ([255, 255, 255])
            - Background pixels (no tissue): Gray ([192, 192, 192])
    """

    tile1 = iterable[0]
    tile2 = iterable[1]
    dir = iterable[2]
    save_img = iterable[3]

    tilename = tile1["Tilename"]
    colocal_dict = {}
    colocal_dict["Tilename"] = tilename
    # Check if both Images contain tissue.If both dont: Flag = -1
    if tile1["Flag"] == 0 and tile2["Flag"] == 0:
        colocal_dict["Flag"] = -1
        return colocal_dict
    # If tile1 contains tissue and tile2 doesn't: Process tile1 only. If save_img is True, coloring for tile1 is green.
    elif tile1["Flag"] == 0:
        colocal_dict["Flag"] = 1
        high_complement, pos_complement, low_complement, negative, background = process_single_tile(
            tile2, save_img, dir, [
                [0, 0, 204], [102, 102, 255], [153, 153, 255]]
        )
        high_overlap, pos_overlap, low_overlap, = (0, 0, 0)
    # If tile2 contains tissue and tile1 doesn't: Process tile2 only. If save_img is True, coloring for tile2 is blue.
    elif tile2["Flag"] == 0:
        colocal_dict["Flag"] = 1
        high_complement, pos_complement, low_complement, negative, background = process_single_tile(
            tile1, save_img, dir, [
                [0, 255, 0], [102, 255, 102], [153, 255, 153]]
        )
        high_overlap, pos_overlap, low_overlap, = (0, 0, 0)
    # Check if Images have the correct shape. If one doesn't: Flag = -2
    elif tile1["Image Array"].shape != (1024, 1024) or tile2["Image Array"].shape != (
        1024,
        1024,
    ):
        colocal_dict["Flag"] = -2
        return colocal_dict
    # If both contain tissue: Process both tiles
    else:
        colocal_dict["Flag"] = 1
        high_overlap, pos_overlap, low_overlap, high_complement, pos_complement, low_complement, negative, background = process_two_tiles(
            tile1, tile2, save_img, dir
        )
    # Sum up the results
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

    # Vals in % for overlap, complement, negative in respect to entire image. Except background with respect to total pixel count
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
    tissue_count = round((tissue_count / 1048576) * 100, 4)
    background = round((background / 1048576) * 100, 4)

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
    colocal_dict["Tissue"] = tissue_count
    colocal_dict["Background / No Tissue"] = background

    return colocal_dict


def process_single_tile(tile, save_img, dir, color):
    """
    Processes a single tile, categorizing pixels into different complement levels and optionally saving the processed
    image. This function is called if only one of the two tiles contains tissue. The coloring of the image can be
    specified by the color parameter and therefore depends on the input. This function is typically called by the
    function compute_dual_antigen_colocalization with either tile1 or tile2 as input. If tile1 is the input, the
    coloring will be variations of green ([0, 255, 0], [153, 238, 153], [204, 255, 204]). If tile2 is the input, the
    coloring will be variations of blue ([0, 0, 204], [102, 204, 255], [153, 204, 255]).

    Args:
        tile (dict): A dictionary containing the image under the key "Image Array".
        save_img (bool): A flag indicating whether create a colored image and save it.
        dir (str): The directory where the image should be saved if save_img is True.
        color (list): A list of RGB color values to be used for different complement levels.

    Returns:
        tuple: A tuple containing counts of pixels in the following order:
            - high_complement (int): Count of pixels with values less than 61.
            - pos_complement (int): Count of pixels with values between 61 and 120.
            - low_complement (int): Count of pixels with values between 121 and 180.
            - negative (int): Count of pixels with values between 181 and 234.
            - background (int): Count of pixels with values 235 and above.

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory. The coloring scheme is depends on the color parameter.
    """
    high_complement = 0
    pos_complement = 0
    low_complement = 0
    negative = 0
    background = 0
    h, w = tile["Image Array"].shape

    if save_img:
        tilename = tile["Tilename"]
        img_out = np.full((h, w, 3), 192, dtype="uint8")
        for y in range(h):
            for x in range(w):
                pixel = tile["Image Array"][x, y]
                if pixel < 61:
                    high_complement += 1
                    img_out[x, y] = color[0]
                elif pixel < 121:
                    pos_complement += 1
                    img_out[x, y] = color[1]
                elif pixel < 181:
                    low_complement += 1
                    img_out[x, y] = color[2]
                elif pixel < 235:
                    negative += 1
                    img_out[x, y] = [255, 255, 255]
                else:
                    background += 1
        img_out = Image.fromarray(img_out.astype("uint8"))
        out = f"{dir}/{tilename}.tif"
        img_out.save(out)
    else:
        for y in range(h):
            for x in range(w):
                pixel = tile["Image Array"][x, y]
                if pixel < 61:
                    high_complement += 1
                elif pixel < 121:
                    pos_complement += 1
                elif pixel < 181:
                    low_complement += 1
                elif pixel < 235:
                    negative += 1
                else:
                    background += 1

    return high_complement, pos_complement, low_complement, negative, background


def process_two_tiles(tile1, tile2, save_img, dir):
    """
    Processes two image tiles and calculates various overlap and complement metrics.

    Args:
        tile1 (dict): A dictionary containing the first tile's data, including "Image Array".
        tile2 (dict): A dictionary containing the second tile's data, including "Image Array".
        save_img (bool): A flag indicating whether to save the output image.
        dir (str): The directory where the output image should be saved if save_img is True.

    Returns:
        tuple: A tuple containing the counts of:
            - high_overlap (int): Number of pixels where both tiles are highly positive.
            - pos_overlap (int): Number of pixels where both tiles are positive.
            - low_overlap (int): Number of pixels where both tiles are low positive.
            - high_complement (int): Number of pixels where one tile is highly positive and the other is not.
            - pos_complement (int): Number of pixels where one tile is positive and the other is not.
            - low_complement (int): Number of pixels where one tile is low positive and the other is not.
            - negative (int): Number of pixels where either tile is negative.
            - background (int): Number of background pixels.

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory. The coloring scheme is as follows:
            - Overlapping pixels (positive in both tiles):
                - Highly positive: Red ([255, 0, 0])
                - Positive: Red ([255, 0, 0])
                - Low positive: Red ([255, 0, 0])
            - Complementary pixels (positive in only one tile):
                - Tile1 highly positive: Green ([0, 255, 0])
                - Tile1 positive: Light Green ([153, 238, 153])
                - Tile1 low positive: Very Light Green ([204, 255, 204])
                - Tile2 highly positive: Blue ([0, 0, 204])
                - Tile2 positive: Light Blue ([102, 204, 255])
                - Tile2 low positive: Very Light Blue ([153, 204, 255])
            - Negative pixels (both tiles negative): White ([255, 255, 255])
            - Background pixels (no tissue): Gray ([192, 192, 192])
    """

    high_overlap = 0
    pos_overlap = 0
    low_overlap = 0
    high_complement = 0
    pos_complement = 0
    low_complement = 0
    negative = 0
    background = 0
    h, w = tile1["Image Array"].shape

    if save_img:
        tilename = tile1["Tilename"]
        img_out = np.full((h, w, 3), 192, dtype="uint8")
        for y in range(h):
            for x in range(w):
                pixel1 = tile1["Image Array"][x, y]
                pixel2 = tile2["Image Array"][x, y]

                # Check if both pixels are positive
                if pixel1 < 181 and pixel2 < 181:
                    if pixel1 < 61 and pixel2 < 61:
                        high_overlap += 1
                        # Color strong red where both are highly positive
                        img_out[x, y] = [255, 0, 0]
                    elif pixel1 < 121 and pixel2 < 121:
                        pos_overlap += 1
                        # Color strong red where both are positive
                        img_out[x, y] = [255, 0, 0]
                    else:
                        low_overlap += 1
                        # Color light red where both are low positive
                        img_out[x, y] = [255, 0, 0]
                # Check if pixel1 is positive and pixel2 is not
                elif pixel1 < 181 and pixel2 > 180:
                    if pixel1 < 61:
                        high_complement += 1
                        # Color strong green if img1 is highly positive
                        img_out[x, y] = [0, 255, 0]
                    elif pixel1 < 121:
                        pos_complement += 1
                        # Color bright green if img1 is positive
                        img_out[x, y] = [153, 238, 153]
                    else:
                        low_complement += 1
                        # Color light green if img1 is low positive
                        img_out[x, y] = [204, 255, 204]
                # Check if pixel2 is positive and pixel1 is not
                elif pixel1 > 180 and pixel2 < 181:
                    if pixel2 < 61:
                        high_complement += 1
                        # Color strong blue if img2 is highly positive
                        img_out[x, y] = [0, 0, 204]
                    elif pixel2 < 121:
                        pos_complement += 1
                        # Color bright blue if img2 is positive
                        img_out[x, y] = [102, 204, 255]
                    else:
                        low_complement += 1
                        # Color light blue if img2 is low positive
                        img_out[x, y] = [153, 204, 255]
                elif pixel1 < 235 or pixel2 < 235:
                    negative += 1
                    # Color white where either are negative
                    img_out[x, y] = [255, 255, 255]
                else:
                    # Background Pixels stay gray
                    background += 1

        img_out = Image.fromarray(img_out.astype("uint8"))
        out = f"{dir}/{tilename}.tif"
        img_out.save(out)
    else:
        for y in range(h):
            for x in range(w):
                pixel1 = tile1["Image Array"][x, y]
                pixel2 = tile2["Image Array"][x, y]

                # Check if both pixels are positive
                if pixel1 < 181 and pixel2 < 181:
                    if pixel1 < 61 and pixel2 < 61:
                        high_overlap += 1
                    elif pixel1 < 121 and pixel2 < 121:
                        pos_overlap += 1
                    else:
                        low_overlap += 1
                # Check if one pixel is positive and the other is not
                elif (pixel1 < 181 and pixel2 > 180) or (pixel1 > 180 and pixel2 < 181):
                    if (pixel1 < 61 and pixel2 > 180) or (pixel2 < 61 and pixel1 > 180):
                        high_complement += 1
                    elif (pixel1 < 121 and pixel2 > 180) or (pixel2 < 121 and pixel1 > 180):
                        pos_complement += 1
                    else:
                        low_complement += 1
                elif pixel1 < 235 or pixel2 < 235:
                    negative += 1
                else:
                    background += 1

    return high_overlap, pos_overlap, low_overlap, high_complement, pos_complement, low_complement, negative, background


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
            img = np.full((h, w, 3), 192, dtype="uint8")
            for y in range(h):
                for x in range(w):
                    pixel_values = [img["Image Array"][x, y]
                                    for img in [img1, img2, img3]]
                    sum_high = sum(1 for val in pixel_values if val < 61)
                    sum_pos = sum(1 for val in pixel_values if val < 121)
                    sum_low = sum(1 for val in pixel_values if val < 181)
                    sum_neg = sum(1 for val in pixel_values if val < 235)

                    # Check if 2 or all 3 pixels are positive
                    if sum_low >= 2:
                        # Check if 2 or all 3 pixels are highly positive
                        if sum_high >= 2:
                            high_overlap += 1
                            # Color strong red where all are highly positive
                            img[x, y] = [255, 0, 0]
                        # Check if 2 or all 3 pixels are positive
                        elif sum_pos >= 2:
                            pos_overlap += 1
                            # Color strong red where all are positive
                            img[x, y] = [255, 0, 0]
                        # Else 2 or all 3 pixels are low positive
                        else:
                            low_overlap += 1
                            # Color light red where all are low positive
                            img[x, y] = [255, 0, 0]
                    # Check if 1 pixel is positive while the other 2 are not
                    elif sum_low == 1:
                        # Check if the pixel is highly positive
                        if sum_high == 1:
                            idx = pixel_values.index(min(pixel_values))
                            high_complement += 1
                            # Colors: img1: Strong Green, img2: Strong Blue, img3: Strong Orange
                            img[x, y] = [[0, 255, 0], [
                                0, 0, 255], [255, 165, 0]][idx]
                        # Check if the pixel is positive
                        elif sum_pos == 1:
                            idx = pixel_values.index(min(pixel_values))
                            pos_complement += 1
                            # Colors: img1: Bright Green, img2: Bright Blue, img3: Bright Orange
                            img[x, y] = [[102, 255, 102], [
                                102, 102, 255], [255, 200, 102]][idx]
                        # Else the pixel is low positive
                        else:
                            idx = pixel_values.index(min(pixel_values))
                            low_complement += 1
                            # Colors: img1: Light Green, img2: Light Blue, img3: Light Orange
                            img[x, y] = [[153, 255, 153], [
                                153, 153, 255], [255, 225, 153]][idx]
                    # Check if any of the pixels are negative
                    elif sum_neg > 0:
                        negative += 1
                        # Color white where either are negative
                        img[x, y] = [255, 255, 255]
                    else:
                        # Else pixel is background: stay gray
                        background += 1

            img = Image.fromarray(img.astype("uint8"))
            out = f"{dir}/{tilename}.tif"
            img.save(out)
        else:
            for y in range(h):
                for x in range(w):
                    pixel_values = [img["Image Array"][x, y]
                                    for img in [img1, img2, img3]]
                    sum_high = sum(1 for val in pixel_values if val < 61)
                    sum_pos = sum(1 for val in pixel_values if val < 121)
                    sum_low = sum(1 for val in pixel_values if val < 181)
                    sum_neg = sum(1 for val in pixel_values if val < 235)

                    # Check if 2 or all 3 pixels are positive
                    if sum_low >= 2:
                        # Check if 2 or all 3 pixels are highly positive
                        if sum_high >= 2:
                            high_overlap += 1
                        # Check if 2 or all 3 pixels are positive
                        elif sum_pos >= 2:
                            pos_overlap += 1
                        # Else 2 or all 3 pixels are low positive
                        else:
                            low_overlap += 1
                    # Check if 1 pixel is positive while the other 2 are not
                    elif sum_low == 1:
                        # Check if the pixel is highly positive
                        if sum_high == 1:
                            high_complement += 1
                        # Check if the pixel is positive
                        elif sum_pos == 1:
                            pos_complement += 1
                        # Else the pixel is low positive
                        else:
                            low_complement += 1
                    # Check if any of the pixels are negative
                    elif sum_neg > 0:
                        negative += 1
                    else:
                        # Else pixel is background
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

        # Vals in % for overlap, complement, negative in respect to entire image. Except background with respect to total pixel count
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
        tissue_count = round((tissue_count / (h * w)) * 100, 4)
        background = round((background / (h * w)) * 100, 4)

        # set dict TODO add tissue Count for clarity
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
        colocal_dict["Tissue"] = tissue_count
        colocal_dict["Background / No Tissue"] = background

    return colocal_dict
