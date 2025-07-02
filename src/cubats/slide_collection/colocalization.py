# Third Party
import numpy as np
from PIL import Image

# CuBATS
from cubats.config import xp
from cubats.cutils import to_numpy

# Define color schemes for each tile
COLOR_OVERLAP = [255, 0, 0]  # Strong Red
COLOR_TILE1 = [[0, 255, 0], [102, 255, 102], [153, 255, 153]]  # Green shades
COLOR_TILE2 = [[0, 0, 204], [102, 204, 255], [153, 204, 255]]  # Blue shades
COLOR_TILE3 = [[255, 165, 0], [255, 200, 102], [255, 225, 153]]  # Orange shades
COLOR_NEGATIVE = [255, 255, 255]  # White
COLOR_BACKGROUND = [192, 192, 192]  # Gray


def analyze_dual_antigen_colocalization(iterable):
    """
    This function analyzes antigen colocalization in the same tile from 2 different WSIs and returns the results as a
    dictionary.

    The function performs pixel-wise comparison across two images to calculate total coverage, overlapping coverage,
    and complementary coverage. The intensity levels are divided into five categories based on their intensity values.
    Intensity values are defined for each tile by the passed antigen profile which can be defined when initializing the
    SlideCollection. If not antigen profiles are defined, intensity values will fall back to default thresholds:
    high positive (0-60), medium positive (61-120), low positive (121-180), negative (181-234),
    and background (235-255). The results are saved in a dictionary and returned. Optionally, an image containing the
    colored results can be saved in the specified directory.

    Args:
        iterable (iterable): An iterable containing the following elements:
            'tile1', the first tile to be compared and analyzed for antigen coverage;
            'tile2', the second tile to be compared and analyzed for antigen coverage;
            Antigen-specific profiles: 'profile1' and 'profile2' containing antigen-specific thresholds for
            tile1 and tile2 respectively;
            'output path', the directory where the output image should be saved;
            'save_img', a flag indicating whether to save the output image.

    Returns:
        dict: A dictionary containing the results of the analysis:
            - Tilename: Name of the tile
            - Flag: Flag indicating if the tile was processed or not. (1: processed; -1: not processed due to both
              tiles not being processable; -2: one of the tiles was not processable due to wrong shape)
            - Total Coverage: Percentage of positive pixels (intensity values < 181), covered by both tiles combined,
              with respect to actual tissue in the images. (Only if Flag == 1)
            - Total Overlap: Percentage of positive pixels that are positive in both images combined, with respect to
              actual tissue in the images. (Only if Flag == 1)
            - Total Complement: Percentage of positive pixels that are only covered by one of the two images, with
              respect to actual tissue in the images. (Only if Flag == 1)
            - High Positive Overlap: Percentage of highly positive pixels covered by both tiles combined, with respect
              to actual tissue in the images. (Only if Flag == 1)
            - High Positive Complement: Percentage of highly positive pixels covered by only one of the two tiles, with
              respect to actual tissue in the images. (Only if Flag == 1)
            - Medium Positive Overlap: Percentage of medium positive pixels covered by both tiles combined, with
              respect to actual tissue in the images. (Only if Flag == 1)
            - Medium Positive Complement: Percentage of medium positive pixels covered by only one of the two tiles,
              with respect to actual tissue in the images. (Only if Flag == 1)
            - Low Positive Overlap: Percentage of low positive pixels covered by both tiles combined, with respect to
              actual tissue in the images. (Only if Flag == 1)
            - Low Positive Complement: Percentage of low positive pixels covered by only one of the two tiles, with
              respect to actual tissue in the images. (Only if Flag == 1)
            - Negative: Percentage of negative pixels not covered by either of the two tiles, with respect to actual
              tissue in the images. (Only if Flag == 1)
            - Tissue: Percentage of actual tissue across the tiles with respect to the total pixel count of the tile
              (1024 x 1024 = 1048576). (Only if Flag == 1)
            - Background / No Tissue: Percentage of the tile that is not covered by tissue across the tiles with
              respect to the total pixel count of the tile (1024 x 1024 = 1048576). (Only if Flag == 1)

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory. The
        coloring scheme is as follows:

            - Overlapping pixels (positive in both tiles):
                - Highly positive: Red ([255, 0, 0])
                - Medium positive: Red ([255, 0, 0])
                - Low positive: Red ([255, 0, 0])
            - Complementary pixels (positive in only one tile):
                - Tile1 highly positive: Green ([0, 255, 0])
                - Tile1 medium positive: Light Green ([153, 238, 153])
                - Tile1 low positive: Very Light Green ([204, 255, 204])
                - Tile2 highly positive: Blue ([0, 0, 204])
                - Tile2 medium positive: Light Blue ([102, 204, 255])
                - Tile2 low positive: Very Light Blue ([153, 204, 255])
            - Negative pixels (both tiles negative): White ([255, 255, 255])
            - Background pixels (no tissue): Gray ([192, 192, 192])
    """

    tile1 = iterable[0]
    tile2 = iterable[1]
    profile1, profile2 = iterable[2]
    dir = iterable[3]
    save_img = iterable[4]

    tilename = tile1["Tilename"]
    colocal_dict = {"Tilename": tilename}

    # Check if both Images contain tissue.If both dont: Flag = -1
    if tile1["Flag"] == 0 and tile2["Flag"] == 0:
        colocal_dict["Flag"] = -1
        return colocal_dict
    # If tile2 contains tissue and tile1 doesn't: Process tile2 only. If save_img is True, coloring for tile2 is blue.
    elif tile1["Flag"] == 0:
        colocal_dict["Flag"] = 1
        high_complement, med_complement, low_complement, negative, background = (
            _process_single_tile(
                tile2, dir, save_img, COLOR_TILE2, antigen_profile=profile2
            )
        )
        high_overlap, med_overlap, low_overlap = (0, 0, 0)
    # If tile1 contains tissue and tile2 doesn't: Process tile1 only. If save_img is True, coloring for tile1 is green.
    elif tile2["Flag"] == 0:
        colocal_dict["Flag"] = 1
        high_complement, med_complement, low_complement, negative, background = (
            _process_single_tile(
                tile1, dir, save_img, COLOR_TILE1, antigen_profile=profile1
            )
        )
        high_overlap, med_overlap, low_overlap = (0, 0, 0)
    # Check if tiles have the correct shape. If one doesn't: Flag = -2 TODO Padding
    elif tile1["Image Array"].shape != (1024, 1024) or tile2["Image Array"].shape != (
        1024,
        1024,
    ):
        colocal_dict["Flag"] = -2
        return colocal_dict
    # If both contain tissue: Process both tiles
    else:
        colocal_dict["Flag"] = 1
        (
            high_overlap,
            med_overlap,
            low_overlap,
            high_complement,
            med_complement,
            low_complement,
            negative,
            background,
        ) = _process_two_tiles(
            tile1,
            tile2,
            dir,
            save_img,
            [COLOR_TILE1, COLOR_TILE2],
            antigen_profiles=[profile1, profile2],
        )

    # Sum up the results
    coverage = (
        high_overlap
        + med_overlap
        + low_overlap
        + high_complement
        + med_complement
        + low_complement
    )
    total_overlap = high_overlap + med_overlap + low_overlap
    total_complement = high_complement + med_complement + low_complement
    tissue_count = coverage + negative

    # overlap, complement, negative with respect to tissue count. Tissue, background with respect to total pixel count.
    coverage = round((coverage / tissue_count) * 100, 4)
    total_overlap = round((total_overlap / tissue_count) * 100, 4)
    total_complement = round((total_complement / tissue_count) * 100, 4)
    high_overlap = round((high_overlap / tissue_count) * 100, 4)
    high_complement = round((high_complement / tissue_count) * 100, 4)
    med_overlap = round((med_overlap / tissue_count) * 100, 4)
    med_complement = round((med_complement / tissue_count) * 100, 4)
    low_overlap = round((low_overlap / tissue_count) * 100, 4)
    low_complement = round((low_complement / tissue_count) * 100, 4)
    negative = round((negative / tissue_count) * 100, 4)
    tissue_count = round((tissue_count / 1048576) * 100, 4)
    background = round((background / 1048576) * 100, 4)

    # set dictionary values
    colocal_dict["Total Coverage"] = coverage
    colocal_dict["Total Overlap"] = total_overlap
    colocal_dict["Total Complement"] = total_complement
    colocal_dict["High Positive Overlap"] = high_overlap
    colocal_dict["High Positive Complement"] = high_complement
    colocal_dict["Medium Positive Overlap"] = med_overlap
    colocal_dict["Medium Positive Complement"] = med_complement
    colocal_dict["Low Positive Overlap"] = low_overlap
    colocal_dict["Low Positive Complement"] = low_complement
    colocal_dict["Negative"] = negative
    colocal_dict["Tissue"] = tissue_count
    colocal_dict["Background / No Tissue"] = background

    return colocal_dict


def analyze_triplet_antigen_colocalization(iterable):
    """
    This function analyzes antigen colocalization in the same tile from 3 different WSIs and returns the results as
    dictionary.

    The function performs pixel-wise comparison across three images to calculate total coverage, overlapping coverage,
    and complementary coverage. The intensity levels are divided into five categories based on their intensity values.
    Intensity values are defined for each tile by the passed antigen profile which can be defined when initializing the
    SlideCollection. If not antigen profiles are defined, intensity values will fall back to default thresholds:
    high positive (0-60), medium positive (61-120), low positive (121-180), negative (181-234),
    and background (235-255). The results are saved in a dictionary and returned. Optionally, an image containing the
    colored results can be saved in the specified directory.

    Args:
        iterable (iterable): An iterable containing the following elements:
            'tile1', the first tile to be compared and analyzed for antigen coverage;
            'tile2', the second tile to be compared and analyzed for antigen coverage,
            'tile3', the third tileto be compared and analyzed for antigen coverage;
            Antigen-specific profiles: 'profile1', 'profile2', and 'profile3' containing antigen-specific thresholds
            for tile1, tile2, and tile 3respectively;
            'output path', the directory where the output image should be saved;
            'save_img', a flag indicating whether to save the output image.

    Returns:
        dict: A dictionary containing the results of the analysis:
            - Tilename: Name of the tile
            - Flag: Flag indicating if the tile was processed or not. (1: processed; 0: not processed due both tiles
              not being processable; -1: all tiles not containing tissue)
            - Total Coverage: Percentage of positive pixels (intensity values < 181), covered by all three tiles
              combined, with respect to actual tissue in the images. (Only if Flag == 1)
            - Total Overlap: Percentage of positive pixels that are positive in at least two or all three images
              combined, with respect to actual tissue in the images. (Only if Flag == 1)
            - Total Complement: Percentage of positive pixels that are only covered by one of the three images, with
              respect to actual tissue in the images. (Only if Flag == 1)
            - High Positive Overlap: Percentage of highly positive pixels covered by all three tiles combined, with
              respect to actual tissue in the images. (Only if Flag == 1)
            - High Positive Complement: Percentage of highly positive pixels covered by only one of the three tiles,
              with respect to actual tissue in the images. (Only if Flag == 1)
            - Medium Positive Overlap: Percentage of medium positive pixels covered by all three tiles combined, with
              respect to actual tissue in the images. (Only if Flag == 1)
            - Medium Positive Complement: Percentage of medium positive pixels covered by only one of the three tiles.
              (Only if Flag == 1)
            - Low Positive Overlap: Percentage of low positive pixels covered by all three tiles combined, with respect
              to actual tissue in the images. (Only if Flag == 1)
            - Low Positive Complement: Percentage of low positive pixels covered by only one of the three tiles, with
              respect to actual tissue in the images. (Only if Flag == 1)
            - Negative: Percentage of negative pixels not covered by any of the three tiles, with respect to actual
              tissue in the images.(Only if Flag == 1)
            - Tissue: Percentage of actual tissue accros the tiles with respect to total pixel count of the tile
              (1024 x 1024 = 1048576). (Only if Flag == 1)
            - Background / No Tissue: Percentage of the tile that is not covered by tissue across the tiles with
              respect to total pixel count of the tile (1024 x 1024 = 1048576). (Only if Flag == 1)

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory. The
        coloring scheme is as follows:

            - Overlapping pixels (positive in both tiles):
                - Highly positive: Red ([255, 0, 0])
                - Medium Positive: Red ([255, 0, 0])
                - Low positive: Red ([255, 0, 0])
            - Complementary pixels (positive in only one tile):
                - Tile1 highly positive: Green ([0, 255, 0])
                - Tile1 medium positive: Light Green ([153, 238, 153])
                - Tile1 low positive: Very Light Green ([204, 255, 204])
                - Tile2 highly positive: Blue ([0, 0, 204])
                - Tile2 medium positive: Light Blue ([102, 204, 255])
                - Tile2 low positive: Very Light Blue ([153, 204, 255])
                - Tile3 highly positive: Orange ([255, 165, 0])
                - Tile3 medium positive: Light Orange ([255, 200, 102])
                - Tile3 low positive: Very Light Orange ([255, 225, 153])
            - Negative pixels (both tiles negative): White ([255, 255, 255])
            - Background pixels (no tissue): Gray ([192, 192, 192])
    """

    tile1 = iterable[0]
    tile2 = iterable[1]
    tile3 = iterable[2]
    profile1, profile2, profile3 = iterable[3]
    dir = iterable[4]
    save_img = iterable[5]

    tilename = tile1["Tilename"]
    colocal_dict = {"Tilename": tilename}

    # Count the number of images with Flag=0
    flag_count = sum([tile1["Flag"] == 0, tile2["Flag"] == 0, tile3["Flag"] == 0])

    # Check if all Images contain tissue. If all dont: Flag = -1
    if flag_count == 3:
        colocal_dict["Flag"] = -1
        return colocal_dict
    # Check if two images have Flag=0 and process the remaining image
    elif flag_count == 2:
        colocal_dict["Flag"] = 1
        # If only tile1 processable, process tile1 with color green
        if tile1["Flag"] != 0:
            (
                high_complement,
                med_complement,
                low_complement,
                negative,
                background,
            ) = _process_single_tile(
                tile1, dir, save_img, COLOR_TILE1, antigen_profile=profile1
            )

        # If only tile2 processable, process tile2 with color blue
        elif tile2["Flag"] != 0:
            (
                high_complement,
                med_complement,
                low_complement,
                negative,
                background,
            ) = _process_single_tile(
                tile2, dir, save_img, COLOR_TILE2, antigen_profile=profile2
            )

        # If only tile3 processable, process tile3 with color orange
        elif tile3["Flag"] != 0:
            (
                high_complement,
                med_complement,
                low_complement,
                negative,
                background,
            ) = _process_single_tile(
                tile3, dir, save_img, COLOR_TILE3, antigen_profile=profile3
            )

        high_overlap, med_overlap, low_overlap = (0, 0, 0)

    # Check if only one tile has Flag=0 and process the remaining two tiles
    elif flag_count == 1:
        colocal_dict["Flag"] = 1
        if tile1["Flag"] == 0:
            (
                high_overlap,
                med_overlap,
                low_overlap,
                high_complement,
                med_complement,
                low_complement,
                negative,
                background,
            ) = _process_two_tiles(
                tile2,
                tile3,
                dir,
                save_img,
                [COLOR_TILE2, COLOR_TILE3],
                antigen_profiles=[profile2, profile3],
            )

        elif tile2["Flag"] == 0:
            (
                high_overlap,
                med_overlap,
                low_overlap,
                high_complement,
                med_complement,
                low_complement,
                negative,
                background,
            ) = _process_two_tiles(
                tile1,
                tile3,
                dir,
                save_img,
                [COLOR_TILE1, COLOR_TILE3],
                antigen_profiles=[profile1, profile3],
            )

        elif tile3["Flag"] == 0:
            (
                high_overlap,
                med_overlap,
                low_overlap,
                high_complement,
                med_complement,
                low_complement,
                negative,
                background,
            ) = _process_two_tiles(
                tile1,
                tile2,
                dir,
                save_img,
                [COLOR_TILE1, COLOR_TILE2],
                antigen_profiles=[profile1, profile2],
            )

    # Check for Error 2: Flag = -2: One of images doesn't have the correct shape
    # TODO Think about padding images to correct shape
    elif (
        tile1["Image Array"].shape != (1024, 1024)
        or tile2["Image Array"].shape != (1024, 1024)
        or tile3["Image Array"].shape != (1024, 1024)
    ):
        colocal_dict["Flag"] = -2
        return colocal_dict

    # Otherwise process all three images
    else:
        colocal_dict["Flag"] = 1
        (
            high_overlap,
            med_overlap,
            low_overlap,
            high_complement,
            med_complement,
            low_complement,
            negative,
            background,
        ) = _process_three_tiles(
            tile1,
            tile2,
            tile3,
            dir,
            save_img,
            antigen_profiles=[profile1, profile2, profile3],
        )

    coverage = (
        high_overlap
        + med_overlap
        + low_overlap
        + high_complement
        + med_complement
        + low_complement
    )
    total_overlap = high_overlap + med_overlap + low_overlap
    total_complement = high_complement + med_complement + low_complement
    tissue_count = coverage + negative

    # overlap, complement, negative with respect to tissue count. Tissue, background with respect to total pixel count.
    coverage = round((coverage / tissue_count) * 100, 4)
    total_overlap = round((total_overlap / tissue_count) * 100, 4)
    total_complement = round((total_complement / tissue_count) * 100, 4)
    high_overlap = round((high_overlap / tissue_count) * 100, 4)
    high_complement = round((high_complement / tissue_count) * 100, 4)
    med_overlap = round((med_overlap / tissue_count) * 100, 4)
    med_complement = round((med_complement / tissue_count) * 100, 4)
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
    colocal_dict["Medium Positive Overlap"] = med_overlap
    colocal_dict["Medium Positive Complement"] = med_complement
    colocal_dict["Low Positive Overlap"] = low_overlap
    colocal_dict["Low Positive Complement"] = low_complement
    colocal_dict["Negative"] = negative
    colocal_dict["Tissue"] = tissue_count
    colocal_dict["Background / No Tissue"] = background

    return colocal_dict


def _process_single_tile_old(tile, dir, save_img, color):
    """
    Processes a single image tile by iterating across the tile. It categorizes pixels into different complement levels,
    negative tissue, and background. Optionally saves the processed image to specified directory if save_img is True.

    Args:
        tile (dict): A dictionary containing the image under the key "Image Array".
        save_img (bool): A flag indicating whether to create a colored image and save it.
        dir (str): The directory where the image should be saved if save_img is True.
        color (list): A list of RGB color values to be used for different complement levels.

    Returns:
        tuple: A tuple containing counts of pixels in the following order:
            - high_complement (int): Count of pixels with values less than 61.
            - pos_complement (int): Count of pixels with values between 61-120.
            - low_complement (int): Count of pixels with values between 121-180.
            - negative (int): Count of pixels with values between 181-234.
            - background (int): Count of pixels with values 235 and above.

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory.
        The coloring scheme depends on the color parameter and is specified by the calling function. For more
        information, refer to the analyze_dual_antigen_colocalization or analyze_triplet_antigen_colocalization
        functions.
    """
    high_complement = 0
    med_complement = 0
    low_complement = 0
    negative = 0
    background = 0
    h, w = tile["Image Array"].shape

    if save_img:
        tilename = tile["Tilename"]
        img_out = np.full((h, w, 3), COLOR_BACKGROUND, dtype="uint8")
        for y in range(h):
            for x in range(w):
                pixel = tile["Image Array"][x, y]
                if pixel < 61:
                    high_complement += 1
                    img_out[x, y] = color[0]
                elif pixel < 121:
                    med_complement += 1
                    img_out[x, y] = color[1]
                elif pixel < 181:
                    low_complement += 1
                    img_out[x, y] = color[2]
                elif pixel < 235:
                    negative += 1
                    img_out[x, y] = COLOR_NEGATIVE
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
                    med_complement += 1
                elif pixel < 181:
                    low_complement += 1
                elif pixel < 235:
                    negative += 1
                else:
                    background += 1

    return high_complement, med_complement, low_complement, negative, background


def _process_single_tile(tile, dir, save_img, color, antigen_profile):
    """
    Processes antigen expression for a single tile.

    Processes a single image tile using vectorization and automatic backend detection for GPU acceleration. It
    categorizes pixels into different complement levels, negative tissue, and background based on the passed antigen
    profile. Optionally creates and saves a colored image of the analysis into the directory if 'save_img' is True.

    Args:
        tile (dict): A dictionary containing the image under the key "Image Array" and masks under the key "Masks".
        save_img (bool): A flag indicating whether to create a colored image and save it.
        dir (str): The directory where the image should be saved if save_img is True.
        color (list): A list of RGB color values to be used for different complement levels.
        antigen_profile (dict): Antigen profile for the tile, specifying the thresholds applied during processing.

    Returns:
        tuple: A tuple containing counts of pixels in the following order:
            - high_complement (int): Count of pixels with values below 'high_positive_threshold' of 'antigen profile'.
            - med_complement (int): Count of pixels with values between 'high_positive_threshold' and
              'medium_positive_threshold' of 'antigen_profile'.
            - low_complement (int): Count of pixels with values between 'medium_positive_threshold' and
              'low_positive_threshold' of 'antigen_profile'.
            - negative (int): Count of pixels with values between 'low_positive_threshold' and 234 of 'antigen_profile'.
            - background (int): Count of pixels 235 and above.

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory.
        The coloring scheme depends on the color parameter and is specified by the calling function.
    """
    # Convert image arrays to CuPy arrays
    img1 = xp.array(tile["Image Array"])

    # Get thresholds from antigen_profile
    high_thresh = antigen_profile["high_positive_threshold"]
    medium_thresh = antigen_profile["medium_positive_threshold"]
    low_thresh = antigen_profile["low_positive_threshold"]

    # Create masks for different intensity zones for tile1
    high_positive_mask = img1 < high_thresh
    medium_positive_mask = (img1 >= high_thresh) & (img1 < medium_thresh)
    low_positive_mask = (img1 >= medium_thresh) & (img1 < low_thresh)
    negative_mask = (img1 >= low_thresh) & (img1 < 235)
    background_mask = img1 >= 235

    # Calculate counts for each zone
    high_complement = xp.sum(high_positive_mask)
    med_complement = xp.sum(medium_positive_mask)
    low_complement = xp.sum(low_positive_mask)
    negative = xp.sum(negative_mask)
    background = xp.sum(background_mask)

    # Save image if required
    if save_img:
        colored_img = xp.full(
            (high_positive_mask.shape[0], high_positive_mask.shape[1], 3),
            xp.array(COLOR_BACKGROUND, dtype=xp.uint8),
            dtype=xp.uint8,
        )
        colored_img[high_positive_mask] = color[0]  # High complement color
        colored_img[medium_positive_mask] = color[1]  # Positive complement color
        colored_img[low_positive_mask] = color[2]  # Low complement color
        colored_img[negative_mask] = [255, 255, 255]  # White for negative
        colored_img[background_mask] = [192, 192, 192]  # Gray for background

        colored_img = to_numpy(colored_img)
        img = Image.fromarray(colored_img)
        img.save(f"{dir}/{tile['Tilename']}.tif")
    total_pixel_count = img1.size
    total_count = (
        high_complement + med_complement + low_complement + negative + background
    )
    assert (
        total_count == total_pixel_count
    ), f"1: Total count {total_count} does not match total pixel count {total_pixel_count}"
    return (
        int(high_complement),
        int(med_complement),
        int(low_complement),
        int(negative),
        int(background),
    )


def _process_two_tiles_old(tile1, tile2, dir, save_img, colors):
    """
    Processes two image tiles by iterating across the tile. It categorizes pixels into overlap, different complement
    levels, negative tissue, and background. Optionally saves the processed image to the specified directory if
    save_img is True. This function is typically called by analyze_dual_antigen_colocalization or
    analyze_triplet_antigen_colocalization when two of the three tiles contain tissue.

    Args:
        tile1 (dict): A dictionary containing tile1's data. The image can be accessed using the key "Image Array".
        tile2 (dict): A dictionary containing tile2's data. The image can be accessed using the key "Image Array".
        dir (str): The directory where the output image should be saved if save_img is True.
        save_img (bool): A flag indicating whether to save the output image.
        colors (list): A list of RGB color values to be used for different complement levels.

    Returns:
        tuple: A tuple containing the counts of:
            - high_overlap (int): Number of pixels where both tiles are highly positive.
            - med_overlap (int): Number of pixels where both tiles are medium positive.
            - low_overlap (int): Number of pixels where both tiles are low positive.
            - high_complement (int): Number of pixels where one tile is highly positive and the other is not.
            - med_complement (int): Number of pixels where one tile is medium positive and the other is not.
            - low_complement (int): Number of pixels where one tile is low positive and the other is not.
            - negative (int): Number of pixels where either tile is negative.
            - background (int): Number of background pixels.

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory.
        The coloring scheme depends on the color parameter and is specified by the calling function. For more
        information, refer to the analyze_dual_antigen_colocalization or analyze_triplet_antigen_colocalization
        functions.
    """

    high_overlap = 0
    med_overlap = 0
    low_overlap = 0
    high_complement = 0
    med_complement = 0
    low_complement = 0
    negative = 0
    background = 0
    h, w = tile1["Image Array"].shape

    if save_img:
        tilename = tile1["Tilename"]
        img_out = np.full((h, w, 3), COLOR_BACKGROUND, dtype="uint8")
        for y in range(h):
            for x in range(w):
                pixel1 = tile1["Image Array"][x, y]
                pixel2 = tile2["Image Array"][x, y]

                # Check if both pixels are positive
                if pixel1 < 181 and pixel2 < 181:
                    if pixel1 < 61 and pixel2 < 61:
                        high_overlap += 1
                        # Color strong red where both are highly positive
                        img_out[x, y] = COLOR_OVERLAP
                    elif pixel1 < 121 and pixel2 < 121:
                        med_overlap += 1
                        # Color strong red where both are positive
                        img_out[x, y] = COLOR_OVERLAP
                    else:
                        low_overlap += 1
                        # Color light red where both are low positive
                        img_out[x, y] = COLOR_OVERLAP
                # Check if pixel1 is positive and pixel2 is not
                elif pixel1 < 181 and pixel2 > 180:
                    if pixel1 < 61:
                        high_complement += 1
                        # Color scheme for tile1 (strong)
                        img_out[x, y] = colors[0][0]
                    elif pixel1 < 121:
                        med_complement += 1
                        # Color scheme for tile1 (bright)
                        img_out[x, y] = colors[0][1]
                    else:
                        low_complement += 1
                        # Color scheme for tile1 (light)
                        img_out[x, y] = colors[0][2]
                elif pixel1 > 180 and pixel2 < 181:
                    if pixel2 < 61:
                        high_complement += 1
                        # Color scheme for tile2 (strong)
                        img_out[x, y] = colors[1][0]
                    elif pixel2 < 121:
                        med_complement += 1
                        # Color scheme for tile2 (bright)
                        img_out[x, y] = colors[1][1]
                    else:
                        low_complement += 1
                        # Color scheme for tile2 (light)
                        img_out[x, y] = colors[1][2]
                elif pixel1 < 235 or pixel2 < 235:
                    negative += 1
                    # Color white where either are negative
                    img_out[x, y] = COLOR_NEGATIVE
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
                        med_overlap += 1
                    else:
                        low_overlap += 1
                # Check if one pixel is positive and the other is not
                elif (pixel1 < 181 and pixel2 > 180) or (pixel1 > 180 and pixel2 < 181):
                    if (pixel1 < 61 and pixel2 > 180) or (pixel2 < 61 and pixel1 > 180):
                        high_complement += 1
                    elif (pixel1 < 121 and pixel2 > 180) or (
                        pixel2 < 121 and pixel1 > 180
                    ):
                        med_complement += 1
                    else:
                        low_complement += 1
                elif pixel1 < 235 or pixel2 < 235:
                    negative += 1
                else:
                    background += 1

    return (
        high_overlap,
        med_overlap,
        low_overlap,
        high_complement,
        med_complement,
        low_complement,
        negative,
        background,
    )


def _process_two_tiles(tile1, tile2, dir, save_img, colors, antigen_profiles):
    """
    Processes antigen expression for two single tiles.

    Processes two image tiles using vectorization and automatic backend detection for GPU acceleration. It categorizes
    pixels into overlap, different complement levels, negative tissue, and background based on the passed antigen
    profiles. Optionally creates and saves a colored image of the analysis into the directory if 'save_img' is True.
    This function is typically called by analyze_dual_antigen_colocalization or analyze_triplet_antigen_colocalization
    when two of the three tiles contain tissue.

    Args:
        tile1 (dict): A dictionary containing tile1's data. The image can be accessed using the key "Image Array".
        tile2 (dict): A dictionary containing tile2's data. The image can be accessed using the key "Image Array".
        dir (str): The directory where the output image should be saved if save_img is True.
        save_img (bool): A flag indicating whether to save the output image.
        colors (list): A list of RGB color values to be used for different complement levels.
        antigen_profiles (list): List of two dicts specifying antigen thresholds for each tile respectively.

    Returns:
        tuple: A tuple containing the counts of:
            - high_overlap (int): Count of pixels where both tiles are highly positive, according to 'antigen_profiles'.
            - med_overlap (int): Count of pixels where both tiles are medium positive, according to 'antigen_profiles'.
            - low_overlap (int): Count of pixels where both tiles are low positive, according to 'antigen_profiles'.
            - high_complement (int): Count of pixels with values below 'high_positive_threshold' of 'antigen profiles'.
            - med_complement (int): Count of pixels with values between 'high_positive_threshold' and
              'medium_positive_threshold' of 'antigen_profiles'
            - low_complement (int): Count of pixels with values between 'medium_positive_threshold' and
              'low_positive_threshold' of 'antigen_profiles'.
            - negative (int): Count of pixels with values between 'low_positive_threshold' and 234 of 'antigen_profile'.
            - background (int): Count of pixels 235 and above.

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory.
        The coloring scheme depends on the color parameter and is specified by the calling function. For more
        information, refer to the analyze_dual_antigen_colocalization or analyze_triplet_antigen_colocalization
        functions.
    """
    # Convert image arrays to CuPy arrays
    img1 = xp.array(tile1["Image Array"])
    img2 = xp.array(tile2["Image Array"])

    # Get thresholds from antigen_profile
    high_thresh1 = antigen_profiles[0]["high_positive_threshold"]
    medium_thresh1 = antigen_profiles[0]["medium_positive_threshold"]
    low_thresh1 = antigen_profiles[0]["low_positive_threshold"]

    high_thresh2 = antigen_profiles[1]["high_positive_threshold"]
    medium_thresh2 = antigen_profiles[1]["medium_positive_threshold"]
    low_thresh2 = antigen_profiles[1]["low_positive_threshold"]

    # Create masks for different intensity zones for tile1
    high_positive_mask1 = img1 < high_thresh1
    medium_positive_mask1 = (img1 >= high_thresh1) & (img1 < medium_thresh1)
    low_positive_mask1 = (img1 >= medium_thresh1) & (img1 < low_thresh1)
    negative_mask1 = (img1 >= low_thresh1) & (img1 < 235)
    background_mask1 = img1 >= 235

    # Create masks for different intensity zones for tile2
    high_positive_mask2 = img2 < high_thresh2
    medium_positive_mask2 = (img2 >= high_thresh2) & (img2 < medium_thresh2)
    low_positive_mask2 = (img2 >= medium_thresh2) & (img2 < low_thresh2)
    negative_mask2 = (img2 >= low_thresh2) & (img2 < 235)
    background_mask2 = img2 >= 235

    # Calculate overlap and complement counts
    high_overlap = xp.sum(high_positive_mask1 & high_positive_mask2)
    med_overlap = xp.sum(
        (high_positive_mask1 & medium_positive_mask2)
        | (medium_positive_mask1 & high_positive_mask2)
        | (medium_positive_mask1 & medium_positive_mask2)
    )
    low_overlap = xp.sum(
        (high_positive_mask1 & low_positive_mask2)
        | (medium_positive_mask1 & low_positive_mask2)
        | (low_positive_mask1 & high_positive_mask2)
        | (low_positive_mask1 & medium_positive_mask2)
        | (low_positive_mask1 & low_positive_mask2)
    )
    high_complement = xp.sum(
        high_positive_mask1
        & ~(high_positive_mask2 | medium_positive_mask2 | low_positive_mask2)
    ) + xp.sum(
        ~(high_positive_mask1 | medium_positive_mask1 | low_positive_mask1)
        & high_positive_mask2
    )
    med_complement = xp.sum(
        medium_positive_mask1
        & ~(high_positive_mask2 | medium_positive_mask2 | low_positive_mask2)
    ) + xp.sum(
        ~(high_positive_mask1 | medium_positive_mask1 | low_positive_mask1)
        & medium_positive_mask2
    )
    low_complement = xp.sum(
        low_positive_mask1
        & ~(high_positive_mask2 | medium_positive_mask2 | low_positive_mask2)
    ) + xp.sum(
        ~(high_positive_mask1 | medium_positive_mask1 | low_positive_mask1)
        & low_positive_mask2
    )
    negative = xp.sum(
        (negative_mask1 & negative_mask2)
        | (negative_mask1 & background_mask2)
        | (background_mask1 & negative_mask2)
    )
    background = xp.sum(background_mask1 & background_mask2)

    # Save image if required
    if save_img:
        colored_img = xp.full(
            (high_positive_mask1.shape[0], high_positive_mask1.shape[1], 3),
            xp.array(COLOR_BACKGROUND, dtype=xp.uint8),
            dtype=xp.uint8,
        )
        colored_img[high_positive_mask1 & high_positive_mask2] = (
            COLOR_OVERLAP  # High overlap color
        )
        colored_img[
            (high_positive_mask1 & medium_positive_mask2)
            | (medium_positive_mask1 & high_positive_mask2)
            | (medium_positive_mask1 & medium_positive_mask2)
        ] = COLOR_OVERLAP  # Positive overlap color
        colored_img[
            (high_positive_mask1 & low_positive_mask2)
            | (medium_positive_mask1 & low_positive_mask2)
            | (low_positive_mask1 & high_positive_mask2)
            | (low_positive_mask1 & medium_positive_mask2)
            | (low_positive_mask1 & low_positive_mask2)
        ] = COLOR_OVERLAP  # Low overlap color
        colored_img[
            high_positive_mask1
            & ~(high_positive_mask2 | medium_positive_mask2 | low_positive_mask2)
        ] = colors[0][
            0
        ]  # High complement color for tile1
        colored_img[
            ~(high_positive_mask1 | medium_positive_mask1 | low_positive_mask1)
            & high_positive_mask2
        ] = colors[1][
            0
        ]  # High complement color for tile2
        colored_img[
            medium_positive_mask1
            & ~(high_positive_mask2 | medium_positive_mask2 | low_positive_mask2)
        ] = colors[0][
            1
        ]  # Positive complement color for tile1
        colored_img[
            ~(high_positive_mask1 | medium_positive_mask1 | low_positive_mask1)
            & medium_positive_mask2
        ] = colors[1][
            1
        ]  # Positive complement color for tile2
        colored_img[
            low_positive_mask1
            & ~(high_positive_mask2 | medium_positive_mask2 | low_positive_mask2)
        ] = colors[0][
            2
        ]  # Low complement color for tile1
        colored_img[
            ~(high_positive_mask1 | medium_positive_mask1 | low_positive_mask1)
            & low_positive_mask2
        ] = colors[1][
            2
        ]  # Low complement color for tile2
        colored_img[negative_mask1 & negative_mask2] = [
            255,
            255,
            255,
        ]  # White for negative
        colored_img[background_mask1 & background_mask2] = [
            192,
            192,
            192,
        ]  # Gray for background

        colored_img = to_numpy(colored_img)
        img = Image.fromarray(colored_img)
        img.save(f"{dir}/{tile1['Tilename']}.tif")
    # Check if the sum of all counts equals the total pixel count
    total_pixel_count = img1.size
    total_count = (
        high_overlap
        + med_overlap
        + low_overlap
        + high_complement
        + med_complement
        + low_complement
        + negative
        + background
    )
    assert (
        total_count == total_pixel_count
    ), f"Pair: Total count {total_count} does not match total pixel count {total_pixel_count}"
    return (
        int(high_overlap),
        int(med_overlap),
        int(low_overlap),
        int(high_complement),
        int(med_complement),
        int(low_complement),
        int(negative),
        int(background),
    )


def _process_three_tiles_old(tile1, tile2, tile3, dir, save_img):
    """
    Processes three tiles by iterating across the tiles. It categorizes pixels into overlap, different complement
    levels, negative tissue and background. Optionally saves the processed image into dir if save_img is True. This
    function is only called by analyze_triplet_antigen_colocalization.

    Args:
        tile1 (dict): A dictionary containing tile1's data. The image can be accessed using the key "Image Array"
        tile2 (dict): A dictionary containing tile2's data. The image can be accessed using the key "Image Array"
        tile3 (dict): A dictionary containing tile3's data. The image can be accessed using the key "Image Array".
        dir (str): The directory where the output image should be saved if save_img is True.
        save_img (bool): A flag indicating whether to save the output image.

    Returns:
        tuple: A tuple containing the counts of:
            - high_overlap (int): Number of pixels where all three tiles are highly positive.
            - med_overlap (int): Number of pixels where all three tiles are medium positive.
            - low_overlap (int): Number of pixels where all three tiles are low positive.
            - high_complement (int): Number of pixels where one tile is highly positive and the others are not.
            - med_complement (int): Number of pixels where one tile is medium positive and the others are not.
            - low_complement (int): Number of pixels where one tile is low positive and the others are not.
            - negative (int): Number of pixels where any tile is negative.
            - background (int): Number of background pixels.

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory.
        For the coloring scheme, see the description of analyze_triplet_antigen_combinations().

    """
    high_overlap = 0
    med_overlap = 0
    low_overlap = 0
    high_complement = 0
    med_complement = 0
    low_complement = 0
    negative = 0
    background = 0
    h, w = tile1["Image Array"].shape

    if save_img:
        # If save_img is True, in addition to the numerical analysis an image containing the results of the analysis
        # will be created and saved in passed directory. Else only numerical analysis will be performed.
        tilename = tile1["Tilename"]
        img_out = np.full((h, w, 3), COLOR_BACKGROUND, dtype="uint8")
        for y in range(h):
            for x in range(w):
                pixel_values = [
                    tile["Image Array"][x, y] for tile in [tile1, tile2, tile3]
                ]
                sum_high = sum(1 for val in pixel_values if val < 61)
                sum_med = sum(1 for val in pixel_values if val < 121)
                sum_low = sum(1 for val in pixel_values if val < 181)
                sum_neg = sum(1 for val in pixel_values if val < 235)

                # Check if 2 or all 3 pixels are positive
                if sum_low >= 2:
                    # Check if 2 or all 3 pixels are highly positive
                    if sum_high >= 2:
                        high_overlap += 1
                        # Color strong red where all are highly positive
                        img_out[x, y] = COLOR_OVERLAP
                    # Check if 2 or all 3 pixels are positive
                    elif sum_med >= 2:
                        med_overlap += 1
                        # Color strong red where all are positive
                        img_out[x, y] = COLOR_OVERLAP
                    # Else 2 or all 3 pixels are low positive
                    else:
                        low_overlap += 1
                        # Color strong red where all are low positive
                        img_out[x, y] = COLOR_OVERLAP
                # Check if 1 pixel is positive while the other 2 are not
                elif sum_low == 1:
                    # Check if the pixel is highly positive
                    if sum_high == 1:
                        idx = pixel_values.index(min(pixel_values))
                        high_complement += 1
                        # Colors: tile1: Strong Green, tile2: Strong Blue, tile3: Strong Orange
                        img_out[x, y] = [
                            COLOR_TILE1[0],
                            COLOR_TILE2[0],
                            COLOR_TILE3[0],
                        ][idx]
                    # Check if the pixel is positive
                    elif sum_med == 1:
                        idx = pixel_values.index(min(pixel_values))
                        med_complement += 1
                        # Colors: tile1: Bright Green, tile2: Bright Blue, tile3: Bright Orange
                        img_out[x, y] = [
                            COLOR_TILE1[1],
                            COLOR_TILE2[1],
                            COLOR_TILE3[1],
                        ][idx]
                    # Else the pixel is low positive
                    else:
                        idx = pixel_values.index(min(pixel_values))
                        low_complement += 1
                        # Colors: tile1: Light Green, tile2: Light Blue, tile3: Light Orange
                        img_out[x, y] = [
                            COLOR_TILE1[2],
                            COLOR_TILE2[2],
                            COLOR_TILE3[2],
                        ][idx]
                # Check if any of the pixels are negative
                elif sum_neg > 0:
                    negative += 1
                    # Color white where either are negative
                    img_out[x, y] = COLOR_NEGATIVE
                else:
                    # Else pixel is background: stay gray
                    background += 1

        img_out = Image.fromarray(img_out.astype("uint8"))
        out = f"{dir}/{tilename}.tif"
        img_out.save(out)
    else:
        for y in range(h):
            for x in range(w):
                pixel_values = [
                    img["Image Array"][x, y] for img in [tile1, tile2, tile3]
                ]
                sum_high = sum(1 for val in pixel_values if val < 61)
                sum_med = sum(1 for val in pixel_values if val < 121)
                sum_low = sum(1 for val in pixel_values if val < 181)
                sum_neg = sum(1 for val in pixel_values if val < 235)

                # Check if 2 or all 3 pixels are positive
                if sum_low >= 2:
                    # Check if 2 or all 3 pixels are highly positive
                    if sum_high >= 2:
                        high_overlap += 1
                    # Check if 2 or all 3 pixels are positive
                    elif sum_med >= 2:
                        med_overlap += 1
                    # Else 2 or all 3 pixels are low positive
                    else:
                        low_overlap += 1
                # Check if 1 pixel is positive while the other 2 are not
                elif sum_low == 1:
                    # Check if the pixel is highly positive
                    if sum_high == 1:
                        high_complement += 1
                    # Check if the pixel is positive
                    elif sum_med == 1:
                        med_complement += 1
                    # Else the pixel is low positive
                    else:
                        low_complement += 1
                # Check if any of the pixels are negative
                elif sum_neg > 0:
                    negative += 1
                else:
                    # Else pixel is background
                    background += 1

    total_pixel_count = tile1["Image Array"].size
    total_count = (
        high_overlap
        + med_overlap
        + low_overlap
        + high_complement
        + med_complement
        + low_complement
        + negative
        + background
    )
    assert (
        total_count == total_pixel_count
    ), f"Total count {total_count} does not match total pixel count {total_pixel_count}"

    return (
        high_overlap,
        med_overlap,
        low_overlap,
        high_complement,
        med_complement,
        low_complement,
        negative,
        background,
    )


def _process_three_tiles(tile1, tile2, tile3, dir, save_img, antigen_profiles):
    """
    Processes antigen expression for three tiles.

    Processes three image tiles using vectorization and automatic backend detection for GPU acceleration. It categorizes
    pixels into overlap, different complement levels, negative tissue, and background based on the passed antigen
    profiles. Optionally creates and saves a colored image of the analysis into the directory if 'save_img' is True.
    This function is only called by analyze_triplet_antigen_colocalization.

    Args:
        tile1 (dict): A dictionary containing tile1's data. The image can be accessed using the key "Image Array".
        tile2 (dict): A dictionary containing tile2's data. The image can be accessed using the key "Image Array".
        tile3 (dict): A dictionary containing tile3's data. The image can be accessed using the key "Image Array".
        dir (str): The directory where the output image should be saved if save_img is True.
        save_img (bool): A flag indicating whether to save the output image.
        antigen_profiles (list): List of three dicts specifying antigen thresholds for each tile respectively.


    Returns:
        tuple: A tuple containing the counts of:
            - high_overlap (int): Count of pixels where both tiles are highly positive, according to 'antigen_profiles'.
            - med_overlap (int): Count of pixels where both tiles are medium positive, according to 'antigen_profiles'.
            - low_overlap (int): Count of pixels where both tiles are low positive, according to 'antigen_profiles'.
            - high_complement (int): Count of pixels with values below 'high_positive_threshold' of 'antigen profiles'.
            - med_complement (int): Count of pixels with values between 'high_positive_threshold' and
              'medium_positive_threshold' of 'antigen_profiles'
            - low_complement (int): Count of pixels with values between 'medium_positive_threshold' and
              'low_positive_threshold' of 'antigen_profiles'.
            - negative (int): Count of pixels with values between 'low_positive_threshold' and 234 of 'antigen_profile'.
            - background (int): Count of pixels 235 and above.

    Optional Image Saving:
        If save_img is True, an image containing the colored results will be saved in the specified directory.
        For the coloring scheme, see the description of analyze_triplet_antigen_combinations().
    """
    # Convert image arrays to CuPy arrays
    img1 = xp.array(tile1["Image Array"])
    img2 = xp.array(tile2["Image Array"])
    img3 = xp.array(tile3["Image Array"])

    # Get thresholds from antigen_profile
    high_thresh1 = antigen_profiles[0]["high_positive_threshold"]
    medium_thresh1 = antigen_profiles[0]["medium_positive_threshold"]
    low_thresh1 = antigen_profiles[0]["low_positive_threshold"]

    high_thresh2 = antigen_profiles[1]["high_positive_threshold"]
    medium_thresh2 = antigen_profiles[1]["medium_positive_threshold"]
    low_thresh2 = antigen_profiles[1]["low_positive_threshold"]

    high_thresh3 = antigen_profiles[2]["high_positive_threshold"]
    medium_thresh3 = antigen_profiles[2]["medium_positive_threshold"]
    low_thresh3 = antigen_profiles[2]["low_positive_threshold"]
    # Create masks for different intensity zones for tile1
    high_positive_mask1 = img1 < high_thresh1
    medium_positive_mask1 = (img1 >= high_thresh1) & (img1 < medium_thresh1)
    low_positive_mask1 = (img1 >= medium_thresh1) & (img1 < low_thresh1)
    negative_mask1 = (img1 >= low_thresh1) & (img1 < 235)
    background_mask1 = img1 >= 235

    # Create masks for different intensity zones for tile2
    high_positive_mask2 = img2 < high_thresh2
    medium_positive_mask2 = (img2 >= high_thresh2) & (img2 < medium_thresh2)
    low_positive_mask2 = (img2 >= medium_thresh2) & (img2 < low_thresh2)
    negative_mask2 = (img2 >= low_thresh2) & (img2 < 235)
    background_mask2 = img2 >= 235

    # Create masks for different intensity zones for tile3
    high_positive_mask3 = img3 < high_thresh3
    medium_positive_mask3 = (img3 >= high_thresh3) & (img3 < medium_thresh3)
    low_positive_mask3 = (img3 >= medium_thresh3) & (img3 < low_thresh3)
    negative_mask3 = (img3 >= low_thresh3) & (img3 < 235)
    background_mask3 = img3 >= 235

    # Calculate overlap and complement counts
    # High overlap if all 3 or 2/3 pixels are high positive, regardless of the 3rd
    high_overlap = xp.sum(
        (high_positive_mask1 & high_positive_mask2 & high_positive_mask3)
        | (high_positive_mask1 & high_positive_mask2 & ~high_positive_mask3)
        | (high_positive_mask1 & high_positive_mask3 & ~high_positive_mask2)
        | (high_positive_mask2 & high_positive_mask3 & ~high_positive_mask1)
    )
    # Positive overlap if at least 2/3 pixels are positive or if 2 below 121 and not high overlap
    med_overlap = xp.sum(
        (medium_positive_mask1 & medium_positive_mask2 & medium_positive_mask3)
        | (medium_positive_mask1 & medium_positive_mask2 & ~medium_positive_mask3)
        | (medium_positive_mask1 & medium_positive_mask3 & ~medium_positive_mask2)
        | (medium_positive_mask2 & medium_positive_mask3 & ~medium_positive_mask1)
        | (
            medium_positive_mask1
            & high_positive_mask2
            & ~high_positive_mask3
            & ~medium_positive_mask3
        )
        | (
            medium_positive_mask1
            & high_positive_mask3
            & ~high_positive_mask2
            & ~medium_positive_mask2
        )
        | (
            high_positive_mask1
            & medium_positive_mask2
            & ~high_positive_mask3
            & ~medium_positive_mask3
        )
        | (
            high_positive_mask1
            & medium_positive_mask3
            & ~high_positive_mask2
            & ~medium_positive_mask2
        )
        | (
            medium_positive_mask2
            & high_positive_mask3
            & ~high_positive_mask1
            & ~medium_positive_mask1
        )
        | (
            medium_positive_mask3
            & high_positive_mask2
            & ~high_positive_mask1
            & ~medium_positive_mask1
        )
    )
    # Low positive overlap if 2/3 pixels are low positive or 2 pixel <181 if not (high-) positive overlap
    low_overlap = xp.sum(
        (low_positive_mask1 & low_positive_mask2 & low_positive_mask3)
        | (low_positive_mask1 & low_positive_mask2 & ~low_positive_mask3)
        | (low_positive_mask1 & low_positive_mask3 & ~low_positive_mask2)
        | (low_positive_mask2 & low_positive_mask3 & ~low_positive_mask1)
        | (
            low_positive_mask1
            & medium_positive_mask2
            & (negative_mask3 | background_mask3)
        )
        | (
            low_positive_mask1
            & high_positive_mask2
            & (negative_mask3 | background_mask3)
        )
        | (
            low_positive_mask1
            & medium_positive_mask3
            & (negative_mask2 | background_mask2)
        )
        | (
            low_positive_mask1
            & high_positive_mask3
            & (negative_mask2 | background_mask2)
        )
        | (
            low_positive_mask2
            & medium_positive_mask3
            & (negative_mask1 | background_mask1)
        )
        | (
            low_positive_mask2
            & high_positive_mask3
            & (negative_mask1 | background_mask1)
        )
        | (
            low_positive_mask2
            & medium_positive_mask1
            & (negative_mask3 | background_mask3)
        )
        | (
            low_positive_mask2
            & high_positive_mask1
            & (negative_mask3 | background_mask3)
        )
        | (
            low_positive_mask3
            & medium_positive_mask1
            & (negative_mask2 | background_mask2)
        )
        | (
            low_positive_mask3
            & high_positive_mask1
            & (negative_mask2 | background_mask2)
        )
        | (
            low_positive_mask3
            & medium_positive_mask2
            & (negative_mask1 | background_mask1)
        )
        | (
            low_positive_mask3
            & high_positive_mask2
            & (negative_mask1 | background_mask1)
        )
    )
    # High complement if only 1 pixel <61 while the other 2 >= 181
    high_complement = xp.sum(
        (
            high_positive_mask1
            & (negative_mask2 | background_mask2)
            & (negative_mask3 | background_mask3)
        )
        | (
            high_positive_mask2
            & (negative_mask1 | background_mask1)
            & (negative_mask3 | background_mask3)
        )
        | (
            high_positive_mask3
            & (negative_mask1 | background_mask1)
            & (negative_mask2 | background_mask2)
        )
    )
    # Positive complement if only 1 pixel <121 while the other 2 >= 181
    med_complement = xp.sum(
        (
            medium_positive_mask1
            & (negative_mask2 | background_mask2)
            & (negative_mask3 | background_mask3)
        )
        | (
            medium_positive_mask2
            & (negative_mask1 | background_mask1)
            & (negative_mask3 | background_mask3)
        )
        | (
            medium_positive_mask3
            & (negative_mask1 | background_mask1)
            & (negative_mask2 | background_mask2)
        )
    )
    # Positive complement if only 1 pixel <181 while the other 2 >= 181
    low_complement = xp.sum(
        (
            low_positive_mask1
            & (negative_mask2 | background_mask2)
            & (negative_mask3 | background_mask3)
        )
        | (
            low_positive_mask2
            & (negative_mask1 | background_mask1)
            & (negative_mask3 | background_mask3)
        )
        | (
            low_positive_mask3
            & (negative_mask1 | background_mask1)
            & (negative_mask2 | background_mask2)
        )
    )
    # Negative if at least 1/3 pixels >180 and <235 while no other pixel <181
    negative = xp.sum(
        negative_mask1 & negative_mask2 & negative_mask3
        | negative_mask1 & negative_mask2 & background_mask3
        | negative_mask1 & background_mask2 & negative_mask3
        | background_mask1 & negative_mask2 & negative_mask3
        | negative_mask1 & background_mask2 & background_mask3
        | background_mask1 & negative_mask2 & background_mask3
        | background_mask1 & background_mask2 & negative_mask3
    )
    # Background only if all 3 pixels are background
    background = xp.sum(background_mask1 & background_mask2 & background_mask3)

    # Check if the sum of all counts equals the total pixel count
    total_pixel_count = img1.size
    total_count = (
        high_overlap
        + med_overlap
        + low_overlap
        + high_complement
        + med_complement
        + low_complement
        + negative
        + background
    )

    assert (
        total_count == total_pixel_count
    ), f"Triple: Total count {total_count} does not match total pixel count {total_pixel_count}"

    # Save image if required
    if save_img:
        colored_img = xp.full(
            (high_positive_mask1.shape[0], high_positive_mask1.shape[1], 3),
            xp.array(COLOR_BACKGROUND, dtype=xp.uint8),
            dtype=xp.uint8,
        )
        # High overlap color
        colored_img[
            (high_positive_mask1 & high_positive_mask2 & high_positive_mask3)
            | (high_positive_mask1 & high_positive_mask2 & ~high_positive_mask3)
            | (high_positive_mask1 & high_positive_mask3 & ~high_positive_mask2)
            | (high_positive_mask2 & high_positive_mask3 & ~high_positive_mask1)
        ] = COLOR_OVERLAP

        # Positive overlap color
        colored_img[
            (medium_positive_mask1 & medium_positive_mask2 & medium_positive_mask3)
            | (medium_positive_mask1 & medium_positive_mask2 & ~medium_positive_mask3)
            | (medium_positive_mask1 & medium_positive_mask3 & ~medium_positive_mask2)
            | (medium_positive_mask2 & medium_positive_mask3 & ~medium_positive_mask1)
            | (
                medium_positive_mask1
                & high_positive_mask2
                & ~high_positive_mask3
                & ~medium_positive_mask3
            )
            | (
                medium_positive_mask1
                & high_positive_mask3
                & ~high_positive_mask2
                & ~medium_positive_mask2
            )
            | (
                high_positive_mask1
                & medium_positive_mask2
                & ~high_positive_mask3
                & ~medium_positive_mask3
            )
            | (
                high_positive_mask1
                & medium_positive_mask3
                & ~high_positive_mask2
                & ~medium_positive_mask2
            )
            | (
                medium_positive_mask2
                & high_positive_mask3
                & ~high_positive_mask1
                & ~medium_positive_mask1
            )
            | (
                medium_positive_mask3
                & high_positive_mask2
                & ~high_positive_mask1
                & ~medium_positive_mask1
            )
        ] = COLOR_OVERLAP

        # Low overlap color
        colored_img[
            (low_positive_mask1 & low_positive_mask2 & low_positive_mask3)
            | (low_positive_mask1 & low_positive_mask2 & ~low_positive_mask3)
            | (low_positive_mask1 & low_positive_mask3 & ~low_positive_mask2)
            | (low_positive_mask2 & low_positive_mask3 & ~low_positive_mask1)
            | (
                low_positive_mask1
                & medium_positive_mask2
                & (negative_mask3 | background_mask3)
            )
            | (
                low_positive_mask1
                & high_positive_mask2
                & (negative_mask3 | background_mask3)
            )
            | (
                low_positive_mask1
                & medium_positive_mask3
                & (negative_mask2 | background_mask2)
            )
            | (
                low_positive_mask1
                & high_positive_mask3
                & (negative_mask2 | background_mask2)
            )
            | (
                low_positive_mask2
                & medium_positive_mask3
                & (negative_mask1 | background_mask1)
            )
            | (
                low_positive_mask2
                & high_positive_mask3
                & (negative_mask1 | background_mask1)
            )
            | (
                low_positive_mask2
                & medium_positive_mask1
                & (negative_mask3 | background_mask3)
            )
            | (
                low_positive_mask2
                & high_positive_mask1
                & (negative_mask3 | background_mask3)
            )
            | (
                low_positive_mask3
                & medium_positive_mask1
                & (negative_mask2 | background_mask2)
            )
            | (
                low_positive_mask3
                & high_positive_mask1
                & (negative_mask2 | background_mask2)
            )
            | (
                low_positive_mask3
                & medium_positive_mask2
                & (negative_mask1 | background_mask1)
            )
            | (
                low_positive_mask3
                & high_positive_mask2
                & (negative_mask1 | background_mask1)
            )
        ] = COLOR_OVERLAP

        # High complement color
        colored_img[
            (
                high_positive_mask1
                & (negative_mask2 | background_mask2)
                & (negative_mask3 | background_mask3)
            )
            | (
                high_positive_mask2
                & (negative_mask1 | background_mask1)
                & (negative_mask3 | background_mask3)
            )
            | (
                high_positive_mask3
                & (negative_mask1 | background_mask1)
                & (negative_mask2 | background_mask2)
            )
        ] = COLOR_TILE1[0]

        # Positive complement color
        colored_img[
            (
                medium_positive_mask1
                & (negative_mask2 | background_mask2)
                & (negative_mask3 | background_mask3)
            )
            | (
                medium_positive_mask2
                & (negative_mask1 | background_mask1)
                & (negative_mask3 | background_mask3)
            )
            | (
                medium_positive_mask3
                & (negative_mask1 | background_mask1)
                & (negative_mask2 | background_mask2)
            )
        ] = COLOR_TILE1[1]

        # Low complement color
        colored_img[
            (
                low_positive_mask1
                & (negative_mask2 | background_mask2)
                & (negative_mask3 | background_mask3)
            )
            | (
                low_positive_mask2
                & (negative_mask1 | background_mask1)
                & (negative_mask3 | background_mask3)
            )
            | (
                low_positive_mask3
                & (negative_mask1 | background_mask1)
                & (negative_mask2 | background_mask2)
            )
        ] = COLOR_TILE1[2]

        # Negative color
        colored_img[
            negative_mask1 & negative_mask2 & negative_mask3
            | negative_mask1 & negative_mask2 & background_mask3
            | negative_mask1 & background_mask2 & negative_mask3
            | background_mask1 & negative_mask2 & negative_mask3
            | negative_mask1 & background_mask2 & background_mask3
            | background_mask1 & negative_mask2 & background_mask3
            | background_mask1 & background_mask2 & negative_mask3
        ] = [255, 255, 255]

        # Background color
        colored_img[background_mask1 & background_mask2 & background_mask3] = [
            192,
            192,
            192,
        ]

        colored_img = to_numpy(colored_img)
        img = Image.fromarray(colored_img)
        img.save(f"{dir}/{tile1['Tilename']}.tif")

    return (
        int(high_overlap),
        int(med_overlap),
        int(low_overlap),
        int(high_complement),
        int(med_complement),
        int(low_complement),
        int(negative),
        int(background),
    )
