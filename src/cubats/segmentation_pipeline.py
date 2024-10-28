# TODO add type hints
# TODO add unit tests
# TODO add multi processing

# Standard Library
import logging
from os import listdir, path
from time import time
from typing import List, Tuple, Union

# Third Party
import numpy as np
import onnx
import torch
import torchstain
import torchvision
from numpy import concatenate
from onnx2torch import convert
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from pyvips import BandFormat
from pyvips import Image as VipsImage
from skimage.transform import resize
from tqdm import tqdm

# CuBATS
import cubats.logging_config as log_config
from cubats import Utils as utils

# Initialize logging
logging.config.dictConfig(log_config.LOGGING)
logger = logging.getLogger(__name__)
# Suppress pyvips logs
logging.getLogger("pyvips").setLevel(logging.WARNING)

# Currently only works for pytorch input order, as some steps are hardcoded and onnx2torch is used
# TODO remove input_size from vars, can be calculated from onnx model via model.graph.input
# TODO add support for a heatmap output (optional or alternative)
# TODO fix tile_size logic -> deepzoom gnerator only takes quadratic tiles, maybe change to int overall?


def init_normalizer(path_to_src_img):
    """Initializes a Reinhard normalizer for image normalization.

    Args:
         path_to_src_img (str): Path to the source image file.

    Returns:
        torchstain.normalizers.ReinhardNormalizer: An instance of the Reinhard normalizer fitted to the source image.
    """
    normalizer = torchstain.normalizers.ReinhardNormalizer(
        method='modified', backend='torch')
    src_img = torchvision.io.read_image(path_to_src_img)
    normalizer.fit(src_img)
    return normalizer


def run_segmentation_pipeline(
    input_path: str,
    model_path: str,
    tile_size: Tuple[int, int],
    model_input_size: List[int],
    output_path: Union[str, None] = None,
    plot_results=False
):
    """ Run the segmentation pipeline on the given input path using the
    specified model.

    Performs segmentation on a single HE stained WSI or all HE stained WSIs in
    a directory using the specified model. The segmentation results are saved
    in the output directory. If no output directory is provided, the results
    are saved in the same directory as the input. Optionally, a thumbnail of
    the segmentation results can be plotted on the original image.

    Args:
        input_path (str): The path to the input file or directory.
        model_path (str): The path to the ONNX model file.
        tile_size (Tuple[int, int]): The size of each tile for segmentation.
        model_input_size ([int]): The input size of the model. For example, [1, 3, 1024, 1024].
        output_path (Union[str, None], optional): The path to the output directory. If not provided, the output will be saved in the same directory as the input. Defaults to None.
        plot_results (bool, optional): Whether to plot the segmentation results. Defaults to False.

    Raises:
        FileNotFoundError: If the input path or output path does not exist.
        ValueError: If the output path is not a directory or the model path is invalid.

    Returns:
        None
    """
    logger.info(
        f'Starting segmentation of {path.splitext(path.basename(input_path))[0]} using model {path.splitext(path.basename(model_path))[0]}')
    start_time_segmentation = time()
    # check if the input path is valid and if it is a file or a directory
    if not path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist.")
    if path.isfile(input_path):
        segment_single_file = True
        input_folder = path.dirname(input_path)
    else:
        segment_single_file = False
        input_folder = input_path

    # check the output path and raise an error if it is not a path or a directory
    if output_path is None:
        output_path = input_folder
    else:
        if not path.exists(output_path):
            logger.error(f"Output path {output_path} does not exist.")
            raise FileNotFoundError(
                f"Output path {output_path} does not exist.")
        if not path.isdir(output_path):
            logger.error(f"Output path {output_path} is not a directory.")
            raise ValueError(f"Output path {output_path} is not a directory.")

    # check if the model is an valid onnx file
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
    except ValueError as e:
        logger.error(f"Model {model_path} is not a valid onnx file.")
        raise e
    except Exception:
        raise ValueError(f"Model {model_path} is not a valid path.")

    model = convert(model)

    # Suppress Pillow warning because of the large WSI
    Image.MAX_IMAGE_PIXELS = None

    # run segmentation pipeline
    if segment_single_file:
        print('segmenting single file')
        _segment_file(input_path, model, tile_size,
                      model_input_size, output_path, plot_results)
        print('file segmented')
    else:
        print('segmenting all files in folder')
        for file in listdir(input_folder):
            if file.endswith(".tif"):
                _segment_file(path.join(input_folder, file), model,
                              tile_size, model_input_size, output_path, plot_results)
        print('all files segmented')

    end_time_segmentation = time()
    logger.info(
        f'Segmentation of {input_path} completed in {end_time_segmentation - start_time_segmentation:.2f} seconds.')


# TODO add better names for the padding logic
# TODO refactor mask-WSI creation logic
def _segment_file(
    file_path,
    model,
    tile_size,
    model_input_size,
    output_path,
    plot_results
):
    """ Performs tumor detection and segmentation on a single WSI file.

    Segments a (WSI) file into tiles, performs tumor detection on each tile and
    save the reconstructed image.

    Args:
        file_path (str): The path to the WSI file.
        model: The segmentation model.
        tile_size (tuple): The size of each tile in pixels.
        model_input_size (tuple): The input size of the model.
        output_path (str): The path to save the segmented image.
        plot_results (bool): Whether to plot the segmentation results on the
        original image.

    Returns:
        None
    """
    logger.info(f"Starting segmentation for file: {file_path}")
    normalizer = init_normalizer(r"C:\Users\mlnot\Desktop\model\21585.png")
    try:
        slide = OpenSlide(file_path)
        logger.debug(f"Opened slide file: {file_path}")
    except Exception as e:
        logger.error(f"Error opening slide file {file_path}: {e}")
        return

    try:
        slide_generator = DeepZoomGenerator(
            slide, tile_size=tile_size[0], overlap=0, limit_bounds=False)
        logger.debug("Created DeepZoomGenerator")
    except Exception as e:
        logger.error(f"Error creating DeepZoomGenerator: {e}")
        return

    # calculate if there is an overhang for the final tile at the right and bottom edge
    needs_padding_right = (
        slide_generator.level_tiles[-1][0] * tile_size[0] - slide.dimensions[0]) > 0
    needs_padding_bottom = (
        slide_generator.level_tiles[-1][1] * tile_size[1] - slide.dimensions[1]) > 0

    # array to save the segmented WSI rows as single numpy arrays
    row_array = []

    # segment each tile
    for row in tqdm(
        range(0, slide_generator.level_tiles[-1][1]),
        desc=f'Segmenting Rows for {path.splitext(path.basename(file_path))[0]}',
    ):
        logger.debug(f"Processing row {row}")
        # initially no padding is required
        requires_padding_right = False
        # array to save the segmented tiles of all columns of a row
        column_array = []

        if row == slide_generator.level_tiles[-1][1] - 1 and needs_padding_right:
            requires_padding_right = True

        for column in range(0, slide_generator.level_tiles[-1][0]):
            requires_padding_bottom = False

            if column == slide_generator.level_tiles[-1][0] - 1 and needs_padding_bottom:
                requires_padding_bottom = True

            if requires_padding_right or requires_padding_bottom:
                raw_tile = slide_generator.get_tile(
                    slide_generator.level_count - 1, (column, row))
                tile = Image.new("RGB", (1024, 1024), (255, 255, 255))
                tile.paste(raw_tile, (0, 0))
                logger.debug(
                    f"Tile at row {row}, column {column} required padding")
            else:
                tile = slide_generator.get_tile(
                    slide_generator.level_count - 1, (column, row))
                logger.debug(f"Retrieved tile at row {row}, column {column}")

            # If tile is mostly empty, skip segmentation, fill with white tile
            if np.array(tile).mean() > 240:
                column_array.append(Image.new("L", tile.size, 255))
            else:
                # Convert the tile to a tensor
                transform = torchvision.transforms.ToTensor()
                tile_tensor = transform(tile).float() * 255

                # Normalize the tensor using the normalizer
                normalized_tile = normalizer.normalize(tile_tensor)

                # recreate tensor with correct shape
                normalized_tile = normalized_tile.cpu().permute(2, 0, 1).unsqueeze(0).float()

                # Segment the tile
                segmented_tile = _segment_tile(
                    normalized_tile, model, model_input_size)

                column_array.append(segmented_tile)

        segmented_row = concatenate(column_array, axis=1)
        row_array.append(segmented_row)

    logger.debug("Constructing segmented WSI")
    segmented_wsi = concatenate(row_array, axis=0)
    segmented_wsi = VipsImage.new_from_array(
        segmented_wsi).cast(BandFormat.INT)

    # Extract the base name and file type
    base_name, file_type = path.splitext(file_path)
    wsi_name = utils.get_name(base_name)

    # Construct output paths for the mask and thumbnail
    mask_out = path.join(output_path, f"{wsi_name}_mask{file_type}")
    thumb_out = path.join(output_path, f"{wsi_name}_mask_thumbnail.png")

    # Save the segmented WSI using pyvips
    save_segmented_wsi(segmented_wsi, mask_out)

    # Create and save PNG thumbnail
    save_thumbnail(segmented_wsi, thumb_out)

    # if plot_results: save cropped segmentation result on top of original image
    if plot_results:
        try:
            _plot_segmentation_on_tissue(file_path, output_path)
            logger.debug("Plotted segmentation on tissue")
        except Exception as e:
            logger.error(f"Error plotting segmentation on tissue: {e}")

    logger.info(f"Finished segmentation for file: {file_path}")


def _segment_tile(tile: torch.Tensor, model, model_input_size) -> Image:
    """
    Segments a given tile using a pre-trained model and returns the segmented tile as a PIL Image.

    Args:
        tile (torch.Tensor): The input tile to be segmented. Expected shape is (1, C, H, W).
        model: The pre-trained segmentation model.
        model_input_size (tuple): The expected input size for the model in the format (1, C, H, W).

    Returns:
        Image: The segmented tile as a binary mask in PIL Image format.
    """
    # Start segmentation
    with torch.no_grad():
        # run model
        segmentation = model(tile)

    # output to binary mask
    segmentation = segmentation.sigmoid()

    # Removes Batch and single color dimension
    segmentation = segmentation.squeeze(0).cpu().numpy()

    # Rescale to original size if necessary
    if tile.shape[2:] != (model_input_size[2], model_input_size[3]):
        original_size = tile.shape[1:]  # (H, W)
        segmented_tile = resize(
            segmentation, original_size, anti_aliasing=True)
    else:
        segmented_tile = segmentation

    # Threshold to binary mask
    segmented_tile = (segmented_tile > 0.5).astype(np.uint8)

    # Invert mask so that tissue is 1 and background is 0
    segmented_tile = 1 - segmented_tile

    # Ensure the segmented tile has the correct shape for conversion to PIL
    if segmented_tile.ndim == 3 and segmented_tile.shape[0] == 1:
        segmented_tile = segmented_tile.squeeze(0)
    elif segmented_tile.ndim == 2:
        segmented_tile = segmented_tile[:, :, None]

    # Convert to PIL image
    segmented_tile_pil = Image.fromarray(segmented_tile.squeeze() * 255)

    return segmented_tile_pil


def save_segmented_wsi(segmented_wsi, mask_out):
    """ Save the segmented WSI to file.

    Args:
        segmented_wsi (pyvips.Image): The segmented WSI.
        mask_out (str): The path to save the segmented WSI.
    """
    try:
        logger.debug(f"Saving segmented WSI to {mask_out}")
        segmented_wsi.crop(0, 0, segmented_wsi.width, segmented_wsi.height). \
            tiffsave(
                mask_out,
                tile=True, compression='jpeg', bigtiff=True,
                pyramid=True, tile_width=1024, tile_height=1024,
                Q=100,  # Set JPEG quality to 100%
                predictor="horizontal",  # Use horizontal predictor for better compression
                strip=True  # Strip metadata to reduce file size

        )
        logger.debug(f"Segmented WSI saved to {mask_out}")
    except Exception as e:
        logger.error(f"Error saving segmented WSI: {e}")


def save_thumbnail(segmented_wsi, thumb_out):
    """ Create and save PNG thumbnail.

    Args:
        segmented_wsi (pyvips.Image): The segmented WSI.
        thumb_out (str): The path to save the PNG thumbnail.
    """
    try:
        thumbnail_size = 10000  # Define the size of the thumbnail
        thumbnail = segmented_wsi.thumbnail_image(thumbnail_size)
        thumbnail.write_to_file(thumb_out)
        logger.debug(f"PNG thumbnail saved to {thumb_out}")
    except Exception as e:
        logger.error(f"Error saving PNG thumbnail: {e}")


def _plot_segmentation_on_tissue(file_path, output_path):
    """
    Plots the segmentation results on the original image and saves the result as a thumbnail.

    Args:
        file_path (str): The path to the original WSI file.
        output_path (str): The path to save the thumbnail.

    Returns:
        None
    """
    start_time = time()
    logger.info("Plotting thumbnail for segmented WSI")

    slide = OpenSlide(file_path)

    wsi_name, file_type = path.splitext(file_path)
    wsi_name = path.splitext(wsi_name)[0] + "_mask" + file_type
    mask_path = path.join(output_path, path.basename(wsi_name))
    mask = OpenSlide(mask_path)

    logger.info("Retrieving thumbnail for slide")
    slide_thumbnail = slide.get_thumbnail(
        (slide.dimensions[0] / 256, slide.dimensions[1] / 256))
    logger.info("Retrieving thumbnail for mask")
    mask_thumbnail = mask.get_thumbnail(
        (mask.dimensions[0] / 256, mask.dimensions[1] / 256))

    # Convert mask to RGBA
    logger.info("Converting mask to RGBA")
    mask_thumbnail = mask_thumbnail.convert("RGBA")

    # Split the mask into its components
    logger.info("Splitting mask into components")
    r, g, b, a = mask_thumbnail.split()

    # Create a new alpha channel where white areas are fully transparent
    logger.info("Creating new alpha channel")
    alpha = Image.eval(a, lambda px: 0 if px == 255 else 255)

    # Combine the mask with the new alpha channel
    logger.info("Combining mask with new alpha channel")
    mask_thumbnail = Image.merge("RGBA", (r, g, b, alpha))

    # Ensure the slide is in RGB mode (no alpha)
    logger.info("Converting slide to RGB")
    slide_thumbnail = slide_thumbnail.convert("RGB")

    # Composite the slide and mask
    logger.info("Compositing slide and mask")
    combined = Image.alpha_composite(
        slide_thumbnail.convert("RGBA"), mask_thumbnail)

    png_name, _ = path.splitext(file_path)
    png_name = path.splitext(png_name)[0] + '_mask' + '.png'
    logger.info(f"Saving thumbnail to {path.join(output_path, png_name)}")
    combined.save(path.join(output_path, png_name))

    end_time = time()
    logger.debug(
        f"Thumbnail creation took {end_time - start_time:.2f} seconds.")
