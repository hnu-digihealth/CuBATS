# TODO add docstrings
# TODO add type hints
# TODO add unit tests
# TODO add logging
# TODO add error (handling) throwing where ever needed
# TODO add multi processing

# Standard Library
import logging
from os import listdir, path
from time import time
from typing import List, Tuple, Union

# Third Party
import onnx
from numpy import asarray, concatenate, ubyte, uint8
from onnx2torch import convert
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from pyvips import BandFormat
from pyvips import Image as VipsImage
from skimage.transform import resize
from torch import float32, from_numpy
from tqdm import tqdm

# CuBATS
import cubats.logging_config as log_config
from cubats import Utils as utils

logging.config.dictConfig(log_config.LOGGING)
logger = logging.getLogger(__name__)
# Currently only works for pytorch input order, as some steps are hardcoded and onnx2torch is used
# TODO remove input_size from vars, can be calculated from onnx model via model.graph.input
# TODO add support for a heatmap output (optional or alternative)
# TODO fix tile_size logic -> deepzoom gnerator only takes quadratic tiles, maybe change to int overall?

# model_input_size = [int] before


def run_segmentation_pipeline(
    input_path: str,
    model_path: str,
    tile_size: Tuple[int, int],
    model_input_size: List[int],
    output_path: Union[str, None] = None,
    plot_results=False
):
    """

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

    # suppress Pillow warning because of the large WSI
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
    """

    """
    slide = OpenSlide(file_path)
    slide_generator = DeepZoomGenerator(
        slide, tile_size=tile_size[0], overlap=0, limit_bounds=False)

    # calculate if there is an overhang for the final tile at the right and bottom edge
    needs_right_padding = (
        slide_generator.level_tiles[-1][0] * tile_size[0] - slide.dimensions[0]) > 0
    needs_bottom_padding = (
        slide_generator.level_tiles[-1][1] * tile_size[1] - slide.dimensions[1]) > 0

    # array to save the segmented WSI rows as single numpy arrays
    row_array = []

    # segment each tile
    for row in tqdm(
        range(0, slide_generator.level_tiles[-1][1]),
        desc=f'Segmenting Rows for {path.splitext(path.basename(file_path))[0]}',
    ):
        # initially no padding is required
        requires_padding_right = False
        # array to save the segmented tiles of all columns of a row
        column_array = []

        if row == slide_generator.level_tiles[-1][1] - 1 and needs_right_padding:
            requires_padding_right = True

        for column in range(0, slide_generator.level_tiles[-1][0]):
            requires_padding_bottom = False

            if column == slide_generator.level_tiles[-1][0] - 1 and needs_bottom_padding:
                requires_padding_bottom = True

            if requires_padding_right or requires_padding_bottom:
                raw_tile = slide_generator.get_tile(
                    slide_generator.level_count - 1, (column, row))
                tile = Image.new("RGB", (1024, 1024), (255, 255, 255))
                tile.paste(raw_tile, (0, 0))
            else:
                tile = slide_generator.get_tile(
                    slide_generator.level_count - 1, (column, row))

            column_array.append(_segment_tile(tile, model, model_input_size))

        segmented_row = concatenate(column_array, axis=1)
        row_array.append(segmented_row)

    # recombine image rows to single wsi amd save to file
    # TODO check tiffsave parameters
    logger.debug("Constructing segmented WSI")
    segmented_wsi = concatenate(row_array, axis=0)
    segmented_wsi = VipsImage.new_from_array(
        segmented_wsi).cast(BandFormat.INT)

    # utils call due to name cleaning
    wsi_name, file_type = path.splitext(file_path)
    wsi_name = utils.get_name(wsi_name) + "_mask" + file_type

    logger.debug(f"Saving segmented WSI to {path.join(output_path, wsi_name)}")
    segmented_wsi.crop(0, 0, slide.dimensions[0], slide.dimensions[1]). \
        tiffsave(
            path.join(output_path, wsi_name),
            tile=True, compression='jpeg', bigtiff=True,
            pyramid=True, tile_width=256, tile_height=256
    )
    logger.debug(f"Segmented WSI saved to {path.join(output_path, wsi_name)}")

    # if plot_results: save cropped segmentation result on top op original image
    if plot_results:
        _plot_segmentation_on_tissue(file_path, output_path)


def _segment_tile(tile: Image, model, model_input_size) -> Image:
    """

    """
    # only pytorch due to hardcoded input shape logic
    resclaed_tile = resize(
        asarray(tile), (model_input_size[2], model_input_size[3]))
    # parse to pytroch format (C, H, W)
    resclaed_tile = from_numpy(resclaed_tile).type(float32).permute(2, 0, 1)
    segmentation = model(resclaed_tile)
    # output to binary mask
    segmentation = segmentation.sigmoid()
    segmentation = segmentation.squeeze(0).squeeze(
        0)  # Removes Batch and single color dimension
    # parse to numpy
    segmentation = segmentation.detach().numpy()
    # rescale to original size
    tile = resize(segmentation, (tile.size[0], tile.size[1]))
    tile = Image.fromarray((tile * 255).astype(uint8))

    # move this in the numpy are and do it on the 0.5 threshold instead
    # remove "heatmap" effect and make binary mask
    tile.point(lambda x: 1 if x > 120 else 0, mode='1')

    return tile


# TODO fix maxk overlay logic
# - only apply mask with alpha and remove white form mask
# - don't use alpha on slide itself
def _plot_segmentation_on_tissue(file_path, output_path):
    """

    """
    logger.debug("Plotting thumbnail for segmented WSI")
    slide = OpenSlide(file_path)

    wsi_name, file_type = path.splitext(file_path)
    wsi_name = wsi_name + "_segmented" + file_type
    mask_path = path.join(output_path, path.basename(wsi_name))
    mask = OpenSlide(mask_path)

    slide = slide.get_thumbnail(
        (slide.dimensions[0] / 8, slide.dimensions[1] / 8))
    mask = mask.get_thumbnail((mask.dimensions[0] / 8, mask.dimensions[1] / 8))

    logger.debug(f"Saving thumbnail to {path.join(output_path, wsi_name)}")
    png_name, _ = path.splitext(file_path)
    png_name = png_name + '_segmentation' + '.png'
    img = ubyte(0.5 * asarray(slide) + 0.5 * asarray(mask))
    img = Image.fromarray(img).save(path.join(output_path, png_name))
