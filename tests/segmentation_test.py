# Standard Library
import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import pyvips
from openslide import OpenSlide
from PIL import Image

# CuBATS
from cubats import logging_config  # Correctly import logging_config
from cubats.segmentation import (_save_segmented_wsi, _save_thumbnail,
                                 _segment_file, run_tumor_segmentation)


class TestRunTumorSegmentation(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.test_dir = tempfile.TemporaryDirectory()

        # Path to a valid image
        self.valid_image_path = os.path.join(
            self.test_dir.name, 'ref_norm.png')
        Image.new('RGB', (512, 512)).save(self.valid_image_path)

        # Path to a valid model
        self.valid_model_path = os.path.join(self.test_dir.name, 'model.onnx')
        with open(self.valid_model_path, 'w') as f:
            f.write('mock model content')

        # Path to an invalid input
        self.invalid_input_path = os.path.join(
            self.test_dir.name, 'nonexistent.png')

        # Path to an invalid model
        self.invalid_model_path = os.path.join(
            self.test_dir.name, 'invalid_model.onnx')
        with open(self.invalid_model_path, 'w') as f:
            f.write('invalid model content')

        # Path to an invalid output
        self.invalid_output_path = os.path.join(
            self.test_dir.name, 'invalid_output')

        # Ensure the logs directory exists
        logs_dir = os.path.join(self.test_dir.name, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Initialize logging configuration
        logging.config.dictConfig(logging_config.LOGGING)
        self.logger = logging.getLogger(__name__)

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    @patch('onnx.checker.check_model')
    @patch('onnx.load', side_effect=FileNotFoundError("Model file not found"))
    def test_invalid_model_path(self, mock_load, mock_check_model):
        # Check if the error is thrown correctly
        with self.assertRaises(ValueError) as context:
            run_tumor_segmentation(
                input_path=self.valid_image_path,
                model_path=self.invalid_model_path,
                tile_size=(256, 256),
                output_path=None,
                normalization=False,
                inversion=False,
                plot_results=False
            )
        self.assertIn("is not a valid path", str(context.exception))

    @patch('onnx.checker.check_model')
    @patch('onnx.load')
    @patch('onnx2torch.convert')
    def test_invalid_input_path(self, mock_convert, mock_load, mock_check_model):
        # Mock the model loading and conversion
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_convert.return_value = mock_model
        mock_check_model.return_value = None  # Mock the check_model function

        # Check if the error is thrown correctly
        with self.assertRaises(FileNotFoundError) as context:
            run_tumor_segmentation(
                input_path=self.invalid_input_path,
                model_path=self.valid_model_path,
                tile_size=(256, 256),
                output_path=None,
                normalization=False,
                inversion=False,
                plot_results=False
            )
        self.assertIn("Input path", str(context.exception))

    @patch('onnx.checker.check_model')
    @patch('onnx.load')
    @patch('onnx2torch.convert')
    def test_invalid_output_path(self, mock_convert, mock_load, mock_check_model):
        # Mock the model loading and conversion
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_convert.return_value = mock_model
        mock_check_model.return_value = None  # Mock the check_model function

        # Check if the error is thrown correctly
        with self.assertRaises(FileNotFoundError) as context:
            run_tumor_segmentation(
                input_path=self.valid_image_path,
                model_path=self.valid_model_path,
                tile_size=(256, 256),
                output_path=self.invalid_output_path,
                normalization=False,
                inversion=False,
                plot_results=False
            )
        self.assertIn("Output path", str(context.exception))

    @patch('onnx.checker.check_model')
    @patch('onnx.load')
    @patch('onnx2torch.convert')
    def test_invalid_output_path_not_directory(self, mock_convert, mock_load, mock_check_model):
        # Mock the model loading and conversion
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_convert.return_value = mock_model
        mock_check_model.return_value = None  # Mock the check_model function

        # Create a file instead of a directory for the invalid output path
        with open(self.invalid_output_path, 'w') as f:
            f.write('This is a file, not a directory.')

        # Check if the error is thrown correctly
        with self.assertRaises(ValueError) as context:
            run_tumor_segmentation(
                input_path=self.valid_image_path,
                model_path=self.valid_model_path,
                tile_size=(256, 256),
                output_path=self.invalid_output_path,
                normalization=False,
                inversion=False,
                plot_results=False
            )
        self.assertIn("is not a directory", str(context.exception))

    @patch('onnx.checker.check_model')
    @patch('onnx.load')
    @patch('onnx2torch.convert', side_effect=RuntimeError("Error converting model"))
    def test_model_conversion_error(self, mock_convert, mock_load, mock_check_model):
        # Mock the model loading and conversion
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_model.graph.input[0].type.tensor_type.shape.dim = [MagicMock(dim_value=1), MagicMock(
            dim_value=3), MagicMock(dim_value=224), MagicMock(dim_value=224)]
        mock_check_model.return_value = None  # Mock the check_model function

        # Check if the error is thrown correctly
        with self.assertRaises(RuntimeError) as context:
            run_tumor_segmentation(
                input_path=self.valid_image_path,
                model_path=self.valid_model_path,
                tile_size=(256, 256),
                output_path=None,
                normalization=False,
                inversion=False,
                plot_results=False
            )
        self.assertIn("Error converting model", str(context.exception))


class TestSegmentFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.test_dir = tempfile.TemporaryDirectory()

        # Path to a valid WSI file
        self.valid_wsi_path = os.path.join(self.test_dir.name, 'valid_wsi.tif')
        Image.new('RGB', (512, 512)).save(self.valid_wsi_path)

        # Path to an invalid WSI file
        self.invalid_wsi_path = os.path.join(
            self.test_dir.name, 'invalid_wsi.tif')

        # Path to the output directory
        self.output_path = os.path.join(self.test_dir.name, 'output')
        os.makedirs(self.output_path, exist_ok=True)

        # Initialize logging configuration
        logging.config.dictConfig(logging_config.LOGGING)
        self.logger = logging.getLogger('cubats.segmentation')

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    @patch('openslide.OpenSlide')
    def test_open_slide_error(self, mock_open_slide):
        # Mock OpenSlide to raise an error
        mock_open_slide.side_effect = Exception("Error opening slide file")

        # Check if the error is logged correctly
        with self.assertLogs(self.logger, level='ERROR') as log:
            _segment_file(
                self.invalid_wsi_path,
                MagicMock(),
                (256, 256),
                (1, 3, 224, 224),
                self.output_path,
                normalization=False,
                inversion=False,
                plot_results=False
            )
            self.assertTrue(len(log.output) > 0)
            self.assertIn("Error opening slide file", log.output[0])


class TestSaveSegmentedWSI(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.test_dir = tempfile.TemporaryDirectory()

        # Path to the output file
        self.output_path = os.path.join(
            self.test_dir.name, 'segmented_wsi.tif')

        # Create a dummy segmented WSI
        self.segmented_wsi = MagicMock(spec=pyvips.Image)
        self.segmented_wsi.width = 512
        self.segmented_wsi.height = 512

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    @patch.object(pyvips.Image, 'crop', return_value=MagicMock())
    def test_save_segmented_wsi_success(self, mock_crop):
        try:
            _save_segmented_wsi(self.segmented_wsi,
                                (256, 256), self.output_path)
        except Exception as e:
            self.fail(f"_save_segmented_wsi raised an exception: {e}")

    @patch('pyvips.Image.tiffsave', side_effect=Exception("Error saving WSI"))
    def test_save_segmented_wsi_error(self, mock_tiffsave):
        # Test error handling
        with self.assertLogs('cubats.segmentation', level='ERROR') as log:
            _save_segmented_wsi(self.segmented_wsi,
                                (256, 256), self.output_path)
            self.assertIn("Error saving segmented WSI", log.output[0])


class TestSaveThumbnail(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.test_dir = tempfile.TemporaryDirectory()

        # Path to the WSI file
        self.wsi_path = os.path.join(self.test_dir.name, 'wsi.tif')
        Image.new('RGB', (512, 512)).save(self.wsi_path)

        # Path to the output file
        self.output_path = os.path.join(self.test_dir.name, 'thumbnail.png')

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    @patch('openslide.OpenSlide.get_thumbnail')
    @patch('openslide.OpenSlide')
    def test_save_thumbnail_success(self, mock_openslide, mock_get_thumbnail):
        # Mock the OpenSlide and get_thumbnail functions
        mock_slide = MagicMock(spec=OpenSlide)
        mock_openslide.return_value = mock_slide
        mock_thumbnail = Image.new('RGB', (512, 512))
        mock_get_thumbnail.return_value = mock_thumbnail

        # Test successful save
        try:
            _save_thumbnail(self.wsi_path, self.output_path)
        except Exception as e:
            self.fail(f"_save_thumbnail raised an exception: {e}")

    @patch('openslide.OpenSlide', side_effect=Exception("Error opening WSI"))
    def test_save_thumbnail_error(self, mock_openslide):
        # Test error handling
        with self.assertLogs('cubats.segmentation', level='ERROR') as log:
            _save_thumbnail(self.wsi_path, self.output_path)
            self.assertIn("Error saving PNG thumbnail", log.output[0])


if __name__ == '__main__':
    unittest.main()
