# Standard Library
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import numpy as np
from PIL import Image

# CuBATS
from cubats.slide_collection.tile_processing import (calculate_pixel_intensity,
                                                     calculate_score,
                                                     ihc_stain_separation,
                                                     quantify_tile)


class TestQuantifyTile(unittest.TestCase):
    @patch('cubats.slide_collection.tile_processing.ihc_stain_separation')
    @patch('cubats.slide_collection.tile_processing.calculate_pixel_intensity')
    def test_quantify_tile_not_processed(self, mock_calculate_pixel_intensity, mock_ihc_stain_separation):
        # Create a mostly white tile
        white_tile = Image.fromarray(np.full((10, 10, 3), 255, dtype=np.uint8))
        iterable = [0, 0, white_tile, '/fake/dir', False]

        result = quantify_tile(iterable)

        self.assertEqual(result['Tilename'], '0_0')
        self.assertEqual(result['Flag'], 0)
        self.assertNotIn('Histogram', result)
        self.assertNotIn('Hist_centers', result)
        self.assertNotIn('Zones', result)
        self.assertNotIn('Percentage', result)
        self.assertNotIn('Score', result)
        self.assertNotIn('Px_count', result)
        self.assertNotIn('Image Array', result)

    @patch('cubats.slide_collection.tile_processing.ihc_stain_separation')
    @patch('cubats.slide_collection.tile_processing.calculate_pixel_intensity')
    def test_quantify_tile_processed(self, mock_calculate_pixel_intensity, mock_ihc_stain_separation):
        # Create a tile that should be processed
        tile = Image.fromarray(np.random.randint(
            0, 255, (10, 10, 3), dtype=np.uint8))
        iterable = [1, 1, tile, '/fake/dir', False]

        # Mock the return values of the called functions
        mock_ihc_stain_separation.return_value = (np.random.randint(
            0, 255, (10, 10, 3), dtype=np.uint8), None, None)
        mock_calculate_pixel_intensity.return_value = (
            np.random.rand(256), np.random.rand(256), np.random.rand(
                5), np.random.rand(5), np.random.rand(5), 100, np.random.rand(10, 10)
        )

        result = quantify_tile(iterable)

        self.assertEqual(result['Tilename'], '1_1')
        self.assertEqual(result['Flag'], 1)
        self.assertIn('Histogram', result)
        self.assertIn('Hist_centers', result)
        self.assertIn('Zones', result)
        self.assertIn('Percentage', result)
        self.assertIn('Score', result)
        self.assertIn('Px_count', result)
        self.assertIn('Image Array', result)

    def test_quantify_tile_empty_iterable(self):
        iterable = []

        with self.assertRaises(IndexError):
            quantify_tile(iterable)

    def test_quantify_tile_none_tile(self):
        iterable = [1, 1, None, '/fake/dir', False]

        with self.assertRaises(AttributeError):
            quantify_tile(iterable)

    def test_quantify_tile_black_tile(self):
        # Create a black tile
        tile_array = np.zeros((10, 10, 3), dtype=np.uint8)
        tile = Image.fromarray(tile_array)
        iterable = [1, 1, tile, '/fake/dir', False]

        result = quantify_tile(iterable)

        self.assertEqual(result['Tilename'], '1_1')
        self.assertEqual(result['Flag'], 0)
        self.assertNotIn('Histogram', result)
        self.assertNotIn('Hist_centers', result)
        self.assertNotIn('Zones', result)
        self.assertNotIn('Percentage', result)
        self.assertNotIn('Score', result)
        self.assertNotIn('Px_count', result)
        self.assertNotIn('Image Array', result)

    @patch('cubats.slide_collection.tile_processing.os.makedirs')
    @patch('cubats.slide_collection.tile_processing.Image.fromarray')
    @patch('cubats.slide_collection.tile_processing.ihc_stain_separation')
    @patch('cubats.slide_collection.tile_processing.calculate_pixel_intensity')
    def test_quantify_tile_save_img_true_dir_none(self, mock_calculate_pixel_intensity, mock_ihc_stain_separation,
                                                  mock_fromarray, mock_makedirs):
        # Create a numpy array with mean < 235 and std > 15 so it passes function conditions
        mock_tile_np = np.random.normal(
            loc=100, scale=20, size=(10, 10, 3)).astype(np.uint8)
        print(
            f"Mock tile mean: {mock_tile_np.mean()}, std: {mock_tile_np.std()}")
        assert mock_tile_np.mean() < 235
        assert mock_tile_np.std() > 15

        # Mock the tile and its conversion to numpy array
        mock_tile = MagicMock()
        mock_tile.convert.return_value = mock_tile
        mock_tile.__array__ = lambda: mock_tile_np

        # Mock ihc_stain_separation
        mock_ihc_stain_separation.return_value = (
            mock_tile_np, mock_tile_np, mock_tile_np)

        # Mock calculate_pixel_intensity
        mock_calculate_pixel_intensity.return_value = (
            [], [], [], [], [], [], [])

        # Call the function and expect a ValueError
        # CuBATS
        from cubats.slide_collection.tile_processing import quantify_tile
        with self.assertRaises(ValueError) as context:
            quantify_tile([0, 1, mock_tile, None, True])

        self.assertEqual(str(context.exception),
                         "Target directory must be specified if save_img is True")


class TestIHCStainSeparation(unittest.TestCase):

    def setUp(self):
        # Create a sample RGB image
        self.ihc_rgb = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 0], [0, 255, 255], [255, 0, 255]],
            [[128, 128, 128], [64, 64, 64], [32, 32, 32]]
        ], dtype=np.uint8)

    def test_ihc_stain_separation_all_false(self):
        ihc_d, ihc_h, ihc_e = ihc_stain_separation(
            self.ihc_rgb, hematoxylin=False, eosin=False)

        # Check that hematoxylin and eosin images are None
        self.assertIsNone(ihc_h)
        self.assertIsNone(ihc_e)

        # Check that DAB image is not None
        self.assertIsNotNone(ihc_d)

    def test_ihc_stain_separation_hematoxylin_true(self):
        ihc_d, ihc_h, ihc_e = ihc_stain_separation(
            self.ihc_rgb, hematoxylin=True, eosin=False)

        # Check that hematoxylin image is not None
        self.assertIsNotNone(ihc_h)

        # Check that eosin image is None
        self.assertIsNone(ihc_e)

        # Check that DAB image is not None
        self.assertIsNotNone(ihc_d)

    def test_ihc_stain_separation_eosin_true(self):
        ihc_d, ihc_h, ihc_e = ihc_stain_separation(
            self.ihc_rgb, hematoxylin=False, eosin=True)

        # Check that hematoxylin image is None
        self.assertIsNone(ihc_h)

        # Check that eosin image is not None
        self.assertIsNotNone(ihc_e)

        # Check that DAB image is not None
        self.assertIsNotNone(ihc_d)

    def test_ihc_stain_separation_all_true(self):
        ihc_d, ihc_h, ihc_e = ihc_stain_separation(
            self.ihc_rgb, hematoxylin=True, eosin=True)

        # Check that hematoxylin image is not None
        self.assertIsNotNone(ihc_h)

        # Check that eosin image is not None
        self.assertIsNotNone(ihc_e)

        # Check that DAB image is not None
        self.assertIsNotNone(ihc_d)


class TestCalculatePixelIntensity(unittest.TestCase):

    def test_simple_image(self):
        # Create a 3x3 image with each pixel representing a different zone
        image = np.array([
            [[0, 0, 0], [61, 61, 61], [120, 120, 120]],
            [[121, 121, 121], [181, 181, 181], [240, 240, 240]],
            [[255, 255, 255], [236, 236, 236], [20, 20, 20]]
        ], dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = calculate_pixel_intensity(
            image)

        # Expected zones
        expected_zones = np.array([2, 2, 1, 1, 3])
        np.testing.assert_array_equal(zones, expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / pixelcount) * 100
        np.testing.assert_array_almost_equal(percentage, expected_percentage)

        # Check pixel count
        self.assertEqual(pixelcount, 9)

        # Check img_analysis: 255 if pixel is < 181, else pixel value
        expected_img_analysis = np.array([
            [0, 61, 120],
            [121, 181, 255],
            [255, 255, 20]
        ], dtype=np.uint8)
        np.testing.assert_array_equal(img_analysis, expected_img_analysis)

    def test_all_high_positive(self):
        # Create an image where all pixels are high positive
        image = np.full((2, 2, 3), 50, dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = calculate_pixel_intensity(
            image)

        # Expected zones
        expected_zones = np.array([4, 0, 0, 0, 0])
        np.testing.assert_array_equal(zones, expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / pixelcount) * 100
        np.testing.assert_array_almost_equal(percentage, expected_percentage)

        # Check pixel count
        self.assertEqual(pixelcount, 4)

        # Check img_analysis
        expected_img_analysis = np.full((2, 2), 50, dtype=np.uint8)
        np.testing.assert_array_equal(img_analysis, expected_img_analysis)

    def test_all_positive(self):
        # Create an image where all pixels are positive
        image = np.full((2, 2, 3), 100, dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = calculate_pixel_intensity(
            image)

        # Expected zones
        expected_zones = np.array([0, 4, 0, 0, 0])
        np.testing.assert_array_equal(zones, expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / pixelcount) * 100
        np.testing.assert_array_almost_equal(percentage, expected_percentage)

        # Check pixel count
        self.assertEqual(pixelcount, 4)

        # Check img_analysis
        expected_img_analysis = np.full((2, 2), 100, dtype=np.uint8)
        np.testing.assert_array_equal(img_analysis, expected_img_analysis)

    def test_all_low_positive(self):
        # Create an image where all pixels are low positive
        image = np.full((2, 2, 3), 180, dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = calculate_pixel_intensity(
            image)

        # Expected zones
        expected_zones = np.array([0, 0, 4, 0, 0])
        np.testing.assert_array_equal(zones, expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / pixelcount) * 100
        np.testing.assert_array_almost_equal(percentage, expected_percentage)

        # Check pixel count
        self.assertEqual(pixelcount, 4)

        # Check img_analysis
        expected_img_analysis = np.full((2, 2), 180, dtype=np.uint8)
        np.testing.assert_array_equal(img_analysis, expected_img_analysis)

    def test_all_negative(self):
        # Create an image where all pixels are negative
        image = np.full((2, 2, 3), 200, dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = calculate_pixel_intensity(
            image)

        # Expected zones
        expected_zones = np.array([0, 0, 0, 4, 0])
        np.testing.assert_array_equal(zones, expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / pixelcount) * 100
        np.testing.assert_array_almost_equal(percentage, expected_percentage)

        # Check pixel count
        self.assertEqual(pixelcount, 4)

        # Check img_analysis
        expected_img_analysis = np.full((2, 2), 200, dtype=np.uint8)
        np.testing.assert_array_equal(img_analysis, expected_img_analysis)

    def test_all_background(self):
        # Create an image where all pixels are background
        image = np.full((2, 2, 3), 255, dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = calculate_pixel_intensity(
            image)

        # Expected zones
        expected_zones = np.array([0, 0, 0, 0, 4])
        np.testing.assert_array_equal(zones, expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / pixelcount) * 100
        np.testing.assert_array_almost_equal(percentage, expected_percentage)

        # Check pixel count
        self.assertEqual(pixelcount, 4)

        # Check img_analysis
        expected_img_analysis = np.full((2, 2), 255, dtype=np.uint8)
        np.testing.assert_array_equal(img_analysis, expected_img_analysis)

    def test_empty_image(self):
        # Create an image with all pixels set to zero
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = calculate_pixel_intensity(
            image)

        # Expected zones
        expected_zones = np.array([4, 0, 0, 0, 0])
        np.testing.assert_array_equal(zones, expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / pixelcount) * 100
        np.testing.assert_array_almost_equal(percentage, expected_percentage)

        # Check pixel count
        self.assertEqual(pixelcount, 4)

        # Check img_analysis
        expected_img_analysis = np.zeros((2, 2), dtype=np.uint8)
        np.testing.assert_array_equal(img_analysis, expected_img_analysis)


class TestCalculateScore(unittest.TestCase):

    def test_normal_case(self):
        zones = np.array([10, 20, 30, 40, 0])
        count = 100
        expected_percentage = np.array([10.0, 20.0, 30.0, 40.0, 0.0])
        expected_score = np.array([0.4, 0.6, 0.6, 0.4, 0.0])

        percentage, score = calculate_score(zones, count)

        np.testing.assert_array_almost_equal(percentage, expected_percentage)
        np.testing.assert_array_almost_equal(score, expected_score)

    def test_edge_case_zeros(self):
        zones = np.array([0, 0, 0, 0])
        count = 100
        expected_percentage = np.array([0.0, 0.0, 0.0, 0.0])
        expected_score = np.array([0.0, 0.0, 0.0, 0.0])

        percentage, score = calculate_score(zones, count)

        np.testing.assert_array_almost_equal(percentage, expected_percentage)
        np.testing.assert_array_almost_equal(score, expected_score)

    def test_invalid_input_zero_count(self):
        zones = np.array([10, 20, 30, 40])
        count = 0

        with self.assertRaises(ZeroDivisionError):
            calculate_score(zones, count)

    def test_large_numbers(self):
        zones = np.array([1000000, 2000000, 3000000, 4000000, 0])
        count = 10000000
        expected_percentage = np.array([10.0, 20.0, 30.0, 40.0, 0.0])
        expected_score = np.array([0.4, 0.6, 0.6, 0.4, 0.0])

        percentage, score = calculate_score(zones, count)

        np.testing.assert_array_almost_equal(percentage, expected_percentage)
        np.testing.assert_array_almost_equal(score, expected_score)


class TestTileProcessing(unittest.TestCase):

    @patch('cubats.slide_collection.tile_processing.os.makedirs')
    @patch('cubats.slide_collection.tile_processing.tiff.imsave')
    @patch('cubats.slide_collection.tile_processing.ihc_stain_separation')
    # Mock tqdm to pass through
    @patch('cubats.slide_collection.tile_processing.tqdm', side_effect=lambda x: x)
    def test_separate_stains_and_save__tiles_as_tif(self, mock_tqdm, mock_ihc_stain_separation, mock_imsave,
                                                    mock_makedirs):
        # Mock deepzoom_object
        mock_deepzoom_object = MagicMock()
        mock_deepzoom_object.level_tiles = {0: (2, 2)}
        mock_tile = MagicMock()
        mock_tile.convert.return_value = mock_tile
        mock_deepzoom_object.get_tile.return_value = mock_tile
        mock_tile_np = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        mock_tile.__array__ = lambda: mock_tile_np

        # Mock ihc_stain_separation
        mock_ihc_stain_separation.return_value = (
            mock_tile_np, mock_tile_np, mock_tile_np)

        # Call the function
        # CuBATS
        from cubats.slide_collection.tile_processing import \
            separate_stains_and_save__tiles_as_tif
        separate_stains_and_save__tiles_as_tif(
            mock_deepzoom_object, 1, '/mock/target_directory')

        # Log calls to mock_imsave
        print(f"mock_imsave.call_count: {mock_imsave.call_count}")
        for call in mock_imsave.call_args_list:
            print(f"mock_imsave called with args: {call}")

        # Check if directories are created
        mock_makedirs.assert_any_call(
            '/mock/target_directory/original_tiles', exist_ok=True)
        mock_makedirs.assert_any_call(
            '/mock/target_directory/DAB_tiles', exist_ok=True)
        mock_makedirs.assert_any_call(
            '/mock/target_directory/H_tiles', exist_ok=True)
        mock_makedirs.assert_any_call(
            '/mock/target_directory/E_tiles', exist_ok=True)

        # Check if tiles are saved
        # Adjusted expected call count
        self.assertEqual(mock_imsave.call_count, 16)

        # Check if ihc_stain_separation is called
        self.assertEqual(mock_ihc_stain_separation.call_count, 4)


if __name__ == '__main__':
    unittest.main()
