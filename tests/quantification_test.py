# Standard Library
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import numpy as np
from PIL import Image

# CuBATS
from cubats.config import xp
from cubats.cutils import to_numpy
from cubats.slide_collection.tile_quantification import (
    calculate_percentage_and_score, color_deconvolution,
    evaluate_staining_intensities, quantify_tile)


class TestQuantifyTile(unittest.TestCase):
    def setUp(self):
        self.antigen_profile = {
            "Name": "default",
            "low_positive_threshold": 181,
            "medium_positive_threshold": 121,
            "high_positive_threshold": 61,
        }

    @patch("cubats.slide_collection.tile_quantification.color_deconvolution")
    @patch("cubats.slide_collection.tile_quantification.evaluate_staining_intensities")
    def test_quantify_tile_not_processed(
        self, mock_evaluate_staining_intensities, mock_color_deconvolution
    ):
        # Create a mostly white tile
        white_tile = Image.fromarray(np.full((10, 10, 3), 255, dtype=np.uint8))
        iterable = [
            0,
            0,
            (white_tile, None),
            "/fake/dir",
            False,
            True,
            self.antigen_profile,
        ]

        result = quantify_tile(iterable)

        self.assertEqual(result["Tilename"], "0_0")
        self.assertEqual(result["Flag"], 0)
        self.assertNotIn("Histogram", result)
        self.assertNotIn("Hist_centers", result)
        self.assertNotIn("Zones", result)
        self.assertNotIn("Percentage", result)
        self.assertNotIn("Score", result)
        self.assertNotIn("Mask Count", result)
        self.assertNotIn("Image Array", result)

    def test_quantify_tile_with_mask(self):
        # Create a tile with high mean
        tile_np = np.full((10, 10, 3), 240, dtype=np.uint8)
        tile = Image.fromarray(tile_np)

        # Boolean mask
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:5, 0:5] = True  # top-left corner

        iterable = [2, 2, (tile, mask), "/fake/dir",
                    False, self.antigen_profile]

        with (
            patch(
                "cubats.slide_collection.tile_quantification.color_deconvolution"
            ) as mock_ihc,
            patch(
                "cubats.slide_collection.tile_quantification.evaluate_staining_intensities"
            ) as mock_calc,
        ):

            mock_ihc.return_value = (tile_np, tile_np, tile_np)
            mock_calc.return_value = ([], [], [], [], [], 25, tile_np)

            result = quantify_tile(iterable)

            self.assertEqual(result["Flag"], 1)
            self.assertEqual(result["Tilename"], "2_2")
            self.assertIn("Mask Count", result)
            self.assertEqual(result["Mask Count"], 25)

    @patch("cubats.slide_collection.tile_quantification.color_deconvolution")
    @patch("cubats.slide_collection.tile_quantification.evaluate_staining_intensities")
    def test_quantify_tile_processed(
        self, mock_evaluate_staining_intensities, mock_color_deconvolution
    ):
        # Create a tile that should be processed

        tile = Image.fromarray(np.random.randint(
            0, 255, (10, 10, 3), dtype=np.uint8))
        iterable = [1, 1, (tile, None), "/fake/dir", False,
                    True, self.antigen_profile]

        # Mock the return values of the called functions
        mock_color_deconvolution.return_value = (
            np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8),
            None,
            None,
        )
        mock_evaluate_staining_intensities.return_value = (
            np.random.rand(256),
            np.random.rand(256),
            np.random.rand(5),
            np.random.rand(5),
            np.random.rand(5),
            100,
            np.random.rand(10, 10),
        )

        result = quantify_tile(iterable)

        self.assertEqual(result["Tilename"], "1_1")
        self.assertEqual(result["Flag"], 1)
        self.assertIn("Histogram", result)
        self.assertIn("Hist_centers", result)
        self.assertIn("Zones", result)
        self.assertIn("Percentage", result)
        self.assertIn("Score", result)
        self.assertIn("Mask Count", result)
        self.assertIn("Image Array", result)

    def test_quantify_tile_processed_with_mask(self):
        tile_np = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        tile = Image.fromarray(tile_np)

        # mask
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True  # middle 6x6 region

        iterable = [1, 1, (tile, mask), "/fake/dir", False,
                    True, self.antigen_profile]

        with (
            patch(
                "cubats.slide_collection.tile_quantification.color_deconvolution"
            ) as mock_ihc,
            patch(
                "cubats.slide_collection.tile_quantification.evaluate_staining_intensities"
            ) as mock_calc,
        ):

            mock_ihc.return_value = (tile_np, tile_np, tile_np)
            mock_calc.return_value = (
                np.random.rand(256),
                np.random.rand(256),
                np.random.rand(5),
                np.random.rand(5),
                np.random.rand(5),
                36,  # 6x6 mask pixels
                tile_np,
            )

            result = quantify_tile(iterable)

            self.assertEqual(result["Flag"], 1)
            self.assertEqual(result["Tilename"], "1_1")
            self.assertIn("Mask Count", result)
            self.assertEqual(result["Mask Count"], 36)
            self.assertIn("Histogram", result)
            self.assertIn("Image Array", result)

    def test_quantify_tile_empty_iterable(self):
        iterable = []

        with self.assertRaises(IndexError):
            quantify_tile(iterable)

    def test_quantify_tile_none_tile(self):
        iterable = [1, 1, (None, None), "/fake/dir", False,
                    True, self.antigen_profile]

        with self.assertRaises(AttributeError):
            quantify_tile(iterable)

    def test_quantify_tile_black_tile(self):
        # Create a black tile
        tile_array = np.zeros((10, 10, 3), dtype=np.uint8)
        tile = Image.fromarray(tile_array)
        iterable = [1, 1, (tile, None), "/fake/dir", False,
                    True, self.antigen_profile]

        result = quantify_tile(iterable)

        self.assertEqual(result["Tilename"], "1_1")
        self.assertEqual(result["Flag"], 0)
        self.assertNotIn("Histogram", result)
        self.assertNotIn("Hist_centers", result)
        self.assertNotIn("Zones", result)
        self.assertNotIn("Percentage", result)
        self.assertNotIn("Score", result)
        self.assertNotIn("Mask Count", result)
        self.assertNotIn("Image Array", result)

    @patch("os.makedirs")
    @patch("cubats.slide_collection.tile_quantification.Image.fromarray")
    @patch("cubats.slide_collection.tile_quantification.color_deconvolution")
    @patch("cubats.slide_collection.tile_quantification.evaluate_staining_intensities")
    def test_quantify_tile_save_img_true_dir_none(
        self,
        mock_evaluate_staining_intensities,
        mock_color_deconvolution,
        mock_fromarray,
        mock_makedirs,
    ):
        # Create a numpy array with mean < 235 and std > 15 so it passes function conditions
        mock_tile_np = np.random.normal(loc=100, scale=20, size=(10, 10, 3)).astype(
            np.uint8
        )
        print(
            f"Mock tile mean: {mock_tile_np.mean()}, std: {mock_tile_np.std()}")
        assert mock_tile_np.mean() < 235
        assert mock_tile_np.std() > 15

        # Mock the tile and its conversion to numpy array
        mock_tile = MagicMock()
        mock_tile.convert.return_value = mock_tile
        mock_tile.__array__ = lambda: mock_tile_np

        # Mock color_deconvolution
        mock_color_deconvolution.return_value = (
            mock_tile_np,
            mock_tile_np,
            mock_tile_np,
        )

        # Mock evaluate_staining_intensities
        mock_evaluate_staining_intensities.return_value = (
            [], [], [], [], [], [], [])

        # Call the function and expect a ValueError
        # CuBATS
        from cubats.slide_collection.tile_quantification import quantify_tile

        with self.assertRaises(ValueError) as context:
            quantify_tile([0, 1, mock_tile, None, True,
                          True, self.antigen_profile])

        self.assertEqual(
            str(context.exception),
            "Target directory must be specified if save_img is True",
        )


class TestColorDeconvolution(unittest.TestCase):

    def setUp(self):
        # Create a sample RGB image
        self.ihc_rgb = np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[255, 255, 0], [0, 255, 255], [255, 0, 255]],
                [[128, 128, 128], [64, 64, 64], [32, 32, 32]],
            ],
            dtype=np.uint8,
        )

    def test_color_deconvolution_all_false(self):
        ihc_d, ihc_h, ihc_e = color_deconvolution(
            self.ihc_rgb, hematoxylin=False, eosin=False
        )

        # Check that hematoxylin and eosin images are None
        self.assertIsNone(ihc_h)
        self.assertIsNone(ihc_e)

        # Check that DAB image is not None
        self.assertIsNotNone(ihc_d)

    def test_color_deconvolution_hematoxylin_true(self):
        ihc_d, ihc_h, ihc_e = color_deconvolution(
            self.ihc_rgb, hematoxylin=True, eosin=False
        )

        # Check that hematoxylin image is not None
        self.assertIsNotNone(ihc_h)

        # Check that eosin image is None
        self.assertIsNone(ihc_e)

        # Check that DAB image is not None
        self.assertIsNotNone(ihc_d)

    def test_color_deconvolution_eosin_true(self):
        ihc_d, ihc_h, ihc_e = color_deconvolution(
            self.ihc_rgb, hematoxylin=False, eosin=True
        )

        # Check that hematoxylin image is None
        self.assertIsNone(ihc_h)

        # Check that eosin image is not None
        self.assertIsNotNone(ihc_e)

        # Check that DAB image is not None
        self.assertIsNotNone(ihc_d)

    def test_color_deconvolution_all_true(self):
        ihc_d, ihc_h, ihc_e = color_deconvolution(
            self.ihc_rgb, hematoxylin=True, eosin=True
        )

        # Check that hematoxylin image is not None
        self.assertIsNotNone(ihc_h)

        # Check that eosin image is not None
        self.assertIsNotNone(ihc_e)

        # Check that DAB image is not None
        self.assertIsNotNone(ihc_d)


class TestCalculatePixelIntensity(unittest.TestCase):
    def setUp(self):
        self.antigen_profile = {
            "Name": "default",
            "low_positive_threshold": 181,
            "medium_positive_threshold": 121,
            "high_positive_threshold": 61,
        }

    def test_simple_image(self):
        # Create a 3x3 image with each pixel representing a different zone
        image = np.array(
            [
                [[0, 0, 0], [61, 61, 61], [120, 120, 120]],
                [[121, 121, 121], [181, 181, 181], [240, 240, 240]],
                [[255, 255, 255], [236, 236, 236], [20, 20, 20]],
            ],
            dtype=np.uint8,
        )

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(image, self.antigen_profile)
        )

        # Expected zones
        expected_zones = np.array([2, 2, 1, 1, 3])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        # Expected percentages
        # expected_percentage = (expected_zones / to_numpy(pixelcount)) * 100
        percentage_tissue = (zones[:4] / 6) * 100
        percentage_background = (zones[4:] / 9) * 100
        expected_percentage = xp.concatenate(
            [percentage_tissue, percentage_background])

        np.testing.assert_array_almost_equal(
            to_numpy(percentage), to_numpy(expected_percentage)
        )

        # Check pixel count
        self.assertEqual(to_numpy(pixelcount), 6)

        # Check img_analysis: 255 if pixel is < 181, else pixel value
        expected_img_analysis = np.array(
            [[0, 61, 120], [121, 181, 240], [255, 236, 20]], dtype=np.uint8
        )
        np.testing.assert_array_equal(img_analysis, expected_img_analysis)

    def test_all_high_positive(self):
        # Create an image where all pixels are high positive
        image = np.full((2, 2, 3), 50, dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(image, self.antigen_profile)
        )

        # Expected zones
        expected_zones = np.array([4, 0, 0, 0, 0])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / to_numpy(pixelcount)) * 100
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        # Check pixel count
        self.assertEqual(to_numpy(pixelcount), 4)

        # Check img_analysis
        expected_img_analysis = np.full((2, 2), 50, dtype=np.uint8)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_all_positive(self):
        # Create an image where all pixels are positive
        image = np.full((2, 2, 3), 100, dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(image, self.antigen_profile)
        )

        # Expected zones
        expected_zones = np.array([0, 4, 0, 0, 0])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / to_numpy(pixelcount)) * 100
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        # Check pixel count
        self.assertEqual(to_numpy(pixelcount), 4)

        # Check img_analysis
        expected_img_analysis = np.full((2, 2), 100, dtype=np.uint8)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_all_low_positive(self):
        # Create an image where all pixels are low positive
        image = np.full((2, 2, 3), 180, dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(image, self.antigen_profile)
        )

        # Expected zones
        expected_zones = np.array([0, 0, 4, 0, 0])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / to_numpy(pixelcount)) * 100
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        # Check pixel count
        self.assertEqual(to_numpy(pixelcount), 4)

        # Check img_analysis
        expected_img_analysis = np.full((2, 2), 180, dtype=np.uint8)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_all_negative(self):
        # Create an image where all pixels are negative
        image = np.full((2, 2, 3), 200, dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(image, self.antigen_profile)
        )

        # Expected zones
        expected_zones = np.array([0, 0, 0, 4, 0])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / to_numpy(pixelcount)) * 100
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        # Check pixel count
        self.assertEqual(to_numpy(pixelcount), 4)

        # Check img_analysis
        expected_img_analysis = np.full((2, 2), 200, dtype=np.uint8)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_all_background_nomask(self):
        # Create an image where all pixels are background
        image = np.full((2, 2, 3), 255, dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(image, self.antigen_profile)
        )

        # Expected zones
        expected_zones = np.array([0, 0, 0, 0, 4])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        # Expected percentages
        expected_percentage = np.array([0, 0, 0, 0, 100])
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        # Check pixel count
        self.assertEqual(pixelcount, 0)

        # Check img_analysis
        expected_img_analysis = np.full((2, 2), 255, dtype=np.float32)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_all_background_with_mask(self):
        # Create an all-background image (all pixels = 255)
        image = np.full((2, 2, 3), 255, dtype=np.uint8)

        # Create a boolean tumor mask covering the entire image
        tumor_mask = np.ones((2, 2), dtype=bool)

        # Run the pixel intensity calculation with mask
        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(
                image, self.antigen_profile, tumor_mask=tumor_mask
            )
        )

        # Expected zones: only background pixels
        expected_zones = np.array([0, 0, 0, 0, 4])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        # Expected percentages: all background
        expected_percentage = np.array([0, 0, 0, 0, 100], dtype=np.float32)
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        # Check that the score reflects background-only tile
        self.assertEqual(score, "Background")

        # Pixel count should match the mask area (since mask defines ROI)
        expected_pixelcount = np.sum(tumor_mask)
        self.assertEqual(pixelcount, expected_pixelcount)

        # Check img_analysis: all 255 (float32)
        expected_img_analysis = np.full((2, 2), 255, dtype=np.float32)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_all_high_positive_with_mask(self):
        image = np.full((2, 2, 3), 50, dtype=np.uint8)
        tumor_mask = np.ones((2, 2), dtype=bool)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(
                image, self.antigen_profile, tumor_mask=tumor_mask
            )
        )

        expected_zones = np.array([4, 0, 0, 0, 0])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        expected_percentage = (expected_zones / np.sum(tumor_mask)) * 100
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        self.assertEqual(pixelcount, np.sum(tumor_mask))

        expected_img_analysis = np.full((2, 2), 50, dtype=np.float32)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_all_positive_with_mask(self):
        image = np.full((2, 2, 3), 100, dtype=np.uint8)
        tumor_mask = np.ones((2, 2), dtype=bool)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(
                image, self.antigen_profile, tumor_mask=tumor_mask
            )
        )

        expected_zones = np.array([0, 4, 0, 0, 0])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        expected_percentage = (expected_zones / np.sum(tumor_mask)) * 100
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        self.assertEqual(pixelcount, np.sum(tumor_mask))

        expected_img_analysis = np.full((2, 2), 100, dtype=np.float32)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_all_low_positive_with_mask(self):
        image = np.full((2, 2, 3), 180, dtype=np.uint8)
        tumor_mask = np.ones((2, 2), dtype=bool)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(
                image, self.antigen_profile, tumor_mask=tumor_mask
            )
        )

        expected_zones = np.array([0, 0, 4, 0, 0])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        expected_percentage = (expected_zones / np.sum(tumor_mask)) * 100
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        self.assertEqual(pixelcount, np.sum(tumor_mask))

        expected_img_analysis = np.full((2, 2), 180, dtype=np.float32)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_all_negative_with_mask(self):
        image = np.full((2, 2, 3), 200, dtype=np.uint8)
        tumor_mask = np.ones((2, 2), dtype=bool)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(
                image, self.antigen_profile, tumor_mask=tumor_mask
            )
        )

        expected_zones = np.array([0, 0, 0, 4, 0])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        expected_percentage = (expected_zones / np.sum(tumor_mask)) * 100
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        self.assertEqual(pixelcount, np.sum(tumor_mask))

        expected_img_analysis = np.full((2, 2), 200, dtype=np.float32)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_empty_image(self):
        # Create an image with all pixels set to zero
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(image, self.antigen_profile)
        )

        # Expected zones
        expected_zones = np.array([4, 0, 0, 0, 0])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        # Expected percentages
        expected_percentage = (expected_zones / to_numpy(pixelcount)) * 100
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        # Check pixel count
        self.assertEqual(to_numpy(pixelcount), 4)

        # Check img_analysis
        expected_img_analysis = np.zeros((2, 2), dtype=np.uint8)
        np.testing.assert_array_equal(
            to_numpy(img_analysis), expected_img_analysis)

    def test_partial_mask_mixed_intensities(self):
        image = np.array(
            [
                [[50, 50, 50], [100, 100, 100], [180, 180, 180]],
                [[200, 200, 200], [240, 240, 240], [255, 255, 255]],
                [[20, 20, 20], [120, 120, 120], [181, 181, 181]],
            ],
            dtype=np.uint8,
        )

        # Mask only the top-left 2x2 area
        tumor_mask = np.array(
            [
                [True, True, False],
                [True, True, False],
                [False, False, False],
            ],
            dtype=bool,
        )

        hist, hist_centers, zones, percentage, score, pixelcount, img_analysis = (
            evaluate_staining_intensities(
                image, self.antigen_profile, tumor_mask=tumor_mask
            )
        )

        # Top-left 2x2 pixels intensities: 50 (high+), 100 (positive), 200 (negative), 240 (background)
        expected_zones = np.array([1, 1, 0, 1, 1])
        np.testing.assert_array_equal(to_numpy(zones), expected_zones)

        # Tissue zones normalized to sum of tissue (1+1+0+1=3)
        # Background zone normalized to mask count (4)
        expected_percentage = np.array(
            [33.333333, 33.333333, 0, 33.333333, 25], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)

        self.assertIsInstance(score, str)

        self.assertEqual(pixelcount, np.sum(tumor_mask))

        img_analysis_np = to_numpy(img_analysis)
        self.assertTrue(np.isnan(img_analysis_np[~tumor_mask]).all())
        expected_masked_values = np.array(
            [[50, 100], [200, 240]], dtype=np.float32)
        self.assertTrue(
            np.array_equal(
                img_analysis_np[tumor_mask].reshape(
                    2, 2), expected_masked_values
            )
        )


class TestCalculateScore(unittest.TestCase):

    def test_edge_case_zeros(self):
        zones = xp.array([0, 0, 0, 0, 0], dtype=xp.float32)

        with self.assertRaises(ValueError):
            calculate_percentage_and_score(zones)

    def test_large_numbers(self):
        zones = xp.array([1000000, 2000000, 3000000,
                         4000000, 0], dtype=xp.float32)
        expected_percentage = np.array(
            [10.0, 20.0, 30.0, 40.0, 0.0], dtype=np.float32)
        # weights: [4,3,2,1], scores=[4000000,6000000,6000000,4000000] → max=6e6 → Positive
        expected_score = "Medium Positive"

        percentage, score = calculate_percentage_and_score(zones)

        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage, decimal=5
        )
        self.assertEqual(score, expected_score)

    def test_dominant_zone(self):
        zones = xp.array([70, 10, 10, 10, 10], dtype=xp.float32)
        expected_percentage = np.array(
            [70.0, 10.0, 10.0, 10.0, 9.0909], dtype=np.float32
        )
        expected_score = "High Positive"  # zone>66.6%

        percentage, score = calculate_percentage_and_score(zones)

        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage, decimal=5
        )
        self.assertEqual(score, expected_score)

    def test_high_positive_score(self):
        zones = xp.array([20, 21, 29, 30, 0])
        expected_percentage = np.array([20.0, 21.0, 29.0, 30.0, 0.0])
        expected_score = "High Positive"  # Based the weights and highest score

        percentage, score = calculate_percentage_and_score(zones)

        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)
        self.assertEqual(score, expected_score)

    def test_positive_score(self):
        zones = xp.array([20, 27, 29, 24, 0])
        expected_percentage = np.array([20.0, 27.0, 29.0, 24.0, 0.0])
        expected_score = "Medium Positive"  # Based on the weights and the highest score

        percentage, score = calculate_percentage_and_score(zones)

        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)
        self.assertEqual(score, expected_score)

    def test_low_positive_score(self):
        zones = xp.array([10, 30, 50, 10, 10])
        expected_percentage = np.array([10.0, 30.0, 50.0, 10.0, 9.090909])
        expected_score = "Low Positive"  # Based on the weights and the highest score

        percentage, score = calculate_percentage_and_score(zones)

        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)
        self.assertEqual(score, expected_score)

    def test_negative_score(self):
        zones = xp.array([10, 19.5, 10, 60.5, 0])
        expected_percentage = np.array([10.0, 19.5, 10.0, 60.5, 0.0])
        expected_score = "Negative"  # Based on the weights and the highest score

        percentage, score = calculate_percentage_and_score(zones)

        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage)
        self.assertEqual(score, expected_score)

    def test_background_score(self):
        zones = xp.array([3, 10, 10, 10, 67])
        expected_percentage = np.array(
            [9.09, 30.30, 30.30, 30.30, 67.0]
        )  # normalized to tissuecount
        expected_score = "Background"  # Based on the weights and the highest score

        percentage, score = calculate_percentage_and_score(zones)

        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage, decimal=2
        )
        self.assertEqual(score, expected_score)

    def test_background_66_percent(self):
        zones = xp.array([10, 20, 10, 10, 66])
        expected_percentage = np.array(
            [20.0, 40.0, 20.0, 20.0, 56.90], dtype=np.float32
        )
        expected_score = (
            "Medium Positive"  # Background is exactly 66%, so the score is calculated
        )

        percentage, score = calculate_percentage_and_score(zones)

        np.testing.assert_array_almost_equal(
            to_numpy(percentage), expected_percentage, decimal=2
        )
        self.assertEqual(score, expected_score)


if __name__ == "__main__":
    unittest.main()
