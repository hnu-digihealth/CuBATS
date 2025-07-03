# Standard Library
import tempfile
import unittest
from unittest.mock import patch

# Third Party
import numpy as np

# CuBATS
from cubats.slide_collection.colocalization import \
    analyze_dual_antigen_colocalization


class TestComputeDualAntigenColocalization(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.test_dir = tempfile.TemporaryDirectory()
        self.antigen_profile = {
            "Name": "default",
            "low_positive_threshold": 181,
            "medium_positive_threshold": 121,
            "high_positive_threshold": 61,
        }
        self.antigen_profiles = [
            self.antigen_profile,
            self.antigen_profile,
        ]

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    def test_both_images_contain_tissue(self):
        img1 = {"Tilename": "tile1", "Flag": 1, "Image Array": np.ones((1024, 1024))}
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": np.ones((1024, 1024))}
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertIn("Total Coverage", result)
        self.assertIn("Total Overlap", result)
        self.assertIn("Total Complement", result)
        self.assertIn("High Positive Overlap", result)
        self.assertIn("High Positive Complement", result)
        self.assertIn("Medium Positive Overlap", result)
        self.assertIn("Medium Positive Complement", result)
        self.assertIn("Low Positive Overlap", result)
        self.assertIn("Low Positive Complement", result)
        self.assertIn("Negative", result)
        self.assertIn("Background / No Tissue", result)

    def test_both_images_do_not_contain_tissue(self):
        img1 = {"Tilename": "tile1", "Flag": 0, "Image Array": np.ones((1024, 1024))}
        img2 = {"Tilename": "tile2", "Flag": 0, "Image Array": np.ones((1024, 1024))}
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], -1)

    def test_images_have_different_shapes(self):
        img1 = {"Tilename": "tile1", "Flag": 1, "Image Array": np.ones((1024, 1024))}
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": np.ones((512, 512))}
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], -2)

    @patch("PIL.Image.Image.save")
    def test_image_saving(self, mock_save):
        img1 = {"Tilename": "tile1", "Flag": 1, "Image Array": np.ones((1024, 1024))}
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": np.ones((1024, 1024))}
        output_path = self.test_dir.name
        save_img = True

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        _ = result
        mock_save.assert_called_once()

    def test_one_image_does_not_contain_tissue_1(self):
        img1 = {"Tilename": "tile1", "Flag": 0}
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": np.ones((1024, 1024))}
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 0.0)
        self.assertAlmostEqual(result["Total Complement"], 100.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 100.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_one_image_does_not_contain_tissue_2(self):
        img1_array = np.full((1024, 1024), 255)
        # Set pixel values for img1 with skewed distribution
        img1_array[0:512, :] = 50  # High positive
        img1_array[512:768, :] = 100  # Positive
        img1_array[768:896, :] = 150  # Low positive
        img1_array[896:1024, :] = 200  # Negative

        img1 = {"Tilename": "tile1", "Flag": 1, "Image Array": img1_array}
        img2 = {"Tilename": "tile2", "Flag": 0}
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 87.5)
        self.assertAlmostEqual(result["Total Overlap"], 0.0)
        self.assertAlmostEqual(result["Total Complement"], 87.5)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 50.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 25.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 12.5)
        self.assertAlmostEqual(result["Negative"], 12.5)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_negative_tissue(self):
        img1 = {
            "Tilename": "tile1",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 234),
        }
        img2 = {
            "Tilename": "tile2",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 200),
        }
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 0.0)
        self.assertAlmostEqual(result["Total Overlap"], 0.0)
        self.assertAlmostEqual(result["Total Complement"], 0.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 100.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_high_positive_overlap(self):
        img1 = {
            "Tilename": "tile1",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 60),
        }
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": np.full((1024, 1024), 0)}
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 100.0)
        self.assertAlmostEqual(result["Total Complement"], 0.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 100.0)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_positive_overlap(self):
        img1 = {
            "Tilename": "tile1",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 81),
        }
        img2 = {
            "Tilename": "tile2",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 120),
        }
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 100.0)
        self.assertAlmostEqual(result["Total Complement"], 0.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 100.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_low_positive_overlap_1(self):
        img1 = {
            "Tilename": "tile1",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 121),
        }
        img2 = {
            "Tilename": "tile2",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 180),
        }
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 100.0)
        self.assertAlmostEqual(result["Total Complement"], 0.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 100.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_low_positive_overlap_2(self):
        img1 = {
            "Tilename": "tile1",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 121),
        }
        img2 = {
            "Tilename": "tile2",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 70),
        }
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 100.0)
        self.assertAlmostEqual(result["Total Complement"], 0.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 100.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_high_positive_complement_1(self):
        img1 = {
            "Tilename": "tile1",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 60),
        }
        img2 = {
            "Tilename": "tile2",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 200),
        }
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 0.0)
        self.assertAlmostEqual(result["Total Complement"], 100.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 100.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_high_positive_complement_2(self):
        img1 = {
            "Tilename": "tile1",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 60),
        }
        img2 = {
            "Tilename": "tile2",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 255),
        }
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 0.0)
        self.assertAlmostEqual(result["Total Complement"], 100.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 100.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_positive_complement_1(self):
        img1 = {
            "Tilename": "tile1",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 100),
        }
        img2 = {
            "Tilename": "tile2",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 181),
        }
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 0.0)
        self.assertAlmostEqual(result["Total Complement"], 100.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 100.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_positive_complement_2(self):
        img1 = {
            "Tilename": "tile1",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 100),
        }
        img2 = {
            "Tilename": "tile2",
            "Flag": 1,
            "Image Array": np.full((1024, 1024), 235),
        }
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 0.0)
        self.assertAlmostEqual(result["Total Complement"], 100.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 100.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_equal_distribution(self):
        img1_array = np.full((1024, 1024), 255)
        img2_array = np.full((1024, 1024), 255)

        # Calculate the number of pixels for each band (1/4 of the total pixels)
        total_pixels = 1024 * 1024
        pixels_per_band = total_pixels // 4

        # Set pixel values for img1 with equal distribution
        img1_array.flat[0:pixels_per_band] = 50  # High positive overlap
        img1_array.flat[pixels_per_band : 2 * pixels_per_band] = 100  # Positive overlap
        img1_array.flat[2 * pixels_per_band : 3 * pixels_per_band] = (
            150  # Low positive overlap
        )
        img1_array.flat[3 * pixels_per_band : 4 * pixels_per_band] = 181  # Negative

        # Reshape the array to its original shape
        img1_array = img1_array.reshape((1024, 1024))

        # Set pixel values for img1 with equal distribution
        img2_array.flat[0:pixels_per_band] = 50  # High positive overlap
        img2_array.flat[pixels_per_band : 2 * pixels_per_band] = 100  # Positive overlap
        img2_array.flat[2 * pixels_per_band : 3 * pixels_per_band] = (
            150  # Low positive overlap
        )
        img2_array.flat[3 * pixels_per_band : 4 * pixels_per_band] = 181  # Negative

        # Reshape the array to its original shape
        img2_array = img2_array.reshape((1024, 1024))

        img1 = {"Tilename": "tile1", "Flag": 1, "Image Array": img1_array}
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": img2_array}
        output_path = self.test_dir.name
        save_img = True

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 75.0)
        self.assertAlmostEqual(result["Total Overlap"], 75.0)
        self.assertAlmostEqual(result["Total Complement"], 0.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 25.0)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 25.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 25.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 25.0)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test_skewed_distribution(self):
        img1_array = np.zeros((1024, 1024))
        img2_array = np.zeros((1024, 1024))

        # Set pixel values for img1 with skewed distribution
        img1_array[0:512, :] = 50  # High positive overlap
        img1_array[512:768, :] = 100  # Positive overlap
        img1_array[768:896, :] = 150  # Low positive overlap
        img1_array[896:1024, :] = 200  # Negative

        # Set pixel values for img2 with skewed distribution
        img2_array[0:256, :] = 50  # High positive overlap
        img2_array[256:512, :] = 100  # Positive overlap
        img2_array[512:768, :] = 150  # Low positive overlap
        img2_array[768:1024, :] = 200  # Negative

        img1 = {"Tilename": "tile1", "Flag": 1, "Image Array": img1_array}
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": img2_array}
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 87.5)
        self.assertAlmostEqual(result["Total Overlap"], 75.0)
        self.assertAlmostEqual(result["Total Complement"], 12.5)
        self.assertAlmostEqual(result["High Positive Overlap"], 25.0)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 25.0)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 25.0)
        self.assertAlmostEqual(result["Low Positive Complement"], 12.5)
        self.assertAlmostEqual(result["Negative"], 12.5)
        self.assertAlmostEqual(result["Tissue"], 100.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 0.0)

    def test__equal_distribution_save_img_true_vs_false(self):
        img1_array = np.full((1024, 1024), 255)
        img2_array = np.full((1024, 1024), 255)

        # Calculate the number of pixels for each band (1/4 of the total pixels)
        total_pixels = 1024 * 1024
        pixels_per_band = total_pixels // 4

        # Set pixel values for img1 with equal distribution
        img1_array.flat[0:pixels_per_band] = 50  # High positive overlap
        img1_array.flat[pixels_per_band : 2 * pixels_per_band] = 100  # Positive overlap
        img1_array.flat[2 * pixels_per_band : 3 * pixels_per_band] = (
            150  # Low positive overlap
        )
        img1_array.flat[3 * pixels_per_band : 4 * pixels_per_band] = 181  # Negative

        # Reshape the array to its original shape
        img1_array = img1_array.reshape((1024, 1024))
        # Set pixel values for img1 with equal distribution
        img2_array.flat[0:pixels_per_band] = 50  # High positive overlap
        img2_array.flat[pixels_per_band : 2 * pixels_per_band] = 100  # Positive overlap
        img2_array.flat[2 * pixels_per_band : 3 * pixels_per_band] = (
            150  # Low positive overlap
        )
        img2_array.flat[3 * pixels_per_band : 4 * pixels_per_band] = 181  # Negative

        # Reshape the array to its original shape
        img2_array = img2_array.reshape((1024, 1024))

        img1 = {"Tilename": "tile1", "Flag": 1, "Image Array": img1_array}
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": img2_array}
        output_path = self.test_dir.name

        # Run with save_img = True
        save_img_true_result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, True]
        )

        # Run with save_img = False
        save_img_false_result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, False]
        )

        # Check if the results are the same
        self.assertEqual(save_img_true_result, save_img_false_result)

    def test_skewed_distribution_save_img_true_vs_false(self):
        img1_array = np.zeros((1024, 1024))
        img2_array = np.zeros((1024, 1024))

        # Set pixel values for img1 with skewed distribution
        img1_array[0:512, :] = 50  # High positive overlap
        img1_array[512:768, :] = 100  # Positive overlap
        img1_array[768:896, :] = 150  # Low positive overlap
        img1_array[896:1024, :] = 200  # Negative

        # Set pixel values for img2 with skewed distribution
        img2_array[0:256, :] = 50  # High positive overlap
        img2_array[256:512, :] = 100  # Positive overlap
        img2_array[512:768, :] = 150  # Low positive overlap
        img2_array[768:1024, :] = 200  # Negative

        img1 = {"Tilename": "tile1", "Flag": 1, "Image Array": img1_array}
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": img2_array}
        output_path = self.test_dir.name

        # Run with save_img = True
        save_img_true_result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, True]
        )

        # Run with save_img = False
        save_img_false_result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, False]
        )

        # Check if the results are the same
        self.assertEqual(save_img_true_result, save_img_false_result)

    def test_equal_distribution_with_background(self):
        img1_array = np.full((1024, 1024), 255)
        img2_array = np.full((1024, 1024), 255)

        # Calculate the number of pixels for each band (1/4 of the total pixels)
        total_pixels = 1024 * 1024
        pixels_per_band = total_pixels // 4

        # Set pixel values for img1 with equal distribution and background
        img1_array.flat[0:pixels_per_band] = 240  # Background
        # High positive overlap
        img1_array.flat[pixels_per_band : 2 * pixels_per_band] = 50
        img1_array.flat[2 * pixels_per_band : 3 * pixels_per_band] = (
            100  # Positive overlap
        )
        img1_array.flat[3 * pixels_per_band : 4 * pixels_per_band] = (
            150  # Low positive overlap
        )

        # Reshape the array to its original shape
        img1_array = img1_array.reshape((1024, 1024))

        # Set pixel values for img1 with equal distribution and background
        img2_array.flat[0:pixels_per_band] = 240  # Background
        # High positive overlap
        img2_array.flat[pixels_per_band : 2 * pixels_per_band] = 50
        img2_array.flat[2 * pixels_per_band : 3 * pixels_per_band] = (
            100  # Positive overlap
        )
        img2_array.flat[3 * pixels_per_band : 4 * pixels_per_band] = (
            150  # Low positive overlap
        )

        # Reshape the array to its original shape
        img2_array = img2_array.reshape((1024, 1024))

        img1 = {"Tilename": "tile1", "Flag": 1, "Image Array": img1_array}
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": img2_array}
        output_path = self.test_dir.name
        save_img = True

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 100.0)
        self.assertAlmostEqual(result["Total Complement"], 0.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 33.3333, places=2)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 33.3333, places=2)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 33.3333, places=2)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 75.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 25.0)

    def test_skewed_distribution_with_background(self):
        img1_array = np.zeros((1024, 1024))
        img2_array = np.zeros((1024, 1024))

        # Set pixel values for img1 with skewed distribution and background
        img1_array[0:256, :] = 240  # Background
        img1_array[256:512, :] = 50  # High positive overlap
        img1_array[512:768, :] = 100  # Positive overlap
        img1_array[768:1024, :] = 150  # Low positive overlap

        # Set pixel values for img2 with different skewed distribution and background
        img2_array[0:256, :] = 240  # Background
        img2_array[256:512, :] = 150  # Low positive overlap
        img2_array[512:768, :] = 100  # Positive overlap
        img2_array[768:1024, :] = 50  # High positive overlap

        img1 = {"Tilename": "tile1", "Flag": 1, "Image Array": img1_array}
        img2 = {"Tilename": "tile2", "Flag": 1, "Image Array": img2_array}
        output_path = self.test_dir.name
        save_img = False

        result = analyze_dual_antigen_colocalization(
            [img1, img2, self.antigen_profiles, output_path, save_img]
        )
        self.assertEqual(result["Flag"], 1)
        self.assertAlmostEqual(result["Total Coverage"], 100.0)
        self.assertAlmostEqual(result["Total Overlap"], 100.0)
        self.assertAlmostEqual(result["Total Complement"], 0.0)
        self.assertAlmostEqual(result["High Positive Overlap"], 0.0)
        self.assertAlmostEqual(result["High Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Medium Positive Overlap"], 33.3333, places=2)
        self.assertAlmostEqual(result["Medium Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Low Positive Overlap"], 66.6667, places=2)
        self.assertAlmostEqual(result["Low Positive Complement"], 0.0)
        self.assertAlmostEqual(result["Negative"], 0.0)
        self.assertAlmostEqual(result["Tissue"], 75.0)
        self.assertAlmostEqual(result["Background / No Tissue"], 25.0)


if __name__ == "__main__":
    unittest.main()
