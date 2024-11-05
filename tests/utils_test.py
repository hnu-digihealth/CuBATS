# Standard Library
import unittest
from unittest.mock import MagicMock

# Third Party
from PIL import Image

# CuBATS
import cubats.Utils as Utils


class TestUtils(unittest.TestCase):

    def test_get_name(self):
        self.assertEqual(Utils.get_name("path/to/image.ome.tiff"), "image")
        self.assertEqual(Utils.get_name("path/to/image.ome.tif"), "image")
        self.assertEqual(Utils.get_name("path/to/image.jpg"), "image")
        self.assertEqual(Utils.get_name(
            "path/to/image_underscore.ome.tif"), "image_underscore")
        with self.assertRaises(ValueError):
            Utils.get_name(123)

        with self.assertRaises(ValueError):
            Utils.get_name("")

    def test_get_score_name(self):
        self.assertEqual(Utils.get_score_name([0, 1, 2, 3]), "Negative")
        self.assertEqual(Utils.get_score_name([3, 2, 1, 0]), "High Positive")
        self.assertEqual(Utils.get_score_name([1, 3, 2, 0]), "Positive")
        self.assertEqual(Utils.get_score_name([2, 1, 3, 0]), "Low Positive")

        # Score list usually contains 5 elements but 5th is irrelevant for score name
        self.assertEqual(Utils.get_score_name([0, 1, 2, 3, 5]), "Negative")
        self.assertEqual(Utils.get_score_name(
            [3, 2, 1, 0, 5]), "High Positive")
        self.assertEqual(Utils.get_score_name([1, 3, 2, 0, 5]), "Positive")
        self.assertEqual(Utils.get_score_name([2, 1, 3, 0, 5]), "Low Positive")

        with self.assertRaises(ValueError):
            Utils.get_score_name("not a list")

        with self.assertRaises(ValueError):
            Utils.get_score_name([])

        with self.assertRaises(ValueError):
            Utils.get_score_name([1, 2, "three", 4])
        with self.assertRaises(ValueError):
            Utils.get_score_name([1, 2, 3])

        with self.assertRaises(ValueError):
            Utils.get_score_name([1, -2, 3, 4])

        with self.assertRaises(ValueError):
            Utils.get_score_name([-1, -2, -3, -4, -5])

    def test_downsample_Openslide_to_PIL(self):
        # Mock the Openslide object
        mock_openslide = MagicMock()
        mock_openslide.dimensions = (10000, 10000)
        mock_openslide.get_best_level_for_downsample.return_value = 0
        mock_openslide.level_dimensions = [(10000, 10000)]
        mock_image = Image.new('RGB', (10000, 10000))
        mock_openslide.read_region.return_value = mock_image

        # Test downsampling with a factor of 10
        img, old_w, old_h, new_w, new_h = Utils.downsample_Openslide_to_PIL(
            mock_openslide, 10)

        self.assertEqual(old_w, 10000)
        self.assertEqual(old_h, 10000)
        self.assertEqual(new_w, 1000)
        self.assertEqual(new_h, 1000)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (1000, 1000))

        # Test downsampling with a factor of 1 (no downsampling)
        img, old_w, old_h, new_w, new_h = Utils.downsample_Openslide_to_PIL(
            mock_openslide, 1)

        self.assertEqual(old_w, 10000)
        self.assertEqual(old_h, 10000)
        self.assertEqual(new_w, 10000)
        self.assertEqual(new_h, 10000)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (10000, 10000))

        # Test downsampling with a factor larger than the image dimensions
        img, old_w, old_h, new_w, new_h = Utils.downsample_Openslide_to_PIL(
            mock_openslide, 10000)

        self.assertEqual(old_w, 10000)
        self.assertEqual(old_h, 10000)
        self.assertEqual(new_w, 1)
        self.assertEqual(new_h, 1)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (1, 1))

        # Test invalid downsampling factor (zero or negative)
        with self.assertRaises(ValueError):
            Utils.downsample_Openslide_to_PIL(mock_openslide, 0)

        with self.assertRaises(ValueError):
            Utils.downsample_Openslide_to_PIL(mock_openslide, -1)

        # Test invalid Openslide object
        with self.assertRaises(ValueError):
            Utils.downsample_Openslide_to_PIL(None, 10)


if __name__ == '__main__':
    unittest.main()
