# Standard Library
import os
import shutil
import tempfile
import unittest

# Third Party
import openslide
import openslide.deepzoom

# CuBATS
from cubats.slide_collection.slide_collection import SlideCollection


class TestSlideCollectionInitialization(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.src_dir = os.path.join(self.temp_dir.name, "test_input")
        self.dst_dir = os.path.join(self.temp_dir.name, "test_output")
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(self.dst_dir, exist_ok=True)

        # Path to the actual valid file
        test_file_path = os.path.join(os.path.dirname(
            __file__), 'test_files', 'test_file.tiff')

        # Verify that the test file exists
        assert os.path.exists(
            test_file_path), f"Test file not found: {test_file_path}"

        # Add the test file to the source directory with different filenames
        self.mock_files = [
            "Pat_ID_HE_mask.tiff",
            "Pat_ID_ROR2.tiff",
            "Pat_ID_B7H3.tiff",
            "Pat_ID_PDPN.tiff",
            "Pat_ID_HE.tiff",
            "Pat_ID_ADAM9.tiff",
            "Pat_ID_PDPN_NZ1.tiff"
        ]
        for file_name in self.mock_files:
            shutil.copy(test_file_path, os.path.join(self.src_dir, file_name))

    def tearDown(self):
        # Debug statement to check if tearDown is called
        print(f"Tearing down: {self.dst_dir}")
        try:
            # Remove temporary directories after tests
            shutil.rmtree(self.dst_dir)
            print(f"Successfully removed: {self.dst_dir}")
        except Exception as e:
            print(f"Error removing {self.dst_dir}: {e}")

    def test_slide_collection_initialization(self):
        # Initialize the SlideCollection
        slide_collection = SlideCollection(
            "Test_Collection", self.src_dir, self.dst_dir)

        # Names with the ID part
        names = ["Pat_ID_HE_mask",
                 "Pat_ID_ROR2",
                 "Pat_ID_B7H3",
                 "Pat_ID_PDPN",
                 "Pat_ID_HE",
                 "Pat_ID_ADAM9",
                 "Pat_ID_PDPN_NZ1"]
        names.sort()

        # Load the test file using OpenSlide to verify its properties
        test_file_path = os.path.join(os.path.dirname(
            __file__), 'test_files', 'test_file.tiff')
        test_slide = openslide.OpenSlide(test_file_path)
        DeepZoomGenerator = openslide.deepzoom.DeepZoomGenerator(
            test_slide, tile_size=1024, overlap=0, limit_bounds=True)
        expected_level_dimensions = DeepZoomGenerator.level_dimensions
        expected_level_count = DeepZoomGenerator.level_count
        expected_tile_count = DeepZoomGenerator.tile_count

        # Verify the SlideCollection properties
        self.assertEqual(slide_collection.collection_name, "Test_Collection")
        self.assertEqual(len(slide_collection.collection_list), 7)
        self.assertTrue(os.path.exists(self.dst_dir))

        self.assertIsNotNone(slide_collection.mask)
        self.assertTrue(slide_collection.mask.is_mask)

        self.assertTrue(slide_collection.reference_slide.is_reference)
        self.assertFalse(slide_collection.reference_slide.is_mask)
        self.assertFalse(slide_collection.mask.is_reference)

        # Sort the actual slide names to match the expected order
        actual_names = sorted(
            [slide.name for slide in slide_collection.collection_list])

        for i in range(len(names)):
            slide_name = actual_names[i]
            self.assertEqual(slide_name, names[i])
            # Verify other properties
            self.assertEqual(
                slide_collection.collection_list[i].level_dimensions, expected_level_dimensions)
            self.assertEqual(
                slide_collection.collection_list[i].level_count, expected_level_count)
            self.assertEqual(
                slide_collection.collection_list[i].tile_count, expected_tile_count)


if __name__ == '__main__':
    unittest.main()
