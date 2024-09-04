# Standard Library
import os
import shutil
import unittest

# CuBATS
from cubats.slide_collection.slide_collection import SlideCollection


class TestSlideCollectionInitialization(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for testing
        self.src_dir = "/Users/moritz.lokal/Desktop/test_cubats/input"
        self.dst_dir = "/Users/moritz.lokal/Desktop/test_cubats/test_output"
        # os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(self.dst_dir, exist_ok=True)

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

        names = ["N_2014_000862_HE.ome_mask", "N_2014_000862_ROR2", "N_2014_000862_B7H3",
                 "N_2014_000862_PDPN", "N_2014_000862_HE", "N_2014_000862_ADAM9", "N_2014_000862_PDPN_NZ1"]
        level_dim = ((1, 1),
                     (1, 2),
                     (2, 3),
                     (3, 5),
                     (5, 10),
                     (9, 19),
                     (17, 37),
                     (33, 73),
                     (66, 146),
                     (132, 291),
                     (264, 581),
                     (528, 1162),
                     (1055, 2323),
                     (2109, 4646),
                     (4217, 9291),
                     (8434, 18581),
                     (16867, 37162),
                     (33734, 74324),
                     (67467, 148648))
        level_count = 19
        tile_count = 12929

        # Verify the SlideCollection properties
        self.assertEqual(slide_collection.collection_name, "Test_Collection")
        self.assertEqual(len(slide_collection.collection_list), 8)
        self.assertTrue(os.path.exists(self.dst_dir))

        self.assertIsNotNone(slide_collection.mask)
        self.assertTrue(slide_collection.mask.is_mask)

        self.assertTrue(slide_collection.reference_slide.is_reference)
        self.assertFalse(slide_collection.reference_slide.is_mask)
        self.assertFalse(slide_collection.mask.is_reference)

        for i in range(len(names)):
            self.assertEqual(
                slide_collection.collection_list[i].name, names[i])
            self.assertEqual(
                slide_collection.collection_list[i].level_dimensions, level_dim)
            self.assertEqual(
                slide_collection.collection_list[i].level_count, level_count)
            self.assertEqual(
                slide_collection.collection_list[i].tile_count, tile_count)


if __name__ == '__main__':
    unittest.main()
