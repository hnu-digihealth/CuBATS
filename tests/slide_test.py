# Standard Library
import os
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import numpy as np
from PIL import Image as PILImage

# CuBATS
from cubats.slide_collection.slide import Slide


class TestSlideQuantify(unittest.TestCase):
    def setUp(self):
        # temporary dirs
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.dst = os.path.join(self.tmp.name, "dst")
        os.makedirs(self.src, exist_ok=True)
        os.makedirs(self.dst, exist_ok=True)

        # test fixture from repo tests/test_files
        self.test_file = os.path.join(
            os.path.dirname(__file__), "test_files", "test_file.tiff"
        )
        assert os.path.exists(
            self.test_file), "missing test fixture test_file.tiff"

        # copy fixture to src for clarity (Slide uses file path)
        self.slide_path = os.path.join(self.src, "Pat_Test.tiff")
        shutil.copy(self.test_file, self.slide_path)

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except Exception:
            pass

    def test_quantify_slide_raises_for_mask_and_reference(self):
        # mask slide should raise
        mask_slide = Slide("MaskSlide", self.slide_path, is_mask=True)
        with self.assertRaises(ValueError):
            mask_slide.quantify_slide([(0, 0)], save_dir=self.dst)

        # reference slide should raise
        ref_slide = Slide("RefSlide", self.slide_path, is_reference=True)
        with self.assertRaises(ValueError):
            ref_slide.quantify_slide([(0, 0)], save_dir=self.dst)

    def test_quantify_slide_requires_img_dir_when_save_img_true(self):
        s = Slide("S_requires_img_dir", self.slide_path)
        with self.assertRaises(ValueError):
            s.quantify_slide([(0, 0)], save_dir=self.dst,
                             save_img=True, img_dir=None)

    def test_quantify_slide_processes_and_saves_pickle_tile_level(self):
        s = Slide("S_process", self.slide_path)
        # Build a fake tile-result expected by summarize_quantification_results
        fake_tile_result = {
            "Flag": 1,
            "Zones": [100, 50, 25, 25, 0],  # 5 zones
            "Mask Count": 1024,  # > 0 to avoid division by zero
        }

        # Patch mask_tile to be a simple passthrough (it will be invoked when building iterable if mask provided)
        with patch("cubats.slide_collection.slide.mask_tile", side_effect=lambda t, m: (t, m)):
            # Patch ProcessPoolExecutor so map returns our fake_tile_result iterator
            with patch("concurrent.futures.ProcessPoolExecutor") as MockExec:
                mock_executor = MockExec.return_value.__enter__.return_value
                mock_executor.map.return_value = iter([fake_tile_result])

                # run quantify_slide for one tile coordinate
                coords = [(0, 0)]
                s.quantify_slide(coords, save_dir=self.dst, save_img=False)

        # assert pickle file exists and contains the same dict
        out_pickle = os.path.join(self.dst, f"{s.name}_processing_info.pickle")
        self.assertTrue(os.path.exists(out_pickle))
        with open(out_pickle, "rb") as fh:
            loaded = pickle.load(fh)
        # the saved pickle should be a dict mapping indices to results and match slide.detailed_quantification_results
        self.assertEqual(loaded, s.detailed_quantification_results)
        self.assertIn("Name", s.quantification_summary)
        self.assertEqual(s.quantification_summary["Name"], s.name)

    def test_quantify_slide_with_save_img_creates_img_dir(self):
        s = Slide("S_with_img", self.slide_path)
        fake_tile_result = {
            "Flag": 1,
            "Zones": [10, 10, 10, 10, 0],
            "Mask Count": 512,
        }
        img_dir = os.path.join(self.dst, "tiles_out")
        with patch("cubats.slide_collection.slide.mask_tile", side_effect=lambda t, m: (t, m)):
            with patch("concurrent.futures.ProcessPoolExecutor") as MockExec:
                mock_executor = MockExec.return_value.__enter__.return_value
                mock_executor.map.return_value = iter([fake_tile_result])

                s.quantify_slide([(0, 0)], save_dir=self.dst,
                                 save_img=True, img_dir=img_dir)

        # dab_tile_dir should be set and directory exists
        self.assertEqual(s.dab_tile_dir, img_dir)
        self.assertTrue(os.path.isdir(img_dir))
        # pickle exists as well
        out_pickle = os.path.join(self.dst, f"{s.name}_processing_info.pickle")
        self.assertTrue(os.path.exists(out_pickle))


class TestSlideReconstruct(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.out = os.path.join(self.tmp.name, "out")
        os.makedirs(self.src, exist_ok=True)
        os.makedirs(self.out, exist_ok=True)

        # small valid test fixture used elsewhere
        self.test_file = os.path.join(
            os.path.dirname(__file__), "test_files", "test_file.tiff"
        )
        assert os.path.exists(
            self.test_file), "missing test fixture test_file.tiff"

        # copy fixture and construct a real Slide (openslide will read the fixture)
        self.slide_path = os.path.join(self.src, "Pat_Test.tiff")
        shutil.copy(self.test_file, self.slide_path)
        self.slide = Slide("Pat_Test", self.slide_path)

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except Exception:
            pass

    def test_reconstruct_slide_raises_when_in_path_missing(self):
        missing = os.path.join(self.tmp.name, "no_such_dir")
        with self.assertRaises(ValueError):
            self.slide.reconstruct_slide(missing, self.out)

    def test_reconstruct_slide_raises_when_no_tif_files(self):
        empty_dir = os.path.join(self.tmp.name, "empty_tiles")
        os.makedirs(empty_dir, exist_ok=True)
        with self.assertRaises(ValueError):
            self.slide.reconstruct_slide(empty_dir, self.out)

    def test_reconstruct_slide_success_calls_vips_and_saves(self):
        # Prepare tiles directory with one tile (0_0.tif). leave other tiles missing to exercise fallback.
        tiles_dir = os.path.join(self.tmp.name, "tiles")
        os.makedirs(tiles_dir, exist_ok=True)

        # Save one small tif tile file that reconstruct_slide will pick up
        tile_path = os.path.join(tiles_dir, "0_0.tif")
        PILImage.new("RGB", (8, 8), (10, 20, 30)).save(tile_path)

        # Patch np.concatenate to avoid operating on PIL images and to return deterministic arrays
        # First call (per-row concatenation) -> small array, second call (rows -> whole) -> small array
        concat_side_effects = [
            np.zeros((8, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
        ]

        # Patch VipsImage.new_from_array chain so we don't require pyvips to write real TIFF
        with patch("cubats.slide_collection.slide.np.concatenate") as mock_concat, patch(
            "cubats.slide_collection.slide.VipsImage.new_from_array"
        ) as mock_new_from_array:
            mock_concat.side_effect = concat_side_effects
            # build a chainable mock: new_from_array().cast().crop().tiffsave()
            mock_vips = MagicMock()
            mock_cast = MagicMock()
            mock_crop = MagicMock()
            mock_tiffsave = MagicMock()
            mock_crop.tiffsave = mock_tiffsave
            mock_cast.crop.return_value = mock_crop
            mock_vips.cast.return_value = mock_cast
            mock_new_from_array.return_value = mock_vips

            out_dir = os.path.join(self.out, "reconst_out")
            # call reconstruct_slide - should not raise
            self.slide.reconstruct_slide(tiles_dir, out_dir)

            # assertions: vips pipeline invoked and tiffsave called once
            mock_new_from_array.assert_called()
            mock_vips.cast.assert_called()
            mock_cast.crop.assert_called_once()
            mock_tiffsave.assert_called_once()

            # file may not actually be written because tiffsave is mocked, but out dir should exist
            self.assertTrue(os.path.isdir(out_dir))


if __name__ == "__main__":
    unittest.main()
