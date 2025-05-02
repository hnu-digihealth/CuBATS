# Standard Library
import unittest
from unittest.mock import MagicMock, patch
import os
import platform
# OpenSlide setup for macOS and Windows for now until Main class
if platform.system() == "Darwin":  # macOS
    path = os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')
    path += (':' if path else '') + '/opt/homebrew/lib'
    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = path
elif platform.system() == "Windows":  # Windows
    OPENSLIDE_PATH = r"C:\\Users\\mlser\\openslide-bin-4.0.0.6-windows-x64\\openslide-bin-4.0.0.6-windows-x64\\bin"
    if hasattr(os, "add_dll_directory"):
        with os.add_dll_directory(OPENSLIDE_PATH):
            pass
    else:
        os.environ['PATH'] += f";{OPENSLIDE_PATH}"

# Import OpenSlide
try:
    # Third Party
    import openslide
except ImportError as e:
    raise ImportError(
        "OpenSlide could not be imported. Ensure the library is installed and the path is set correctly."
    ) from e

# CuBATS
from cubats.registration import register, register_with_ref


class TestRegistration(unittest.TestCase):
    def setUp(self):
        # Openslide setup as it is not used in the tests
        _ = openslide
        # Common mocks
        self.mock_registrar = MagicMock()
        self.mock_path_exists = patch(
            "cubats.registration.os.path.exists").start()
        self.mock_listdir = patch("cubats.registration.os.listdir").start()
        self.mock_path_join = patch("cubats.registration.os.path.join").start()
        self.mock_mkdir = patch("pathlib.Path.mkdir").start()
        self.mock_valis = patch("valis.registration.Valis").start()
        self.mock_kill_jvm = patch("valis.registration.kill_jvm").start()

        # Default mock behaviors
        self.mock_valis.return_value = self.mock_registrar
        self.mock_path_join.side_effect = lambda *args: "/".join(args)
        self.mock_mkdir.return_value = None

        # Mock return values for registrar.register()
        self.mock_registrar.register.return_value = (
            MagicMock(), MagicMock(), MagicMock())

    def tearDown(self):
        # Stop all patches
        patch.stopall()

    def mock_paths(self, existing_paths):
        """Helper to mock os.path.exists for specific paths."""
        self.mock_path_exists.side_effect = lambda path: path in existing_paths

    def test_register_with_ref(self):
        # Mock paths and listdir
        self.mock_paths(["/dummy/src", "reference_slide"])
        self.mock_listdir.return_value = ["file1.tif", "file2.tif"]

        # Call the function with microregistration=False
        register_with_ref(
            "/dummy/src", "/dummy/dst", "reference_slide", microregistration=False
        )

        # Assertions for microregistration=False
        self.mock_valis.assert_called_with(
            "/dummy/src", "/dummy/dst", reference_img_f="reference_slide"
        )
        self.mock_registrar.register.assert_called()
        self.mock_registrar.warp_and_save_slides.assert_called_with(
            "/dummy/dst/registered_slides", crop="overlap"
        )
        self.mock_kill_jvm.assert_called()

        # Reset mocks
        self.mock_registrar.reset_mock()
        self.mock_kill_jvm.reset_mock()

        # Call the function with microregistration=True
        register_with_ref(
            "/dummy/src", "/dummy/dst", "reference_slide", microregistration=True
        )

        # Assertions for microregistration=True
        self.mock_registrar.register_micro.assert_called_with(
            max_non_rigid_registration_dim_px=2000, align_to_reference=True
        )
        self.mock_kill_jvm.assert_called()

    def test_register(self):
        # Mock paths and listdir
        self.mock_paths(["/dummy/src"])
        self.mock_listdir.return_value = ["file1.tif", "file2.tif"]

        # Call the function with microregistration=False
        register("/dummy/src", "/dummy/dst", microregistration=False)

        # Assertions for microregistration=False
        self.mock_valis.assert_called_with("/dummy/src", "/dummy/dst")
        self.mock_registrar.register.assert_called()
        self.mock_registrar.warp_and_save_slides.assert_called_with(
            "/dummy/dst/registered_slides", crop="overlap"
        )
        self.mock_kill_jvm.assert_called()

        # Reset mocks
        self.mock_registrar.reset_mock()
        self.mock_kill_jvm.reset_mock()

        # Call the function with microregistration=True
        register("/dummy/src", "/dummy/dst", microregistration=True)

        # Assertions for microregistration=True
        self.mock_registrar.register_micro.assert_called_with(
            max_non_rigid_registartion_dim_px=2000
        )
        self.mock_kill_jvm.assert_called()

    def test_invalid_source_directory(self):
        with self.assertRaises(ValueError) as context:
            register_with_ref(
                123, "/dummy/dst", "reference_slide", microregistration=False
            )
        self.assertEqual(
            str(context.exception), "Invalid or non-existent source directory"
        )

    def test_invalid_destination_directory(self):
        # Mock the source directory to exist
        self.mock_paths(["/dummy/src"])
        with self.assertRaises(ValueError) as context:
            register_with_ref(
                "/dummy/src", 123, "reference_slide", microregistration=False
            )
        self.assertEqual(str(context.exception),
                         "Invalid destination directory")

    def test_invalid_reference_slide(self):
        self.mock_paths(["/dummy/src", "/dummy/dst"])
        with self.assertRaises(ValueError) as context:
            register_with_ref(
                "/dummy/src", "/dummy/dst", 123, microregistration=False
            )
        self.assertEqual(
            str(context.exception), "Invalid or non-existent reference slide"
        )

    def test_invalid_microregistration(self):
        self.mock_paths(["/dummy/src", "/dummy/dst", "reference_slide"])
        with self.assertRaises(ValueError) as context:
            register_with_ref(
                "/dummy/src",
                "/dummy/dst",
                "reference_slide",
                microregistration="not_a_bool",
            )
        self.assertEqual(
            str(context.exception), "microregistration must be a boolean"
        )

    def test_invalid_max_non_rigid_registartion_dim_px(self):
        self.mock_paths(["/dummy/src", "/dummy/dst", "reference_slide"])
        with self.assertRaises(ValueError) as context:
            register_with_ref(
                "/dummy/src",
                "/dummy/dst",
                "reference_slide",
                microregistration=False,
                max_non_rigid_registration_dim_px="not_an_int",
            )
        self.assertEqual(
            str(context.exception),
            "max_non_rigid_registartion_dim_px must be an integer",
        )


if __name__ == "__main__":
    unittest.main()
