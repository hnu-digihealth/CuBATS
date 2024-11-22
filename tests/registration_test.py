# Standard Library
import unittest
from unittest.mock import MagicMock, patch

# CuBATS
from cubats.registration import register, register_with_ref


class TestRegistration(unittest.TestCase):

    @patch('valis.registration.kill_jvm')
    @patch('valis.registration.Valis')
    @patch('cubats.registration.os.path.join')
    @patch('cubats.registration.os.listdir')
    @patch('cubats.registration.os.path.exists')
    @patch('pathlib.Path.mkdir')
    def test_register_with_ref(self, mock_mkdir, mock_path_exists, mock_listdir, mock_path_join,
                               mock_valis, mock_kill_jvm):
        # Mock the os.path.exists to return True for specific paths
        mock_path_exists.side_effect = lambda path: path in [
            '/dummy/src', 'reference_slide']

        # Mock the os.listdir to return a dummy list of files
        mock_listdir.return_value = ['file1.tif', 'file2.tif']

        # Mock the os.path.join to return a dummy path
        mock_path_join.side_effect = lambda *args: '/'.join(args)

        # Mock the mkdir method to do nothing
        mock_mkdir.return_value = None

        # Mock the Valis class and its methods
        mock_registrar = MagicMock()
        mock_valis.return_value = mock_registrar

        # Mock the return value of the register method
        mock_registrar.register.return_value = (
            MagicMock(), MagicMock(), MagicMock())

        # Call the function with microregistration=False
        register_with_ref('/dummy/src', '/dummy/dst',
                          'reference_slide', microregistration=False)

        # Check if the Valis class was instantiated correctly
        mock_valis.assert_called_with(
            '/dummy/src', '/dummy/dst', reference_img_f='reference_slide')

        # Check if the register method was called
        mock_registrar.register.assert_called()

        # Check if the warp_and_save_slides method was called with the correct arguments
        mock_registrar.warp_and_save_slides.assert_called_with(
            '/dummy/dst/registered_slides', crop='overlap')

        # Check if the kill_jvm method was called
        mock_kill_jvm.assert_called()

        # Reset mock calls
        mock_registrar.reset_mock()
        mock_kill_jvm.reset_mock()

        # Call the function with microregistration=True
        register_with_ref('/dummy/src', '/dummy/dst',
                          'reference_slide', microregistration=True)

        # Check if the register_micro method was called with the correct arguments
        mock_registrar.register_micro.assert_called_with(
            max_non_rigid_registartion_dim_px=3000, align_to_reference=True)

        # Check if the kill_jvm method was called again
        mock_kill_jvm.assert_called()

    def test_invalid_source_directory(self):
        with self.assertRaises(ValueError) as context:
            register_with_ref(123, '/dummy/dst',
                              'reference_slide', microregistration=False)
        self.assertEqual(str(context.exception),
                         "Invalid or non-existent source directory")

    def test_invalid_destination_directory(self):
        with patch('cubats.registration.os.path.exists') as mock_exists:
            # Mock the source directory to exist
            mock_exists.side_effect = lambda path: path == '/dummy/src'

            with self.assertRaises(ValueError) as context:
                register_with_ref('/dummy/src', 123,
                                  'reference_slide', microregistration=False)
            self.assertEqual(str(context.exception),
                             "Invalid destination directory")

    def test_invalid_reference_slide(self):
        with patch('cubats.registration.os.path.exists') as mock_exists:
            # Mock the source and destination directories to exist
            mock_exists.side_effect = lambda path: path in [
                '/dummy/src', '/dummy/dst']

            with self.assertRaises(ValueError) as context:
                register_with_ref('/dummy/src', '/dummy/dst',
                                  123, microregistration=False)
            self.assertEqual(str(context.exception),
                             "Invalid or non-existent reference slide")

    def test_invalid_microregistration(self):
        with patch('cubats.registration.os.path.exists') as mock_exists:
            # Mock the source, destination directories, and reference slide to exist
            mock_exists.side_effect = lambda path: path in [
                '/dummy/src', '/dummy/dst', 'reference_slide']

            with self.assertRaises(ValueError) as context:
                register_with_ref('/dummy/src', '/dummy/dst',
                                  'reference_slide', microregistration='not_a_bool')
            self.assertEqual(str(context.exception),
                             "microregistration must be a boolean")

    def test_invalid_max_non_rigid_registartion_dim_px(self):
        with patch('cubats.registration.os.path.exists') as mock_exists:
            # Mock the source, destination directories, and reference slide to exist
            mock_exists.side_effect = lambda path: path in [
                '/dummy/src', '/dummy/dst', 'reference_slide']

            with self.assertRaises(ValueError) as context:
                register_with_ref('/dummy/src', '/dummy/dst', 'reference_slide',
                                  microregistration=False, max_non_rigid_registartion_dim_px='not_an_int')
            self.assertEqual(
                str(context.exception), "max_non_rigid_registartion_dim_px must be an integer")

    @patch('valis.registration.kill_jvm')
    @patch('valis.registration.Valis')
    @patch('cubats.registration.os.path.join')
    @patch('cubats.registration.os.listdir')
    @patch('cubats.registration.os.path.exists')
    @patch('pathlib.Path.mkdir')
    def test_register(self, mock_mkdir, mock_path_exists, mock_listdir, mock_path_join, mock_valis, mock_kill_jvm):
        # Mock the os.path.exists to return True for specific paths
        mock_path_exists.side_effect = lambda path: path in ['/dummy/src']

        # Mock the os.listdir to return a dummy list of files
        mock_listdir.return_value = ['file1.tif', 'file2.tif']

        # Mock the os.path.join to return a dummy path
        mock_path_join.side_effect = lambda *args: '/'.join(args)

        # Mock the mkdir method to do nothing
        mock_mkdir.return_value = None

        # Mock the Valis class and its methods
        mock_registrar = MagicMock()
        mock_valis.return_value = mock_registrar

        # Mock the return value of the register method to return a tuple with three values
        mock_registrar.register.return_value = (
            MagicMock(), MagicMock(), MagicMock())

        # Call the function with microregistration=False
        register('/dummy/src', '/dummy/dst', microregistration=False)

        # Check if the Valis class was instantiated correctly
        mock_valis.assert_called_with('/dummy/src', '/dummy/dst')

        # Check if the register method was called
        mock_registrar.register.assert_called()

        # Check if the warp_and_save_slides method was called with the correct arguments
        mock_registrar.warp_and_save_slides.assert_called_with(
            '/dummy/dst/registered_slides', crop='overlap')

        # Check if the kill_jvm method was called
        mock_kill_jvm.assert_called()

        # Reset mock calls
        mock_registrar.reset_mock()
        mock_kill_jvm.reset_mock()

        # Call the function with microregistration=True
        register('/dummy/src', '/dummy/dst', microregistration=True)

        # Check if the register_micro method was called with the correct arguments
        mock_registrar.register_micro.assert_called_with(
            max_non_rigid_registartion_dim_px=3000)

        # Check if the kill_jvm method was called again
        mock_kill_jvm.assert_called()


if __name__ == '__main__':
    unittest.main()
