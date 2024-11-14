# Standard Library
import logging.config
import os
import unittest
from io import StringIO

# CuBATS
from cubats import logging_config


class TestLoggingConfiguration(unittest.TestCase):
    def setUp(self):
        # Ensure the logs directory exists
        logs_dir = os.path.join(os.path.dirname(
            logging_config.__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Initialize logging configuration
        logging.config.dictConfig(logging_config.LOGGING)
        self.logger = logging.getLogger(__name__)

    def test_logging_to_stdout(self):
        # Create a StringIO object to capture the logging output
        log_capture_string = StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(ch)

        # Log an INFO level message
        self.logger.info("This is an info message")

        # Verify stdout
        log_capture_string.seek(0)
        stdout_content = log_capture_string.read()
        self.assertIn("INFO: This is an info message", stdout_content)

        # Remove the custom handler
        self.logger.removeHandler(ch)

    def test_logging_to_stderr(self):
        # Create a StringIO object to capture the logging output
        log_capture_string = StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(ch)

        # Log a WARNING level message
        self.logger.warning("This is a warning message")

        # Verify stderr
        log_capture_string.seek(0)
        stderr_content = log_capture_string.read()
        self.assertIn("WARNING: This is a warning message", stderr_content)

        # Remove the custom handler
        self.logger.removeHandler(ch)

    def test_logging_to_file(self):
        # Log a DEBUG level message
        self.logger.debug("This is a debug message")

        # Verify file log
        log_file_path = os.path.join(os.path.dirname(
            logging_config.__file__), 'logs', 'cubats.log')
        with open(log_file_path, 'r') as log_file:
            log_content = log_file.read()
            self.assertIn("[DEBUG]", log_content)
            self.assertIn("This is a debug message", log_content)


if __name__ == '__main__':
    unittest.main()
