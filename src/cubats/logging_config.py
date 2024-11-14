# import logging.config
# Standard Library
import os

# Determine the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Ensure the logs directory exists
logs_dir = os.path.join(project_root, 'logs')
os.makedirs(logs_dir, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(levelname)s: %(message)s",
        },
        "detailed": {
            "format": "[%(levelname)s] %(asctime)s | %(name)s | L %(lineno)d: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "handlers": {
        "stdout": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "simple",
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "stream": "ext://sys.stderr",
            "formatter": "simple",
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": os.path.join(logs_dir, "cubats.log"),
            "maxBytes": 10485760,
            "backupCount": 3,
        }
    },
    "loggers": {
        "": {
            "handlers": ["stdout", "stderr", "file"],
            "level": "DEBUG",
        },
    },
}
