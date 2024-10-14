import os

# Define the log directory outside the src directory
log_dir = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'logs')
print(f"Log directory: {log_dir}")  # Debugging statement
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'cubats.log')
print(f"Log file: {log_file}")  # Debugging statement


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
            "filename": log_file,
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

# logging.config.dictConfig(LOGGING)
