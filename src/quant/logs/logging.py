"""
logging configuration for quant package --
Sets a logger which logs the exact script; function; line number; log level; and message. This part is prefixed to every log message and in colors for better visibility.
info  is in black
debug is in blue
warning is in yellow
error is in red
no critical logs yet
"""

import logging

from colorlog import ColoredFormatter

logger = logging.getLogger("quant_logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

logger = logging.getLogger("quant_logger")
logger.propagate = (
    False  # Add this line to prevent log messages from being propagated to the root logger
)
formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
ch.setFormatter(formatter)
logger.addHandler(ch)
# Example usage:
# from quant.logs.logging import logger
# logger.info("This is an info message")
# logger.debug("This is a debug message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")

# if __name__ == "__main__":
#     logger.info("Logger is configured and ready to use.")
#     logger.debug("This is a debug message for testing purposes.")
#     logger.warning("This is a warning message for testing purposes.")
#     logger.error("This is an error message for testing purposes.")
#     logger.critical("This is a critical message for testing purposes.")
