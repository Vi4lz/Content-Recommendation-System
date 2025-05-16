import logging
import sys
from logging import Logger

def setup_logging() -> Logger:
    """
    Sets up the logging configuration.

    This function configures logging to print log messages to both the console (stdout)
    and a log file ('app.log'). It sets the log level to INFO by default.

    Returns:
        Logger: The configured logger instance.
    """
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log', encoding="utf-8")
        ]
    )
    logger = logging.getLogger(__name__)
    return logger
