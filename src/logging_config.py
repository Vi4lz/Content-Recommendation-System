import logging
import sys

def setup_logging():
    """
    Sets up logging configuration.
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
