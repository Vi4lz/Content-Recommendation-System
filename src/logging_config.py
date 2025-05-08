import logging

def setup_logging():
    """
    Sets up logging configuration.
    """
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    logger = logging.getLogger(__name__)
    return logger
