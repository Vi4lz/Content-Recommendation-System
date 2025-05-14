import joblib
import os
from logging_config import setup_logging

logger = setup_logging()

def save_model(model, filename):
    """
    Saves a trained model to a file using joblib.

    Returns:
        None
    """
    try:
        joblib.dump(model, filename)
        logger.info(f"Model saved to: {filename}")
    except Exception as e:
        logger.error(f"Failed to save model to '{filename}': {e}")


def load_model(filename):
    """
    Loads a trained model from a file using joblib.

    Returns:
        BaseEstimator or None: Loaded model if file exists and is valid; otherwise None.
    """
    if not os.path.exists(filename):
        logger.warning(f"Model file '{filename}' not found.")
        return None
    try:
        model = joblib.load(filename)
        logger.info(f"Model loaded from: {filename}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from '{filename}': {e}")
        return None
