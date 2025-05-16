import joblib
import os
from logging_config import setup_logging
from sklearn.base import BaseEstimator
from typing import Optional

logger = setup_logging()

def save_model(model: BaseEstimator, filename: str) -> None:
    """
    Saves a trained model to a file using joblib.

    Args:
        model (BaseEstimator): The trained machine learning model to be saved.
        filename (str): The path where the model should be saved.

    Returns:
        None.
    """
    try:
        joblib.dump(model, filename)
        logger.info(f"Model saved to: {filename}")
    except Exception as e:
        logger.error(f"Failed to save model to '{filename}': {e}")


def load_model(filename: str) -> Optional[BaseEstimator]:
    """
    Loads a trained model from a file using joblib.

    Args:
        filename (str): The path to the saved model file.

    Returns:
        Optional[BaseEstimator]: The loaded model if the file exists, otherwise None.
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
