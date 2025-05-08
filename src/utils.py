import joblib
import os


def save_model(model, filename):
    """
    Saves a trained model to a file using joblib.

    Args:
        model (BaseEstimator): Trained model to be saved.
        filename (str): Path to the output file where the model will be stored.

    Returns:
        None
    """
    try:
        joblib.dump(model, filename)
        print(f"[INFO] Model saved to: {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save model to '{filename}': {e}")


def load_model(filename):
    """
    Loads a trained model from a file using joblib.

    Args:
        filename (str): Path to the model file.

    Returns:
        BaseEstimator or None: Loaded model if file exists and is valid; otherwise None.
    """
    if not os.path.exists(filename):
        print(f"[WARNING] Model file '{filename}' not found.")
        return None
    try:
        model = joblib.load(filename)
        print(f"[INFO] Model loaded from: {filename}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model from '{filename}': {e}")
        return None
