import os
import joblib

def save_joblib(obj, filename):
    """Save object to a file using joblib."""
    try:
        joblib.dump(obj, filename)
        print(f"[INFO] Joblib file saved: {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save file {filename}: {e}")

def load_joblib(filename):
    """Load object using joblib."""
    if not os.path.exists(filename):
        return None
    try:
        return joblib.load(filename)
    except Exception as e:
        print(f"[ERROR] Failed to load file {filename}: {e}")
        return None
