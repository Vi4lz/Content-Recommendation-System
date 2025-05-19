import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MERGED_CACHE_PATH = os.path.join(DATA_DIR, 'merged_metadata.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'nn_model.joblib')
MATRIX_PATH = os.path.join(DATA_DIR, 'count_matrix.joblib')

