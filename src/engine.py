import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from recommender import fuzzy_search, get_recommendations, get_or_train_model, get_top_movies
from data_preprocessing import load_and_merge_metadata
from utils import load_model, save_model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MERGED_CACHE_PATH = os.path.join(DATA_DIR, 'merged_metadata.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'nn_model.joblib')
MATRIX_PATH = os.path.join(DATA_DIR, 'count_matrix.joblib')

# Global resource cache
_resources = {}

def load_resources():
    global _resources
    if _resources:
        return _resources  # already loaded

    metadata_path = os.path.join(DATA_DIR, 'movies_metadata.csv')
    credits_path = os.path.join(DATA_DIR, 'credits.csv')
    keywords_path = os.path.join(DATA_DIR, 'keywords.csv')

    metadata = load_and_merge_metadata(metadata_path, credits_path, keywords_path, MERGED_CACHE_PATH)
    if metadata is None or metadata.empty:
        raise ValueError("Metadata failed to load.")

    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

    count_matrix = load_model(MATRIX_PATH)
    if count_matrix is None:
        vectorizer = CountVectorizer(stop_words='english')
        count_matrix = vectorizer.fit_transform(metadata['soup'])
        save_model(count_matrix, MATRIX_PATH)

    nn_model = get_or_train_model(count_matrix, MODEL_PATH)

    _resources = {
        'metadata': metadata,
        'indices': indices,
        'count_matrix': count_matrix,
        'nn_model': nn_model
    }
    return _resources


def get_matches(user_input: str) -> pd.DataFrame:
    res = load_resources()
    return fuzzy_search(user_input, res['metadata'])


def get_recommendations_by_title(title: str, top_n=15):
    res = load_resources()

    if title not in res['indices']:
        return pd.DataFrame()  # Returning empty DataFrame if no match found

    return get_recommendations(
        title,
        res['nn_model'],
        res['metadata'],
        res['indices'],
        res['count_matrix'],
        top_n=top_n
    )


def get_top_rated_movies(top_n=100, percentile=0.90):
    res = load_resources()
    metadata = res['metadata']
    top_movies_df = get_top_movies(metadata, top_n=top_n, percentile=percentile)
    return top_movies_df.fillna('Unknown')


