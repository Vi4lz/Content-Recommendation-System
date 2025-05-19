import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from recommender import fuzzy_search, get_recommendations, get_or_train_model, get_top_movies
from data_preprocessing import load_and_merge_metadata
from utils import load_model, save_model
from config import BASE_DIR, DATA_DIR, MERGED_CACHE_PATH, MATRIX_PATH, MODEL_PATH


# Global resource cache
_resources = {}


def load_resources():
    """
    Loads and caches all necessary resources for movie recommendation:
    - Metadata (merged)
    - Index mapping from titles
    - Count matrix (text vectorization)
    - NearestNeighbors model

    Returns:
        dict: Dictionary containing:
            - 'metadata' (pd.DataFrame): Merged and cleaned movie metadata.
            - 'indices' (pd.Series): Mapping from movie titles to DataFrame indices.
            - 'count_matrix' (csr_matrix): CountVectorizer-transformed text features.
            - 'nn_model' (NearestNeighbors): Trained recommendation model.
    """
    global _resources
    if _resources:
        return _resources  # Already loaded

    metadata_path = os.path.join(DATA_DIR, 'movies_metadata.csv')
    credits_path = os.path.join(DATA_DIR, 'credits.csv')
    keywords_path = os.path.join(DATA_DIR, 'keywords.csv')
    zip_path = os.path.join(DATA_DIR, 'raw_data.zip')  # pridėta
    extract_to = DATA_DIR

    metadata = load_and_merge_metadata(
        metadata_path,
        credits_path,
        keywords_path,
        MERGED_CACHE_PATH,
        zip_path=zip_path,
        extract_to=extract_to)
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
    """
    Performs fuzzy search on movie titles based on user input.

    Args:
        user_input (str): Partial or full movie title to search for.

    Returns:
        pd.DataFrame: Top matched movie titles with their scores and metadata (e.g. genres, release date).
    """
    res = load_resources()
    return fuzzy_search(user_input, res['metadata'])


def get_recommendations_by_title(title: str, top_n=15) -> pd.DataFrame:
    """
    Generates movie recommendations based on a given movie title.

    Args:
        title (str): Title of the reference movie.
        top_n (int, optional): Number of recommendations to return. Defaults to 15.

    Returns:
        pd.DataFrame: DataFrame containing recommended movies with metadata (title, genres, release date).
                     Returns empty DataFrame if the title is not found.
    """
    res = load_resources()

    if title not in res['indices']:
        return pd.DataFrame()  # Title not found

    return get_recommendations(
        title,
        res['nn_model'],
        res['metadata'],
        res['indices'],
        res['count_matrix'],
        top_n=top_n
    )


def get_top_rated_movies(top_n=100, percentile=0.90) -> pd.DataFrame:
    """
    Returns the top N movies based on a weighted IMDb-style rating.

    Args:
        top_n (int, optional): Number of top-rated movies to return. Defaults to 100.
        percentile (float, optional): Minimum vote count threshold percentile. Defaults to 0.90.

    Returns:
        pd.DataFrame: DataFrame of top-rated movies, sorted by weighted rating.
                      Missing values are filled with 'Unknown'.
    """
    res = load_resources()
    metadata = res['metadata']
    top_movies_df = get_top_movies(metadata, top_n=top_n, percentile=percentile)
    return top_movies_df.fillna('Unknown')
