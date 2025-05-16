import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from recommender import fuzzy_search, get_recommendations, get_or_train_model, get_top_movies
from data_preprocessing import load_and_merge_metadata
from utils import load_model, save_model
from logging_config import setup_logging
from typing import Dict, Any

setup_logging()
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MERGED_CACHE_PATH = os.path.join(DATA_DIR, 'merged_metadata.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'nn_model.joblib')
MATRIX_PATH = os.path.join(DATA_DIR, 'count_matrix.joblib')

_resources: Dict[str, Any] = {}

def load_resources() -> Dict[str, Any]:
    """
    Loads and caches metadata, indices, count matrix, and NearestNeighbors model.

    Returns:
        dict: Dictionary containing loaded resources:
            - 'metadata' (pd.DataFrame)
            - 'indices' (pd.Series)
            - 'count_matrix' (scipy.sparse matrix)
            - 'nn_model' (NearestNeighbors)
    """
    global _resources
    if _resources:
        logger.debug("Resources already loaded from cache.")
        return _resources

    logger.info("Loading metadata and models...")

    metadata_path = os.path.join(DATA_DIR, 'movies_metadata.csv')
    credits_path = os.path.join(DATA_DIR, 'credits.csv')
    keywords_path = os.path.join(DATA_DIR, 'keywords.csv')

    try:
        metadata = load_and_merge_metadata(metadata_path, credits_path, keywords_path, MERGED_CACHE_PATH)
    except Exception as e:
        logger.exception("Failed to load and merge metadata.")
        raise e

    if metadata is None or metadata.empty:
        logger.error("Metadata is empty or failed to load.")
        raise ValueError("Metadata failed to load.")

    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
    logger.info("Metadata and indices loaded successfully.")

    count_matrix = load_model(MATRIX_PATH)
    if count_matrix is None:
        logger.info("Count matrix not found. Generating...")
        vectorizer = CountVectorizer(stop_words='english')
        count_matrix = vectorizer.fit_transform(metadata['soup'])
        save_model(count_matrix, MATRIX_PATH)
        logger.info("Count matrix generated and saved.")

    nn_model = get_or_train_model(count_matrix, MODEL_PATH)
    logger.info("NearestNeighbors model loaded.")

    _resources = {
        'metadata': metadata,
        'indices': indices,
        'count_matrix': count_matrix,
        'nn_model': nn_model
    }

    logger.debug("All resources loaded successfully.")
    return _resources


def get_matches(user_input: str) -> pd.DataFrame:
    """
    Returns fuzzy-matched movie titles based on user input.

    Args:
        user_input (str): The string entered by the user.

    Returns:
        pd.DataFrame: DataFrame containing matching movie titles and metadata.
    """
    logger.debug(f"Performing fuzzy search for input: {user_input}")
    res = load_resources()
    matches = fuzzy_search(user_input, res['metadata'])
    logger.info(f"Found {len(matches)} matches for input '{user_input}'")
    return matches


def get_recommendations_by_title(title: str, top_n: int = 15) -> pd.DataFrame:
    """
    Gets content-based movie recommendations for a given title.

    Args:
        title (str): Movie title to base recommendations on.
        top_n (int, optional): Number of recommendations to return. Defaults to 15.

    Returns:
        pd.DataFrame: DataFrame of recommended movies with metadata.
    """
    logger.debug(f"Getting recommendations for: {title}")
    res = load_resources()

    if title not in res['indices']:
        logger.warning(f"Title '{title}' not found in indices.")
        return pd.DataFrame()

    recommendations = get_recommendations(
        title,
        res['nn_model'],
        res['metadata'],
        res['indices'],
        res['count_matrix'],
        top_n=top_n
    )
    logger.info(f"Returning {len(recommendations)} recommendations for title: {title}")
    return recommendations


def get_top_rated_movies(top_n: int = 100, percentile: float = 0.90) -> pd.DataFrame:
    """
    Returns the top-rated movies based on IMDb-style weighted rating.

    Args:
        top_n (int, optional): Number of top movies to return. Defaults to 100.
        percentile (float, optional): Minimum vote threshold percentile. Defaults to 0.90.

    Returns:
        pd.DataFrame: DataFrame of top-rated movies.
    """
    logger.debug("Fetching top rated movies.")
    res = load_resources()
    metadata = res['metadata']
    top_movies_df = get_top_movies(metadata, top_n=top_n, percentile=percentile)
    logger.info(f"Top {top_n} movies fetched based on weighted rating.")
    return top_movies_df.fillna('Unknown')
