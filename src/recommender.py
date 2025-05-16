import pandas as pd
from sklearn.neighbors import NearestNeighbors
from utils import save_model, load_model
from logging_config import setup_logging
from fuzzywuzzy import process
from typing import Optional

logger = setup_logging()


def get_top_movies(df: pd.DataFrame, top_n: int = 100, percentile: float = 0.90) -> pd.DataFrame:
    """
    Returns the top N movies ranked by IMDb-style weighted rating.
    The weighted rating combines the movie's average rating and the number of votes it received.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'vote_count' and 'vote_average' columns.
        top_n (int): Number of top-rated movies to return (default is 100).
        percentile (float): Minimum vote count threshold (percentile-based, default is 0.90).

    Returns:
        pd.DataFrame: Top N movies sorted by weighted rating.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning empty result.")
        return pd.DataFrame()

    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(percentile)

    qualified = df[df['vote_count'] >= m].copy()

    if qualified.empty:
        logger.warning("No movies meet the minimum vote count threshold.")
        return pd.DataFrame()

    def weighted_rating(x: pd.Series, m: float = m, C: float = C) -> float:
        """
        Calculates the weighted rating for a movie based on its vote count and average rating.

        Args:
            x (pd.Series): Series containing 'vote_count' and 'vote_average' for a movie.
            m (float): Minimum vote count threshold.
            C (float): Mean of all movie ratings.

        Returns:
            float: Weighted rating of the movie.
        """
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (v + m) * C)

    qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)
    return qualified.sort_values('weighted_rating', ascending=False).head(top_n)[
        ['title', 'release_date', 'vote_count', 'vote_average', 'weighted_rating']
    ]


def train_model(count_matrix: pd.DataFrame) -> NearestNeighbors:
    """
    Trains a NearestNeighbors model using the given count matrix.

    Args:
        count_matrix (pd.DataFrame): The matrix of features (e.g., from CountVectorizer).

    Returns:
        NearestNeighbors: Trained NearestNeighbors model.
    """
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11, n_jobs=-1)
    model.fit(count_matrix)
    return model


def get_or_train_model(count_matrix: pd.DataFrame, model_path: str) -> NearestNeighbors:
    """
    Loads a trained NearestNeighbors model if it exists; otherwise trains and saves a new one.

    Args:
        count_matrix (pd.DataFrame): The matrix of features (e.g., from CountVectorizer).
        model_path (str): Path where the model is saved or will be saved.

    Returns:
        NearestNeighbors: Trained model.
    """
    model = load_model(model_path)
    if model is None:
        logger.info("No pre-trained model found. Training now...")
        model = train_model(count_matrix)
        save_model(model, model_path)
    return model


def get_recommendations(
    title: str, nn_model: NearestNeighbors, metadata: pd.DataFrame, indices: pd.Series,
    count_matrix: pd.DataFrame, top_n: int = 15
) -> pd.DataFrame:
    """
    Returns a list of top N movie recommendations based on a given movie title.
    The recommendations are generated using a NearestNeighbors model trained on a count matrix.

    Args:
        title (str): Movie title to base recommendations on.
        nn_model (NearestNeighbors): Trained NearestNeighbors model.
        metadata (pd.DataFrame): DataFrame with movie metadata.
        indices (pd.Series): Series mapping movie titles to their DataFrame indices.
        count_matrix (pd.DataFrame): Count matrix used during training.
        top_n (int): Number of recommendations to return (default is 15).

    Returns:
        pd.DataFrame: DataFrame with titles, release date, genres, and director of the recommended movies.
    """
    if title not in indices:
        logger.warning(f"Movie '{title}' not found in dataset.")
        return pd.DataFrame()

    idx = indices[title]
    distances, neighbor_indices = nn_model.kneighbors(count_matrix[idx], n_neighbors=top_n + 1)
    recommended_indices = neighbor_indices.flatten()[1:]  # Exclude the queried movie itself

    recommended_titles = metadata['title'].iloc[recommended_indices].unique()
    recommendations_with_details = metadata[metadata['title'].isin(recommended_titles)].copy()
    recommendations_with_details['release_date'] = recommendations_with_details['release_date'].fillna('Unknown')
    recommendations_with_details['genres'] = recommendations_with_details['genres'].fillna('Unknown')

    return recommendations_with_details[['title', 'release_date', 'genres']].head(top_n)


def fuzzy_search(query: str, metadata: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Searches for movies that match a given query string using fuzzy matching.
    Returns the top N matches based on similarity to the query.

    Args:
        query (str): Movie title to search for.
        metadata (pd.DataFrame): DataFrame with movie metadata.
        top_n (int): Number of top matches to return (default is 10).

    Returns:
        pd.DataFrame: DataFrame with movie titles, similarity score, genres, and release date.
    """
    query = query.strip()

    if len(query) <= 3:
        candidates = metadata['title']
    else:
        candidates = metadata[metadata['title'].str.len() > 3]['title']

    raw_results = process.extract(query, candidates, limit=top_n)
    results = [(title, score) for title, score, _ in raw_results]
    matches = pd.DataFrame(results, columns=['title', 'score'])
    matches = matches[matches['score'] > 70]

    matches_with_details = metadata[metadata['title'].isin(matches['title'])].copy()
    matches_with_details = pd.merge(matches, matches_with_details, on='title')

    return matches_with_details[['title', 'score', 'genres', 'release_date']].head(top_n)
