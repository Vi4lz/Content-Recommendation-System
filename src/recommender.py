import pandas as pd
from sklearn.neighbors import NearestNeighbors
from utils import save_model, load_model
from logging_config import setup_logging
from fuzzywuzzy import process

logger = setup_logging()


def get_top_movies(df, top_n=10, percentile=0.90):
    """
    Returns the top N movies ranked by IMDb-style weighted rating.

    Args:
        df (pd.DataFrame): DataFrame containing at least 'vote_count' and 'vote_average'.
        top_n (int): Number of top-rated movies to return.
        percentile (float): Minimum vote count threshold (percentile-based).

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

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (v + m) * C)

    qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)
    return qualified.sort_values('weighted_rating', ascending=False).head(top_n)[
        ['title', 'vote_count', 'vote_average', 'weighted_rating']
    ]


def train_model(count_matrix):
    """
    Trains a NearestNeighbors model using the given count matrix.

    Args:
        count_matrix (csr_matrix): CountVectorizer matrix of the 'soup' field.

    Returns:
        NearestNeighbors: Trained nearest neighbor model.
    """
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11, n_jobs=-1)
    model.fit(count_matrix)
    return model


def get_or_train_model(count_matrix, model_path):
    """
    Loads a trained NearestNeighbors model if it exists; otherwise trains and saves a new one.

    Args:
        count_matrix (csr_matrix): CountVectorizer matrix.
        model_path (str): File path to store/load the trained model.

    Returns:
        NearestNeighbors: Trained model.
    """
    model = load_model(model_path)
    if model is None:
        logger.info("No pre-trained model found. Training now...")
        model = train_model(count_matrix)
        save_model(model, model_path)
    return model


def get_recommendations(title, nn_model, metadata, indices, count_matrix, top_n=10):
    """
    Returns a list of top N movie recommendations based on a given title.

    Args:
        title (str): Movie title to base recommendations on.
        nn_model (NearestNeighbors): Trained NearestNeighbors model.
        metadata (pd.DataFrame): DataFrame with movie metadata.
        indices (pd.Series): Series mapping movie titles to their DataFrame indices.
        count_matrix (csr_matrix): CountVectorizer matrix used during training.
        top_n (int): Number of recommendations to return.

    Returns:
        pd.Series: Titles of the recommended movies.
    """
    if title not in indices:
        logger.warning(f" Movie '{title}' not found in dataset.")
        return pd.Series(dtype=str)

    idx = indices[title]
    distances, neighbor_indices = nn_model.kneighbors(count_matrix[idx], n_neighbors=top_n + 1)
    recommended_indices = neighbor_indices.flatten()[1:]  # Exclude the queried movie itself
    unique_recommendations = list(set(metadata['title'].iloc[recommended_indices]))
    return pd.Series(unique_recommendations[:top_n])



def fuzzy_search(query: str, metadata: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Patobulinta fuzzy paieška su išfiltruotais trumpais filmais ir išvalytais rezultatais.
    """
    query = query.strip()

    if len(query) <= 3:
        candidates = metadata['title']
    else:
        candidates = metadata[metadata['title'].str.len() > 3]['title']

    raw_results = process.extract(query, candidates, limit=top_n)
    results = [(title, score) for title, score, _ in raw_results]  # Pašalinam indeksą
    matches = pd.DataFrame(results, columns=['title', 'score'])
    matches = matches[matches['score'] > 70]

    return matches.head(top_n)  # Limitavimo kodas

