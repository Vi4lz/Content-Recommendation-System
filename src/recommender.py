import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger(__name__)

def get_top_movies(df, top_n=10, percentile=0.90):
    """
     Returns the top N movies ranked by IMDb-style weighted rating.

    Parameters:
        df (pd.DataFrame): Aggregated movie ratings with 'vote_count' and 'vote_average'.
        top_n (int): Number of top movies to return.
        percentile (float): Percentile to determine minimum vote count (m).

    Returns:
        pd.DataFrame: Top N movies sorted by weighted rating.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning empty result.")
        return pd.DataFrame()

    C = df['vote_average'].mean()    # mean rating across all movies
    m = df['vote_count'].quantile(percentile)   # number of votes received by a movie in the percentile param.

    has_enough_votes = df['vote_count'] >= m   # condition to filter out movies having greater than equal to given percent vote counts.
    qualified = df[has_enough_votes].copy()   # new independent df with calculations.

    if qualified.empty:
        logger.warning("No movies meet the minimum vote count threshold.")
        return pd.DataFrame()

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v+m) * R) + (m / (v + m) * C)   # Applies IMDb weighted rating formula to a single movie.

    qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)
    top_movies = qualified.sort_values('weighted_rating', ascending=False).head(top_n)

    return top_movies[['title', 'vote_count', 'vote_average', 'weighted_rating']]


def compute_tfidf_matrix(df, column='overview'):
    """
    Computes a TF-IDF matrix for the specified text column in the given DataFrame.

    This function transforms a column of text data (e.g., tags or descriptions)
    into a matrix of TF-IDF features, ignoring common English stop words.
    Returns:
        scipy.sparse.csr.csr_matrix: Sparse matrix of TF-IDF features.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df[column] = df[column].fillna('')   # Replace Nan with empty string
    logger.info(f"Column '{column}' processed. Number of rows: {len(df)}")

    try:
        tfidf = TfidfVectorizer(stop_words='english')  # Initialize TF-IDF vectorizer
        tfidf_matrix = tfidf.fit_transform(df[column])  # Apply vectorizer to the specified column
        logger.info(f"TF-IDF matrix created. Shape: {tfidf_matrix.shape}")
    except Exception as e:
        logger.error(f"Error during TF-IDF matrix creation: {str(e)}")
        raise
    return tfidf_matrix


def compute_cosine_similarity_matrix(tfidf_matrix):
    """
    [DEVELOPMENT USE ONLY]
    Compute a cosine similarity matrix from a TF-IDF matrix.
    Use only for testing purposes, because it's a slow and RAM demanding operation.
    Parameter:
        tfidf_matrix (scipy.sparse matrix)
    Returns:
        ndarray: Cosine similarity matrix (NxN), where N - movie count.
    """
    if tfidf_matrix is None or tfidf_matrix.shape[0] == 0:
        raise ValueError("TF-IDF matrix is empty or None.")

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return cosine_sim



def get_recommendations(title, cosine_sim, indices, metadata, top_n=10):
    """
    Given a movie title, returns the top N most similar movies based on cosine similarity.

    Args:
        title (str): The title of the movie.
        cosine_sim (pd.DataFrame): The cosine similarity matrix.
        indices (pd.Series): A series that maps movie titles to their corresponding indices.
        metadata (pd.DataFrame): The dataframe containing the movie metadata.
        top_n (int): The number of top recommendations to return.

    Returns:
        pd.DataFrame: The top N recommended movies and their similarity scores.
    """
    if title not in indices:
        print(f"Movie '{title}' not found.")
        return None

    idx = indices[title]  # Get the index of the input movie title
    sim_scores = list(enumerate(cosine_sim[idx]))  # Get pairwise similarity scores of all movies with that movie
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort movies based on similarity score
    sim_scores = sim_scores[1:top_n + 1]  # Get top N most similar movies (exclude the movie itself)

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the top N most similar movies
    return metadata['title'].iloc[movie_indices]
