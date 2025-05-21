import pandas as pd
from sklearn.neighbors import NearestNeighbors
from utils import save_model, load_model
from logging_config import setup_logging
from fuzzywuzzy import process

logger = setup_logging()


def get_top_movies(df, top_n=100, percentile=0.90):
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

    C = df['vote_average'].mean()  # All movies average score
    m = df['vote_count'].quantile(percentile) # Minimum requirement of votes (90%)

    qualified = df[df['vote_count'] >= m].copy()

    if qualified.empty:
        logger.warning("No movies meet the minimum vote count threshold.")
        return pd.DataFrame()

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count'] # Single movie vote count
        R = x['vote_average'] # Single movie vote average
        return (v / (v + m) * R) + (m / (v + m) * C)

    qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)
    return qualified.sort_values('weighted_rating', ascending=False).head(top_n)[
        ['title', 'vote_count', 'vote_average', 'weighted_rating', 'release_date']
    ]


def train_model(count_matrix):
    """
    Trains a NearestNeighbors model using the given count matrix.

    Returns:
        NearestNeighbors: Trained nearest neighbor model.
    """
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11, n_jobs=-1)
    model.fit(count_matrix)
    return model


def get_or_train_model(count_matrix, model_path):
    """
    Loads a trained NearestNeighbors model if it exists; otherwise trains and saves a new one.

    Returns:
        NearestNeighbors: Trained model.
    """
    model = load_model(model_path)
    if model is None:
        logger.info("No pre-trained model found. Training now...")
        model = train_model(count_matrix)
        save_model(model, model_path)
    return model


def get_recommendations(title, nn_model, metadata, indices, count_matrix, top_n=15):
    """
    Returns a list of top N movie recommendations based on a given title.
    This function now also returns additional metadata for each recommended movie.

    Args:
        title (str): Movie title to base recommendations on.
        nn_model (NearestNeighbors): Trained NearestNeighbors model.
        metadata (pd.DataFrame): DataFrame with movie metadata.
        indices (pd.Series): Series mapping movie titles to their DataFrame indices.
        count_matrix (csr_matrix): CountVectorizer matrix used during training.
        top_n (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: DataFrame with titles, release date, genres, and director of the recommended movies.
    """
    if title not in indices:
        logger.warning(f"Movie '{title}' not found in dataset.")
        return pd.DataFrame()  # Returning empty DataFrame if no match found

    idx = indices[title]
    distances, neighbor_indices = nn_model.kneighbors(count_matrix[idx], n_neighbors=top_n + 1)
    recommended_indices = neighbor_indices.flatten()[1:]  # Exclude the queried movie itself

    recommended_titles = metadata['title'].iloc[recommended_indices].unique()

    # Get additional details like release_date, genres, and director
    recommendations_with_details = metadata[metadata['title'].isin(recommended_titles)].copy()


    recommendations_with_details['release_date'] = recommendations_with_details['release_date'].fillna('Unknown')
    recommendations_with_details['genres'] = recommendations_with_details['genres'].astype(str)
    recommendations_with_details['genres'] = recommendations_with_details['genres'].str.replace(r"[\[\]']", '',
                                                                                                regex=True)
    recommendations_with_details['genres'] = recommendations_with_details['genres'].replace('', 'Unknown')

    return recommendations_with_details[['title', 'release_date', 'genres']].head(top_n)


def fuzzy_search(query: str, metadata: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Performs fuzzy search to find movies in the metadata that closely match the input query.

    Args:
        query (str): User input or partial movie title to search for.
        metadata (pd.DataFrame): Movie metadata containing at least 'title', 'genres', and 'release_date'.
        top_n (int, optional): Maximum number of results to return. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame containing the top matching movie titles.
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

    # Now, we also fetch 'genres' and 'release_date' based on the matched titles
    matches_with_details = metadata[metadata['title'].isin(matches['title'])].copy()

    matches_with_details['genres'] = matches_with_details['genres'].astype(str)
    matches_with_details['genres'] = matches_with_details['genres'].str.replace(r"[\[\]']", '',
                                                                                                regex=True)
    matches_with_details['genres'] = matches_with_details['genres'].replace('', 'Unknown')

    # Merge the score with the matched results
    matches_with_details = pd.merge(matches, matches_with_details, on='title')

    return matches_with_details[['title', 'score', 'genres', 'release_date']].head(top_n)

