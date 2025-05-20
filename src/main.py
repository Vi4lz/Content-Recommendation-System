import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from data_preprocessing import load_and_merge_metadata
from utils import save_model, load_model
from recommender import get_recommendations, get_top_movies, fuzzy_search
from logging_config import setup_logging
from config import  BASE_DIR, DATA_DIR, MERGED_CACHE_PATH, MATRIX_PATH, MODEL_PATH

logger = setup_logging()

def main():
    """
    Entry point for the recommendation system.
    Loads data, prepares features, vectorizes soup, fits or loads NearestNeighbors model,
    and provides movie recommendations.
    """

    # Define raw data paths
    metadata_path = os.path.join(DATA_DIR, 'movies_metadata.csv')
    credits_path = os.path.join(DATA_DIR, 'credits.csv')
    keywords_path = os.path.join(DATA_DIR, 'keywords.csv')
    zip_path = os.path.join(DATA_DIR, 'raw_data.zip')
    extract_to = DATA_DIR

    # Load metadata
    metadata = load_and_merge_metadata(metadata_path, credits_path, keywords_path, MERGED_CACHE_PATH, zip_path, extract_to)

    if metadata is None or metadata.empty:
        logger.error("Failed to load metadata.")
        return

    # Get top movies based on IMDb-style weighted rating
    top_movies = get_top_movies(metadata)
    logger.info("Top Movies based on weighted rating:")
    logger.info(top_movies.head(10).to_string(index=False))  # Printing top 10

    # Create reverse index
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

    # Vectorize soup (or load from file)
    count_matrix = load_model(MATRIX_PATH)
    if count_matrix is None:
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(metadata['soup'])
        save_model(count_matrix, MATRIX_PATH)

    # Load or fit NearestNeighbors model
    nn_model = load_model(MODEL_PATH)
    if nn_model is None:
        nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11, n_jobs=-1)
        nn_model.fit(count_matrix)
        save_model(nn_model, MODEL_PATH)

    user_input = input("Enter a movie title: ").strip()
    matches = fuzzy_search(user_input, metadata)

    if matches.empty:
        logger.warning(f"Similar movies not found: {user_input}")
        return

    logger.info(f"\nMaybe you had in mind:\n{matches.to_string(index=False)}\n")

    title = matches.iloc[0]['title']
    logger.info(f"Chosen Movie: {title}")

    if title not in indices:
        logger.warning(f"Movie '{title}' not found in dataset.")
        return

    logger.info(f"\nGenerating recommendations for: {title}\n")
    recommendations = get_recommendations(title, nn_model, metadata, indices, count_matrix, top_n=10)

    logger.info("[RECOMMENDATIONS]")
    logger.info(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()
