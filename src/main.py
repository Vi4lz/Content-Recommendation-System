import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from data_preprocessing import load_and_merge_metadata
from utils import save_model, load_model
from recommender import get_recommendations

def main():
    """
    Entry point for the recommendation system.
    Loads data, prepares features, vectorizes soup, fits or loads NearestNeighbors model,
    and provides movie recommendations.
    """
    # Define base paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    MODEL_PATH = os.path.join(DATA_DIR, 'nn_model.joblib')
    MATRIX_PATH = os.path.join(DATA_DIR, 'count_matrix.joblib')

    # Load metadata
    metadata_path = os.path.join(DATA_DIR, 'movies_metadata.csv')
    credits_path = os.path.join(DATA_DIR, 'credits.csv')
    keywords_path = os.path.join(DATA_DIR, 'keywords.csv')

    metadata = load_and_merge_metadata(metadata_path, credits_path, keywords_path)

    if metadata is None or metadata.empty:
        print("[ERROR] Failed to load metadata.")
        return

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

    # Request recommendations
    title = "The Dark Knight Rises"
    if title not in indices:
        print(f"[WARN] Movie '{title}' not found in dataset.")
        return

    print(f"\n[INFO] Generating recommendations for: {title}\n")
    recommendations = get_recommendations(title, nn_model, metadata, indices, count_matrix, top_n=10)

    # Print results
    print("[RECOMMENDATIONS]")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
