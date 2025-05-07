import pandas as pd
from data_preprocessing import load_or_create_aggregated_movies, load_all_csv_files, load_and_merge_metadata
from recommender import get_top_movies, compute_tfidf_matrix, compute_cosine_similarity_matrix, get_recommendations
from utils import save_joblib, load_joblib
import os

# Keliai iki duomenų
data_dir = '../data'
tfidf_pickle_path = os.path.join(data_dir, 'tfidf_matrix.pkl')
cosine_pickle_path = os.path.join(data_dir, 'cosine_sim_matrix.pkl')
aggregated_file = os.path.join(data_dir, 'aggregated_movie_ratings.csv')
metadata_path = '../data/movies_metadata.csv'
credits_path = '../data/credits.csv'
keywords_path = '../data/keywords.csv'

# 1. Įkeliame metaduomenis
metadata = load_and_merge_metadata(metadata_path, credits_path, keywords_path)
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# 2. Krauname arba kuriame TF-IDF matricą
tfidf_matrix = load_joblib(tfidf_pickle_path)
if tfidf_matrix is None:
    print("[INFO] TF-IDF matrix not found. Computing and saving...")
    tfidf_matrix = compute_tfidf_matrix(metadata, column='overview')
    save_joblib(tfidf_matrix, tfidf_pickle_path)

# 3. Krauname arba kuriame Cosine panašumo matricą
cosine_sim_matrix = load_joblib(cosine_pickle_path)
if cosine_sim_matrix is None:
    print("[INFO] Cosine similarity matrix not found. Computing and saving...")
    cosine_sim_matrix = compute_cosine_similarity_matrix(tfidf_matrix)
    save_joblib(cosine_sim_matrix, cosine_pickle_path)

# 4. Gauti rekomendacijas
title = 'The Dark Knight Rises'
recommended_movies = get_recommendations(title, cosine_sim_matrix, indices, metadata, top_n=10)

if recommended_movies is not None:
    print(f"\n[RECOMMENDATIONS for '{title}']\n")
    print(recommended_movies.to_string(index=False))

# [DEV ONLY]
# aggregated_movies = load_or_create_aggregated_movies(data_dir, aggregated_file)
#
# if aggregated_movies is not None:
#     top_movies = get_top_movies(aggregated_movies, top_n=15)
#     print(top_movies.to_string(index=False))

print(metadata.head(2).to_string())