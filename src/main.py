import pandas as pd
import os
from data_preprocessing import load_and_merge_metadata, clean_data, create_soup
from recommender import get_top_movies, get_recommendations
from vectorizers import compute_tfidf_matrix, compute_count_vector_matrix, compute_cosine_similarity_matrix
from utils import save_joblib, load_joblib


data_dir = '../data'
tfidf_pickle_path = os.path.join(data_dir, 'tfidf_matrix.pkl')
cosine_pickle_path = os.path.join(data_dir, 'cosine_sim_matrix.pkl')
count_pickle_path = os.path.join(data_dir, 'count_matrix.pkl')
cosine_count_pickle_path = os.path.join(data_dir, 'cosine_sim_count.pkl')

metadata_path = '../data/movies_metadata.csv'
credits_path = '../data/credits.csv'
keywords_path = '../data/keywords.csv'

metadata = load_and_merge_metadata(metadata_path, credits_path, keywords_path)

metadata = metadata.sample(5000, random_state=42).reset_index(drop=True)

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

metadata['soup'] = metadata.apply(create_soup, axis=1)
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

method = 'metadata' # select method 'overview' or 'metadata'

if method == 'overview':
    tfidf_matrix = load_joblib(tfidf_pickle_path) # load or create TF-IDF matrix.
    if tfidf_matrix is None:
        print("[INFO] TF-IDF matrix not found. Computing and saving...")
        tfidf_matrix = compute_tfidf_matrix(metadata, column='overview')
        save_joblib(tfidf_matrix, tfidf_pickle_path)

    # load or create cosine similarity matrix.
    cosine_sim_matrix = load_joblib(cosine_pickle_path)
    if cosine_sim_matrix is None:
        print("[INFO] Cosine similarity matrix not found. Computing and saving...")
        cosine_sim_matrix = compute_cosine_similarity_matrix(tfidf_matrix)
        save_joblib(cosine_sim_matrix, cosine_pickle_path)

else:  # method == 'metadata'
    count_matrix = load_joblib(count_pickle_path)
    if count_matrix is None:
        print("[INFO] Count vector matrix not found. Computing and saving...")
        count_matrix = compute_count_vector_matrix(metadata, column='soup')
        save_joblib(count_matrix, count_pickle_path)

    cosine_sim_matrix = load_joblib(cosine_count_pickle_path)
    if cosine_sim_matrix is None:
        print("[INFO] Cosine similarity (count-based) not found. Computing and saving...")
        cosine_sim_matrix = compute_cosine_similarity_matrix(count_matrix, use_linear_kernel=False)
        save_joblib(cosine_sim_matrix, cosine_count_pickle_path)

# GET RECOMMENDATIONS.
title = 'The Dark Knight Rises'
recommended_movies = get_recommendations(title, cosine_sim_matrix, indices, metadata, top_n=10)

if recommended_movies is not None:
    print(f"\n[RECOMMENDATIONS for '{title}']\n")
    print(recommended_movies.to_string(index=False))


# SHOW SOME FOR TESTING,
print("\n[Sample Metadata Preview]")
print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3).to_string(index=False))
print("\n[Soup Column Preview]")
print(metadata[['soup']].head(2).to_string(index=False))


# [DEV ONLY]
# aggregated_movies = load_or_create_aggregated_movies(data_dir, aggregated_file)
#
# if aggregated_movies is not None:
#     top_movies = get_top_movies(aggregated_movies, top_n=15)
#     print(top_movies.to_string(index=False))