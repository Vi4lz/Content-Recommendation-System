from data_preprocessing import load_or_create_aggregated_movies, load_all_csv_files
import pandas as pd
from model import get_top_movies, compute_tfidf_matrix, compute_cosine_cimilarity_matrix

aggregated_file = '../data/aggregated_movie_ratings.csv'
aggregated_movies = load_or_create_aggregated_movies('../data', aggregated_file)

metadata_df = pd.read_csv('../data/movies_metadata.csv', low_memory=False)



# if aggregated_movies is not None:
#     top_movies = get_top_movies(aggregated_movies, top_n=15)
#     print(top_movies.to_string(index=False))

if metadata_df is not None:
    tfidf_matrix = compute_tfidf_matrix(metadata_df, column='overview')
    print(f"Shape of TF-IDF matrix {tfidf_matrix.shape}.")
    cos_sim_matrix = compute_cosine_cimilarity_matrix(tfidf_matrix)
    print(cos_sim_matrix.shape)


