import os
import pandas as pd
from data_preprocessing import load_all_csv_files, merge_ratings_with_movies, aggregate_movie_ratings
from model import get_top_movies

aggregated_file = '../data/aggregated_movie_ratings.csv'

if os.path.exists(aggregated_file):
    print(f"Found existing aggregated file: {aggregated_file}")
    aggregated_movies = pd.read_csv(aggregated_file)
else:
    print("Aggregated file not found. Loading full datasets...")
    datasets = load_all_csv_files('../data')

    if 'movies' in datasets and 'ratings' in datasets:
        movies_df = datasets['movies']
        ratings_df = datasets['ratings']

        merged_data = merge_ratings_with_movies(ratings_df, movies_df)

        if merged_data is not None:
            aggregated_movies = aggregate_movie_ratings(merged_data, output_file=aggregated_file)
        else:
            aggregated_movies = None
    else:
        print("Required datasets (movies, ratings) are missing!")
        aggregated_movies = None

if aggregated_movies is not None:
    top_10 = get_top_movies(aggregated_movies, top_n=10)
    print(top_10.to_string(index=False))


