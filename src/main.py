from data_preprocessing import (
    load_or_create_aggregated_movies,
    load_or_create_movies_with_tags
)
from model import get_top_movies

aggregated_file = '../data/aggregated_movie_ratings.csv'
movies_with_tags_file = '../data/movies_with_tags.csv'

aggregated_movies = load_or_create_aggregated_movies('../data', aggregated_file)
movies_with_tags = load_or_create_movies_with_tags('../data', movies_with_tags_file)

if aggregated_movies is not None:
    top_movies = get_top_movies(aggregated_movies, top_n=15)
    print(top_movies.to_string(index=False))


if movies_with_tags is not None:
    print(movies_with_tags.head().to_string())