from data_preprocessing import load_all_csv_files, merge_ratings_with_movies

datasets = load_all_csv_files('../data')

movies = datasets.get('movies')
ratings = datasets.get('ratings')

# if movies is not None:
#     print("\n Movies:")
#     print(movies.head())
#
# if ratings is not None:
#     print("\n Ratings:")
#     print(ratings.head())


if movies is not None and ratings is not None:
    merged_data = merge_ratings_with_movies(ratings, movies)
    print(merged_data.head().to_string())