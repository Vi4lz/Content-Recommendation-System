from data_preprocessing import load_all_csv_files

datasets = load_all_csv_files('../data')

movies = datasets.get('movies')
ratings = datasets.get('ratings')

if movies is not None:
    print("\n Movies:")
    print(movies.head())

if ratings is not None:
    print("\n Ratings:")
    print(ratings.head())