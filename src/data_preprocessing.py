import os
import pandas as pd

def load_all_csv_files(data_dir='../data'):
    """
    Function reads all CSV files in the given directory
    and returns a dictionary, where:
    - key: file name without the .csv extension
    - value: a specific pandas DataFrame
    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are filenames (without .csv extension)
                                 and values are the corresponding pandas DataFrames.
    """

    datasets = {}

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return {}

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            try:
                df = pd.read_csv(file_path)
                key_name = file_name.replace('.csv', '')
                datasets[key_name] = df
                print(f"Loaded: {file_name} ({df.shape[0]} rows., {df.shape[1]} columns.)")
            except Exception as e:
                print(f"Error reading {file_name} ({file_path}): {e}")

    return datasets


def merge_ratings_with_movies(ratings_df, movies_df):
    """
    Merge the ratings DataFrame with the movies DataFrame using the 'movieId' column.
    Returns:
        pd.DataFrame: A merged DataFrame that combines rating data with corresponding movie details.
    """
    try:
        merged = pd.merge(ratings_df, movies_df, on='movieId', how='left')
        print(f"Merged table: {merged.shape[0]} rows, {merged.shape[1]} columns.")
        return merged
    except Exception as e:
        print(f"Error while merging the DataFrames: {e}")
        return None


def aggregate_movie_ratings(merged_df, output_file='aggregated_movie_ratings.csv'):
    """
    Aggregates movie ratings by calculating the total number of ratings (vote_count)
    and the average rating (vote_average) for each movie.

    Saves the aggregated data to a CSV file.

    Returns:
        pd.DataFrame: A new DataFrame containing columns:
                      - movieId
                      - title
                      - vote_count (number of ratings per movie)
                      - vote_average (average rating per movie)
    """
    agg = merged_df.groupby(['movieId', 'title']).agg(
        vote_count=('rating', 'count'),
        vote_average=('rating', 'mean')
    ).reset_index()

    agg.to_csv(output_file, index=False)
    print(f"Aggregated data saved to {output_file}.")
    print(f"Aggregated {agg.shape[0]} movies with calculated rating statistics.")
    return agg


def merge_movies_with_tags(movies_df, tags_df):
    """
    Merges movies and tags DataFrames on 'movieId'.
    Tags are grouped and concatenated into a single string per movie.
    Returns:
        pd.DataFrame: A merged DataFrame containing movie details and concatenated tags.
    """
    if movies_df.empty or tags_df.empty:
        print("One or both input DataFrames are empty.")
        return pd.DataFrame()

    grouped = tags_df.groupby('movieId')['tag']
    tags_grouped = grouped.apply(lambda tags: ' '.join(str(tag) for tag in tags)).reset_index()

    merged_df = pd.merge(movies_df, tags_grouped, on='movieId', how='left')

    print(f"Merged movies and tags: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns.")
    return merged_df


def load_or_create_aggregated_movies(data_dir, output_file):
    """
    Loads the aggregated movie ratings file if it exists, or creates it by merging
    the ratings and movies data.
    """
    if os.path.exists(output_file):
        print(f"Found existing aggregated file: {output_file}")
        return pd.read_csv(output_file)

    datasets = load_all_csv_files(data_dir)
    movies = datasets.get('movies')
    ratings = datasets.get('ratings')

    if movies is not None and ratings is not None:
        merged = merge_ratings_with_movies(ratings, movies)
        return aggregate_movie_ratings(merged, output_file) if merged is not None else None
    else:
        print("Missing ratings or movies data.")
        return None


def load_or_create_movies_with_tags(data_dir, output_file):
    """
    Loads the movies with tags file if it exists, or creates it by merging the movies and tags data.
    """
    if os.path.exists(output_file):
        print(f"Loaded existing movies with tags file: {output_file}")
        return pd.read_csv(output_file)

    datasets = load_all_csv_files(data_dir)
    movies = datasets.get('movies')
    tags = datasets.get('tags')

    if movies is not None and tags is not None:
        merged = merge_movies_with_tags(movies, tags)
        merged.to_csv(output_file, index=False)
        return merged
    else:
        print("Missing movies or tags data.")
        return None
