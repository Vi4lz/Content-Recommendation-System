import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

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
                if file_name == 'movies_metadata.csv':
                    df = pd.read_csv(file_path, low_memory=False)
                else:
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


def load_and_merge_metadata(metadata_path, credits_path, keywords_path):
    try:
        if not os.path.exists(metadata_path):
            print(f"Error: {metadata_path} not found!")
            return None
        if not os.path.exists(credits_path):
            print(f"Error: {credits_path} not found!")
            return None
        if not os.path.exists(keywords_path):
            print(f"Error: {keywords_path} not found!")
            return None

        metadata = pd.read_csv(metadata_path, low_memory=False)
        credits = pd.read_csv(credits_path)
        keywords = pd.read_csv(keywords_path)

        metadata = metadata.drop([19730, 29503, 35587])
        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        metadata['id'] = metadata['id'].astype('int')

        metadata = metadata.merge(credits, on='id', how='left')
        metadata = metadata.merge(keywords, on='id', how='left')

        print("Successfully merged metadata, credits, and keywords.")
        return metadata
    except Exception as e:
        print(f"Error loading and merging datasets: {e}")
        raise