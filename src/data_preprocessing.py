import os
import pandas as pd
from ast import literal_eval
from data_cleaning import clean_data, get_list, get_director
from logging_config import setup_logging

logger = setup_logging()

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
        logger.error(f"Directory {data_dir} does not exist.")
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
                logger.info(f"Loaded: {file_name} ({df.shape[0]} rows, {df.shape[1]} columns.)")
            except Exception as e:
                logger.error(f"Error reading {file_name} ({file_path}): {e}")

    return datasets


def merge_ratings_with_movies(ratings_df, movies_df):
    """
    Merge the ratings DataFrame with the movies DataFrame using the 'movieId' column.
    Returns:
        pd.DataFrame: A merged DataFrame that combines rating data with corresponding movie details.
    """
    try:
        merged = pd.merge(ratings_df, movies_df, on='movieId', how='left')
        logger.info(f"Merged ratings and movies tables: {merged.shape[0]} rows, {merged.shape[1]} columns.")
        return merged
    except Exception as e:
        logger.error(f"Error while merging the DataFrames: {e}")
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
    logger.info(f"Aggregated data saved to {output_file}.")
    logger.info(f"Aggregated {agg.shape[0]} movies with calculated rating statistics.")
    return agg


def load_or_create_aggregated_movies(data_dir, output_file):
    """
    Loads the aggregated movie ratings file if it exists, or creates it by merging
    the ratings and movies data.
    """
    if os.path.exists(output_file):
        logger.info(f"Found existing aggregated file: {output_file}")
        return pd.read_csv(output_file)

    datasets = load_all_csv_files(data_dir)
    movies = datasets.get('movies')
    ratings = datasets.get('ratings')

    if movies is not None and ratings is not None:
        merged = merge_ratings_with_movies(ratings, movies)
        return aggregate_movie_ratings(merged, output_file) if merged is not None else None
    else:
        logger.error("Missing ratings or movies data.")
        return None


def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []


# function to create 'soup' by combining cast, keywords, director, and genres
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


def load_and_merge_metadata(metadata_path, credits_path, keywords_path, merged_cache_path='merged_metadata.parquet'):
    try:
        if os.path.exists(merged_cache_path):
            logger.info(f"Found cached merged metadata at: {merged_cache_path}")
            return pd.read_csv(merged_cache_path)

        if not os.path.exists(metadata_path):
            logger.error(f"Error: {metadata_path} not found!")
            return None
        if not os.path.exists(credits_path):
            logger.error(f"Error: {credits_path} not found!")
            return None
        if not os.path.exists(keywords_path):
            logger.error(f"Error: {keywords_path} not found!")
            return None

        metadata = pd.read_csv(metadata_path, low_memory=False)
        credits = pd.read_csv(credits_path)
        keywords = pd.read_csv(keywords_path)

        metadata = metadata.drop([19730, 29503, 35587], errors='ignore')

        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        metadata['id'] = metadata['id'].astype('int')

        metadata = metadata.merge(credits, on='id', how='left')
        metadata = metadata.merge(keywords, on='id', how='left')

        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            metadata[feature] = metadata[feature].apply(safe_literal_eval)

        metadata['director'] = metadata['crew'].apply(get_director)

        for feature in ['cast', 'keywords', 'genres']:
            metadata[feature] = metadata[feature].apply(get_list)

        for feature in ['cast', 'keywords', 'director', 'genres']:
            metadata[feature] = metadata[feature].apply(clean_data)

        metadata['soup'] = metadata.apply(create_soup, axis=1)

        metadata.to_csv(merged_cache_path, index=False)
        logger.info(f"Merged metadata processed and saved to {merged_cache_path}")
        return metadata

    except Exception as e:
        logger.error(f"Error loading and processing datasets: {e}")
        raise
