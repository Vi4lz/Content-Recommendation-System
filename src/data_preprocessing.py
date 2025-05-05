import os
import pandas as pd

def load_all_csv_files(data_dir='../data'):
    """
    Function reads all CSV files in the given directory
    and returns a dictionary, where:
    - key: file name without the .csv extension
    - value: a specific pandas DataFrame
    Returns:
        datasets from the given directory
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

