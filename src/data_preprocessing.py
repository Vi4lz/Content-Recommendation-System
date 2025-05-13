import os
import pandas as pd
from data_cleaning import clean_data, get_list, get_director, clean_metadata, clean_features
from logging_config import setup_logging

logger = setup_logging()



# Function to create 'soup' by combining cast, keywords, director, and genres
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


def load_and_merge_metadata(metadata_path, credits_path, keywords_path, merged_cache_path='merged_metadata.parquet'):
    """
    Loads and merges movie metadata, credits, and keywords datasets.
    Performs data merging and stores the processed dataset.

    Args:
        metadata_path (str): Path to the movie metadata CSV file.
        credits_path (str): Path to the movie credits CSV file.
        keywords_path (str): Path to the movie keywords CSV file.
        merged_cache_path (str): Path to cache the processed merged dataset.

    Returns:
        pd.DataFrame: The processed metadata with merged data.
    """
    try:
        # Check if cached file exists
        if os.path.exists(merged_cache_path):
            logger.info(f"Found cached merged metadata at: {merged_cache_path}")
            return pd.read_csv(merged_cache_path)

        # Check if input files exist
        if not os.path.exists(metadata_path):
            logger.error(f"Error: {metadata_path} not found!")
            return None
        if not os.path.exists(credits_path):
            logger.error(f"Error: {credits_path} not found!")
            return None
        if not os.path.exists(keywords_path):
            logger.error(f"Error: {keywords_path} not found!")
            return None

        # Load datasets
        metadata = pd.read_csv(metadata_path, low_memory=False)
        credits = pd.read_csv(credits_path)
        keywords = pd.read_csv(keywords_path)

        # Clean metadata
        metadata = clean_metadata(metadata)

        # Merge datasets on 'id'
        metadata = metadata.merge(credits, on='id', how='left')
        metadata = metadata.merge(keywords, on='id', how='left')

        # Clean features: 'cast', 'crew', 'keywords', 'genres'
        metadata = clean_features(metadata)

        # Create 'soup' for content-based recommendation system
        metadata['soup'] = metadata.apply(create_soup, axis=1)

        # Save the cleaned and merged dataset to a file
        metadata.to_csv(merged_cache_path, index=False)
        logger.info(f"Merged metadata processed and saved to {merged_cache_path}")

        return metadata

    except Exception as e:
        logger.error(f"Error loading and processing datasets: {e}")
        raise