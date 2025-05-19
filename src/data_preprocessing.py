import os
import pandas as pd
import zipfile
from data_cleaning import clean_data, get_list, get_director, clean_metadata, clean_features
from logging_config import setup_logging

logger = setup_logging()

def extract_raw_data(zip_path, extract_to):
    """
    Extracts raw_data.zip if not already extracted.

    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory to extract files to.

    Returns:
        Extracted expected files.
    """
    try:
        # Check if already extracted (by checking for one of the expected files)
        expected_files = ['keywords.csv', 'credits.csv', 'movies_metadata.csv']
        all_exist = all(os.path.exists(os.path.join(extract_to, f)) for f in expected_files)

        if not all_exist:
            if os.path.exists(zip_path):
                logger.info(f"Extracting {zip_path} to {extract_to}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                logger.info("Extraction complete.")
            else:
                logger.error(f"{zip_path} not found!")
                raise FileNotFoundError(f"{zip_path} not found!")
        else:
            logger.info("Raw data already extracted.")
    except Exception as e:
        logger.error(f"Failed to extract raw data: {e}")
        raise


# Function to create 'soup' by combining cast, keywords, director, and genres
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


def load_and_merge_metadata(metadata_path, credits_path, keywords_path, merged_cache_path='merged_metadata.csv', zip_path=None, extract_to=None):
    """
    Loads and merges movie metadata, credits, and keywords datasets.
    Performs data merging and stores the processed dataset.

    Args:
        metadata_path (str): Path to the movie metadata CSV file.
        credits_path (str): Path to the movie credits CSV file.
        keywords_path (str): Path to the movie keywords CSV file.
        merged_cache_path (str): Path to cache the processed merged dataset.
        zip_path (str): Path to the zip file (if data needs extraction).
        extract_to (str): Directory to extract files to (if data needs extraction).

    Returns:
        pd.DataFrame: The processed metadata with merged data.
    """
    try:
        if zip_path and extract_to:
            extract_raw_data(zip_path, extract_to)

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
