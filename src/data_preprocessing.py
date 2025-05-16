import os
import pandas as pd
import zipfile
from typing import List, Optional
from data_cleaning import clean_data, get_list, get_director, clean_metadata, clean_features
from logging_config import setup_logging

logger = setup_logging()


def ensure_unzipped(zip_path: str, extract_dir: str, expected_files: List[str]) -> None:
    """
    Ensures the required CSV files are extracted from a ZIP archive.

    If any expected file is missing from the target directory, this function attempts to
    extract them from the specified ZIP file.

    Args:
        zip_path (str): Full path to the ZIP archive (e.g. 'data/raw_data.zip').
        extract_dir (str): Directory where files should be extracted to.
        expected_files (List[str]): List of filenames expected to be present after extraction.

    Returns:
        None

    Raises:
        FileNotFoundError: If the ZIP archive does not exist.
        Exception: If any error occurs during extraction.
    """
    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(extract_dir, f))]
    if missing_files:
        if not os.path.exists(zip_path):
            logger.error(f"ZIP archive not found: {zip_path}")
            raise FileNotFoundError(f"ZIP archive not found: {zip_path}")

        logger.info(f"Extracting missing files from: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info("Extraction complete.")
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise


def create_soup(x: pd.Series) -> str:
    """
    Concatenates relevant movie features into a single string ("soup") for vectorization.

    Args:
        x (pd.Series): A row from the movie metadata DataFrame.

    Returns:
        str: Concatenated string of keywords, cast, director, and genres.
    """
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


def load_and_merge_metadata(
    metadata_path: str,
    credits_path: str,
    keywords_path: str,
    merged_cache_path: str = 'merged_metadata.csv'
) -> Optional[pd.DataFrame]:
    """
    Loads and merges movie metadata, credits, and keywords datasets.
    If a cached version exists, it is returned. Otherwise, raw data is processed and saved.

    Args:
        metadata_path (str): Path to the movie metadata CSV file.
        credits_path (str): Path to the movie credits CSV file.
        keywords_path (str): Path to the movie keywords CSV file.
        merged_cache_path (str, optional): Path where the merged dataset will be saved. Defaults to 'merged_metadata.csv'.

    Returns:
        Optional[pd.DataFrame]: The merged and processed movie metadata,
        or None if loading fails.
    """
    try:
        # Extract ZIP if necessary
        zip_path = os.path.join(os.path.dirname(metadata_path), 'raw_data.zip')
        extract_dir = os.path.dirname(metadata_path)
        expected_files = [
            os.path.basename(metadata_path),
            os.path.basename(credits_path),
            os.path.basename(keywords_path)
        ]
        ensure_unzipped(zip_path, extract_dir, expected_files)

        # Use cache if available
        if os.path.exists(merged_cache_path):
            logger.info(f"Found cached merged metadata at: {merged_cache_path}")
            return pd.read_csv(merged_cache_path)

        # Check if individual files exist
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}")
            return None
        if not os.path.exists(credits_path):
            logger.error(f"Credits file not found: {credits_path}")
            return None
        if not os.path.exists(keywords_path):
            logger.error(f"Keywords file not found: {keywords_path}")
            return None

        # Load data
        metadata = pd.read_csv(metadata_path, low_memory=False)
        credits = pd.read_csv(credits_path)
        keywords = pd.read_csv(keywords_path)

        # Clean and merge
        metadata = clean_metadata(metadata)
        metadata = metadata.merge(credits, on='id', how='left')
        metadata = metadata.merge(keywords, on='id', how='left')

        metadata = clean_features(metadata)
        metadata['soup'] = metadata.apply(create_soup, axis=1)

        # Save merged version
        metadata.to_csv(merged_cache_path, index=False)
        logger.info(f"Merged metadata processed and saved to {merged_cache_path}")

        return metadata

    except Exception as e:
        logger.error(f"Error loading and processing datasets: {e}")
        raise
