import numpy as np
import pandas as pd
from ast import literal_eval
from typing import Any, List, Union
from logging_config import setup_logging

logger = setup_logging()


def safe_literal_eval(val: Any) -> List:
    """
    A safer version of `literal_eval` that returns an empty list if parsing fails.

    Args:
        val (Any): A string expected to represent a Python literal (e.g., list of dicts).

    Returns:
        List: Parsed Python object (usually a list), or empty list on failure.
    """
    if isinstance(val, str):
        try:
            logger.debug(f"Attempting to safely evaluate: {val}")
            return literal_eval(val)
        except (ValueError, SyntaxError) as e:
            logger.error(f"Failed to parse value: {val}. Error: {e}")
            return []
    return []


def clean_data(x: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Cleans input data by removing spaces and converting text to lowercase.

    Args:
        x (Union[str, List[str]]): A string or list of strings.

    Returns:
        Union[str, List[str]]: Cleaned string or list of strings.
    """
    if isinstance(x, list):
        logger.debug(f"Cleaning list data: {x}")
        cleaned_data = [str.lower(i.replace(" ", "")) for i in x if isinstance(i, str)]
        logger.info(f"Cleaned list data: {cleaned_data}")
        return cleaned_data
    elif isinstance(x, str):
        logger.debug(f"Cleaning string data: {x}")
        cleaned_string = str.lower(x.replace(" ", ""))
        logger.info(f"Cleaned string data: {cleaned_string}")
        return cleaned_string
    else:
        logger.warning(f"Invalid data type for cleaning: {x}")
        return ''  # Return empty string for invalid types


def get_director(x: List[dict]) -> Union[str, float]:
    """
    Extracts the director's name from a crew list.

    Args:
        x (List[dict]): List of crew members with their roles.

    Returns:
        Union[str, float]: Name of the director if found, otherwise NaN.
    """
    logger.debug(f"Extracting director from crew list: {x}")
    for i in x:
        if i.get('job') == 'Director':
            logger.info(f"Found director: {i.get('name')}")
            return i.get('name')
    logger.warning("No director found in the crew list.")
    return np.nan


def get_list(x: Any) -> List[str]:
    """
    Extracts a list of names (e.g., actors or genres), limiting the output to top 3.

    Args:
        x (Any): A list of dictionaries expected to contain 'name' keys.

    Returns:
        List[str]: List of names (max 3). Empty list if input is invalid.
    """
    logger.debug(f"Extracting list from: {x}")
    if isinstance(x, list):
        names = [i['name'] for i in x if isinstance(i, dict) and 'name' in i]
        names = names[:3]  # Limit to top 3
        logger.info(f"Extracted names: {names}")
        return names
    logger.warning(f"Invalid list format: {x}")
    return []


def clean_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw movie metadata by removing duplicates.

    Args:
        metadata (pd.DataFrame): Raw metadata dataset.

    Returns:
        pd.DataFrame: Cleaned metadata dataset.
    """
    logger.info("Cleaning metadata by removing duplicates.")
    metadata = metadata.drop_duplicates(subset='id', keep='first')

    # Log number of duplicates removed
    logger.debug(f"Duplicates removed. Remaining entries: {metadata.shape[0]}")

    # Convert IDs to integers for merging
    metadata['id'] = metadata['id'].astype(int)

    logger.info("Metadata cleaning complete.")
    return metadata


def clean_features(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and processes specific feature columns: 'cast', 'crew', 'keywords', and 'genres'.

    Args:
        metadata (pd.DataFrame): Merged metadata dataset.

    Returns:
        pd.DataFrame: Metadata with processed features and added 'director' and cleaned 'soup' fields.
    """
    logger.info("Cleaning feature columns: 'cast', 'crew', 'keywords', 'genres'.")

    # Clean features: 'cast', 'crew', 'keywords', 'genres'
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        logger.debug(f"Cleaning feature: {feature}")
        metadata[feature] = metadata[feature].apply(safe_literal_eval)

    # Extract director from 'crew'
    logger.debug("Extracting directors from 'crew' feature.")
    metadata['director'] = metadata['crew'].apply(get_director)

    # Limit 'cast', 'keywords', 'genres' to top 3 entries
    for feature in ['cast', 'keywords', 'genres']:
        logger.debug(f"Limiting {feature} to top 3 entries.")
        metadata[feature] = metadata[feature].apply(get_list)

    # Clean data for text fields
    for feature in ['cast', 'keywords', 'director', 'genres']:
        logger.debug(f"Cleaning text data for feature: {feature}")
        metadata[feature] = metadata[feature].apply(clean_data)

    logger.info("Feature columns cleaned.")
    return metadata
