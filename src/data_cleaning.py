import numpy as np
import pandas as pd
from ast import literal_eval
from logging_config import setup_logging

logger = setup_logging()


def safe_literal_eval(val):
    """
    Safely evaluates a string representation of a Python literal structure (e.g. list, dict).

    Args:
        val (str): String to be evaluated.

    Returns:
        list: Parsed list if successful, otherwise an empty list.
    """
    if isinstance(val, str):
        try:
            return literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []


def clean_data(x):
    """
    Cleans input data by removing spaces and converting text to lowercase.

    Args:
        x (list or str): A list of strings or a single string to be cleaned.

    Returns:
        list or str: Cleaned list of strings or cleaned single string. Returns empty string for invalid input.
    """
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x if isinstance(i, str)]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        return ''  # return empty string for invalid types


def get_director(x):
    """
    Extracts the director's name from a list of crew members.

    Args:
        x (list): List of crew member dictionaries.

    Returns:
        str or np.nan: Name of the director if found, otherwise NaN.
    """
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    """
    Returns a list of names, limiting to top 3.

    Args:
        x (list): List of dictionaries containing 'name' key.

    Returns:
        list: List of up to 3 names.
    """
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


def clean_metadata(metadata):
    """
    Performs cleaning of the raw metadata dataset, e.g., dropping duplicates and handling the 'adult' column.

    Args:
        metadata (pd.DataFrame): Raw metadata dataset.

    Returns:
        pd.DataFrame: Cleaned metadata dataset.
    """
    # Drop duplicate movie entries based on 'id'
    metadata = metadata.drop_duplicates(subset='id', keep='first')

    # Clean 'adult' column values
    invalid_adult_values = metadata[~metadata['adult'].isin(['True', 'False'])]
    logger.info(f"Found {invalid_adult_values.shape[0]} rows with invalid 'adult' values.")
    metadata = metadata[metadata['adult'].isin(['True', 'False'])]

    # Convert IDs to integers for merging
    metadata['id'] = metadata['id'].astype(int)

    return metadata


def clean_features(metadata):
    """
    Performs feature-specific cleaning of 'cast', 'crew', 'keywords', 'genres' columns.

    Args:
        metadata (pd.DataFrame): Merged metadata dataset.

    Returns:
        pd.DataFrame: Cleaned metadata dataset with feature columns processed.
    """
    # Clean features: 'cast', 'crew', 'keywords', 'genres'
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(safe_literal_eval)

    # Extract director from 'crew'
    metadata['director'] = metadata['crew'].apply(get_director)

    # Limit 'cast', 'keywords', 'genres' to top 3 entries
    for feature in ['cast', 'keywords', 'genres']:
        metadata[feature] = metadata[feature].apply(get_list)

    # Clean data for text fields
    for feature in ['cast', 'keywords', 'director', 'genres']:
        metadata[feature] = metadata[feature].apply(clean_data)

    return metadata
