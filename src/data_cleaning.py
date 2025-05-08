import numpy as np

def get_director(x):
    """
    Extracts director's name from the crew list.
    """
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    """
    Returns a list of names, limiting to top 3.
    """
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

def clean_data(x):
    """
    Cleans input data by removing spaces and converting text to lowercase.
    """
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x if isinstance(i, str)]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        return ''  # return empty string for invalid types
