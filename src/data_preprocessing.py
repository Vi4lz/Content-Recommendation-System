import os
import pandas as pd

def load_all_csv_files(data_dir='../data'):
    """
    Function reads all CSV files in the given directory
    and returns a dictionary, where:
    - key: file name without the .csv extension
    - value: a specific pandas DataFrame
    :param data_dir:
    :return: datasets
    """

    datasets = {}

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            try:
                df = pd.read_csv(file_path)
                key_name = file_name.replace('.csv', '')
                datasets[key_name] = df
                print(f"Loaded: {file_name} ({df.shape[0]} row., {df.shape[1]} column.)")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    return datasets
