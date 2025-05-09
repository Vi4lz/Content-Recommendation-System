import os
import pandas as pd


def run_diagnostics(metadata):
    print("=== Dataset Overview ===")
    print(f"Rows: {metadata.shape[0]}, Columns: {metadata.shape[1]}")

    print("\n=== Column Types ===")
    print(metadata.dtypes)

    print("\n=== Missing Values per Column ===")
    print(metadata.isnull().sum().sort_values(ascending=False))

    print("\n=== Unique Values in Selected Columns ===")
    for col in ['title', 'genres', 'adult', 'release_date']:
        if col in metadata.columns:
            print(f"{col}: {metadata[col].nunique()} unique values")

    if 'adult' in metadata.columns:
        print("\n=== 'adult' Value Distribution ===")
        print(metadata['adult'].value_counts(dropna=False))

    if 'genres' in metadata.columns:
        print("\n=== Genre Frequency ===")
        if metadata['genres'].apply(lambda x: isinstance(x, list)).any():
            genre_counts = pd.Series([g for sublist in metadata['genres'] for g in sublist])
            print(genre_counts.value_counts().head(10))

    if 'vote_average' in metadata.columns:
        print("\n=== Vote Average Statistics ===")
        print(metadata['vote_average'].describe())

    print("\n=== Sample Entries ===")
    print(metadata.sample(5)[['title', 'adult', 'release_date']])

    print("\n=== Rows with Missing Values ===")
    print(metadata[metadata.isnull().any(axis=1)].head())

    if 'id' in metadata.columns:
        if metadata['id'].duplicated().any():
            print("Duplicate IDs found:")
            print(metadata[metadata['id'].duplicated(keep=False)][['id', 'title']])
        else:
            print("All IDs are unique.")

    if 'vote_average' in metadata.columns:
        print("\n=== Invalid Vote Averages (Out of Bounds) ===")
        outliers = metadata[(metadata['vote_average'] < 0) | (metadata['vote_average'] > 10)]
        print(outliers[['title', 'vote_average']])

    if 'release_date' in metadata.columns:
        metadata['release_date'] = pd.to_datetime(metadata['release_date'], errors='coerce')
        print("\n=== Invalid or Missing Release Dates ===")
        print(metadata[metadata['release_date'].isnull()][['title', 'release_date']])


if __name__ == "__main__":
    # Automatically determine the correct path to the data file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "../data/movies_metadata.csv")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found at: {DATA_PATH}")
    else:
        try:
            metadata = pd.read_csv(DATA_PATH, low_memory=False)
            run_diagnostics(metadata)
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
