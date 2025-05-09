import os
import pandas as pd
from ast import literal_eval
from data_cleaning import clean_data, get_list, get_director
from logging_config import setup_logging

logger = setup_logging()

def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []

# Function to create 'soup' by combining cast, keywords, director, and genres
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def load_and_merge_metadata(metadata_path, credits_path, keywords_path, merged_cache_path='merged_metadata.parquet'):
    try:
        # Patikrinimas ar mes jau turime apdorotą failą
        if os.path.exists(merged_cache_path):
            logger.info(f"Found cached merged metadata at: {merged_cache_path}")
            return pd.read_csv(merged_cache_path)

        # Patikrinimai, ar visi reikalingi failai egzistuoja
        if not os.path.exists(metadata_path):
            logger.error(f"Error: {metadata_path} not found!")
            return None
        if not os.path.exists(credits_path):
            logger.error(f"Error: {credits_path} not found!")
            return None
        if not os.path.exists(keywords_path):
            logger.error(f"Error: {keywords_path} not found!")
            return None

        # Įkeliame duomenis
        metadata = pd.read_csv(metadata_path, low_memory=False)
        credits = pd.read_csv(credits_path)
        keywords = pd.read_csv(keywords_path)

        # Pašalinti nereikalingas eilutes pagal 'id' (pašalinsime eilutes su netinkamais 'adult' stulpeliais)
        # Pirmiausia, pašalinsime pasikartojančius 'id'
        metadata = metadata.drop_duplicates(subset='id', keep='first')

        # Patikrinkime, kurios eilutės turi netinkamas reikšmes 'adult' stulpelyje
        invalid_adult_values = metadata[~metadata['adult'].isin(['True', 'False'])]
        logger.info(f"Found {invalid_adult_values.shape[0]} rows with invalid 'adult' values.")

        # Pašalinti tas eilutes, kur 'adult' stulpelyje nėra 'True' arba 'False'
        metadata = metadata[metadata['adult'].isin(['True', 'False'])]

        # Konvertuojame id į int, kad galėtume atlikti sujungimus
        keywords['id'] = keywords['id'].astype('int')
        credits['id'] = credits['id'].astype('int')
        metadata['id'] = metadata['id'].astype('int')

        # Sujungiame duomenis
        metadata = metadata.merge(credits, on='id', how='left')
        metadata = metadata.merge(keywords, on='id', how='left')

        # Apdorojame tekstinius duomenis, kad gautume reikalingus laukus
        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            metadata[feature] = metadata[feature].apply(safe_literal_eval)

        # Gaukite režisierių iš "crew" sąrašo
        metadata['director'] = metadata['crew'].apply(get_director)

        # Paverčiame "cast", "keywords", "genres" į sąrašus su maksimaliai 3 elementais
        for feature in ['cast', 'keywords', 'genres']:
            metadata[feature] = metadata[feature].apply(get_list)

        # Pateikiame visą tekstą mažosiomis raidėmis ir pašaliname nereikalingas tarpus
        for feature in ['cast', 'keywords', 'director', 'genres']:
            metadata[feature] = metadata[feature].apply(clean_data)

        # Sukuriame 'soup' stulpelį, kuris bus naudojamas rekomendacijų sistemoje
        metadata['soup'] = metadata.apply(create_soup, axis=1)

        # Konvertuojame vote_count ir vote_average į teisingus tipus (numerius)
        metadata['vote_count'] = pd.to_numeric(metadata['vote_count'], errors='coerce').fillna(0).astype(int)
        metadata['vote_average'] = pd.to_numeric(metadata['vote_average'], errors='coerce').fillna(0.0)

        # Išsaugome sujungtą ir paruoštą metadata į failą
        metadata.to_csv(merged_cache_path, index=False)
        logger.info(f"Merged metadata processed and saved to {merged_cache_path}")

        return metadata

    except Exception as e:
        logger.error(f"Error loading and processing datasets: {e}")
        raise
