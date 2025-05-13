import pytest
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from src.recommender import get_top_movies, get_recommendations

def test_get_top_movies():
    test_data = {
        'title': ['Movie A', 'Movie B', 'Movie C'],
        'vote_count': [100, 500, 1000],
        'vote_average': [7.5, 8.2, 8.0],
    }
    df = pd.DataFrame(test_data)

    top_movies = get_top_movies(df, top_n=2)

    assert top_movies.shape[0] == 2
    assert 'weighted_rating' in top_movies.columns
    assert top_movies['title'].iloc[0] == 'Movie B'
    print("Top movies test passed.")


def test_get_recommendations():
    test_data = {
        'title': ['Movie A', 'Movie B', 'Movie C'],
        'soup': ['action adventure', 'romance drama', 'action thriller'],
    }
    df = pd.DataFrame(test_data)

    count_matrix = csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 0]])

    nn_model = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine')
    nn_model.fit(count_matrix)

    indices = pd.Series(df.index, index=df['title'])

    recommendations = get_recommendations('Movie A', nn_model, df, indices, count_matrix)

    assert recommendations.shape[0] > 0
    print("Recommendations test passed.")


def test_non_existing_movie():
    test_data = {
        'title': ['Movie A', 'Movie B', 'Movie C'],
        'soup': ['action adventure', 'romance drama', 'action thriller'],
    }
    df = pd.DataFrame(test_data)

    count_matrix = csr_matrix([[1, 0, 1], [0, 1, 0], [1, 0, 0]])

    nn_model = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine')
    nn_model.fit(count_matrix)

    indices = pd.Series(df.index, index=df['title'])

    recommendations = get_recommendations('Nonexistent Movie', nn_model, df, indices, count_matrix)

    assert recommendations.empty
    print("Non-existing movie test passed.")

