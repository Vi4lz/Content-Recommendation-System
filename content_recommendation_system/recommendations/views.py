from typing import List, Dict
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from .forms import MovieSearchForm
from engine import get_matches, get_recommendations_by_title, get_top_rated_movies


def home(request: HttpRequest) -> HttpResponse:
    """
    Render the homepage with a movie search form.

    Args:
        request (HttpRequest): The incoming HTTP request.

    Returns:
        HttpResponse: Rendered homepage with search form.
    """
    form = MovieSearchForm()
    return render(request, 'recommendations/home.html', {'form': form})


def matches(request: HttpRequest) -> HttpResponse:
    """
    Handle movie title search and return matching results using fuzzy search.

    Args:
        request (HttpRequest): The incoming HTTP request.

    Returns:
        HttpResponse: Rendered results page with list of similar movie titles or a message if none found.
    """
    form = MovieSearchForm()
    matches: List[Dict] = []
    query: str = ""
    message: str = ""

    if request.method == 'POST':
        form = MovieSearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['title']
            results = get_matches(query)
            results = results.drop_duplicates(subset='title', keep='first')

            if not results.empty:
                matches = results.to_dict(orient='records')
            else:
                message = "No matches found."

    return render(request, 'recommendations/matches.html', {
        'form': form,
        'query': query,
        'matches': matches,
        'message': message,
    })


def recommend(request: HttpRequest) -> HttpResponse:
    """
    Generate and display movie recommendations based on the most similar title.

    Args:
        request (HttpRequest): The incoming HTTP request.

    Returns:
        HttpResponse: Rendered recommendations page with similar movies.
    """
    title: str = request.GET.get('title', '')

    if not title:
        return render(request, 'recommendations/home.html', {'form': MovieSearchForm()})

    matches_df = get_matches(title)
    if matches_df.empty:
        return render(request, 'recommendations/recommendations.html', {
            'title': title,
            'recommendations': [],
            'message': "No similar titles found."
        })

    best_match: str = matches_df.iloc[0]['title']
    recommendations = get_recommendations_by_title(best_match)

    return render(request, 'recommendations/recommendations.html', {
        'title': best_match,
        'recommendations': recommendations[['title', 'release_date', 'genres']].to_dict(orient='records')
    })


def top_movies(request: HttpRequest) -> HttpResponse:
    """
    Display a list of the top 100 highest-rated movies based on IMDb-style weighted rating.

    Args:
        request (HttpRequest): The incoming HTTP request.

    Returns:
        HttpResponse: Rendered page with top 100 movies.
    """
    top_movies_df = get_top_rated_movies()
    top_movies: List[Dict] = top_movies_df.to_dict(orient='records')

    return render(request, 'recommendations/top_movies.html', {
        'top_movies': top_movies
    })
