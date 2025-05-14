from django.shortcuts import render
from .forms import MovieSearchForm
from engine import get_matches, get_recommendations_by_title, get_top_rated_movies

def home(request):
    form = MovieSearchForm()
    return render(request, 'recommendations/home.html', {'form': form})


def matches(request):
    form = MovieSearchForm()
    matches = []
    query = ""
    message = ""

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


def recommend(request):
    title = request.GET.get('title')
    if not title:
        return render(request, 'recommendations/home.html', {'form': MovieSearchForm()})

    matches_df = get_matches(title)
    if matches_df.empty:
        return render(request, 'recommendations/recommendations.html', {
            'title': title,
            'recommendations': [],
            'message': "No similar titles found."
        })

    best_match = matches_df.iloc[0]['title']  # pirmas, geriausiai atitikęs
    recommendations = get_recommendations_by_title(best_match)

    return render(request, 'recommendations/recommendations.html', {
        'title': best_match,
        'recommendations': recommendations[['title', 'release_date', 'genres']].to_dict(orient='records')
    })


def top_movies(request):
    top_movies_df = get_top_rated_movies()
    top_movies = top_movies_df.to_dict(orient='records')

    return render(request, 'recommendations/top_movies.html', {
        'top_movies': top_movies
    })
