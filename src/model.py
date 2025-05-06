import pandas as pd

def get_top_movies(df, top_n=10, percentile=0.90):
    """
     Returns the top N movies ranked by IMDb-style weighted rating.

    Parameters:
        df (pd.DataFrame): Aggregated movie ratings with 'vote_count' and 'vote_average'.
        top_n (int): Number of top movies to return.
        percentile (float): Percentile to determine minimum vote count (m).

    Returns:
        pd.DataFrame: Top N movies sorted by weighted rating.
    """
    if df.empty:
        print("Input DataFrame is empty.")
        return pd.DataFrame()

    C = df['vote_average'].mean()    # mean rating across all movies
    m = df['vote_count'].quantile(percentile)   # number of votes received by a movie in the percentile param.

    has_enough_votes = df['vote_count'] >= m
    qualified = df[has_enough_votes].copy()   # new independent df with calculations.

    if qualified.empty:
        print("No movies meet the minimum vote count threshold.")
        return pd.DataFrame()

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v+m) * R) + (m / (v + m) * C)   # imdb weighted rating formula.

    qualified['weighted_rating'] = qualified.apply(weighted_rating, axis=1)
    top_movies = qualified.sort_values('weighted_rating', ascending=False).head(top_n)

    return top_movies[['title', 'vote_count', 'vote_average', 'weighted_rating']]