import logging
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

logger = logging.getLogger(__name__)

def compute_tfidf_matrix(df, column='overview'):
    """
    Computes a TF-IDF matrix for the specified text column in the given DataFrame.

    This function transforms a column of text data (e.g., tags or descriptions)
    into a matrix of TF-IDF features, ignoring common English stop words.
    Returns:
        scipy.sparse.csr.csr_matrix: Sparse matrix of TF-IDF features.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df[column] = df[column].fillna('')
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        matrix = tfidf.fit_transform(df[column])
        logger.info(f"TF-IDF matrix created. Shape: {matrix.shape}")
        return matrix
    except Exception as e:
        logger.error(f"TF-IDF error: {str(e)}")
        raise

def compute_count_vector_matrix(df, column='soup'):
    """
    Computes a CountVectorizer matrix for the specified text column.
    Returns:
        scipy.sparse.csr.csr_matrix: Count-based feature matrix.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df[column] = df[column].fillna('')
    try:
        count = CountVectorizer(stop_words='english')
        matrix = count.fit_transform(df[column])
        logger.info(f"Count matrix created. Shape: {matrix.shape}")
        return matrix
    except Exception as e:
        logger.error(f"Count Vectorizer error: {str(e)}")
        raise

def compute_cosine_similarity_matrix(matrix, use_linear_kernel=True):
    """
    Computes a cosine similarity matrix from a given vectorized feature matrix.
    Parameters:
        matrix (scipy.sparse matrix): Vectorized feature matrix (TF-IDF or Count).
        use_linear_kernel (bool): Whether to use linear_kernel (default) or cosine_similarity.
    Returns:
        ndarray: Cosine similarity matrix.
    """
    try:
        if matrix is None or matrix.shape[0] == 0:
            raise ValueError("Input matrix is empty or None.")

        if use_linear_kernel:
            return linear_kernel(matrix, matrix)
        else:
            return cosine_similarity(matrix, matrix, dense_output=False)
    except Exception as e:
        logger.error(f"Cosine similarity computation failed: {e}")
        raise