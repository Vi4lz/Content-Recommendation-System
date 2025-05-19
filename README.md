# 🎓 Content Recommendation System

## 🎯 Project Goal

This project is a **content-based movie recommendation system** that:

- Analyzes movie features (genres, cast, director, keywords)
- Recommends similar movies based on a selected input movie
- Displays both similar movies and top-rated movies using a weighted IMDb-style score
- Provides a user-friendly web interface built with Django

---

📦 The program automatically unpacks `raw_data.zip` and processes the CSV files (e.g. `movies_metadata.csv`, `credits.csv`, `keywords.csv`) on first execution. Make sure the archive is present inside the `data/` directory.


## 🧱 Project Structure

ContentRecommendationSystem/
│
├── data/ → CSV files and pre-trained models (joblib)
├── src/ → Core logic: preprocessing, recommendation engine, utilities
│ ├── recommender.py
│ ├── data_preprocessing.py
│ ├── engine.py
│ └── utils.py
├── content_recommendation_system/
│ ├── recommendations/ → Django application (views, forms, templates)
│ └── manage.py → Django management script


---

## ⚙️ How It Works

### 1. Data Processing

- Merges data from `movies_metadata.csv`, `credits.csv`, and `keywords.csv`
- Cleans inaccurate entries and duplicates
- Extracts relevant features and creates a unified **"soup"** column combining:
  - Genres
  - Cast
  - Director
  - Keywords

---

### 2. IMDb-Style Weighted Rating

A weighted rating formula is used to prioritize both the **quantity** and **quality** of votes:

WR = (v / (v + m)) * R + (m / (v + m)) * C


Where:
- **v** = number of votes for the movie  
- **R** = average rating of the movie  
- **C** = mean rating across all movies  
- **m** = minimum votes required to be listed (defined by a percentile threshold)

✅ This produces a reliable **Top 100 movies** list based on quality.

---

### 3. Content-Based Recommendation

- Uses `CountVectorizer` to convert the soup column into a count matrix
- Applies `NearestNeighbors` with cosine similarity to identify similar movies
- Given an input movie, the system returns the top 10 most similar titles

---

### 4. Fuzzy Matching with FuzzyWuzzy

- Supports user input tolerance (e.g. typos or partial names)
- Displays the closest matching titles if no exact match is found

---

### 5. Django Web Interface

- `home.html`: Homepage with movie search bar
- `matches.html`: Displays possible matches from user input
- `recommendations.html`: Displays similar recommended movies
- `top_movies.html`: Shows Top 100 highest rated movies based on IMDb-style formula

---

## 🚀 How to Run the Project

Follow these steps to set up and run the project locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vi4lz/Content-Recommendation-System.git
   cd Content-Recommendation-System

2. **Create and activate a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate        # Linux/Mac
    venv\Scripts\activate           # Windows
   
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt

4. **Run Django server**
    ```bash
    cd content_recommendation_system
    python manage.py runserver

5. **Acces in your browser**
    http://127.0.0.1:8000    

Ensure you have raw_data.zip placed in the data/ folder. 
The program will automatically extract the contents on first run.

➡️ If the file structure looks like this:

data/
└── raw_data.zip

You’re good to go! No need to manually extract anything.


📦 Requirements
Ensure these packages are installed (via requirements.txt):

* pandas
* numpy
* scikit-learn
* joblib
* fuzzywuzzy
* python-Levenshtein
* nltk
* Django>=3.0


🧠 Technologies Used
*  Python for backend logic and modeling

*  pandas and NumPy for data preprocessing

*  scikit-learn for vectorization and Nearest Neighbors model

*  fuzzywuzzy for tolerant text matching

*  Django for the web interface

*  joblib for saving/loading models


🔍 Example Features
*  Recommends: “The Matrix” → “Ghost in the shell”, “I, Robot”, “The Matrix Reloaded”

*  Handles typos: “Godfatr” → suggests “The Godfather”

*  Real-time top movies: Top 100 IMDb-style page


🤖 Author

Built by Šarūnas Tumasonis as a final 'Python' course project - ""Content Recommendation System"