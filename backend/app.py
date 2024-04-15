import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import ast
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))

json_file_path = os.path.join(current_directory, 'init.json')

with open(json_file_path, 'r') as file:
    data = json.load(file)
    movies_df = pd.DataFrame(data)

app = Flask(__name__)
CORS(app)

def json_search(query):
    matches = movies_df[movies_df['title'].str.lower().str.contains(query.lower()) | movies_df['original_title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'overview', 'vote_average']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

def genre_search(genre):
    matches = movies_df[movies_df['genres'].apply(lambda g: genre.lower() in (genre_name.lower() for genre_name in g))]
    matches_filtered = matches[['title', 'overview', 'vote_average']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

def filter_movies_by_genre(genre):
    def is_genre_present(genres_str, genre):
        try:
            genres_list = ast.literal_eval(genres_str)
            for genre_dict in genres_list:
                if genre.lower() == genre_dict['name'].lower():
                    return True
        except ValueError:
            return False
        return False

    matches = movies_df[movies_df['genres'].apply(lambda g: is_genre_present(g, genre))]
    matches_filtered = matches[['title', 'overview', 'vote_average']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

def compute_similarities(overviews, query):
    combined_texts = overviews + [query]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    cosine_similarities = linear_kernel(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    similarity_scores = [(score, idx) for idx, score in enumerate(cosine_similarities)]
    similarity_scores = sorted(similarity_scores, reverse=True)
    top_matches = similarity_scores[:50]
    return top_matches

def tokenize(text):
    if text is None:
        return ""
    return text.lower().split()

def build_vocabulary(descriptions):
    vocab = set()
    for description in descriptions:
        vocab.update(tokenize(description))
    return list(vocab)

def vectorize(text, vocabulary):
    word_counts = {word: 0 for word in vocabulary}
    for word in tokenize(text):
        if word in word_counts:
            word_counts[word] += 1
    return [word_counts[word] for word in vocabulary]

def cosine_similarity(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v**2 for v in vec1))
    magnitude2 = math.sqrt(sum(v**2 for v in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def get_genre_frequencies(dataframe):
    genre_counts = {}
    for index, row in dataframe.iterrows():
        genres = ast.literal_eval(row['genres'])
        for genre in genres:
            genre_name = genre['name']
            if genre_name in genre_counts:
                genre_counts[genre_name] += 1
            else:
                genre_counts[genre_name] = 1
    return genre_counts

def genresuggests(genre):
    genres = set()
    for item in data:
        genre_list = json.loads(item['genres'].replace("'", "\""))
        for genre in genre_list:
            if genre in genre['name'].lower():
                genres.add(genre['name'])

    return jsonify(list(genres))

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    query= request.args.get("query")
    json_text= json.loads((filter_movies_by_genre(text)))
    overviews = [overview['overview'] if overview['overview'] is not None else "" for overview in json_text]
    texts = overviews + ([query] if query is not None else [''])
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_similarities = linear_kernel(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    movie_scores = list(enumerate(cosine_similarities))
    sorted_movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
    top_half_movie_indices = [index for index, score in sorted_movie_scores[:len(sorted_movie_scores) // 2]]
    filtered_movies = [json_text[index] for index in top_half_movie_indices]
    return filtered_movies

@app.route('/genre_suggestions')
def genre_suggestions():
    query = request.args.get('query', '').lower()
    threshold = 10
    genre_counts = get_genre_frequencies(movies_df)
    filtered_genres = {genre for genre, count in genre_counts.items() if count >= threshold}

    matching_genres = set()
    for genre in filtered_genres:
        if query in genre.lower():
            matching_genres.add(genre)

    return jsonify(list(matching_genres))

@app.route('/predict_preference')
def predict_preference():
    liked_genres_query = request.args.get('liked_genres', '')
    disliked_genres_query = request.args.get('disliked_genres', '')

    liked_genres_list = liked_genres_query.split(',')
    disliked_genres_list = disliked_genres_query.split(',')
    try:
        nb_model, genre_encoder = load_model()
        user_vector = preprocess_user_input(liked_genres_list, disliked_genres_list, genre_encoder)
        prob_like = nb_model.predict_proba([user_vector])[0, 1]
        return jsonify({'like_probability': prob_like})
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)})

@app.route('/recommended_movies')
def recommended_movies():
    like_probability = float(request.args.get('like_probability', 0))

    print(like_probability)

    threshold = 0.5

    if like_probability >= threshold:
        recommended_movies = movies_df[movies_df['vote_average'] >= threshold*10]
        recommended_movies = recommended_movies[['title', 'overview', 'vote_average']]
    else:
        recommended_movies = movies_df[movies_df['vote_average'] < threshold*10]
        recommended_movies = recommended_movies[['title', 'overview', 'vote_average']]

    recommended_movies_json = recommended_movies.to_json(orient='records')

    return jsonify(recommended_movies_json)

def load_model():
    X, _, genre_encoder = preprocess_genres(movies_df, threshold=10)
    y = simulate_user_preferences(movies_df)
    nb_model = train_naive_bayes(X, y)
    return nb_model, genre_encoder

def preprocess_user_input(liked_genres, disliked_genres, genre_encoder):
    user_vector = np.zeros(len(genre_encoder.classes_))
    valid_genres = set(genre_encoder.classes_)

    for genre in liked_genres:
        if genre in valid_genres:
            user_vector[genre_encoder.transform([[genre]])[0]] = 1

    for genre in disliked_genres:
        if genre in valid_genres:
            user_vector[genre_encoder.transform([[genre]])[0]] = -1

    return user_vector

def preprocess_genres(dataframe, threshold=10):
    genre_counts = get_genre_frequencies(dataframe)

    mlb = MultiLabelBinarizer()
    genres = dataframe['genres'].apply(lambda x: [genre['name'] for genre in ast.literal_eval(x) if genre_counts.get(genre['name'], 0) >= threshold])

    genres_encoded = mlb.fit_transform(genres)

    return genres_encoded, mlb.classes_, mlb

def simulate_user_preferences(dataframe, like_threshold=7.0):
    likes = dataframe['vote_average'].apply(lambda x: 1 if x >= like_threshold else 0)
    return likes

def train_naive_bayes(X, y):
    model = BernoulliNB()
    model.fit(X, y)
    return model

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)