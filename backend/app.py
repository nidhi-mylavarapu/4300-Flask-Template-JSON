import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import ast

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    # episodes_df = pd.DataFrame(data['episodes'])
    # reviews_df = pd.DataFrame(data['reviews'])
    movies_df = pd.DataFrame(data)

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    # matches = []
    # merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    # matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    # matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    # matches_filtered_json = matches_filtered.to_json(orient='records')
    # return matches_filtered_json
    matches = movies_df[movies_df['title'].str.lower().str.contains(query.lower()) | movies_df['original_title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'overview', 'vote_average']]  # Adjusted to match relevant fields in the new JSON
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


import math

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


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    query= request.args.get("query")
    json_text= json.loads((json_search(text)))
    overviews=[]
    for movie in json_text: 
        overviews.append(movie['overview'])
    vocabulary = build_vocabulary(overviews + [query])
    movie_vectors = [vectorize(desc, vocabulary) for desc in overviews]
    user_vector = vectorize(query, vocabulary)
    similarities = {desc: cosine_similarity(user_vector, movie_vector) for desc, movie_vector in zip(overviews, movie_vectors)}
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    print(len(sorted_similarities))
    print("before")
    top_50_percent = sorted_similarities[:len(sorted_similarities) // 2]
    print(len(top_50_percent))
    print("after")
    top_descriptions_set = set(desc for desc, _ in top_50_percent)
    filtered_movies = [movie for movie in json_text if movie["overview"] in top_descriptions_set]
    return filtered_movies

@app.route('/genre_suggestions')
def genre_suggestions():
    query = request.args.get('query', '').lower()
    genres = set()

    for item in data:
        genre_list = json.loads(item['genres'].replace("'", "\""))
        for genre in genre_list:
            if query in genre['name'].lower():
                genres.add(genre['name'])

    return jsonify(list(genres))

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)

