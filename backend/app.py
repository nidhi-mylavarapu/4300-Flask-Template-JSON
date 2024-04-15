import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report



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
    matches_filtered = matches[['title', 'overview', 'vote_average', 'reviews']]  # Adjusted to match relevant fields in the new JSON
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

import math
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
def genresuggests(genre): 
    genres = set()
    for item in data:
        genre_list = json.loads(item['genres'].replace("'", "\""))
        for genre in genre_list:
            if genre in genre['name'].lower():
                genres.add(genre['name'])

    return jsonify(list(genres))


train_data = pd.read_csv('train.tsv.zip', sep='\t')
train_data['full_sentence'] = train_data.groupby('SentenceId')['Phrase'].transform(lambda x: ' '.join(x))
train_data = train_data.drop_duplicates('SentenceId').reset_index(drop=True)

model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
model.fit(train_data['full_sentence'], train_data['Sentiment'])

def classify_and_score_reviews(movies_df):
    results = []
    for index, row in movies_df.iterrows():
        movie_reviews = row['reviews']
        if movie_reviews:
            sentiments = model.predict(movie_reviews)
            if sentiments.size > 0:
                average_sentiment = sentiments.mean()
                results.append((row['title'], average_sentiment))
            else:
                results.append((row['title'], None))
        else:
            results.append((row['title'], None))
    return dict(results)


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    query= request.args.get("query")
    print("before")
    movies_df = pd.read_json('init.json')
    if 'reviews' not in movies_df.columns:
        movies_df['reviews'] = [[] for _ in range(len(movies_df))]

    sentiment_scores = classify_and_score_reviews(movies_df)


    json_text= json.loads((filter_movies_by_genre(text)))
    if 'reviews' not in movies_df.columns:
        movies_df['reviews'] = [[] for _ in range(len(movies_df))]
    overviews = [overview['overview'] if overview['overview'] is not None else "" for overview in json_text]
    texts = overviews + ([query] if query is not None else [''])
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_similarities = linear_kernel(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    movie_scores = list(enumerate(cosine_similarities))
    sorted_movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
    # top_half_movie_indices = [index for index, score in sorted_movie_scores[:len(sorted_movie_scores) // 2]]
    # filtered_movies = [json_text[index] for index in top_half_movie_indices]

    combined_scores = [(cosine_similarities[i], sentiment_scores[movies_df.iloc[i]['title']]) for i in range(len(movies_df))]

    combined_scores_sorted = sorted(combined_scores, key=lambda x: (x[0] + (x[1] if x[1] is not None else 0)), reverse=True)
    top_half_indices = [index for index, score in combined_scores_sorted[:len(combined_scores_sorted) // 2]]

    filtered_movies = [json_text[index] for index in top_half_indices]
    return filtered_movies


    # overviews=[]
    # for movie in json_text: 
    #     overviews.append(movie['overview'])
    
    # vocabulary = build_vocabulary(overviews + [query])
    # movie_vectors = [vectorize(desc, vocabulary) for desc in overviews]
    # user_vector = vectorize(query, vocabulary)
    # similarities = {desc: cosine_similarity(user_vector, movie_vector) for desc, movie_vector in zip(overviews, movie_vectors)}
    # sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    # print(len(sorted_similarities))
    # top_50_percent = sorted_similarities[:len(sorted_similarities) // 2]
    # print(len(top_50_percent))
    # top_descriptions_set = set(desc for desc, _ in top_50_percent)
    # filtered_movies = [movie for movie in json_text if movie["overview"] in top_descriptions_set]
    # return filtered_movies

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

