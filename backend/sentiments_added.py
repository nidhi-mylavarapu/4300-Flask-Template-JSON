import json
import os
import ast
import sklearn as sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

train_data = pd.read_csv('train.tsv.zip', sep='\t')
train_data['full_sentence'] = train_data.groupby('SentenceId')['Phrase'].transform(lambda x: ' '.join(x))
train_data = train_data.drop_duplicates('SentenceId').reset_index(drop=True)

model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
model.fit(train_data['full_sentence'], train_data['Sentiment'])

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# Define the function to process JSON data and append sentiment scores
def classify_and_score_reviews(json_data, model):
    # Load the JSON data into a DataFrame
    movies_df = pd.read_json(json_data)

    # Initialize a new column for sentiment scores
    movies_df['sentiment_score'] = None

    # Iterate over each movie and calculate sentiment scores
    for index, row in movies_df.iterrows():
        movie_reviews = row['reviews']
        if movie_reviews:
            sentiments = model.predict(movie_reviews)
            if len(sentiments) > 0:
                average_sentiment = np.mean(sentiments)
                movies_df.at[index, 'sentiment_score'] = average_sentiment

    # Return the modified DataFrame as a JSON string
    return movies_df.to_json(orient='records')


def main(input_json_path, output_json_path):
    # Read the original JSON data
    original_data = read_json_file(input_json_path)
    
    # Convert the original data to a JSON string since our function expects it
    original_json_string = json.dumps(original_data)
    
    # Classify and score reviews
    updated_json_string = classify_and_score_reviews(original_json_string, model)
    
    # Convert the JSON string back to a dictionary
    updated_data = json.loads(updated_json_string)
    
    # Write the updated JSON data to a file
    write_json_file(updated_data, output_json_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script_name.py input_json_path output_json_path")
    else:
        input_json_path = "/Users/mahithapenmetsa/Desktop/4300-Flask-Template-JSON/backend/init.json"
        output_json_path = "/Users/mahithapenmetsa/Desktop/4300-Flask-Template-JSON/backend/init.json"
        main(input_json_path, output_json_path)
