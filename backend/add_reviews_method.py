import json
import os
import pandas as pd


directory = r'c:\Users\Nathan Palamuttam\Downloads\2_reviews_per_movie_raw'

csv_names = []

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        processed_name = filename[:-9]
        print(processed_name)
        csv_names.append((processed_name, filename))


with open('init.json', 'r') as file:
    data = json.load(file)

    for item in data:
        for item1 in csv_names:
            if item['title'] is not None and item['title'] in item1[0]:
                csv_file_path = os.path.join(directory, item1[1])
                df = pd.read_csv(csv_file_path)
                df_sorted= df.sort_values(by='helpful', ascending=False)
                reviews = df_sorted['review'].head(10).tolist()
                item['reviews'] = reviews
                print(item['title'])
                print(reviews)
with open('init.json', 'w') as file:
    json.dump(data, file, indent = 4)



