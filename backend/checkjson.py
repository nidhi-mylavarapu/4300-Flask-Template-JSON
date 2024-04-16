import json

with open('init.json', 'r') as file:
    data = json.load(file)
    for item in data:
        print(item['popularity'], item['image'])
