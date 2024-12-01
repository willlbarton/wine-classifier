import pandas as pd
import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://candidate-llm.extraction.artificialos.com/v1/chat/completions"
API_KEY = os.getenv("LLM_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Please set the LLM_API_KEY environment variable.")

cache_file = "llm_cache.json"
try:
    with open(cache_file, "r") as f:
        cache = json.load(f)
except FileNotFoundError:
    cache = {}

def query_llm_with_retry(description, retries=3, wait_time=5):
    if description in cache:
        return cache[description]
    
    for attempt in range(retries):
        try:
            headers = {
                "x-api-key": API_KEY,
                "Content-Type": "application/json",
            }
            
            prompt = (
                f"Analyze the following wine description:\n\n{description}\n\n"
                "1. Does it mention a specific location? If yes, is the location in Spain, France, Italy, or the US?\n"
                "2. Extract the most relevant keywords from the description.\n\n"
                "Answer in this JSON format:\n"
                '{"contains_location": true/false, "country": "Spain/France/Italy/USA/None", "keywords": ["keyword1", "keyword2", ...]}'
            )
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.2,
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                output = json.loads(result["choices"][0]["message"]["content"])
                cache[description] = output
                return output
            elif response.status_code == 429:
                print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"API error: {response.status_code}, {response.text}")
        
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(wait_time)
    
    print("Max retries reached. Skipping this description.")
    cache[description] = {"contains_location": False, "country": None, "keywords": []}
    return cache[description]

def process_dataset_with_llm(file_path, output_file="processed_data.csv"):
    wine_data = pd.read_csv(file_path)
    
    contains_location = []
    countries = []
    keywords_list = []
    
    for index, description in enumerate(wine_data['description']):
        if isinstance(description, str):
            output = query_llm_with_retry(description)
            contains_location.append(output.get("contains_location", False))
            countries.append(output.get("country", None) if output.get("country", None) != "USA" else "US")
            keywords_list.append(output.get("keywords", []))
            
            if (index + 1) % 10 == 0:
                with open(cache_file, "w") as f:
                    json.dump(cache, f)
                print(f"Processed {index + 1}/{len(wine_data)} rows.")
        else:
            contains_location.append(False)
            countries.append(None)
            keywords_list.append([])
    
    with open(cache_file, "w") as f:
        json.dump(cache, f)
    
    wine_data['contains_location'] = contains_location
    wine_data['country_from_location'] = countries
    wine_data['keywords'] = keywords_list
    wine_data.drop(columns=['Unnamed: 0'], inplace=True)
    wine_data.to_csv(output_file, index=False)
    return wine_data

file_path = "data/wine_quality.csv"
output_file = "data/wine_quality_with_locations_and_keywords.csv"
processed_data = process_dataset_with_llm(file_path, output_file)

print(processed_data.head())
filtered_data = processed_data[processed_data['contains_location'] == True].copy()
filtered_data['matches'] = filtered_data['country_from_location'] == filtered_data['country']

matches_count = filtered_data['matches'].sum()
total_filtered = len(filtered_data)
match_percentage = (matches_count / total_filtered) * 100
print(matches_count, total_filtered, match_percentage)

mismatches = filtered_data[filtered_data['matches'] == False]
print(mismatches[['description']])

