import argparse
import os
import json
import time
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from xgboost import XGBClassifier
import wandb
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

def process_dataset_with_llm(file_path, output_file):
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
    
    wine_data['country_from_location'] = countries
    tfidf = TfidfVectorizer(max_features=250)
    tfidf_matrix = tfidf.fit_transform([" ".join(x) for x in keywords_list])
    tfidf_features = pd.DataFrame(tfidf_matrix.toarray(), 
                                  columns=tfidf.get_feature_names_out(), 
                                  index=wine_data.index)
    
    wine_data = pd.concat([wine_data, tfidf_features], axis=1)
    
    numerical_columns = ['points', 'price']
    scaler = StandardScaler()
    wine_data[numerical_columns] = scaler.fit_transform(wine_data[numerical_columns])

    map = {'US': 0, 'France': 1, 'Italy': 2, 'Spain': 3}
    wine_data['country'] = wine_data['country'].map(map)
    wine_data['country_from_location'] = wine_data['country_from_location'].map(map)

    label_encoder = LabelEncoder()
    threshold = 3
    variety_counts = wine_data['variety'].value_counts()
    rare_varieties = variety_counts[variety_counts < threshold].index
    wine_data['variety'] = wine_data['variety'].replace(rare_varieties, 'Other')
    wine_data['variety'] = label_encoder.fit_transform(wine_data['variety'])

    wine_data.drop(columns=['Unnamed: 0', 'description'], inplace=True)

    wine_data.to_csv(output_file, index=False)
    return wine_data

parser = argparse.ArgumentParser()
parser.add_argument("--hyper_tune", action="store_true", help="Perform hyperparameter tuning using WandB.")
parser.add_argument("--dataset", default="data/wine_quality.csv", help="Path to the dataset.")
args = parser.parse_args()

file_path = args.dataset
base, ext = os.path.splitext(args.dataset)
output_file = f"{base}_processed.csv"

try:
    input = pd.read_csv(file_path)
    length = len(input)
except FileNotFoundError:
    print("File not found.")
    exit(1)

if False and os.path.exists(output_file) and len(pd.read_csv(output_file)) == len(pd.read_csv(file_path)):
    wine_data = pd.read_csv(output_file)
    print("Processed data already exists. Skipping processing.")
else:
    wine_data = process_dataset_with_llm(file_path, output_file)

print(wine_data.head())

X, Y = wine_data.drop(columns=['country']), wine_data['country']

X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)
X_train_val = np.array(X_train_val)
Y_train_val = np.array(Y_train_val)

class_weights = compute_class_weight('balanced', classes=np.unique(Y_train_val), y=Y_train_val)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_model(config=None):
    fold_f1 = []
    fold_acc = []
    with wandb.init(config=config):
        config = wandb.config

        model = XGBClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_weight=config.min_child_weight,
            gamma=0,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss'
        )
        
        conf = np.zeros((4, 4))
        for train_indices, val_indices in kfold.split(X_train_val, Y_train_val):
            X_train, X_val = X_train_val[train_indices], X_train_val[val_indices]
            Y_train, Y_val = Y_train_val[train_indices], Y_train_val[val_indices]
            
            sample_weights_train = np.array([class_weights[cls] for cls in Y_train])
            
            model.fit(X_train, Y_train, sample_weight=sample_weights_train)
            
            val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(Y_val, val_pred)
            val_f1 = f1_score(Y_val, val_pred, average='weighted')
            conf += confusion_matrix(Y_val, val_pred) / 5
            fold_f1.append(val_f1)
            fold_acc.append(val_accuracy)

        mean_f1 = np.mean(fold_f1)
        mean_accuracy = np.mean(fold_acc)
        wandb.log({"mean_accuracy": mean_accuracy, "mean_f1": mean_f1})
        print(f"Mean Accuracy across folds: {mean_accuracy}")
        print(f"Mean F1 Score across folds: {mean_f1}")

if args.hyper_tune:
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'mean_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'n_estimators': {
                'values': [100, 200, 300, 400, 500]
            },
            'learning_rate': {
                'min': 0.01,
                'max': 0.2
            },
            'max_depth': {
                'values': [3, 4, 5, 6, 7, 8, 9]
            },
            'subsample': {
                'min': 0.6,
                'max': 1.0
            },
            'colsample_bytree': {
                'min': 0.6,
                'max': 1.0
            },
            'min_child_weight': {
                'min': 1,
                'max': 10
            },
            'reg_alpha': {
                'min': 0,
                'max': 5
            },
            'reg_lambda': {
                'min': 0,
                'max': 10
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="tf-idf-xgboost")
    wandb.agent(sweep_id, function=train_model, count=250)
else:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_f1_scores = []
    conf_matrix = np.zeros((len(np.unique(Y)), len(np.unique(Y))))  # For cumulative confusion matrix

    for fold, (train_indices, val_indices) in enumerate(kfold.split(X, Y)):
        X_train_fold, X_val_fold = X.iloc[train_indices], X.iloc[val_indices]
        Y_train_fold, Y_val_fold = Y.iloc[train_indices], Y.iloc[val_indices]

        sample_weights_train = np.array([class_weights[cls] for cls in Y_train_fold])

        model = XGBClassifier()
        model.fit(X_train_fold, Y_train_fold, sample_weight=sample_weights_train)

        val_pred = model.predict(X_val_fold)
        val_accuracy = accuracy_score(Y_val_fold, val_pred)
        val_f1 = f1_score(Y_val_fold, val_pred, average='weighted')

        conf_matrix += confusion_matrix(Y_val_fold, val_pred)

        fold_accuracies.append(val_accuracy)
        fold_f1_scores.append(val_f1)

        print(f"Fold {fold + 1} - Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    mean_accuracy = np.mean(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)

    print("\nCross-Validation Results:")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    print(f"Cumulative Confusion Matrix:\n{conf_matrix}")