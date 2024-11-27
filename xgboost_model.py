import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import wandb

# Load dataset
file_path = 'data/wine_quality_1000.csv'
wine_data = pd.read_csv(file_path)

# Drop unnecessary columns
wine_data.drop(columns=['Unnamed: 0'], inplace=True)

# Encode target variable
label_encoder = LabelEncoder()
wine_data['country'] = label_encoder.fit_transform(wine_data['country'])

# Preprocessing steps BEFORE splitting
# 1. Sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
wine_data['description_emb'] = list(model.encode(wine_data['description'].tolist()))

# 2. Group rare varieties
threshold = 5
for country in wine_data['country'].unique():
    rare_varieties = wine_data[wine_data['country'] == country]['variety'].value_counts()
    rare_varieties = rare_varieties[rare_varieties < threshold].index
    wine_data.loc[wine_data['country'] == country, 'variety'] = wine_data.loc[
        wine_data['country'] == country, 'variety'].replace(rare_varieties, 'Other')

# 3. One-hot encode 'variety'
variety_dummies = pd.get_dummies(wine_data['variety'], prefix='variety')
wine_data = pd.concat([wine_data, variety_dummies], axis=1)
wine_data.drop(columns=['variety'], inplace=True)

# 4. Separate features and target
X = wine_data.drop(columns=['country', 'description'])
y = wine_data['country']

# Train-test-validation split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

# Preprocessing steps AFTER splitting
# 1. Scale numerical features
numerical_columns = ['points', 'price']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Combine processed features
X_train_combined = np.hstack([np.vstack(X_train['description_emb']), X_train.drop(columns=['description_emb']).values])
X_val_combined = np.hstack([np.vstack(X_val['description_emb']), X_val.drop(columns=['description_emb']).values])
X_test_combined = np.hstack([np.vstack(X_test['description_emb']), X_test.drop(columns=['description_emb']).values])

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights_train = np.array([class_weights[cls] for cls in y_train])

# Define train_model function for W&B
def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Initialize XGBoost model with hyperparameters
        model = XGBClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss'
        )
        
        # Fit model with sample weights
        model.fit(X_train_combined, y_train, sample_weight=sample_weights_train)
        
        # Predict and evaluate
        val_pred = model.predict(X_val_combined)
        val_accuracy = accuracy_score(y_val, val_pred)
        wandb.log({"val_accuracy": val_accuracy})
        
        print("Validation Accuracy:", val_accuracy)
        print(classification_report(y_val, val_pred))

# Define W&B sweep configuration
sweep_config = {
    'method': 'bayes',  # Options: 'grid', 'random', 'bayes'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'n_estimators': {
            'values': [n for n in range(50, 310, 10)]
        },
        'learning_rate': {
            'min': 0.01,
            'max': 0.3
        },
        'max_depth': {
            'values': [n for n in range(2, 10)]
        },
        'subsample': {
            'min': 0.6,
            'max': 1.0
        },
        'colsample_bytree': {
            'min': 0.6,
            'max': 1.0
        }
    }
}

# Run W&B sweep
sweep_id = wandb.sweep(sweep_config, project="wine-country-classification")
wandb.agent(sweep_id, function=train_model, count=250)
