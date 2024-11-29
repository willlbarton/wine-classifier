import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--hyper_tune", action="store_true", help="Perform hyperparameter tuning using WandB.")
args = parser.parse_args()

file_path = 'data/wine_quality_1000.csv'
wine_data = pd.read_csv(file_path)
wine_data.drop(columns=['Unnamed: 0'], inplace=True)

label_encoder = LabelEncoder()
wine_data['country'] = label_encoder.fit_transform(wine_data['country'])

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
wine_data['description_emb'] = list(model.encode(wine_data['description'].tolist()))

numerical_columns = ['points', 'price']
scaler = StandardScaler()
wine_data[numerical_columns] = scaler.fit_transform(wine_data[numerical_columns])

threshold = 3
variety_counts = wine_data['variety'].value_counts()
rare_varieties = variety_counts[variety_counts < threshold].index
wine_data['variety'] = wine_data['variety'].replace(rare_varieties, 'Other')
wine_data['variety'] = label_encoder.fit_transform(wine_data['variety'])

X = np.hstack([
    np.vstack(wine_data['description_emb']),
    wine_data.drop(columns=['description_emb', 'country', 'description']).values
])
Y = wine_data['country']

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
            gamma=config.gamma,
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
            
            wandb.log({"fold_accuracy": val_accuracy, "fold_f1": val_f1})

        mean_f1 = np.mean(fold_f1)
        mean_accuracy = np.mean(fold_acc)
        wandb.log({"mean_accuracy": mean_accuracy, "mean_f1": mean_f1})
        print(f"Mean Accuracy across folds: {mean_accuracy}")
        print(f"Mean F1 Score across folds: {mean_f1}")

if args.hyper_tune:
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'mean_f1',
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
            'gamma': {
                'min': 0,
                'max': 5
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

    sweep_id = wandb.sweep(sweep_config, project="wine-classification")
    wandb.agent(sweep_id, function=train_model, count=250)
else:
    api = wandb.Api()
    project_path = "wb122-imperial-college-london/wine-classification"
    runs = api.runs(project_path)
    best_run = sorted(runs, key=lambda run: run.summary.get("mean_f1", 0), reverse=True)[0]
    best_hyperparameters = best_run.config

    final_model = XGBClassifier(
        n_estimators=best_hyperparameters["n_estimators"],
        learning_rate=best_hyperparameters["learning_rate"],
        max_depth=best_hyperparameters["max_depth"],
        subsample=best_hyperparameters["subsample"],
        colsample_bytree=best_hyperparameters["colsample_bytree"],
        min_child_weight=best_hyperparameters["min_child_weight"],
        gamma=best_hyperparameters["gamma"],
        reg_alpha=best_hyperparameters["reg_alpha"],
        reg_lambda=best_hyperparameters["reg_lambda"],
        random_state=42,
        objective='multi:softprob',
        eval_metric='mlogloss'
    )

    final_model.fit(X_train_val, Y_train_val)

    test_pred = final_model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, test_pred)
    test_f1 = f1_score(Y_test, test_pred, average='weighted')
    test_conf = confusion_matrix(Y_test, test_pred)

    print("Test Set Evaluation")
    print(f"Accuracy: {test_accuracy}")
    print(f"F1 Score: {test_f1}")
    print(f"Confusion Matrix:\n{test_conf}")