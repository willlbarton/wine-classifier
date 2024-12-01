# Wine Quality Classification with XGBoost

This repository implements a classification model using XGBoost to predict the country of origin of wines based on a dataset containing descriptions, numerical features, and categorical features. The model includes preprocessing, feature engineering, training, and evaluation components, and supports hyperparameter tuning via Weights & Biases (W&B).

---

## Features

- Preprocessing includes scaling numerical features, encoding categorical variables, and applying TF-IDF to textual data.
- Handles imbalanced datasets by computing class weights.
- Supports hyperparameter tuning through W&B sweeps for optimal parameter selection.
- Final model evaluation using a dedicated test set.

---

## Requirements

- xgboost
- scikit-learn
- pandas
- numpy
- Weights & Biases (optional for hyperparameter tuning)

To install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

## Running the script

```bash
python3 xgboost_model.py [options]
```

### Options

1. **Hyperparameter Tuning**  
   Perform hyperparameter tuning using Weights & Biases (WANDB).  
   **Command:**

   ```bash
   python3 xgboost_model.py --hyper_tune
   ```

   **Description:**  
   Activates hyperparameter optimisation to search for the best model configuration. If this option is not present, the model will be evaluated using the default hyperparameters.

2. **Training on different dataset**  
   Train the model using a specified dataset
   **Command:**

   ```bash
   python3 xgboost_model.py --dataset <path_to_dataset>
   ```

   **Description:**  
   Runs script using a dataset specified by user, defaults to the provided dataset if this option is not present
