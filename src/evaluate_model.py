"""
Filename: src/evaluate_model.py
Purpose: Functions to evaluate models using MAE, RMSE, and R2. Also handles saving the best model.
How to run: Import into notebooks.
Output: Evaluation metrics and saved model files.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculates MAE, RMSE, and R2 score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

def compare_models(results_list):
    """
    Takes a list of evaluation dictionaries and returns a formatted DataFrame.
    """
    df_results = pd.DataFrame(results_list)
    # Sort by lowest MAE
    df_results = df_results.sort_values(by='MAE').reset_index(drop=True)
    return df_results

def save_model(model, filepath='models/best_model.pkl'):
    """
    Saves the trained model (pipeline) to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved successfully at {filepath}")

def load_model(filepath='models/best_model.pkl'):
    """
    Loads a saved model from disk.
    """
    return joblib.load(filepath)

if __name__ == "__main__":
    print("This module provides evaluation and saving functions.")
