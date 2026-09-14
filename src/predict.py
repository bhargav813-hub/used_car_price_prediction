"""
Filename: src/predict.py
Purpose: Provides a prediction interface for single or multiple new observations using the saved model.
How to run: Import into the Streamlit app or API.
Output: Predicted price.
"""
import pandas as pd
from src.evaluate_model import load_model
from src.feature_engineering import apply_feature_engineering
import os

def make_prediction(model_path, input_data):
    """
    Loads the model and makes a prediction on the provided dictionary of input data.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
    model = load_model(model_path)
    
    # Convert dictionary to single-row DataFrame
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data.copy()
        
    # Apply the same feature engineering used during training
    df_engineered = apply_feature_engineering(df_input)
    
    prediction = model.predict(df_engineered)
    return prediction[0]

if __name__ == "__main__":
    print("This module provides prediction capabilities.")
