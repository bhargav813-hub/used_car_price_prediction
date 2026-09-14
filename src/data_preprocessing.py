"""
Filename: src/data_preprocessing.py
Purpose: Contains functions to clean and preprocess raw used car data, handling missing values, extracting numeric values from strings (like currency or mileage), and standardizing text.
How to run: This module is intended to be imported into training scripts or notebooks.
Output: Cleaned Pandas DataFrame.
"""
import pandas as pd
import numpy as np

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(filepath)

def standardize_columns(df):
    """Renames columns from the specific kaggle dataset to match our expected format."""
    column_mapping = {
        'brand': 'Brand',
        'model': 'Model',
        'model_year': 'Model Year',
        'milage': 'Mileage',
        'fuel_type': 'Fuel Type',
        'engine': 'Engine Type',
        'transmission': 'Transmission',
        'ext_col': 'Exterior Color',
        'int_col': 'Interior Color',
        'accident': 'Accident History',
        'clean_title': 'Clean Title',
        'price': 'Price'
    }
    df = df.rename(columns=column_mapping)
    # Re-create Brand & Model if it was split
    if 'Brand' in df.columns and 'Model' in df.columns and 'Brand & Model' not in df.columns:
        df['Brand & Model'] = df['Brand'].astype(str) + ' ' + df['Model'].astype(str)
    return df

def clean_currency(df, column_name):
    """Removes currency symbols and commas, converts to float."""
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def clean_mileage(df, column_name):
    """Removes units (e.g., 'km', 'miles') and converts to numeric."""
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def handle_missing_values(df):
    """
    Handles missing values by filling or dropping them.
    Numerical columns: fill with median.
    Categorical columns: fill with 'Unknown' or mode.
    """
    df = df.copy()
    
    # Fill numerical NaNs with median
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical NaNs with 'Unknown'
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
    
    return df

def remove_duplicates(df):
    """Removes duplicate rows."""
    return df.drop_duplicates()

def preprocess_data(df, target_col='Price'):
    """
    Executes the full preprocessing pipeline.
    """
    df = standardize_columns(df)
    df = remove_duplicates(df)
    
    # Clean standard target and feature if they exist in string format
    if target_col in df.columns and df[target_col].dtype == 'object':
        df = clean_currency(df, target_col)
        
    if 'Mileage' in df.columns and df['Mileage'].dtype == 'object':
        df = clean_mileage(df, 'Mileage')
        
    df = handle_missing_values(df)
    
    # Remove rows where Target is still NaN after cleaning
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
        
    return df

if __name__ == "__main__":
    print("This module provides data preprocessing functions.")
    # Example usage:
    # df = load_data('../data/raw/used_cars.csv')
    # df_clean = preprocess_data(df)
