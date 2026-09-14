"""
Filename: src/feature_engineering.py
Purpose: Creates new predictive features from existing ones. E.g., Vehicle Age, extracting Brand/Model, and Mileage per Year.
How to run: Import into notebooks or training scripts.
Output: DataFrame with new engineered columns.
"""
import pandas as pd
import numpy as np

def calculate_vehicle_age(df, year_col='Model Year', reference_year=2024):
    """
    Calculates Vehicle Age.
    Why: Older cars generally depreciate. Age is a stronger linear predictor than raw Model Year.
    """
    if year_col in df.columns:
        df['Vehicle Age'] = reference_year - df[year_col]
        # Handle cases where year is in the future
        df['Vehicle Age'] = df['Vehicle Age'].apply(lambda x: x if x >= 0 else 0)
    return df

def calculate_mileage_per_year(df):
    """
    Calculates Mileage per Year.
    Why: High mileage per year indicates heavy usage (e.g., taxi, delivery), which drastically lowers value.
    """
    if 'Mileage' in df.columns and 'Vehicle Age' in df.columns:
        # Avoid division by zero for brand new cars
        df['Mileage per Year'] = df['Mileage'] / (df['Vehicle Age'] + 1)
    return df

def extract_brand_model(df, col='Brand & Model'):
    """
    Splits a combined Brand and Model column.
    Why: Some datasets provide 'Toyota Camry'. We want 'Brand' = 'Toyota' and 'Model' = 'Camry'.
    """
    if col in df.columns:
        # Assuming format "Brand Model..."
        split_data = df[col].astype(str).str.split(' ', n=1, expand=True)
        if split_data.shape[1] == 2:
            df['Extracted_Brand'] = split_data[0]
            df['Extracted_Model'] = split_data[1]
        else:
            df['Extracted_Brand'] = split_data[0]
            df['Extracted_Model'] = 'Unknown'
    return df

def extract_engine_capacity(df, col='Engine Type'):
    """
    Extracts numeric engine capacity (e.g., '2.0' from '2.0L 4-Cylinder').
    Why: Larger engine capacity often correlates with higher trim levels and price.
    """
    if col in df.columns:
        # Extract the first float-like number
        df['Engine Capacity (L)'] = df[col].astype(str).str.extract(r'(\d+\.\d+)').astype(float)
        # Fill missing with median
        median_val = df['Engine Capacity (L)'].median()
        if pd.isna(median_val):
            median_val = 2.0 # Default fallback
        df['Engine Capacity (L)'] = df['Engine Capacity (L)'].fillna(median_val)
    return df

def apply_feature_engineering(df, reference_year=2024):
    """Runs all feature engineering steps."""
    df = df.copy()
    df = calculate_vehicle_age(df, reference_year=reference_year)
    df = calculate_mileage_per_year(df)
    df = extract_brand_model(df)
    df = extract_engine_capacity(df)
    
    # Drop original columns that are now redundant if they exist and if it helps lower dimensionality, 
    # but for safety, we'll keep them for the ML pipeline to process or ignore.
    
    return df

if __name__ == "__main__":
    print("This module provides feature engineering functions.")
