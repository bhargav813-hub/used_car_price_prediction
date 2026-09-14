"""
Filename: notebooks/01_data_understanding.py
Purpose: Script to load the raw dataset and display basic statistics, missing values, and data types.
How to run: Run directly in a terminal or import into Jupyter/Colab.
Output: Print statements showing dataset characteristics.
"""
import pandas as pd
import os
import sys

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_data

def run_data_understanding():
    filepath = 'data/raw/used_cars.csv'
    
    if not os.path.exists(filepath):
        print(f"Dataset not found at {filepath}.")
        print("Please place the dataset there before running this script.")
        return
        
    df = load_data(filepath)
    
    print("=== Dataset Head ===")
    print(df.head(), "\n")
    
    print("=== Dataset Tail ===")
    print(df.tail(), "\n")
    
    print("=== Dataset Shape ===")
    print(df.shape, "\n")
    
    print("=== Dataset Info ===")
    df.info()
    print("\n")
    
    print("=== Dataset Description ===")
    print(df.describe(include='all'), "\n")
    
    print("=== Missing Values ===")
    print(df.isnull().sum(), "\n")
    
    print("=== Duplicate Records ===")
    print(df.duplicated().sum(), "\n")
    
    # Identify Data Types
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numerical Features: {num_cols}")
    print(f"Categorical Features: {cat_cols}")

if __name__ == "__main__":
    run_data_understanding()
