"""
Filename: notebooks/02_eda.py
Purpose: Generates exploratory visualizations to understand feature relationships with the target variable (Price).
How to run: Run directly in a terminal or import into Jupyter/Colab.
Output: Saves visualization plots to the 'notebooks/plots' directory.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import apply_feature_engineering

def run_eda():
    filepath = 'data/raw/used_cars.csv'
    if not os.path.exists(filepath):
        print(f"Dataset not found at {filepath}.")
        return

    print("Loading and preprocessing data for EDA...")
    df_raw = load_data(filepath)
    df_clean = preprocess_data(df_raw)
    df = apply_feature_engineering(df_clean)
    
    os.makedirs('notebooks/plots', exist_ok=True)
    
    # 1. Price Distribution
    if 'Price' in df.columns:
        plt.figure(figsize=(10, 6))
        plt_sns.histplot(df['Price'], bins=50, kde=True)
        plt.title('Distribution of Used Car Prices')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.savefig('notebooks/plots/price_distribution.png')
        plt.close()
    
    # 2. Price vs Model Year
    if 'Model Year' in df.columns and 'Price' in df.columns:
        plt.figure(figsize=(10, 6))
        plt_sns.scatterplot(data=df, x='Model Year', y='Price', alpha=0.5)
        plt.title('Price vs Model Year')
        plt.savefig('notebooks/plots/price_vs_year.png')
        plt.close()
        
    # 3. Price vs Mileage
    if 'Mileage' in df.columns and 'Price' in df.columns:
        plt.figure(figsize=(10, 6))
        plt_sns.scatterplot(data=df, x='Mileage', y='Price', alpha=0.5)
        plt.title('Price vs Mileage')
        plt.savefig('notebooks/plots/price_vs_mileage.png')
        plt.close()
        
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=['number']).corr()
    plt_sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('notebooks/plots/correlation_heatmap.png')
    plt.close()

    print("EDA completed. Plots saved in 'notebooks/plots/' directory.")
    
if __name__ == "__main__":
    run_eda()
