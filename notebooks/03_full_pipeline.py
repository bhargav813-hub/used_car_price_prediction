"""
Filename: notebooks/03_full_pipeline.py
Purpose: Ties everything together. Loads data, cleans, engineers features, splits, trains models, evaluates, tunes, and saves the best model.
How to run: python notebooks/03_full_pipeline.py
Output: Printed metrics comparison and saved model pipeline.
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import apply_feature_engineering
from src.train_model import get_preprocessor, split_data, get_models, train_pipeline, tune_random_forest
from src.evaluate_model import evaluate_model, compare_models, save_model

def run_pipeline():
    filepath = 'data/raw/used_cars.csv'
    if not os.path.exists(filepath):
        print(f"Dataset not found at {filepath}.")
        print("Creating a dummy dataset just to demonstrate the pipeline architecture...")
        os.makedirs('data/raw', exist_ok=True)
        data = {
            'Brand & Model': ['Toyota Camry', 'Honda Civic', 'Ford Mustang', 'Tesla Model 3', 'BMW 3 Series'] * 20,
            'Model Year': [2018, 2019, 2020, 2021, 2022] * 20,
            'Mileage': ['50000 km', '40,000 miles', '30000', '20000 km', '10000'] * 20,
            'Fuel Type': ['Gasoline', 'Gasoline', 'Gasoline', 'Electric', 'Gasoline'] * 20,
            'Engine Type': ['2.5L 4-Cyl', '2.0L 4-Cyl', '5.0L V8', 'Electric', '3.0L 6-Cyl'] * 20,
            'Transmission': ['Auto', 'Auto', 'Manual', 'Auto', 'Auto'] * 20,
            'Price': ['$15000', '$16000', '$25000', '$35000', '$30000'] * 20
        }
        pd.DataFrame(data).to_csv(filepath, index=False)
        print("Dummy data created.\n")
        
    print("1. Loading Data...")
    df = load_data(filepath)
    
    print("2. Preprocessing Data...")
    df = preprocess_data(df, target_col='Price')
    
    print("3. Feature Engineering...")
    df = apply_feature_engineering(df)
    
    print("4. Train/Test Split...")
    X_train, X_test, y_train, y_test = split_data(df, target_col='Price')
    
    print("5. Getting Preprocessor (Encoding/Scaling)...")
    preprocessor = get_preprocessor(X_train)
    
    print("6. Training and Evaluating Models...")
    models = get_models()
    results = []
    
    best_baseline_model = None
    best_baseline_score = float('inf')
    best_model_name = ""
    
    for name, model in models.items():
        print(f"   Training {name}...")
        pipeline = train_pipeline(X_train, y_train, preprocessor, model)
        
        # Predict on Test set
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        eval_metrics = evaluate_model(y_test, y_pred, name)
        results.append(eval_metrics)
        
        if eval_metrics['MAE'] < best_baseline_score:
            best_baseline_score = eval_metrics['MAE']
            best_baseline_model = model
            best_model_name = name
            
    print("\n--- Model Comparison ---")
    df_results = compare_models(results)
    print(df_results.to_string(index=False))
    
    print(f"\n7. Hyperparameter Tuning (Demonstrating on Random Forest)")
    # For demonstration, we'll tune Random Forest.
    print("   Tuning Random Forest...")
    best_tuned_pipeline = tune_random_forest(X_train, y_train, preprocessor)
    
    print("8. Saving Best Model...")
    # Evaluate tuned model
    y_pred_tuned = best_tuned_pipeline.predict(X_test)
    tuned_metrics = evaluate_model(y_test, y_pred_tuned, "Tuned Random Forest")
    print(f"   Tuned Model MAE: {tuned_metrics['MAE']:.2f}")
    
    save_model(best_tuned_pipeline, 'models/best_model.pkl')
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    run_pipeline()
