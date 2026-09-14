"""
Filename: src/train_model.py
Purpose: Contains functions to split data, build a preprocessing pipeline (encoding/scaling), and train regression models.
How to run: Import into notebooks or run directly.
Output: Trained Scikit-Learn model/pipeline.
"""
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

def get_preprocessor(df, target_col='Price'):
    """
    Creates a ColumnTransformer to handle numerical and categorical data.
    """
    # Exclude target from features
    features = df.drop(columns=[target_col]) if target_col in df.columns else df
    
    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # We want to scale numerical features and one-hot encode categorical features.
    # We use handle_unknown='ignore' so the model doesn't break if it sees a new car model in the future.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    return preprocessor

def split_data(df, target_col='Price', test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_models():
    """Returns a dictionary of un-trained regression models."""
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42)
    }

def train_pipeline(X_train, y_train, preprocessor, model):
    """
    Creates and trains a pipeline consisting of the preprocessor and the given model.
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def tune_random_forest(X_train, y_train, preprocessor):
    """
    Example of hyperparameter tuning for Random Forest using RandomizedSearchCV.
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    
    random_search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_grid, 
        n_iter=10, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        random_state=42, 
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

if __name__ == "__main__":
    print("This module provides training functions and pipelines.")
