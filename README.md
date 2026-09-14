# Used Car Price Prediction System

This is a complete Machine Learning-based Used Car Price Prediction System built in Python. 

## 1. Project Objective
This project trains multiple regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) on historical used-car data to predict the listed price of a used vehicle based on its specifications, condition, and history. The final solution is served via a Streamlit web application.

## 2. Dataset Source & Recommendation
**Where to obtain the dataset:**
We recommend downloading a public used-car dataset from **Kaggle**, such as the "Craigslist Vehicles Dataset" or "Used Cars Dataset" by Austin Reese. 
When downloading, place the `.csv` file in the `data/raw/` folder and name it `used_cars.csv`.

Ensure the dataset has columns roughly mapping to:
- Brand & Model
- Model Year
- Mileage
- Fuel Type
- Engine Type
- Transmission
- Exterior/Interior Color
- Accident History
- Clean Title
- Price (Target Variable)

## 3. Evaluation Metrics Explained
- **MAE (Mean Absolute Error):** The average absolute difference between predicted prices and actual prices. It tells us, on average, how many dollars off our predictions are.
- **RMSE (Root Mean Squared Error):** The square root of the average squared errors. RMSE heavily penalizes large errors (e.g., if we predict a $100k car as $10k). If RMSE is much larger than MAE, it means the model has a few very large errors (likely outliers).
- **R² Score:** Explains the proportion of variance in the vehicle prices that our model explains. An R² of 0.85 means our model explains 85% of the price variance. 1.0 is perfect, 0 means it's no better than guessing the average.

## 4. Complete Project Flow
1. **Dataset**: Raw CSV is loaded.
2. **Data Cleaning**: Missing values are imputed, strings (like currency) are parsed into numbers, and duplicates are removed.
3. **Feature Engineering**: Features like `Vehicle Age` (Current Year - Model Year) and `Mileage per Year` are derived because they show stronger linear relationships with price.
4. **EDA (Exploratory Data Analysis)**: Visualizations are generated to understand relationships (e.g., how Mileage reduces Price).
5. **Preprocessing & Encoding**: Categorical data (like Brand, Transmission) is One-Hot Encoded. Numerical data is scaled. This is wrapped in a Scikit-Learn `Pipeline`.
6. **Train/Test Split**: Data is split 80% (train) and 20% (test). Preprocessing pipelines are *only fitted on training data* to prevent data leakage.
7. **Multiple Regression Models**: We train Linear, Tree, Forest, and Gradient Boosting models.
8. **Evaluation**: Models are compared using MAE, RMSE, & R².
9. **Hyperparameter Tuning**: The best baseline model is tuned using `RandomizedSearchCV`.
10. **Save Model**: The final `Pipeline` is saved via `joblib`.
11. **User Input / Streamlit App**: Users enter specs on the web app.
12. **Predicted Price**: The model outputs the estimated price.

## 5. How to Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Add Data:**
   Place your dataset as `used_cars.csv` inside `data/raw/`.
3. **Explore Data & Train Model:**
   You can run the notebook scripts interactively in Jupyter, or just run the pipeline script:
   ```bash
   python notebooks/03_full_pipeline.py
   ```
   This will output the evaluation metrics and save `best_model.pkl` in the `models/` directory.
4. **Run Streamlit App:**
   ```bash
   streamlit run app/app.py
   ```

## 6. How to Run in Google Colab
1. Upload the entire `used-car-price-prediction` folder to your Google Drive.
2. Open a new Colab Notebook and mount your drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/used-car-price-prediction
   ```
3. Install missing requirements:
   ```python
   !pip install streamlit xgboost
   ```
4. Run the training script:
   ```python
   !python notebooks/03_full_pipeline.py
   ```
5. To view the Streamlit app in Colab, you'll need `localtunnel` or `ngrok`.
   ```bash
   !npm install localtunnel
   !streamlit run app/app.py &>/content/logs.txt &
   !npx localtunnel --port 8501
   ```
   Click the URL provided by localtunnel.

## 7. Streamlit Deployment (Cloud)
To deploy this for a college project:
1. Push the repository to GitHub (ensure `data/` and `models/` are pushed if small, or recreate the model on deploy).
2. Go to `share.streamlit.io` and connect your GitHub account.
3. Select this repository and specify `app/app.py` as the Main file path.
4. Click Deploy.
