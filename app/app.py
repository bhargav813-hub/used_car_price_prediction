"""
Filename: app/app.py
Purpose: Streamlit web application providing a UI for the used car price prediction system.
How to run: streamlit run app/app.py
Output: Interactive web interface.
"""
import streamlit as st
import pandas as pd
import sys
import os

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import make_prediction

st.set_page_config(page_title="Used Car Price Prediction", layout="centered")

st.title("🚗 Used Car Price Prediction")
st.markdown("Enter the specifications of a used vehicle below to get an estimated selling price based on historical data.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Vehicle Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand_model = st.text_input("Brand & Model (e.g. Toyota Camry)", "Toyota Camry")
        model_year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2018)
        mileage = st.number_input("Mileage (distance driven)", min_value=0, value=50000)
        fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid", "Other"])
        engine_type = st.text_input("Engine Type (e.g. 2.5L 4-Cylinder)", "2.5L")
        
    with col2:
        transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT", "Other"])
        exterior_color = st.text_input("Exterior Color", "White")
        interior_color = st.text_input("Interior Color", "Black")
        accident_history = st.selectbox("Accident History", ["Clean", "Reported Accident", "Salvage"])
        clean_title = st.selectbox("Clean Title", ["Yes", "No"])
        
    submit_button = st.form_submit_button(label="Predict Price")

if submit_button:
    brand_parts = brand_model.split(' ', 1)
    brand = brand_parts[0] if len(brand_parts) > 0 else 'Unknown'
    model_name = brand_parts[1] if len(brand_parts) > 1 else 'Unknown'

    # Prepare input dictionary matching training features
    input_data = {
        'Brand & Model': brand_model,
        'Brand': brand,
        'Model': model_name,
        'Model Year': model_year,
        'Mileage': mileage,
        'Fuel Type': fuel_type,
        'Engine Type': engine_type,
        'Transmission': transmission,
        'Exterior Color': exterior_color,
        'Interior Color': interior_color,
        'Accident History': accident_history,
        'Clean Title': clean_title
    }
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
    
    try:
        predicted_price = make_prediction(model_path, input_data)
        
        st.success("Prediction Successful!")
        st.markdown(f"### Estimated Used-Car Price: **${predicted_price:,.2f}**")
        
        st.info(
            "**Note on Prediction & Confidence:**\n"
            "This prediction is based on historical data. Factors such as exact vehicle condition (scratches, interior wear), "
            "local market fluctuations, and urgent selling needs can cause the actual selling price to deviate. "
            "This is an estimation, not a guarantee."
        )
        
        # Display Feature Importance conceptual note
        st.markdown("#### What affects this price?")
        st.write("- **Vehicle Age:** Older cars typically depreciate (except classics).")
        st.write("- **Mileage per Year:** High mileage drastically reduces value.")
        st.write("- **Brand:** Premium brands hold value differently than economy brands.")
        
    except FileNotFoundError:
        st.error(f"Model not found at `{model_path}`. Please train the model first by running `notebooks/03_full_pipeline.py`.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
