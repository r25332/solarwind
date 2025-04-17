import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow import keras

# === Load Models & Scalers ===
wind_model = joblib.load('wind_model.joblib')
wind_scaler = joblib.load('scaler.joblib')

solar_model = joblib.load('best_model_solar.joblib')
solar_scaler = joblib.load('scaler_solar.joblib')

# === Streamlit App ===
st.title("🔋 Energy Prediction App")

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose Prediction Type:", ("Wind Power Prediction", "Solar Energy Prediction"))


# === WIND POWER PREDICTION ===
if option == "Wind Power Prediction":
    st.header("🌬️ Wind Power Prediction")
    st.write("Enter the following details to predict the wind power:")

    # Input form for wind prediction (now using number_input)
    input_data = {
        'Day': st.number_input('Day', min_value=1, max_value=31, value=15),
        'Month': st.number_input('Month', min_value=1, max_value=12, value=6),
        'Year': st.number_input('Year', min_value=2006, max_value=2016, value=2010),
        'hour': st.number_input('Hour', min_value=0, max_value=23, value=12),
        'Theoretical_Power_Curve (KWh)': st.number_input('Theoretical Power Curve (KWh)', value=500.0),
        'Wind Direction (°)': st.number_input('Wind Direction (°)', min_value=0.0, max_value=360.0, value=180.0),
        'Wind Speed (m/s)': st.number_input('Wind Speed (m/s)', min_value=0.0, value=5.0)
    }

    def preprocess_wind_input(user_input):
        input_df = pd.DataFrame([user_input])
        input_df = input_df[['Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)', 'Day', 'Month', 'Year', 'hour']]
        return wind_scaler.transform(input_df)

    if st.button('Predict Wind Power'):
        try:
            processed_input = preprocess_wind_input(input_data)
            prediction = wind_model.predict(processed_input)
            st.subheader('Predicted LV Active Power (kW)')
            st.success(f"{prediction[0]:.2f} kW")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


# === SOLAR ENERGY PREDICTION ===
elif option == "Solar Energy Prediction":
    st.header("☀️ Solar Energy Prediction")
    st.write("Enter the following details to predict the solar energy:")

    # Input form for solar prediction
    wind_speed = st.number_input("Wind Speed", value=7.5)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0)
    avg_wind_speed = st.number_input("Average Wind Speed (Period)", value=8.0)
    avg_pressure = st.number_input("Average Pressure (Period)", value=29.82)
    temperature = st.number_input("Temperature (°F)", value=69.0)
    day = st.number_input("Day", value=8)
    month = st.number_input("Month", value=3)
    hour = st.number_input("Time (Hour)", value=0)

    solar_input_df = pd.DataFrame({
        'wind-speed': [wind_speed],
        'humidity': [humidity],
        'average-wind-speed-(period)': [avg_wind_speed],
        'average-pressure-(period)': [avg_pressure],
        'temperature': [temperature],
        'Day': [day],
        'Month': [month],
        'Time': [hour]
    })

    if st.button("Predict Solar Energy"):
        try:
            scaled_input = solar_scaler.transform(solar_input_df)
            prediction = solar_model.predict(scaled_input)
            st.subheader("Predicted Solar Energy Output")
            st.success(f"{prediction[0]:.2f} MW")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
