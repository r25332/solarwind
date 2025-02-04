import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved models and scalers
wind_model = joblib.load('wind_model.joblib',mmap_mode=None)
wind_scaler = joblib.load('scaler.joblib',mmap_mode=None)

solar_model = joblib.load('best_model_solar.joblib',mmap_mode=None)
solar_scaler = joblib.load('scaler_solar.joblib',mmap_mode=None)

# Define the Streamlit app
st.title("Energy Prediction App")

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose Prediction Type:", ("Wind Power Prediction", "Solar Energy Prediction"))

if option == "Wind Power Prediction":
    st.header("Wind Power Prediction")
    st.write("Enter the following details to predict the wind power:")
    
    # User input fields for wind power prediction
    temperature_2m = st.number_input("Temperature at 2m (°C)", value=20.0)
    relativehumidity_2m = st.number_input("Relative Humidity at 2m (%)", value=50.0)
    dewpoint_2m = st.number_input("Dew Point at 2m (°C)", value=10.0)
    windspeed_10m = st.number_input("Wind Speed at 10m (m/s)", value=5.0)
    windspeed_100m = st.number_input("Wind Speed at 100m (m/s)", value=10.0)
    winddirection_10m = st.number_input("Wind Direction at 10m (°)", value=180.0)
    winddirection_100m = st.number_input("Wind Direction at 100m (°)", value=180.0)
    windgusts_10m = st.number_input("Wind Gusts at 10m (m/s)", value=15.0)
    day = st.number_input("Day", value=1, min_value=1, max_value=31)
    month = st.number_input("Month", value=1, min_value=1, max_value=12)
    year = st.number_input("Year", value=2021, min_value=2000, max_value=2100)
    hour = st.number_input("Hour", value=12, min_value=0, max_value=23)
    
    # Prediction button for wind power
    if st.button("Predict Wind Power"):
        # Create a numpy array from the inputs
        input_data = np.array([[temperature_2m, relativehumidity_2m, dewpoint_2m, windspeed_10m, 
                                windspeed_100m, winddirection_10m, winddirection_100m, windgusts_10m, 
                                day, month, year, hour]])

        # Scale the input data
        scaled_input_data = wind_scaler.transform(input_data)

        # Make prediction
        prediction = wind_model.predict(scaled_input_data)

        # Display the prediction
        st.success(f"Predicted Wind Power: {prediction[0]:.4f}")

elif option == "Solar Energy Prediction":
    st.header("Solar Energy Prediction")
    st.write("Enter the following details to predict the solar energy:")
    
    # User input fields for solar energy prediction
    wind_speed = st.number_input("Wind Speed", value=7.5)
    humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=75.0)
    average_wind_speed = st.number_input("Average Wind Speed (Period)", value=8.0)
    average_pressure = st.number_input("Average Pressure (Period)", value=29.82)
    temperature = st.number_input("Temperature", value=69.0)
    day = st.number_input("Day", value=8)
    month = st.number_input("Month", value=3)
    time = st.number_input("Time (Hour)", value=0)
    
    # Create a DataFrame for the input features
    input_data = pd.DataFrame({
        'wind-speed': [wind_speed],
        'humidity': [humidity],
        'average-wind-speed-(period)': [average_wind_speed],
        'average-pressure-(period)': [average_pressure],
        'temperature': [temperature],
        'Day': [day],
        'Month': [month],
        'Time': [time]
    })

    # Scale the input features
    input_data_scaled = solar_scaler.transform(input_data)

    # Make predictions using the loaded model
    if st.button("Predict Solar Energy"):
        prediction = solar_model.predict(input_data_scaled)
        st.success(f"Predicted Solar MW: {prediction[0]:.2f}")




