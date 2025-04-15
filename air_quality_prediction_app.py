import streamlit as st
import joblib
import numpy as np

# Load model and label encoder
model = joblib.load('air_quality_model.pkl')
le = joblib.load('label_encoder.pkl')

st.title("🌫️ Air Quality Prediction App")

# Input form
st.sidebar.header("Enter Environmental Features:")
Tm = st.sidebar.number_input("Mean Temp (Tm)", value=25.0)
T = st.sidebar.number_input("Temperature (T)", value=27.0)
TM = st.sidebar.number_input("Max Temp (TM)", value=30.0)
VV = st.sidebar.number_input("Visibility (VV)", value=2.0)
SLP = st.sidebar.number_input("Sea Level Pressure (SLP)", value=1010.0)
H = st.sidebar.number_input("Humidity (H)", value=80.0)
V = st.sidebar.number_input("Wind Speed (V)", value=5.0)
VM = st.sidebar.number_input("Min Wind Speed / Variation (VM)", value=1.0)

# Predict
if st.button("Predict Air Quality"):
    input_data = np.array([[Tm, T, TM, VV, SLP, H, V, VM]])
    prediction = model.predict(input_data)
    label = le.inverse_transform(prediction)[0]
    st.success(f"Predicted Air Quality: **{label}**")
