import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Cyclone Intensity Prediction", layout="centered", page_icon="🌪️")

# Custom CSS
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #1f77b4;
}
div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
    height: 3em;
    width: 100%;
    border-radius: 10px;
    font-size: 18px;
}
div.stButton > button:first-child:hover {
    background-color: #e63946;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div style='text-align: center;'>
    <h1 style='color: #1f77b4;'>🌪️ Cyclone Intensity Prediction</h1>
    <p style='font-size:16px;'>Predict cyclone occurrence using wind, pressure, and location</p>
</div>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv(r"C:\Users\rakes\Downloads\atlantic.csv")
df.columns = df.columns.str.strip()

# Convert Latitude & Longitude to numeric floats
def convert_lat(val):
    if isinstance(val, str):
        if val.endswith("N"):
            return float(val[:-1])
        elif val.endswith("S"):
            return -float(val[:-1])
    return float(val)

def convert_lon(val):
    if isinstance(val, str):
        if val.endswith("E"):
            return float(val[:-1])
        elif val.endswith("W"):
            return -float(val[:-1])
    return float(val)

if "Latitude" in df.columns:
    df["Latitude"] = df["Latitude"].apply(convert_lat)
if "Longitude" in df.columns:
    df["Longitude"] = df["Longitude"].apply(convert_lon)

# Create Cyclone column if missing
if "Cyclone" not in df.columns:
    df["Cyclone"] = (df["Maximum Wind"] >= 64).astype(int)

# Features to use
feature_cols = ["Maximum Wind", "Minimum Pressure", "Latitude", "Longitude"]
df = df.dropna(subset=feature_cols)

X = df[feature_cols]
y = df["Cyclone"]

# Scaling + Model
scaler_path = "scaler.pkl"
model_path = "cyclone_model.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    st.info("Training model, please wait...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    st.success("Model training completed and saved!")

# Input sliders
st.subheader("Enter Input Parameters")
max_wind = st.slider("Maximum Wind (knots)", min_value=0, max_value=200, step=1, value=50)
min_pressure = st.slider("Minimum Pressure (hPa)", min_value=850, max_value=1100, step=1, value=1000)
latitude = st.slider("Latitude", min_value=-90, max_value=90, step=1, value=15)
longitude = st.slider("Longitude", min_value=-180, max_value=180, step=1, value=-60)

# Input summary
st.subheader("Input Summary")
st.markdown(f"""
<div style='padding:10px; border:1px solid #ddd; border-radius:10px; display:flex; flex-wrap:wrap; justify-content:space-around; background-color:#f0f2f6;'>
    <div><strong>Maximum Wind:</strong> {max_wind} knots</div>
    <div><strong>Minimum Pressure:</strong> {min_pressure} hPa</div>
    <div><strong>Latitude:</strong> {latitude}</div>
    <div><strong>Longitude:</strong> {longitude}</div>
</div>
""", unsafe_allow_html=True)

# Prepare input
input_df = pd.DataFrame([{
    "Maximum Wind": max_wind,
    "Minimum Pressure": min_pressure,
    "Latitude": latitude,
    "Longitude": longitude
}])

input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Cyclone"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.markdown("""
            <div style='padding:20px; border-radius:10px; background-color:#ff4b4b; color:white; text-align:center;'>
                <h2>🌪️ Cyclone Predicted!</h2>
                <p>Take necessary precautions.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='padding:20px; border-radius:10px; background-color:#4CAF50; color:white; text-align:center;'>
                <h2>✅ No Cyclone</h2>
                <p>Weather conditions are safe.</p>
            </div>
        """, unsafe_allow_html=True)
