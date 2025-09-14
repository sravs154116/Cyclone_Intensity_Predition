# app.py
import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Cyclone Intensity Prediction", layout="centered", page_icon="🌪️")

# Custom CSS for sidebar and button
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #1f77b4;
}

/* Predict button style */
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
    <p style='font-size:16px;'>Predict cyclone occurrence using Maximum Wind and Minimum Pressure</p>
</div>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv(r"C:\Users\rakes\Downloads\atlantic.csv")
df.columns = df.columns.str.strip()

# Binary target
df['Cyclone_Occurrence'] = df['Maximum Wind'].apply(lambda x: 'Yes' if x >= 34 else 'No')

# Features and target
X = df[['Maximum Wind', 'Minimum Pressure']]
y = df['Cyclone_Occurrence']

# Load or train model
le_path = "label_encoder.pkl"
model_path = "cyclone_model.pkl"

if os.path.exists(model_path) and os.path.exists(le_path):
    model = joblib.load(model_path)
    le = joblib.load(le_path)
else:
    st.info("Training model, please wait...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    joblib.dump(le, le_path)
    st.success("Model training completed and saved!")

# Input sliders
st.subheader("Enter Input Parameters")
max_wind = st.slider("Maximum Wind (knots)", min_value=0, max_value=400, step=1)
min_pressure = st.slider("Minimum Pressure (hPa)", min_value=800, max_value=1100, step=1)

# Input summary in one line box
st.subheader("Input Summary")
st.markdown(f"""
<div style='padding:10px; border:1px solid #ddd; border-radius:10px; display:flex; justify-content:space-around; background-color:#f0f2f6;'>
    <div><strong>Maximum Wind:</strong> {max_wind} knots</div>
    <div><strong>Minimum Pressure:</strong> {min_pressure} hPa</div>
</div>
""", unsafe_allow_html=True)

# Prepare input dataframe
input_df = pd.DataFrame({'Maximum Wind': [max_wind], 'Minimum Pressure': [min_pressure]})

# Prediction button
if st.button("Predict Cyclone"):
    prediction = model.predict(input_df)
    prediction_label = le.inverse_transform(prediction)[0]

    # Display prediction with color-coded card
    if prediction_label == 'Yes':
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
