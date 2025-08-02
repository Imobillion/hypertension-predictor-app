# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and features
model = joblib.load("hypertension_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Hypertension Predictor", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .result {
        font-size: 24px;
        padding: 10px;
        background-color: #e0e0e0;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🩺 Hypertension Risk Prediction App</div>', unsafe_allow_html=True)
st.write("Fill out the form below to check hypertension risk.")

# User input form
with st.form("hypertension_form"):
    age = st.slider("Age", 18, 100, 45)
    salt = st.slider("Salt Intake (g/day)", 0.0, 20.0, 9.0)
    stress = st.slider("Stress Score (0-10)", 0, 10, 5)
    bp_history = st.selectbox("Blood Pressure History", ["Normal", "Prehypertension", "Hypertension"])
    sleep = st.slider("Sleep Duration (hours/night)", 0.0, 12.0, 6.0)
    bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 24.5)
    medication = st.selectbox("Medication", ["None", "ACE Inhibitor", "Other"])
    family_history = st.selectbox("Family History of Hypertension", ["No", "Yes"])
    exercise = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])
    smoker = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])

    submit = st.form_submit_button("🔍 Predict")

if submit:
    # Create input dataframe
    input_data = pd.DataFrame([{
        "Age": age,
        "Salt_Intake": salt,
        "Stress_Score": stress,
        "BP_History": bp_history,
        "Sleep_Duration": sleep,
        "BMI": bmi,
        "Medication": medication,
        "Family_History": family_history,
        "Exercise_Level": exercise,
        "Smoking_Status": smoker
    }])

    # One-hot encode input and align with training columns
    input_encoded = pd.get_dummies(input_data)
    for col in features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[features]

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.markdown("---")
    if pred == 1:
        st.markdown(
            f"<div class='result' style='background-color:#f8d7da;'>⚠️ <strong>High Risk:</strong> This person is likely to have hypertension.<br>Probability: {prob:.2f}</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='result' style='background-color:#d4edda;'>✅ <strong>Low Risk:</strong> This person is not likely to have hypertension.<br>Probability: {prob:.2f}</div>",
            unsafe_allow_html=True)
