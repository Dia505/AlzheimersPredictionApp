import pickle

import joblib
import numpy as np
import streamlit as st

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")  # You must save and load your scaler used during training

st.title("Alzheimer’s Risk Predictor")

# User inputs
age = st.slider("Age", 60, 89, 75)
sleep_hours = st.slider("Average Sleep Hours", 2.0, 10.0, 6.5)
bp_systolic = st.slider("Systolic Blood Pressure", 90, 180, 120)
cholesterol = st.slider("Cholesterol Level", 120, 300, 180)
diet_score = st.slider("Diet Score (0–10)", 0.0, 0.99, 0.5)
physical_activity = st.selectbox("Physical Activity Level", [0, 1, 2])  
education_level = st.selectbox("Education Level", [0, 1, 2, 3]) 
access_healthcare = st.selectbox("Access to Healthcare", [0, 1, 2])  
gender = st.selectbox("Gender", ["Female", "Male"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
depression = st.selectbox("Depression Diagnosis", ["No", "Yes"])
cognitive_decline = st.selectbox("Cognitive Decline Noted?", ["No", "Yes"])

smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
alcohol_use = st.selectbox("Alcohol Use", ["None", "Occasional", "Frequent"])

# Binary encoding
gender_binary = 1 if gender == "Female" else 0
diabetes_binary = 1 if diabetes == "Yes" else 0
depression_binary = 1 if depression == "Yes" else 0
cognitive_binary = 1 if cognitive_decline == "Yes" else 0

# One-hot encoding for smoking (order: current, former, never)
smoking_current = 1 if smoking_status == "Current" else 0
smoking_former = 1 if smoking_status == "Former" else 0
smoking_never = 1 if smoking_status == "Never" else 0

# One-hot encoding for alcohol (order: none, occasional, frequent)
alcohol_none = 1 if alcohol_use == "None" else 0
alcohol_occasional = 1 if alcohol_use == "Occasional" else 0
alcohol_frequent = 1 if alcohol_use == "Frequent" else 0

# Scale the continuous inputs (must match training scaler)
numeric_input = np.array([[age, sleep_hours, bp_systolic, cholesterol]])
scaled_numeric = scaler.transform(numeric_input)
full_numeric_input = np.hstack((scaled_numeric[0], [diet_score]))

# Combine all features in order used in training
final_input = np.hstack((
    [cognitive_binary],
    full_numeric_input,
    [physical_activity, depression_binary],
    [education_level, access_healthcare, gender_binary, diabetes_binary],
    [smoking_never, smoking_former, smoking_current],
    [alcohol_occasional, alcohol_none, alcohol_frequent]
)).reshape(1, -1)

# Predict
if st.button("Predict"):
    prediction = model.predict(final_input)
    st.success(f"Prediction: {'At Risk' if prediction[0] == 1 else 'Not at Risk'}")
