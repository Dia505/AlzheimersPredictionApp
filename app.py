import joblib
import numpy as np
import streamlit as st
import pandas as pd

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Alzheimer’s Risk Predictor")

# User inputs
age = st.slider("Age", 60, 89, 75)
sleep_hours = st.slider("Average Sleep Hours", 2.0, 10.0, 6.5, format="%.1f")
bp_systolic = st.slider("Systolic Blood Pressure", 90.0, 180.0, 120.0, format="%.1f")
cholesterol = st.slider("Cholesterol Level", 120.0, 300.0, 180.0, format="%.1f")

depression = st.selectbox("Depression Diagnosis", ["No", "Yes"])
cognitive_decline = st.selectbox("Cognitive Decline Noted?", ["No", "Yes"])
alcohol_use = st.selectbox("Alcohol Use", ["None", "Occasional", "Frequent"])

physical_activity_map = {"Low": 0, "Moderate": 1, "High": 2}
physical_activity_str = st.selectbox("Physical Activity Level", list(physical_activity_map.keys()))
physical_activity_ord = physical_activity_map[physical_activity_str]

# Binary encoding
depression_binary = 1 if depression == "Yes" else 0
cognitive_decline_binary = 1 if cognitive_decline == "Yes" else 0

# Alcohol occasional binary (Since alcohol_occasional feature was selected due to significant weight)
alcohol_use_occasional = 1 if alcohol_use == "Occasional" else 0

# Scale numeric features
numeric_input_df = pd.DataFrame([[age, sleep_hours, bp_systolic, cholesterol]], columns=['age', 'sleep_hours', 'bp_systolic', 'cholesterol'])
scaled_numeric = scaler.transform(numeric_input_df)

# st.write("Scaled numeric features:", scaled_numeric[0][:3])

# Combine features in EXACT order from training
final_input = np.hstack((
    scaled_numeric[0][:3],  # age, sleep_hours, bp_systolic
    [depression_binary, cognitive_decline_binary, alcohol_use_occasional, physical_activity_ord]
)).reshape(1, -1)

# st.write("Final input to model:", final_input)

# Predict
if st.button("Predict"):
    prediction = model.predict(final_input)
    st.success(f"Prediction: {'At Risk' if prediction[0] in ['yes'] else 'Not at Risk'}")
    probability = model.predict_proba(final_input)[0, 1]
    st.info(f"Probability of being at risk: {probability:.3f}")
