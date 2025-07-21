import joblib
import numpy as np
import streamlit as st

# Load model and scaler
model = joblib.load("random_forest_model.pkl")

st.title("Alzheimer’s Risk Predictor")

# User inputs
age = st.slider("Age", 60, 89, 75)
sleep_hours = st.slider("Average Sleep Hours", 2.0, 10.0, 6.5, format="%.1f")
diet_score = st.slider("Diet Score", 0.0, 0.99, 0.5)
bp_systolic = st.slider("Systolic Blood Pressure", 90.0, 180.0, 120.0, format="%.1f")
cholesterol = st.slider("Cholesterol Level", 120.0, 300.0, 180.0, format="%.1f")
 
gender = st.selectbox("Gender", ["Male", "Female"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
depression = st.selectbox("Depression Diagnosis", ["No", "Yes"])
cognitive_decline = st.selectbox("Cognitive Decline Noted?", ["No", "Yes"])

smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
alcohol_use = st.selectbox("Alcohol Use", ["None", "Occasional", "Frequent"])

# Ordinal string display → numeric encoding
education_level_map = {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}
physical_activity_map = {"Low": 0, "Moderate": 1, "High": 2}
access_to_healthcare_map = {"Poor": 0, "Moderate": 1, "Good": 2}

education_level_str = st.selectbox("Education Level", list(education_level_map.keys()))
physical_activity_str = st.selectbox("Physical Activity Level", list(physical_activity_map.keys()))
access_to_healthcare_str = st.selectbox("Access to Healthcare", list(access_to_healthcare_map.keys()))

# Convert to ordinal values
education_level_ord = education_level_map[education_level_str]
physical_activity_ord = physical_activity_map[physical_activity_str]
access_to_healthcare_ord = access_to_healthcare_map[access_to_healthcare_str]

# Binary encoding
gender_binary = 1 if gender == "Female" else 0
diabetes_binary = 1 if diabetes == "Yes" else 0
depression_binary = 1 if depression == "Yes" else 0
cognitive_decline_binary = 1 if cognitive_decline == "Yes" else 0

# One-hot encoding for smoking (order: current, former, never)
smoking_status_current = 1 if smoking_status == "Current" else 0
smoking_status_former = 1 if smoking_status == "Former" else 0
smoking_status_never = 1 if smoking_status == "Never" else 0

# One-hot encoding for alcohol (order: none, occasional, frequent)
alcohol_use_none = 1 if alcohol_use == "None" else 0
alcohol_use_occasional = 1 if alcohol_use == "Occasional" else 0
alcohol_use_frequent = 1 if alcohol_use == "Frequent" else 0

# Combine all features in order used in training
final_input = np.hstack((
    age,
    sleep_hours,
    diet_score,
    bp_systolic,
    cholesterol,
    gender_binary,
    diabetes_binary,
    depression_binary,
    cognitive_decline_binary,
    smoking_status_current,
    smoking_status_former,
    smoking_status_never,
    alcohol_use_frequent,
    alcohol_use_none,
    alcohol_use_occasional,
    education_level_ord,
    physical_activity_ord,
    access_to_healthcare_ord
)).reshape(1, -1)

# Predict
if st.button("Predict"):
    prediction = model.predict(final_input)
    st.success(f"Prediction: {'At Risk' if prediction[0].lower() == 'yes' else 'Not at Risk'}")


