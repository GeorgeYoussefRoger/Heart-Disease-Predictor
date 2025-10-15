import streamlit as st
import pandas as pd
import joblib

# Loading the saved model
model = joblib.load("models/final_model.pkl")

st.title("🫀 Heart Disease Predictor")
st.write("Enter patient details below to predict the risk of heart disease.")

# Input fields
thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversible Defect"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
sex = st.selectbox("Sex", ["Female", "Male"])

# Convert inputs into feature format
features = {
    "thal_3.0": thal == "Normal",
    "cp_4.0": cp == "Atypical Angina",
    "exang_0.0": exang == "No",
    "ca": float(ca),
    "thal_7.0": thal == "Reversible Defect",
    "slope_2.0": slope == "Flat",
    "sex_0.0": sex == "Female",
}

# DataFrame with correct column order
X_input = pd.DataFrame([features], columns=[
    "thal_3.0", "cp_4.0", "exang_0.0", "ca", 
    "thal_7.0", "slope_2.0", "sex_0.0"
])

# Predict button
if st.button("Predict"):
    prediction = model.predict(X_input)[0]
    if prediction == 0:
        st.success("✅ No Heart Disease Detected")
    else:
        st.error("⚠️ Risk of Heart Disease")