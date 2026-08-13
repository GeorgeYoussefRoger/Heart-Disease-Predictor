import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Navigation Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Data Visualization"])

# Load Model and original Dataset
model = joblib.load("models/final_model.pkl")
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df = pd.read_csv("data/processed.cleveland.data", names=column_names)


# Page 1: Predictor
if page == "Prediction":
    st.title("🫀 Heart Disease Predictor")
    st.write("Enter patient details to predict heart disease risk.")

    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"]) 
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversible Defect"])

    features = {
        "cp_4.0": 1 if cp == "Non-Anginal" else 0,
        "ca": ca,
        "thal_7.0": 1 if thal == "Reversible Defect" else 0,
        "slope_1.0": 1 if slope == "Flat" else 0,
        "thal_3.0": 1 if thal == "Normal" else 0,
        "cp_3.0": 1 if cp == "Asymptomatic" else 0,
        "oldpeak": oldpeak
    }

    if st.button("Predict"):
        X_input = pd.DataFrame([features], columns=features.keys())
        prediction = model.predict(X_input)[0]
        if prediction == 0:
            st.success("✅ Low Risk of Heart Disease")
        else:
            st.error("⚠️ High Risk of Heart Disease")

# Page 2: Trends
else:
    st.title("📊 Heart Disease Data Visualization")
    st.write("Explore patterns and trends in the UCI Heart Disease dataset.")

    # Cholesterol vs Max Heart Rate
    fig_scatter = px.scatter(df, x="chol", y="thalach", title="Cholesterol vs Max Heart Rate", color="target",  hover_data=['trestbps'],
                             labels={'thalach': 'Max Heart Rate', 'chol': 'Cholesterol','trestbps': 'Resting Blood Pressure', 
                                     'target': 'Heart Disease Level'})
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Chest Pain Type
    fig_cp = px.histogram(df, x="cp", title="Heart Disease Levels by Chest Pain Type", color="target", barmode="group", 
                          labels={"cp": "Chest Pain Type (1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic)", 
                                  "target": "Heart Disease Level"})
    st.plotly_chart(fig_cp, use_container_width=True)

    # Age Distribution
    fig_age = px.histogram(df, x="age", title="Age Distribution by Heart Disease Level", color="target", barmode="stack", 
                           labels={"age": "Age", "target": "Heart Disease Level"})
    st.plotly_chart(fig_age, use_container_width=True)