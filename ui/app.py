import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the saved model
model = joblib.load("models/final_model.pkl")

df = pd.read_csv("data/heart_disease.csv")

# Sidebar navigation
st.sidebar.title("Navigation Bar")
page = st.sidebar.radio("Go to", ["Prediction", "Data Visualization"])

if page == "Prediction":
    st.title("❤️ Heart Disease Prediction App")
    st.write("Enter patient details below to predict the risk of heart disease.")

    # Input fields
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    cp = st.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"))
    if cp == "Typical Angina" or cp == "Atypical Angina" or cp == "Non-Anginal Pain":
        cp = False
    else:
        cp = True
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=70, max_value=220, value=150)
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    ca = st.selectbox("Number of Major Vessels", (0, 1, 2, 3))
    thal = st.selectbox("Thal", ("Normal", "Fixed Defect", "Reversable Defect"))
    if thal == "Normal" or thal == "Fixed Defect":
        thal = False
    else:
        thal = True

    features = np.array([[thalach, age, oldpeak, chol, trestbps, ca, cp, thal]])

    # Predict button
    if st.button("Predict"):
        prediction = model.predict(features)[0]
        if prediction == 0:
            st.success("✅ No Heart Disease Detected")
        else:
            st.error("⚠️ High Risk of Heart Disease")

elif page == "Data Visualization":
    st.title("📊 Explore Heart Disease Trends")

    # Histogram for Age
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot for Cholesterol Levels
    st.subheader("Cholesterol Levels vs Heart Disease")
    fig, ax = plt.subplots()
    sns.boxplot(x="target", y="chol", data=df, ax=ax)
    ax.set_xlabel("Heart Disease (1-4 = Yes, 0 = No)")
    st.pyplot(fig)

    # Pair Plot for selected features
    st.subheader("Feature Relationships")
    st.write("Scatterplots of key features colored by disease presence.")
    selected = df[["age", "trestbps", "chol", "thalach", "target"]]
    fig = sns.pairplot(selected, hue="target", palette="coolwarm")
    st.pyplot(fig)