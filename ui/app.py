import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Navigation Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🫀 Predictor", "📊 Data Visualization"])

# Load Model and original Dataset
model = joblib.load("models/final_model.pkl")
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df = pd.read_csv("data/heart+disease/processed.cleveland.data", names=column_names)


# Page 1: Predictor
if page == "🫀 Predictor":
    st.title("🫀 Heart Disease Predictor")
    st.write("Enter patient details to predict heart disease risk.")

    col1, col2 = st.columns(2)
    with col1:
        thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversible Defect"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    with col2:
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        sex = st.selectbox("Sex", ["Female", "Male"])

    features = {
        "thal_3.0": thal == "Normal",
        "cp_4.0": cp == "Atypical Angina",
        "exang_0.0": exang == "No",
        "ca": float(ca),
        "thal_7.0": thal == "Reversible Defect",
        "slope_2.0": slope == "Flat",
        "sex_0.0": sex == "Female",
    }

    X_input = pd.DataFrame([features], columns=[
        "thal_3.0", "cp_4.0", "exang_0.0", "ca",
        "thal_7.0", "slope_2.0", "sex_0.0"
    ])

    if st.button("Predict"):
        prediction = model.predict(X_input)[0]
        if prediction == 0:
            st.success("✅ No Heart Disease Detected")
        else:
            st.error("⚠️ Risk of Heart Disease")


# Page 2: Trends
else:
    st.title("📊 Heart Disease Data Visualization")
    st.write("Explore patterns and trends in the UCI Heart Disease dataset.")

    # Target Distribution
    fig_target = px.pie(df, names="target", title="Heart Disease Distribution", color_discrete_sequence=px.colors.sequential.RdBu, 
                        labels={"target": "Heart Disease Level"})
    st.plotly_chart(fig_target, use_container_width=True)

    # Cholesterol vs Max Heart Rate
    fig_scatter = px.scatter(df, x="chol", y="thalach", title="Cholesterol vs Max Heart Rate", color="target",  hover_data=['trestbps'],
                             labels={'thalach': 'Max Heart Rate', 'chol': 'Cholesterol','trestbps': 'Resting Blood Pressure', 
                                     'target': 'Heart Disease Level'})
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Disease by Sex
    fig_sex = px.histogram(df, x="sex", title="Heart Disease Levels by Sex", color="target", barmode="group",
                     labels={"sex": "Sex (0: Female, 1: Male)", "target": "Heart Disease Level"})
    st.plotly_chart(fig_sex, use_container_width=True)

    # Chest Pain Type
    fig_cp = px.histogram(df, x="cp", title="Heart Disease Levels by Chest Pain Type", color="target", barmode="group", 
                          labels={"cp": "Chest Pain Type (1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic)", 
                                  "target": "Heart Disease Level"})
    st.plotly_chart(fig_cp, use_container_width=True)

    # Age Distribution
    fig_age = px.histogram(df, x="age", title="Age Distribution by Heart Disease Level", color="target", barmode="stack", 
                           labels={"age": "Age", "target": "Heart Disease Level"})
    st.plotly_chart(fig_age, use_container_width=True)

    # Correlation Heatmap
    corr = df.corr(numeric_only=True)
    fig_corr = px.imshow(corr, title="Feature Correlation Heatmap", text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)