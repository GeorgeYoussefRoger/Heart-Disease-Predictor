# ❤️ Heart Disease Predictor

## 📌 Overview
This project applies Machine Learning techniques to predict the presence of heart disease using the UCI Heart Disease dataset. It includes data preprocessing, supervised & unsupervised learning, model evaluation, hyperparameter tuning and deployment through a Streamlit web app with Ngrok for public access.

## 📂 Project Structure
```
Heart_Disease_Project/
├── data/
│ ├── heart_disease.csv # Processed dataset
├── notebooks/
│ ├── 01_data_preprocessing.ipynb # Data cleaning & preprocessing
│ ├── 02_pca_analysis.ipynb # PCA & dimensionality reduction
│ ├── 03_feature_selection.ipynb # Feature selection techniques
│ ├── 04_supervised_learning.ipynb # Classification models
│ ├── 05_unsupervised_learning.ipynb # Clustering (KMeans, Hierarchical)
│ ├── 06_hyperparameter_tuning.ipynb # GridSearch & RandomizedSearch
├── models/
│ ├── final_model.pkl # Tuned best-performing model
├── ui/
│ ├── app.py # Streamlit app for deployment
├── deployment/
│ ├── ngrok_setup.txt # Instructions for sharing app via ngrok
├── results/
│ ├── evaluation_metrics.txt # Accuracy, F1, AUC scores
├── README.md # Project documentation
├── requirements.txt # Dependencies
├── .gitignore
```

## ⚙️ Methodology
1. Data Preprocessing
   - Handled missing values using imputation
   - Encoded categorical variables with One-Hot Encoding
   - Visualized distributions and outliers with boxplots
2. Dimensionality Reduction (PCA)
   - Reduced feature space while retaining 90%+ variance
   - Cumulative variance plot to determine optimal components
3. Feature Selection
   - Random Forest feature importance
   - Recursive Feature Elimination (RFE)
   - Chi-Square test
4. Supervised Learning
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - Metrics evaluated: Accuracy, Precision, Recall, F1-Score, AUC
5. Unsupervised Learning
   - KMeans (Elbow + Silhouette methods)
   - Hierarchical Clustering with dendrograms
   - Compared clustering with actual labels (ARI, crosstab)
6. Hyperparameter Tuning
   - Applied GridSearchCV and RandomizedSearchCV
   - Selected the best model based on tuned performance

## 📊 Dataset
- Source: UCI Heart Disease Dataset => https://archive.ics.uci.edu/dataset/45/heart+disease
- Used Cleveland database

## 🚀 Deployment

1. Clone this repository and install dependencies:
   ```
   git clone https://github.com/GeorgeYoussefRoger/heart-disease-predictor.git
   ```
   ```
   cd heart-disease-predictor
   ```
   ```
   pip install -r requirements.txt
3. Run the Streamlit app
   ```
   streamlit run ui/app.py
4. Share to the internet with Ngrok:
   See deployment/ngrok_setup.txt for step-by-step instructions.

## 🌐 Features in the App
- Users can input patient details (age, blood pressure, cholesterol, chest pain type, etc.).
- The model predicts risk of heart disease (No Risk vs. Risk Levels).
