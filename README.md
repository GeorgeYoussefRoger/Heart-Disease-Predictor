# 🫀 Heart Disease Predictor

A machine learning project that predicts the likelihood of heart disease using the UCI Heart Disease dataset. This project demonstrates a complete end-to-end ML workflow including data preprocessing, dimensionality reduction, model evaluation, feature selection, hyperparameter tuning, and deployment via Streamlit with Ngrok for public access.

## 🚀 Features

- Prediction App
  - Interactive Streamlit web app for heart disease risk prediction
  - Uses the best-performing trained model selected after extensive experimentation
  - Clean, user-friendly input interface with real-time predictions

- Data Visualization Dashboard
  - Interactive exploration of heart disease trends using `Plotly`
  - Cholesterol vs Max Heart Rate relationship
  - Disease distribution by sex
  - Disease distribution by chest pain type
  - Age distribution across disease levels

## 🧠 Methodology

- Data Preprocessing
  - Removed missing values and handled inconsistencies
  - Encoded categorical variables using One-Hot Encoding
  - Visualized distributions and detected outliers via boxplots

- Dimensionality Reduction (PCA)
  - Applied Principal Component Analysis to reduce feature space while maintaining >90% variance
  - Determined optimal components through cumulative variance plots

- Feature Selection
  - Used RandomForest / XGBoost feature importance, Recursive Feature Elimination (RFE), and Chi-Square test
  - Identified an optimal subset of features for best model performance

- Supervised Learning
  - Trained and evaluated multiple classifiers:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
  - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

- Unsupervised Learning
  - Applied KMeans (Elbow & Silhouette methods) and Hierarchical Clustering
  - Compared clusters with actual labels using ARI and crosstab analysis

- Hyperparameter Tuning
  - Optimized models with GridSearchCV and RandomizedSearchCV
  - Selected the best-performing model (Logistic Regression) for deployment

## 📂 Project Structure

```
Heart_Disease_Project/
├── data/
│ ├── heart+disease                      # UCI dataset
│ ├── cleaned_heart_disease.csv          # Processed data after cleaning & preprocessing
│ ├── pca_heart_disease.csv              # Data after PCA
│ └── selected_feature_heart_disease.csv # Data with optimized feature subset
├── notebooks/
│ ├── 01_data_preprocessing.ipynb        # Data cleaning & preprocessing
│ ├── 02_pca_analysis.ipynb              # PCA & dimensionality reduction
│ ├── 03_feature_selection.ipynb         # Feature selection techniques
│ ├── 04_supervised_learning.ipynb       # Classification models
│ ├── 05_unsupervised_learning.ipynb     # Clustering (KMeans, Hierarchical)
│ └── 06_hyperparameter_tuning.ipynb     # GridSearch & RandomizedSearch
├── models/
│ └── final_model.pkl                    # Tuned best-performing model
├── ui/
│ └── app.py                             # Streamlit app for deployment
├── deployment/
│ └── ngrok_setup.txt                    # Instructions for sharing app via ngrok
├── results/
│ └── evaluation_metrics.txt             # Accuracy, F1, AUC scores
├── requirements.txt                     # Dependencies
├── README.md
├── .gitignore
└── LICENSE
```

## 🧰 Technologies Used

- Language: Python (3.12)
- Data Analysis: `NumPy` `Pandas`
- Data Visualization: `Matplotlib` `Seaborn` `Plotly`
- Machine Learning: `Scikit-Learn` `XGBoost` `SciPy`
- Model Persistence: `Joblib`
- Web UI and Deployment: `Streamlit` `Ngrok`

## 📦 Installation & Usage

1. Clone the repository

```
git clone https://github.com/GeorgeYoussefRoger/Heart-Disease-Predictor.git
cd Heart-Disease-Predictor
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the Streamlit app

```
streamlit run ui/app.py
```

4. Share your app publicly (optional)
Follow the steps in `deployment/ngrok_setup.txt` to share your app using Ngrok.

## 📂 Dataset

- Source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Used Subset: Cleveland database

## 📜 License

- This project is licensed under the MIT License.
- See the `LICENSE` file for details.
