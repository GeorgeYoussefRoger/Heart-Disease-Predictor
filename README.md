# 🫀 Heart Disease Predictor

An end-to-end machine learning project for predicting heart disease using the UCI Heart Disease dataset.

The project was built to demonstrate a complete classical machine learning workflow from data preprocessing and exploratory analysis to feature selection, dimensionality reduction, model comparison, hyperparameter tuning and an interactive prediction application.

- Data preprocessing with missing-value handling, one-hot encoding and Min-Max scaling
- Feature selection using XGBoost, Random Forest, RFE and Chi-Square
- Dimensionality reduction analysis using PCA
- Comparison of Logistic Regression, Decision Tree, Random Forest and SVM
- Hyperparameter optimization with GridSearchCV and RandomizedSearchCV
- Reusable preprocessing and model pipeline
- Unsupervised pattern discovery using K-Means and Hierarchical Clustering
- Interactive predictions and data visualization through Streamlit

> Dataset: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

- Used Cleveland subset
- This project is for educational purposes only and is not intended to provide medical diagnosis or advice.

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `SciPy` `XGBoost` `Matplotlib` `Seaborn` `Plotly` `Streamlit`

## Methodology

- Data Preprocessing
  - Removed missing values and handled inconsistencies
  - Encoded categorical variables using One-Hot Encoding
  - Scaled numerical features using Min-Max Scaling
  - Visualized distributions and detected potential outliers using boxplots
- Dimensionality Reduction (PCA)
  - Applied Principal Component Analysis to reduce the feature space while maintaining 96% of the variance
  - Determined the number of components using cumulative explained variance
- Feature Selection
  - Used Random Forest and XGBoost feature importance
  - Applied Recursive Feature Elimination (RFE)
  - Applied the Chi-Square statistical test
  - Selected features that were consistently identified by multiple feature selection methods
- Supervised Learning
  - Trained and evaluated multiple classifiers:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
  - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Unsupervised Learning
  - Applied K-Means clustering and the Elbow Method
  - Applied Hierarchical Clustering with dendrogram analysis
  - Compared discovered clusters with the actual disease labels
- Hyperparameter Tuning
  - Optimized models using GridSearchCV and RandomizedSearchCV
  - Compared baseline and tuned model performance
  - Selected Logistic Regression as the final deployment model based on performance, lower complexity and faster inference

## Getting Started

### Prerequisites

- Python 3.12+

### Installation

Clone the repository

```
git clone https://github.com/GeorgeYoussefRoger/Heart-Disease-Predictor.git
cd Heart-Disease-Predictor
```

Create a Virtual Environment

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

### Usage

Run the Streamlit app

```
streamlit run ui/app.py
```

Access UI: http://localhost:8501

Share your app publicly (optional)

- Follow the steps in `deployment/ngrok_setup.txt` to share your app using Ngrok

## License

This project is licensed under the MIT License.
