# Loan-Approval-Predictor
### Overview

This project develops a machine learning model to predict loan approval outcomes (loan_status: 0 = non-default, 1 = default) using historical loan data. The Random Forest Classifier, optimized with RandomizedSearchCV, achieves high accuracy, minimizes default risk, promotes fairness, streamlines decision-making, and ensures interpretability.

### Objectives

+ Enhance loan approval accuracy.

+ Minimize default risk by identifying high-risk applicants.

+ Promote fairness by analyzing biases.

+ Streamline decision-making through automation.

+ Improve interpretability for stakeholders.

### Key Features

* Dataset: ~45,000 rows with features like previous_loan_defaults_on_file, loan_percent_income, credit_score, person_gender, person_education.

* Preprocessing: SMOTE for class imbalance, encoding for categorical variables, outlier handling.

* Model: Random Forest Classifier (best_rf) saved as random_forest_loan_model.joblib.

* Evaluation: Precision, recall, F1 score, ROC AUC; visualizations include Feature Importance Plot, Confusion Matrix (implied), and recommended Precision-Recall, SHAP, ROC curves.

### Key Insights

- Top predictors: previous_loan_defaults_on_file, loan_percent_income, credit_score.

- SMOTE improves default detection.

- Feature Importance Plot clarifies key factors; SHAP (recommended) enhances fairness analysis.

- Confusion Matrix (implied) supports reliable automation.

- ROC Curve (drafted) shows Random Forest outperforms Logistic Regression.

### Setup

- Clone the repository.

- Install dependencies: pip install -r requirements.txt.

- Run index.ipynb in Jupyter Notebook.

### Requirements

+ Python 3.x

+ Libraries: pandas, numpy, scikit-learn, imblearn, matplotlib, seaborn, shap, lime, joblib

### Usage

+ Load data and preprocess in index.ipynb.

+ Train model and evaluate using provided metrics and plots.

+ Deploy model with random_forest_loan_model.joblib for predictions.

### Recommendations

+ Validate SMOTE on test set; explore class weights.

+ Implement SHAP and fairness metrics for equity.

+ Optimize thresholds with Precision-Recall Curve.

+ Deploy and monitor model performance.
