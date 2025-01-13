# Loan-Approval-Prediction-Model
Machine learning model for predicting loan approval

## Overview
This project is part of the Kaggle Playground October 2024 Challenge. The goal is to predict whether a loan is approved based on their application information. This README provides an overview of the project workflow, key learnings, and outcomes.

## Objective
The primary objective of this challenge is to:
- Predict the probability of loan approval for applicants using given features.
- Achieve a high score based on the ROC AUC evaluation metric.

## Evaluation
Submissions are evaluated using the Area Under the ROC Curve (AUC). This metric measures how well the model identifies approved and rejected applicants, based on predicted probabilities compared to ground truth labels.

## Key Learnings
This project marked one of my first hand-on experience with machine learning, in which I utilized several newly learned techniques and tools:
- Exploratory Data Analysis: Utilized matplotlib and seaborn libraries to produce bar charts, box plots, heatmaps, and more.
- Pipeline and Column Transformer: Used for efficient feature preprocessing.
- Handling class imbalance: Experimented with SMOTE (Synthetic Minority Oversampling Technique) and applied scale_pos_weight parameter in models.
- Optimizing hyperparameters: Utilized optuna to find models' best hyperparameters combination.
- Predicting probabilities: Utilized predict_proba for Random Forest and Gradient Boosting (XGB, CatBoost, LightGBM) models to generate accurate probability outputs.

## Workflow
The following steps summarize the end-to-end process for this project:
### Preparation:
- Loaded libraries and datasets
### Exploratory Data Analysis (EDA):
- Analyzed data distribution, correlations, and relationships between features.
### Split Data:
- Split train data into train and validation data.
### Preprocessing:
- Handled missing values and ensured proper data format.
### Feature Engineering:
- Balanced the dataset using SMOTE for minority class oversampling.
- Applied one-hot encoding and other transformations using Column Transformer.
### Model Training:
- Experimented with multiple algorithms: Random Forest, XGB, CatBoost, and LightGBM.
- Tuned hyperparameters to optimize ROC AUC score.
### Evaluation and Ensemble:
- Ensembled predictions using 2 methods: Soft Voting and Weighted Average.
- Evaluated ensemble methods.
### Submission:
- Generated predictions on the test dataset and prepared the submission file.

## Achievements:
Achieved a ROC AUC score of 0.96 on the public leaderboard.

## Reproducibility
To replicate the results, follow these steps:
1. Clone the repository:
git clone https://github.com/DangPalm/Loan-Approval-Prediction-Model.git
cd loan-approval-prediction
2. Install dependencies:
pip install -r requirements.txt
3. Run the Jupyter Notebook:
Open LoanApprovalPrediction.ipynb and execute all cells to preprocess data, train models, and generate predictions.
4. Submit predictions:
Generate the submission file using the notebook and upload it to Kaggle.

## Dataset
Source: Kaggle Playground October 2024 Challenge.
Size: 58,645 rows and 12 features for training; 39,098 rows for testing.
Target: Binary classification (1 = Approved, 0 = Rejected).

## Technologies Used
Programming: Python<br>
pandas, numpy: Data manipulation and numerical computations.<br>
matplotlib, seaborn: Data visualization.<br>
scikit-learn: Preprocessing, pipelines, and machine learning models.<br>
xgboost, catboost, lightgbm: Gradient boosting frameworks.<br>
optuna: Hyperparameter optimization.<br>

## Contact
For any questions or collaboration opportunities, feel free to contact me: <br>
Name: Dang Pham<br>
Email: pham2dh@mail.uc.edu<br>
LinkedIn: www.linkedin.com/in/phamhaidang<br>
