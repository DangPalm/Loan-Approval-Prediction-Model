# Loan-Approval-Prediction-Model
Machine learning model for predicting loan approval
Overview

This project is part of the Kaggle Playground October 2024 Challenge. The goal is to predict whether an applicant is approved for a loan based on their application details. This README provides an overview of the project workflow, key learnings, and outcomes.

Objective

The primary objective of this challenge is to:

Predict the probability of loan approval for applicants using given features.
Achieve a high score based on the ROC AUC evaluation metric.
Evaluation

Submissions are evaluated using the Area Under the ROC Curve (AUC). This metric measures how well the model distinguishes between approved and rejected applicants, based on predicted probabilities compared to ground truth labels.

Key Learnings

During this project, I implemented several key techniques and tools:

Handling class imbalance: Applied SMOTE (Synthetic Minority Oversampling Technique) and scale_pos_weight parameter in algorithms.
Predicting probabilities: Leveraged predict_proba for Gradient Boosting models to generate accurate probability outputs.
Pipeline and Column Transformer: Used for efficient feature preprocessing and model training.
Workflow

The following steps summarize the end-to-end process for this project:

Preparation:
Loaded and cleaned the dataset.
Addressed missing values and ensured proper formatting of numerical and categorical features.
Exploratory Data Analysis (EDA):
Analyzed data distribution, correlations, and relationships between features.
Identified important features such as credit history, income, and loan amount.
Feature Engineering:
Balanced the dataset using SMOTE for minority class oversampling.
Applied one-hot encoding and other transformations using Column Transformer.
Model Training:
Experimented with multiple algorithms:
Gradient Boosting (final model).
Random Forest, Logistic Regression (for comparison).
Tuned hyperparameters to optimize ROC AUC score.
AutoML with LightAutoML:
Used LightAutoML for automated feature selection and rapid experimentation.
Evaluation and Ensemble:
Evaluated models on validation data and ensembled predictions for better results.
Final Submission:
Generated predictions on the test dataset and prepared the submission file.
Achievements

Achieved a ROC AUC score of 0.98 on the public leaderboard.
Successfully placed in the Top 10% of participants during the competition.
For more insights, read my detailed analysis on Medium.
Reproducibility

To replicate the results, follow these steps:

Clone the repository:
git clone https://github.com/yourusername/loan-approval-prediction.git
cd loan-approval-prediction
Install dependencies:
pip install -r requirements.txt
Run the Jupyter Notebook:
Open LoanApprovalPrediction.ipynb and execute all cells to preprocess data, train models, and generate predictions.
Submit predictions:
Generate the submission file using the notebook and upload it to Kaggle.
Dataset

Source: Kaggle Playground October 2024 Challenge.
Size: 10,000 rows and 15 features for training; 39,098 rows for testing.
Target: Binary classification (1 = Approved, 0 = Rejected).
Important Features:
ApplicantIncome, LoanAmount, Credit_History, Dependents.
File Structure

loan-approval-prediction/
├── data/
│   ├── train.csv                # Training dataset
│   ├── test.csv                 # Test dataset
│   └── sample_submission.csv    # Kaggle submission format
├── notebooks/
│   └── LoanApprovalPrediction.ipynb  # Main notebook
├── submission.csv               # Final submission file
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
Technologies Used

Programming: Python
Libraries:
Pandas, NumPy: Data manipulation and analysis.
Scikit-learn: Model training and evaluation.
LightAutoML: Automated machine learning.
Matplotlib, Seaborn: Data visualization.
Contact

For any questions or collaboration opportunities, feel free to contact me:

Name: Your Name
Email: your.email@example.com
LinkedIn: Your LinkedIn Profile
Medium Blog: Detailed Analysis
