# Loan Default Prediction

## Synopsis
This project uses machine learning to predict whether a person may fail to repay a loan. It analyzes income, loan amount, credit history, employment status, debt-to-income ratio, and other applicant information.

## Objectives
- Explore loan application data.
- Analyze factors related to loan repayment.
- Prepare the data for machine learning.
- Train a classification model.
- Predict loan default.
- Check the accuracy of the model.

## Tools
Python, Pandas, Matplotlib, Scikit-learn

## Dataset Features
- applicant_id
- age
- annual_income
- loan_amount
- credit_score
- employment_status
- loan_term_months
- existing_loans
- debt_to_income_ratio
- home_ownership
- loan_default

Target:
- 0 = Likely Repayment
- 1 = Potential Default

## Data Preparation
- Loaded the loan application dataset using Pandas.
- Checked for missing values.
- Filled missing numerical values with median values.
- Filled missing categorical values with the most frequent value.
- Standardized numerical features.
- One-hot encoded categorical features.
- Split the dataset into training and testing sets.

## Machine Learning Algorithm
Logistic Regression is used for binary classification.

## Evaluation
The program calculates:
- Accuracy
- Classification report
- Confusion matrix
- Default probability for a new applicant
- Important model factors using logistic-regression coefficients

It also generates visualizations for credit score versus loan amount and default rate by employment status.

## How to Run
1. Install Python.
2. Open a terminal in this project folder.
3. Install dependencies:
   pip install -r requirements.txt
4. Run:
   python loan_default_prediction.py

## Dataset Note
The included dataset is synthetic and intended for educational/classroom machine-learning practice. It does not contain real applicant information.

## Important Note
This project is for educational purposes. Model predictions should not be used as the sole basis for real-world lending or credit decisions.
