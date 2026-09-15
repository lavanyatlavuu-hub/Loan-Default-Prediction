# Loan-Default-Prediction
Loan Default Prediction is a machine learning project that predicts whether a loan applicant is likely to repay or default based on income, loan amount, credit score, employment status, existing loans, and debt-to-income ratio. The project uses Logistic Regression with Python and Scikit-learn.
# Loan Default Prediction

## Overview

This project uses machine learning to predict whether a loan applicant is likely to repay a loan or default. It analyzes applicant details such as income, loan amount, credit score, employment status, existing loans, and debt-to-income ratio.

## Objectives

* Analyze loan application data
* Handle missing values
* Prepare data for machine learning
* Train a classification model
* Predict loan default
* Evaluate model performance
* Identify important factors related to loan default

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Machine Learning Algorithm

Logistic Regression is used to classify applicants into two categories:

0 = Likely Repayment

1 = Potential Default

## Dataset Features

* Age
* Annual Income
* Loan Amount
* Credit Score
* Employment Status
* Loan Term
* Existing Loans
* Debt to Income Ratio
* Home Ownership

## Methodology

1. Load the loan dataset
2. Check and handle missing values
3. Separate features and target
4. Standardize numerical features
5. Encode categorical features
6. Split data into training and testing sets
7. Train the Logistic Regression model
8. Generate predictions
9. Evaluate the model using accuracy, classification report, and confusion matrix
10. Predict default probability for a new applicant

## Visualizations

The project generates visualizations for:

* Credit Score versus Loan Amount
* Loan Default Rate by Employment Status

## How to Run

Install the required packages:

pip install -r requirements.txt

Run the project:

python loan_default_prediction.py

## Dataset Note

The dataset is synthetic and intended for educational and machine learning practice. It does not contain real applicant information.

## Conclusion

This project demonstrates how machine learning can be used to predict loan default risk and identify factors that may influence loan repayment. The model is intended for educational purposes and should not be used as the sole basis for real-world lending decisions.
