# Loan-Default-Prediction
Loan Default Prediction is a machine learning project that predicts whether a loan applicant may repay or default. It analyzes factors such as income, loan amount, credit score, employment status, existing loans, and debt-to-income ratio. Logistic Regression is used for classification, with preprocessing, evaluation, and visualization techniques.
Absolutely. 

## Loan Default Prediction Using Machine Learning

### 1. Introduction

Loan default is an important problem in the financial sector. A borrower is considered to have defaulted when they fail to repay a loan according to the agreed terms. Predicting potential loan defaults can help financial institutions understand repayment risk.

The **Loan Default Prediction** project applies machine learning to predict whether a loan applicant is likely to **repay the loan or default**. The project analyzes applicant information such as age, annual income, loan amount, credit score, employment status, existing loans, debt-to-income ratio, loan term, and home ownership.

The project uses **Python, Pandas, Matplotlib, and Scikit-learn**. Logistic Regression is used as the classification algorithm because the target variable has two possible outcomes.

> **Note:** The dataset supplied with the project is synthetic and intended for educational/classroom machine-learning practice. It does not contain real applicant information.

---

# 2. Objectives

The main objectives of this project are:

1. To explore and understand loan application data.
2. To identify factors related to loan repayment and default.
3. To clean and preprocess the dataset.
4. To handle missing values in the dataset.
5. To convert categorical data into numerical form.
6. To standardize numerical features.
7. To divide the data into training and testing sets.
8. To train a Logistic Regression classification model.
9. To predict whether an applicant may default on a loan.
10. To evaluate the performance of the machine-learning model.
11. To visualize important relationships in the loan data.
12. To identify important factors associated with loan default.

---

# 3. Dataset Description

The project dataset contains **700 records and 11 columns**.

| Feature              | Description                     |
| -------------------- | ------------------------------- |
| applicant_id         | Unique applicant identification |
| age                  | Age of the applicant            |
| annual_income        | Annual income                   |
| loan_amount          | Amount of loan requested        |
| credit_score         | Applicant's credit score        |
| employment_status    | Employment category             |
| loan_term_months     | Loan repayment period           |
| existing_loans       | Number of existing loans        |
| debt_to_income_ratio | Ratio of debt to income         |
| home_ownership       | Home ownership status           |
| loan_default         | Target variable                 |

### Target Variable

* **0 = Likely Repayment**
* **1 = Potential Default**

---

# 4. Methodology

The project follows several steps to build the prediction model.

### Step 1: Data Loading

The dataset is loaded using the **Pandas** library. The program reads the `loan_default.csv` file and examines its structure.

### Step 2: Data Exploration

The dataset is examined to understand:

* Number of records
* Number of features
* Data types
* Missing values
* Initial data patterns

The dataset contains **700 rows and 11 columns**.

### Step 3: Handling Missing Values

Missing values are handled before training the model.

* Missing numerical values are replaced with the **median**.
* Missing categorical values are replaced with the **most frequent value**.

This prevents missing data from causing problems during model training.

### Step 4: Feature Selection

The applicant information is used as input features, while `loan_default` is used as the target variable.

The applicant ID is not used as a meaningful predictive feature.

### Step 5: Feature Scaling

Numerical features are standardized using **StandardScaler**.

Scaling ensures that numerical variables with different ranges can be processed effectively by the machine-learning algorithm.

### Step 6: Categorical Encoding

Categorical features such as:

* Employment status
* Home ownership

are converted into numerical representations using **One-Hot Encoding**.

### Step 7: Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

The training data is used to build the model, while the testing data is used to evaluate its performance.

### Step 8: Model Training

The project uses **Logistic Regression**.

Logistic Regression is suitable because the problem is a **binary classification problem**, where the output is either repayment or potential default.

### Step 9: Prediction

After training, the model predicts the class of unseen loan applicants.

It can also calculate the **probability of default** for an applicant.

### Step 10: Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
The project also analyzes Logistic Regression coefficients to identify important factors related to loan default.

