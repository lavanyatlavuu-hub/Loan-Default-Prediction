# Loan Default Prediction
# Tools: Python, Pandas, Matplotlib, Scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("loan_default.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:", data.shape)

print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Features and target
X = data.drop(columns=["applicant_id", "loan_default"])
y = data["loan_default"]

numeric_features = [
    "age", "annual_income", "loan_amount", "credit_score",
    "loan_term_months", "existing_loans", "debt_to_income_ratio"
]

categorical_features = [
    "employment_status", "home_ownership"
]

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Classification model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Repaid", "Defaulted"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new loan applicant
new_applicant = pd.DataFrame([{
    "age": 35,
    "annual_income": 48000,
    "loan_amount": 90000,
    "credit_score": 590,
    "employment_status": "Self-employed",
    "loan_term_months": 60,
    "existing_loans": 3,
    "debt_to_income_ratio": 0.48,
    "home_ownership": "Rent"
}])

prediction = model.predict(new_applicant)[0]
default_probability = model.predict_proba(new_applicant)[0][1]

print("\nExample Loan Prediction:",
      "Potential Default" if prediction == 1 else "Likely Repayment")
print(f"Default Probability: {default_probability * 100:.2f}%")

# Feature importance from logistic regression coefficients
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["classifier"].coef_[0]

importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "absolute_importance": abs(coefficients)
}).sort_values("absolute_importance", ascending=False)

print("\nTop Factors Related to Loan Default:")
print(importance.head(10)[["feature", "coefficient"]])

# Visualization: credit score vs loan amount
plt.figure(figsize=(8, 5))
plt.scatter(
    data["credit_score"],
    data["loan_amount"],
    c=data["loan_default"],
    alpha=0.6
)
plt.xlabel("Credit Score")
plt.ylabel("Loan Amount")
plt.title("Credit Score and Loan Amount vs Default")
plt.tight_layout()
plt.savefig("credit_score_vs_loan_amount.png", dpi=150)
plt.show()

# Visualization: default rate by employment status
default_by_employment = (
    data.groupby("employment_status", dropna=False)["loan_default"].mean() * 100
)

plt.figure(figsize=(8, 5))
default_by_employment.plot(kind="bar")
plt.xlabel("Employment Status")
plt.ylabel("Default Rate (%)")
plt.title("Loan Default Rate by Employment Status")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("default_by_employment.png", dpi=150)
plt.show()
