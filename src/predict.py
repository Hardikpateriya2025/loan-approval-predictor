import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "loan_model.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

MAPS = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}

def predict_loan(applicant: dict):
    row = {}
    for col in feature_columns:
        if col in MAPS:
            row[col] = MAPS[col][applicant[col]]
        else:
            row[col] = applicant[col]

    X = pd.DataFrame([row], columns=feature_columns)
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    result = "Approved" if pred == 1 else "Rejected"
    print(f"Prediction: {result} (confidence: {prob:.2%})")
    return result, prob

if __name__ == "__main__":
    sample_applicant = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 0,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
    }
    predict_loan(sample_applicant)
