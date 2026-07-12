import os
import sys
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details to predict loan approval likelihood.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150, step=10)
    loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 84, 60, 36, 12])
    credit_history = st.selectbox("Credit History", ["Has credit history", "No credit history"])

if st.button("Predict Loan Approval", type="primary"):
    applicant = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1.0 if credit_history == "Has credit history" else 0.0,
        "Property_Area": property_area,
    }

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

    st.divider()
    if pred == 1:
        st.success(f"### ✅ Loan Approved")
        st.write(f"Confidence: **{prob:.2%}**")
    else:
        st.error(f"### ❌ Loan Rejected")
        st.write(f"Confidence: **{(1-prob):.2%}**")

    st.progress(float(prob))
    st.caption(f"Approval probability: {prob:.2%}")
