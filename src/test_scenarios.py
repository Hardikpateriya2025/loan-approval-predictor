import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict_loan

scenarios = {
    "Strong applicant": {
        "Gender": "Male", "Married": "Yes", "Dependents": "0",
        "Education": "Graduate", "Self_Employed": "No",
        "ApplicantIncome": 8000, "CoapplicantIncome": 2000,
        "LoanAmount": 120, "Loan_Amount_Term": 360,
        "Credit_History": 1.0, "Property_Area": "Urban",
    },
    "Low income, no credit history": {
        "Gender": "Female", "Married": "No", "Dependents": "0",
        "Education": "Not Graduate", "Self_Employed": "No",
        "ApplicantIncome": 1800, "CoapplicantIncome": 0,
        "LoanAmount": 100, "Loan_Amount_Term": 360,
        "Credit_History": 0.0, "Property_Area": "Rural",
    },
    "High loan amount relative to income": {
        "Gender": "Male", "Married": "Yes", "Dependents": "2",
        "Education": "Graduate", "Self_Employed": "Yes",
        "ApplicantIncome": 3000, "CoapplicantIncome": 0,
        "LoanAmount": 400, "Loan_Amount_Term": 180,
        "Credit_History": 1.0, "Property_Area": "Semiurban",
    },
    "Self-employed, good credit, moderate income": {
        "Gender": "Female", "Married": "Yes", "Dependents": "1",
        "Education": "Graduate", "Self_Employed": "Yes",
        "ApplicantIncome": 4500, "CoapplicantIncome": 1500,
        "LoanAmount": 130, "Loan_Amount_Term": 360,
        "Credit_History": 1.0, "Property_Area": "Semiurban",
    },
    "Large family, no coapplicant income": {
        "Gender": "Male", "Married": "Yes", "Dependents": "3+",
        "Education": "Not Graduate", "Self_Employed": "No",
        "ApplicantIncome": 3200, "CoapplicantIncome": 0,
        "LoanAmount": 150, "Loan_Amount_Term": 360,
        "Credit_History": 1.0, "Property_Area": "Rural",
    },
}

print("=" * 60)
for name, applicant in scenarios.items():
    print(f"\nScenario: {name}")
    predict_loan(applicant)
print("\n" + "=" * 60)
