# Loan Approval Predictor

A machine learning project that predicts whether a loan application will be approved based on applicant details like income, credit history, education, and loan amount. Includes a trained classification model and an interactive Streamlit web app.

## Demo

Run the app locally to get a form-based UI where you enter applicant details and instantly see an Approved/Rejected prediction with a confidence score.

## Project Structure

loan-approval-predictor/
├── data/
│   └── loan_data.csv          # Training dataset
├── models/
│   ├── loan_model.pkl         # Trained Logistic Regression model
│   ├── feature_columns.pkl    # Feature column order
│   └── scaler.pkl             # StandardScaler for numeric features
├── src/
│   ├── preprocess.py          # Data cleaning and encoding
│   ├── train.py                # Model training and evaluation
│   ├── predict.py              # Single prediction (CLI)
│   └── test_scenarios.py       # Batch scenario testing
├── app.py                      # Streamlit web UI
└── requirements.txt

## Setup

Clone the repo:

    git clone https://github.com/Hardikpateriya2025/loan-approval-predictor.git
    cd loan-approval-predictor

Create and activate a virtual environment:

    python3 -m venv venv
    source venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## Usage

Retrain the model:

    cd src
    python train.py

Run a single prediction from the command line:

    cd src
    python predict.py

Run the web app:

    streamlit run app.py

Then open the local URL shown in your terminal (usually http://localhost:8501).

## Model Details

- Algorithm: Logistic Regression (also compares against Random Forest during training)
- Accuracy: ~84% on held-out test data
- Key features: Credit history is the strongest predictor, followed by marital status, education, and loan amount relative to income
- Numeric features (income, loan amount, loan term) are scaled with StandardScaler before training so all features contribute meaningfully to the model.

## Dataset

Based on the classic Loan Prediction dataset, containing applicant details such as:
- Gender, Marital Status, Dependents, Education, Self-Employment status
- Applicant/Coapplicant Income
- Loan Amount and Term
- Credit History
- Property Area

## License

This project is for educational purposes.
