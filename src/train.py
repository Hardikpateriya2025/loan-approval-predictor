import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import load_and_clean

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

def train():
    df = load_and_clean()

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))
        print(confusion_matrix(y_test, preds))
        print()

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    print(f"Best model: {best_name} with accuracy {best_score:.4f}")

    if best_name == "LogisticRegression":
        print("\nFeature coefficients (scaled):")
        for col, coef in zip(X.columns, best_model.coef_[0]):
            print(f"  {col}: {coef:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "loan_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"Saved model to {MODELS_DIR}/loan_model.pkl")

if __name__ == "__main__":
    train()
