import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")

def load_and_clean(path=None):
    if path is None:
        path = DEFAULT_PATH
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    return df

if __name__ == "__main__":
    df = load_and_clean()
    print(df.head())
    print(df.shape)
    print(df.dtypes)
