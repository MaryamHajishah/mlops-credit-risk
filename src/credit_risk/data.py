"""Download and load the German Credit dataset (OpenML 'credit-g', 1,000 loan applicants)."""
from pathlib import Path

import pandas as pd

NUMERIC = ["duration", "credit_amount", "installment_commitment", "residence_since",
           "age", "existing_credits", "num_dependents"]
CATEGORICAL = ["checking_status", "credit_history", "purpose", "savings_status", "employment",
               "personal_status", "other_parties", "property_magnitude", "other_payment_plans",
               "housing", "job", "own_telephone", "foreign_worker"]
TARGET = "class"


def download(path: str = "data/credit.csv") -> Path:
    from sklearn.datasets import fetch_openml

    frame = fetch_openml("credit-g", version=1, as_frame=True, parser="auto").frame
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in CATEGORICAL + [TARGET]:
        df[col] = df[col].astype(str)
    for col in NUMERIC:
        df[col] = df[col].astype(float)  # floats keep the model signature valid if a value is missing
    return df


if __name__ == "__main__":
    print(f"Saved {download()}")
