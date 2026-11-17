"""Data checks that run before training. A failed check stops the pipeline."""
import pandas as pd

from .data import CATEGORICAL, NUMERIC, TARGET


class DataValidationError(ValueError):
    pass


def validate(df: pd.DataFrame, positive_label: str = "bad") -> dict:
    problems = []
    missing = [c for c in NUMERIC + CATEGORICAL + [TARGET] if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if len(df) < 100:
        problems.append(f"too few rows: {len(df)}")
    nulls = df[NUMERIC + CATEGORICAL + [TARGET]].isna().sum()
    if nulls.any():
        problems.append(f"null values: {nulls[nulls > 0].to_dict()}")
    for col in NUMERIC:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col} is not numeric")
        elif (df[col] < 0).any():
            problems.append(f"{col} has negative values")
    if not df["age"].between(18, 100).all():
        problems.append("age outside 18 to 100")
    labels = set(df[TARGET].unique())
    if positive_label not in labels or len(labels) != 2:
        problems.append(f"target must be binary and contain '{positive_label}', got {labels}")
    if problems:
        raise DataValidationError("; ".join(problems))
    return {"rows": len(df), "positive_rate": float((df[TARGET] == positive_label).mean())}
