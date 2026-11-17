import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.data import CATEGORICAL, NUMERIC  # noqa: E402

LEVELS = {c: [f"{c}_{i}" for i in range(3)] for c in CATEGORICAL}


def synthetic_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Same schema as credit-g, with a learnable signal so the quality gate can pass."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(LEVELS[c], n) for c in CATEGORICAL})
    df["duration"] = rng.integers(4, 72, n)
    df["credit_amount"] = rng.integers(250, 18000, n)
    df["installment_commitment"] = rng.integers(1, 5, n)
    df["residence_since"] = rng.integers(1, 5, n)
    df["age"] = rng.integers(19, 75, n)
    df["existing_credits"] = rng.integers(1, 4, n)
    df["num_dependents"] = rng.integers(1, 3, n)
    risk = 0.04 * df["duration"] + 0.0001 * df["credit_amount"] - 0.03 * df["age"]
    risk += (df["checking_status"] == "checking_status_0") * 1.5
    df["class"] = np.where(risk + rng.normal(0, 0.5, n) > risk.median(), "bad", "good")
    return df[NUMERIC + CATEGORICAL + ["class"]]


@pytest.fixture
def frame():
    return synthetic_frame()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A params file, a CSV and a private MLflow backend for one test."""
    import mlflow

    csv = tmp_path / "credit.csv"
    synthetic_frame().to_csv(csv, index=False)
    params = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "params.yaml"))
    params["data"]["path"] = str(csv)
    params["model"]["gradient_boosting"]["n_estimators"] = 50
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    monkeypatch.chdir(tmp_path)
    return path
