"""Validate data, train a model, log everything to MLflow, and promote it only if it passes the gate."""
import argparse

import mlflow
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, load
from .validate import validate


def build_model(cfg: dict) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    kind = cfg["type"]
    if kind == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=0, **cfg["gradient_boosting"])
    elif kind == "logistic_regression":
        clf = LogisticRegression(**cfg["logistic_regression"])
    else:
        raise ValueError(f"unknown model type {kind}")
    return Pipeline([("pre", pre), ("clf", clf)])


def promote(client: MlflowClient, name: str, version: str, alias: str, auc: float) -> bool:
    """Move the alias to the new version only if it beats the current champion."""
    try:
        current = client.get_model_version_by_alias(name, alias)
        best = client.get_run(current.run_id).data.metrics["roc_auc"]
    except Exception:
        best = -1.0
    if auc > best:
        client.set_registered_model_alias(name, alias, version)
        return True
    return False


def run(params_path: str = "params.yaml") -> dict:
    p = yaml.safe_load(open(params_path))
    d, m, g = p["data"], p["model"], p["promotion"]
    df = load(d["path"])
    stats = validate(df, d["positive_label"])

    X, y = df.drop(columns=[d["target"]]), (df[d["target"]] == d["positive_label"]).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=d["test_size"], random_state=d["random_state"], stratify=y)

    mlflow.set_experiment("credit-risk")
    with mlflow.start_run() as active:
        mlflow.log_params({"model_type": m["type"], **m[m["type"]], **stats})
        model = build_model(m).fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y_te, proba),
            "f1": f1_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", signature=infer_signature(X_te, proba),
                                 input_example=X_te.head(2))

        result = {"run_id": active.info.run_id, **metrics, "registered": False, "promoted": False}
        if metrics["roc_auc"] < g["min_roc_auc"]:
            mlflow.set_tag("gate", "failed")
            print(f"Gate failed: roc_auc {metrics['roc_auc']:.3f} < {g['min_roc_auc']}")
            return result
        mlflow.set_tag("gate", "passed")
        mv = mlflow.register_model(f"runs:/{active.info.run_id}/model", g["registered_model"])
        result["registered"] = True
        result["version"] = mv.version
        result["promoted"] = promote(MlflowClient(), g["registered_model"], mv.version,
                                     g["alias"], metrics["roc_auc"])
    print(result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    run(ap.parse_args().params)
