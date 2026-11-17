"""HTTP service that serves the current champion model from the MLflow registry."""
import os
from functools import lru_cache

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_URI = os.getenv("MODEL_URI", "models:/credit-risk@champion")
app = FastAPI(title="credit-risk")


class Applicant(BaseModel):
    checking_status: str
    duration: float = Field(ge=0)
    credit_history: str
    purpose: str
    credit_amount: float = Field(ge=0)
    savings_status: str
    employment: str
    installment_commitment: float = Field(ge=0)
    personal_status: str
    other_parties: str
    residence_since: float = Field(ge=0)
    property_magnitude: str
    age: float = Field(ge=18, le=100)
    other_payment_plans: str
    housing: str
    existing_credits: float = Field(ge=0)
    job: str
    num_dependents: float = Field(ge=0)
    own_telephone: str
    foreign_worker: str


@lru_cache(maxsize=1)
def model():
    return mlflow.sklearn.load_model(MODEL_URI)


@app.get("/health")
def health():
    try:
        model()
        return {"status": "ok", "model_uri": MODEL_URI}
    except Exception as exc:  # model registry not reachable or alias missing
        raise HTTPException(503, f"model not loaded: {exc}")


@app.post("/predict")
def predict(applicant: Applicant):
    frame = pd.DataFrame([applicant.model_dump()])
    risk = float(model().predict_proba(frame)[0, 1])
    return {"default_probability": round(risk, 4), "decision": "review" if risk >= 0.5 else "approve"}
