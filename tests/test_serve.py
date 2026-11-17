import importlib

from fastapi.testclient import TestClient

from credit_risk.train import run
from conftest import synthetic_frame


def test_predict_uses_champion(workspace, monkeypatch):
    run(str(workspace))
    monkeypatch.setenv("MODEL_URI", "models:/credit-risk@champion")
    import credit_risk.serve as serve
    importlib.reload(serve)
    client = TestClient(serve.app)

    assert client.get("/health").status_code == 200
    applicant = synthetic_frame(1).drop(columns=["class"]).iloc[0].to_dict()
    body = client.post("/predict", json=applicant).json()
    assert 0 <= body["default_probability"] <= 1
    assert body["decision"] in {"approve", "review"}


def test_invalid_input_is_rejected(workspace, monkeypatch):
    import credit_risk.serve as serve
    client = TestClient(serve.app)
    applicant = synthetic_frame(1).drop(columns=["class"]).iloc[0].to_dict()
    applicant["age"] = 5
    assert client.post("/predict", json=applicant).status_code == 422
