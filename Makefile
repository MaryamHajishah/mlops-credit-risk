export PYTHONPATH := src
export MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db

.PHONY: install data train ui serve test

install: ; pip install -r requirements.txt
data:    ; python -m credit_risk.data
train:   ; python -m credit_risk.train
ui:      ; mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI)
serve:   ; uvicorn credit_risk.serve:app --reload
test:    ; pytest -q
