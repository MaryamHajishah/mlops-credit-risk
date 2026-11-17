# Results

What was actually run, and what it produced. Runs were made on 17 Nov 2026.

## Environment

- CPU laptop (Intel Core i7-10510U), no GPU
- Python 3.10.12, mlflow 2.16.2, scikit-learn 1.7.2, SQLAlchemy 2.0.58
- MLflow backend: local SQLite (`mlflow.db`) with artifacts in `mlruns/`
- Data: OpenML `credit-g`, 1,000 rows, 30% labelled `bad`
- Metrics are on the held-out 20% test split (stratified, `random_state` 42). Threshold 0.5 for F1, precision and recall.

## Training runs

Quality gate: `min_roc_auc` 0.70. Promotion needs a strictly higher ROC AUC than the current champion.

| Run | Model | ROC AUC | F1 | Precision | Recall | Registered | Promoted |
|---|---|---|---|---|---|---|---|
| 1 | gradient boosting (200 trees, lr 0.05, depth 3) | 0.794 | 0.623 | 0.717 | 0.550 | v1 | yes |
| 2 | gradient boosting, same params | 0.794 | 0.623 | 0.717 | 0.550 | v2 | no, tie with v1 |
| 3 | logistic regression (C 1.0) | 0.804 | 0.593 | 0.667 | 0.533 | v3 | yes |

Champion after the three runs: `credit-risk` version 3 (logistic regression, ROC AUC 0.804).

Run 2 shows the promotion rule working: the same model scored the same, so the alias stayed on v1.
Run 3 has a higher ROC AUC but lower F1 than run 1. Promotion uses ROC AUC only.

## API check

`make serve`, then `POST /predict` with the first row of `data/credit.csv` (all 20 features, `class` removed):

```json
{"default_probability": 0.0377, "decision": "approve"}
```

The same request with `"age": 5` returns HTTP 422, because the schema accepts ages 18 to 100.

## Docker image

Built locally from `Dockerfile` (`python:3.11-slim`, about 940 MB). It was run with the project folder
mounted and `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`. `/health` returned 200 and the same request
returned the same response as above.

The first build failed to load the model: a fresh install pulled SQLAlchemy 2.1, which mlflow 2.16.2
cannot import. `requirements.txt` now pins `sqlalchemy<2.1`.

## Not run

- A remote MLflow tracking server (only the local SQLite backend was used)
- Kubernetes or any other cluster deployment
- Drift monitoring and scheduled retraining
