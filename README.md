# mlops-credit-risk

An end-to-end MLOps pipeline for a credit-risk model: data validation, experiment tracking, a model
registry with a quality gate, a served model, a container image, and CI.

The model predicts whether a loan applicant is a credit risk, using the public German Credit dataset
(OpenML `credit-g`, 1,000 applicants, 20 features). The model is deliberately simple. The point of the
project is everything around it.

## What happens on every training run

```
data/credit.csv ─► validate ─► train ─► log to MLflow ─► quality gate ─► register ─► promote?
                   (schema,     (sklearn   (params,        (roc_auc ≥      (new        (only if it
                    ranges,      pipeline)  metrics,        min_roc_auc)    version)    beats the
                    nulls)                  model, signature)                           champion)
```

1. **Validate.** `validate.py` checks columns, types, nulls, value ranges and the label before any
   training. A failed check stops the run.
2. **Train and track.** `train.py` fits a scikit-learn pipeline and logs parameters, data stats,
   ROC AUC, F1, precision, recall and the model with its input signature to MLflow.
3. **Gate.** A model below `min_roc_auc` in `params.yaml` is never registered.
4. **Promote.** A model that passes is registered as a new version. The `champion` alias moves to it
   only if its ROC AUC beats the current champion, so a worse model cannot replace a better one.
5. **Serve.** `serve.py` is a FastAPI service that loads `models:/credit-risk@champion`, validates
   each request against a typed schema, and returns a default probability.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
make install
make data        # downloads credit-g from OpenML into data/credit.csv
make train       # validate, train, log, gate, register, promote
make ui          # MLflow UI at http://localhost:5000 to compare runs
make serve       # API at http://localhost:8000/docs
make test        # 9 tests on synthetic data with a throwaway MLflow backend
```

Try changing `model.type` or the hyperparameters in `params.yaml`, run `make train` again, and see
whether the champion moves.

## Results

The current champion is a logistic regression with ROC AUC 0.804 on the held-out 20% test split.
It replaced a gradient boosting model at 0.794. A second run of that same gradient boosting model
was registered but not promoted, because it did not beat the champion. Details, the API check and
what was not run are in [RESULTS.md](RESULTS.md).

## CI

`.github/workflows/ci.yml` runs the tests on every push and pull request, then downloads the real
dataset, trains, and fails the build if the model does not pass the quality gate. A second job builds
the container image.

## Layout

```
src/credit_risk/  data.py  validate.py  train.py  serve.py
tests/            synthetic data with the real schema, so tests need no network
params.yaml       every setting that changes a run
Dockerfile        image for the API (built in CI)
```

## Not covered yet

Drift monitoring on live predictions, scheduled retraining, a remote MLflow tracking server, and deployment to a cluster.
