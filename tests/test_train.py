import yaml
from mlflow.tracking import MlflowClient

from credit_risk.train import run


def test_good_model_is_registered_and_promoted(workspace):
    result = run(str(workspace))
    assert result["roc_auc"] > 0.70
    assert result["registered"] and result["promoted"]
    champion = MlflowClient().get_model_version_by_alias("credit-risk", "champion")
    assert champion.version == result["version"]


def test_gate_blocks_weak_model(workspace):
    params = yaml.safe_load(open(workspace))
    params["promotion"]["min_roc_auc"] = 0.999
    workspace.write_text(yaml.safe_dump(params))
    result = run(str(workspace))
    assert not result["registered"]


def test_worse_model_does_not_replace_champion(workspace):
    first = run(str(workspace))
    params = yaml.safe_load(open(workspace))
    params["model"]["gradient_boosting"].update(n_estimators=1, max_depth=1, learning_rate=0.01)
    params["promotion"]["min_roc_auc"] = 0.0
    workspace.write_text(yaml.safe_dump(params))
    second = run(str(workspace))
    assert second["registered"] and not second["promoted"]
    champion = MlflowClient().get_model_version_by_alias("credit-risk", "champion")
    assert champion.version == first["version"]
