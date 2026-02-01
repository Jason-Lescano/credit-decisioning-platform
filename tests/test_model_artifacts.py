from pathlib import Path
import joblib


def test_model_artifact_exists_and_has_expected_keys():
    model_path = Path("artifacts/models/lgbm_model.joblib")
    assert model_path.exists(), "Expected model artifact at artifacts/models/lgbm_model.joblib"

    bundle = joblib.load(model_path)
    assert "model" in bundle
    assert "feature_names" in bundle
    assert isinstance(bundle["feature_names"], list)
    assert len(bundle["feature_names"]) > 0
