import pytest

from credit_risk.validate import DataValidationError, validate


def test_valid_frame_passes(frame):
    stats = validate(frame)
    assert stats["rows"] == len(frame)
    assert 0 < stats["positive_rate"] < 1


def test_missing_column_fails(frame):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate(frame.drop(columns=["age"]))


def test_out_of_range_age_fails(frame):
    frame.loc[0, "age"] = 7
    with pytest.raises(DataValidationError, match="age"):
        validate(frame)


def test_nulls_fail(frame):
    frame.loc[0, "duration"] = None
    with pytest.raises(DataValidationError, match="null"):
        validate(frame)
