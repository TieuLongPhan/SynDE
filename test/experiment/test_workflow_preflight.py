from importlib import import_module
from pathlib import Path

import pytest

_require_inputs = import_module("Experiment.scripts.07_train_energy")._require_inputs
_load = import_module("Experiment.scripts.13_validate")._load


def test_training_preflight_reports_all_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        _require_inputs(
            {
                "training labels": tmp_path / "training.csv",
                "test labels": tmp_path / "test.csv",
            }
        )

    message = str(caught.value)
    assert "training labels" in message
    assert "test labels" in message
    assert "03_run_xtb.py" in message


def test_training_preflight_accepts_existing_files(tmp_path: Path) -> None:
    source = tmp_path / "training.csv"
    source.touch()

    assert _require_inputs({"training labels": source}) == {"training labels": source}


def test_release_loader_rejects_missing_or_invalid_json(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not JSON", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing release artifact"):
        _load(missing)
    with pytest.raises(ValueError, match="Invalid JSON release artifact"):
        _load(invalid)
