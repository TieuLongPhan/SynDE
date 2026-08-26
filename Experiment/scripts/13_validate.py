#!/usr/bin/env python3
"""Validate the committed SynDE global/local energy benchmark.

This read-only check does not load the large feature
caches, fit a model, or use external-validation labels. Work-directory paths
inside the JSON records are retained as run provenance; this script resolves
their release counterparts through the canonical repository layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "training_molecules": 78_513,
    "training_groups": 11_993,
    "evaluation_molecules": 19_940,
    "evaluation_groups": 3_005,
    "connectivity_terms": 633,
    "one_shot_producer_sha256": "7349ab1296a617b212b3425f6d5f75b7302855089642e5c0c8355b7750843790",
}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing release artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON release artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in release artifact: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_hash(path: Path, expected: str, label: str) -> None:
    _require(path.is_file(), f"Missing {label}: {path}")
    observed = _sha256(path)
    _require(observed == expected, f"SHA256 mismatch for {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synde-result",
        type=Path,
        default=Path("Experiment/results/global_comparators/synde.json"),
    )
    parser.add_argument(
        "--comparator-result",
        type=Path,
        default=Path("Experiment/results/global_comparators/benchmark.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("Experiment/results/global_comparators/artifacts"),
    )
    parser.add_argument(
        "--energy-model",
        type=Path,
        default=Path("synde/models/synde_energy_model.json"),
    )
    args = parser.parse_args()

    synde_result = _load(args.synde_result)
    comparator_result = _load(args.comparator_result)
    frozen_model_path = PROJECT_ROOT / "synde/models/synde_frozen_model.json"
    energy_model_path = args.energy_model
    frozen_model = _load(frozen_model_path)
    energy_model = _load(energy_model_path)

    _require(
        synde_result["protocol"]
        == "synde-total-energy-amended-domain-external-validation-v2",
        "Unexpected SynDE benchmark protocol.",
    )
    _require(
        comparator_result["protocol"] == "synde-global-local-external-comparators-v1",
        "Unexpected comparator protocol.",
    )
    split = synde_result["split"]
    data = comparator_result["data"]
    for key in (
        "training_molecules",
        "training_groups",
        "evaluation_molecules",
        "evaluation_groups",
    ):
        _require(split[key] == EXPECTED[key], f"Unexpected SynDE {key}: {split[key]}")
        _require(
            data[key] == EXPECTED[key], f"Unexpected comparator {key}: {data[key]}"
        )

    _require(split["formula_disjoint"], "Formula-disjoint firewall is false.")
    _require(split["connectivity_disjoint"], "Connectivity-disjoint firewall is false.")
    _require(not split["unseen_elements"], "Release cohort contains unseen elements.")
    _require(
        comparator_result["design"]["one_prediction_for_global_and_local_evaluation"],
        "Global and local metrics do not share one prediction vector.",
    )

    frozen_weights = frozen_model.get("weights")
    energy_weights = energy_model.get("connectivity_weights")
    _require(isinstance(frozen_weights, dict), "Frozen model has no weight mapping.")
    _require(
        isinstance(energy_weights, dict), "Energy model has no connectivity weights."
    )
    _require(
        len(energy_weights) == EXPECTED["connectivity_terms"],
        f"Expected {EXPECTED['connectivity_terms']} connectivity terms.",
    )
    _require(
        frozen_weights == energy_weights,
        "Energy calibration changed the frozen weights.",
    )

    _check_hash(
        frozen_model_path,
        synde_result["data"]["formula_relative_model_sha256"],
        "frozen connectivity model",
    )
    _check_hash(
        energy_model_path,
        synde_result["energy_model"]["sha256"],
        "packaged energy model",
    )
    validation = synde_result["amended_connectivity_equation_validation"]
    _require(
        validation["external_performance"]["groups"] == EXPECTED["evaluation_groups"],
        "Amended connectivity validation group count differs.",
    )
    _require(
        validation["external_performance"]["molecules"]
        == EXPECTED["evaluation_molecules"],
        "Amended connectivity validation molecule count differs.",
    )
    _require(
        synde_result["program_sha256"] == EXPECTED["one_shot_producer_sha256"],
        "One-shot SynDE producer hash differs from the frozen provenance record.",
    )

    artifact_dir = args.artifact_dir
    for key in ("ecfp4_ridge", "rdk5_ridge", "maccs_ridge"):
        model = comparator_result["models"][key]
        artifact = artifact_dir / "fingerprints" / Path(model["artifact"]).name
        _check_hash(artifact, model["artifact_sha256"], key)
        _require(
            model["composition_calibration"]["training_only"],
            f"{key} calibration is not train-only.",
        )

    chemprop = comparator_result["models"]["chemprop_dmpnn"]
    for fold in chemprop["folds"]:
        checkpoint = (
            artifact_dir
            / "chemprop"
            / f"fold_{fold['validation_fold']}"
            / "model/model_0/best.pt"
        )
        _check_hash(
            checkpoint,
            fold["checkpoint_sha256"],
            f"Chemprop fold {fold['validation_fold']}",
        )
    calibration_path = artifact_dir / "chemprop/composition_calibration.json"
    _require(
        _load(calibration_path) == chemprop["composition_calibration"],
        "Chemprop calibration artifact differs from benchmark record.",
    )
    _require(
        chemprop["composition_calibration"]["training_only"],
        "Chemprop calibration is not train-only.",
    )

    synde_global = synde_result["cross_formula_metrics"][
        "composition_plus_synde_connectivity"
    ]
    synde_local = synde_result["new_model_same_formula_ranking"]
    for model_name, metrics in {
        "SynDE": {"global": synde_global, "local": synde_local},
        **{
            name: model["metrics"]
            for name, model in comparator_result["models"].items()
        },
    }.items():
        _require(
            metrics["global"]["molecules"] == EXPECTED["evaluation_molecules"],
            f"{model_name} global cohort differs.",
        )
        if model_name != "maccs_ridge":
            _require(
                metrics["local"]["groups"] == EXPECTED["evaluation_groups"],
                f"{model_name} local cohort differs.",
            )

    summary = {
        "status": "valid",
        "training": [EXPECTED["training_molecules"], EXPECTED["training_groups"]],
        "external_validation": [
            EXPECTED["evaluation_molecules"],
            EXPECTED["evaluation_groups"],
        ],
        "connectivity_terms": EXPECTED["connectivity_terms"],
        "synde_global_mae_rmse_eV": [synde_global["mae_eV"], synde_global["rmse_eV"]],
        "synde_local_pearson_spearman": [
            synde_local["mean_pearson"],
            synde_local["mean_spearman"],
        ],
        "verified_comparator_artifacts": 8,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
