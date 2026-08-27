#!/usr/bin/env python3
"""Postfit support, margin, failure, and fixed-prediction null diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.scripts._helpers import (
    _feature_family,
    correlation,
    load_calibration_seed,
    public_record,
    rank,
)
from synde.energy import V3_FEATURE_FAMILIES

PROTOCOL = "synde-ord-v4-postfit-diagnostics-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile_bins(
    values: np.ndarray, outcomes: np.ndarray, count: int
) -> list[dict[str, object]]:
    edges = np.quantile(values, np.linspace(0, 1, count + 1))
    output = []
    for index in range(count):
        mask = (values >= edges[index]) & (
            values <= edges[index + 1]
            if index == count - 1
            else values < edges[index + 1]
        )
        output.append(
            {
                "bin": index + 1,
                "lower_inclusive": float(edges[index]),
                "upper_inclusive": float(edges[index + 1]),
                "pairs": int(mask.sum()),
                "mean_pairwise_correctness": float(np.mean(outcomes[mask])),
            }
        )
    return output


def _fixed_prediction_null(
    rows: list[dict[str, object]], replicates: int
) -> dict[str, object]:
    seed = int(hashlib.sha256(f"{PROTOCOL}:null".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    pearson_null = np.empty(replicates, dtype=float)
    spearman_null = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        pearson = []
        spearman = []
        for row in rows:
            predictions = [float(value) for value in row["predictions"]]
            labels = np.asarray(row["labels"], dtype=float)
            permuted = labels[rng.permutation(len(labels))].tolist()
            pearson.append(correlation(predictions, permuted))
            spearman.append(
                correlation(rank(predictions).tolist(), rank(permuted).tolist())
            )
        pearson_null[replicate] = np.mean(pearson)
        spearman_null[replicate] = np.mean(spearman)
    observed_pearson = float(np.mean([row["pearson"] for row in rows]))
    observed_spearman = float(np.mean([row["spearman"] for row in rows]))
    return {
        "replicates": replicates,
        "scope": (
            "labels permuted within formula groups against already selected "
            "outer predictions; selection pipeline not repeated"
        ),
        "observed_pearson": observed_pearson,
        "observed_spearman": observed_spearman,
        "pearson_null_mean": float(np.mean(pearson_null)),
        "pearson_null_95_range": [
            float(np.quantile(pearson_null, 0.025)),
            float(np.quantile(pearson_null, 0.975)),
        ],
        "pearson_null_maximum": float(np.max(pearson_null)),
        "pearson_add_one_p": float(
            (np.sum(pearson_null >= observed_pearson) + 1) / (replicates + 1)
        ),
        "spearman_null_mean": float(np.mean(spearman_null)),
        "spearman_null_95_range": [
            float(np.quantile(spearman_null, 0.025)),
            float(np.quantile(spearman_null, 0.975)),
        ],
        "spearman_null_maximum": float(np.max(spearman_null)),
        "spearman_add_one_p": float(
            (np.sum(spearman_null >= observed_spearman) + 1) / (replicates + 1)
        ),
    }


def run(
    result_path: Path, model_path: Path, calibration_seed_path: Path
) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    _, seed_model = load_calibration_seed(calibration_seed_path)
    rows = result["group_predictions"]["synde"]

    margins = []
    correctness = []
    for row in rows:
        predictions = row["predictions"]
        labels = row["labels"]
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                predicted_delta = float(predictions[left] - predictions[right])
                label_delta = float(labels[left] - labels[right])
                margins.append(abs(predicted_delta))
                product = predicted_delta * label_delta
                correctness.append(1.0 if product > 0 else 0.5 if product == 0 else 0.0)

    selected = set(model["selected_terms"])
    seed_selected = set(seed_model["selected_terms"])
    frequencies = result["outer_expanded_support_frequency"]
    standardized = model["standardized_coefficients"]
    family_counts = Counter(
        V3_FEATURE_FAMILIES.get(name, _feature_family(name)) for name in selected
    )
    worst_pearson = sorted(rows, key=lambda row: float(row["pearson"]))[:20]
    worst_spearman = sorted(rows, key=lambda row: float(row["spearman"]))[:20]
    return {
        "protocol": PROTOCOL,
        "development_only": True,
        "untouched_test_complete": False,
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "support": {
            "available_features": int(result["available_feature_count"]),
            "selected_features": len(selected),
            "seed_features": len(seed_selected),
            "retained_from_seed": len(selected & seed_selected),
            "dropped_from_seed": len(seed_selected - selected),
            "new_relative_to_seed": len(selected - seed_selected),
            "family_counts": dict(sorted(family_counts.items())),
            "outer_frequency_counts": dict(
                sorted(Counter(frequencies.values()).items())
            ),
        },
        "largest_absolute_standardized_coefficients": [
            {"feature": name, "standardized_coefficient": float(value)}
            for name, value in sorted(
                standardized.items(), key=lambda item: abs(item[1]), reverse=True
            )[:30]
        ],
        "fixed_prediction_within_group_permutation": _fixed_prediction_null(rows, 100),
        "absolute_score_margin": {
            "pairs": len(margins),
            "quintiles": _quantile_bins(
                np.asarray(margins, dtype=float),
                np.asarray(correctness, dtype=float),
                5,
            ),
            "interpretation": (
                "Descriptive correctness versus absolute cross-fitted score "
                "difference; not a calibrated probability."
            ),
        },
        "worst_groups": {
            "by_pearson": [
                {
                    "key": row["key"],
                    "size": row["size"],
                    "pearson": row["pearson"],
                    "spearman": row["spearman"],
                }
                for row in worst_pearson
            ],
            "by_spearman": [
                {
                    "key": row["key"],
                    "size": row["size"],
                    "pearson": row["pearson"],
                    "spearman": row["spearman"],
                }
                for row in worst_spearman
            ],
        },
        "interpretation": (
            "All diagnostics are postfit development descriptions and did not "
            "alter the selected profile, support, or coefficients."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("Experiment/results/calibration_results.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("synde/models/synde_frozen_model.json"),
    )
    parser.add_argument(
        "--calibration-seed",
        type=Path,
        default=Path("Experiment/calibration_seed.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("Experiment/results/diagnostics.json")
    )
    args = parser.parse_args()
    for label, path in {
        "calibration result": args.result,
        "frozen model": args.model,
        "calibration seed": args.calibration_seed,
    }.items():
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    payload = public_record(run(args.result, args.model, args.calibration_seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
