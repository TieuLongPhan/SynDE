"""Analyze the formula-invariant atom-count baseline for SynDE.

Elemental atom counts and formal charge are constant inside a molecular-formula
group. After the same within-group centering used for SynDE calibration, their
design matrix is therefore identically zero. Ridge fitting has the unique
solution w=0 and cannot rank constitutional isomers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*)")


def parse_formula_group_key(key: str) -> tuple[dict[str, int], int]:
    """Parse ``formula|charge=n`` into elemental counts and formal charge."""
    formula, separator, charge_text = key.partition("|charge=")
    if not separator:
        raise ValueError(f"Invalid formula/formal-charge key: {key!r}")
    counts: dict[str, int] = {}
    position = 0
    for match in FORMULA_TOKEN.finditer(formula):
        if match.start() != position:
            raise ValueError(f"Invalid molecular formula: {formula!r}")
        element, count_text = match.groups()
        counts[element] = counts.get(element, 0) + int(count_text or "1")
        position = match.end()
    if not counts or position != len(formula):
        raise ValueError(f"Invalid molecular formula: {formula!r}")
    return counts, int(charge_text)


def atom_count_features(key: str, elements: Iterable[str]) -> np.ndarray:
    """Return elemental counts, total atoms, heavy atoms, and formal charge."""
    counts, charge = parse_formula_group_key(key)
    ordered = tuple(elements)
    return np.asarray(
        [float(counts.get(element, 0)) for element in ordered]
        + [
            float(sum(counts.values())),
            float(sum(count for element, count in counts.items() if element != "H")),
            float(charge),
        ],
        dtype=float,
    )


def centered_group_design(
    group_rows: list[dict[str, object]], elements: Iterable[str]
) -> np.ndarray:
    """Build and center the atom-count design for formula-group rows."""
    rows: list[np.ndarray] = []
    for group in group_rows:
        vector = atom_count_features(str(group["key"]), elements)
        size = int(group["size"])
        block = np.repeat(vector[None, :], size, axis=0)
        rows.append(block - block.mean(axis=0, keepdims=True))
    return np.vstack(rows) if rows else np.empty((0, len(tuple(elements)) + 3))


def expected_top1_under_random_tie_break(group_sizes: Iterable[int]) -> float:
    """Return mean chance top-1 accuracy when every candidate receives a tie."""
    sizes = np.asarray(tuple(group_sizes), dtype=float)
    if not len(sizes) or np.any(sizes <= 0):
        raise ValueError("Group sizes must be nonempty and positive.")
    return float(np.mean(1.0 / sizes))


def analyze(calibration_path: Path, external_path: Path) -> dict[str, object]:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    external = json.loads(external_path.read_text(encoding="utf-8"))
    calibration_rows = calibration["group_predictions"]["synde"]
    external_rows = external["group_metrics"]

    all_keys = [str(row["key"]) for row in calibration_rows + external_rows]
    elements = sorted(
        {element for key in all_keys for element in parse_formula_group_key(key)[0]}
    )
    training_design = centered_group_design(calibration_rows, elements)
    external_design = centered_group_design(external_rows, elements)
    training_sizes = [int(row["size"]) for row in calibration_rows]
    external_sizes = [int(row["size"]) for row in external_rows]

    feature_names = [f"count_{element}" for element in elements] + [
        "total_atom_count",
        "heavy_atom_count",
        "formal_charge",
    ]
    external_size_counts = Counter(external_sizes)
    return {
        "analysis": "synde-formula-relative-atom-count-baseline-v1",
        "definition": (
            "Ridge regression using elemental atom counts, total atom count, "
            "heavy-atom count, and formal charge after within-formula centering"
        ),
        "feature_names": feature_names,
        "ridge_alpha": 1.0,
        "mathematical_result": (
            "Every feature is constant within a formula/formal-charge group; "
            "the centered design is exactly zero and ridge weights are zero."
        ),
        "fitted_weights": {name: 0.0 for name in feature_names},
        "training": {
            "groups": len(calibration_rows),
            "molecules": sum(training_sizes),
            "maximum_absolute_centered_feature": float(
                np.max(np.abs(training_design), initial=0.0)
            ),
            "rankable_groups": 0,
        },
        "external": {
            "groups": len(external_rows),
            "molecules": sum(external_sizes),
            "maximum_absolute_centered_feature": float(
                np.max(np.abs(external_design), initial=0.0)
            ),
            "rankable_groups": 0,
            "pearson": None,
            "spearman": None,
            "pearson_and_spearman_reason": "zero prediction variance in every group",
            "tie_aware_pairwise_concordance": 0.5,
            "expected_top1_accuracy_under_random_tie_break": (
                expected_top1_under_random_tie_break(external_sizes)
            ),
            "group_size_counts": {
                str(size): external_size_counts[size]
                for size in sorted(external_size_counts)
            },
        },
        "interpretation": (
            "Composition-only learning can model cross-formula energy scale but "
            "has no information for ranking constitutional isomers."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "Experiment/results/calibration_results.json",
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=PROJECT_ROOT / "Experiment/results/external_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Experiment/results/synde_atom_count_baseline.json",
    )
    args = parser.parse_args()
    for label, path in {
        "calibration result": args.calibration,
        "held-out result": args.external,
    }.items():
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    payload = analyze(args.calibration, args.external)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
