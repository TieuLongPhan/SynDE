#!/usr/bin/env python3
"""Nested baseline and descriptor-block analysis for SynDE.

The analysis compares formula-invariant atom counts, a coarse chemical-bond
inventory, and cumulative blocks of the named SynDE descriptor library. Ridge
penalties are selected inside each outer training pool using complete
formula/formal-charge groups. Only the bond-inventory model is additionally
fitted on all ORD train groups and transferred to the established external
cohort; no external label enters fitting or penalty selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import math
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from rdkit import Chem
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.scripts._helpers import (  # noqa: E402
    _bootstrap_interval,
    _composite,
    _feature_family,
    _hash_bucket,
    _prediction_rows,
    _prepare,
    correlation,
    rank,
    public_record,
)
from synde.energy import V3_FEATURE_FAMILIES  # noqa: E402

_calibration = importlib.import_module("Experiment.scripts.05_calibrate")
INNER_FOLDS = _calibration.INNER_FOLDS
INNER_NAMESPACE = _calibration.INNER_NAMESPACE
OUTER_FOLDS = _calibration.OUTER_FOLDS
OUTER_NAMESPACE = _calibration.OUTER_NAMESPACE

_external = importlib.import_module("Experiment.scripts.06_evaluate")
_load_external = _external._load_external
_sample_group = _external._sample_group

RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
LOCAL_FAMILIES = frozenset({"atom_state", "bond_environment"})
RING_GRAPH_FAMILIES = frozenset(
    {
        "ring_topology",
        "classical_graph_index",
        "named_chemical_graph_descriptor",
        "other_named_term",
    }
)
ELECTRONIC_FAMILIES = frozenset(
    {
        "electrotopological_state",
        "gasteiger_charge",
        "huckel_fukui",
        "v3_charge_topology",
        "v3_huckel_density_spectral",
    }
)
NONLOCAL_FAMILIES = frozenset(
    {
        "first_order_v1",
        "v3_resonance_topology",
        "v3_cycle_ring_junction",
        "v3_graph_steric",
    }
)


def descriptor_family(name: str) -> str:
    """Return the canonical family for a named SynDE coordinate."""
    return V3_FEATURE_FAMILIES.get(name, _feature_family(name))


def bond_inventory(molecule: Chem.Mol) -> dict[str, float]:
    """Count element-pair/bond-order classes, including implicit X--H bonds."""
    counts: Counter[str] = Counter()
    for bond in molecule.GetBonds():
        left, right = sorted(
            (bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol())
        )
        order = str(bond.GetBondType()).lower()
        counts[f"bond_count[{left}-{right};order={order}]"] += 1
    for atom in molecule.GetAtoms():
        hydrogens = int(atom.GetTotalNumHs(includeNeighbors=True))
        if hydrogens:
            left, right = sorted(("H", atom.GetSymbol()))
            counts[f"bond_count[{left}-{right};order=single]"] += hydrogens
    return {name: float(value) for name, value in sorted(counts.items())}


def groups_with_bond_inventory(
    groups: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Replace the full feature map by coarse bond-inventory coordinates."""
    output: list[dict[str, object]] = []
    for group in groups:
        molecules = []
        for row in group["molecules"]:
            molecule = row.get("molecule")
            if molecule is None:
                molecule = Chem.MolFromSmiles(str(row["smiles"]))
            if molecule is None:
                raise ValueError(f"Could not parse {row['smiles']!r}.")
            molecules.append(
                {
                    "smiles": str(row["smiles"]),
                    "label": float(row["label"]),
                    "features": bond_inventory(molecule),
                }
            )
        output.append({"key": str(group["key"]), "molecules": molecules})
    return output


def prediction_rows_with_ties(
    groups: list[dict[str, object]], predictions: np.ndarray
) -> list[dict[str, object]]:
    """Return group metrics without discarding constant predicted rankings."""
    rows: list[dict[str, object]] = []
    position = 0
    for group in groups:
        molecules = group["molecules"]
        count = len(molecules)
        predicted = np.asarray(predictions[position : position + count], dtype=float)
        labels = np.asarray([row["label"] for row in molecules], dtype=float)
        position += count
        pearson = correlation(predicted.tolist(), labels.tolist())
        spearman = correlation(rank(predicted).tolist(), rank(labels).tolist())
        concordance: list[float] = []
        for left in range(count):
            for right in range(left + 1, count):
                product = (predicted[left] - predicted[right]) * (
                    labels[left] - labels[right]
                )
                concordance.append(1.0 if product > 0 else 0.5 if product == 0 else 0.0)
        predicted_minima = np.flatnonzero(predicted == np.min(predicted))
        reference_minima = set(np.flatnonzero(labels == np.min(labels)).tolist())
        top1 = sum(index in reference_minima for index in predicted_minima) / len(
            predicted_minima
        )
        rows.append(
            {
                "key": str(group["key"]),
                "size": count,
                "pearson": float(pearson) if math.isfinite(pearson) else None,
                "spearman": float(spearman) if math.isfinite(spearman) else None,
                "pairwise_concordance": float(np.mean(concordance)),
                "top1_accuracy": float(top1),
                "predictions": predicted.tolist(),
                "labels": labels.tolist(),
            }
        )
    return rows


def summary_with_ties(
    rows: list[dict[str, object]], namespace: str
) -> dict[str, object]:
    """Summarize correlations conditionally and ranking metrics unconditionally."""
    rankable = [
        row
        for row in rows
        if row["pearson"] is not None and row["spearman"] is not None
    ]
    pearson = [float(row["pearson"]) for row in rankable]
    spearman = [float(row["spearman"]) for row in rankable]
    rankable_sizes = np.asarray([row["size"] for row in rankable], dtype=float)
    concordance = [float(row["pairwise_concordance"]) for row in rows]
    top1 = [float(row["top1_accuracy"]) for row in rows]
    return {
        "groups": len(rows),
        "molecules": int(sum(int(row["size"]) for row in rows)),
        "rankable_groups": len(rankable),
        "mean_pearson": float(np.mean(pearson)),
        "mean_pearson_bootstrap_95ci": _bootstrap_interval(
            pearson, f"{namespace}:pearson"
        ),
        "mean_spearman": float(np.mean(spearman)),
        "mean_spearman_bootstrap_95ci": _bootstrap_interval(
            spearman, f"{namespace}:spearman"
        ),
        "weighted_pearson": float(np.average(pearson, weights=rankable_sizes)),
        "weighted_spearman": float(np.average(spearman, weights=rankable_sizes)),
        "positive_pearson_fraction": float(np.mean(np.asarray(pearson) > 0)),
        "mean_pairwise_concordance": float(np.mean(concordance)),
        "top1_accuracy": float(np.mean(top1)),
    }


def fit_profile_predictions(
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    support: set[str],
    alpha: float,
) -> np.ndarray:
    """Fit a ridge profile and return predictions for all held-out molecules."""
    _, _, train_x, test_x, target = _prepare(train, test, selected_names=support)
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-7, max_iter=10_000).fit(
        train_x, target
    )
    return np.asarray(model.predict(test_x), dtype=float)


def _outer_split(
    groups: list[dict[str, object]], outer: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [
        group
        for group in groups
        if _hash_bucket(OUTER_NAMESPACE, str(group["key"]), OUTER_FOLDS) != outer
    ]
    test = [
        group
        for group in groups
        if _hash_bucket(OUTER_NAMESPACE, str(group["key"]), OUTER_FOLDS) == outer
    ]
    return train, test


def _inner_splits(
    groups: list[dict[str, object]], outer: int
) -> list[tuple[list[dict[str, object]], list[dict[str, object]]]]:
    namespace = f"{INNER_NAMESPACE}:outer={outer}"
    return [
        (
            [
                group
                for group in groups
                if _hash_bucket(namespace, str(group["key"]), INNER_FOLDS) != fold
            ],
            [
                group
                for group in groups
                if _hash_bucket(namespace, str(group["key"]), INNER_FOLDS) == fold
            ],
        )
        for fold in range(INNER_FOLDS)
    ]


def _tune_ridge(
    splits: list[tuple[list[dict[str, object]], list[dict[str, object]]]],
    allowed: set[str] | None,
) -> tuple[float, list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for alpha in RIDGE_ALPHA_GRID:
        rows: list[dict[str, object]] = []
        for train, validation in splits:
            vectorizer, _, train_x, validation_x, target = _prepare(
                train, validation, selected_names=allowed
            )
            model = Ridge(alpha=alpha, solver="lsqr", tol=1e-7, max_iter=10_000).fit(
                train_x, target
            )
            rows.extend(_prediction_rows(validation, model.predict(validation_x)))
        candidates.append(
            {
                "alpha": alpha,
                "composite": _composite(rows),
                "mean_pearson": float(np.mean([row["pearson"] for row in rows])),
                "mean_spearman": float(np.mean([row["spearman"] for row in rows])),
            }
        )
    best = max(
        candidates,
        key=lambda row: (
            float(row["composite"]),
            -abs(np.log10(float(row["alpha"])) - 1.0),
        ),
    )
    return float(best["alpha"]), candidates


def nested_ridge_profile(
    groups: list[dict[str, object]],
    allowed: set[str] | None,
    name: str,
) -> dict[str, object]:
    """Evaluate one ridge profile with nested formula-group cross-validation."""
    profile_names = allowed or {
        feature
        for group in groups
        for row in group["molecules"]
        for feature in row["features"]
    }
    rows: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for outer in range(OUTER_FOLDS):
        train, test = _outer_split(groups, outer)
        alpha, candidates = _tune_ridge(_inner_splits(train, outer), profile_names)
        predictions = fit_profile_predictions(train, test, profile_names, alpha)
        rows.extend(prediction_rows_with_ties(test, predictions))
        folds.append(
            {
                "outer_fold": outer + 1,
                "train_groups": len(train),
                "test_groups": len(test),
                "ridge_alpha": alpha,
                "inner_candidates": candidates,
            }
        )
        print(
            f"{name}: completed outer fold {outer + 1}/{OUTER_FOLDS}",
            file=sys.stderr,
            flush=True,
        )
    return {
        "coordinate_count": len(profile_names),
        "metrics": summary_with_ties(rows, f"synde-method-ablation:{name}"),
        "outer_folds": folds,
    }


def _cross_validated_alpha(
    groups: list[dict[str, object]],
) -> tuple[float, list[dict[str, object]]]:
    splits = [_outer_split(groups, outer) for outer in range(OUTER_FOLDS)]
    return _tune_ridge(splits, None)


def _external_primary_groups(
    external_path: Path,
    training_groups: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    training_keys = {str(group["key"]) for group in training_groups}
    training_connectivities = {
        str(row["connectivity"])
        for group in training_groups
        for row in group["molecules"]
    }
    groups, audit = _load_external(external_path)
    eligible = {
        key: rows
        for key, rows in groups.items()
        if len(rows) >= 3 and len({float(row["label"]) for row in rows}) >= 2
    }
    graph_disjoint = {
        key: [
            row
            for row in rows
            if str(row["connectivity"]) not in training_connectivities
        ]
        for key, rows in eligible.items()
    }
    graph_disjoint = {
        key: rows
        for key, rows in graph_disjoint.items()
        if len(rows) >= 3 and len({float(row["label"]) for row in rows}) >= 2
    }
    primary = {
        key: _sample_group(key, rows)
        for key, rows in sorted(graph_disjoint.items())
        if key not in training_keys
    }
    return (
        [{"key": key, "molecules": rows} for key, rows in primary.items()],
        audit,
    )


def external_bond_inventory(
    training: list[dict[str, object]],
    external: list[dict[str, object]],
) -> dict[str, object]:
    """Fit the bond-inventory ridge on ORD train and evaluate external groups."""
    alpha, tuning = _cross_validated_alpha(training)
    vectorizer, scaler, train_x, external_x, target = _prepare(training, external)
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-7, max_iter=10_000).fit(
        train_x, target
    )
    rows = prediction_rows_with_ties(external, model.predict(external_x))
    raw_weights = model.coef_ / scaler.scale_
    return {
        "selected_ridge_alpha": alpha,
        "coordinate_count": len(vectorizer.feature_names_),
        "training_only_tuning": tuning,
        "metrics": summary_with_ties(rows, "synde-bond-inventory:external"),
        "weights": {
            name: float(weight)
            for name, weight in zip(vectorizer.feature_names_, raw_weights)
        },
    }


def analyze(
    training_cache_path: Path,
    external_path: Path,
    calibration_path: Path,
    external_result_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    groups = joblib.load(training_cache_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    available = {
        name
        for group in groups
        for row in group["molecules"]
        for name in row["features"]
    }
    family_names = {
        family: {name for name in available if descriptor_family(name) == family}
        for family in sorted({descriptor_family(name) for name in available})
    }
    local = set().union(*(family_names.get(family, set()) for family in LOCAL_FAMILIES))
    local_ring_graph = local | set().union(
        *(family_names.get(family, set()) for family in RING_GRAPH_FAMILIES)
    )
    local_ring_electronic = local_ring_graph | set().union(
        *(family_names.get(family, set()) for family in ELECTRONIC_FAMILIES)
    )
    complete = local_ring_electronic | set().union(
        *(family_names.get(family, set()) for family in NONLOCAL_FAMILIES)
    )

    bond_training = groups_with_bond_inventory(groups)
    profiles = {
        "bond_inventory_ridge": nested_ridge_profile(
            bond_training, None, "bond_inventory"
        ),
        "local_valence_ridge": nested_ridge_profile(groups, local, "local_valence"),
        "local_plus_ring_graph_ridge": nested_ridge_profile(
            groups, local_ring_graph, "local_plus_ring_graph"
        ),
        "local_ring_graph_plus_electronic_ridge": nested_ridge_profile(
            groups, local_ring_electronic, "local_ring_graph_plus_electronic"
        ),
        "complete_library_ridge": nested_ridge_profile(
            groups, complete, "complete_library"
        ),
    }

    external_groups, external_audit = _external_primary_groups(external_path, groups)
    bond_external = groups_with_bond_inventory(external_groups)
    external_bond = external_bond_inventory(bond_training, bond_external)

    return {
        "analysis": "synde-formal-method-baselines-and-ablation-v1",
        "design": (
            "Formula-centered ridge comparators with complete formula groups "
            "held out; ridge penalties selected inside each outer training pool."
        ),
        "training": {
            "groups": len(groups),
            "molecules": sum(len(group["molecules"]) for group in groups),
        },
        "atom_count_control": {
            "rankable_groups": 0,
            "pearson": None,
            "spearman": None,
            "tie_aware_pairwise_concordance": 0.5,
            "reason": "All atom-count coordinates are constant within formula.",
        },
        "descriptor_blocks": {
            "local_valence": sorted(LOCAL_FAMILIES),
            "ring_and_global_topology": sorted(RING_GRAPH_FAMILIES),
            "charge_and_pi_electronic": sorted(ELECTRONIC_FAMILIES),
            "first_order_and_nonlocal_graph": sorted(NONLOCAL_FAMILIES),
        },
        "profiles": profiles,
        "synde_nested_reference": calibration["synde_nested"],
        "external": {
            "groups": len(bond_external),
            "molecules": sum(len(group["molecules"]) for group in bond_external),
            "source_audit": external_audit,
            "bond_inventory_ridge": external_bond,
            "synde_reference": json.loads(
                external_result_path.read_text(encoding="utf-8")
            )["external_performance"],
        },
        "interpretation": (
            "Bond inventories capture bond-order composition but not local "
            "substitution, conjugation, ring junctions, or nonlocal electronic "
            "topology. Cumulative profiles quantify the additional ranking "
            "information supplied by these descriptor blocks."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path("/tmp/synde-energy-external-validation/training.joblib"),
    )
    parser.add_argument(
        "--external-csv",
        type=Path,
        default=PROJECT_ROOT / "data/ord_test_xtb.csv",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "Experiment/results/calibration_results.json",
    )
    parser.add_argument(
        "--external-result",
        type=Path,
        default=PROJECT_ROOT / "Experiment/results/external_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Experiment/results/synde_baselines_ablation.json",
    )
    args = parser.parse_args()
    for label, path in {
        "training cache": args.training_cache,
        "test labels": args.external_csv,
        "calibration result": args.calibration,
        "held-out result": args.external_result,
    }.items():
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    payload = public_record(
        analyze(
            args.training_cache,
            args.external_csv,
            args.calibration,
            args.external_result,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
