#!/usr/bin/env python3
"""Calibrate and externally validate the SynDE total-energy model.

The newly fitted formula-relative connectivity equation is frozen before this
stage and an element-count calibration is fitted to its raw-energy residual.
The resulting single artifact predicts protocol-defined GFN2-xTB optimized total energy
globally and ranks constitutional isomers locally; composition cancels exactly
for molecules with the same formula.

The training and external-validation cohorts are formula- and
connectivity-disjoint. All elemental calibration is fitted on training data;
the external cohort is used only once for global and within-formula scoring.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import joblib
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.scripts._helpers import _compact  # noqa: E402
from synde.energy import (  # noqa: E402
    FirstOrderTwoDEnergyScorer,
    SynDEEnergyModelCard,
    SynDEEnergyPredictor,
    extract_named_empirical_two_d_features,
    extract_quantum_graph_v3_features,
    molecular_composition,
)
from synde.energy.interpretable_two_d_v3 import (  # noqa: E402
    V3_FEATURE_FAMILIES,
)
from synde.graph import GraphBuilder  # noqa: E402
from synde.chem import (  # noqa: E402
    ISOTOPE_EXCLUSION_REASON,
    SUPPORTED_ELEMENTS,
    SUPPORTED_ELEMENT_SET,
    has_isotopically_labelled_atom,
    normalize_ordinary_explicit_hydrogens,
)

PROTOCOL = "synde-total-energy-amended-domain-external-validation-v2"
REFERENCE_PROTOCOL = (
    "single retained RDKit ETKDGv3 conformer; MMFF/UFF preoptimization; "
    "gas-phase GFN2-xTB 6.7.1 --opt extreme --acc 0.01; neutral closed-shell; "
    "final optimized total energy converted from Hartree to eV"
)
TARGET = "single-conformer gas-phase GFN2-xTB 6.7.1 optimized total energy"
SOURCE_SUPPORTED_ELEMENTS = set(SUPPORTED_ELEMENT_SET)
HEAVY_ELEMENTS = tuple(element for element in SUPPORTED_ELEMENTS if element != "H")
MAX_MOLECULES_PER_GROUP = 10
MIN_GROUP_SIZE = 3
SAMPLE_NAMESPACE = "synde-external-validation-v1"
DIRECT_JOINT_RIDGE_ALPHA = 100.0
FIRST_ORDER_SCORER = FirstOrderTwoDEnergyScorer()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smiles_column(fieldnames: Iterable[str] | None) -> str:
    names = set(fieldnames or ())
    for candidate in ("SMILES", "Canonical_SMILES", "smiles", "mol"):
        if candidate in names:
            return candidate
    raise ValueError(f"No recognized SMILES column in {sorted(names)}")


def _identity(molecule: Chem.Mol) -> tuple[str, str]:
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    charge = Chem.GetFormalCharge(molecule)
    connectivity = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    return f"{formula}|charge={charge}", connectivity


def _sample_group(key: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SAMPLE_NAMESPACE}:{key}:{row['connectivity']}".encode()
        ).hexdigest(),
    )[:MAX_MOLECULES_PER_GROUP]


def _extract_features(smiles: str) -> tuple[dict[str, float], dict[str, int], int]:
    graph = GraphBuilder.from_smiles(smiles)
    first_order = FIRST_ORDER_SCORER.score(graph)
    features = _compact(extract_named_empirical_two_d_features(graph, first_order))
    features.update(extract_quantum_graph_v3_features(graph))
    composition = molecular_composition(graph)
    heavy_atoms = sum(count for element, count in composition.items() if element != "H")
    return features, composition, heavy_atoms


def _extract_external_group(
    item: tuple[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, list[dict[str, str]], int]:
    key, rows = item
    molecules = []
    failures = []
    for row in rows:
        try:
            features, composition, heavy_atoms = _extract_features(row["smiles"])
        except Exception as exc:  # noqa: BLE001 - audit every scoring failure.
            failures.append(
                {
                    "key": key,
                    "smiles": str(row["smiles"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        molecules.append(
            row
            | {
                "features": features,
                "composition": composition,
                "heavy_atoms": heavy_atoms,
                "formal_charge": 0,
            }
        )
    group = (
        {"key": key, "molecules": molecules}
        if len(molecules) >= MIN_GROUP_SIZE
        and len({row["label"] for row in molecules}) >= 2
        else None
    )
    return group, failures, len(molecules)


def build_external_cache(
    source: Path,
    training_groups: list[dict[str, Any]],
    output: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a training formula/connectivity firewall and extract fixed features."""

    started = time.perf_counter()
    training_keys = {str(group["key"]) for group in training_groups}
    training_connectivities = {
        str(row["connectivity"])
        for group in training_groups
        for row in group["molecules"]
    }
    audit: Counter[str] = Counter()
    candidates: dict[str, dict[str, dict[str, Any] | None]] = defaultdict(dict)
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        smiles_column = _smiles_column(reader.fieldnames)
        for row in reader:
            audit["source_rows"] += 1
            if row.get("status") != "success":
                audit["excluded_non_success"] += 1
                continue
            try:
                label = float(row.get("energy") or row.get("energy_eV"))
            except (TypeError, ValueError):
                audit["excluded_missing_label"] += 1
                continue
            if not math.isfinite(label):
                audit["excluded_nonfinite_label"] += 1
                continue
            smiles = str(row.get(smiles_column, "")).strip()
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                audit["excluded_parse_failure"] += 1
                continue
            if has_isotopically_labelled_atom(molecule):
                audit[f"excluded_{ISOTOPE_EXCLUSION_REASON}"] += 1
                continue
            molecule = normalize_ordinary_explicit_hydrogens(molecule)
            if len(Chem.GetMolFrags(molecule)) != 1:
                audit["excluded_disconnected"] += 1
                continue
            if Chem.GetFormalCharge(molecule) != 0:
                audit["excluded_nonzero_charge"] += 1
                continue
            if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
                audit["excluded_non_closed_shell"] += 1
                continue
            elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
            if not elements <= SOURCE_SUPPORTED_ELEMENTS:
                audit["excluded_unsupported_element"] += 1
                continue
            key, connectivity = _identity(molecule)
            if key in training_keys:
                audit["excluded_training_formula"] += 1
                continue
            if connectivity in training_connectivities:
                audit["excluded_training_connectivity"] += 1
                continue
            if connectivity in candidates[key]:
                if candidates[key][connectivity] is not None:
                    candidates[key][connectivity] = None
                    audit["excluded_duplicate_connectivity_rows"] += 2
                else:
                    audit["excluded_duplicate_connectivity_rows"] += 1
                continue
            candidates[key][connectivity] = {
                "smiles": Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=True
                ),
                "connectivity": connectivity,
                "label": label,
            }
            audit["clean_rows_before_achiral_deduplication"] += 1

    selected: list[tuple[str, list[dict[str, Any]]]] = []
    for key, by_connectivity in candidates.items():
        rows = [row for row in by_connectivity.values() if row is not None]
        if len(rows) < MIN_GROUP_SIZE or len({row["label"] for row in rows}) < 2:
            audit["excluded_nonrankable_formula_groups"] += 1
            audit["excluded_nonrankable_molecules"] += len(rows)
            continue
        selected.append((key, _sample_group(key, rows)))

    groups: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    processed = 0
    with multiprocessing.Pool(processes=workers) as pool:
        extracted = pool.imap(_extract_external_group, sorted(selected), chunksize=8)
        for position, (group, group_failures, extracted_count) in enumerate(
            extracted, start=1
        ):
            if group is not None:
                groups.append(group)
            failures.extend(group_failures)
            processed += extracted_count
            if position % 250 == 0:
                print(
                    f"cross-formula external features {position}/{len(selected)} groups",
                    file=sys.stderr,
                    flush=True,
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(groups, output, compress=3)
    summary = {
        "protocol": f"{PROTOCOL}-external-cache",
        "source": str(source),
        "source_sha256": _sha256(source),
        "training_formula_count": len(training_keys),
        "training_connectivity_count": len(training_connectivities),
        "formula_disjoint": True,
        "connectivity_disjoint": True,
        "groups": len(groups),
        "molecules": sum(len(group["molecules"]) for group in groups),
        "audit": dict(sorted(audit.items())),
        "feature_extraction_failures": failures,
        "cache": str(output),
        "cache_sha256": _sha256(output),
        "elapsed_seconds": time.perf_counter() - started,
        "mean_seconds_per_extracted_molecule": (
            (time.perf_counter() - started) / processed if processed else None
        ),
    }
    return groups, summary


def _ensure_training_metadata(groups: list[dict[str, Any]]) -> None:
    for group in groups:
        for row in group["molecules"]:
            if "composition" in row:
                continue
            graph = GraphBuilder.from_smiles(str(row["smiles"]))
            composition = molecular_composition(graph)
            row["composition"] = composition
            row["heavy_atoms"] = sum(
                count for element, count in composition.items() if element != "H"
            )
            row["formal_charge"] = 0


def _flatten(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row | {"group_key": str(group["key"])}
        for group in groups
        for row in group["molecules"]
    ]


def _composition_matrix(
    rows: list[dict[str, Any]],
    *,
    elements: tuple[str, ...] = SUPPORTED_ELEMENTS,
    enhanced: bool = False,
) -> tuple[np.ndarray, list[str]]:
    names = [f"atom_count[{element}]" for element in elements]
    matrix = np.asarray(
        [
            [float(row["composition"].get(element, 0)) for element in elements]
            for row in rows
        ],
        dtype=float,
    )
    if not enhanced:
        return matrix, names
    atomic_numbers = {
        element: Chem.GetPeriodicTable().GetAtomicNumber(element)
        for element in elements
    }
    extra = []
    for row in rows:
        composition = row["composition"]
        total_atoms = sum(composition.values())
        heavy_atoms = int(row["heavy_atoms"])
        electron_count = sum(
            atomic_numbers[element] * composition.get(element, 0)
            for element in elements
        )
        extra.append(
            [total_atoms, heavy_atoms, electron_count, total_atoms**2, heavy_atoms**2]
        )
    names.extend(
        [
            "total_atoms",
            "heavy_atoms",
            "electron_count",
            "total_atoms_squared",
            "heavy_atoms_squared",
        ]
    )
    return np.column_stack((matrix, np.asarray(extra, dtype=float))), names


def _feature_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in names] for row in rows],
        dtype=np.float32,
    )


def _fit_linear(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    model = LinearRegression().fit(train_x, train_y)
    return model.predict(test_x), float(model.intercept_), np.asarray(model.coef_)


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    group_keys: list[str],
    *,
    namespace: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    residual = predictions - labels
    ss_res = float(np.sum(residual**2))
    ss_total = float(np.sum((labels - labels.mean()) ** 2))
    pearson = (
        float(np.corrcoef(labels, predictions)[0, 1])
        if np.std(labels) > 0 and np.std(predictions) > 0
        else None
    )
    ranked_labels = rankdata(labels)
    ranked_predictions = rankdata(predictions)
    spearman = (
        float(np.corrcoef(ranked_labels, ranked_predictions)[0, 1])
        if np.std(ranked_labels) > 0 and np.std(ranked_predictions) > 0
        else None
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        grouped[key].append(index)
    group_absolute_error_sums = np.asarray(
        [float(np.sum(np.abs(residual[indices]))) for indices in grouped.values()]
    )
    group_squared_error_sums = np.asarray(
        [float(np.sum(residual[indices] ** 2)) for indices in grouped.values()]
    )
    group_counts = np.asarray(
        [len(indices) for indices in grouped.values()], dtype=float
    )
    seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    boot_mae = np.empty(bootstrap_replicates, dtype=float)
    boot_rmse = np.empty(bootstrap_replicates, dtype=float)
    for start in range(0, bootstrap_replicates, 250):
        count = min(250, bootstrap_replicates - start)
        indices = rng.integers(0, len(group_counts), size=(count, len(group_counts)))
        sampled_counts = np.sum(group_counts[indices], axis=1)
        boot_mae[start : start + count] = (
            np.sum(group_absolute_error_sums[indices], axis=1) / sampled_counts
        )
        boot_rmse[start : start + count] = np.sqrt(
            np.sum(group_squared_error_sums[indices], axis=1) / sampled_counts
        )
    return {
        "molecules": len(labels),
        "formula_groups": len(grouped),
        "mae_eV": float(np.mean(np.abs(residual))),
        "mae_group_bootstrap_95ci_eV": [
            float(np.quantile(boot_mae, 0.025)),
            float(np.quantile(boot_mae, 0.975)),
        ],
        "rmse_eV": float(np.sqrt(np.mean(residual**2))),
        "rmse_group_bootstrap_95ci_eV": [
            float(np.quantile(boot_rmse, 0.025)),
            float(np.quantile(boot_rmse, 0.975)),
        ],
        "median_absolute_error_eV": float(np.median(np.abs(residual))),
        "r2": float(1.0 - ss_res / ss_total) if ss_total > 0 else None,
        "pearson": pearson,
        "spearman": spearman,
        "mean_signed_error_eV": float(np.mean(residual)),
    }


def _ranking_metrics(
    rows: list[dict[str, Any]], predictions: np.ndarray, bootstrap_replicates: int
) -> dict[str, Any]:
    by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        by_group[str(row["group_key"])].append((float(prediction), float(row["label"])))
    correlation_records = []
    ranking_records = []
    for values in by_group.values():
        scores = np.asarray([value[0] for value in values])
        labels = np.asarray([value[1] for value in values])
        if len(values) < MIN_GROUP_SIZE or len(np.unique(labels)) < 2:
            continue
        pearson = float(np.corrcoef(scores, labels)[0, 1])
        spearman = float(np.corrcoef(rankdata(scores), rankdata(labels))[0, 1])
        if math.isfinite(pearson) and math.isfinite(spearman):
            correlation_records.append([pearson, spearman, float(len(values))])
        concordance = []
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                product = (scores[left] - scores[right]) * (
                    labels[left] - labels[right]
                )
                concordance.append(1.0 if product > 0 else 0.5 if product == 0 else 0.0)
        predicted_minima = np.isclose(scores, np.min(scores), rtol=0.0, atol=1e-12)
        reference_minima = np.isclose(labels, np.min(labels), rtol=0.0, atol=1e-12)
        ranking_records.append(
            [
                float(np.mean(concordance)),
                float(
                    np.sum(predicted_minima & reference_minima)
                    / np.sum(predicted_minima)
                ),
                float(len(values)),
            ]
        )
    correlation_array = np.asarray(correlation_records)
    ranking_array = np.asarray(ranking_records)
    output: dict[str, Any] = {
        "groups": len(ranking_records),
        "molecules": int(np.sum(ranking_array[:, 2])),
        "correlation_groups": len(correlation_records),
        "correlation_molecules": int(np.sum(correlation_array[:, 2])),
        "concordance_groups": len(ranking_records),
        "top1_groups": len(ranking_records),
    }
    rng = np.random.default_rng(
        int(hashlib.sha256(f"{PROTOCOL}:ranking".encode()).hexdigest()[:16], 16)
    )
    metric_arrays = (
        ("mean_pearson", correlation_array[:, 0]),
        ("mean_spearman", correlation_array[:, 1]),
        ("mean_pairwise_concordance", ranking_array[:, 0]),
        ("top1_accuracy", ranking_array[:, 1]),
    )
    for name, values in metric_arrays:
        means = np.empty(bootstrap_replicates)
        for start in range(0, bootstrap_replicates, 250):
            count = min(250, bootstrap_replicates - start)
            indices = rng.integers(0, len(values), size=(count, len(values)))
            means[start : start + count] = np.mean(values[indices], axis=1)
        output[name] = float(np.mean(values))
        output[f"{name}_group_bootstrap_95ci"] = [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ]
    return output


def _stratified_metrics(
    test_rows: list[dict[str, Any]],
    labels: np.ndarray,
    predictions: np.ndarray,
    distances: np.ndarray,
    q99: float,
) -> dict[str, Any]:
    heavy = np.asarray([row["heavy_atoms"] for row in test_rows], dtype=float)
    quartiles = np.quantile(heavy, (0.25, 0.5, 0.75))

    def subset(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {"molecules": 0}
        keys = [test_rows[index]["group_key"] for index in np.flatnonzero(mask)]
        return _metrics(
            labels[mask],
            predictions[mask],
            keys,
            namespace=f"{PROTOCOL}:stratum:{hashlib.sha256(mask.tobytes()).hexdigest()}",
            bootstrap_replicates=500,
        )

    size_masks = {
        "q1_smallest": heavy <= quartiles[0],
        "q2": (heavy > quartiles[0]) & (heavy <= quartiles[1]),
        "q3": (heavy > quartiles[1]) & (heavy <= quartiles[2]),
        "q4_largest": heavy > quartiles[2],
    }
    composition = {}
    for element in HEAVY_ELEMENTS:
        mask = np.asarray(
            [row["composition"].get(element, 0) > 0 for row in test_rows], dtype=bool
        )
        if np.any(mask):
            composition[f"contains_{element}"] = subset(mask)
    return {
        "heavy_atom_quartile_boundaries": quartiles.tolist(),
        "molecular_size": {name: subset(mask) for name, mask in size_masks.items()},
        "element_presence": composition,
        "formal_charge": {"charge_0": subset(np.ones(len(test_rows), dtype=bool))},
        "applicability": {
            "inside_training_q99": subset(distances <= q99),
            "outside_training_q99": subset(distances > q99),
        },
    }


def run(
    *,
    training_cache: Path,
    training_cache_summary: Path,
    formula_relative_model: Path,
    external_csv: Path,
    external_cache: Path,
    external_cache_summary_path: Path | None,
    model_output: Path,
    bootstrap_replicates: int,
    rebuild_external_cache: bool,
    workers: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    training_groups = joblib.load(training_cache)
    _ensure_training_metadata(training_groups)
    training_summary = json.loads(training_cache_summary.read_text(encoding="utf-8"))
    formula_model = json.loads(formula_relative_model.read_text(encoding="utf-8"))
    connectivity_weights = {
        str(name): float(value) for name, value in formula_model["weights"].items()
    }
    if not connectivity_weights:
        raise ValueError("The fitted SynDE connectivity equation is empty.")
    feature_names = sorted(connectivity_weights)

    if external_cache.exists() and not rebuild_external_cache:
        external_groups = joblib.load(external_cache)
        if external_cache_summary_path is not None:
            external_cache_summary = json.loads(
                external_cache_summary_path.read_text(encoding="utf-8")
            )
            external_cache_summary["reused_compact_cache"] = str(external_cache)
            external_cache_summary["reused_compact_cache_sha256"] = _sha256(
                external_cache
            )
        else:
            external_cache_summary = {
                "protocol": f"{PROTOCOL}-external-cache-reused",
                "source": str(external_csv),
                "source_sha256": _sha256(external_csv),
                "formula_disjoint": True,
                "connectivity_disjoint": True,
                "groups": len(external_groups),
                "molecules": sum(len(group["molecules"]) for group in external_groups),
                "cache": str(external_cache),
                "cache_sha256": _sha256(external_cache),
            }
    else:
        external_groups, external_cache_summary = build_external_cache(
            external_csv, training_groups, external_cache, workers
        )
    _ensure_training_metadata(external_groups)

    trained_elements = tuple(
        element
        for element in SUPPORTED_ELEMENTS
        if any(
            row["composition"].get(element, 0) > 0
            for group in training_groups
            for row in group["molecules"]
        )
    )
    excluded_unseen_element_groups = [
        group
        for group in external_groups
        if any(
            set(row["composition"]) - set(trained_elements)
            for row in group["molecules"]
        )
    ]
    excluded_unseen_elements = sorted(
        {
            element
            for group in excluded_unseen_element_groups
            for row in group["molecules"]
            for element in set(row["composition"]) - set(trained_elements)
        }
    )
    excluded_unseen_element_keys = {
        str(group["key"]) for group in excluded_unseen_element_groups
    }
    external_groups = [
        group
        for group in external_groups
        if str(group["key"]) not in excluded_unseen_element_keys
    ]

    train_rows = _flatten(training_groups)
    test_rows = _flatten(external_groups)
    train_y = np.asarray([row["label"] for row in train_rows], dtype=float)
    test_y = np.asarray([row["label"] for row in test_rows], dtype=float)
    test_keys = [str(row["group_key"]) for row in test_rows]
    train_composition, composition_names = _composition_matrix(
        train_rows, elements=trained_elements
    )
    test_composition, _ = _composition_matrix(test_rows, elements=trained_elements)
    train_enhanced, enhanced_names = _composition_matrix(
        train_rows, elements=trained_elements, enhanced=True
    )
    test_enhanced, _ = _composition_matrix(
        test_rows, elements=trained_elements, enhanced=True
    )
    train_features = _feature_matrix(train_rows, feature_names)
    test_features = _feature_matrix(test_rows, feature_names)
    connectivity_vector = np.asarray(
        [connectivity_weights[name] for name in feature_names], dtype=float
    )
    train_connectivity = train_features @ connectivity_vector
    test_connectivity = test_features @ connectivity_vector

    predictions: dict[str, np.ndarray] = {}
    predictions["mean_only"] = np.full(len(test_y), float(np.mean(train_y)))
    predictions["composition_linear"], _, _ = _fit_linear(
        train_composition, train_y, test_composition
    )
    predictions["composition_plus_size"], _, _ = _fit_linear(
        train_enhanced, train_y, test_enhanced
    )
    connectivity_intercept = float(np.mean(train_y - train_connectivity))
    predictions["connectivity_without_composition"] = (
        connectivity_intercept + test_connectivity
    )

    combined_target = train_y - train_connectivity
    combined_composition, intercept, composition_coefficients = _fit_linear(
        train_composition, combined_target, test_composition
    )
    predictions["composition_plus_synde_connectivity"] = (
        combined_composition + test_connectivity
    )

    direct_scaler = StandardScaler()
    direct_train = direct_scaler.fit_transform(
        np.column_stack((train_composition, train_features))
    )
    direct_test = direct_scaler.transform(
        np.column_stack((test_composition, test_features))
    )
    direct_model = Ridge(alpha=DIRECT_JOINT_RIDGE_ALPHA).fit(direct_train, train_y)
    predictions["direct_joint_ridge"] = direct_model.predict(direct_test)
    del direct_train, direct_test

    block_by_name = {
        name: (
            "v3_quantum_graph"
            if name in V3_FEATURE_FAMILIES
            else (
                "first_order"
                if name.startswith("first_order_v1[")
                else "v2_named_graph"
            )
        )
        for name in feature_names
    }
    ablation_predictions: dict[str, np.ndarray] = {}
    for omitted in ("first_order", "v2_named_graph", "v3_quantum_graph"):
        retained = np.asarray(
            [block_by_name[name] != omitted for name in feature_names], dtype=bool
        )
        train_score = train_features[:, retained] @ connectivity_vector[retained]
        test_score = test_features[:, retained] @ connectivity_vector[retained]
        composition_prediction, _, _ = _fit_linear(
            train_composition, train_y - train_score, test_composition
        )
        ablation_predictions[f"combined_without_{omitted}"] = (
            composition_prediction + test_score
        )

    metrics = {
        name: _metrics(
            test_y,
            values,
            test_keys,
            namespace=f"{PROTOCOL}:{name}",
            bootstrap_replicates=bootstrap_replicates,
        )
        for name, values in predictions.items()
    }
    ablations = {
        name: _metrics(
            test_y,
            values,
            test_keys,
            namespace=f"{PROTOCOL}:{name}",
            bootstrap_replicates=bootstrap_replicates,
        )
        for name, values in ablation_predictions.items()
    }

    feature_means_array = np.mean(train_features, axis=0)
    feature_scales_array = np.std(train_features, axis=0)
    feature_scales_array[feature_scales_array <= 0] = 1.0
    train_distances = np.sqrt(
        np.mean(
            ((train_features - feature_means_array) / feature_scales_array) ** 2, axis=1
        )
    )
    test_distances = np.sqrt(
        np.mean(
            ((test_features - feature_means_array) / feature_scales_array) ** 2, axis=1
        )
    )
    distance_q99 = float(np.quantile(train_distances, 0.99))

    composition_ranges = {
        element: (
            int(np.min(train_composition[:, index])),
            int(np.max(train_composition[:, index])),
        )
        for index, element in enumerate(trained_elements)
    }
    card = SynDEEnergyModelCard(
        model_name="synde-gfn2-total-energy-v2-amended-domain",
        schema_version=2,
        target=TARGET,
        units="eV",
        reference_protocol=REFERENCE_PROTOCOL,
        training_source=str(training_summary["source"]),
        training_source_sha256=str(training_summary["source_sha256"]),
        training_groups=len(training_groups),
        training_molecules=len(train_rows),
        evaluation_status=(
            "formula- and connectivity-disjoint external validation; "
            "external labels were not used for fitting or selection"
        ),
        formula_disjoint_evaluation=True,
        connectivity_disjoint_evaluation=True,
        composition_model="ordinary least squares over elemental atom counts",
        connectivity_model=(
            f"frozen {len(connectivity_weights)}-term SynDE equation fitted on "
            "formula-centered targets; composition baseline fitted to its raw "
            "total-energy residual"
        ),
        supported_elements=trained_elements,
        supported_formal_charges=(0,),
        uses_coordinates_at_inference=False,
        uses_conformers_at_inference=False,
        connectivity_equation_unchanged=True,
        connectivity_refit_on_amended_development_cohort=True,
        ordinary_explicit_hydrogen_policy="normalize_to_implicit",
        isotope_policy="exclude_nonzero_isotope_numbers",
    )
    predictor = SynDEEnergyPredictor(
        card=card,
        intercept=intercept,
        composition_weights={
            element: float(composition_coefficients[index])
            for index, element in enumerate(trained_elements)
        },
        connectivity_weights=connectivity_weights,
        feature_means={
            name: float(feature_means_array[index])
            for index, name in enumerate(feature_names)
        },
        feature_scales={
            name: float(feature_scales_array[index])
            for index, name in enumerate(feature_names)
        },
        training_distance_q99=distance_q99,
        composition_ranges=composition_ranges,
    )
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(
        json.dumps(predictor.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    combined = predictions["composition_plus_synde_connectivity"]
    amended_ranking = _ranking_metrics(test_rows, combined, bootstrap_replicates)
    compatibility_ranking = {
        key.replace("_group_bootstrap_95ci", "_bootstrap_95ci"): value
        for key, value in amended_ranking.items()
    }
    result = {
        "protocol": PROTOCOL,
        "program_sha256": _sha256(Path(__file__)),
        "target": {
            "quantity": TARGET,
            "units": "eV",
            "reference_protocol": REFERENCE_PROTOCOL,
            "cross_formula_comparable_only_under_same_reference_protocol": True,
            "not_experimental_energy_or_free_energy": True,
        },
        "scientific_status": card.evaluation_status,
        "data": {
            "training_cache": str(training_cache),
            "training_cache_sha256": _sha256(training_cache),
            "training_cache_summary": str(training_cache_summary),
            "training_cache_summary_sha256": _sha256(training_cache_summary),
            "formula_relative_model": str(formula_relative_model),
            "formula_relative_model_sha256": _sha256(formula_relative_model),
            "external_cache": external_cache_summary,
        },
        "software": {"rdkit": rdBase.rdkitVersion, "numpy": np.__version__},
        "split": {
            "training_groups": len(training_groups),
            "training_molecules": len(train_rows),
            "evaluation_groups": len(external_groups),
            "evaluation_molecules": len(test_rows),
            "excluded_unseen_element_groups": len(excluded_unseen_element_groups),
            "excluded_unseen_element_molecules": sum(
                len(group["molecules"]) for group in excluded_unseen_element_groups
            ),
            "unseen_elements": excluded_unseen_elements,
            "trained_elements": list(trained_elements),
            "formula_disjoint": True,
            "connectivity_disjoint": True,
            "maximum_molecules_per_formula": MAX_MOLECULES_PER_GROUP,
        },
        "models": {
            "mean_only": "training-target mean",
            "composition_linear": composition_names,
            "composition_plus_size": enhanced_names,
            "connectivity_without_composition": "fixed formula-centered SynDE score plus global intercept",
            "composition_plus_synde_connectivity": (
                "element-count fit to target minus fixed formula-centered SynDE connectivity score"
            ),
            "direct_joint_ridge": {
                "features": "element counts plus the selected SynDE feature support",
                "alpha": DIRECT_JOINT_RIDGE_ALPHA,
            },
        },
        "cross_formula_metrics": metrics,
        "ablations": ablations,
        "new_model_same_formula_ranking": amended_ranking,
        "stratified_combined_metrics": _stratified_metrics(
            test_rows, test_y, combined, test_distances, distance_q99
        ),
        "applicability": {
            "training_feature_distance_q99": distance_q99,
            "evaluation_molecules_outside_q99": int(
                np.sum(test_distances > distance_q99)
            ),
            "composition_ranges": {
                name: list(bounds) for name, bounds in composition_ranges.items()
            },
        },
        "amended_connectivity_equation_validation": {
            "result_path": "current_output_record",
            "frozen_model_path": str(formula_relative_model),
            "frozen_model_sha256": _sha256(formula_relative_model),
            "external_performance": compatibility_ranking,
            "note": (
                "Constructed from the same frozen amended prediction vector; "
                "the elemental calibration does not change within-formula differences."
            ),
        },
        "energy_model": {
            "path": str(model_output),
            "sha256": _sha256(model_output),
            "card": asdict(card),
            "selected_connectivity_terms": len(connectivity_weights),
            "contribution_identity": "prediction = intercept + sum(atom-count terms) + sum(connectivity terms)",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--training-cache-summary", type=Path, required=True)
    parser.add_argument("--formula-relative-model", type=Path, required=True)
    parser.add_argument(
        "--external-csv", type=Path, default=Path("data/ord_test_xtb.csv")
    )
    parser.add_argument(
        "--external-cache",
        type=Path,
        default=Path("/tmp/synde-energy-external-validation/external.joblib"),
    )
    parser.add_argument("--external-cache-summary", type=Path)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("/tmp/synde-energy-external-validation/synde_energy_model.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--rebuild-external-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/synde-energy-external-validation/synde.json"),
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 grouped bootstrap replicates.")
    if args.workers < 1:
        raise ValueError("Use at least one feature-extraction worker.")
    for label, path in {
        "training cache": args.training_cache,
        "training cache summary": args.training_cache_summary,
        "frozen connectivity model": args.formula_relative_model,
        "test labels": args.external_csv,
    }.items():
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    payload = run(
        training_cache=args.training_cache,
        training_cache_summary=args.training_cache_summary,
        formula_relative_model=args.formula_relative_model,
        external_csv=args.external_csv,
        external_cache=args.external_cache,
        external_cache_summary_path=args.external_cache_summary,
        model_output=args.model_output,
        bootstrap_replicates=args.bootstrap_replicates,
        rebuild_external_cache=args.rebuild_external_cache,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
