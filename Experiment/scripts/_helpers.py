"""Shared experiment helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from synde.chem import (
    ISOTOPE_EXCLUSION_REASON,
    has_isotopically_labelled_atom,
    normalize_ordinary_explicit_hydrogens,
)
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

BOOTSTRAP_REPLICATES = 10_000
MIN_GROUP_SIZE = 3


def load_calibration_seed(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Load the version-neutral seed and restore runtime descriptor names."""
    seed = json.loads(path.read_text(encoding="utf-8"))

    def runtime_name(name: str) -> str:
        return name.replace("quantum_graph_", "v3_", 1)

    stored_model = seed["model"]
    model = {
        "weights": {
            runtime_name(str(name)): float(value)
            for name, value in stored_model["weights"].items()
        },
        "selected_terms": [
            runtime_name(str(name)) for name in stored_model["selected_terms"]
        ],
    }
    return seed, model


def public_record(value: object) -> object:
    """Replace historical descriptor prefixes in serialized result records."""
    if isinstance(value, dict):
        return {
            str(key).replace("v3_", "quantum_graph_"): public_record(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [public_record(item) for item in value]
    if isinstance(value, str):
        return value.replace("v3_", "quantum_graph_").replace(
            "synde-ord-v4-", "synde-ord-"
        )
    return value


def rank(values: list[float]) -> np.ndarray:
    """Return average ranks, including ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def correlation(left: list[float], right: list[float]) -> float:
    """Return Pearson correlation, or NaN for an undefined correlation."""
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _hash_bucket(namespace: str, key: str, count: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{key}".encode()).hexdigest()
    return int(digest[:8], 16) % count


def _compact(features: dict[str, float]) -> dict[str, float]:
    """Remove high-cardinality paths and nearest-neighbour strings."""
    excluded = ("path_1_3[", "path_1_4[", "benson_nn[", "inactive_empirical[")
    return {
        name: value for name, value in features.items() if not name.startswith(excluded)
    }


def _center_sparse(
    matrix: sparse.spmatrix, bounds: list[tuple[int, int]]
) -> sparse.csr_matrix:
    blocks = []
    for start, end in bounds:
        block = matrix[start:end].toarray()
        block -= block.mean(axis=0, keepdims=True)
        blocks.append(sparse.csr_matrix(block))
    if not blocks:
        return sparse.csr_matrix((0, matrix.shape[1]), dtype=float)
    return sparse.vstack(blocks, format="csr")


def _prepare(
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    selected_names: set[str] | None = None,
) -> tuple[
    DictVectorizer,
    StandardScaler,
    sparse.csr_matrix,
    sparse.csr_matrix,
    np.ndarray,
]:
    vectorizer = DictVectorizer(sparse=True, sort=True)
    train_features = [
        _compact(row["features"]) for group in train for row in group["molecules"]
    ]
    test_features = [
        _compact(row["features"]) for group in test for row in group["molecules"]
    ]
    train_matrix = vectorizer.fit_transform(train_features)
    test_matrix = (
        vectorizer.transform(test_features)
        if test_features
        else sparse.csr_matrix((0, train_matrix.shape[1]), dtype=float)
    )
    train_bounds: list[tuple[int, int]] = []
    targets: list[float] = []
    position = 0
    for group in train:
        molecules = group["molecules"]
        train_bounds.append((position, position + len(molecules)))
        values = np.asarray([row["label"] for row in molecules], dtype=float)
        targets.extend((values - values.mean()).tolist())
        position += len(molecules)
    test_bounds: list[tuple[int, int]] = []
    position = 0
    for group in test:
        count = len(group["molecules"])
        test_bounds.append((position, position + count))
        position += count
    train_matrix = _center_sparse(train_matrix, train_bounds)
    test_matrix = _center_sparse(test_matrix, test_bounds)
    if selected_names is not None:
        indices = [
            index
            for index, name in enumerate(vectorizer.feature_names_)
            if name in selected_names
        ]
        train_matrix = train_matrix[:, indices]
        test_matrix = test_matrix[:, indices]
        vectorizer.feature_names_ = [
            vectorizer.feature_names_[index] for index in indices
        ]
        vectorizer.vocabulary_ = {
            name: index for index, name in enumerate(vectorizer.feature_names_)
        }
    scaler = StandardScaler(with_mean=False)
    train_scaled = scaler.fit_transform(train_matrix).tocsr()
    test_scaled = (
        scaler.transform(test_matrix).tocsr()
        if test_matrix.shape[0]
        else sparse.csr_matrix((0, train_matrix.shape[1]), dtype=float)
    )
    return (
        vectorizer,
        scaler,
        train_scaled,
        test_scaled,
        np.asarray(targets, dtype=float),
    )


def _bootstrap_interval(values: list[float], namespace: str) -> list[float]:
    seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = np.mean(
        array[rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))],
        axis=1,
    )
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _feature_family(name: str) -> str:
    if name.startswith("atom_state["):
        return "atom_state"
    if name.startswith("bond_environment["):
        return "bond_environment"
    if name.startswith("ring[") or name.startswith("ring_"):
        return "ring_topology"
    if name.startswith("fukui_") or name.startswith("huckel_"):
        return "huckel_fukui"
    if name.startswith("first_order_v1["):
        return "first_order_v1"
    if name.startswith("inactive_empirical["):
        return "inactive_empirical"
    if name.startswith("estate_"):
        return "electrotopological_state"
    if name.startswith("gasteiger_"):
        return "gasteiger_charge"
    if name.startswith("rdkit_"):
        return "named_chemical_graph_descriptor"
    if "charge" in name:
        return "formal_charge_topology"
    if name.startswith("graph_"):
        return "classical_graph_index"
    return "other_named_term"


def _selected_terms(
    train: list[dict[str, object]],
    validation: list[dict[str, object]],
    *,
    alpha: float,
    allowed: set[str],
) -> set[str]:
    vectorizer, _, matrix, _, target = _prepare(
        train, validation, selected_names=allowed
    )
    if matrix.shape[1] == 0:
        return set()
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=1.0,
        fit_intercept=False,
        max_iter=30_000,
        tol=1e-5,
        selection="cyclic",
    ).fit(matrix, target)
    return {
        vectorizer.feature_names_[index]
        for index, value in enumerate(model.coef_)
        if abs(value) > 1e-10
    }


def _prediction_rows(
    groups: list[dict[str, object]], predictions: np.ndarray
) -> list[dict[str, object]]:
    rows = []
    position = 0
    for group in groups:
        molecules = group["molecules"]
        count = len(molecules)
        predicted = np.asarray(predictions[position : position + count], dtype=float)
        labels = np.asarray([row["label"] for row in molecules], dtype=float)
        position += count
        pearson = correlation(predicted.tolist(), labels.tolist())
        spearman = correlation(rank(predicted).tolist(), rank(labels).tolist())
        if not (math.isfinite(pearson) and math.isfinite(spearman)):
            continue
        concordance_values = []
        for left in range(count):
            for right in range(left + 1, count):
                product = (predicted[left] - predicted[right]) * (
                    labels[left] - labels[right]
                )
                concordance_values.append(
                    1.0 if product > 0 else 0.5 if product == 0 else 0.0
                )
        rows.append(
            {
                "key": str(group["key"]),
                "size": count,
                "pearson": float(pearson),
                "spearman": float(spearman),
                "pairwise_concordance": float(np.mean(concordance_values)),
                "top1_accuracy": int(np.argmin(predicted) == np.argmin(labels)),
                "predictions": predicted.tolist(),
                "labels": labels.tolist(),
            }
        )
    return rows


def _summary(rows: list[dict[str, object]], namespace: str) -> dict[str, object]:
    pearson = [float(row["pearson"]) for row in rows]
    spearman = [float(row["spearman"]) for row in rows]
    concordance = [float(row["pairwise_concordance"]) for row in rows]
    top1 = [float(row["top1_accuracy"]) for row in rows]
    sizes = np.asarray([row["size"] for row in rows], dtype=float)
    return {
        "groups": len(rows),
        "molecules": int(np.sum(sizes)),
        "mean_pearson": float(np.mean(pearson)),
        "mean_pearson_bootstrap_95ci": _bootstrap_interval(
            pearson, f"{namespace}:pearson"
        ),
        "mean_spearman": float(np.mean(spearman)),
        "mean_spearman_bootstrap_95ci": _bootstrap_interval(
            spearman, f"{namespace}:spearman"
        ),
        "weighted_pearson": float(np.average(pearson, weights=sizes)),
        "weighted_spearman": float(np.average(spearman, weights=sizes)),
        "positive_pearson_fraction": float(np.mean(np.asarray(pearson) > 0)),
        "mean_pairwise_concordance": float(np.mean(concordance)),
        "top1_accuracy": float(np.mean(top1)),
    }


def _composite(rows: list[dict[str, object]]) -> float:
    return float(
        (
            np.mean([float(row["pearson"]) for row in rows])
            + np.mean([float(row["spearman"]) for row in rows])
        )
        / 2
    )


def _fit_predict(
    train: list[dict[str, object]],
    test: list[dict[str, object]],
    support: set[str],
    ridge_alpha: float,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, float]]:
    vectorizer, scaler, train_x, test_x, train_y = _prepare(
        train, test, selected_names=support
    )
    if train_x.shape[1] == 0:
        return [], {}, {}
    model = Ridge(alpha=ridge_alpha, solver="lsqr", tol=1e-7, max_iter=10_000).fit(
        train_x, train_y
    )
    rows = _prediction_rows(test, model.predict(test_x))
    raw = model.coef_ / scaler.scale_
    return (
        rows,
        {name: float(value) for name, value in zip(vectorizer.feature_names_, raw)},
        {
            name: float(value)
            for name, value in zip(vectorizer.feature_names_, model.coef_)
        },
    )


def _available_names(groups: list[dict[str, object]]) -> set[str]:
    return {
        name
        for group in groups
        for row in group["molecules"]
        for name in row["features"]
        if not name.startswith(
            ("path_1_3[", "path_1_4[", "benson_nn[", "inactive_empirical[")
        )
    }


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


def _load_candidates(
    path: Path, *, supported_elements: set[str]
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Load clean, unique achiral connectivities without using label magnitude."""
    audit = Counter()
    candidates: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        smiles_column = _smiles_column(reader.fieldnames)
        for row in reader:
            audit["source_rows"] += 1
            if row.get("status") != "success":
                audit["excluded_non_success"] += 1
                continue
            try:
                raw_label = row.get("energy") or row.get("energy_eV")
                label = float(raw_label)
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
            if not elements <= supported_elements:
                audit["excluded_unsupported_element"] += 1
                continue
            key, connectivity = _identity(molecule)
            candidates[key][connectivity].append(
                {
                    "smiles": Chem.MolToSmiles(
                        molecule, canonical=True, isomericSmiles=True
                    ),
                    "label": label,
                    "connectivity": connectivity,
                    "scaffold": MurckoScaffold.MurckoScaffoldSmiles(
                        mol=molecule, includeChirality=False
                    ),
                    "molecule": molecule,
                    "elements": sorted(elements),
                }
            )
            audit["clean_rows_before_achiral_deduplication"] += 1

    groups: dict[str, list[dict[str, object]]] = {}
    duplicate_connectivities = 0
    duplicate_rows = 0
    for key, by_connectivity in candidates.items():
        rows = []
        for values in by_connectivity.values():
            if len(values) != 1:
                duplicate_connectivities += 1
                duplicate_rows += len(values)
                continue
            rows.append(values[0])
        groups[key] = rows
    audit["excluded_achiral_duplicate_connectivities"] = duplicate_connectivities
    audit["excluded_achiral_duplicate_rows"] = duplicate_rows
    audit["formula_charge_groups_after_cleaning"] = len(groups)
    audit["eligible_groups_before_firewall"] = sum(
        len(rows) >= MIN_GROUP_SIZE for rows in groups.values()
    )
    audit["eligible_molecules_before_firewall"] = sum(
        len(rows) for rows in groups.values() if len(rows) >= MIN_GROUP_SIZE
    )
    return groups, dict(sorted(audit.items()))
