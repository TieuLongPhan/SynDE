#!/usr/bin/env python3
"""Train fair global/local fingerprint and Chemprop comparators.

Every structural model is trained on energy centered within formula groups.
An elemental composition layer is then fitted, using training data only, to
the raw-energy residual left by that structural model.  The resulting single
prediction is evaluated globally (MAE, RMSE, R2, Pearson, and Spearman) and
locally after grouping by formula (Pearson, Spearman, concordance, and top-1).

The script consumes the exact training and external caches produced by
``07_train_energy.py`` so every comparator uses the same molecules and
formula/connectivity firewall as SynDE.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import joblib
import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.scripts._helpers import _hash_bucket  # noqa: E402
from Experiment.scripts._fit_energy import (  # noqa: E402
    SUPPORTED_ELEMENTS,
    _composition_matrix,
    _ensure_training_metadata,
    _flatten,
    _metrics,
    _ranking_metrics,
)

PROTOCOL = "synde-global-local-external-comparators-v1"
CV_NAMESPACE = "synde-comparator-formula-folds-v1"
CV_FOLDS = 5
RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cohorts(
    training_cache: Path, external_cache: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]]:
    training_groups = joblib.load(training_cache)
    external_groups = joblib.load(external_cache)
    _ensure_training_metadata(training_groups)
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
    supported = set(trained_elements)
    external_groups = [
        group
        for group in external_groups
        if all(set(row["composition"]) <= supported for row in group["molecules"])
    ]
    return training_groups, external_groups, trained_elements


def _group_bounds(groups: list[dict[str, Any]]) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for group in groups:
        end = start + len(group["molecules"])
        bounds.append((start, end))
        start = end
    return bounds


def _center_values(values: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    centered = values.copy()
    for start, end in bounds:
        centered[start:end] -= np.mean(centered[start:end])
    return centered


def _center_sparse(
    matrix: sparse.csr_matrix, bounds: list[tuple[int, int]]
) -> sparse.csr_matrix:
    blocks: list[sparse.csr_matrix] = []
    for start, end in bounds:
        block = matrix[start:end]
        mean = sparse.csr_matrix(np.asarray(block.mean(axis=0), dtype=np.float32))
        blocks.append(block - sparse.vstack([mean] * (end - start), format="csr"))
    return sparse.vstack(blocks, format="csr")


def _row_folds(groups: list[dict[str, Any]]) -> np.ndarray:
    folds: list[int] = []
    for group in groups:
        fold = _hash_bucket(CV_NAMESPACE, str(group["key"]), CV_FOLDS)
        folds.extend([fold] * len(group["molecules"]))
    return np.asarray(folds, dtype=np.int8)


def _mean_group_rank_score(
    rows: list[dict[str, Any]], predictions: np.ndarray
) -> tuple[float, float, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["group_key"])].append(index)
    pearsons: list[float] = []
    spearmans: list[float] = []
    for indices in grouped.values():
        labels = np.asarray([float(rows[index]["label"]) for index in indices])
        scores = predictions[indices]
        if np.std(labels) == 0 or np.std(scores) == 0:
            continue
        pearsons.append(float(np.corrcoef(labels, scores)[0, 1]))
        spearmans.append(float(np.corrcoef(rankdata(labels), rankdata(scores))[0, 1]))
    mean_pearson = float(np.mean(pearsons))
    mean_spearman = float(np.mean(spearmans))
    return (mean_pearson + mean_spearman) / 2.0, mean_pearson, mean_spearman


def _evaluate_prediction(
    name: str,
    external_rows: list[dict[str, Any]],
    predictions: np.ndarray,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    labels = np.asarray([float(row["label"]) for row in external_rows])
    keys = [str(row["group_key"]) for row in external_rows]
    return {
        "global": _metrics(
            labels,
            predictions,
            keys,
            namespace=f"{PROTOCOL}:{name}:global",
            bootstrap_replicates=bootstrap_replicates,
        ),
        "local": _ranking_metrics(external_rows, predictions, bootstrap_replicates),
    }


def _fit_composition_layer(
    train_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    train_structural: np.ndarray,
    external_structural: np.ndarray,
    elements: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    train_y = np.asarray([float(row["label"]) for row in train_rows])
    train_composition, names = _composition_matrix(train_rows, elements=elements)
    external_composition, _ = _composition_matrix(external_rows, elements=elements)
    model = LinearRegression().fit(train_composition, train_y - train_structural)
    predictions = model.predict(external_composition) + external_structural
    return predictions, {
        "model": "ordinary least squares on elemental atom counts",
        "features": names,
        "intercept": float(model.intercept_),
        "weights": {
            element: float(model.coef_[index]) for index, element in enumerate(elements)
        },
        "fit_target": "raw energy minus final structural-model score",
        "training_only": True,
    }


def _explicit_hydrogen_molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")
    return Chem.AddHs(molecule)


def _bit_indices(bit_vector: DataStructs.ExplicitBitVect) -> np.ndarray:
    return np.asarray(bit_vector.GetOnBits(), dtype=np.int32)


def _fingerprinters() -> (
    dict[str, tuple[int, Callable[[Chem.Mol], np.ndarray], dict[str, Any]]]
):
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return {
        "ecfp4_ridge": (
            2048,
            lambda molecule: _bit_indices(morgan.GetFingerprint(molecule)),
            {"algorithm": "Morgan/ECFP", "radius": 2, "bits": 2048},
        ),
        "rdk5_ridge": (
            2048,
            lambda molecule: _bit_indices(
                Chem.RDKFingerprint(
                    molecule,
                    minPath=1,
                    maxPath=5,
                    fpSize=2048,
                    nBitsPerHash=2,
                    useHs=True,
                    branchedPaths=True,
                    useBondOrder=True,
                )
            ),
            {"algorithm": "RDKit topological path", "maximum_path": 5, "bits": 2048},
        ),
        "maccs_ridge": (
            167,
            lambda molecule: _bit_indices(MACCSkeys.GenMACCSKeys(molecule)),
            {"algorithm": "MACCS structural keys", "bits": 167},
        ),
    }


def _fingerprint_matrix(
    rows: list[dict[str, Any]],
    width: int,
    fingerprint: Callable[[Chem.Mol], np.ndarray],
) -> sparse.csr_matrix:
    row_indices: list[int] = []
    column_indices: list[int] = []
    for index, row in enumerate(rows):
        bits = fingerprint(_explicit_hydrogen_molecule(str(row["smiles"])))
        row_indices.extend([index] * len(bits))
        column_indices.extend(bits.tolist())
    return sparse.csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.float32),
            (np.asarray(row_indices), np.asarray(column_indices)),
        ),
        shape=(len(rows), width),
        dtype=np.float32,
    )


def _ridge(alpha: float) -> Ridge:
    return Ridge(
        alpha=alpha,
        fit_intercept=False,
        solver="lsqr",
        tol=1e-7,
        max_iter=10_000,
    )


def _train_fingerprint_models(
    training_groups: list[dict[str, Any]],
    external_groups: list[dict[str, Any]],
    elements: tuple[str, ...],
    work_dir: Path,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    train_rows = _flatten(training_groups)
    external_rows = _flatten(external_groups)
    train_y = np.asarray([float(row["label"]) for row in train_rows])
    centered_y = _center_values(train_y, _group_bounds(training_groups))
    folds = _row_folds(training_groups)
    results: dict[str, Any] = {}
    for name, (width, fingerprint, definition) in _fingerprinters().items():
        started = time.perf_counter()
        print(f"[comparators] featurizing {name}", flush=True)
        raw_train = _fingerprint_matrix(train_rows, width, fingerprint)
        raw_external = _fingerprint_matrix(external_rows, width, fingerprint)
        centered_train = _center_sparse(raw_train, _group_bounds(training_groups))
        candidates: list[dict[str, Any]] = []
        oof_by_alpha = {
            alpha: np.empty(len(train_rows), dtype=float) for alpha in RIDGE_ALPHA_GRID
        }
        for fold in range(CV_FOLDS):
            fit_indices = np.flatnonzero(folds != fold)
            validation_indices = np.flatnonzero(folds == fold)
            scaler = StandardScaler(with_mean=False)
            fit_x = sparse.csr_matrix(scaler.fit_transform(centered_train[fit_indices]))
            validation_x = sparse.csr_matrix(
                scaler.transform(centered_train[validation_indices])
            )
            for alpha in RIDGE_ALPHA_GRID:
                model = _ridge(alpha).fit(fit_x, centered_y[fit_indices])
                oof_by_alpha[alpha][validation_indices] = model.predict(validation_x)
        for alpha, oof in oof_by_alpha.items():
            composite, pearson, spearman = _mean_group_rank_score(train_rows, oof)
            candidates.append(
                {
                    "alpha": alpha,
                    "mean_rank_correlation": composite,
                    "mean_pearson": pearson,
                    "mean_spearman": spearman,
                }
            )
        chosen = max(
            candidates,
            key=lambda row: (
                float(row["mean_rank_correlation"]),
                -abs(np.log10(float(row["alpha"])) - 1.0),
            ),
        )
        alpha = float(chosen["alpha"])
        scaler = StandardScaler(with_mean=False)
        final_train_x = sparse.csr_matrix(scaler.fit_transform(centered_train))
        model = _ridge(alpha).fit(final_train_x, centered_y)
        train_structural = np.asarray(
            model.predict(sparse.csr_matrix(scaler.transform(raw_train))), dtype=float
        )
        external_structural = np.asarray(
            model.predict(sparse.csr_matrix(scaler.transform(raw_external))),
            dtype=float,
        )
        predictions, composition = _fit_composition_layer(
            train_rows,
            external_rows,
            train_structural,
            external_structural,
            elements,
        )
        model_dir = work_dir / "artifacts" / "fingerprints"
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact = model_dir / f"{name}.joblib"
        joblib.dump(
            {
                "definition": definition,
                "explicit_hydrogens": True,
                "ridge_alpha": alpha,
                "scaler": scaler,
                "structural_model": model,
                "composition": composition,
            },
            artifact,
            compress=3,
        )
        results[name] = {
            "representation": definition,
            "explicit_hydrogens": True,
            "structural_target": "energy centered within formula/formal-charge group",
            "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
            "training_only_selection": candidates,
            "selected_ridge_alpha": alpha,
            "composition_calibration": composition,
            "metrics": _evaluate_prediction(
                name, external_rows, predictions, bootstrap_replicates
            ),
            "artifact": str(artifact),
            "artifact_sha256": _sha256(artifact),
            "elapsed_seconds": time.perf_counter() - started,
        }
    return results


def _write_chemprop_training_csv(
    path: Path,
    groups: list[dict[str, Any]],
    validation_fold: int,
) -> None:
    bounds = _group_bounds(groups)
    rows = _flatten(groups)
    labels = np.asarray([float(row["label"]) for row in rows])
    centered = _center_values(labels, bounds)
    folds = _row_folds(groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("smiles", "centered_energy", "split")
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "smiles": row["smiles"],
                    "centered_energy": centered[index],
                    "split": "val" if folds[index] == validation_fold else "train",
                }
            )


def _write_smiles_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("smiles",))
        writer.writeheader()
        writer.writerows({"smiles": row["smiles"]} for row in rows)


def _run_logged(command: list[str], environment: dict[str, str], log: Path) -> None:
    print(f"[comparators] running: {' '.join(command)}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-40:])
        raise RuntimeError(f"Chemprop command failed; see {log}\n{tail}")


def _read_chemprop_predictions(path: Path, expected_smiles: list[str]) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [str(row["smiles"]) for row in rows] != expected_smiles:
        raise RuntimeError(f"Chemprop prediction order differs in {path}.")
    columns = [name for name in rows[0] if name != "smiles"]
    if len(columns) != 1:
        raise RuntimeError(f"Expected one Chemprop target column, found {columns}.")
    return np.asarray([float(row[columns[0]]) for row in rows], dtype=float)


def _train_chemprop(
    args: argparse.Namespace,
    training_groups: list[dict[str, Any]],
    external_groups: list[dict[str, Any]],
    elements: tuple[str, ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    train_rows = _flatten(training_groups)
    external_rows = _flatten(external_groups)
    executable = Path(sys.executable).with_name("chemprop")
    chemprop = str(executable) if executable.is_file() else shutil.which("chemprop")
    if chemprop is None:
        raise RuntimeError("Chemprop is not installed in the active environment.")
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(args.work_dir / "matplotlib")
    environment["PYTHONHASHSEED"] = str(args.seed)
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    chemprop_dir = args.work_dir / "artifacts" / "chemprop"
    train_prediction_input = chemprop_dir / "training_smiles.csv"
    external_prediction_input = chemprop_dir / "external_smiles.csv"
    _write_smiles_csv(train_prediction_input, train_rows)
    _write_smiles_csv(external_prediction_input, external_rows)
    train_smiles = [str(row["smiles"]) for row in train_rows]
    external_smiles = [str(row["smiles"]) for row in external_rows]
    train_predictions: list[np.ndarray] = []
    external_predictions: list[np.ndarray] = []
    fold_records: list[dict[str, Any]] = []
    for fold in args.chemprop_folds:
        fold_started = time.perf_counter()
        fold_dir = chemprop_dir / f"fold_{fold + 1}"
        data_path = fold_dir / "data.csv"
        output_dir = fold_dir / "model"
        checkpoint = output_dir / "model_0" / "best.pt"
        _write_chemprop_training_csv(data_path, training_groups, fold)
        if args.force_chemprop or not checkpoint.is_file():
            command = [
                chemprop,
                "train",
                "--data-path",
                str(data_path),
                "--task-type",
                "regression",
                "--smiles-columns",
                "smiles",
                "--target-columns",
                "centered_energy",
                "--splits-column",
                "split",
                "--output-dir",
                str(output_dir),
                "--multi-hot-atom-featurizer-mode",
                "V2",
                "--ignore-stereo",
                "--add-h",
                "--message-hidden-dim",
                str(args.hidden_dim),
                "--depth",
                str(args.depth),
                "--dropout",
                str(args.dropout),
                "--aggregation",
                "norm",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--epochs",
                str(args.epochs),
                "--warmup-epochs",
                "1",
                "--patience",
                str(args.patience),
                "--accelerator",
                args.accelerator,
                "--devices",
                args.devices,
                "--pytorch-seed",
                str(args.seed),
                "--data-seed",
                str(args.seed),
                "-q",
            ]
            _run_logged(command, environment, fold_dir / "train.log")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing Chemprop checkpoint: {checkpoint}")
        for split_name, input_path, smiles, collection in (
            ("training", train_prediction_input, train_smiles, train_predictions),
            (
                "external",
                external_prediction_input,
                external_smiles,
                external_predictions,
            ),
        ):
            prediction_path = fold_dir / f"{split_name}_predictions.csv"
            command = [
                chemprop,
                "predict",
                "--test-path",
                str(input_path),
                "--model-paths",
                str(checkpoint),
                "--preds-path",
                str(prediction_path),
                "--smiles-columns",
                "smiles",
                "--ignore-stereo",
                "--add-h",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--accelerator",
                args.accelerator,
                "--devices",
                args.devices,
                "-q",
            ]
            _run_logged(command, environment, fold_dir / f"predict_{split_name}.log")
            collection.append(_read_chemprop_predictions(prediction_path, smiles))
        fold_records.append(
            {
                "validation_fold": fold + 1,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
        )
    train_structural = np.mean(np.vstack(train_predictions), axis=0)
    external_structural = np.mean(np.vstack(external_predictions), axis=0)
    predictions, composition = _fit_composition_layer(
        train_rows,
        external_rows,
        train_structural,
        external_structural,
        elements,
    )
    calibration_path = chemprop_dir / "composition_calibration.json"
    calibration_path.write_text(
        json.dumps(composition, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "representation": {
            "algorithm": "Chemprop directed-message-passing neural network",
            "explicit_hydrogens": True,
            "stereochemistry": "ignored to match constitutional graphs",
        },
        "structural_target": "energy centered within formula/formal-charge group",
        "architecture": {
            "ensemble_size": len(args.chemprop_folds),
            "hidden_dimension": args.hidden_dim,
            "depth": args.depth,
            "dropout": args.dropout,
            "epochs_maximum": args.epochs,
            "early_stopping_patience": args.patience,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "accelerator": args.accelerator,
            "devices": args.devices,
        },
        "folds": fold_records,
        "composition_calibration": composition,
        "metrics": _evaluate_prediction(
            "chemprop_dmpnn", external_rows, predictions, args.bootstrap_replicates
        ),
        "calibration_artifact": str(calibration_path),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _write_summary_tables(
    payload: dict[str, Any], output: Path, synde_result: Path
) -> None:
    rows: list[tuple[str, dict[str, Any]]] = []
    if synde_result.is_file():
        synde = json.loads(synde_result.read_text(encoding="utf-8"))
        rows.append(
            (
                "SynDE",
                {
                    "global": synde["cross_formula_metrics"][
                        "composition_plus_synde_connectivity"
                    ],
                    "local": synde["new_model_same_formula_ranking"],
                },
            )
        )
    rows.extend((name, record["metrics"]) for name, record in payload["models"].items())
    tables_dir = output.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    global_path = tables_dir / "global_metrics.csv"
    local_path = tables_dir / "local_metrics.csv"
    with global_path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "model",
            "mae_eV",
            "rmse_eV",
            "median_absolute_error_eV",
            "r2",
            "pearson",
            "spearman",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, metrics in rows:
            writer.writerow(
                {"model": name}
                | {field: metrics["global"].get(field) for field in fields[1:]}
            )
    with local_path.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "model",
            "mean_pearson",
            "mean_spearman",
            "mean_pairwise_concordance",
            "top1_accuracy",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, metrics in rows:
            writer.writerow(
                {"model": name}
                | {field: metrics["local"].get(field) for field in fields[1:]}
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--external-cache", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("fingerprints", "chemprop"),
        default=("fingerprints", "chemprop"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/synde-external-global-comparators-v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/synde-external-global-comparators-v1/benchmark.json"),
    )
    parser.add_argument(
        "--synde-result",
        type=Path,
        default=Path("Experiment/results/global_comparators/synde.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument(
        "--chemprop-folds", type=int, nargs="+", default=list(range(CV_FOLDS))
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "auto"), default="gpu")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-chemprop", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 grouped bootstrap replicates.")
    if any(not 0 <= fold < CV_FOLDS for fold in args.chemprop_folds):
        raise ValueError(f"Chemprop folds must be in [0, {CV_FOLDS - 1}].")
    for label, path in {
        "training cache": args.training_cache,
        "test cache": args.external_cache,
        "SynDE result": args.synde_result,
    }.items():
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    training_groups, external_groups, elements = _load_cohorts(
        args.training_cache, args.external_cache
    )
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "target": "single-conformer gas-phase GFN2-xTB 6.7.1 optimized total energy in eV",
        "design": {
            "one_prediction_for_global_and_local_evaluation": True,
            "structural_fit": "formula-centered training target",
            "global_calibration": "training-only elemental OLS on raw-energy residual",
            "primary_global_endpoints": ["mae_eV", "rmse_eV"],
            "correlations_reported_but_composition_dominated": True,
            "formula_disjoint": True,
            "connectivity_disjoint": True,
        },
        "software": {
            "python": sys.version.split()[0],
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "data": {
            "training_cache": str(args.training_cache),
            "training_cache_sha256": _sha256(args.training_cache),
            "external_cache": str(args.external_cache),
            "external_cache_sha256": _sha256(args.external_cache),
            "training_groups": len(training_groups),
            "training_molecules": sum(
                len(group["molecules"]) for group in training_groups
            ),
            "evaluation_groups": len(external_groups),
            "evaluation_molecules": sum(
                len(group["molecules"]) for group in external_groups
            ),
            "trained_elements": list(elements),
        },
        "models": {},
    }
    if "fingerprints" in args.models:
        payload["models"].update(
            _train_fingerprint_models(
                training_groups,
                external_groups,
                elements,
                args.work_dir,
                args.bootstrap_replicates,
            )
        )
    if "chemprop" in args.models:
        payload["software"]["chemprop"] = importlib.metadata.version("chemprop")
        payload["software"]["torch"] = importlib.metadata.version("torch")
        payload["models"]["chemprop_dmpnn"] = _train_chemprop(
            args, training_groups, external_groups, elements
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_summary_tables(payload, args.output, args.synde_result)
    print(
        json.dumps(
            {name: record["metrics"] for name, record in payload["models"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
