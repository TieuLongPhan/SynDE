#!/usr/bin/env python3
"""Evaluate the frozen SynDE model once on the external source, without refit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable, Iterable

import joblib
import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.scripts._helpers import correlation, rank  # noqa: E402
from synde.energy import SynDEScorer  # noqa: E402
from synde.graph import GraphBuilder  # noqa: E402
from synde.chem import (  # noqa: E402
    ISOTOPE_EXCLUSION_REASON,
    SUPPORTED_ELEMENT_SET,
    has_isotopically_labelled_atom,
    normalize_ordinary_explicit_hydrogens,
)

PROTOCOL = "synde-ord-v5-frozen-external-v1"
PROTOCOL_DOCUMENT = "Experiment/README.md"
SAMPLE_NAMESPACE = "synde-ord-v4-external-sample-v1"
MIN_GROUP_SIZE = 3
MAX_MOLECULES_PER_GROUP = 10
BOOTSTRAP_REPLICATES = 10_000
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
GROUP_SIMILARITY_FIELD = "nearest_training_tanimoto_median"
EXPECTED_MODEL_SHA256 = (
    "6b04dd12dd22643662f0ea894266bde07d8750b7506447da7545bc675ed0c166"
)
SUPPORTED_ELEMENTS = set(SUPPORTED_ELEMENT_SET)


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


def _scaffold(molecule: Chem.Mol) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)


def _load_external(
    path: Path,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, int]]:
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
            if not elements <= SUPPORTED_ELEMENTS:
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
                    "molecule": molecule,
                    "scaffold": _scaffold(molecule),
                    "elements": sorted(elements),
                }
            )
            audit["clean_rows_before_achiral_deduplication"] += 1

    groups: dict[str, list[dict[str, object]]] = {}
    for key, by_connectivity in candidates.items():
        rows = []
        for values in by_connectivity.values():
            if len(values) != 1:
                audit["excluded_achiral_duplicate_connectivities"] += 1
                audit["excluded_achiral_duplicate_rows"] += len(values)
                continue
            rows.append(values[0])
        groups[key] = rows
    audit["formula_charge_groups_after_cleaning"] = len(groups)
    return groups, dict(sorted(audit.items()))


def _sample_group(key: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SAMPLE_NAMESPACE}:{key}:{row['connectivity']}".encode()
        ).hexdigest(),
    )[:MAX_MOLECULES_PER_GROUP]


def _metric_row(
    key: str, scores: list[float], labels: list[float]
) -> dict[str, object] | None:
    if len(labels) < MIN_GROUP_SIZE or len(set(labels)) < 2:
        return None
    pearson = correlation(scores, labels)
    spearman = correlation(rank(scores).tolist(), rank(labels).tolist())
    if not (math.isfinite(pearson) and math.isfinite(spearman)):
        return None
    concordance = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            product = (scores[left] - scores[right]) * (labels[left] - labels[right])
            concordance.append(1.0 if product > 0 else 0.5 if product == 0 else 0.0)
    return {
        "key": key,
        "size": len(labels),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "pairwise_concordance": float(np.mean(concordance)),
        "top1_accuracy": int(np.argmin(scores) == np.argmin(labels)),
    }


def _bootstrap_mean_interval(values: list[float], namespace: str) -> list[float]:
    array = np.asarray(values, dtype=float)
    seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    chunk = 250
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        count = min(chunk, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = np.mean(array[indices], axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _summary(rows: list[dict[str, object]], namespace: str) -> dict[str, object]:
    if not rows:
        return {
            "groups": 0,
            "molecules": 0,
            "mean_pearson": None,
            "mean_spearman": None,
            "mean_pairwise_concordance": None,
            "top1_accuracy": None,
        }
    output: dict[str, object] = {
        "groups": len(rows),
        "molecules": sum(int(row["size"]) for row in rows),
    }
    for metric in ("pearson", "spearman", "pairwise_concordance", "top1_accuracy"):
        values = [float(row[metric]) for row in rows]
        key = "top1_accuracy" if metric == "top1_accuracy" else f"mean_{metric}"
        output[key] = float(np.mean(values))
        output[f"{key}_bootstrap_95ci"] = _bootstrap_mean_interval(
            values, f"{namespace}:{metric}"
        )
    output["positive_pearson_fraction"] = float(
        np.mean([float(row["pearson"]) > 0 for row in rows])
    )
    return output


def _stratum(
    rows: list[dict[str, object]],
    metadata: dict[str, dict[str, object]],
    predicate: Callable[[dict[str, object]], bool],
    namespace: str,
) -> dict[str, object]:
    selected = [row for row in rows if predicate(metadata[str(row["key"])])]
    return _summary(selected, namespace)


def _similarity_strata(
    rows: list[dict[str, object]],
    metadata: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize groups by their median nearest-development similarity."""
    return {
        "far_group_median_tanimoto_le_0_50": _stratum(
            rows,
            metadata,
            lambda row: float(row[GROUP_SIMILARITY_FIELD]) <= 0.50,
            f"{PROTOCOL}:median-similarity:far",
        ),
        "intermediate_group_median_tanimoto_0_50_to_0_70": _stratum(
            rows,
            metadata,
            lambda row: 0.50 < float(row[GROUP_SIMILARITY_FIELD]) <= 0.70,
            f"{PROTOCOL}:median-similarity:intermediate",
        ),
        "near_group_median_tanimoto_gt_0_70": _stratum(
            rows,
            metadata,
            lambda row: float(row[GROUP_SIMILARITY_FIELD]) > 0.70,
            f"{PROTOCOL}:median-similarity:near",
        ),
    }


def _group_median_similarity_quantiles(
    metadata: dict[str, dict[str, object]],
) -> dict[str, float]:
    values = [float(row[GROUP_SIMILARITY_FIELD]) for row in metadata.values()]
    if not values:
        return {}
    return {
        name: float(value)
        for name, value in zip(
            ("min", "q05", "q25", "median", "q75", "q95", "max"),
            np.quantile(values, (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)),
        )
    }


def _similarity_analysis_record(*, reaggregated: bool) -> dict[str, object]:
    return {
        "molecule_level_definition": (
            "maximum ECFP4 Tanimoto similarity to any development molecule"
        ),
        "group_level_definition": (
            "median molecule-level nearest-development Tanimoto similarity"
        ),
        "group_similarity_field": GROUP_SIMILARITY_FIELD,
        "stratum_thresholds": {
            "far": "<=0.50",
            "intermediate": ">0.50 and <=0.70",
            "near": ">0.70",
        },
        "reaggregated_from_stored_group_records_without_model_rescoring": (
            reaggregated
        ),
    }


def _reaggregate_existing(payload: dict[str, object]) -> dict[str, object]:
    """Rebuild similarity-derived summaries from stored group-level records."""
    rows = list(payload["group_metrics"])
    metadata = dict(payload["group_metadata"])
    previous_strata = dict(payload["strata"])
    retained_strata = {
        key: value
        for key, value in previous_strata.items()
        if "group_max_tanimoto" not in key and "group_median_tanimoto" not in key
    }
    similarity_strata = _similarity_strata(rows, metadata)
    if sum(int(value["groups"]) for value in similarity_strata.values()) != len(rows):
        raise RuntimeError("Median-similarity strata do not partition all groups.")
    if sum(int(value["molecules"]) for value in similarity_strata.values()) != sum(
        int(row["size"]) for row in rows
    ):
        raise RuntimeError("Median-similarity strata do not partition all molecules.")
    payload["strata"] = similarity_strata | retained_strata

    distance_audit = dict(payload["distance_audit"])
    distance_audit.pop("group_max_nearest_training_tanimoto_quantiles", None)
    distance_audit["group_median_nearest_training_tanimoto_quantiles"] = (
        _group_median_similarity_quantiles(metadata)
    )
    payload["distance_audit"] = distance_audit
    payload["program_sha256"] = _sha256(Path(__file__))
    payload["similarity_analysis"] = _similarity_analysis_record(reaggregated=True)
    return payload


def _training_domain(
    cache_path: Path,
    cache_summary_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
    actual_hash = _sha256(cache_path)
    if actual_hash != summary["output_cache_sha256"]:
        raise RuntimeError("Training cache hash does not match its frozen summary.")
    groups = joblib.load(cache_path)
    if len(groups) != int(summary["groups"]):
        raise RuntimeError("Training cache group count does not match its summary.")
    return groups, summary


def run(
    external_path: Path,
    training_cache_path: Path,
    training_cache_summary_path: Path,
    model_path: Path,
    expected_model_sha256: str = EXPECTED_MODEL_SHA256,
) -> dict[str, object]:
    started = time.perf_counter()
    if _sha256(model_path) != expected_model_sha256:
        raise RuntimeError("The model does not match the protocol-frozen hash.")
    scorer = SynDEScorer.load(model_path)
    training_groups, cache_summary = _training_domain(
        training_cache_path, training_cache_summary_path
    )
    training_keys = {str(group["key"]) for group in training_groups}
    training_rows = [
        molecule for group in training_groups for molecule in group["molecules"]
    ]
    training_connectivities = {
        str(molecule["connectivity"]) for molecule in training_rows
    }
    training_scaffolds = {
        str(molecule["scaffold"])
        for molecule in training_rows
        if molecule.get("scaffold")
    }

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
        includeChirality=False,
    )
    training_molecules = [
        Chem.MolFromSmiles(str(row["smiles"])) for row in training_rows
    ]
    if any(molecule is None for molecule in training_molecules):
        raise RuntimeError("A frozen training molecule failed RDKit parsing.")
    training_fingerprints = [
        fingerprint_generator.GetFingerprint(molecule)
        for molecule in training_molecules
        if molecule is not None
    ]

    groups, audit = _load_external(external_path)
    eligible_before_firewall = {
        key: rows
        for key, rows in groups.items()
        if len(rows) >= MIN_GROUP_SIZE
        and len({float(row["label"]) for row in rows}) >= 2
    }
    graph_disjoint = {
        key: [
            row
            for row in rows
            if str(row["connectivity"]) not in training_connectivities
        ]
        for key, rows in eligible_before_firewall.items()
    }
    graph_disjoint = {
        key: rows
        for key, rows in graph_disjoint.items()
        if len(rows) >= MIN_GROUP_SIZE
        and len({float(row["label"]) for row in rows}) >= 2
    }
    primary = {
        key: rows for key, rows in graph_disjoint.items() if key not in training_keys
    }
    sampled = {key: _sample_group(key, rows) for key, rows in sorted(primary.items())}

    metric_rows: list[dict[str, object]] = []
    metadata: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []
    for position, (key, rows) in enumerate(sampled.items(), start=1):
        try:
            graphs = [GraphBuilder.from_smiles(str(row["smiles"])) for row in rows]
            labels = [float(row["label"]) for row in rows]
            results = scorer.score_group(graphs)
            scores = [float(result.score) for result in results]
            metric = _metric_row(key, scores, labels)
            if metric is None:
                failures.append({"key": key, "error": "invalid_group_metric"})
                continue
        except Exception as exc:  # noqa: BLE001 - all external failures are serialized.
            failures.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
            continue

        nearest_similarities = []
        for row in rows:
            fingerprint = fingerprint_generator.GetFingerprint(row["molecule"])
            nearest_similarities.append(
                max(
                    DataStructs.BulkTanimotoSimilarity(
                        fingerprint, training_fingerprints
                    )
                )
            )
        scaffolds = {str(row["scaffold"]) for row in rows if row["scaffold"]}
        outside_q99 = sum(
            scorer.DISTANCE_WARNING in result.warnings for result in results
        )
        metadata[key] = {
            "molecules": len(rows),
            "outside_q99": outside_q99,
            "outside_q99_fraction": outside_q99 / len(rows),
            "exactly_scaffold_unseen": bool(scaffolds)
            and not bool(scaffolds & training_scaffolds),
            "nonempty_scaffold_count": len(scaffolds),
            "nearest_training_tanimoto_min": float(np.min(nearest_similarities)),
            "nearest_training_tanimoto_median": float(np.median(nearest_similarities)),
            "nearest_training_tanimoto_max": float(np.max(nearest_similarities)),
            "sulfur_containing": any("S" in row["elements"] for row in rows),
        }
        metric_rows.append(metric)
        if position % 250 == 0:
            print(
                f"scored {position}/{len(sampled)} frozen-model external groups",
                file=sys.stderr,
                flush=True,
            )

    scoreable_coverage = len(metric_rows) / len(sampled) if sampled else 0.0
    external_summary = _summary(metric_rows, f"{PROTOCOL}:primary")
    strata = {
        **_similarity_strata(metric_rows, metadata),
        "exactly_scaffold_unseen": _stratum(
            metric_rows,
            metadata,
            lambda row: bool(row["exactly_scaffold_unseen"]),
            f"{PROTOCOL}:scaffold-unseen",
        ),
        "sulfur_containing": _stratum(
            metric_rows,
            metadata,
            lambda row: bool(row["sulfur_containing"]),
            f"{PROTOCOL}:sulfur",
        ),
        "group_size_3": _stratum(
            metric_rows,
            metadata,
            lambda row: int(row["molecules"]) == 3,
            f"{PROTOCOL}:size3",
        ),
        "group_size_4": _stratum(
            metric_rows,
            metadata,
            lambda row: int(row["molecules"]) == 4,
            f"{PROTOCOL}:size4",
        ),
        "group_size_5_to_10": _stratum(
            metric_rows,
            metadata,
            lambda row: 5 <= int(row["molecules"]) <= 10,
            f"{PROTOCOL}:size5to10",
        ),
    }

    criteria = {
        "scoreable_coverage_at_least_0_99": scoreable_coverage >= 0.99,
        "mean_pearson_at_least_0_90": float(external_summary["mean_pearson"]) >= 0.90,
        "pearson_lower_95_at_least_0_88": float(
            external_summary["mean_pearson_bootstrap_95ci"][0]
        )
        >= 0.88,
        "mean_spearman_at_least_0_85": float(external_summary["mean_spearman"]) >= 0.85,
        "spearman_lower_95_at_least_0_82": float(
            external_summary["mean_spearman_bootstrap_95ci"][0]
        )
        >= 0.82,
        "mean_pairwise_concordance_at_least_0_90": float(
            external_summary["mean_pairwise_concordance"]
        )
        >= 0.90,
        "top1_accuracy_at_least_0_80": float(external_summary["top1_accuracy"]) >= 0.80,
    }
    criteria["all_external_performance_criteria_met"] = all(criteria.values())

    return {
        "protocol": PROTOCOL,
        "protocol_document": PROTOCOL_DOCUMENT,
        "frozen_model_only_no_refit": True,
        "external_labels_used_for_training": False,
        "training_source": cache_summary["source"],
        "training_source_sha256": cache_summary["source_sha256"],
        "training_cache": str(training_cache_path),
        "training_cache_sha256": _sha256(training_cache_path),
        "training_groups": len(training_groups),
        "training_molecules": len(training_rows),
        "external_source": str(external_path),
        "external_source_sha256": _sha256(external_path),
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "program_sha256": _sha256(Path(__file__)),
        "software": {
            "rdkit_generation_version": "2025.9.6",
            "xtb_generation_version": "6.7.1",
            "rdkit_runtime_version": rdBase.rdkitVersion,
        },
        "source_audit": audit,
        "firewall": {
            "training_formula_groups": len(training_keys),
            "training_achiral_connectivities": len(training_connectivities),
            "all_clean_eligible_external_groups": len(eligible_before_firewall),
            "all_clean_eligible_external_molecules": sum(
                len(rows) for rows in eligible_before_firewall.values()
            ),
            "graph_disjoint_external_groups": len(graph_disjoint),
            "graph_disjoint_external_molecules": sum(
                len(rows) for rows in graph_disjoint.values()
            ),
            "formula_and_graph_disjoint_external_groups": len(primary),
            "formula_and_graph_disjoint_molecules_before_sampling": sum(
                len(rows) for rows in primary.values()
            ),
            "sampled_external_groups": len(sampled),
            "sampled_external_molecules": sum(len(rows) for rows in sampled.values()),
        },
        "coverage": {
            "selected_groups": len(sampled),
            "scoreable_groups": len(metric_rows),
            "group_coverage_fraction": scoreable_coverage,
            "scoring_failures": failures,
            "molecules_outside_training_q99": sum(
                int(row["outside_q99"]) for row in metadata.values()
            ),
            "groups_with_any_molecule_outside_training_q99": sum(
                int(row["outside_q99"]) > 0 for row in metadata.values()
            ),
        },
        "external_performance": external_summary,
        "distance_audit": {
            "fingerprint": (
                f"Morgan radius={MORGAN_RADIUS}, bits={MORGAN_BITS}, "
                "includeChirality=False"
            ),
            "training_nonempty_scaffolds": len(training_scaffolds),
            "exactly_scaffold_unseen_groups": sum(
                bool(row["exactly_scaffold_unseen"]) for row in metadata.values()
            ),
            "group_median_nearest_training_tanimoto_quantiles": (
                _group_median_similarity_quantiles(metadata)
            ),
        },
        "similarity_analysis": _similarity_analysis_record(reaggregated=False),
        "strata": strata,
        "external_performance_criteria": criteria,
        "group_metrics": metric_rows,
        "group_metadata": metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-csv", type=Path, default=Path("data/ord_test_xtb.csv")
    )
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path("/tmp/synde-energy-external-validation/training.joblib"),
    )
    parser.add_argument(
        "--training-cache-summary",
        type=Path,
        default=Path("Experiment/results/training_cache_record.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("synde/models/synde_frozen_model.json"),
    )
    parser.add_argument(
        "--expected-model-sha256",
        default=EXPECTED_MODEL_SHA256,
        help="Required SHA-256 of the model frozen before external evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Experiment/results/external_results.json"),
    )
    parser.add_argument(
        "--reaggregate-existing",
        action="store_true",
        help=(
            "Rebuild similarity strata in --output from its stored group metrics "
            "and metadata without rescoring the frozen model."
        ),
    )
    args = parser.parse_args()
    if args.reaggregate_existing:
        if not args.output.is_file():
            parser.error(f"missing stored external result: {args.output}")
        payload = _reaggregate_existing(
            json.loads(args.output.read_text(encoding="utf-8"))
        )
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "similarity_analysis": payload["similarity_analysis"],
                    "similarity_strata": {
                        key: value
                        for key, value in payload["strata"].items()
                        if "group_median_tanimoto" in key
                    },
                },
                indent=2,
            )
        )
        return
    for label, path in {
        "test labels": args.external_csv,
        "training cache": args.training_cache,
        "training cache summary": args.training_cache_summary,
        "frozen model": args.model,
    }.items():
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    payload = run(
        args.external_csv,
        args.training_cache,
        args.training_cache_summary,
        args.model,
        args.expected_model_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
