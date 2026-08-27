#!/usr/bin/env python3
"""Benchmark SynDE scoring against external-cohort GFN2-xTB target generation."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synde.energy import SynDEScorer  # noqa: E402
from synde.graph import GraphBuilder  # noqa: E402

PROTOCOL = "synde-target-generation-runtime-v1"
EXPECTED_GROUPS = 3_005
EXPECTED_MOLECULES = 19_940


def _summary(seconds: list[float], molecules: int) -> dict[str, object]:
    values = np.asarray(seconds, dtype=float)
    per_molecule = values / molecules
    return {
        "distribution": (
            "repeated warm selected-cohort SMILES parsing, graph featurization, "
            "and frozen connectivity scoring"
        ),
        "observations": len(values),
        "molecules_per_observation": molecules,
        "mean_seconds": float(np.mean(values)),
        "std_seconds_sample": float(np.std(values, ddof=1)),
        "mean_milliseconds_per_molecule": float(np.mean(per_molecule) * 1000.0),
        "std_milliseconds_per_molecule_sample": float(
            np.std(per_molecule, ddof=1) * 1000.0
        ),
        "mean_molecules_per_second": float(molecules / np.mean(values)),
    }


def _xtb_times(path: Path, connectivities: set[str]) -> dict[str, object]:
    elapsed: list[float] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("status") == "success"
                and row.get("connectivity") in connectivities
            ):
                elapsed.append(float(row["elapsed_seconds"]))
    if len(elapsed) != len(connectivities):
        raise RuntimeError(
            f"Expected {len(connectivities)} xTB timings, found {len(elapsed)}."
        )
    values = np.asarray(elapsed, dtype=float)
    return {
        "distribution": "individual molecule elapsed times",
        "molecules": len(values),
        "mean_seconds_per_molecule": float(np.mean(values)),
        "std_seconds_per_molecule_sample": float(np.std(values, ddof=1)),
        "median_seconds_per_molecule": float(np.median(values)),
        "p95_seconds_per_molecule": float(np.quantile(values, 0.95)),
        "mean_milliseconds_per_molecule": float(np.mean(values) * 1000.0),
        "std_milliseconds_per_molecule_sample": float(np.std(values, ddof=1) * 1000.0),
        "mean_molecules_per_second_per_worker": float(1.0 / np.mean(values)),
        "scope": "SMILES embedding, GFN2-xTB extreme optimization, and energy extraction",
    }


def _synde_times(
    groups: list[dict[str, object]], model_path: Path, repeats: int, molecules: int
) -> tuple[dict[str, object], float]:
    load_started = time.perf_counter()
    scorer = SynDEScorer.load(model_path)
    load_seconds = time.perf_counter() - load_started
    totals: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for group in groups:
            graphs = [
                GraphBuilder.from_smiles(str(row["smiles"]))
                for row in group["molecules"]
            ]
            scorer.score_group(graphs)
        totals.append(time.perf_counter() - started)
    return _summary(totals, molecules), load_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-cache",
        type=Path,
        default=Path("/tmp/synde-external-global-comparators-v1/external.joblib"),
    )
    parser.add_argument(
        "--external-csv", type=Path, default=Path("data/ord_test_xtb.csv")
    )
    parser.add_argument(
        "--synde-model",
        type=Path,
        default=Path("synde/models/synde_frozen_model.json"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        help=(
            "Optional seed-recovery audit. When supplied, benchmark only "
            "freshly calculated successful external connectivities with "
            "nonzero xTB elapsed times."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Experiment/results/inference_runtime_benchmark.json"),
    )
    args = parser.parse_args()

    if args.repeats < 2:
        parser.error("--repeats must be at least two")
    required = {
        "test cache": args.external_cache,
        "test labels": args.external_csv,
        "frozen model": args.synde_model,
    }
    if args.subset_manifest is not None:
        required["subset manifest"] = args.subset_manifest
    for label, path in required.items():
        if not path.is_file():
            parser.error(f"missing {label}: {path}")

    groups = joblib.load(args.external_cache)
    selection = "complete external cohort"
    if args.subset_manifest is not None:
        manifest = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
        requested = {
            str(row["connectivity"])
            for row in manifest["fresh_calculation_records"]
            if row.get("split") == "external"
        }
        successful = set()
        with args.external_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("connectivity") in requested
                    and row.get("status") == "success"
                    and float(row.get("elapsed_seconds") or 0.0) > 0.0
                ):
                    successful.add(str(row["connectivity"]))
        groups = [
            {
                "key": group["key"],
                "molecules": [
                    row
                    for row in group["molecules"]
                    if str(row["connectivity"]) in successful
                ],
            }
            for group in groups
        ]
        groups = [group for group in groups if group["molecules"]]
        selection = (
            "freshly recalculated successful external recovery subset with "
            "nonzero xTB elapsed times"
        )
    molecules = sum(len(group["molecules"]) for group in groups)
    if args.subset_manifest is None and (
        len(groups) != EXPECTED_GROUPS or molecules != EXPECTED_MOLECULES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_MOLECULES} molecules/{EXPECTED_GROUPS} groups, "
            f"found {molecules}/{len(groups)}."
        )
    connectivities = {
        str(row["connectivity"]) for group in groups for row in group["molecules"]
    }
    if len(connectivities) != molecules:
        raise RuntimeError("External benchmark connectivities are not unique.")
    synde, load_seconds = _synde_times(
        groups, args.synde_model, args.repeats, molecules
    )
    payload = {
        "protocol": PROTOCOL,
        "fairness_scope": (
            "Both methods use the same selected external molecules and cover SMILES input "
            "through final energy output; model fitting is excluded."
        ),
        "cohort": {
            "groups": len(groups),
            "molecules": molecules,
            "selection": selection,
            "subset_manifest": (
                str(args.subset_manifest) if args.subset_manifest is not None else None
            ),
        },
        "repeats": args.repeats,
        "hardware": {
            "platform": platform.platform(),
            "cpu": "AMD Ryzen Threadripper 7970X (32 cores, 64 threads)",
            "ram_gib": 125,
        },
        "software": {
            "python": sys.version.split()[0],
            "rdkit": importlib.metadata.version("rdkit"),
            "numpy": np.__version__,
            "xtb": "6.7.1",
        },
        "xtb_gfn2_extreme": _xtb_times(args.external_csv, connectivities),
        "synde": synde,
        "excluded_setup_seconds": {"synde_model_load": load_seconds},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
