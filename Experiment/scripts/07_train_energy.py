#!/usr/bin/env python3
"""Train and evaluate the SynDE energy-and-ranking artifact.

This entry point rebuilds the active training feature cache, fits the elemental
calibration around the frozen connectivity equation, and evaluates the
complete model on the held-out cohort. With
``--publish``, the verified model and result record replace their canonical
repository files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _input_path(path: Path) -> Path:
    """Resolve an input path from the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require_inputs(paths: dict[str, Path]) -> dict[str, Path]:
    """Resolve required files and fail before creating run outputs."""
    resolved = {name: _input_path(path) for name, path in paths.items()}
    missing = {name: path for name, path in resolved.items() if not path.is_file()}
    if missing:
        details = "\n".join(f"- {name}: {path}" for name, path in missing.items())
        hint = ""
        if {"training labels", "test labels"} & set(missing):
            hint = (
                "\nGenerate the missing xTB labels with "
                "`python Experiment/scripts/03_run_xtb.py`."
            )
        raise FileNotFoundError(f"Missing required experiment inputs:\n{details}{hint}")
    return resolved


def _run(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SynDE] running: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(f"Stage failed; see {log_path}\n{tail}")


def _verify(
    connectivity_model: Path,
    energy_model: Path,
    result_path: Path,
) -> dict[str, object]:
    source = json.loads(connectivity_model.read_text(encoding="utf-8"))
    model = json.loads(energy_model.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    source_weights = source.get("weights")
    energy_weights = model.get("connectivity_weights")
    if not isinstance(source_weights, dict) or not isinstance(energy_weights, dict):
        raise RuntimeError("Missing connectivity coefficients in a model artifact.")
    if not energy_weights:
        raise RuntimeError("The fitted connectivity equation is empty.")
    if source_weights != energy_weights:
        raise RuntimeError(
            "Energy calibration changed the frozen connectivity weights."
        )
    reported_terms = result.get("energy_model", {}).get("selected_connectivity_terms")
    if reported_terms != len(energy_weights):
        raise RuntimeError("Benchmark record has the wrong connectivity-term count.")
    return {
        "model_name": model["card"]["model_name"],
        "connectivity_terms": len(energy_weights),
        "training_groups": result["split"]["training_groups"],
        "evaluation_groups": result["split"]["evaluation_groups"],
        "evaluation_molecules": result["split"]["evaluation_molecules"],
        "global_metrics": result["cross_formula_metrics"][
            "composition_plus_synde_connectivity"
        ],
        "local_metrics": result["new_model_same_formula_ranking"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-source",
        type=Path,
        default=Path("data/ord_training_xtb.csv"),
    )
    parser.add_argument(
        "--external-source",
        type=Path,
        default=Path("data/ord_test_xtb.csv"),
    )
    parser.add_argument(
        "--connectivity-model",
        type=Path,
        default=Path("synde/models/synde_frozen_model.json"),
    )
    parser.add_argument(
        "--calibration-seed",
        type=Path,
        default=Path("Experiment/calibration_seed.json"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/synde-energy-external-validation"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument(
        "--rebuild-training-cache",
        action="store_true",
        help="Discard a reusable training feature cache in the work directory.",
    )
    parser.add_argument(
        "--rebuild-external-cache",
        action="store_true",
        help="Discard a reusable external feature cache in the work directory.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Install verified generated artifacts into their canonical repository paths.",
    )
    args = parser.parse_args()

    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 grouped bootstrap replicates.")
    inputs = _require_inputs(
        {
            "training labels": args.training_source,
            "test labels": args.external_source,
            "connectivity model": args.connectivity_model,
            "calibration seed": args.calibration_seed,
        }
    )
    args.training_source = inputs["training labels"]
    args.external_source = inputs["test labels"]
    args.connectivity_model = inputs["connectivity model"]
    args.calibration_seed = inputs["calibration seed"]
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    training_cache = work_dir / "training.joblib"
    training_record = work_dir / "training_cache_record.json"
    external_cache = work_dir / "external.joblib"
    energy_model = work_dir / "synde_energy_model.json"
    benchmark_result = work_dir / "synde.json"

    if args.rebuild_training_cache or not (
        training_cache.is_file() and training_record.is_file()
    ):
        _run(
            [
                sys.executable,
                "Experiment/scripts/04_build_cache.py",
                "--source",
                str(args.training_source),
                "--calibration-seed",
                str(args.calibration_seed),
                "--output-cache",
                str(training_cache),
                "--output",
                str(training_record),
            ],
            work_dir / "build_training_cache.log",
        )
    else:
        print(f"[SynDE] reusing {training_cache}", flush=True)

    benchmark_command = [
        sys.executable,
        "Experiment/scripts/_fit_energy.py",
        "--training-cache",
        str(training_cache),
        "--training-cache-summary",
        str(training_record),
        "--formula-relative-model",
        str(args.connectivity_model),
        "--external-csv",
        str(args.external_source),
        "--external-cache",
        str(external_cache),
        "--model-output",
        str(energy_model),
        "--output",
        str(benchmark_result),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
    ]
    if args.rebuild_external_cache or not external_cache.is_file():
        benchmark_command.append("--rebuild-external-cache")
    _run(benchmark_command, work_dir / "benchmark.log")

    summary = _verify(args.connectivity_model, energy_model, benchmark_result)
    if args.publish:
        canonical_model = PROJECT_ROOT / "synde/models/synde_energy_model.json"
        canonical_result = (
            PROJECT_ROOT / "Experiment/results/global_comparators/synde.json"
        )
        shutil.copy2(energy_model, canonical_model)
        shutil.copy2(benchmark_result, canonical_result)
        summary["published"] = True
    else:
        summary["published"] = False
    summary["work_dir"] = str(work_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
