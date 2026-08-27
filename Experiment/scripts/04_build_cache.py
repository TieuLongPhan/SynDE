#!/usr/bin/env python3
"""Build the fixed-feature SynDE training cache."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.scripts._helpers import (  # noqa: E402
    _compact,
    _load_candidates,
    load_calibration_seed,
    public_record,
)
from synde.energy import (  # noqa: E402
    FirstOrderTwoDEnergyScorer,
    V3_FEATURE_NAMES as FROZEN_FEATURE_NAMES,
    extract_named_empirical_two_d_features,
    extract_quantum_graph_v3_features as extract_frozen_graph_features,
)
from synde.graph import GraphBuilder  # noqa: E402
from synde.chem import SUPPORTED_ELEMENT_SET  # noqa: E402

PROTOCOL = "synde-ord-v5-training-cache-v1"
SAMPLE_NAMESPACE = "synde-ord-v4-calibration-sample-v1"
MAX_MOLECULES_PER_GROUP = 10
SUPPORTED_ELEMENTS = set(SUPPORTED_ELEMENT_SET)
EXPECTED_FEATURE_DEFINITION_SHA256 = (
    "a1749aea7a455d57677e0165224747f9aa674800411bd585f6a5b6f90d892a24"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_group(key: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SAMPLE_NAMESPACE}:{key}:{row['connectivity']}".encode()
        ).hexdigest(),
    )[:MAX_MOLECULES_PER_GROUP]


def _extract_group(
    item: tuple[str, list[dict[str, object]]],
) -> tuple[dict[str, object] | None, list[dict[str, str]], Counter[str]]:
    key, rows = item
    scorer = FirstOrderTwoDEnergyScorer()
    molecules = []
    failures = []
    nonzero: Counter[str] = Counter()
    for row in _sample_group(key, rows):
        smiles = str(row["smiles"])
        try:
            graph = GraphBuilder.from_smiles(smiles)
            first_order = scorer.score(graph)
            if first_order.score is None:
                raise RuntimeError("first-order feature extraction returned no score")
            features = _compact(
                extract_named_empirical_two_d_features(graph, first_order)
            )
            frozen_features = extract_frozen_graph_features(graph)
            if tuple(frozen_features) != FROZEN_FEATURE_NAMES:
                raise RuntimeError("Feature row differs from frozen manifest")
            features.update(frozen_features)
        except Exception as exc:  # noqa: BLE001 - every failure is serialized.
            failures.append(
                {"key": key, "smiles": smiles, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        nonzero.update(name for name, value in features.items() if value)
        molecules.append(
            {
                "smiles": smiles,
                "connectivity": str(row["connectivity"]),
                "label": float(row["label"]),
                "features": features,
                "scaffold": str(row["scaffold"]),
                "elements": list(row["elements"]),
            }
        )
    group = (
        {"key": key, "molecules": molecules}
        if len(molecules) >= 3 and len({row["label"] for row in molecules}) >= 2
        else None
    )
    return group, failures, nonzero


def build(
    source: Path,
    calibration_seed: Path,
    output_cache: Path,
    workers: int,
) -> dict[str, object]:
    started = time.perf_counter()
    manifest, frozen = load_calibration_seed(calibration_seed)
    if manifest["feature_definition_sha256"] != EXPECTED_FEATURE_DEFINITION_SHA256:
        raise RuntimeError("Unexpected frozen feature-definition hash.")
    frozen_support = set(frozen["selected_terms"])

    source_groups, source_audit = _load_candidates(
        source, supported_elements=SUPPORTED_ELEMENTS
    )
    eligible = {
        key: rows
        for key, rows in source_groups.items()
        if len(rows) >= 3 and len({float(row["label"]) for row in rows}) >= 2
    }

    groups: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    feature_nonzero = Counter()
    with multiprocessing.Pool(processes=workers) as pool:
        extracted = pool.imap(_extract_group, sorted(eligible.items()), chunksize=8)
        for position, (group, group_failures, nonzero) in enumerate(extracted, start=1):
            if group is not None:
                groups.append(group)
            failures.extend(group_failures)
            feature_nonzero.update(nonzero)
            if position % 250 == 0:
                print(
                    f"extracted {position}/{len(eligible)} SynDE groups",
                    file=sys.stderr,
                    flush=True,
                )

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(groups, output_cache, compress=3)
    molecule_count = sum(len(group["molecules"]) for group in groups)
    feature_names = sorted(
        {
            name
            for group in groups
            for row in group["molecules"]
            for name in row["features"]
        }
    )
    elapsed = time.perf_counter() - started
    return {
        "protocol": PROTOCOL,
        "protocol_document": "Experiment/README.md",
        "development_only": True,
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_audit": source_audit,
        "eligible_groups_before_sampling": len(eligible),
        "eligible_molecules_before_sampling": sum(
            len(rows) for rows in eligible.values()
        ),
        "groups": len(groups),
        "molecules": molecule_count,
        "sample_namespace": SAMPLE_NAMESPACE,
        "maximum_molecules_per_group": MAX_MOLECULES_PER_GROUP,
        "calibration_seed": str(calibration_seed),
        "calibration_seed_sha256": _sha256(calibration_seed),
        "feature_definition_sha256": EXPECTED_FEATURE_DEFINITION_SHA256,
        "feature_count_observed": len(feature_names),
        "seed_feature_count": len(FROZEN_FEATURE_NAMES),
        "seed_support_count": len(frozen_support),
        "seed_support_sha256": hashlib.sha256(
            "\n".join(sorted(frozen_support)).encode()
        ).hexdigest(),
        "feature_extraction_failures": failures,
        "nonzero_molecule_counts": {
            name: int(feature_nonzero[name]) for name in feature_names
        },
        "output_cache": str(output_cache),
        "output_cache_sha256": _sha256(output_cache),
        "elapsed_seconds": elapsed,
        "mean_seconds_per_molecule": elapsed / molecule_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/ord_training_xtb.csv")
    )
    parser.add_argument(
        "--calibration-seed",
        type=Path,
        default=Path("Experiment/calibration_seed.json"),
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=Path("/tmp/synde-energy-external-validation/training.joblib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Experiment/results/training_cache_record.json"),
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    for label, path in {
        "training labels": args.source,
        "calibration seed": args.calibration_seed,
    }.items():
        if not path.is_file():
            parser.error(f"missing {label}: {path}")
    payload = public_record(
        build(
            args.source,
            args.calibration_seed,
            args.output_cache,
            args.workers,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
