#!/usr/bin/env python3
"""Create label-blind, formula-disjoint ORD manifests before running xTB.

The input is ``data/ord.csv`` from ``01_prepare_ord.py``. Complete molecular
formula groups are assigned deterministically to training, validation, or
external partitions before quantum labels exist. Within each group, a greedy
ECFP4 max-min rule retains at most ten structurally diverse connectivities.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tempfile

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
NAMESPACE = "synde-ord-ff7427-formula-split-v2"
MIN_GROUP_SIZE = 3
DEFAULT_MAX_PER_GROUP = 10


def recorded_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: str) -> str:
    """Return a namespaced deterministic hexadecimal key."""
    return hashlib.sha256(": ".join((NAMESPACE, *parts)).encode()).hexdigest()


def hash_fraction(*parts: str) -> float:
    """Map a stable key to the half-open interval [0, 1)."""
    return int(stable_hash(*parts)[:16], 16) / float(2**64)


def assigned_split(
    formula: str, validation_fraction: float, external_fraction: float
) -> str:
    """Assign one complete formula group without consulting an energy label."""
    value = hash_fraction("formula", formula)
    if value < external_fraction:
        return "external"
    if value < external_fraction + validation_fraction:
        return "validation"
    return "training"


def diverse_rows(rows: list[dict[str, str]], maximum: int) -> list[dict[str, str]]:
    """Select a deterministic ECFP4 max-min subset from one formula group."""
    ordered = sorted(rows, key=lambda row: stable_hash("molecule", row["connectivity"]))
    if len(ordered) <= maximum:
        return ordered
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for row in ordered:
        molecule = Chem.MolFromSmiles(row["SMILES"])
        if molecule is None:
            raise ValueError(f"Canonical SMILES no longer parses: {row['SMILES']}")
        fingerprints.append(generator.GetFingerprint(molecule))

    selected = [0]
    remaining = set(range(1, len(ordered)))
    minimum_distances = {index: 1.0 for index in remaining}
    while len(selected) < maximum:
        latest = selected[-1]
        indices = sorted(remaining)
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[latest], [fingerprints[index] for index in indices]
        )
        for index, similarity in zip(indices, similarities):
            minimum_distances[index] = min(
                minimum_distances[index], 1.0 - float(similarity)
            )
        chosen = min(
            remaining,
            key=lambda index: (
                -minimum_distances[index],
                stable_hash("molecule", ordered[index]["connectivity"]),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return [ordered[index] for index in selected]


def atomic_write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    """Write a CSV atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_groups(path: Path) -> tuple[dict[str, list[dict[str, str]]], Counter[str]]:
    """Load eligible neutral rows grouped by molecular formula."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    audit: Counter[str] = Counter()
    seen_connectivities: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "SMILES",
            "connectivity",
            "formula",
            "formal_charge",
            "paper_eligible",
            "exclusion_reason",
            "heavy_atoms",
            "elements",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            audit["source_rows"] += 1
            if row["paper_eligible"] != "1":
                audit["excluded_not_paper_eligible"] += 1
                for reason in filter(None, row["exclusion_reason"].split(";")):
                    audit[f"excluded_{reason}"] += 1
                continue
            if int(row["formal_charge"]) != 0:
                raise ValueError("paper_eligible row has nonzero formal charge")
            connectivity = row["connectivity"]
            if connectivity in seen_connectivities:
                raise ValueError(f"Duplicate eligible connectivity: {connectivity}")
            seen_connectivities.add(connectivity)
            groups[row["formula"]].append(row)
            audit["eligible_rows"] += 1
    return dict(groups), audit


def scaffold(smiles: str) -> str:
    """Return an achiral Bemis-Murcko scaffold, or an empty string."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)


def scaffold_component_assignments(
    groups: dict[str, list[dict[str, str]]],
    validation_fraction: float,
    external_fraction: float,
) -> tuple[dict[str, str], dict[str, object]]:
    """Assign components of formula groups linked by a nonempty scaffold."""
    parent = {formula: formula for formula in groups}

    def find(formula: str) -> str:
        while parent[formula] != formula:
            parent[formula] = parent[parent[formula]]
            formula = parent[formula]
        return formula

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if stable_hash("union", left_root) <= stable_hash("union", right_root):
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    scaffold_owner: dict[str, str] = {}
    formula_scaffolds: dict[str, set[str]] = {}
    for formula, rows in groups.items():
        values = {scaffold(row["SMILES"]) for row in rows}
        values.discard("")
        formula_scaffolds[formula] = values
        for value in values:
            owner = scaffold_owner.setdefault(value, formula)
            union(formula, owner)

    components: dict[str, list[str]] = defaultdict(list)
    for formula in groups:
        components[find(formula)].append(formula)
    units = sorted(
        (sorted(formulas) for formulas in components.values()),
        key=lambda formulas: (
            -len(formulas),
            stable_hash("scaffold-component", *formulas),
        ),
    )
    fractions = {
        "training": 1.0 - validation_fraction - external_fraction,
        "validation": validation_fraction,
        "external": external_fraction,
    }
    active_splits = [name for name, fraction in fractions.items() if fraction > 0]
    targets = {name: fraction * len(groups) for name, fraction in fractions.items()}
    assigned_counts = Counter()
    assignments: dict[str, str] = {}
    for formulas in units:
        size = len(formulas)

        def assignment_cost(split: str) -> tuple[float, str]:
            cost = sum(
                (assigned_counts[name] + (size if name == split else 0) - targets[name])
                ** 2
                for name in fractions
            )
            return cost, stable_hash("component-split-tie", *formulas, split)

        chosen = min(active_splits, key=assignment_cost)
        assigned_counts[chosen] += size
        for formula in formulas:
            assignments[formula] = chosen

    return assignments, {
        "components": len(units),
        "largest_component_formula_groups": max(map(len, units), default=0),
        "nonempty_unique_scaffolds": len(scaffold_owner),
        "acyclic_formula_groups": sum(
            not values for values in formula_scaffolds.values()
        ),
    }


def select(
    source: Path,
    training_output: Path,
    validation_output: Path,
    external_output: Path,
    audit_output: Path,
    *,
    validation_fraction: float,
    external_fraction: float,
    maximum_per_group: int,
    maximum_groups: int | None,
    maximum_heavy_atoms: int | None,
    scaffold_disjoint: bool,
) -> None:
    """Create immutable, label-blind candidate manifests."""
    if not 0 <= validation_fraction < 1 or not 0 <= external_fraction < 1:
        raise ValueError("Split fractions must lie in [0, 1)")
    if validation_fraction + external_fraction >= 1:
        raise ValueError("Validation plus external fractions must be below 1")
    if maximum_per_group < MIN_GROUP_SIZE:
        raise ValueError(f"maximum_per_group must be at least {MIN_GROUP_SIZE}")
    if maximum_heavy_atoms is not None and maximum_heavy_atoms < 1:
        raise ValueError("maximum_heavy_atoms must be positive")

    groups, source_audit = load_groups(source)
    rankable = {
        formula: rows
        for formula, rows in groups.items()
        if len(rows) >= MIN_GROUP_SIZE
        and (
            maximum_heavy_atoms is None
            or max(int(row["heavy_atoms"]) for row in rows) <= maximum_heavy_atoms
        )
    }
    excluded_small = sum(
        len(rows) for rows in groups.values() if len(rows) < MIN_GROUP_SIZE
    )
    excluded_large_groups = {
        formula: rows
        for formula, rows in groups.items()
        if len(rows) >= MIN_GROUP_SIZE
        and maximum_heavy_atoms is not None
        and max(int(row["heavy_atoms"]) for row in rows) > maximum_heavy_atoms
    }
    formulas = sorted(
        rankable,
        key=lambda formula: stable_hash("group-order", formula),
    )
    if maximum_groups is not None:
        if maximum_groups < 1:
            raise ValueError("maximum_groups must be positive")
        formulas = formulas[:maximum_groups]

    selected_groups = {formula: rankable[formula] for formula in formulas}
    scaffold_split_audit: dict[str, object] = {
        "enabled": scaffold_disjoint,
        "policy": (
            "formula groups sharing any nonempty achiral Bemis-Murcko scaffold "
            "are assigned as one connected component; empty scaffolds are ignored"
            if scaffold_disjoint
            else "not enforced; nonempty scaffold overlap is reported diagnostically"
        ),
    }
    if scaffold_disjoint:
        formula_assignments, component_audit = scaffold_component_assignments(
            selected_groups, validation_fraction, external_fraction
        )
        scaffold_split_audit.update(component_audit)
    else:
        formula_assignments = {
            formula: assigned_split(formula, validation_fraction, external_fraction)
            for formula in formulas
        }

    outputs: dict[str, list[dict[str, object]]] = {
        "training": [],
        "validation": [],
        "external": [],
    }
    formula_sets: dict[str, set[str]] = {name: set() for name in outputs}
    connectivity_sets: dict[str, set[str]] = {name: set() for name in outputs}
    scaffold_sets: dict[str, set[str]] = {name: set() for name in outputs}
    available_sizes = Counter()
    selected_sizes = Counter()

    for formula in formulas:
        available = rankable[formula]
        split = formula_assignments[formula]
        retained = diverse_rows(available, maximum_per_group)
        formula_sets[split].add(formula)
        available_sizes[len(available)] += 1
        selected_sizes[len(retained)] += 1
        for selection_rank, row in enumerate(retained, start=1):
            connectivity_sets[split].add(row["connectivity"])
            molecular_scaffold = scaffold(row["SMILES"])
            if molecular_scaffold:
                scaffold_sets[split].add(molecular_scaffold)
            outputs[split].append(
                {
                    "split": split,
                    "group_id": f"{formula}|charge=0",
                    "formula": formula,
                    "group_size_available": len(available),
                    "selection_rank": selection_rank,
                    "SMILES": row["SMILES"],
                    "connectivity": row["connectivity"],
                    "heavy_atoms": int(row["heavy_atoms"]),
                    "elements": row["elements"],
                    "selection_status": "selected_before_xtb",
                }
            )

    split_names = tuple(outputs)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if formula_sets[left] & formula_sets[right]:
                raise RuntimeError(f"Formula leakage between {left} and {right}")
            if connectivity_sets[left] & connectivity_sets[right]:
                raise RuntimeError(f"Connectivity leakage between {left} and {right}")
            shared_scaffolds = scaffold_sets[left] & scaffold_sets[right]
            if scaffold_disjoint and shared_scaffolds:
                raise RuntimeError(f"Scaffold leakage between {left} and {right}")

    fields = [
        "split",
        "group_id",
        "formula",
        "group_size_available",
        "selection_rank",
        "SMILES",
        "connectivity",
        "heavy_atoms",
        "elements",
        "selection_status",
    ]
    paths = {
        "training": training_output,
        "validation": validation_output,
        "external": external_output,
    }
    for split, path in paths.items():
        atomic_write_csv(path, fields, outputs[split])

    training_scaffolds = scaffold_sets["training"]
    external_rows = outputs["external"]
    external_scaffolds = [scaffold(row["SMILES"]) for row in external_rows]
    external_unseen = sum(
        bool(value) and value not in training_scaffolds for value in external_scaffolds
    )
    payload = {
        "protocol": NAMESPACE,
        "selection_is_label_blind": True,
        "source": recorded_path(source),
        "source_sha256": sha256_file(source),
        "rdkit_version": rdBase.rdkitVersion,
        "rules": {
            "group_key": "molecular formula plus formal charge (charge fixed at zero)",
            "minimum_group_size": MIN_GROUP_SIZE,
            "maximum_molecules_per_group": maximum_per_group,
            "maximum_heavy_atoms": maximum_heavy_atoms,
            "within_group_selection": (
                "deterministic greedy max-min ECFP4 (Morgan radius 2, 2048 bits)"
            ),
            "validation_fraction": validation_fraction,
            "external_fraction": external_fraction,
            "training_fraction": 1.0 - validation_fraction - external_fraction,
            "maximum_groups": maximum_groups,
            "split_assignment": (
                "greedy size-balanced assignment of scaffold-linked formula components"
                if scaffold_disjoint
                else "SHA-256 of the complete formula group"
            ),
            "scaffold_disjoint": scaffold_disjoint,
        },
        "source_counts": {
            **dict(sorted(source_audit.items())),
            "excluded_eligible_molecules_in_groups_smaller_than_3": excluded_small,
            "excluded_formula_groups_above_heavy_atom_limit": len(
                excluded_large_groups
            ),
            "excluded_molecules_above_heavy_atom_limit": sum(
                len(rows) for rows in excluded_large_groups.values()
            ),
            "rankable_formula_groups_before_budget_cap": len(rankable),
            "selected_formula_groups": len(formulas),
        },
        "splits": {
            split: {
                "formula_groups": len(formula_sets[split]),
                "molecules": len(outputs[split]),
                "unique_connectivities": len(connectivity_sets[split]),
                "unique_murcko_scaffolds": len(scaffold_sets[split]),
                "output": recorded_path(paths[split]),
                "output_sha256": sha256_file(paths[split]),
            }
            for split in outputs
        },
        "external_diagnostics": {
            "molecules_with_scaffold_unseen_in_training": external_unseen,
            "fraction_with_scaffold_unseen_in_training": (
                external_unseen / len(external_rows) if external_rows else None
            ),
        },
        "scaffold_split": scaffold_split_audit,
        "available_group_size_distribution": {
            str(size): count for size, count in sorted(available_sizes.items())
        },
        "selected_group_size_distribution": {
            str(size): count for size, count in sorted(selected_sizes.items())
        },
        "leakage_checks": {
            "formula_intersections": 0,
            "connectivity_intersections": 0,
            "nonempty_scaffold_intersections": {
                f"{left}:{right}": len(scaffold_sets[left] & scaffold_sets[right])
                for left_index, left in enumerate(split_names)
                for right in split_names[left_index + 1 :]
            },
        },
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "Selected "
        + ", ".join(
            f"{split}={len(formula_sets[split]):,} groups/"
            f"{len(outputs[split]):,} molecules"
            for split in outputs
        )
    )


def main() -> None:
    """Parse command-line arguments and select cohorts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATA_DIR / "ord.csv")
    parser.add_argument(
        "--training-output", type=Path, default=DATA_DIR / "ord_training_candidates.csv"
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DATA_DIR / "ord_validation_candidates.csv",
    )
    parser.add_argument(
        "--external-output", type=Path, default=DATA_DIR / "ord_external_candidates.csv"
    )
    parser.add_argument(
        "--audit", type=Path, default=DATA_DIR / "ord_cohort_selection.audit.json"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--external-fraction", type=float, default=0.20)
    parser.add_argument("--maximum-per-group", type=int, default=DEFAULT_MAX_PER_GROUP)
    parser.add_argument(
        "--maximum-heavy-atoms",
        type=int,
        default=None,
        help="Optional pre-label molecular-size ceiling used to control xTB cost.",
    )
    parser.add_argument(
        "--maximum-groups",
        type=int,
        default=None,
        help="Optional xTB budget cap applied deterministically before splitting.",
    )
    parser.add_argument(
        "--scaffold-disjoint",
        action="store_true",
        help=(
            "Keep formula groups sharing a nonempty Bemis-Murcko scaffold in "
            "the same partition. Empty scaffolds are handled by distance diagnostics."
        ),
    )
    args = parser.parse_args()
    select(
        args.source.resolve(),
        args.training_output.resolve(),
        args.validation_output.resolve(),
        args.external_output.resolve(),
        args.audit.resolve(),
        validation_fraction=args.validation_fraction,
        external_fraction=args.external_fraction,
        maximum_per_group=args.maximum_per_group,
        maximum_groups=args.maximum_groups,
        maximum_heavy_atoms=args.maximum_heavy_atoms,
        scaffold_disjoint=args.scaffold_disjoint,
    )


if __name__ == "__main__":
    main()
