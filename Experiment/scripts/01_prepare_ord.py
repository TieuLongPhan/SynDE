#!/usr/bin/env python3
"""Fetch, extract, and canonicalize a pinned Open Reaction Database snapshot.

The pipeline has three explicit stages:

1. clone the official ``ord-data`` repository at ``ORD_COMMIT`` and fetch the
   pinned Git-LFS protobuf objects;
2. extract unique raw molecular strings to ``data/ord_raw.csv``; and
3. split disconnected records with RDKit, canonicalize every component, and
   write ``data/ord.csv`` plus a formula-group inventory.

No xTB calculation is launched by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
ORD_REPOSITORY = "https://github.com/open-reaction-database/ord-data.git"
ORD_COMMIT = "ff7427ff6e65e9c5cf25caeccb9ed6407179d3d6"
EXPECTED_PROTOBUF_FILES = 550
from synde.chem import (  # noqa: E402
    ISOTOPE_EXCLUSION_REASON,
    SUPPORTED_ELEMENT_SET,
    has_isotopically_labelled_atom,
    normalize_ordinary_explicit_hydrogens,
)

SUPPORTED_ELEMENTS = SUPPORTED_ELEMENT_SET
RAW_EXTRACTION_PROTOCOL = "synde-ord-official-snapshot-raw-v3"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run(
    command: list[str], *, cwd: Path, environment: dict[str, str] | None = None
) -> None:
    """Run a checked subprocess and echo the reproducible command."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def git_output(snapshot_dir: Path, *arguments: str) -> str:
    """Return stripped output from a Git command in the ORD checkout."""
    return subprocess.check_output(
        ["git", "-C", str(snapshot_dir), *arguments], text=True
    ).strip()


def fetch_snapshot(snapshot_dir: Path) -> None:
    """Clone and materialize the pinned official ORD protobuf snapshot."""
    if shutil.which("git") is None:
        raise RuntimeError("git is required")
    try:
        subprocess.run(
            ["git", "lfs", "version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "git-lfs is required; update the environment from env.yml first"
        ) from error

    environment = dict(os.environ)
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    if not snapshot_dir.exists():
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                ORD_REPOSITORY,
                str(snapshot_dir),
            ],
            cwd=PROJECT_ROOT,
            environment=environment,
        )
    elif not (snapshot_dir / ".git").is_dir():
        raise RuntimeError(f"Refusing to overwrite non-Git path: {snapshot_dir}")

    origin = git_output(snapshot_dir, "remote", "get-url", "origin")
    if origin.rstrip("/") not in {
        ORD_REPOSITORY.rstrip("/"),
        "git@github.com:open-reaction-database/ord-data.git",
    }:
        raise RuntimeError(f"Unexpected ORD origin: {origin}")
    run(
        ["git", "-C", str(snapshot_dir), "fetch", "origin", ORD_COMMIT],
        cwd=PROJECT_ROOT,
        environment=environment,
    )
    run(
        ["git", "-C", str(snapshot_dir), "checkout", "--detach", ORD_COMMIT],
        cwd=PROJECT_ROOT,
        environment=environment,
    )
    run(
        ["git", "-C", str(snapshot_dir), "lfs", "install", "--local"],
        cwd=PROJECT_ROOT,
        environment=dict(os.environ),
    )
    run(
        [
            "git",
            "-C",
            str(snapshot_dir),
            "lfs",
            "pull",
            "--include=data/**/*.pb.gz",
            "--exclude=data/**/*.parquet",
        ],
        cwd=PROJECT_ROOT,
        environment=dict(os.environ),
    )

    head = git_output(snapshot_dir, "rev-parse", "HEAD")
    if head != ORD_COMMIT:
        raise RuntimeError(
            f"ORD checkout mismatch: expected {ORD_COMMIT}, found {head}"
        )
    paths = sorted(snapshot_dir.glob("data/*/ord_dataset-*.pb.gz"))
    if len(paths) != EXPECTED_PROTOBUF_FILES:
        raise RuntimeError(
            f"Pinned snapshot should contain {EXPECTED_PROTOBUF_FILES} protobuf files; "
            f"found {len(paths)}"
        )
    pointers = []
    for path in paths:
        with path.open("rb") as handle:
            if handle.read(42) == b"version https://git-lfs.github.com/spec/v1":
                pointers.append(path)
    if pointers:
        raise RuntimeError(f"{len(pointers)} ORD files are still Git-LFS pointers")
    print(f"Pinned ORD snapshot ready: {head} ({len(paths)} protobuf files)")


def reaction_smiles_components(value: str) -> Iterator[tuple[str, str]]:
    """Yield components and side labels from an ORD reaction SMILES string."""
    # ORD reaction identifiers may use reaction CXSMILES. Its annotation starts
    # after whitespace and can contain dots (for example ``f:0.1``); those dots
    # are metadata, not molecular-component separators.
    core = value.split(maxsplit=1)[0]
    sides = core.split(">")
    if len(sides) != 3:
        return
    for side_name, side in zip(("reactant", "reagent", "product"), sides):
        for component in side.split("."):
            component = component.strip()
            if component:
                yield component, f"reaction_smiles_{side_name}"


def initialize_raw_database(database: Path) -> sqlite3.Connection:
    """Open the resumable raw-string database and validate its provenance."""
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute("""
        CREATE TABLE IF NOT EXISTS raw (
            smiles TEXT PRIMARY KEY,
            occurrences INTEGER NOT NULL,
            example_dataset_id TEXT NOT NULL,
            example_reaction_id TEXT NOT NULL,
            example_location TEXT NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS processed (
            source_path TEXT PRIMARY KEY,
            source_sha256 TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            counts_json TEXT NOT NULL
        )
        """)
    stored = connection.execute(
        "SELECT value FROM metadata WHERE key = 'ord_commit'"
    ).fetchone()
    if stored is not None and stored[0] != ORD_COMMIT:
        connection.close()
        raise RuntimeError(
            f"Extraction database belongs to ORD commit {stored[0]}, not {ORD_COMMIT}"
        )
    connection.execute(
        "INSERT OR IGNORE INTO metadata VALUES ('ord_commit', ?)", (ORD_COMMIT,)
    )
    stored_protocol = connection.execute(
        "SELECT value FROM metadata WHERE key = 'raw_extraction_protocol'"
    ).fetchone()
    if stored_protocol is not None and stored_protocol[0] != RAW_EXTRACTION_PROTOCOL:
        connection.close()
        raise RuntimeError(
            "Extraction database uses an incompatible raw-extraction protocol: "
            f"{stored_protocol[0]}"
        )
    if stored_protocol is None:
        existing_rows = int(
            connection.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
        )
        if existing_rows:
            connection.close()
            raise RuntimeError(
                "Extraction database predates the CXSMILES fix; use a new "
                "--work-database path"
            )
        connection.execute(
            "INSERT INTO metadata VALUES ('raw_extraction_protocol', ?)",
            (RAW_EXTRACTION_PROTOCOL,),
        )
    connection.commit()
    return connection


def extract_one(source_file: Path, work_database: Path) -> None:
    """Extract one ORD dataset in an isolated, restart-safe subprocess."""
    try:
        from ord_schema import message_helpers
        from ord_schema.proto import dataset_pb2
    except ImportError as error:
        raise RuntimeError(
            "ord_schema is required; update the environment from env.yml first"
        ) from error

    source_file = source_file.resolve()
    connection = initialize_raw_database(work_database)
    source_key = str(source_file)
    if connection.execute(
        "SELECT 1 FROM processed WHERE source_path = ?", (source_key,)
    ).fetchone():
        connection.close()
        return

    audit = Counter()
    try:
        dataset = message_helpers.load_message(str(source_file), dataset_pb2.Dataset)
        dataset_id = str(dataset.dataset_id or source_file.stem)
        connection.execute("BEGIN IMMEDIATE")

        def observe(
            smiles: str | None,
            reaction_id: str,
            location: str,
        ) -> None:
            audit["molecular_string_occurrences"] += 1
            value = (smiles or "").strip()
            if not value:
                audit["excluded_empty_strings"] += 1
                return
            connection.execute(
                """
                INSERT INTO raw VALUES (?, 1, ?, ?, ?)
                ON CONFLICT(smiles) DO UPDATE SET occurrences = occurrences + 1
                """,
                (value, dataset_id, reaction_id, location),
            )

        audit["dataset_files"] += 1
        for reaction in dataset.reactions:
            audit["reactions"] += 1
            reaction_id = str(reaction.reaction_id)
            for input_name, reaction_input in reaction.inputs.items():
                for component_index, compound in enumerate(reaction_input.components):
                    audit["structured_compounds"] += 1
                    try:
                        smiles = message_helpers.smiles_from_compound(
                            compound, canonical=False
                        )
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        audit["structured_compounds_without_structure"] += 1
                        continue
                    observe(
                        smiles, reaction_id, f"input:{input_name}:{component_index}"
                    )
            for outcome_index, outcome in enumerate(reaction.outcomes):
                for product_index, product in enumerate(outcome.products):
                    audit["structured_products"] += 1
                    try:
                        smiles = message_helpers.smiles_from_compound(
                            product, canonical=False
                        )
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        audit["structured_products_without_structure"] += 1
                        continue
                    observe(
                        smiles,
                        reaction_id,
                        f"outcome:{outcome_index}:product:{product_index}",
                    )
            try:
                reaction_smiles = message_helpers.get_reaction_smiles(
                    reaction,
                    generate_if_missing=False,
                    canonical=False,
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                reaction_smiles = None
                audit["invalid_reaction_smiles_identifiers"] += 1
            if reaction_smiles:
                audit["reactions_with_reaction_smiles"] += 1
                for component, location in reaction_smiles_components(reaction_smiles):
                    observe(component, reaction_id, location)

        connection.execute(
            "INSERT INTO processed VALUES (?, ?, ?, ?)",
            (
                source_key,
                sha256_file(source_file),
                dataset_id,
                json.dumps(dict(sorted(audit.items())), sort_keys=True),
            ),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        connection.close()
        raise
    connection.close()


def extract_raw(
    snapshot_dir: Path,
    output: Path,
    audit_output: Path,
    work_database: Path,
    workers: int,
) -> None:
    """Extract raw strings with restart-safe isolated ORD subprocesses."""
    head = git_output(snapshot_dir, "rev-parse", "HEAD")
    if head != ORD_COMMIT:
        raise RuntimeError(f"Expected ORD commit {ORD_COMMIT}; found {head}")
    source_paths = sorted(
        path.resolve() for path in snapshot_dir.glob("data/*/ord_dataset-*.pb.gz")
    )
    if len(source_paths) != EXPECTED_PROTOBUF_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_PROTOBUF_FILES} protobuf files; found {len(source_paths)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    work_database = work_database.resolve()
    connection = initialize_raw_database(work_database)
    processed = {
        row[0] for row in connection.execute("SELECT source_path FROM processed")
    }
    connection.close()
    completed = sum(str(path) in processed for path in source_paths)
    if completed:
        print(
            f"Resuming extraction: {completed}/{len(source_paths)} datasets complete",
            flush=True,
        )

    pending_paths = [path for path in source_paths if str(path) not in processed]

    def process(path: Path, database: Path) -> Path:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "extract-one",
                "--source-file",
                str(path),
                "--work-database",
                str(database),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        return path

    if workers == 1:
        for path in pending_paths:
            process(path, work_database)
            completed += 1
            print(
                f"Extracted {completed}/{len(source_paths)} ORD datasets: {path.name}",
                flush=True,
            )
    elif pending_paths:
        chunks = [pending_paths[index::workers] for index in range(workers)]

        def process_chunk(paths: list[Path], database: Path) -> Path:
            for path in paths:
                process(path, database)
            return database

        with tempfile.TemporaryDirectory(prefix="synde-ord-raw-") as directory:
            shard_paths = [
                Path(directory) / f"worker-{index}.sqlite3" for index in range(workers)
            ]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(process_chunk, chunk, shard)
                    for chunk, shard in zip(chunks, shard_paths)
                    if chunk
                ]
                for future in futures:
                    future.result()
            for chunk, shard in zip(chunks, shard_paths):
                if chunk:
                    connection = initialize_raw_database(work_database)
                    connection.execute("ATTACH DATABASE ? AS shard", (str(shard),))
                    connection.execute("""
                        INSERT INTO raw
                        SELECT smiles, occurrences, example_dataset_id,
                               example_reaction_id, example_location
                        FROM shard.raw WHERE true
                        ON CONFLICT(smiles) DO UPDATE SET
                            occurrences = raw.occurrences + excluded.occurrences
                        """)
                    connection.execute("""
                        INSERT INTO processed
                        SELECT source_path, source_sha256, dataset_id, counts_json
                        FROM shard.processed
                        """)
                    connection.commit()
                    connection.execute("DETACH DATABASE shard")
                    connection.close()
                    completed += len(chunk)
                    print(
                        f"Merged {len(chunk)} datasets from {shard.name}; "
                        f"{completed}/{len(source_paths)} complete",
                        flush=True,
                    )

    connection = initialize_raw_database(work_database)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    processed_rows = list(
        connection.execute("SELECT source_path, counts_json FROM processed")
    )
    expected_paths = {str(path) for path in source_paths}
    completed_paths = {row[0] for row in processed_rows}
    missing = expected_paths - completed_paths
    if missing:
        connection.close()
        raise RuntimeError(f"Extraction incomplete: {len(missing)} datasets missing")

    audit = Counter()
    for source_path, counts_json in processed_rows:
        if source_path in expected_paths:
            audit.update(json.loads(counts_json))
    unique_raw = int(connection.execute("SELECT COUNT(*) FROM raw").fetchone()[0])
    audit["unique_raw_molecular_strings"] = unique_raw
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "SMILES",
                "occurrences",
                "example_dataset_id",
                "example_reaction_id",
                "example_location",
            ]
        )
        query = (
            "SELECT smiles, occurrences, example_dataset_id, "
            "example_reaction_id, example_location "
            "FROM raw ORDER BY smiles"
        )
        for row in connection.execute(query):
            writer.writerow(row)
    connection.close()
    temporary_output.replace(output)

    payload = {
        "protocol": RAW_EXTRACTION_PROTOCOL,
        "repository": ORD_REPOSITORY,
        "commit": ORD_COMMIT,
        "source_format": "ord_schema Dataset protobuf (.pb.gz)",
        "source_file_policy": (
            "all 550 protobuf datasets; Parquet siblings and the two merged "
            "Parquet derivatives excluded to prevent duplicate reactions"
        ),
        "restart_policy": (
            "one isolated process and one atomic SQLite transaction per dataset"
        ),
        "work_database": recorded_path(work_database),
        "output": recorded_path(output),
        "output_sha256": sha256_file(output),
        "counts": dict(sorted(audit.items())),
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output} ({unique_raw:,} unique raw strings)")


def smiles_column(fieldnames: Iterable[str] | None) -> str:
    """Return a recognized molecular-string column."""
    names = set(fieldnames or ())
    for candidate in ("SMILES", "smiles", "Canonical_SMILES", "mol"):
        if candidate in names:
            return candidate
    raise ValueError(f"No recognized SMILES column in {sorted(names)}")


def _canonicalize_molecule(
    payload: tuple[bytes, int],
) -> tuple[list[tuple[object, ...]], dict[str, int]]:
    """Canonicalize one parsed parent in a process worker."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    molecule = Chem.Mol(payload[0])
    occurrences = payload[1]
    audit = Counter()
    try:
        fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    except (RuntimeError, ValueError):
        audit["excluded_fragment_sanitization_failure"] += 1
        return [], dict(audit)
    audit["parsed_parent_rows"] += 1
    audit["component_occurrences"] += len(fragments)
    if len(fragments) > 1:
        audit["disconnected_parent_rows"] += 1
    records: list[tuple[object, ...]] = []
    for fragment in fragments:
        for atom in fragment.GetAtoms():
            atom.SetAtomMapNum(0)
        isotopically_labelled = has_isotopically_labelled_atom(fragment)
        try:
            Chem.SanitizeMol(fragment)
            if not isotopically_labelled:
                fragment = normalize_ordinary_explicit_hydrogens(fragment)
            canonical = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
            connectivity = Chem.MolToSmiles(
                fragment, canonical=True, isomericSmiles=False
            )
            formula = rdMolDescriptors.CalcMolFormula(fragment)
        except (RuntimeError, ValueError):
            audit["excluded_canonicalization_failure"] += 1
            continue
        charge = int(Chem.GetFormalCharge(fragment))
        elements_set = {atom.GetSymbol() for atom in fragment.GetAtoms()}
        if any(
            atom.GetTotalNumHs(includeNeighbors=True) > 0
            for atom in fragment.GetAtoms()
        ):
            elements_set.add("H")
        reasons = []
        if isotopically_labelled:
            reasons.append(ISOTOPE_EXCLUSION_REASON)
        if charge != 0:
            reasons.append("nonzero_charge")
        if any(atom.GetNumRadicalElectrons() for atom in fragment.GetAtoms()):
            reasons.append("radical")
        if not elements_set <= SUPPORTED_ELEMENTS:
            reasons.append("unsupported_element")
        records.append(
            (
                canonical,
                connectivity,
                formula,
                charge,
                ";".join(sorted(elements_set)),
                int(fragment.GetNumHeavyAtoms()),
                occurrences,
                int(not reasons),
                ";".join(reasons),
            )
        )
    return records, dict(audit)


def canonicalize(
    source: Path,
    output: Path,
    groups_output: Path,
    audit_output: Path,
    workers: int,
) -> None:
    """Split, canonicalize, classify, and deduplicate ORD molecular components."""
    try:
        from rdkit import Chem, RDLogger, rdBase
    except ImportError as error:
        raise RuntimeError("RDKit is required") from error

    # Invalid molecular strings remain fully counted in the audit, without
    # emitting millions of expected parser diagnostics from heterogeneous ORD
    # identifier fields.
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    output.parent.mkdir(parents=True, exist_ok=True)
    groups_output.parent.mkdir(parents=True, exist_ok=True)
    audit = Counter()
    with tempfile.TemporaryDirectory(
        prefix="synde-ord-canonical-", dir=output.parent
    ) as tmp:
        database = Path(tmp) / "canonical.sqlite3"
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("""
            CREATE TABLE molecules (
                smiles TEXT PRIMARY KEY,
                connectivity TEXT NOT NULL,
                formula TEXT NOT NULL,
                formal_charge INTEGER NOT NULL,
                elements TEXT NOT NULL,
                heavy_atoms INTEGER NOT NULL,
                source_occurrences INTEGER NOT NULL,
                basic_eligible INTEGER NOT NULL,
                exclusion_reason TEXT NOT NULL
            )
            """)
        with source.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        column = smiles_column(header)
        supplier = Chem.MultithreadedSmilesMolSupplier(
            str(source),
            delimiter=",",
            smilesColumn=header.index(column),
            nameColumn=-1,
            titleLine=True,
            sanitize=True,
            numWriterThreads=workers,
        )

        def parsed_payloads() -> Iterator[tuple[bytes, int]]:
            for molecule in supplier:
                audit["raw_rows"] += 1
                if molecule is None:
                    audit["excluded_parse_failure"] += 1
                    continue
                try:
                    occurrences = max(
                        1,
                        int(
                            molecule.GetProp("occurrences")
                            if molecule.HasProp("occurrences")
                            else "1"
                        ),
                    )
                except ValueError:
                    occurrences = 1
                yield molecule.ToBinary(), occurrences

        insert_sql = """
            INSERT INTO molecules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(smiles) DO UPDATE SET
                source_occurrences = source_occurrences + excluded.source_occurrences
        """
        processed = 0
        with multiprocessing.Pool(processes=workers) as pool:
            for records, worker_audit in pool.imap_unordered(
                _canonicalize_molecule, parsed_payloads(), chunksize=64
            ):
                audit.update(worker_audit)
                if records:
                    connection.executemany(insert_sql, records)
                processed += 1
                if processed % 100_000 == 0:
                    connection.commit()
                    print(f"Canonicalized {processed:,} parsed raw rows", flush=True)
        # RDKit emits one empty sentinel after the final record.
        if audit["excluded_parse_failure"] < 1:
            raise RuntimeError(
                "Multithreaded SMILES supplier did not terminate normally"
            )
        audit["raw_rows"] -= 1
        audit["excluded_parse_failure"] -= 1
        connection.commit()

        connection.execute("CREATE INDEX connectivity_index ON molecules(connectivity)")
        connection.execute(
            "CREATE INDEX formula_index ON molecules(formula, formal_charge)"
        )
        connection.commit()

        temporary_output = output.with_suffix(output.suffix + ".tmp")
        with temporary_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "SMILES",
                    "connectivity",
                    "formula",
                    "formal_charge",
                    "elements",
                    "heavy_atoms",
                    "source_occurrences",
                    "connectivity_variants",
                    "basic_eligible",
                    "paper_eligible",
                    "exclusion_reason",
                ]
            )
            query = """
                WITH variants AS (
                    SELECT connectivity, COUNT(*) AS count
                    FROM molecules GROUP BY connectivity
                )
                SELECT m.smiles, m.connectivity, m.formula, m.formal_charge,
                       m.elements, m.heavy_atoms, m.source_occurrences,
                       v.count, m.basic_eligible,
                       CASE WHEN m.basic_eligible = 1 AND v.count = 1 THEN 1 ELSE 0 END,
                       CASE
                           WHEN m.basic_eligible = 1 AND v.count > 1
                           THEN 'stereochemical_identity_ambiguity'
                           ELSE m.exclusion_reason
                       END
                FROM molecules AS m JOIN variants AS v USING(connectivity)
                ORDER BY m.smiles
            """
            for row in connection.execute(query):
                writer.writerow(row)
        temporary_output.replace(output)

        temporary_groups = groups_output.with_suffix(groups_output.suffix + ".tmp")
        with temporary_groups.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "formula",
                    "formal_charge",
                    "canonical_molecules",
                    "unique_connectivities",
                    "paper_eligible_molecules",
                    "rankable_group",
                ]
            )
            group_query = """
                WITH variants AS (
                    SELECT connectivity, COUNT(*) AS count
                    FROM molecules GROUP BY connectivity
                )
                SELECT m.formula, m.formal_charge,
                       COUNT(*) AS canonical_molecules,
                       COUNT(DISTINCT m.connectivity) AS unique_connectivities,
                       SUM(CASE WHEN m.basic_eligible = 1 AND v.count = 1 THEN 1 ELSE 0 END)
                           AS paper_eligible_molecules
                FROM molecules AS m JOIN variants AS v USING(connectivity)
                GROUP BY m.formula, m.formal_charge
                ORDER BY m.formal_charge, m.formula
            """
            for (
                formula,
                charge,
                canonical_count,
                connectivity_count,
                eligible,
            ) in connection.execute(group_query):
                writer.writerow(
                    [
                        formula,
                        charge,
                        canonical_count,
                        connectivity_count,
                        eligible,
                        int(charge == 0 and eligible >= 3),
                    ]
                )
        temporary_groups.replace(groups_output)

        total = int(connection.execute("SELECT COUNT(*) FROM molecules").fetchone()[0])
        connectivities = int(
            connection.execute(
                "SELECT COUNT(DISTINCT connectivity) FROM molecules"
            ).fetchone()[0]
        )
        basic = int(
            connection.execute(
                "SELECT COUNT(*) FROM molecules WHERE basic_eligible = 1"
            ).fetchone()[0]
        )
        excluded_isotopically_labelled = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM molecules
                WHERE instr(exclusion_reason, ?) > 0
                """,
                (ISOTOPE_EXCLUSION_REASON,),
            ).fetchone()[0]
        )
        paper_eligible = int(connection.execute("""
                WITH variants AS (
                    SELECT connectivity, COUNT(*) AS count
                    FROM molecules GROUP BY connectivity
                )
                SELECT COUNT(*) FROM molecules AS m JOIN variants AS v USING(connectivity)
                WHERE m.basic_eligible = 1 AND v.count = 1
                """).fetchone()[0])
        group_counts = [int(row[0]) for row in connection.execute("""
                WITH variants AS (
                    SELECT connectivity, COUNT(*) AS count
                    FROM molecules GROUP BY connectivity
                ), eligible AS (
                    SELECT m.formula, COUNT(*) AS count
                    FROM molecules AS m JOIN variants AS v USING(connectivity)
                    WHERE m.basic_eligible = 1 AND v.count = 1 AND m.formal_charge = 0
                    GROUP BY m.formula
                )
                SELECT count FROM eligible
                """)]
        connection.close()

    size_distribution = Counter(group_counts)
    payload = {
        "protocol": "synde-ord-component-canonicalization-v1",
        "source": recorded_path(source),
        "source_sha256": sha256_file(source),
        "rdkit_version": rdBase.rdkitVersion,
        "component_policy": (
            "RDKit GetMolFrags; atom-map numbers removed; canonical isomeric SMILES "
            "retained; achiral connectivity stored separately"
        ),
        "paper_eligibility": {
            "formal_charge": 0,
            "radical_electrons": 0,
            "supported_elements": sorted(SUPPORTED_ELEMENTS),
            "isotope_numbers": [0],
            "ordinary_explicit_hydrogen_policy": "normalize_to_implicit",
            "achiral_connectivity_variants": 1,
        },
        "counts": {
            **dict(sorted(audit.items())),
            "canonical_molecules": total,
            "unique_achiral_connectivities": connectivities,
            "basic_eligible_molecules": basic,
            "excluded_isotopically_labelled": excluded_isotopically_labelled,
            "paper_eligible_molecules": paper_eligible,
            "neutral_formula_groups": len(group_counts),
            "rankable_formula_groups_size_at_least_3": sum(
                count >= 3 for count in group_counts
            ),
            "molecules_in_rankable_formula_groups": sum(
                count for count in group_counts if count >= 3
            ),
        },
        "eligible_formula_group_size_distribution": {
            str(size): count for size, count in sorted(size_distribution.items())
        },
        "outputs": {
            recorded_path(output): sha256_file(output),
            recorded_path(groups_output): sha256_file(groups_output),
        },
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {output}: {total:,} canonical molecules; "
        f"{paper_eligible:,} paper-eligible; "
        f"{payload['counts']['rankable_formula_groups_size_at_least_3']:,} "
        "rankable formula groups"
    )


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "stage", choices=("fetch", "extract-one", "extract", "canonicalize", "all")
    )
    result.add_argument("--snapshot-dir", type=Path, default=DATA_DIR / "ord-data")
    result.add_argument("--raw-output", type=Path, default=DATA_DIR / "ord_raw.csv")
    result.add_argument("--output", type=Path, default=DATA_DIR / "ord.csv")
    result.add_argument(
        "--groups-output", type=Path, default=DATA_DIR / "ord_formula_groups.csv"
    )
    result.add_argument(
        "--raw-audit", type=Path, default=DATA_DIR / "ord_raw.audit.json"
    )
    result.add_argument(
        "--work-database", type=Path, default=DATA_DIR / ".ord_raw_work_v3.sqlite3"
    )
    result.add_argument("--source-file", type=Path)
    result.add_argument("--extraction-workers", type=int, default=1)
    result.add_argument("--canonicalization-workers", type=int, default=6)
    result.add_argument("--audit", type=Path, default=DATA_DIR / "ord.audit.json")
    return result


def main() -> None:
    """Run one or all pipeline stages."""
    args = parser().parse_args()
    if args.stage == "extract-one":
        if args.source_file is None:
            raise ValueError("--source-file is required for extract-one")
        extract_one(args.source_file, args.work_database.resolve())
        return
    if args.extraction_workers < 1:
        raise ValueError("--extraction-workers must be at least one")
    if args.canonicalization_workers < 1:
        raise ValueError("--canonicalization-workers must be at least one")
    if args.stage in {"fetch", "all"}:
        fetch_snapshot(args.snapshot_dir.resolve())
    if args.stage in {"extract", "all"}:
        extract_raw(
            args.snapshot_dir.resolve(),
            args.raw_output.resolve(),
            args.raw_audit.resolve(),
            args.work_database.resolve(),
            args.extraction_workers,
        )
    if args.stage in {"canonicalize", "all"}:
        canonicalize(
            args.raw_output.resolve(),
            args.output.resolve(),
            args.groups_output.resolve(),
            args.audit.resolve(),
            args.canonicalization_workers,
        )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"ERROR: {error}")
