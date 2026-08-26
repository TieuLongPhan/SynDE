#!/usr/bin/env python3
"""Run the frozen ORD candidate manifests through the declared xTB protocol.

The runner is restart-safe: every completed molecule is committed to SQLite
before another result is accepted. It uses at most 16 process workers and one
thread per xTB process, so ``--workers 16`` consumes at most 16 computational
CPU threads. No optimized XYZ files or full xTB logs are retained.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAX_WORKERS = 16
PROTOCOL = "synde-ord-gfn2-xtb-extreme-v1"
EXPECTED_XTB_VERSION = "6.7.1"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def sha256_file(path: Path) -> str:
    """Return a SHA-256 file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worker_environment() -> None:
    """Pin every numerical backend in a worker to one CPU thread."""
    os.environ.update(THREAD_ENVIRONMENT)


def optimize_one(task: dict[str, str], timeout: int) -> dict[str, object]:
    """Optimize one candidate and return only compact, persistent fields."""
    worker_environment()
    import logging

    # Per-molecule failures are persisted below; suppress verbose library logs
    # that would otherwise make a 98k-molecule workstation run unwieldy.
    logging.disable(logging.CRITICAL)
    from synde.geometry.xtb.xtb_minimize import XTBMinimize

    started = time.monotonic()
    optimizer = XTBMinimize(
        task["SMILES"],
        embed_max_attempts=3,
        embed_seed=42,
        charge=0,
        multiplicity=1,
        gfn=2,
    )
    with tempfile.TemporaryDirectory(prefix="synde-xtb-") as temporary:
        result = optimizer.optimize(
            save_dir=temporary,
            level="extreme",
            timeout=timeout,
            clean=True,
            keep_intermediates=False,
            xtb_omp_threads=1,
        )
    status = str(result.get("status", "error"))
    if status == "success" and result.get("energy_Eh") is None:
        status = "missing_energy"
    return {
        "split": task["split"],
        "group_id": task["group_id"],
        "formula": task["formula"],
        "SMILES": task["SMILES"],
        "connectivity": task["connectivity"],
        "status": status,
        "message": str(result.get("message", ""))[:1000],
        "energy_Eh": result.get("energy_Eh"),
        "energy_eV": result.get("energy_eV"),
        "elapsed_seconds": time.monotonic() - started,
    }


def initialize_database(database: Path) -> sqlite3.Connection:
    """Open the result database and create its restart metadata."""
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute("""
        CREATE TABLE IF NOT EXISTS results (
            split TEXT NOT NULL,
            group_id TEXT NOT NULL,
            formula TEXT NOT NULL,
            smiles TEXT NOT NULL,
            connectivity TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT NOT NULL,
            energy_Eh REAL,
            energy_eV REAL,
            elapsed_seconds REAL NOT NULL,
            protocol TEXT NOT NULL,
            PRIMARY KEY (split, connectivity)
        )
        """)
    stored = connection.execute(
        "SELECT value FROM metadata WHERE key = 'protocol'"
    ).fetchone()
    if stored is not None and stored[0] != PROTOCOL:
        connection.close()
        raise RuntimeError(f"Result database uses incompatible protocol {stored[0]}")
    connection.execute(
        "INSERT OR IGNORE INTO metadata VALUES ('protocol', ?)", (PROTOCOL,)
    )
    connection.commit()
    return connection


def validate_manifest(connection: sqlite3.Connection, name: str, path: Path) -> str:
    """Bind one immutable candidate manifest to the result database."""
    digest = sha256_file(path)
    key = f"manifest_{name}_sha256"
    stored = connection.execute(
        "SELECT value FROM metadata WHERE key = ?", (key,)
    ).fetchone()
    if stored is not None and stored[0] != digest:
        raise RuntimeError(
            f"{name} manifest changed after calculations began: {stored[0]} != {digest}"
        )
    connection.execute("INSERT OR IGNORE INTO metadata VALUES (?, ?)", (key, digest))
    connection.commit()
    return digest


def manifest_rows(path: Path, expected_split: str) -> Iterator[dict[str, str]]:
    """Yield validated candidate rows in their frozen order."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "group_id", "formula", "SMILES", "connectivity"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            if row["split"] != expected_split:
                raise RuntimeError(
                    f"Unexpected split {row['split']!r} in {path}; "
                    f"expected {expected_split!r}"
                )
            yield row


def pending_rows(
    connection: sqlite3.Connection,
    manifests: list[tuple[str, Path]],
    retry_failures: bool,
) -> list[dict[str, str]]:
    """Return candidates not already completed in the SQLite database."""
    existing = {
        (str(split), str(connectivity)): str(status)
        for split, connectivity, status in connection.execute(
            "SELECT split, connectivity, status FROM results"
        )
    }
    pending = []
    for expected_split, path in manifests:
        for row in manifest_rows(path, expected_split):
            status = existing.get((expected_split, row["connectivity"]))
            if status is None or (retry_failures and status != "success"):
                pending.append(row)
    return pending


def store_result(connection: sqlite3.Connection, result: dict[str, object]) -> None:
    """Atomically insert or replace one completed calculation."""
    connection.execute(
        """
        INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["split"],
            result["group_id"],
            result["formula"],
            result["SMILES"],
            result["connectivity"],
            result["status"],
            result["message"],
            result["energy_Eh"],
            result["energy_eV"],
            result["elapsed_seconds"],
            PROTOCOL,
        ),
    )
    connection.commit()


def export_results(connection: sqlite3.Connection, path: Path, split: str) -> None:
    """Export one split from SQLite to an atomic CSV snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "split",
                "group_id",
                "formula",
                "SMILES",
                "connectivity",
                "status",
                "message",
                "energy_Eh",
                "energy_eV",
                "elapsed_seconds",
                "protocol",
            ]
        )
        writer.writerows(
            connection.execute(
                """
                SELECT split, group_id, formula, smiles, connectivity, status,
                       message, energy_Eh, energy_eV, elapsed_seconds, protocol
                FROM results WHERE split = ? ORDER BY group_id, connectivity
                """,
                (split,),
            )
        )
    temporary.replace(path)


def xtb_version() -> str:
    """Return a compact xTB version record and fail if xTB is unavailable."""
    process = subprocess.run(
        ["xtb", "--version"], capture_output=True, text=True, check=True
    )
    version = " ".join((process.stdout or process.stderr).split())[:500]
    if EXPECTED_XTB_VERSION not in version:
        raise RuntimeError(
            f"Expected xTB {EXPECTED_XTB_VERSION}; detected version record: {version}"
        )
    return version


def run_calculations(args: argparse.Namespace) -> None:
    """Load manifests, resume calculations, and export result snapshots."""
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f"--workers must be between 1 and {MAX_WORKERS}")
    if args.timeout < 1:
        raise ValueError("--timeout must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    choices = {
        "training": ("training", args.training_manifest.resolve()),
        "test": ("external", args.test_manifest.resolve()),
    }
    selected_names = ("training", "test") if args.split == "all" else (args.split,)
    manifests = [choices[name] for name in selected_names]
    for name in selected_names:
        _, path = choices[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} candidate manifest: {path}")
    connection = initialize_database(args.database.resolve())
    for name in selected_names:
        _, path = choices[name]
        validate_manifest(connection, name, path)
    pending = pending_rows(connection, manifests, args.retry_failures)
    total_pending = len(pending)
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"Candidates pending: {total_pending:,}; scheduled now: {len(pending):,}; "
        f"workers: {args.workers} (one xTB thread each)",
        flush=True,
    )
    if args.dry_run:
        connection.close()
        return

    version = xtb_version()
    connection.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('xtb_version', ?)", (version,)
    )
    connection.commit()
    worker_environment()

    completed = 0
    failures = 0
    iterator = iter(pending)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=worker_environment
    ) as executor:
        futures = {}
        for _ in range(min(len(pending), args.workers * 2)):
            task = next(iterator, None)
            if task is None:
                break
            futures[executor.submit(optimize_one, task, args.timeout)] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                try:
                    result = future.result()
                except BaseException as error:
                    result = {
                        "split": task["split"],
                        "group_id": task["group_id"],
                        "formula": task["formula"],
                        "SMILES": task["SMILES"],
                        "connectivity": task["connectivity"],
                        "status": "worker_error",
                        "message": repr(error)[:1000],
                        "energy_Eh": None,
                        "energy_eV": None,
                        "elapsed_seconds": 0.0,
                    }
                store_result(connection, result)
                completed += 1
                failures += result["status"] != "success"
                if completed % 100 == 0 or completed == len(pending):
                    print(
                        f"Completed {completed:,}/{len(pending):,}; "
                        f"failures this run: {failures:,}",
                        flush=True,
                    )
                task = next(iterator, None)
                if task is not None:
                    futures[executor.submit(optimize_one, task, args.timeout)] = task

    export_results(connection, args.training_output.resolve(), "training")
    export_results(connection, args.test_output.resolve(), "external")
    connection.close()


def parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--training-manifest",
        type=Path,
        default=DATA_DIR / "ord_training_candidates.csv",
    )
    result.add_argument(
        "--test-manifest",
        type=Path,
        default=DATA_DIR / "ord_external_candidates.csv",
    )
    result.add_argument(
        "--database", type=Path, default=DATA_DIR / "ord_xtb_results.sqlite3"
    )
    result.add_argument(
        "--training-output", type=Path, default=DATA_DIR / "ord_training_xtb.csv"
    )
    result.add_argument(
        "--test-output", type=Path, default=DATA_DIR / "ord_test_xtb.csv"
    )
    result.add_argument("--split", choices=("training", "test", "all"), default="all")
    result.add_argument("--workers", type=int, default=MAX_WORKERS)
    result.add_argument("--timeout", type=int, default=1800)
    result.add_argument("--limit", type=int)
    result.add_argument("--retry-failures", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    """Run the command-line program."""
    run_calculations(parser().parse_args())


if __name__ == "__main__":
    try:
        main()
    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
        subprocess.CalledProcessError,
    ) as error:
        sys.exit(f"ERROR: {error}")
