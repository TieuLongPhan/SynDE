#!/usr/bin/env python3
"""Validate built SynDE distributions and smoke-test the wheel."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile

PACKAGE_FILES = (
    "models/synde_energy_model.json",
    "models/synde_external_validation.json",
    "models/synde_frozen_model.json",
)
PACKAGE_MODULES = (
    "cli.py",
    "errors.py",
    "formatting.py",
    "report.py",
    "energy/refinement.py",
)


def _one_artifact(directory: Path, pattern: str) -> Path:
    """Return the only artifact matching a glob."""
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {pattern!r} artifact in {directory}, found {matches}."
        )
    return matches[0]


def _assert_members(members: set[str], artifact: Path) -> None:
    """Require package source, data, and metadata paths in an archive."""
    required = [
        "synde/__init__.py",
        *[f"synde/{name}" for name in PACKAGE_FILES],
        *[f"synde/{name}" for name in PACKAGE_MODULES],
    ]
    missing = [
        suffix
        for suffix in required
        if not any(member.endswith(suffix) for member in members)
    ]
    if missing:
        raise RuntimeError(f"{artifact.name} is missing: {', '.join(missing)}")


def inspect_archives(wheel: Path, sdist: Path) -> None:
    """Inspect wheel and source archives for required files."""
    with zipfile.ZipFile(wheel) as archive:
        _assert_members(set(archive.namelist()), wheel)
    with tarfile.open(sdist, "r:gz") as archive:
        _assert_members(set(archive.getnames()), sdist)


def smoke_test_wheel(wheel: Path, project_root: Path) -> None:
    """Install and import a wheel outside the source checkout."""
    with (project_root / "pyproject.toml").open("rb") as handle:
        expected_version = tomllib.load(handle)["project"]["version"]

    with tempfile.TemporaryDirectory(prefix="synde-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
            check=True,
            cwd=root,
        )
        code = f"""
from importlib.metadata import entry_points, version
from importlib.resources import files
import synde

expected = {expected_version!r}
installed = version("synde")
if installed != expected:
    raise SystemExit("version mismatch: %s != %s" % (installed, expected))
for filename in {PACKAGE_FILES!r}:
    resource = files("synde").joinpath(filename)
    if not resource.is_file():
        raise SystemExit("missing installed package file: %s" % filename)
for symbol in ("EnergyRefinementRecord", "SynDEEnergyRefiner"):
    if not hasattr(synde, symbol):
        raise SystemExit("missing public API: synde.%s" % symbol)
commands = [entry for entry in entry_points(group="console_scripts") if entry.name == "synde"]
if len(commands) != 1 or commands[0].value != "synde.cli:main":
    raise SystemExit("missing or invalid synde console entry point: %r" % commands)
print("synde %s: wheel import, CLI, refinement API, and model files verified" % installed)
"""
        subprocess.run([str(python), "-c", code], check=True, cwd=root)
        command = environment / (
            "Scripts/synde.exe" if sys.platform == "win32" else "bin/synde"
        )
        completed = subprocess.run(
            [str(command), "--version"],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        )
        observed = completed.stdout.strip()
        expected = f"synde {expected_version}"
        if observed != expected:
            raise RuntimeError(
                f"Console version mismatch: {observed!r} != {expected!r}"
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("dist"),
        help="directory containing exactly one wheel and one source archive",
    )
    return parser.parse_args()


def main() -> None:
    """Run archive inspection and the isolated wheel smoke test."""
    args = parse_args()
    directory = args.directory.resolve()
    wheel = _one_artifact(directory, "*.whl")
    sdist = _one_artifact(directory, "*.tar.gz")
    project_root = Path(__file__).resolve().parents[1]
    inspect_archives(wheel, sdist)
    smoke_test_wheel(wheel, project_root)


if __name__ == "__main__":
    main()
