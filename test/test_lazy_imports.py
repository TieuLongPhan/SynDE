from __future__ import annotations

import subprocess
import sys

import pytest

HEAVY = ("rdkit", "networkx", "numpy", "synkit")


def _loaded(statement: str) -> set[str]:
    """Return the top-level module names loaded by one import statement.

    :param statement: Python source executed in a fresh interpreter.
    :type statement: str
    :return: Distinct top-level module names present afterwards.
    :rtype: set[str]
    """
    code = (
        f"{statement}\n"
        "import sys\n"
        "print(' '.join(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return set(result.stdout.split())


def test_importing_the_package_stays_free_of_heavy_dependencies() -> None:
    loaded = _loaded("import synde")
    assert "synde" in loaded
    assert not [name for name in HEAVY if name in loaded]


def test_importing_the_cli_stays_free_of_heavy_dependencies() -> None:
    loaded = _loaded("import synde.cli")
    assert not [name for name in HEAVY if name in loaded]


def test_lazy_access_still_resolves_every_public_name() -> None:
    import synde

    for name in synde.__all__:
        assert getattr(synde, name) is not None


def test_lazy_access_still_resolves_every_energy_name() -> None:
    import synde.energy as energy

    for name in energy.__all__:
        assert getattr(energy, name) is not None


def test_unknown_attribute_raises_attribute_error() -> None:
    import synde

    with pytest.raises(AttributeError, match="no attribute"):
        synde.NotARealSymbol


def test_dir_lists_the_public_exports() -> None:
    import synde

    assert set(synde.__all__) <= set(dir(synde))


def test_inference_import_does_not_pull_in_conformer_machinery() -> None:
    """The predictor claims 2D-only inference; its imports must match."""
    code = (
        "from synde.energy import SynDEEnergyPredictor\n"
        "import sys\n"
        "geo = [m for m in sys.modules if m.startswith('synde.geometry')]\n"
        "print(len(geo))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "0"
