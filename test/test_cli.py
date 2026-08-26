from __future__ import annotations

import json
from pathlib import Path

import pytest

from synde import cli

MODEL_PATH = Path("synde/models/synde_energy_model.json")
requires_model = pytest.mark.skipif(
    not MODEL_PATH.is_file(), reason="packaged energy model is not present"
)


def test_parser_exposes_every_documented_subcommand() -> None:
    parser = cli.build_parser()
    cases = {
        "predict": ["CCO"],
        "explain": ["CCO"],
        "rank": ["CCO", "CCC"],
        "card": [],
        "info": [],
    }
    for command, arguments in cases.items():
        parsed = parser.parse_args([command, *arguments])
        assert parsed.handler is not None


def test_no_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == cli.EXIT_OK
    assert "usage: synde" in capsys.readouterr().out


def test_version_flag_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert "synde" in capsys.readouterr().out


def test_info_lists_the_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["info"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "synde predict" in out
    assert "synde rank" in out


def test_read_smiles_merges_arguments_and_file(tmp_path: Path) -> None:
    source = tmp_path / "molecules.smi"
    source.write_text("# comment\nCCO ethanol\n\nCCC\n", encoding="utf-8")
    assert cli._read_smiles(["CO"], str(source)) == ["CO", "CCO", "CCC"]


def test_read_smiles_reports_a_missing_file() -> None:
    with pytest.raises(cli.SynDEError, match="Input file not found"):
        cli._read_smiles([], "does-not-exist.smi")


def test_predict_without_input_explains_how_to_supply_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["predict"]) == cli.EXIT_ERROR
    assert "--input" in capsys.readouterr().err


def test_rank_requires_two_candidates(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["rank", "CCO"]) == cli.EXIT_ERROR
    assert "at least two" in capsys.readouterr().err


def test_missing_model_artifact_is_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["card", "--model", "absent.json"]) == cli.EXIT_ERROR
    assert "Model artifact not found" in capsys.readouterr().err


@requires_model
def test_predict_renders_a_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--no-color", "predict", "CCO", "CC(=O)O"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "energy (eV)" in out
    assert "C2H6O" in out
    assert "\033[" not in out


@requires_model
def test_predict_json_is_machine_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["predict", "CCO", "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    entry = payload["predictions"][0]
    assert entry["input"] == "CCO"
    assert entry["formula"] == "C2H6O"
    assert entry["units"] == "eV"
    assert payload["failures"] == []
    assert entry["top_connectivity_terms"]


@requires_model
def test_explain_shows_the_contribution_breakdown(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--no-color", "explain", "CC(=O)NC", "--top", "3"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "top connectivity terms" in out
    assert "composition" in out
    assert "domain distance" in out


@requires_model
def test_rank_orders_isomers_lowest_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["--no-color", "rank", "CCCCC", "CC(C)CC", "CC(C)(C)C"]
    assert cli.main(argv) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "C5H12" in out
    assert "Δ vs best" in out


@requires_model
def test_rank_json_reports_deltas(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["rank", "CCCCC", "CC(C)CC", "CC(C)(C)C", "--json"]
    assert cli.main(argv) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    ranking = payload["ranking"]
    assert [entry["rank"] for entry in ranking] == [1, 2, 3]
    assert ranking[0]["delta_vs_best"] == 0.0
    assert all(
        ranking[index]["predicted_energy"] <= ranking[index + 1]["predicted_energy"]
        for index in range(len(ranking) - 1)
    )


@requires_model
def test_rank_across_formulas_is_rejected_with_guidance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["rank", "CCO", "CCCO"]) == cli.EXIT_ERROR
    assert "formula" in capsys.readouterr().err


@requires_model
def test_keep_going_skips_bad_inputs_and_reports_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["--no-color", "predict", "CCO", "CC(=O)[O-]", "--keep-going"]
    assert cli.main(argv) == cli.EXIT_ERROR
    captured = capsys.readouterr()
    assert "C2H6O" in captured.out
    assert "skipped" in captured.err


@requires_model
def test_predict_stops_on_bad_input_without_keep_going(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["predict", "CCO", "CC(=O)[O-]"]) == cli.EXIT_ERROR
    assert "formal charge" in capsys.readouterr().err


@requires_model
def test_card_reports_provenance_and_domain(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--no-color", "card"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "model" in out
    assert "elements" in out


@requires_model
def test_card_json_includes_the_artifact_digest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["card", "--json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["units"] == "eV"
    assert "model_sha256" in payload


@requires_model
def test_input_file_and_stdin_are_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "molecules.smi"
    source.write_text("CCO\nCCC\n", encoding="utf-8")
    assert cli.main(["--no-color", "predict", "--input", str(source)]) == cli.EXIT_OK
    assert "C2H6O" in capsys.readouterr().out

    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("CCO\n"))
    assert cli.main(["--no-color", "predict", "--input", "-"]) == cli.EXIT_OK
    assert "C2H6O" in capsys.readouterr().out


def test_format_shorthands_map_to_the_shared_destination() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["predict", "CCO"]).format == "table"
    assert parser.parse_args(["predict", "CCO", "--json"]).format == "json"
    assert parser.parse_args(["predict", "CCO", "--csv"]).format == "csv"
    assert parser.parse_args(["predict", "CCO", "--format", "csv"]).format == "csv"


def test_resolve_jobs_maps_zero_to_every_core() -> None:
    import os

    assert cli._resolve_jobs(1) == 1
    assert cli._resolve_jobs(4) == 4
    assert cli._resolve_jobs(0) == max(1, os.cpu_count() or 1)


def test_plan_workers_caps_by_available_work() -> None:
    assert cli._plan_workers(10, 8) == 1
    assert cli._plan_workers(1000, 4) == 4
    assert cli._plan_workers(1000, 100) == 10


@requires_model
def test_predict_csv_is_valid_and_complete(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import csv
    import io

    assert cli.main(["predict", "CCO", "CC(=O)O", "--format", "csv"]) == cli.EXIT_OK
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [row["input"] for row in rows] == ["CCO", "CC(=O)O"]
    assert rows[0]["formula"] == "C2H6O"
    assert rows[0]["units"] == "eV"
    assert float(rows[0]["predicted_energy"]) < 0


@requires_model
def test_rank_csv_carries_rank_and_delta(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import csv
    import io

    argv = ["rank", "CCCCC", "CC(C)CC", "CC(C)(C)C", "--csv"]
    assert cli.main(argv) == cli.EXIT_OK
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [row["rank"] for row in rows] == ["1", "2", "3"]
    assert float(rows[0]["delta_vs_best"]) == 0.0


@requires_model
def test_parallel_and_serial_agree(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["predict", "CCO", "CCC", "CCCC", "--format", "csv"]
    assert cli.main(argv) == cli.EXIT_OK
    serial = capsys.readouterr().out
    assert cli.main([*argv, "--jobs", "2"]) == cli.EXIT_OK
    assert capsys.readouterr().out == serial


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_completion_scripts_are_emitted(
    shell: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["completion", shell]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "synde" in out
    assert "predict" in out
    assert "rank" in out


def test_completion_rejects_an_unknown_shell() -> None:
    with pytest.raises(SystemExit):
        cli.main(["completion", "tcsh"])


def test_progress_writes_a_single_updating_line() -> None:
    import io

    stream = io.StringIO()
    cli._progress(50, 200, stream)
    assert stream.getvalue() == "\r  scoring 50/200 (25%)"
