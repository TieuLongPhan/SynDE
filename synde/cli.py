"""Command-line interface for SynDE energy prediction and isomer ranking.

Installed as the ``synde`` console script.  Every subcommand accepts SMILES as
positional arguments, from a file with ``--input``, or on standard input, and
every subcommand can emit machine-readable output with ``--format json`` or
``--format csv`` so results pipe cleanly into other tools.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
from typing import IO, Any, Sequence

from .errors import SynDEError
from .formatting import Palette, format_float, render_rule, render_table, truncate
from .report import composition_formula, model_card_summary, ranking_summary

__all__ = ["main", "build_parser"]

EXIT_OK = 0
EXIT_ERROR = 1

_EPILOG = """examples:
  synde predict CCO 'CC(=O)O'
  synde predict --input molecules.smi --format json > energies.json
  synde explain 'CC(=O)NC' --top 12
  synde rank CCCCC 'CC(C)CC' 'CC(C)(C)C'
  cat molecules.smi | synde predict --input - --format csv
  synde predict --input big.smi --jobs 8 --format csv > energies.csv
  synde card
  synde completion zsh >> ~/.zshrc
"""


def _package_version() -> str:
    """Return the installed distribution version.

    :return: Version string, or ``"unknown"`` when metadata is unavailable.
    :rtype: str
    """
    from . import __version__

    return __version__


def _read_smiles(argument_values: Sequence[str], source: str | None) -> list[str]:
    """Collect SMILES from positional arguments and an optional file or stdin.

    Blank lines and ``#`` comments are ignored, and only the first whitespace
    separated field of each line is read so ``.smi`` files with trailing names
    work unchanged.

    :param argument_values: SMILES given as positional arguments.
    :type argument_values: Sequence[str]
    :param source: File path, ``"-"`` for standard input, or ``None``.
    :type source: str | None
    :return: SMILES strings in the order they were supplied.
    :rtype: list[str]
    """
    collected = [value for value in argument_values if value]
    if source is None:
        return collected
    if source == "-":
        text = sys.stdin.read()
    else:
        path = Path(source)
        if not path.is_file():
            raise SynDEError(f"Input file not found: {source}")
        text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        collected.append(stripped.split()[0])
    return collected


def _load_predictor(model: str | None):
    """Load the packaged predictor or an artifact from disk.

    :param model: Path to a saved artifact, or ``None`` for the packaged one.
    :type model: str | None
    :return: Ready-to-use energy predictor.
    :rtype: synde.energy.SynDEEnergyPredictor
    """
    from .energy import SynDEEnergyPredictor

    if model is None:
        return SynDEEnergyPredictor.load_default()
    path = Path(model)
    if not path.is_file():
        raise SynDEError(f"Model artifact not found: {model}")
    return SynDEEnergyPredictor.load(path)


def _palette(args: argparse.Namespace, stream: IO[str]) -> Palette:
    """Resolve the colour switch for one invocation.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :param stream: Stream the rendered output is written to.
    :type stream: IO[str]
    :return: Palette honouring ``--color``, ``--no-color``, and the terminal.
    :rtype: Palette
    """
    if getattr(args, "format", "table") != "table":
        return Palette(False)
    if args.color == "never":
        return Palette(False)
    if args.color == "always":
        return Palette(True)
    return Palette.automatic(stream)


def _prediction_record(smiles: str, prediction: Any, top: int) -> dict[str, Any]:
    """Build the JSON record emitted for one successful prediction.

    :param smiles: SMILES exactly as supplied by the caller.
    :type smiles: str
    :param prediction: Prediction produced for that input.
    :type prediction: synde.energy.SynDEEnergyPrediction
    :param top: Number of connectivity terms included.
    :type top: int
    :return: JSON-ready mapping describing the prediction.
    :rtype: dict[str, Any]
    """
    return {
        "input": smiles,
        "canonical_smiles": prediction.canonical_smiles,
        "formula": composition_formula(prediction.composition),
        "status": prediction.status,
        "predicted_energy": prediction.predicted_energy,
        "units": prediction.units,
        "composition_total": prediction.composition_total,
        "connectivity_total": prediction.connectivity_total,
        "top_connectivity_terms": [
            {"term": name, "contribution": value}
            for name, value in prediction.top_contributions(top)
        ],
        "warnings": list(prediction.warnings),
        "provenance": prediction.provenance,
    }


PROGRESS_THRESHOLD = 200

_WORKER_CACHE: dict[str | None, Any] = {}


def _resolve_jobs(jobs: int) -> int:
    """Turn the ``--jobs`` value into a concrete worker count.

    :param jobs: Requested workers; zero or negative means every core.
    :type jobs: int
    :return: Worker count of at least one.
    :rtype: int
    """
    if jobs is None or jobs == 1:
        return 1
    if jobs <= 0:
        return max(1, os.cpu_count() or 1)
    return jobs


def _progress(done: int, total: int, stream: IO[str]) -> None:
    """Overwrite a single-line progress indicator on an interactive stream.

    :param done: Structures scored so far.
    :type done: int
    :param total: Structures to score in total.
    :type total: int
    :param stream: Stream the indicator is written to.
    :type stream: IO[str]
    """
    percent = 100.0 * done / total if total else 100.0
    stream.write(f"\r  scoring {done}/{total} ({percent:.0f}%)")
    stream.flush()


MIN_CHUNK = 100


def _worker_predict_chunk(chunk: Sequence[str], model: str | None):
    """Score a contiguous chunk in a worker, reusing a cached predictor.

    Each worker pays the model-load cost once rather than once per structure,
    which is why work is dispatched in chunks instead of item by item.

    :param chunk: SMILES strings handled by this worker.
    :type chunk: Sequence[str]
    :param model: Artifact path, or ``None`` for the packaged model.
    :type model: str | None
    :return: Input, prediction, and error triples in chunk order.
    :rtype: list[tuple[str, Any, str | None]]
    """
    predictor = _WORKER_CACHE.get(model)
    if predictor is None:
        predictor = _load_predictor(model)
        _WORKER_CACHE[model] = predictor
    results = []
    for smiles in chunk:
        try:
            results.append((smiles, predictor.predict_smiles(smiles), None))
        except (SynDEError, ValueError) as error:
            results.append((smiles, None, str(error).splitlines()[0]))
    return results


def _plan_workers(total: int, jobs: int) -> int:
    """Cap the worker count so every worker gets enough work to be worth it.

    Spawning a worker costs an interpreter start, an RDKit import, and a model
    load, so more workers than ``total / MIN_CHUNK`` makes a run slower.

    :param total: Number of structures to score.
    :type total: int
    :param jobs: Worker count requested on the command line.
    :type jobs: int
    :return: Worker count to actually use.
    :rtype: int
    """
    return max(1, min(_resolve_jobs(jobs), total // MIN_CHUNK))


def _score_parallel(smiles: Sequence[str], model: str | None, jobs: int):
    """Score every input across worker processes with joblib.

    :param smiles: SMILES strings to score.
    :type smiles: Sequence[str]
    :param model: Artifact path, or ``None`` for the packaged model.
    :type model: str | None
    :param jobs: Worker process count.
    :type jobs: int
    :return: Results in input order as input, prediction, error triples.
    :rtype: list[tuple[str, Any, str | None]]
    """
    from joblib import Parallel, delayed

    size = -(-len(smiles) // jobs)
    chunks = [smiles[start : start + size] for start in range(0, len(smiles), size)]
    batches = Parallel(n_jobs=jobs, prefer="processes")(
        delayed(_worker_predict_chunk)(chunk, model) for chunk in chunks
    )
    return [result for batch in batches for result in batch]


def _partition(results, keep_going: bool):
    """Split worker results into successes and failures.

    :param results: Input, prediction, and error triples.
    :type results: Sequence[tuple[str, Any, str | None]]
    :param keep_going: Record per-input failures rather than raising.
    :type keep_going: bool
    :return: Parallel lists of successes and failures.
    :rtype: tuple[list[tuple[str, Any]], list[tuple[str, str]]]
    :raises SynDEError: If an input failed and ``keep_going`` is false.
    """
    successes: list[tuple[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for item, prediction, error in results:
        if error is None:
            successes.append((item, prediction))
        elif keep_going:
            failures.append((item, error))
        else:
            raise SynDEError(error)
    return successes, failures


def _try_parallel(smiles: Sequence[str], model: str | None, workers: int):
    """Score in parallel, returning ``None`` when joblib is unavailable.

    :param smiles: SMILES strings to score.
    :type smiles: Sequence[str]
    :param model: Artifact path, or ``None`` for the packaged model.
    :type model: str | None
    :param workers: Worker process count.
    :type workers: int
    :return: Result triples, or ``None`` to fall back to serial scoring.
    :rtype: list[tuple[str, Any, str | None]] | None
    """
    try:
        return _score_parallel(smiles, model, workers)
    except ImportError:
        print(
            "note: --jobs needs joblib; install '.[experiment]'. "
            "Scoring serially instead.",
            file=sys.stderr,
        )
        return None


def _score_serial(
    predictor,
    smiles: Sequence[str],
    keep_going: bool,
    progress_stream: IO[str] | None,
):
    """Score every input in this process, reporting progress when useful.

    :param predictor: Loaded energy predictor.
    :type predictor: synde.energy.SynDEEnergyPredictor
    :param smiles: SMILES strings to score.
    :type smiles: Sequence[str]
    :param keep_going: Record per-input failures rather than aborting.
    :type keep_going: bool
    :param progress_stream: Stream for the progress indicator, or ``None``.
    :type progress_stream: IO[str] | None
    :return: Parallel lists of successes and failures.
    :rtype: tuple[list[tuple[str, Any]], list[tuple[str, str]]]
    """
    total = len(smiles)
    show = progress_stream is not None and total >= PROGRESS_THRESHOLD
    successes: list[tuple[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for index, item in enumerate(smiles, start=1):
        try:
            successes.append((item, predictor.predict_smiles(item)))
        except (SynDEError, ValueError) as error:
            if not keep_going:
                if show:
                    progress_stream.write("\n")
                raise
            failures.append((item, str(error).splitlines()[0]))
        if show and (index % 50 == 0 or index == total):
            _progress(index, total, progress_stream)
    if show:
        progress_stream.write("\n")
        progress_stream.flush()
    return successes, failures


def _score_each(
    predictor,
    smiles: Sequence[str],
    keep_going: bool,
    *,
    jobs: int = 1,
    model: str | None = None,
    progress_stream: IO[str] | None = None,
):
    """Predict every input, in parallel when the batch is large enough.

    :param predictor: Loaded energy predictor used for serial scoring.
    :type predictor: synde.energy.SynDEEnergyPredictor
    :param smiles: SMILES strings to score.
    :type smiles: Sequence[str]
    :param keep_going: Record per-input failures rather than aborting.
    :type keep_going: bool
    :param jobs: Worker processes; one scores serially in this process.
    :type jobs: int
    :param model: Artifact path handed to workers, or ``None``.
    :type model: str | None
    :param progress_stream: Stream for the progress indicator, or ``None``.
    :type progress_stream: IO[str] | None
    :return: Parallel lists of successes and failures.
    :rtype: tuple[list[tuple[str, Any]], list[tuple[str, str]]]
    """
    workers = _plan_workers(len(smiles), jobs)
    if workers > 1:
        results = _try_parallel(smiles, model, workers)
        if results is not None:
            return _partition(results, keep_going)
    return _score_serial(predictor, smiles, keep_going, progress_stream)


def _emit_failures(failures: Sequence[tuple[str, str]], palette: Palette) -> None:
    """Write skipped inputs to standard error.

    :param failures: Input and first-line reason pairs.
    :type failures: Sequence[tuple[str, str]]
    :param palette: Styling switch applied to the marker.
    :type palette: Palette
    """
    for item, reason in failures:
        marker = palette("skipped", "yellow", "bold")
        print(f"{marker} {item}: {reason}", file=sys.stderr)


def command_predict(args: argparse.Namespace) -> int:
    """Run the ``predict`` subcommand.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Process exit status.
    :rtype: int
    """
    palette = _palette(args, sys.stdout)
    smiles = _read_smiles(args.smiles, args.input)
    if not smiles:
        raise SynDEError(
            "No input structures given.\n"
            "  Hint: pass SMILES as arguments, or use --input FILE (or "
            "--input - to read standard input)."
        )
    predictor = _load_predictor(args.model)
    successes, failures = _score_each(
        predictor,
        smiles,
        args.keep_going,
        jobs=args.jobs,
        model=args.model,
        progress_stream=sys.stderr if sys.stderr.isatty() else None,
    )

    if args.format == "csv":
        _write_csv(
            [
                "input",
                "canonical_smiles",
                "formula",
                "status",
                "predicted_energy",
                "units",
                "composition_total",
                "connectivity_total",
                "warnings",
            ],
            [
                [
                    item,
                    prediction.canonical_smiles,
                    composition_formula(prediction.composition),
                    prediction.status,
                    prediction.predicted_energy,
                    prediction.units,
                    prediction.composition_total,
                    prediction.connectivity_total,
                    ";".join(prediction.warnings),
                ]
                for item, prediction in successes
            ],
        )
        _emit_failures(failures, palette)
        return EXIT_OK if not failures else EXIT_ERROR

    if args.format == "json":
        payload = {
            "model": predictor.card.model_name,
            "units": predictor.card.units,
            "predictions": [
                _prediction_record(item, prediction, args.top)
                for item, prediction in successes
            ],
            "failures": [{"input": item, "error": reason} for item, reason in failures],
        }
        print(json.dumps(payload, indent=2))
        return EXIT_OK if not failures else EXIT_ERROR

    if args.explain:
        for index, (item, prediction) in enumerate(successes):
            if index:
                print()
            print(prediction.summary(color=palette.enabled, top=args.top))
    elif successes:
        units = predictor.card.units
        rows = [
            [
                truncate(item, 44),
                composition_formula(prediction.composition),
                format_float(prediction.predicted_energy, args.precision),
                format_float(prediction.composition_total, args.precision),
                format_float(prediction.connectivity_total, args.precision),
                "!" if prediction.warnings else "",
            ]
            for item, prediction in successes
        ]
        print(
            render_table(
                [
                    "structure",
                    "formula",
                    f"energy ({units})",
                    "composition",
                    "connectivity",
                    "",
                ],
                rows,
                aligns=["l", "l", "r", "r", "r", "l"],
                palette=palette,
            )
        )
        if any(prediction.warnings for _, prediction in successes):
            print()
            print(palette("! applicability warning; rerun with --explain", "yellow"))
    _emit_failures(failures, palette)
    return EXIT_OK if not failures else EXIT_ERROR


def command_explain(args: argparse.Namespace) -> int:
    """Run the ``explain`` subcommand.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Process exit status.
    :rtype: int
    """
    args.explain = True
    return command_predict(args)


def command_rank(args: argparse.Namespace) -> int:
    """Run the ``rank`` subcommand.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Process exit status.
    :rtype: int
    """
    palette = _palette(args, sys.stdout)
    smiles = _read_smiles(args.smiles, args.input)
    if len(smiles) < 2:
        raise SynDEError(
            "Ranking needs at least two constitutional isomers.\n"
            "  Hint: pass several SMILES that share one molecular formula, "
            "for example: synde rank CCCCC 'CC(C)CC' 'CC(C)(C)C'"
        )
    predictor = _load_predictor(args.model)
    ranking = predictor.rank_smiles(smiles)

    if args.format in {"json", "csv"}:
        best = ranking[0][1].predicted_energy
        payload = {
            "model": predictor.card.model_name,
            "units": predictor.card.units,
            "formula": composition_formula(ranking[0][1].composition),
            "ranking": [
                {
                    "rank": position,
                    "input": smiles[index],
                    "canonical_smiles": prediction.canonical_smiles,
                    "predicted_energy": prediction.predicted_energy,
                    "delta_vs_best": prediction.predicted_energy - best,
                    "connectivity_total": prediction.connectivity_total,
                    "warnings": list(prediction.warnings),
                }
                for position, (index, prediction) in enumerate(ranking, start=1)
            ],
        }
        if args.format == "csv":
            _write_csv(
                [
                    "rank",
                    "input",
                    "canonical_smiles",
                    "predicted_energy",
                    "delta_vs_best",
                    "connectivity_total",
                    "warnings",
                ],
                [
                    [
                        entry["rank"],
                        entry["input"],
                        entry["canonical_smiles"],
                        entry["predicted_energy"],
                        entry["delta_vs_best"],
                        entry["connectivity_total"],
                        ";".join(entry["warnings"]),
                    ]
                    for entry in payload["ranking"]
                ],
            )
            return EXIT_OK
        print(json.dumps(payload, indent=2))
        return EXIT_OK

    print(
        ranking_summary(
            ranking, labels=smiles, palette=palette, precision=args.precision
        )
    )
    return EXIT_OK


def command_card(args: argparse.Namespace) -> int:
    """Run the ``card`` subcommand.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Process exit status.
    :rtype: int
    """
    palette = _palette(args, sys.stdout)
    predictor = _load_predictor(args.model)
    if args.format == "json":
        from dataclasses import asdict

        payload = asdict(predictor.card)
        payload["model_sha256"] = predictor.model_sha256
        print(json.dumps(payload, indent=2, default=str))
        return EXIT_OK
    print(
        model_card_summary(
            predictor.card, palette=palette, model_sha256=predictor.model_sha256
        )
    )
    return EXIT_OK


def command_info(args: argparse.Namespace) -> int:
    """Run the ``info`` subcommand.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Process exit status.
    :rtype: int
    """
    palette = _palette(args, sys.stdout)
    lines = [
        palette(f"SynDE {_package_version()}", "bold", "cyan"),
        render_rule(72, palette=palette),
        "  Interpretable 2D prediction of protocol-defined GFN2-xTB energies.",
        "",
        "  " + palette("commands", "bold"),
        "    synde predict SMILES...   total energy across formulas",
        "    synde explain SMILES      full signed contribution breakdown",
        "    synde rank SMILES...      order constitutional isomers",
        "    synde card                model provenance and domain limits",
        "",
        "    synde completion zsh      shell completion script",
        "",
        "  Add --format json or --format csv to any scoring command.",
    ]
    print("\n".join(lines))
    return EXIT_OK


COMPLETIONS = {
    "bash": """# SynDE bash completion. Add to ~/.bashrc:  synde completion bash >> ~/.bashrc
_synde_complete() {
  local cur prev
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  if [ "$COMP_CWORD" -eq 1 ]; then
    COMPREPLY=($(compgen -W "predict explain rank card info completion" -- "$cur"))
    return
  fi
  case "$prev" in
    --format) COMPREPLY=($(compgen -W "table json csv" -- "$cur")); return ;;
    --color) COMPREPLY=($(compgen -W "auto always never" -- "$cur")); return ;;
    --input|--model) COMPREPLY=($(compgen -f -- "$cur")); return ;;
  esac
  COMPREPLY=($(compgen -W "--input --model --format --json --csv --jobs \
    --precision --explain --top --keep-going --color --no-color --help" -- "$cur"))
}
complete -F _synde_complete synde""",
    "zsh": """# SynDE zsh completion. Add to ~/.zshrc:  synde completion zsh >> ~/.zshrc
_synde() {
  local -a subcommands
  subcommands=(
    'predict:predict total energy for one or more structures'
    'explain:show the full contribution breakdown for a structure'
    'rank:order constitutional isomers by predicted energy'
    'card:print the active model card and domain limits'
    'info:show a short command overview'
    'completion:print a shell completion script'
  )
  if (( CURRENT == 2 )); then
    _describe 'synde command' subcommands
    return
  fi
  _arguments \
    '--input[read SMILES from FILE]:file:_files' \
    '--model[score with a saved artifact]:file:_files' \
    '--format[output format]:format:(table json csv)' \
    '--jobs[worker processes]:count:' \
    '--precision[decimal digits]:digits:' \
    '--top[connectivity terms shown]:count:' \
    '--color[colour output]:mode:(auto always never)' \
    '--json[shorthand for --format json]' \
    '--csv[shorthand for --format csv]' \
    '--explain[print the full breakdown]' \
    '--keep-going[skip out-of-domain inputs]' \
    '--no-color[disable colour]'
}
compdef _synde synde""",
    "fish": """# SynDE fish completion. Save as ~/.config/fish/completions/synde.fish
complete -c synde -f
complete -c synde -n __fish_use_subcommand -a predict -d 'predict total energy'
complete -c synde -n __fish_use_subcommand -a explain -d 'full contribution breakdown'
complete -c synde -n __fish_use_subcommand -a rank -d 'order constitutional isomers'
complete -c synde -n __fish_use_subcommand -a card -d 'model card and domain limits'
complete -c synde -n __fish_use_subcommand -a info -d 'short command overview'
complete -c synde -n __fish_use_subcommand -a completion -d 'shell completion script'
complete -c synde -l input -r -d 'read SMILES from FILE'
complete -c synde -l model -r -d 'score with a saved artifact'
complete -c synde -l format -x -a 'table json csv' -d 'output format'
complete -c synde -l jobs -x -d 'worker processes'
complete -c synde -l precision -x -d 'decimal digits'
complete -c synde -l top -x -d 'connectivity terms shown'
complete -c synde -l color -x -a 'auto always never' -d 'colour output'
complete -c synde -l json -d 'shorthand for --format json'
complete -c synde -l csv -d 'shorthand for --format csv'
complete -c synde -l explain -d 'print the full breakdown'
complete -c synde -l keep-going -d 'skip out-of-domain inputs'""",
}


def _write_csv(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Write rows to standard output as RFC 4180 CSV.

    :param headers: Column names written as the first record.
    :type headers: Sequence[str]
    :param rows: Row values, written in order.
    :type rows: Sequence[Sequence[Any]]
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    sys.stdout.write(buffer.getvalue())


def command_completion(args: argparse.Namespace) -> int:
    """Run the ``completion`` subcommand.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :return: Process exit status.
    :rtype: int
    """
    print(COMPLETIONS[args.shell])
    return EXIT_OK


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Attach options shared by every scoring subcommand.

    :param parser: Subcommand parser to extend.
    :type parser: argparse.ArgumentParser
    """
    parser.add_argument("smiles", nargs="*", help="SMILES strings to score")
    parser.add_argument(
        "--input",
        metavar="FILE",
        help="read SMILES from FILE, one per line ('-' reads standard input)",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        help="score with a saved artifact instead of the packaged model",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="output format (default: table)",
    )
    parser.add_argument(
        "--json",
        dest="format",
        action="store_const",
        const="json",
        help="shorthand for --format json",
    )
    parser.add_argument(
        "--csv",
        dest="format",
        action="store_const",
        const="csv",
        help="shorthand for --format csv",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="worker processes for scoring; 0 uses every core (default: 1)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        metavar="N",
        help="digits after the decimal point (default: 4)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the full command-line parser.

    :return: Parser covering every SynDE subcommand.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="synde",
        description=(
            "Predict protocol-defined GFN2-xTB molecular energies and rank "
            "constitutional isomers from 2D structure alone."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"synde {_package_version()}"
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="control ANSI colour output (default: auto)",
    )
    parser.add_argument(
        "--no-color",
        dest="color",
        action="store_const",
        const="never",
        help="disable ANSI colour output",
    )
    subparsers = parser.add_subparsers(dest="command")

    predict = subparsers.add_parser(
        "predict", help="predict total energy for one or more structures"
    )
    _add_common(predict)
    predict.add_argument(
        "--explain",
        action="store_true",
        help="print the full signed contribution breakdown for each input",
    )
    predict.add_argument(
        "--top",
        type=int,
        default=8,
        metavar="N",
        help="connectivity terms shown per structure (default: 8)",
    )
    predict.add_argument(
        "--keep-going",
        action="store_true",
        help="skip inputs outside the model domain instead of stopping",
    )
    predict.set_defaults(handler=command_predict)

    explain = subparsers.add_parser(
        "explain", help="show the full contribution breakdown for a structure"
    )
    _add_common(explain)
    explain.add_argument(
        "--top",
        type=int,
        default=12,
        metavar="N",
        help="connectivity terms shown per structure (default: 12)",
    )
    explain.add_argument(
        "--keep-going",
        action="store_true",
        help="skip inputs outside the model domain instead of stopping",
    )
    explain.set_defaults(handler=command_explain)

    rank = subparsers.add_parser(
        "rank", help="order constitutional isomers by predicted energy"
    )
    _add_common(rank)
    rank.set_defaults(handler=command_rank)

    card = subparsers.add_parser(
        "card", help="print the active model card and domain limits"
    )
    card.add_argument("--model", metavar="PATH", help="inspect a saved artifact")
    card.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="output format (default: table)",
    )
    card.add_argument(
        "--json",
        dest="format",
        action="store_const",
        const="json",
        help="shorthand for --format json",
    )
    card.set_defaults(handler=command_card)

    completion = subparsers.add_parser(
        "completion", help="print a shell completion script"
    )
    completion.add_argument("shell", choices=tuple(COMPLETIONS))
    completion.set_defaults(handler=command_completion, format="table")

    info = subparsers.add_parser("info", help="show a short command overview")
    info.set_defaults(format="table")
    info.set_defaults(handler=command_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SynDE command-line interface.

    :param argv: Argument vector; defaults to ``sys.argv[1:]``.
    :type argv: Sequence[str] | None
    :return: Process exit status.
    :rtype: int
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "handler", None) is None:
        parser.print_help()
        return EXIT_OK
    try:
        return int(args.handler(args))
    except SynDEError as error:
        palette = Palette.automatic(sys.stderr)
        print(f"{palette('error', 'red', 'bold')} {error}", file=sys.stderr)
        return EXIT_ERROR
    except (BrokenPipeError, KeyboardInterrupt):  # pragma: no cover - interactive
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
