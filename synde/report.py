"""Human-readable renderings of SynDE predictions, rankings, and model cards.

This module is the single presentation layer for the package.  The command
line interface, the interactive ``__repr__`` implementations, and the Jupyter
``_repr_html_`` hooks all render through the functions defined here, so terminal
and notebook output stay consistent.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from .formatting import (
    Palette,
    format_float,
    render_fields,
    render_rule,
    render_table,
    terminal_width,
    truncate,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .energy.energy_predictor import (
        SynDEEnergyModelCard,
        SynDEEnergyPrediction,
    )

__all__ = [
    "WARNING_EXPLANATIONS",
    "top_contributions",
    "composition_formula",
    "prediction_headline",
    "prediction_summary",
    "prediction_html",
    "ranking_summary",
    "ranking_html",
    "model_card_summary",
]

WARNING_EXPLANATIONS = {
    "SYNDE_ENERGY_OUTSIDE_TRAINING_Q99_FEATURE_DISTANCE": (
        "descriptor vector is farther from the training centre than 99% of the "
        "training cohort; treat the value as an extrapolation"
    ),
    "SYNDE_ENERGY_OUTSIDE_TRAINING_COMPOSITION_RANGE": (
        "one or more element counts fall outside the fitted composition range"
    ),
}


def top_contributions(
    prediction: "SynDEEnergyPrediction",
    limit: int = 8,
) -> list[tuple[str, float]]:
    """Return the largest-magnitude nonzero connectivity terms.

    :param prediction: Prediction whose connectivity block is inspected.
    :type prediction: SynDEEnergyPrediction
    :param limit: Maximum number of terms returned.
    :type limit: int
    :return: Term name and signed contribution pairs, largest magnitude first.
    :rtype: list[tuple[str, float]]
    """
    nonzero = [
        (name, float(value))
        for name, value in prediction.connectivity_contributions.items()
        if abs(float(value)) > 1e-12
    ]
    nonzero.sort(key=lambda item: (-abs(item[1]), item[0]))
    return nonzero[: max(0, int(limit))]


def composition_formula(composition: dict[str, int]) -> str:
    """Render an element-count mapping in Hill notation.

    :param composition: Element symbol to count mapping.
    :type composition: dict[str, int]
    :return: Hill-ordered molecular formula, or ``"-"`` when empty.
    :rtype: str
    """
    counts = {name: int(value) for name, value in composition.items() if int(value)}
    if not counts:
        return "-"
    ordered: list[str] = []
    for symbol in ("C", "H"):
        if symbol in counts:
            ordered.append(symbol)
    ordered.extend(sorted(name for name in counts if name not in {"C", "H"}))
    return "".join(
        name if counts[name] == 1 else f"{name}{counts[name]}" for name in ordered
    )


def _identity(prediction: "SynDEEnergyPrediction") -> str:
    """Return the best available display label for one prediction.

    :param prediction: Prediction to label.
    :type prediction: SynDEEnergyPrediction
    :return: Canonical SMILES when known, otherwise the graph identity.
    :rtype: str
    """
    descriptors = prediction.descriptors
    return str(
        descriptors.get("canonical_smiles")
        or descriptors.get("graph_identity")
        or "<graph>"
    )


def prediction_headline(
    prediction: "SynDEEnergyPrediction",
    *,
    precision: int = 4,
) -> str:
    """Render a single compact line describing one prediction.

    :param prediction: Prediction to describe.
    :type prediction: SynDEEnergyPrediction
    :param precision: Digits kept after the decimal point.
    :type precision: int
    :return: One-line summary suitable for ``__repr__``.
    :rtype: str
    """
    flag = f", warnings={len(prediction.warnings)}" if prediction.warnings else ""
    return (
        f"{_identity(prediction)} "
        f"{format_float(prediction.predicted_energy, precision)} "
        f"{prediction.units}{flag}"
    )


def _warning_lines(
    prediction: "SynDEEnergyPrediction",
    palette: Palette,
) -> list[str]:
    """Render explained applicability warnings for one prediction.

    :param prediction: Prediction whose warnings are rendered.
    :type prediction: SynDEEnergyPrediction
    :param palette: Styling switch for the warning marker.
    :type palette: Palette
    :return: Zero or more rendered warning lines.
    :rtype: list[str]
    """
    lines: list[str] = []
    for code in prediction.warnings:
        explanation = WARNING_EXPLANATIONS.get(code, "")
        marker = palette("warning", "yellow", "bold")
        lines.append(f"  {marker}  {code}")
        if explanation:
            lines.append(f"           {explanation}")
    return lines


def prediction_summary(
    prediction: "SynDEEnergyPrediction",
    *,
    palette: Palette | None = None,
    precision: int = 4,
    top: int = 8,
    width: int | None = None,
) -> str:
    """Render the full auditable breakdown of one prediction.

    :param prediction: Prediction to render.
    :type prediction: SynDEEnergyPrediction
    :param palette: Styling switch applied to headings and markers.
    :type palette: Palette | None
    :param precision: Digits kept after the decimal point.
    :type precision: int
    :param top: Number of connectivity terms listed.
    :type top: int
    :param width: Rendering width; defaults to the terminal width.
    :type width: int | None
    :return: Multi-line report without a trailing newline.
    :rtype: str
    """
    palette = palette or Palette(False)
    width = width or terminal_width()
    descriptors = prediction.descriptors
    units = prediction.units
    blocks: list[str] = []

    formula = composition_formula(descriptors.get("composition", {}))
    title = palette(_identity(prediction), "bold", "cyan")
    blocks.append(f"{title}   {palette(formula, 'dim')}")
    blocks.append(render_rule(width, palette=palette))

    energy = palette(
        f"{format_float(prediction.predicted_energy, precision)} {units}",
        "bold",
    )
    blocks.append(
        render_fields(
            [
                ("energy", energy),
                (
                    "composition",
                    f"{format_float(prediction.composition_total, precision)} {units}"
                    f"   (intercept "
                    f"{format_float(prediction.intercept_contribution, precision)})",
                ),
                (
                    "connectivity",
                    f"{format_float(prediction.connectivity_total, precision)} {units}",
                ),
                ("status", str(prediction.status)),
            ],
            palette=palette,
            indent="  ",
        )
    )

    terms = top_contributions(prediction, top)
    if terms:
        blocks.append("")
        selected = int(descriptors.get("selected_connectivity_terms", 0))
        nonzero = sum(
            1
            for value in prediction.connectivity_contributions.values()
            if abs(float(value)) > 1e-12
        )
        heading = (
            f"top connectivity terms  ({len(terms)} of {nonzero} active, "
            f"{selected} in model)"
        )
        blocks.append("  " + palette(heading, "bold"))
        name_width = max(24, min(56, width - 24))
        blocks.append(
            render_table(
                ["term", f"contribution ({units})"],
                [
                    [
                        truncate(name, name_width),
                        format_float(value, precision, signed=True),
                    ]
                    for name, value in terms
                ],
                aligns=["l", "r"],
                palette=palette,
                indent="  ",
            )
        )

    distance = float(descriptors.get("selected_feature_distance", 0.0))
    threshold = float(descriptors.get("training_distance_q99", 0.0))
    blocks.append("")
    blocks.append(
        render_fields(
            [
                (
                    "domain distance",
                    f"{format_float(distance, 3)} / q99 {format_float(threshold, 3)}",
                ),
                ("model", str(prediction.provenance.get("model_name", "-"))),
                (
                    "protocol",
                    truncate(
                        str(prediction.provenance.get("reference_protocol", "-")),
                        max(20, width - 20),
                    ),
                ),
            ],
            palette=palette,
            indent="  ",
        )
    )

    warnings = _warning_lines(prediction, palette)
    if warnings:
        blocks.append("")
        blocks.extend(warnings)
    return "\n".join(blocks)


def _html_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a minimal HTML table with inline styling.

    :param headers: Column titles.
    :type headers: Sequence[str]
    :param rows: Row cells rendered with ``str``.
    :type rows: Iterable[Sequence[Any]]
    :return: HTML table markup.
    :rtype: str
    """
    head = "".join(
        f'<th style="text-align:left;padding:2px 10px 2px 0">{escape(str(name))}</th>'
        for name in headers
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td style="padding:2px 10px 2px 0;font-family:monospace">'
            f"{escape(str(cell))}</td>"
            for cell in row
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<table style="border-collapse:collapse;font-size:0.9em">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def prediction_html(
    prediction: "SynDEEnergyPrediction",
    *,
    precision: int = 4,
    top: int = 8,
) -> str:
    """Render one prediction as notebook-friendly HTML.

    :param prediction: Prediction to render.
    :type prediction: SynDEEnergyPrediction
    :param precision: Digits kept after the decimal point.
    :type precision: int
    :param top: Number of connectivity terms listed.
    :type top: int
    :return: HTML fragment for ``_repr_html_``.
    :rtype: str
    """
    descriptors = prediction.descriptors
    units = escape(str(prediction.units))
    rows = [
        ["energy", f"{format_float(prediction.predicted_energy, precision)} {units}"],
        [
            "composition",
            f"{format_float(prediction.composition_total, precision)} {units}",
        ],
        [
            "connectivity",
            f"{format_float(prediction.connectivity_total, precision)} {units}",
        ],
        ["status", str(prediction.status)],
    ]
    parts = [
        f'<div style="font-family:system-ui,sans-serif">'
        f"<strong>{escape(_identity(prediction))}</strong> "
        f'<span style="opacity:.6">'
        f"{escape(composition_formula(descriptors.get('composition', {})))}</span>",
        _html_table(["field", "value"], rows),
    ]
    terms = top_contributions(prediction, top)
    if terms:
        parts.append(
            '<div style="margin-top:6px;opacity:.7">top connectivity terms' "</div>"
        )
        parts.append(
            _html_table(
                ["term", f"contribution ({units})"],
                [[name, format_float(value, precision)] for name, value in terms],
            )
        )
    for code in prediction.warnings:
        parts.append(
            '<div style="margin-top:6px;color:#a15c00">⚠ '
            f"{escape(code)} — {escape(WARNING_EXPLANATIONS.get(code, ''))}</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def ranking_summary(
    entries: Sequence[tuple[int, "SynDEEnergyPrediction"]],
    *,
    labels: Sequence[str] | None = None,
    palette: Palette | None = None,
    precision: int = 4,
    width: int | None = None,
) -> str:
    """Render an ordered same-formula ranking table.

    :param entries: Input index and prediction pairs, already ordered.
    :type entries: Sequence[tuple[int, SynDEEnergyPrediction]]
    :param labels: Display labels indexed by original input position.
    :type labels: Sequence[str] | None
    :param palette: Styling switch applied to headings.
    :type palette: Palette | None
    :param precision: Digits kept after the decimal point.
    :type precision: int
    :param width: Rendering width; defaults to the terminal width.
    :type width: int | None
    :return: Multi-line ranking report without a trailing newline.
    :rtype: str
    """
    palette = palette or Palette(False)
    width = width or terminal_width()
    if not entries:
        return "no candidates"
    best = entries[0][1].predicted_energy
    units = entries[0][1].units
    formula = composition_formula(entries[0][1].descriptors.get("composition", {}))
    rows = []
    for position, (index, prediction) in enumerate(entries, start=1):
        label = labels[index] if labels is not None and index < len(labels) else None
        rows.append(
            [
                str(position),
                label or _identity(prediction),
                format_float(prediction.predicted_energy, precision),
                format_float(
                    prediction.predicted_energy - best, precision, signed=True
                ),
                format_float(prediction.connectivity_total, precision),
                "!" if prediction.warnings else "",
            ]
        )
    header = (
        f"{palette(formula, 'bold', 'cyan')}   "
        f"{palette(f'{len(entries)} candidates, lowest predicted energy first', 'dim')}"
    )
    label_width = max(16, min(48, width - 52))
    table = render_table(
        ["#", "structure", f"energy ({units})", "Δ vs best", "connectivity", ""],
        rows,
        aligns=["r", "l", "r", "r", "r", "l"],
        max_widths=[0, label_width, 0, 0, 0, 0],
        palette=palette,
        indent="  ",
    )
    return "\n".join([header, render_rule(width, palette=palette), table])


def ranking_html(
    entries: Sequence[tuple[int, "SynDEEnergyPrediction"]],
    *,
    labels: Sequence[str] | None = None,
    precision: int = 4,
) -> str:
    """Render an ordered ranking as notebook-friendly HTML.

    :param entries: Input index and prediction pairs, already ordered.
    :type entries: Sequence[tuple[int, SynDEEnergyPrediction]]
    :param labels: Display labels indexed by original input position.
    :type labels: Sequence[str] | None
    :param precision: Digits kept after the decimal point.
    :type precision: int
    :return: HTML fragment showing the ranking table.
    :rtype: str
    """
    if not entries:
        return "<div>no candidates</div>"
    best = entries[0][1].predicted_energy
    units = escape(str(entries[0][1].units))
    rows = []
    for position, (index, prediction) in enumerate(entries, start=1):
        label = labels[index] if labels is not None and index < len(labels) else None
        rows.append(
            [
                position,
                label or _identity(prediction),
                format_float(prediction.predicted_energy, precision),
                format_float(
                    prediction.predicted_energy - best, precision, signed=True
                ),
            ]
        )
    return _html_table(["#", "structure", f"energy ({units})", "Δ vs best"], rows)


def model_card_summary(
    card: "SynDEEnergyModelCard",
    *,
    palette: Palette | None = None,
    model_sha256: str | None = None,
    width: int | None = None,
) -> str:
    """Render the provenance and applicability boundary of one artifact.

    :param card: Model card describing the active artifact.
    :type card: SynDEEnergyModelCard
    :param palette: Styling switch applied to headings.
    :type palette: Palette | None
    :param model_sha256: Digest of the loaded artifact, when available.
    :type model_sha256: str | None
    :param width: Rendering width; defaults to the terminal width.
    :type width: int | None
    :return: Multi-line model-card report without a trailing newline.
    :rtype: str
    """
    palette = palette or Palette(False)
    width = width or terminal_width()
    fields = [
        ("model", card.model_name),
        ("target", card.target),
        ("units", card.units),
        ("protocol", card.reference_protocol),
        ("evaluation", card.evaluation_status),
        ("composition", card.composition_model),
        ("connectivity", card.connectivity_model),
        ("elements", " ".join(card.supported_elements)),
        ("charges", ", ".join(str(value) for value in card.supported_formal_charges)),
        (
            "training",
            f"{card.training_molecules} molecules in {card.training_groups} groups",
        ),
        ("source", card.training_source),
        (
            "inference",
            (
                "2D graph only (no coordinates, no conformers)"
                if not card.uses_coordinates_at_inference
                else "requires coordinates"
            ),
        ),
    ]
    if model_sha256:
        fields.append(("sha256", model_sha256))
    return "\n".join(
        [
            palette("SynDE energy model card", "bold", "cyan"),
            render_rule(width, palette=palette),
            render_fields(fields, palette=palette, indent="  "),
        ]
    )
