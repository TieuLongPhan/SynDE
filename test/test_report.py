from __future__ import annotations

from pathlib import Path

import pytest

from synde.formatting import Palette
from synde.report import (
    WARNING_EXPLANATIONS,
    composition_formula,
    model_card_summary,
    prediction_headline,
    prediction_html,
    prediction_summary,
    ranking_html,
    ranking_summary,
    top_contributions,
)

MODEL_PATH = Path("synde/models/synde_energy_model.json")
requires_model = pytest.mark.skipif(
    not MODEL_PATH.is_file(), reason="packaged energy model is not present"
)


@pytest.fixture(scope="module")
def predictor():
    """Load the packaged predictor once for the rendering tests."""
    from synde.energy import SynDEEnergyPredictor

    return SynDEEnergyPredictor.load_default()


@pytest.mark.parametrize(
    "composition,expected",
    [
        ({"C": 2, "H": 6, "O": 1}, "C2H6O"),
        ({"C": 1, "H": 4}, "CH4"),
        ({"O": 2}, "O2"),
        ({"H": 2, "O": 1}, "H2O"),
        ({}, "-"),
        ({"C": 0}, "-"),
    ],
)
def test_composition_formula_uses_hill_order(
    composition: dict[str, int], expected: str
) -> None:
    assert composition_formula(composition) == expected


def test_warning_explanations_cover_the_predictor_codes() -> None:
    from synde.energy import SynDEEnergyPredictor

    assert SynDEEnergyPredictor.DISTANCE_WARNING in WARNING_EXPLANATIONS
    assert SynDEEnergyPredictor.COMPOSITION_WARNING in WARNING_EXPLANATIONS


@requires_model
def test_top_contributions_are_ordered_by_magnitude(predictor) -> None:
    prediction = predictor.predict_smiles("CC(=O)NC")
    terms = top_contributions(prediction, 5)
    assert len(terms) == 5
    magnitudes = [abs(value) for _, value in terms]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(abs(value) > 0 for _, value in terms)


@requires_model
def test_top_contributions_respects_a_zero_limit(predictor) -> None:
    assert top_contributions(predictor.predict_smiles("CCO"), 0) == []


@requires_model
def test_prediction_headline_is_one_line(predictor) -> None:
    headline = prediction_headline(predictor.predict_smiles("CCO"))
    assert "\n" not in headline
    assert "CCO" in headline
    assert "eV" in headline


@requires_model
def test_prediction_summary_reports_the_sum_identity(predictor) -> None:
    prediction = predictor.predict_smiles("CCO")
    text = prediction_summary(prediction, palette=Palette(False), width=100)
    assert "energy" in text
    assert "composition" in text
    assert "connectivity" in text
    assert "domain distance" in text
    assert "\033[" not in text


@requires_model
def test_prediction_summary_is_bounded_by_the_requested_width(predictor) -> None:
    prediction = predictor.predict_smiles("CC(=O)NC")
    text = prediction_summary(prediction, palette=Palette(False), width=90)
    assert all(len(line) <= 100 for line in text.splitlines())


@requires_model
def test_prediction_html_escapes_and_tabulates(predictor) -> None:
    html = prediction_html(predictor.predict_smiles("CCO"))
    assert html.startswith("<div")
    assert "<table" in html
    assert "<script" not in html


@requires_model
def test_ranking_summary_lists_candidates_lowest_first(predictor) -> None:
    ranking = predictor.rank_smiles(["CCCCC", "CC(C)CC", "CC(C)(C)C"])
    text = ranking_summary(ranking, palette=Palette(False), width=100)
    assert "C5H12" in text
    assert "Δ vs best" in text
    assert text.splitlines()[-1].strip().startswith("3")


def test_ranking_summary_handles_no_candidates() -> None:
    assert ranking_summary([]) == "no candidates"
    assert "no candidates" in ranking_html([])


@requires_model
def test_ranking_html_renders_a_table(predictor) -> None:
    ranking = predictor.rank_smiles(["CCCCC", "CC(C)(C)C"])
    assert "<table" in ranking_html(ranking)


@requires_model
def test_model_card_summary_states_the_domain(predictor) -> None:
    text = model_card_summary(
        predictor.card, palette=Palette(False), model_sha256="abc123"
    )
    assert "elements" in text
    assert "charges" in text
    assert "abc123" in text
    assert "2D graph only" in text
