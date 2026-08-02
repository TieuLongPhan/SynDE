from __future__ import annotations

import pytest

from synde.energy import (
    NAMED_FEATURE_SCHEMA,
    SparseInterpretableTwoDModelCard,
    SparseInterpretableTwoDScorer,
)
from synde.graph import GraphBuilder


def _model() -> SparseInterpretableTwoDScorer:
    card = SparseInterpretableTwoDModelCard(
        model_name="test-sparse-2d",
        target="test formula-relative label",
        units="relative_coordinate",
        feature_schema=NAMED_FEATURE_SCHEMA,
        feature_profile="test",
        training_groups=10,
        training_molecules=40,
        selection_method="test stability selection",
        selection_alpha=0.01,
        selection_l1_ratio=1.0,
        selection_folds=3,
        minimum_selection_frequency=3,
        refit_method="test ridge",
        refit_alpha=10.0,
        formula_centered_training=True,
        uses_coordinates=False,
        calibrated=True,
        fitted_coefficients=True,
    )
    return SparseInterpretableTwoDScorer(
        card=card,
        weights={"graph_zagreb_m1": 0.5},
        training_scale={"graph_zagreb_m1": 2.0},
    )


def test_sparse_v2_runtime_is_explicitly_calibrated_and_two_d() -> None:
    result = _model().score(GraphBuilder.from_smiles("CC(C)CC"))

    assert result.status == "success"
    assert result.score == pytest.approx(result.components["graph_zagreb_m1"])
    assert result.provenance["calibrated"] is True
    assert result.provenance["fitted_coefficients"] is True
    assert result.provenance["uses_coordinates"] is False
    assert result.provenance["uses_conformers"] is False
    assert result.descriptors["comparable_within_formula_only"] is True


def test_sparse_v2_rejects_schema_or_scale_mismatch() -> None:
    model = _model()
    wrong_card = SparseInterpretableTwoDModelCard(
        **{**model.card.__dict__, "feature_schema": "wrong"}
    )
    with pytest.raises(ValueError, match="Unsupported feature schema"):
        SparseInterpretableTwoDScorer(
            card=wrong_card,
            weights={"x": 1.0},
            training_scale={"x": 1.0},
        )
    with pytest.raises(ValueError, match="same terms"):
        SparseInterpretableTwoDScorer(
            card=model.card,
            weights={"x": 1.0},
            training_scale={"y": 1.0},
        )


def test_sparse_v2_group_scoring_checks_formula_and_distance() -> None:
    model = _model()
    model.applicability["centered_feature_distance_q99"] = 0.0
    same_formula = [
        GraphBuilder.from_smiles("CCCCC"),
        GraphBuilder.from_smiles("CC(C)CC"),
    ]

    results = model.score_group(same_formula)

    assert all(
        row.descriptors["centered_standardized_feature_distance"] is not None
        for row in results
    )
    assert any(row.status == "partial" for row in results)
    with pytest.raises(ValueError, match="one formula"):
        model.score_group(
            [
                GraphBuilder.from_smiles("CC"),
                GraphBuilder.from_smiles("CCC"),
            ]
        )
