from __future__ import annotations

import json
from pathlib import Path

import pytest

from synde.energy import (
    FROZEN_MODEL_SHA256,
    SynDEModelCard,
    SynDEScorer,
    SynDEValidationRecord,
)
from synde.graph import GraphBuilder


def _card() -> SynDEModelCard:
    return SynDEModelCard(
        model_name="test-calibration-artifact",
        status_at_freeze="development_only_no_untouched_test",
        target="test GFN2-xTB relative target",
        rdkit_generation_version="2025.9.6",
        xtb_generation_version="6.7.1",
        feature_definition_sha256="test",
        training_source="test.csv",
        training_source_sha256="abc",
        training_groups=2,
        training_molecules=6,
        selected_profile="expanded_stable_recalibration",
        selection_method="test",
        selection_alpha=0.001,
        refit_alpha=1.0,
        formula_centered=True,
        uses_coordinates=False,
        training_labels_previously_used_for_prior_model_evaluation=True,
        external_validation_complete_at_freeze=False,
    )


def _validation(model_sha256: str) -> SynDEValidationRecord:
    return SynDEValidationRecord(
        protocol="test-frozen-external",
        executed_on="2026-07-31",
        model_sha256=model_sha256,
        external_formula_and_graph_disjoint=True,
        external_groups=10,
        external_molecules=40,
        group_scoreability=1.0,
        mean_group_pearson=0.94,
        mean_group_spearman=0.89,
        mean_pairwise_concordance=0.92,
        top1_accuracy=0.85,
        external_labels_used_for_fit_or_tuning=False,
        post_external_model_revision=False,
        all_prespecified_external_performance_criteria_met=True,
        claim_boundary="test boundary",
    )


def _scorer(q99: float = 10.0) -> SynDEScorer:
    return SynDEScorer(
        card=_card(),
        weights={"v3_charge_product_r2": -0.5},
        feature_scales={"v3_charge_product_r2": 0.25},
        training_distance_q99=q99,
    )


def test_synde_score_is_deterministic_and_attributable() -> None:
    molecule = GraphBuilder.from_smiles("CC(=O)NC")
    first = _scorer().score(molecule)
    second = _scorer().score(molecule)

    assert first.score == pytest.approx(second.score)
    assert first.score == pytest.approx(sum(first.components.values()))
    assert first.units == "GFN2-xTB_formula_relative_model_coordinate"
    assert first.warnings == (_scorer().CALIBRATION_ONLY_WARNING,)
    assert first.provenance["external_validation_loaded"] is False


def test_synde_group_distance_and_formula_guard() -> None:
    group = [
        GraphBuilder.from_smiles("CCCO"),
        GraphBuilder.from_smiles("CC(C)O"),
        GraphBuilder.from_smiles("COCC"),
    ]
    results = _scorer(q99=1e-9).score_group(group)

    assert len(results) == 3
    assert all(
        "centered_selected_feature_distance" in row.descriptors for row in results
    )
    assert any(_scorer().DISTANCE_WARNING in row.warnings for row in results)

    with pytest.raises(ValueError, match="one formula"):
        _scorer().score_group(
            [GraphBuilder.from_smiles("CCC"), GraphBuilder.from_smiles("CCCC")]
        )


def test_synde_model_round_trip_without_validation(tmp_path: Path) -> None:
    scorer = _scorer()
    path = tmp_path / "model.json"
    path.write_text(json.dumps(scorer.to_dict()), encoding="utf-8")

    loaded = SynDEScorer.load(path)

    assert loaded.card == scorer.card
    assert loaded.weights == scorer.weights
    assert loaded.feature_scales == scorer.feature_scales
    assert loaded.externally_validated is False


def test_synde_rejects_coordinate_cards_and_mismatched_validation() -> None:
    with pytest.raises(ValueError, match="coordinate-free"):
        SynDEScorer(
            card=SynDEModelCard(**(_card().__dict__ | {"uses_coordinates": True})),
            weights={"x": 1.0},
            feature_scales={"x": 1.0},
            training_distance_q99=1.0,
        )

    with pytest.raises(ValueError, match="does not match"):
        SynDEScorer(
            card=_card(),
            weights={"x": 1.0},
            feature_scales={"x": 1.0},
            training_distance_q99=1.0,
            validation=_validation(FROZEN_MODEL_SHA256),
            model_sha256="wrong",
        )


def test_default_frozen_model_loads_as_externally_validated() -> None:
    scorer = SynDEScorer.load_default()
    group = [
        GraphBuilder.from_smiles("CCCCC"),
        GraphBuilder.from_smiles("CC(C)CC"),
        GraphBuilder.from_smiles("CC(C)(C)C"),
    ]

    results = scorer.score_group(group)

    assert scorer.card.training_groups == 11993
    assert scorer.model_sha256 == FROZEN_MODEL_SHA256
    assert scorer.externally_validated is True
    assert scorer.validation is not None
    assert scorer.validation.external_groups == 3005
    assert scorer.validation.external_molecules == 19940
    assert not scorer.validation.all_prespecified_external_performance_criteria_met
    assert len(scorer.weights) == 633
    assert len(results) == 3
    assert all(scorer.CALIBRATION_ONLY_WARNING not in row.warnings for row in results)
    assert all(row.descriptors["externally_validated"] for row in results)
    assert all(row.status == "success" for row in results)
    assert all(
        row.provenance["all_prespecified_external_performance_criteria_met"] is False
        for row in results
    )
