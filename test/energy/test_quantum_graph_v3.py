from __future__ import annotations

import json

import pytest

from synde.energy import QuantumGraphV3ModelCard, QuantumGraphV3Scorer
from synde.graph import GraphBuilder


def _card() -> QuantumGraphV3ModelCard:
    return QuantumGraphV3ModelCard(
        model_name="test-v3",
        status="development_only_nonconfirmatory",
        target="test relative label",
        feature_definition_sha256="test",
        training_groups=1,
        training_molecules=2,
        admitted_v3_families=("v3_charge_topology",),
        selection_method="test",
        selection_alpha=0.01,
        refit_alpha=10.0,
        formula_centered=True,
        uses_coordinates=False,
        spent_v2_holdout_loaded=False,
        external_validation_complete=False,
    )


def test_quantum_graph_v3_score_is_deterministic_and_attributable() -> None:
    scorer = QuantumGraphV3Scorer(
        card=_card(),
        weights={
            "v3_charge_product_r2": -0.5,
            "v3_huckel_population_variance": 0.25,
        },
    )
    molecule = GraphBuilder.from_smiles("CC(=O)NC")
    first = scorer.score(molecule)
    second = scorer.score(molecule)

    assert first.score == pytest.approx(second.score)
    assert first.score == pytest.approx(sum(first.components.values()))
    assert first.units == "reference_label_relative_coordinate"
    assert first.warnings == ("V3_DEVELOPMENT_ONLY_NONCONFIRMATORY",)
    assert first.provenance["uses_coordinates"] is False


def test_quantum_graph_v3_model_round_trip(tmp_path) -> None:
    scorer = QuantumGraphV3Scorer(card=_card(), weights={"v3_charge_product_r2": -0.5})
    path = tmp_path / "model.json"
    path.write_text(json.dumps(scorer.to_dict()), encoding="utf-8")

    loaded = QuantumGraphV3Scorer.load(path)

    assert loaded.card == scorer.card
    assert loaded.weights == scorer.weights


def test_quantum_graph_v3_group_scoring_rejects_mixed_formulae() -> None:
    scorer = QuantumGraphV3Scorer(card=_card(), weights={"v3_charge_product_r2": -0.5})
    candidates = [
        GraphBuilder.from_smiles("CCC"),
        GraphBuilder.from_smiles("CCCC"),
    ]

    with pytest.raises(ValueError, match="one formula"):
        scorer.score_group(candidates)


def test_quantum_graph_v3_rejects_coordinate_or_holdout_cards() -> None:
    payload = _card().__dict__ | {"uses_coordinates": True}
    with pytest.raises(ValueError, match="coordinate-free"):
        QuantumGraphV3Scorer(
            card=QuantumGraphV3ModelCard(**payload),
            weights={"v3_charge_product_r2": 1.0},
        )

    payload = _card().__dict__ | {"spent_v2_holdout_loaded": True}
    with pytest.raises(ValueError, match="spent holdout"):
        QuantumGraphV3Scorer(
            card=QuantumGraphV3ModelCard(**payload),
            weights={"v3_charge_product_r2": 1.0},
        )
