from __future__ import annotations

import json
from pathlib import Path

import pytest

from synde.energy import (
    EnergyRefinementRecord,
    SynDEEnergyModelCard,
    SynDEEnergyPredictor,
    molecular_composition,
)
from synde.graph import GraphBuilder


def _card() -> SynDEEnergyModelCard:
    return SynDEEnergyModelCard(
        model_name="synde-energy-test-v1",
        schema_version=1,
        target="test total energy",
        units="eV",
        reference_protocol="test protocol",
        training_source="test.csv",
        training_source_sha256="abc",
        training_groups=2,
        training_molecules=4,
        evaluation_status="synthetic test only",
        formula_disjoint_evaluation=True,
        connectivity_disjoint_evaluation=True,
        composition_model="test element-count linear baseline",
        connectivity_model="test named SynDE residual",
        supported_elements=("H", "C", "O"),
        supported_formal_charges=(0,),
        uses_coordinates_at_inference=False,
        uses_conformers_at_inference=False,
        connectivity_equation_unchanged=True,
    )


def _predictor() -> SynDEEnergyPredictor:
    return SynDEEnergyPredictor(
        card=_card(),
        intercept=1.5,
        composition_weights={"H": -0.5, "C": -2.0, "O": -3.0},
        connectivity_weights={"v3_charge_product_r2": 0.25},
        feature_means={"v3_charge_product_r2": 0.0},
        feature_scales={"v3_charge_product_r2": 1.0},
        training_distance_q99=100.0,
        composition_ranges={"H": (0, 20), "C": (1, 10), "O": (0, 4)},
    )


def test_composition_counts_implicit_and_explicit_hydrogen_identically() -> None:
    implicit = GraphBuilder.from_smiles("CC")
    explicit = GraphBuilder.from_smiles("[H]C([H])([H])C([H])([H])[H]")

    assert molecular_composition(implicit) == {"C": 2, "H": 6}
    assert molecular_composition(explicit) == molecular_composition(implicit)


def test_prediction_is_cross_formula_and_contributions_sum() -> None:
    predictor = _predictor()
    outputs = predictor.predict_many(
        [GraphBuilder.from_smiles("CC"), GraphBuilder.from_smiles("CCO")]
    )

    assert len(outputs) == 2
    assert outputs[0].predicted_energy != outputs[1].predicted_energy
    for output in outputs:
        assert output.predicted_energy == pytest.approx(
            output.composition_total + output.connectivity_total
        )
        assert output.composition_total == pytest.approx(
            output.intercept_contribution
            + sum(output.composition_contributions.values())
        )
        assert output.connectivity_total == pytest.approx(
            sum(output.connectivity_contributions.values())
        )
        assert output.units == "eV"
        assert output.descriptors["cross_formula_comparable_under_same_target_protocol"]
        assert not output.descriptors["connectivity_subtotal_used_as_absolute_energy"]


def test_one_artifact_ranks_same_formula_by_connectivity() -> None:
    predictor = _predictor()
    molecules = [
        GraphBuilder.from_smiles("CCO"),
        GraphBuilder.from_smiles("COC"),
    ]

    outputs = predictor.predict_group(molecules)
    ranked = predictor.rank_group(molecules)

    assert outputs[0].composition_total == pytest.approx(outputs[1].composition_total)
    assert [index for index, _ in ranked] == sorted(
        range(2), key=lambda index: outputs[index].connectivity_total
    )

    with pytest.raises(ValueError, match="one formula/formal-charge group"):
        predictor.predict_group([GraphBuilder.from_smiles("CC"), molecules[0]])


def test_model_round_trip(tmp_path: Path) -> None:
    predictor = _predictor()
    path = tmp_path / "energy.json"
    path.write_text(json.dumps(predictor.to_dict()), encoding="utf-8")

    loaded = SynDEEnergyPredictor.load(path)

    assert loaded.card == predictor.card
    assert loaded.model_sha256 is not None
    assert loaded.predict(
        GraphBuilder.from_smiles("CCO")
    ).predicted_energy == pytest.approx(
        predictor.predict(GraphBuilder.from_smiles("CCO")).predicted_energy
    )


def test_external_refinement_returns_new_unvalidated_artifact(tmp_path: Path) -> None:
    predictor = _predictor()
    molecules = [
        GraphBuilder.from_smiles("CC"),
        GraphBuilder.from_smiles("CCC"),
        GraphBuilder.from_smiles("CCO"),
        GraphBuilder.from_smiles("COC"),
    ]
    records = [
        EnergyRefinementRecord(
            identifier=f"external-{index}",
            graph=molecule,
            target=(
                predictor.predict(molecule).predicted_energy
                + 0.75 * molecular_composition(molecule).get("C", 0)
            ),
            provenance={"source": "synthetic-test"},
        )
        for index, molecule in enumerate(molecules)
    ]

    refined, report = predictor.refine(
        records,
        dataset_name="external-test",
        alpha=0.01,
        refine_connectivity=False,
    )

    assert report.refined_metrics["rmse"] < report.baseline_metrics["rmse"]
    assert report.refined_blocks == ("intercept", "composition")
    assert len(report.dataset_sha256) == 64
    assert refined is not predictor
    assert refined.composition_weights != predictor.composition_weights
    assert refined.connectivity_weights == predictor.connectivity_weights
    assert not refined.card.formula_disjoint_evaluation
    assert not refined.card.connectivity_disjoint_evaluation
    assert "independent validation required" in refined.card.evaluation_status
    assert predictor.card.evaluation_status == "synthetic test only"

    path = tmp_path / "refined.json"
    refined.save(path)
    loaded = SynDEEnergyPredictor.load(path)
    assert loaded.card == refined.card
    assert loaded.refinement_report["dataset_name"] == "external-test"
    assert loaded.refinement_report["dataset_sha256"] == report.dataset_sha256
    assert loaded.predict(molecules[0]).predicted_energy == pytest.approx(
        refined.predict(molecules[0]).predicted_energy
    )


def test_external_refinement_validates_records() -> None:
    predictor = _predictor()
    record = EnergyRefinementRecord.from_smiles("one", "CC", -1.0)

    with pytest.raises(ValueError, match="At least two"):
        predictor.refine([record], dataset_name="external-test")
    with pytest.raises(ValueError, match="dataset name"):
        predictor.refine(
            [record, EnergyRefinementRecord.from_smiles("two", "CCC", -2.0)],
            dataset_name="",
        )


def test_domain_guards_and_warnings() -> None:
    predictor = _predictor()

    with pytest.raises(ValueError, match="one connected molecule"):
        predictor.predict(GraphBuilder.from_smiles("CC.O"))
    with pytest.raises(ValueError, match="Unsupported formal charge"):
        predictor.predict(GraphBuilder.from_smiles("C[OH2+]"))

    outside = predictor.predict(GraphBuilder.from_smiles("CCCCCCCCCCC"))
    assert predictor.COMPOSITION_WARNING in outside.warnings


def test_rejects_artifact_that_claims_coordinate_inference() -> None:
    invalid = SynDEEnergyModelCard(
        **(_card().__dict__ | {"uses_coordinates_at_inference": True})
    )
    with pytest.raises(ValueError, match="remain 2D"):
        SynDEEnergyPredictor(
            card=invalid,
            intercept=0.0,
            composition_weights={"C": -1.0},
            connectivity_weights={"x": 1.0},
            feature_means={"x": 0.0},
            feature_scales={"x": 1.0},
            training_distance_q99=1.0,
            composition_ranges={"C": (1, 10)},
        )


def test_default_cross_formula_model_is_current_or_explicitly_unavailable() -> None:
    model_path = Path("synde/models/synde_energy_model.json")
    if not model_path.is_file():
        with pytest.raises(FileNotFoundError, match="externally validated"):
            SynDEEnergyPredictor.load_default()
        return

    predictor = SynDEEnergyPredictor.load_default()
    outputs = predictor.predict_many(
        [GraphBuilder.from_smiles("CCO"), GraphBuilder.from_smiles("CCCC")]
    )

    assert predictor.card.units == "eV"
    assert predictor.card.training_groups == 11993
    assert len(predictor.connectivity_weights) == 633
    assert outputs[0].predicted_energy != outputs[1].predicted_energy
    for output in outputs:
        assert output.predicted_energy == pytest.approx(
            output.composition_total + output.connectivity_total
        )
        assert "external validation" in output.provenance["evaluation_status"]


def test_prediction_repr_and_summary_describe_the_result() -> None:
    output = _predictor().predict(GraphBuilder.from_smiles("CCO"))

    assert repr(output).startswith("SynDEEnergyPrediction(")
    assert "CCO" in repr(output)
    summary = output.summary(color=False)
    assert "energy" in summary
    assert "connectivity" in summary
    assert "\033[" not in summary
    assert "<table" in output._repr_html_()


def test_prediction_exposes_convenience_accessors() -> None:
    output = _predictor().predict(GraphBuilder.from_smiles("CCO"))

    assert output.canonical_smiles == "CCO"
    assert output.composition == {"C": 2, "H": 6, "O": 1}
    assert output.top_contributions(1) == output.top_contributions(5)[:1]


def test_predictor_repr_and_summary_expose_provenance() -> None:
    predictor = _predictor()

    assert "synde-energy-test-v1" in repr(predictor)
    assert "connectivity_terms=1" in repr(predictor)
    summary = predictor.summary(color=False)
    assert "synde-energy-test-v1" in summary
    assert "elements" in summary
    assert "SynDEEnergyPredictor" in predictor._repr_html_()


def test_model_card_repr_and_summary() -> None:
    card = _card()

    assert "synde-energy-test-v1" in repr(card)
    assert "elements" in card.summary(color=False)


def test_smiles_convenience_matches_the_graph_interface() -> None:
    predictor = _predictor()

    direct = predictor.predict(GraphBuilder.from_smiles("CCO"))
    convenience = predictor.predict_smiles("CCO")
    assert convenience.predicted_energy == pytest.approx(direct.predicted_energy)

    batch = predictor.predict_many_smiles(["CCO", "CC"])
    assert [item.canonical_smiles for item in batch] == ["CCO", "CC"]


def test_ranking_keeps_the_tuple_contract_and_renders() -> None:
    predictor = _predictor()
    ranking = predictor.rank_smiles(["CCO", "COC"])

    assert isinstance(ranking, list)
    first_index, first_prediction = ranking[0]
    assert isinstance(first_index, int)
    assert first_prediction.predicted_energy <= ranking[1][1].predicted_energy
    assert "COC" in repr(ranking) or "CCO" in repr(ranking)
    assert "Δ vs best" in ranking.summary(color=False)
    assert "<table" in ranking._repr_html_()


def test_domain_errors_carry_input_rule_and_next_step() -> None:
    from synde.errors import SynDEDomainError

    predictor = _predictor()

    with pytest.raises(SynDEDomainError) as fragments:
        predictor.predict(GraphBuilder.from_smiles("CC.O"))
    message = str(fragments.value)
    assert "one connected molecule" in message
    assert "2 disconnected fragments" in message
    assert "Hint:" in message
    assert fragments.value.details["components"] == 2

    with pytest.raises(SynDEDomainError) as charge:
        predictor.predict(GraphBuilder.from_smiles("C[OH2+]"))
    assert "Unsupported formal charge" in str(charge.value)
    assert charge.value.details["formal_charge"] == 1

    with pytest.raises(SynDEDomainError) as elements:
        predictor.predict(GraphBuilder.from_smiles("CCN"))
    assert elements.value.details["unsupported_elements"] == ["N"]

    with pytest.raises(SynDEDomainError) as radical:
        predictor.predict(GraphBuilder.from_smiles("C[CH2]"))
    assert "closed-shell" in str(radical.value)


def test_group_mismatch_names_the_observed_formulas() -> None:
    from synde.errors import SynDEDomainError

    with pytest.raises(SynDEDomainError) as excinfo:
        _predictor().rank_smiles(["CCO", "CCCO"])
    message = str(excinfo.value)
    assert "one formula/formal-charge group" in message
    assert "C2H6O" in message
    assert "C3H8O" in message
    assert len(excinfo.value.details["observed_groups"]) == 2


def test_domain_description_lists_the_supported_chemistry() -> None:
    description = _predictor().domain_description()

    assert "C" in description
    assert "closed-shell" in description
