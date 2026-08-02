import json
import unittest
from dataclasses import replace

from synde.energy import (
    FormulaCalibrationRecord,
    FormulaRelativeRidgeCalibrator,
    molecular_formula_charge_key,
)
from synde.graph import GraphBuilder


def record(identifier, group, x, target):
    return FormulaCalibrationRecord(
        identifier, group, {"x": x, "constant": 4.0}, target, {}
    )


class TestFormulaRelativeRidgeCalibrator(unittest.TestCase):
    def test_fits_differences_within_formula_not_group_offsets(self):
        rows = [
            record("a1", "A", 0.0, 100.0),
            record("a2", "A", 1.0, 102.0),
            record("a3", "A", 2.0, 104.0),
            record("b1", "B", 0.0, -50.0),
            record("b2", "B", 1.0, -48.0),
            record("b3", "B", 2.0, -46.0),
        ]
        model = FormulaRelativeRidgeCalibrator(alpha=1e-10, units="arb")
        card = model.fit(rows)

        low = model.predict({"x": 0.0, "constant": 400.0})
        high = model.predict({"x": 2.0, "constant": -400.0})

        self.assertAlmostEqual(high.value - low.value, 4.0, places=6)
        self.assertTrue(high.comparable_within_formula_only)
        self.assertGreaterEqual(high.applicability, 0.0)
        self.assertLessEqual(high.applicability, 1.0)
        self.assertEqual(card.training_groups, 2)
        self.assertEqual(card.feature_schema, "synde-2d-features-v2")
        self.assertEqual(card.feature_profile, "custom")
        self.assertEqual(model.weights[model.feature_names.index("constant")], 0.0)

        far = model.predict({"x": 100.0})
        near = model.predict({"x": 1.0})
        self.assertGreater(far.feature_distance, near.feature_distance)
        self.assertLess(far.applicability, near.applicability)

    def test_save_and_load_round_trip(self):
        model = FormulaRelativeRidgeCalibrator(alpha=0.1, target="label", units="u")
        model.fit([record("a1", "A", 0.0, 1.0), record("a2", "A", 1.0, 3.0)])

        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = FormulaRelativeRidgeCalibrator.load(path)
            self.assertAlmostEqual(
                loaded.predict({"x": 0.75}).value,
                model.predict({"x": 0.75}).value,
            )
            self.assertEqual(json.loads(path.read_text())["card"]["units"], "u")

    def test_group_predictions_are_zero_centered(self):
        model = FormulaRelativeRidgeCalibrator(alpha=1e-10)
        model.fit([record("a1", "A", 0.0, 1.0), record("a2", "A", 1.0, 3.0)])
        predictions = model.predict_group([{"x": 1.0}, {"x": 2.0}, {"x": 4.0}])
        self.assertAlmostEqual(sum(row.value for row in predictions), 0.0)
        self.assertLess(predictions[0].value, predictions[-1].value)

        symmetric = model.predict_group([{"x": 0.0}, {"x": 2.0}])
        self.assertAlmostEqual(
            symmetric[0].feature_distance, symmetric[1].feature_distance
        )

    def test_predicts_directly_from_normalized_graph(self):
        graph_a = GraphBuilder.from_smiles("CCO")
        graph_b = GraphBuilder.from_smiles("COC")
        from synde.energy import MoleculeScorer, extract_two_d_features

        scorer = MoleculeScorer()
        records = []
        for identifier, graph, target in (("a", graph_a, 0.0), ("b", graph_b, 1.0)):
            result = scorer.score(graph)
            records.append(
                FormulaCalibrationRecord(
                    identifier,
                    "C2H6O|charge=0",
                    extract_two_d_features(graph, result),
                    target,
                    {},
                )
            )
        model = FormulaRelativeRidgeCalibrator(alpha=0.1)
        model.fit(records)
        prediction = model.predict_graph(graph_b, scorer)
        self.assertGreater(prediction.value, model.predict_graph(graph_a, scorer).value)

        group = model.predict_graph_group([graph_a, graph_b], scorer)
        self.assertAlmostEqual(sum(row.value for row in group), 0.0)

    def test_graph_group_rejects_different_formulas(self):
        graph_a = GraphBuilder.from_smiles("CCO")
        graph_b = GraphBuilder.from_smiles("CCC")
        model = FormulaRelativeRidgeCalibrator(alpha=0.1)
        model.fit([record("a1", "A", 0.0, 1.0), record("a2", "A", 1.0, 2.0)])

        with self.assertRaisesRegex(ValueError, "one molecular formula"):
            model.predict_graph_group([graph_a, graph_b])

    def test_graph_group_rejects_disconnected_structures(self):
        disconnected = GraphBuilder.from_smiles("CC.O")
        model = FormulaRelativeRidgeCalibrator(alpha=0.1)
        model.fit([record("a1", "A", 0.0, 1.0), record("a2", "A", 1.0, 2.0)])

        with self.assertRaisesRegex(ValueError, "single-component"):
            model.predict_graph_group([disconnected])

    def test_predicts_same_formula_smiles_group(self):
        graph_a = GraphBuilder.from_smiles("CCO")
        graph_b = GraphBuilder.from_smiles("COC")
        from synde.energy import MoleculeScorer, extract_two_d_features

        scorer = MoleculeScorer()
        records = []
        for identifier, graph, target in (("a", graph_a, 0.0), ("b", graph_b, 1.0)):
            result = scorer.score(graph)
            records.append(
                FormulaCalibrationRecord(
                    identifier,
                    "C2H6O|charge=0",
                    extract_two_d_features(graph, result),
                    target,
                    {},
                )
            )
        model = FormulaRelativeRidgeCalibrator(alpha=0.1)
        model.fit(records)

        predictions = model.predict_smiles_group(["CCO", "COC"], scorer)
        self.assertAlmostEqual(sum(row.value for row in predictions), 0.0)

    def test_formula_key_handles_implicit_and_explicit_hydrogen(self):
        implicit = GraphBuilder.from_smiles("CC")
        explicit = GraphBuilder.from_smiles("[H]C([H])([H])C([H])([H])[H]")

        self.assertEqual(molecular_formula_charge_key(implicit), "C2H6|charge=0")
        self.assertEqual(
            molecular_formula_charge_key(implicit),
            molecular_formula_charge_key(explicit),
        )

    def test_graph_prediction_rejects_incompatible_feature_schema(self):
        model = FormulaRelativeRidgeCalibrator(alpha=0.1)
        model.fit([record("a1", "A", 0.0, 1.0), record("a2", "A", 1.0, 2.0)])
        model.card = replace(model.card, feature_schema="obsolete-schema")

        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            model.predict_graph(GraphBuilder.from_smiles("CC"))


if __name__ == "__main__":
    unittest.main()
