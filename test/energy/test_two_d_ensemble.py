import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from synde.energy import (
    FormulaCalibrationRecord,
    FormulaRelativeEnsemble,
    FormulaRelativeRidgeCalibrator,
)


def fitted_model(slope: float, profile: str) -> FormulaRelativeRidgeCalibrator:
    rows = [
        FormulaCalibrationRecord("a", "A", {"x": 0.0}, 0.0, {}),
        FormulaCalibrationRecord("b", "A", {"x": 1.0}, slope, {}),
    ]
    model = FormulaRelativeRidgeCalibrator(
        alpha=1e-10,
        target="relative energy",
        units="u",
        feature_profile=profile,
    )
    model.fit(rows)
    return model


class TestFormulaRelativeEnsemble(unittest.TestCase):
    def test_fixed_average_is_zero_centered_and_inspectable(self):
        ensemble = FormulaRelativeEnsemble(
            [fitted_model(2.0, "full"), fitted_model(4.0, "compact")],
            [0.5, 0.5],
        )
        predictions = ensemble.predict_group([{"x": 0.0}, {"x": 1.0}])

        self.assertAlmostEqual(sum(row.value for row in predictions), 0.0)
        self.assertAlmostEqual(predictions[1].value - predictions[0].value, 3.0)
        self.assertEqual(len(predictions[0].member_predictions), 2)
        self.assertIsNone(predictions[0].uncertainty)
        self.assertEqual(ensemble.card.member_profiles, ("full", "compact"))

    def test_manifest_loads_relative_member_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fitted_model(2.0, "full").save(root / "full.json")
            fitted_model(4.0, "compact").save(root / "compact.json")
            manifest = root / "ensemble.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model_name": "test-ensemble",
                        "members": ["full.json", "compact.json"],
                        "weights": [0.5, 0.5],
                    }
                ),
                encoding="utf-8",
            )

            ensemble = FormulaRelativeEnsemble.load(manifest)
            prediction = ensemble.predict_group([{"x": 0.0}, {"x": 1.0}])
            self.assertAlmostEqual(prediction[1].value - prediction[0].value, 3.0)

    def test_rejects_incompatible_members(self):
        first = fitted_model(2.0, "full")
        second = fitted_model(4.0, "compact")
        second.units = "different"
        second.card = None

        with self.assertRaisesRegex(RuntimeError, "fitted or loaded"):
            FormulaRelativeEnsemble([first, second])


if __name__ == "__main__":
    unittest.main()
