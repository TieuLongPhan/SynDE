import unittest
from synde.energy.syn import CalibrationRecord, RidgeCalibrator, deterministic_split


class TestCalibration(unittest.TestCase):
    def test_ridge_fit_and_prediction(self):
        rows = [
            CalibrationRecord(str(i), {"x": float(i)}, 2.0 * i + 1.0, {})
            for i in range(6)
        ]
        model = RidgeCalibrator(target="synthetic", alpha=1e-10)
        card = model.fit(rows)
        pred = model.predict({"x": 3.5})
        self.assertEqual(card.target, "synthetic")
        self.assertAlmostEqual(pred.value, 8.0, places=5)
        self.assertEqual(pred.units, "kcal/mol")

    def test_deterministic_split(self):
        rows = [CalibrationRecord(str(i), {}, float(i), {}) for i in range(10)]
        self.assertEqual(deterministic_split(rows), deterministic_split(rows))
