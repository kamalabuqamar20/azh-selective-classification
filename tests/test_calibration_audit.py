"""Small known-probability cases for the held-out calibration summary."""
import unittest

import numpy as np

from run_calibration_audit import probability_metrics


class CalibrationMetricsTests(unittest.TestCase):
    def test_perfect_and_confidently_wrong_predictions(self):
        metrics = probability_metrics([[1, 0], [1, 0]], ["a", "b"], ["a", "b"])
        self.assertEqual(metrics["errors"], 1)
        self.assertAlmostEqual(metrics["accuracy"], .5)
        self.assertAlmostEqual(metrics["brier_multiclass"], 1.0)
        self.assertAlmostEqual(metrics["ece_top_label_10_equal_width_bins"], .5)
        self.assertEqual(metrics["bins"][-1]["count"], 2)
        self.assertEqual(sum(b["count"] for b in metrics["bins"]), 2)
        self.assertAlmostEqual(metrics["nll"], -np.log(1e-15) / 2)

    def test_uniform_predictions_have_expected_metrics(self):
        metrics = probability_metrics([[.5, .5], [.5, .5]], ["a", "b"], ["a", "b"])
        self.assertAlmostEqual(metrics["ece_top_label_10_equal_width_bins"], 0.)
        self.assertAlmostEqual(metrics["brier_multiclass"], .5)
        self.assertAlmostEqual(metrics["nll"], np.log(2))

    def test_invalid_probability_and_label_inputs_are_rejected(self):
        for probabilities, labels, classes in [
                ([[.2, .2]], ["a"], ["a", "b"]),
                ([[np.nan, 0]], ["a"], ["a", "b"]),
                ([[1, 0]], ["c"], ["a", "b"]),
                ([[1, 0]], ["a"], ["a", "a"])]:
            with self.subTest(probabilities=probabilities, labels=labels):
                with self.assertRaises(ValueError):
                    probability_metrics(probabilities, labels, classes)


if __name__ == "__main__":
    unittest.main()
