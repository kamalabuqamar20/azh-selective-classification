"""Tests for the descriptive, non-inferential split-sensitivity summary."""
import unittest

from run_split_sensitivity import describe_differences


class SplitSensitivityTests(unittest.TestCase):
    def test_risk_difference_signs_and_range(self):
        result = describe_differences([{"joint_minus_confidence": -.05},
                                       {"joint_minus_confidence": 0.0},
                                       {"joint_minus_confidence": .08}])
        self.assertAlmostEqual(result["mean_difference"], .01)
        self.assertEqual(result["minimum_difference"], -.05)
        self.assertEqual(result["maximum_difference"], .08)
        self.assertEqual(result["joint_fewer_errors"], 1)
        self.assertEqual(result["equal_errors"], 1)
        self.assertEqual(result["joint_more_errors"], 1)

    def test_empty_and_nonfinite_differences_are_rejected(self):
        with self.assertRaises(ValueError):
            describe_differences([])
        with self.assertRaises(ValueError):
            describe_differences([{"joint_minus_confidence": float("nan")}])


if __name__ == "__main__":
    unittest.main()
