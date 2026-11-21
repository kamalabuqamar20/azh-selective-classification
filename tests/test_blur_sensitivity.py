"""Synthetic checks of paired blur, clean training, and frozen thresholds."""
import unittest

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from run_azh_pilot import describe_image
from run_blur_sensitivity import (RADII, blur_features, evaluate_conditions,
                                  fixed_summary, threshold_for_coverage)
from run_train_cv import evaluate_fold
from study_utils import outer_folds


class BlurFeatureTests(unittest.TestCase):
    def test_zero_radius_matches_existing_extraction_and_blur_reduces_edges(self):
        yy, xx = np.indices((91, 173))
        pattern = (((xx // 4 + yy // 4) % 2) * 255).astype(np.uint8)
        image = Image.fromarray(np.repeat(pattern[:, :, None], 3, axis=2))
        original_features, original_focus = describe_image(image)
        clean_features, clean_focus = blur_features(image, 0)
        _, blurred_focus = blur_features(image, 1.5)
        np.testing.assert_array_equal(clean_features, original_features)
        self.assertEqual(clean_focus, original_focus)
        self.assertLess(blurred_focus, clean_focus)

    def test_invalid_blur_radius_fails(self):
        for radius in (-1, float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                blur_features(Image.new("RGB", (10, 10)), radius)


class ThresholdTests(unittest.TestCase):
    def test_threshold_uses_development_scores_and_retains_ties(self):
        scores = np.array([.1, .4, .4, .8, 1.])
        threshold = threshold_for_coverage(scores, .6)
        self.assertEqual(threshold, .4)
        self.assertEqual(int((scores >= threshold).sum()), 4)

    def test_fixed_threshold_counts_transitions_and_empty_retention(self):
        errors = np.array([0, 1, 0, 1])
        scores = np.array([.9, .5, .2, .8])
        clean_keep = np.array([True, True, False, False])
        result = fixed_summary(errors, scores, .7, clean_keep)
        self.assertEqual((result["accepted"], result["errors"]), (2, 1))
        self.assertEqual(result["accepted_clean_deferred_here"], 1)
        self.assertEqual(result["deferred_clean_accepted_here"], 1)
        self.assertIsNone(fixed_summary(errors, scores, 1.1, clean_keep)["risk"])


class PairedFoldTests(unittest.TestCase):
    def test_clean_fold_matches_previous_analysis_and_thresholds_ignore_test_labels(self):
        rng = np.random.default_rng(123)
        labels = np.array(["D"] * 40 + ["P"] * 40 + ["S"] * 40 + ["V"] * 40)
        clean = rng.normal(size=(160, 12))
        clean_focus = rng.uniform(.1, 10, size=160)
        x = {radius: clean + radius * .1 for radius in RADII}
        focus = {radius: clean_focus / (1 + radius) for radius in RADII}
        train, test = next(outer_folds(labels))
        with threadpool_limits(limits=1):
            previous = evaluate_fold(clean, clean_focus, labels, train, test, 1)
            result = evaluate_conditions(x, focus, labels, train, test, 1)
            changed_labels = labels.copy()
            changed_labels[test] = np.roll(changed_labels[test], 1)
            changed = evaluate_conditions(x, focus, changed_labels, train, test, 1)
        self.assertEqual(result["conditions"][0]["matched"], previous["policies"])
        self.assertEqual(result["fixed_thresholds_from_clean_calibration"],
                         changed["fixed_thresholds_from_clean_calibration"])
        initial = result["conditions"][0]
        for condition in result["conditions"]:
            self.assertEqual(
                condition["unfiltered_errors"],
                initial["unfiltered_errors"] + condition["correct_clean_wrong_here"]
                - condition["wrong_clean_correct_here"],
            )
            for policy, fixed in condition["fixed"].items():
                clean_accepted = initial["fixed"][policy]["accepted"]
                self.assertEqual(fixed["accepted"], clean_accepted
                                 - fixed["accepted_clean_deferred_here"]
                                 + fixed["deferred_clean_accepted_here"])


if __name__ == "__main__":
    unittest.main()
