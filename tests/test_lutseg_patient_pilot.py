"""Synthetic tests of grouped selection, reference input and leakage guards."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from lutseg_data import checked_download, safe_image_path, validate_rows
from patient_splits import patient_disjoint_folds
from run_lutseg_patient_pilot import fit_and_score, register_decoded_image, selection_mask, selection_metrics


class LutsegInputTests(unittest.TestCase):
    def test_rejects_missing_group_duplicate_and_unsafe_path(self):
        row = {"patient_id": "anonymous-a", "image_file_name": "Images/P1/test.jpg",
               "mask_file_name": "Masks/P1/test.png", "classes_present": [0, 4]}
        self.assertEqual(len(validate_rows((json.dumps(row) + "\n").encode())), 1)
        for invalid in ({**row, "patient_id": " "}, {**row, "image_file_name": "Images/../../secret"}):
            with self.assertRaises(ValueError):
                validate_rows((json.dumps(invalid) + "\n").encode())
        with self.assertRaises(ValueError):
            validate_rows((json.dumps(row) + "\n" + json.dumps(row)).encode())
        with self.assertRaises(ValueError):
            safe_image_path(Path("/tmp/dataset"), "Images/../outside.jpg")

    def test_wrong_cached_hash_is_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.jpg"
            path.write_bytes(b"not the verified image")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                checked_download("https://unused.invalid", path, "0" * 64, download=False)

    def test_duplicate_image_cannot_cross_patient_groups_or_change_label(self):
        image = Image.new("RGB", (3, 3), color=(10, 20, 30))
        seen = {}
        self.assertFalse(register_decoded_image(seen, image, "patient-a", True))
        self.assertTrue(register_decoded_image(seen, image, "patient-a", True))
        with self.assertRaisesRegex(ValueError, "different patient groups"):
            register_decoded_image(seen, image, "patient-b", True)
        with self.assertRaisesRegex(ValueError, "conflicting target labels"):
            register_decoded_image(seen, image, "patient-a", False)


class LutsegSelectionTests(unittest.TestCase):
    def test_matched_coverage_fixed_threshold_and_patient_macro(self):
        scores = np.array([.9, .8, .7, .6, .5])
        np.testing.assert_array_equal(selection_mask(scores, .75, "matched_coverage"), [True, True, True, True, False])
        np.testing.assert_array_equal(selection_mask(scores, .75, "fixed_threshold"), [True, True, False, False, False])
        m = selection_metrics(np.array([1, 0, 0, 1]),
                              np.array(["absent", "present", "absent", "present"]),
                              np.array(["a", "a", "b", "c"]),
                              np.array([True, True, True, False]))
        self.assertAlmostEqual(m["risk"], 1 / 3)
        self.assertAlmostEqual(m["mean_patient_error_rate"], .25)
        self.assertEqual(m["patients_with_retained_images"], 2)
        self.assertEqual(m["patients_all_images_deferred"], 1)
        empty = selection_metrics(np.array([0, 1]), np.array(["absent", "present"]),
                                  np.array(["a", "b"]), np.array([False, False]))
        self.assertIsNone(empty["risk"])
        self.assertIsNone(empty["mean_patient_error_rate"])
        self.assertEqual(empty["patients_all_images_deferred"], 2)

    def test_test_labels_do_not_change_fitted_scores_or_thresholds(self):
        rng = np.random.default_rng(813)
        groups = np.repeat([f"group-{i:02d}" for i in range(30)], 4)
        labels = np.tile(["absent", "present", "absent", "present"], 30)
        features = rng.normal(size=(120, 5))
        focus = np.exp(rng.normal(size=120))
        train, test = patient_disjoint_folds(labels, groups, n_splits=5, seed=42)[0]
        changed = labels.copy()
        changed[test] = np.where(labels[test] == "present", "absent", "present")
        for weight in (None, "balanced"):
            _, scores, thresholds, _, _ = fit_and_score(features, focus, labels, groups, train, test, weight)
            _, other_scores, other_thresholds, _, _ = fit_and_score(features, focus, changed, groups, train, test, weight)
            for policy in scores:
                np.testing.assert_allclose(scores[policy], other_scores[policy])
            self.assertEqual(thresholds, other_thresholds)


if __name__ == "__main__":
    unittest.main()
