"""Offline checks for patient-group metadata and pretrained preprocessing."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from audit_woundsdb import audit_rows
from patient_splits import patient_disjoint_folds, patient_disjoint_inner
from run_pretrained_comparison import ensure_weights, prepare_image


class MetadataAuditTests(unittest.TestCase):
    def test_counts_repeated_patients_and_never_outputs_identifiers(self):
        rows = [
            {"Path to files": "case_12/day_1/results/scene_1", "Photo": "1", "Wound aetiology": "A"},
            {"Path to files": "case_12/day_2/results/scene_1", "Photo": "1", "Wound aetiology": "B"},
            {"Path to files": "case_27/day_1/results/scene_1", "Photo": "1", "Wound aetiology": "A"},
        ]
        result = audit_rows(rows)
        self.assertEqual(result["photo_records"], 3)
        self.assertEqual(result["distinct_case_identifiers"], 2)
        self.assertEqual(result["distinct_case_day_pairs"], 3)
        self.assertEqual(result["case_groups_with_multiple_source_labels"], 1)
        counts = {item["source_aetiology"]: item for item in result["source_aetiology_counts"]}
        self.assertEqual((counts["A"]["photo_records"], counts["A"]["case_groups"]), (2, 2))
        self.assertNotIn("case_12", json.dumps(result))

    def test_missing_identifiers_and_duplicate_paths_fail(self):
        row = {"Path to files": "case_1/day_1/scene_1", "Photo": "1", "Wound aetiology": "A"}
        with self.assertRaises(ValueError):
            audit_rows([row, row])
        with self.assertRaises(ValueError):
            audit_rows([{**row, "Path to files": "scene_1"}])


class PretrainedInputTests(unittest.TestCase):
    def test_preprocessing_has_expected_shape_and_blur_changes_pixels(self):
        yy, xx = np.indices((128, 128))
        pattern = (((xx // 4 + yy // 4) % 2) * 255).astype(np.uint8)
        image = Image.fromarray(np.repeat(pattern[:, :, None], 3, axis=2))
        clean = prepare_image(image, 0)
        blurred = prepare_image(image, 1.5)
        self.assertEqual(clean.shape, (3, 224, 224))
        self.assertEqual(clean.dtype, np.float32)
        self.assertTrue(np.isfinite(clean).all())
        self.assertFalse(np.array_equal(clean, blurred))

    def test_bad_weights_and_invalid_radii_fail_without_downloading(self):
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder, "weights.pth")
            weights.write_bytes(b"not the official checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                ensure_weights(weights, False)
            with self.assertRaises(FileNotFoundError):
                ensure_weights(Path(folder, "missing.pth"), False)
        with self.assertRaises(ValueError):
            prepare_image(Image.new("RGB", (8, 8)), -1)


class PatientSplitTests(unittest.TestCase):
    def test_outer_and_inner_partitions_never_share_patient_groups(self):
        labels, groups = [], []
        for label in ("D", "P", "S", "V"):
            for patient in range(20):
                labels.extend([label, label])
                groups.extend([f"{label}-{patient}"] * 2)
        labels, groups = np.asarray(labels), np.asarray(groups)
        folds = patient_disjoint_folds(labels, groups)
        for outer_train, outer_test in folds:
            self.assertFalse(set(groups[outer_train]) & set(groups[outer_test]))
            inner = patient_disjoint_inner(labels, groups, outer_train)
            parts = [set(groups[idx]) for idx in (*inner, outer_test)]
            for i in range(4):
                for j in range(i + 1, 4):
                    self.assertFalse(parts[i] & parts[j])
            for indices in (*inner, outer_test):
                self.assertEqual(set(labels[indices]), {"D", "P", "S", "V"})

    def test_rare_class_and_missing_ids_fail_closed(self):
        labels = ["A"] * 5 + ["B"] * 5
        groups = ["a1", "a2", "a3", "a4", "a5"] + ["b1"] * 5
        with self.assertRaisesRegex(ValueError, "patient groups"):
            patient_disjoint_folds(labels, groups)
        groups[-1] = ""
        with self.assertRaisesRegex(ValueError, "identifier"):
            patient_disjoint_folds(labels, groups)


if __name__ == "__main__":
    unittest.main()
