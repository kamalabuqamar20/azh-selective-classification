"""Synthetic checks for aggregate wound-area summaries."""
import unittest
import numpy as np
from audit_lutseg_masks import summarise_areas


class LutsegAreaAuditTests(unittest.TestCase):
    def test_area_thresholds_count_images_and_distinct_patients(self):
        s = summarise_areas([0, .0001, .01, .05, .2], ["a", "a", "b", "b", "c"])
        self.assertEqual(s["positive_images"], 4)
        self.assertEqual(s["positive_patient_groups"], 3)
        self.assertEqual(s["minimum_area_thresholds"][1]["images"], 3)
        self.assertEqual(s["minimum_area_thresholds"][1]["patient_groups"], 2)
        self.assertEqual(s["minimum_area_thresholds"][3]["images"], 1)

    def test_rejects_invalid_or_mismatched_areas(self):
        for values, groups in (([1.2], ["a"]), ([np.nan], ["a"]), ([0, .1], ["a"])):
            with self.assertRaises(ValueError):
                summarise_areas(values, groups)


if __name__ == "__main__":
    unittest.main()
