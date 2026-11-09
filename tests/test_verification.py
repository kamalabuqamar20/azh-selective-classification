"""Checks for the reproducibility comparison."""
import unittest

from verify_results import differences


class VerifyResultsTests(unittest.TestCase):
    def test_tolerance_and_environment_are_handled(self):
        expected={"count":4,"risk":.1,"environment":{"python":"3.12.14"}}
        actual={"count":4,"risk":.10000000001,"environment":{"python":"3.12.0"}}
        self.assertEqual(differences(expected,actual),[])

    def test_changed_counts_are_not_tolerated(self):
        self.assertTrue(differences({"retained":148},{"retained":147}))

    def test_missing_or_extra_fields_are_reported(self):
        self.assertTrue(differences({"a":[1,2]},{"a":[1]}))
        self.assertTrue(differences({"a":1},{"a":1,"b":2}))


if __name__=="__main__":
    unittest.main()
