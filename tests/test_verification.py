"""Checks for the reproducibility comparison."""
import unittest
import subprocess
import sys
from pathlib import Path

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

    def test_cli_verifies_one_requested_output_and_requires_an_input(self):
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(root / "verify_results.py")]
        result = subprocess.run(command + ["--woundsdb", str(root / "results/woundsdb_audit")],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("All 1 requested outputs match", result.stdout)
        missing = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("provide at least one", missing.stderr)


if __name__=="__main__":
    unittest.main()
