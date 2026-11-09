"""Offline, synthetic checks for the research code; no clinical data required."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import softmax

from run_azh_pilot import confidence_ece, describe_image, verified_archive
from run_train_cv import evaluate_fold
from study_utils import (fit_correctness_selectors, inner_partitions, outer_folds,
                         scores_for_policies, selector_inputs, validated_risk)


class SelectiveRiskTests(unittest.TestCase):
    def test_retention_and_error_are_computed_from_selected_predictions(self):
        labels=np.array(["D","D","P","P","S","S","V","V"])
        errors=np.array([0,1,1,0,0,1,1,0])
        scores=np.array([.9,.1,.8,.2,.7,.3,.6,.4])
        result=validated_risk(errors,labels,scores,.5)
        self.assertEqual((result["accepted"],result["errors"]),(4,2))
        self.assertEqual(result["risk"],.5)
        self.assertEqual(result["within_class_matched_accepted"],4)

    def test_class_standardisation_uses_full_test_class_weights(self):
        labels=np.array(["D"]*4+["P"]*2+["S"]*2+["V"]*2)
        errors=np.array([1,1,1,1,0,0,0,0,0,0])
        scores=np.array([10,9,8,7,6,5,4,1,3,2])
        result=validated_risk(errors,labels,scores,.8)
        self.assertEqual((result["accepted"],result["errors"]),(8,4))
        self.assertAlmostEqual(result["risk"],.5)
        self.assertAlmostEqual(result["standardised_risk"],.4)

    def test_class_with_no_selected_images_has_undefined_standardised_risk(self):
        labels=np.array(["D","D","P","P","S","S","V","V"])
        result=validated_risk(np.zeros(8,dtype=int),labels,np.arange(8),.25)
        self.assertIsNone(result["standardised_risk"])
        self.assertIsNone(result["classes"]["Diabetic"]["risk"])

    def test_invalid_inputs_are_rejected(self):
        labels=np.array(["D","P","S","V"])
        errors=np.zeros(4,dtype=int)
        for scores,coverage in [(np.zeros(4),0), (np.zeros(4),1.1),
                                (np.array([0.,1.,2.,np.nan]),.8)]:
            with self.subTest(coverage=coverage),self.assertRaises(ValueError):
                validated_risk(errors,labels,scores,coverage)
        with self.assertRaises(ValueError):
            validated_risk(errors,labels[:3],np.ones(4),.8)
        with self.assertRaises(ValueError):
            validated_risk(errors,np.array(["D","D","P","S"]),np.ones(4),.8)
        with self.assertRaises(ValueError):
            validated_risk(np.array([0,1,2,0]),labels,np.ones(4),.8)


class PartitionAndSelectorTests(unittest.TestCase):
    def setUp(self):
        self.labels=np.array(["D"]*20+["P"]*20+["S"]*20+["V"]*20)

    def test_each_image_appears_in_one_outer_test_fold(self):
        folds=list(outer_folds(self.labels))
        observed=[]
        for train,test in folds:
            self.assertFalse(set(train)&set(test))
            self.assertEqual(set(self.labels[test]),{"D","P","S","V"})
            observed.extend(test.tolist())
        self.assertEqual(sorted(observed),list(range(len(self.labels))))

    def test_inner_partitions_are_disjoint_and_stay_in_outer_train(self):
        train,test=next(outer_folds(self.labels))
        fit,calibration,selector=inner_partitions(self.labels,train)
        groups=[set(x.tolist()) for x in (fit,calibration,selector,test)]
        self.assertEqual(sum(map(len,groups)),len(set.union(*groups)))
        self.assertEqual(set.union(*groups[:3]),set(train))

    def test_stratification_failure_is_explicit(self):
        with self.assertRaises(ValueError):
            list(outer_folds(np.array(["D","P","S","V"]),n_splits=5))

    def test_confidence_only_policy_does_not_use_focus_or_labels_at_prediction(self):
        p=np.array([[.7,.1,.1,.1],[.4,.2,.2,.2],[.55,.15,.15,.15],
                    [.85,.05,.05,.05],[.6,.2,.1,.1],[.45,.25,.15,.15]])
        q=np.array([1.,20.,3.,8.,4.,10.])
        one,two=fit_correctness_selectors(p,q,np.array([1,0,0,1,1,0]))
        a=scores_for_policies(p,q,one,two)
        b=scores_for_policies(p,q[::-1],one,two)
        np.testing.assert_allclose(a["confidence"],b["confidence"])
        np.testing.assert_allclose(a["learned_confidence"],b["learned_confidence"])
        self.assertFalse(np.array_equal(a["focus"],b["focus"]))

    def test_bad_probability_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            selector_inputs(np.array([[.8,.8]]),np.array([1.]))
        with self.assertRaises(ValueError):
            selector_inputs(np.array([[.5,.5]]),np.array([-1.]))

    def test_end_to_end_fold_without_real_images(self):
        rng=np.random.default_rng(123)
        labels=np.array(["D"]*40+["P"]*40+["S"]*40+["V"]*40)
        features=rng.normal(size=(160,12))
        focus=rng.uniform(.1,10,size=160)
        outer_train,outer_test=next(outer_folds(labels))
        result=evaluate_fold(features,focus,labels,outer_train,outer_test,1)
        policies=result["policies"]
        self.assertEqual(result["sizes"]["outer_test"],32)
        self.assertEqual({p["accepted"] for p in policies.values()},{26})
        for p in policies.values():
            self.assertTrue(0<=p["risk"]<=1)
            self.assertEqual(sum(x["errors"] for x in p["classes"].values()),p["errors"])


class ScientificAndInputSanityTests(unittest.TestCase):
    def test_marginal_calibration_can_hide_group_disparities(self):
        # Both groups have confidence .5; one has 100% accuracy and one 0%.
        p=np.tile([.5,.5],(4,1))
        pred=np.array(["a"]*4)
        actual=np.array(["a","a","b","b"])
        self.assertEqual(confidence_ece(p,pred,actual),0.0)
        self.assertNotEqual(np.mean(pred[:2]==actual[:2]),np.mean(pred[2:]==actual[2:]))

    def test_temperature_preserves_predicted_class_but_can_change_multiclass_rank(self):
        logits=np.array([[0.,-4.,-4.],[0.,-3.,-6.]])
        a,b=softmax(logits,axis=1),softmax(logits/3,axis=1)
        np.testing.assert_array_equal(a.argmax(axis=1),b.argmax(axis=1))
        self.assertGreater(a[0].max(),a[1].max())
        self.assertLess(b[0].max(),b[1].max())

    def test_constant_image_has_zero_laplacian_variance(self):
        features,focus=describe_image(Image.new("RGB",(60,90),(128,128,128)))
        self.assertEqual(features.shape,(704,))
        self.assertEqual(focus,0)
        self.assertTrue(np.isfinite(features).all())

    def test_wrong_dataset_hash_is_rejected_without_network_access(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder,"Train.zip").write_bytes(b"not the authors' archive")
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                verified_archive(Path(folder),"Train",False)


if __name__ == "__main__":
    unittest.main()
