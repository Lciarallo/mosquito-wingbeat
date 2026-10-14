"""Regression tests for the previously demonstrated scientific validation failures."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from corrected_validation import assert_disjoint, select_thresholds
from swarm_adapted_ensemble import SwarmAdaptedEnsemble
from models_committee import StackingMetaCommittee
from benchmark_all_methods import evaluate_predictions


class TreeStub:
    def predict_proba(self,x):
        p=np.asarray(x)[:,0]
        return np.column_stack([1-p,p])


class MetaStub:
    def predict_proba(self,x):
        self.last_input=x.copy()
        return np.column_stack([1-x[:,0],x[:,0]])


class CorrectedValidationTests(unittest.TestCase):
    def test_shared_recording_and_identical_file_fail_partition_check(self):
        for column in ['group','content_sha256']:
            f=pd.DataFrame({'group':['a','b'],'split':['train','test'],'content_sha256':['x','y']})
            f[column]=['same','same']
            with self.assertRaises(ValueError):assert_disjoint(f)

    def test_test_labels_cannot_fit_threshold_or_reference(self):
        with self.assertRaises(ValueError):select_thresholds([0,1],[.1,.9],[1,1],partition='test')
        ensemble=SwarmAdaptedEnsemble()
        with self.assertRaises(ValueError):ensemble.fit_reference(np.ones((2,3)),partition='test')
        with self.assertRaises(ValueError):ensemble.calibrate_threshold([0,1],[.1,.9],partition='test')

    def test_same_recording_is_stable_across_batch_members_and_sizes(self):
        train=np.array([[.1,.9,.2],[.3,.7,.4],[.5,.5,.6],[.7,.3,.8],[.9,.1,1.]])
        e=SwarmAdaptedEnsemble().fit_reference(train,partition='train')
        query=np.array([[.42,.38,.51],[.8,.8,.8],[.12,.12,.12]])
        expected=e.blend_components(query)
        separately=np.concatenate([e.blend_components(row[None,:]) for row in query])
        np.testing.assert_allclose(expected,separately,atol=1e-15)
        reordered=e.blend_components(query[[2,0,1]])
        np.testing.assert_allclose(expected,reordered[[1,2,0]],atol=1e-15)

    def test_stack_inference_uses_same_raw_probability_scale_as_oof(self):
        meta=MetaStub();model=TreeStub()
        committee=StackingMetaCommittee({'tree':{'model':model,'slope':7.,'bias':-3.}},meta)
        x=np.array([[.1],[.8]])
        committee.predict_proba(x)
        np.testing.assert_array_equal(meta.last_input,x)

    def test_unknown_species_remains_unlabelled_in_benchmark(self):
        result=evaluate_predictions(np.array([1,0,-1]),np.array([.9,.1,.99]),np.ones(3,bool),.2,.8,'test','dummy')
        self.assertEqual(result['unknown_windows'],1)
        self.assertEqual(result['negative_windows'],1)
        self.assertEqual(result['false_positive_emissions'],0)
        self.assertEqual(result['unresolved_positive_emissions'],1)

    def test_actual_oof_has_zero_recording_overlap(self):
        r=Path(__file__).resolve().parents[1]
        oof=pd.read_csv(r/'results/corrected_2026_10_01/models_trees/oof_train_predictions.csv')
        f=pd.read_csv(r/'results/multisource/segments.csv',low_memory=False)
        self.assertEqual(oof.groupby('group').fold.nunique().max(),1)
        self.assertEqual(oof.fold.nunique(),5)
        self.assertTrue(f.iloc[oof.row_index].split.eq('train').all())
        self.assertFalse(oof.row_index.duplicated().any())

    def test_actual_optical_calibration_and_test_share_no_ids_or_content(self):
        r=Path(__file__).resolve().parents[1]/'results/corrected_2026_10_01'
        cal=pd.read_csv(r/'d5_calibration_manifest.csv');test=pd.read_csv(r/'d5_test_manifest.csv')
        for field in ['id','sha256','pcm_sha256']:
            self.assertFalse(set(cal[field])&set(test[field]))


if __name__=='__main__':unittest.main()
