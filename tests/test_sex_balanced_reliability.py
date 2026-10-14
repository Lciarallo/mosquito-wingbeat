"""Guard training feedback roles, deployed-threshold selection and paired CIs."""
import unittest,json
import numpy as np,pandas as pd
from train_sex_balanced_reliability import OUT,ROOT,cut
from evaluate_sex_balanced_reliability import verify,bootstrap_values
from corrected_validation import assert_disjoint
from reliability_metrics import binary_metrics


class Adaptation(unittest.TestCase):
    def test_calibration_partition_required(self):
        with self.assertRaises(ValueError):cut(pd.DataFrame(),[],partition='test')

    def test_female_feedback_is_training_and_not_validation(self):
        f=pd.read_csv(OUT/'segments.csv',low_memory=False);assert_disjoint(f)
        groups=json.loads((OUT/'protocol.json').read_text())['adapted_groups']
        self.assertEqual(len(groups),3);self.assertEqual(set(f.loc[f.group.isin(groups),'split']),{'train'})
        self.assertFalse(f[f.split.eq('train')].path.str.contains('private|download.audio|S_1_D_3|ufrgs',case=False,regex=True).any())

    def test_selection_uses_the_deployed_cut(self):
        s=json.loads((OUT/'selection.json').read_text());candidates=json.loads((OUT/'validation_search.json').read_text())
        best=max(candidates,key=lambda r:r['selection_score']);self.assertEqual(best['name'],s['selected_model'])
        for row in candidates:
            score=min(row['validation_precision'] or 0,row['validation_recall'],row['validation_minimum_recall'])+.02*row['validation_macro_f1']
            self.assertAlmostEqual(score,row['selection_score'])

    def test_frozen_artifacts_remain_identical(self):verify()

    def test_paired_bootstrap_identical_predictions_have_zero_difference(self):
        f=pd.DataFrame(dict(group=['a','b','c','d'],source=['p','p','n','n'],label=[1,1,0,0],score=[.9,.3,.2,.8],quality_ok=[True]*4))
        a,_=bootstrap_values(f,.5,100);b,_=bootstrap_values(f.copy(),.5,100)
        for k in a:np.testing.assert_array_equal(a[k]-b[k],np.zeros(100))

    @unittest.skipUnless((OUT/'fresh_selection.json').exists(),'No fresh acquisition yet')
    def test_next_acquisition_excludes_all_previous_source_groups(self):
        fresh=json.loads((OUT/'fresh_selection.json').read_text());old=json.loads((ROOT/'results/reliability_2026_10_01/fresh_selection.json').read_text())
        self.assertFalse({x['group'] for x in fresh['files']}&{x['group'] for x in old['files']})
        self.assertFalse({x['sha256'] for x in fresh['files']}&{x['sha256'] for x in old['files']})
        freeze=json.loads((OUT/'freeze.json').read_text());self.assertGreater(fresh['selection_utc'],freeze['frozen_utc'])


if __name__=='__main__':unittest.main()
