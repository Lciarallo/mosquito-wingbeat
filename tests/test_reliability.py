"""Checks against misleading accuracy and data leakage in binary detection."""
import unittest,json
import numpy as np,pandas as pd,torch
from reliability_metrics import binary_metrics,precision_at_prevalence
from reliability_models import PCENFrontend,JointDetector
from fetch_reliability_data import canonical_non_aedes,OUT
from train_reliability import choose_cut
from evaluate_reliability import intervals,verify,infer_stages
from train_reliability import infer
from reproduction_models import normalize


class Reliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_high_accuracy_constant_negative_detector_is_useless(self):
        r=binary_metrics([0]*99+[1],np.zeros(100),np.ones(100,bool),.5)
        self.assertEqual(r['accuracy'],.99);self.assertEqual(r['aedes_recall'],0);self.assertEqual(r['aedes_f1'],0)
        self.assertEqual(r['balanced_accuracy'],.5);self.assertIsNone(r['aedes_precision'])

    def test_invalid_and_unknown_audio_cannot_inflate_specificity(self):
        r=binary_metrics([0,0,1,1,-1],[.1,.9,.9,.1,.9],[False,True,True,False,True],.5)
        self.assertEqual(r['confusion'],[[0,1,1],[0,1,1]])
        self.assertEqual(r['specificity'],0);self.assertEqual(r['aedes_recall'],.5)
        self.assertEqual(r['aedes_precision'],.5);self.assertEqual(r['unknown_windows'],1)

    def test_positive_only_private_audio_does_not_prove_precision(self):
        r=binary_metrics([1,1],[.9,.8],[True,True],.5)
        self.assertEqual(r['aedes_precision'],1);self.assertFalse(r['precision_informative_for_binary_detection'])
        self.assertIsNone(r['specificity']);self.assertIsNone(r['macro_f1'])

    def test_genus_known_complex_is_negative_but_unknown_is_not(self):
        self.assertEqual(canonical_non_aedes('culex pipiens complex'),'Culex pipiens complex')
        self.assertEqual(canonical_non_aedes('an funestus sl'),'Anopheles funestus sl')
        for s in ['unknown','mosquito','ae aegypti',np.nan]:self.assertIsNone(canonical_non_aedes(s))

    def test_test_labels_cannot_select_threshold(self):
        with self.assertRaises(ValueError):choose_cut(pd.DataFrame(),[],partition='test')

    def test_rare_aedes_has_lower_precision_at_same_error_rates(self):
        self.assertLess(precision_at_prevalence(.8,.1,.01),.1)
        self.assertGreater(precision_at_prevalence(.8,.1,.5),.8)

    def test_ema_implementation_matches_causal_recurrence(self):
        front=PCENFrontend();e=torch.rand(2,96,125)*10;s=front.smoothing
        decay=(1-s)**torch.arange(e.shape[-1],dtype=e.dtype)
        vector=decay*((1-s)*e[...,:1]+s*(e/decay).cumsum(-1))
        expected=e[...,0];values=[expected]
        for j in range(1,e.shape[-1]):expected=(1-s)*expected+s*e[...,j];values.append(expected)
        torch.testing.assert_close(vector,torch.stack(values,-1),atol=2e-5,rtol=1e-5)

    def test_real_duration_and_batch_invariance(self):
        model=JointDetector('pcen_large',np.zeros(138),np.ones(138)).eval();x=torch.randn(2,2400);f=torch.randn(2,138)
        with torch.no_grad():
            z=model(x,f)[0];one=torch.cat([model(x[i:i+1],f[i:i+1])[0] for i in range(2)])
            torch.testing.assert_close(z,one,atol=2e-5,rtol=1e-5)
            self.assertTrue(torch.isfinite(model(torch.randn(1,7936),f[:1])[0]).all())

    def test_explanation_preserves_the_frozen_inference_rule(self):
        f=pd.DataFrame(dict(bucket=['short','long'],wave_index=[0,0]));x=np.zeros((2,138),dtype='float32')
        w={'short':np.random.default_rng(3).normal(size=(1,2400)).astype('float32'),'long':np.random.default_rng(4).normal(size=(1,7936)).astype('float32')}
        for name in ['raw_expanded','pcen_large']:
            m=JointDetector(name,np.zeros(138),np.ones(138)).eval()
            a=infer(m,name,f,w,x,'cpu');b,p,s=infer_stages(m,name,f,w,x,'cpu')
            np.testing.assert_allclose(a,b,atol=1e-6);self.assertTrue(np.all((p>=0)&(p<=1)))

    def test_bootstrap_f1_counts_invalid_predictions_consistently(self):
        f=pd.DataFrame(dict(group=['a','b','c','d'],source=['p','p','n','n'],label=[1,1,0,0],score=[.9,.1,.1,.9],quality_ok=[True,False,True,True]))
        ci=intervals(f,.5,replicates=100);self.assertIn('macro_f1',ci['metrics'])
        self.assertLessEqual(ci['metrics']['macro_f1']['lower95'],binary_metrics(f.label,f.score,f.quality_ok,.5)['macro_f1'])

    @unittest.skipUnless((OUT/'freeze.json').exists(),'Training has not frozen yet')
    def test_frozen_sources_and_models(self):verify()


if __name__=='__main__':unittest.main()
