"""Regression checks for scientific partitioning and frozen binary inference."""
import json,unittest
import numpy as np,pandas as pd,torch
from reproduction_models import FernandesBinary,MosquitoSongBinary,DomainResidualBinary,normalize
from train_reproduction import choose_binary_cut,ROOT,OUT
from robust_reproduction_ensemble import robust_cut,blend,load as load_robust
from evaluate_reproduction import summarize,verify_freeze
from evaluate_hard_negative_reproduction import load as load_hard,pcmsha
from aedes_binary import targets
from corrected_validation import assert_disjoint,sha


class BinaryInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.manual_seed(5)

    def test_models_accept_real_short_and_long_audio(self):
        x=torch.randn(2,2400)
        for model in [FernandesBinary(),DomainResidualBinary()]:
            model.eval()
            with torch.no_grad():
                for size in [2400,7936]:
                    z,_=model(torch.randn(2,size));self.assertEqual(tuple(z.shape),(2,));self.assertTrue(torch.isfinite(z).all())
        model=MosquitoSongBinary().eval()
        with torch.no_grad():self.assertEqual(tuple(model(x)[0].shape),(2,))
        with self.assertRaises(ValueError):model(torch.randn(2,7936))

    def test_eval_output_independent_of_other_batch_examples(self):
        x=torch.randn(3,2400)
        for m in [FernandesBinary(),MosquitoSongBinary(),DomainResidualBinary()]:
            m.eval()
            with torch.no_grad():
                p=m(x)[0];one=torch.cat([m(a[None,:])[0] for a in x])
            torch.testing.assert_close(p,one,atol=2e-5,rtol=1e-5)

    def test_dc_and_gain_invariance(self):
        x=torch.randn(2,2400)
        torch.testing.assert_close(normalize(x),normalize(x*.4+.12),atol=1e-6,rtol=1e-5)
        m=FernandesBinary().eval()
        with torch.no_grad():torch.testing.assert_close(m(x)[0],m(x*.4+.12)[0],atol=1e-6,rtol=1e-5)

    def test_test_labels_cannot_choose_cut(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.3,.7,.9])
        with self.assertRaises(ValueError):choose_binary_cut(y,p,np.ones(4),partition='test')
        f=pd.DataFrame(dict(label=['noise','noise','Aedes aegypti','Aedes aegypti'],domain=['x']*4,group=['1','2','3','4']))
        with self.assertRaises(ValueError):robust_cut(f,p,partition='test')

    def test_invalid_audio_is_never_counted_as_successful_rejection(self):
        r=summarize([0,0,1,1],[.1,.8,.9,.2],[False,True,True,True],.5)
        self.assertEqual(r['balanced_accuracy'],.5);self.assertEqual(r['operational_balanced_accuracy'],.25)
        self.assertEqual(r['invalid'],1);self.assertEqual(r['negative_emissions'],1)

    def test_unknown_mosquito_not_negative(self):
        f=pd.DataFrame(dict(label=['unknown_mosquito','Aedes aegypti','Anopheles ziemanni','noise']))
        np.testing.assert_array_equal(targets(f),[-1,1,0,0])

    def test_recording_and_byte_duplicates_are_rejected(self):
        for groups,hashes in [(['same','same'],['one','two']),(['one','two'],['same','same'])]:
            f=pd.DataFrame(dict(group=groups,content_sha256=hashes,split=['train','test']))
            with self.assertRaises(ValueError):assert_disjoint(f)

    def test_ensemble_is_batch_independent(self):
        r=load_robust();rng=np.random.default_rng(7);p={n:rng.random(20) for n in r['model_names']}
        one=np.concatenate([blend(r,{n:v[i:i+1] for n,v in p.items()}) for i in range(20)])
        np.testing.assert_allclose(blend(r,p),one,atol=1e-15)

    def test_all_checkpoints_and_original_weights_remain_frozen(self):
        verify_freeze();load_hard();load_robust()
        originals=json.loads((OUT/'original_weights_sha256.json').read_text())
        self.assertTrue(all(sha(ROOT/p)==h for p,h in originals.items()))

    def test_final_test_was_selected_after_freeze_and_is_not_training(self):
        f=json.loads((OUT/'hard_negative/freeze.json').read_text())
        plan=json.loads((OUT/'hard_negative/final_selection.json').read_text())
        self.assertGreater(plan['selection_utc'],f['frozen_utc']);self.assertEqual(plan['model_hashes'],f['model_hashes'])
        prior=json.loads((OUT/'fresh_selection.json').read_text())
        self.assertFalse({p['group'] for p in prior['files']}&{p['group'] for p in plan['files']})
        self.assertFalse({p['sha256'] for p in prior['files']}&{p['sha256'] for p in plan['files']})
        audit=pd.read_csv(OUT/'hard_negative/final_novelty_audit.csv')
        self.assertEqual(int(audit.exact_pcm_windows_seen.sum()),0)

    def test_pcm_hash_same_decoded_data_different_float_storage(self):
        x=np.array([0,.125,-.25],dtype='float32')
        self.assertEqual(pcmsha(x),pcmsha(x.astype('float64')))


if __name__=='__main__':unittest.main()
