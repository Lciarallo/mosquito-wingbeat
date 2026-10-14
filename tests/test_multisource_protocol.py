"""Check real dataset boundaries that would invalidate external evaluation."""
from pathlib import Path
import unittest
import numpy as np,pandas as pd,torch
from large_audio_model import LargeAudioCNN
ROOT=Path(__file__).resolve().parents[1]
class MultisourceProtocolTests(unittest.TestCase):
 def test_original_holdouts_and_stream_timing_are_preserved(self):
  old=pd.read_csv(ROOT/'results/expanded/segments.csv',low_memory=False)
  new=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False).iloc[:len(old)]
  for field in ['path','group','split','start_s','step_s','label']:
   pd.testing.assert_series_equal(old[field].reset_index(drop=True),new[field].reset_index(drop=True),check_dtype=False)
 def test_ufrgs_negative_labels_and_entire_domain_holdout(self):
  df=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False);u=df[df.domain.eq('ufrgs')]
  self.assertEqual(u.path.nunique(),15);self.assertTrue(u.split.eq('test').all())
  negative=u.path.str.contains('/No_Aedes_aegypti/')
  self.assertEqual(u[negative].path.nunique(),5);self.assertTrue(u.loc[negative,'label'].eq('not_aedes').all())
  self.assertEqual(u[~negative].path.nunique(),10);self.assertTrue(u.loc[~negative,'label'].eq('Aedes aegypti').all())
 def test_all_source_families_are_disjoint(self):
  df=pd.read_csv(ROOT/'results/multisource/segments.csv',low_memory=False)
  self.assertEqual(df.groupby('group').split.nunique().max(),1)
  m=df[df.domain.eq('mosquitosong')];challenge=m[m.path.str.contains('Ae.Aegypti_1[FM]_5D_',regex=True)]
  self.assertGreater(len(challenge),0);self.assertTrue(challenge.split.eq('challenge').all())
  self.assertFalse(df[df.split.eq('train')].path.str.contains('private-audio|large_fresh|fresh_external',regex=True).any())
 def test_real_short_input_needs_no_waveform_padding(self):
  torch.set_num_threads(2);model=LargeAudioCNN().eval()
  with torch.no_grad():p,g=model(torch.randn(2,4800),augment=True)
  self.assertEqual(tuple(p.shape),(2,20));self.assertTrue(torch.isfinite(p).all());self.assertTrue(torch.isfinite(g).all())
if __name__=='__main__':unittest.main()
