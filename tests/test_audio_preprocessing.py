"""Verify filters preserve real duration and spectral cues instead of manufacturing tones."""
import unittest
import numpy as np
from audio_preprocessing import METHODS,apply
class PreprocessingTests(unittest.TestCase):
 def test_silence_stays_silent_for_every_method(self):
  for method in METHODS:
   result=apply(np.zeros((2,15872),dtype='float32'),method)
   self.assertEqual(result.shape,(2,15872));self.assertTrue(np.isfinite(result).all());self.assertEqual(float(np.max(np.abs(result))),0)
 def test_no_duration_extension_for_short_recordings(self):
  x=np.random.default_rng(1).normal(0,.01,(2,4800)).astype('float32')
  for method in METHODS:
   y=apply(x,method);self.assertEqual(x.shape,y.shape);self.assertTrue(np.isfinite(y).all())
 def test_bandpass_removes_rumble_and_preserves_two_harmonics(self):
  t=np.arange(15872)/16000;freqs=[50,550,1100]
  x=sum(np.sin(2*np.pi*f*t) for f in freqs).astype('float32')[None,:]
  y=apply(x,'bandpass_150_3500')[0];cut=slice(1600,-1600)
  amplitudes=[abs(2*np.mean(y[cut]*np.exp(-2j*np.pi*f*t[cut]))) for f in freqs]
  self.assertLess(amplitudes[0],.01);self.assertGreater(amplitudes[1],.95);self.assertGreater(amplitudes[2],.95)
 def test_original_is_identical_and_input_is_not_mutated(self):
  x=np.random.default_rng(2).normal(size=(2,15872)).astype('float32');before=x.copy()
  self.assertTrue(np.array_equal(apply(x,'original'),x))
  for method in METHODS:apply(x,method);self.assertTrue(np.array_equal(x,before))
if __name__=='__main__':unittest.main()
