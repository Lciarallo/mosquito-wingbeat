import unittest
import numpy as np
from aedes_classical import features, FEATURE_NAMES


class AcousticFeaturesTests(unittest.TestCase):
    def test_dc_and_gain_do_not_change_features(self):
        time=np.arange(15872)/16000
        x=(.2*np.sin(2*np.pi*550*time)+.1*np.sin(2*np.pi*1100*time)).astype('float32')
        np.testing.assert_allclose(features(x),features(x*2+.03),atol=.02,rtol=.02)

    def test_short_audio_keeps_real_duration(self):
        x=np.zeros((2,4800),dtype='float32')
        original=x.copy();out=features(x)
        self.assertEqual(out.shape,(2,len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_array_equal(x,original)
        with self.assertRaises(ValueError):features(np.zeros(3000))

    def test_frequency_features_track_known_tone(self):
        t=np.arange(15872)/16000;x=.2*np.sin(2*np.pi*550*t)
        f=features(x)[0]
        self.assertLess(abs(f[FEATURE_NAMES.index('low_peak_hz_mean')]-550),10)
        self.assertLess(abs(f[FEATURE_NAMES.index('centroid_hz_mean')]-550),10)

if __name__=='__main__':unittest.main()
