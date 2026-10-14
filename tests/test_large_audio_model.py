"""Check frontend invariance and full checkpoint restoration for offline CNN."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from large_audio_model import Spectrogram,LargeAudioCNN,load_model,SAMPLES,VERSION


class LargeAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frontend_gain_and_dc_invariance(self):
        t=torch.arange(SAMPLES)/16000
        wave=(torch.sin(2*torch.pi*550*t)+.2*torch.sin(2*torch.pi*1100*t)).unsqueeze(0)
        frontend=Spectrogram()
        a,b=frontend(wave),frontend(wave*.005+.03)
        self.assertEqual(tuple(a.shape),(1,2,128,55))
        self.assertTrue(torch.isfinite(a).all())
        self.assertLess(float((a-b).abs().max()),.005)

    def test_silence_is_finite(self):
        features=Spectrogram()(torch.zeros(2,SAMPLES))
        self.assertTrue(torch.isfinite(features).all())

    def test_checkpoint_restores_exact_outputs(self):
        torch.manual_seed(3)
        model=LargeAudioCNN().eval()
        waveform=torch.randn(2,SAMPLES)
        with torch.no_grad():before=model(waveform)
        self.assertGreater(sum(p.numel() for p in model.parameters()),3000000)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cnn.pt'
            torch.save(dict(version=VERSION,state_dict=model.state_dict(),classes=[str(i) for i in range(20)]),path)
            restored,_=load_model(path)
            with torch.no_grad():after=restored(waveform)
        for a,b in zip(before,after):
            self.assertTrue(torch.equal(a,b))
        self.assertEqual(tuple(before[0].shape),(2,20))


if __name__=='__main__':unittest.main()
