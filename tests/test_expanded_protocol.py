"""Check source balancing and streaming policy in the expansion pipeline."""
import unittest
import numpy as np
import pandas as pd
from train_expanded import source_weights, decide


class ExpandedProtocolTests(unittest.TestCase):
    def test_each_class_and_domain_has_equal_total_weight(self):
        frame = pd.DataFrame(dict(group=['a','a','b','c','c','d','e'],
                                  domain=['old','old','old','new','new','new','old']))
        target = np.array([0,0,0,0,0,0,1])
        weights = source_weights(frame, target)
        self.assertAlmostEqual(weights[target==0].sum(), weights[target==1].sum())
        self.assertAlmostEqual(weights[:3].sum(), weights[3:6].sum())
        self.assertAlmostEqual(weights[:2].sum(), weights[2])
        self.assertAlmostEqual(weights[3:5].sum(), weights[5])

    def test_confirmations_reset_after_gap_and_invalid_audio(self):
        frame = pd.DataFrame(dict(path=['a']*8,start_s=[0,.992,1.984,4,4.992,5.984,6.976,7.968],
                                  step_s=[.992]*8,quality_ok=[True]*5+[False,True,True]))
        p = np.tile([.9,.1],(8,1))
        _, result = decide(frame,p,np.ones(8),.5,.5)
        self.assertEqual(result.tolist(),[-1,-1,0,-1,-1,-1,-1,-1])

    def test_current_uncertain_window_cannot_emit_stale_species(self):
        frame = pd.DataFrame(dict(path=['a']*4,start_s=[0,.992,1.984,2.976],
                                  step_s=[.992]*4,quality_ok=[True]*4))
        p = np.array([[.9,.1],[.9,.1],[.51,.49],[.9,.1]])
        _, result = decide(frame,p,np.ones(4),.5,.8)
        self.assertEqual(result.tolist(),[-1,-1,-1,0])


if __name__ == '__main__':
    unittest.main()
