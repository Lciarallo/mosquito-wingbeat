import unittest
import numpy as np
import pandas as pd
from aedes_binary import targets, weights, decisions


class BinaryProtocolTests(unittest.TestCase):
    def test_unresolved_mosquito_is_not_a_negative(self):
        f=pd.DataFrame({'label':['Aedes aegypti','unknown_mosquito','noise','Aedes albopictus','not_aedes']})
        np.testing.assert_array_equal(targets(f),[1,-1,0,0,0])

    def test_source_oversampling_does_not_dominate(self):
        f=pd.DataFrame({'label':['Aedes aegypti']*5+['noise','Aedes albopictus'],
                        'domain':['a']*4+['b','a','a'],
                        'group':['one']*4+['two','three','four']})
        w=weights(f);y=targets(f)
        self.assertAlmostEqual(w[y==1].sum(),w[y==0].sum())
        self.assertAlmostEqual(w[:4].sum(),w[4])
        self.assertAlmostEqual(w[5],w[6])

    def test_reject_band_and_invalid_audio(self):
        states=decisions(np.array([.01,.4,.95,.99]),[True,True,True,False],.2,.8)
        self.assertEqual(states.tolist(),['NAO_AEDES_AEGYPTI','INCERTO','AEDES_AEGYPTI','AUDIO_INVALIDO'])

if __name__=='__main__':unittest.main()
