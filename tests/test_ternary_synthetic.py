"""Physical mixing, held-out ancestry, abstentions and frontend consistency."""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import evaluate_ternary_synthetic as synthetic


def test_mixture_ratio_measured_from_independent_frequency_components():
    t = np.arange(synthetic.SR) / synthetic.SR
    signal = .6 * np.sin(2*np.pi*400*t)
    other = .8 * np.cos(2*np.pi*900*t)
    mixed, declared = synthetic.mix_at_ratio(signal, other, -10)
    spectrum = np.abs(np.fft.rfft(mixed))
    assert 20*np.log10(spectrum[400]/spectrum[900]) == pytest.approx(-10, abs=1e-8)
    assert declared == pytest.approx(-10)
    assert np.abs(mixed).max() <= .9500001
    with pytest.raises(ValueError):
        synthetic.mix_at_ratio(signal, other[:-1], 0)


def test_pink_noise_has_more_low_frequency_power_than_white_noise():
    white = synthetic.coloured_noise(64000, synthetic.generator("white"), "white")
    pink = synthetic.coloured_noise(64000, synthetic.generator("pink"), "pink")
    frequency = np.fft.rfftfreq(len(white), 1/synthetic.SR)
    def ratio(wave):
        power = np.abs(np.fft.rfft(wave))**2
        return power[(frequency>=50)&(frequency<250)].mean()/power[(frequency>=2000)&(frequency<4000)].mean()
    assert ratio(pink) > 10*ratio(white)


def test_control_and_attenuation_preserve_length_and_do_not_mutate_donor():
    wave = np.sin(np.arange(synthetic.SIZE)*.2).astype('float32')*.1
    before = wave.copy()
    control, _ = synthetic.transform(wave, "clean_control", synthetic.generator("control"))
    attenuated, _ = synthetic.transform(wave, "attenuation18db", synthetic.generator("quiet"))
    assert np.array_equal(control, before) and np.array_equal(wave, before)
    assert len(attenuated) == len(wave)
    assert 20*np.log10(synthetic.rms(attenuated)/synthetic.rms(wave)) == pytest.approx(-18,abs=1e-6)


def test_donor_selection_never_labels_noise_windows_in_mosquito_groups_as_pure_noise():
    rows, partition = [], []
    for fold in range(2):
        for group, labels in [("noise_a",[0]),("noise_b",[0]),("mixed",[0,1]),("other",[1]),("aedes",[2])]:
            for label in labels:
                row_id=len(rows)
                rows.append(dict(row_id=row_id,group=f"{fold}:{group}",true_label=label,
                                 domain="source",label=str(label),pcm_block=f"{fold}:{group}"))
                partition.append(dict(row_id=row_id,fold=fold,role="test"))
    frame=pd.DataFrame(rows)
    parents=synthetic.choose_parents(frame,pd.DataFrame(partition),per_negative_fold=1)
    group_truth=frame.groupby('group').true_label.max()
    assert parents.group.nunique()==len(parents)==6
    assert parents.true_label.eq(parents.group.map(group_truth)).all()
    assert not parents.loc[parents.true_label.eq(0),'group'].str.contains('mixed').any()


@pytest.mark.parametrize('bundle',[
    {'fit_rows':[1],'fit_pcm_blocks':['unrelated']},
    {'fit_rows':[999],'fit_pcm_blocks':['related_original']},
])
def test_related_donor_rows_or_pcm_blocks_cannot_enter_heldout_model(bundle):
    parent=pd.DataFrame({'row_id':[1],'pcm_block':['related_original']})
    with pytest.raises(ValueError):
        synthetic.assert_heldout(bundle,parent)


def test_batch_frontend_matches_actual_inference_and_preserves_invalid_windows(tmp_path):
    from predict_ternary import audio_features
    paths=[]
    for key, wave in [('tone',synthetic.procedural('fan_harmonics',synthetic.generator('feature'))),
                      ('silent',np.zeros(synthetic.SIZE,dtype='float32'))]:
        path=tmp_path/(key+'.wav');sf.write(path,wave,synthetic.SR,subtype='FLOAT');paths.append(path)
    manifest=pd.DataFrame([dict(clip_id=p.stem,path=str(p),suite='procedural_negatives',
                                parent_pcm_block=None,true_label=0,fold=-1) for p in paths])
    frame,x=synthetic.audio_batch(manifest,batch_size=2)
    for path in paths:
        expected_frame,expected_x,_=audio_features(path)
        mask=frame.group.eq(path.stem)
        np.testing.assert_allclose(x[mask],expected_x,rtol=2e-6,atol=2e-6)
        assert np.array_equal(frame.loc[mask,'quality_ok'],expected_frame.quality_ok)
    assert not frame.loc[frame.group.eq('silent'),'quality_ok'].any()


def test_invalid_predictions_remain_in_positive_and_negative_denominators():
    result=synthetic.metrics([0,1,2,2],[-1,1,2,-1])
    assert result['aedes_total']==2 and result['aedes_tp']==1
    assert result['aedes_recall']==.5 and result['coverage']==.5
    assert result['noise_total']==1 and result['invalid_audio']==2
    assert result['confusion_rows_noise_other_aedes_columns_noise_other_aedes_invalid'][0]==[0,0,0,1]


def test_short_donor_is_rejected_instead_of_repeated(monkeypatch):
    monkeypatch.setattr(synthetic,'source_wave',lambda path: np.ones(5000,dtype='float32'))
    with pytest.raises(ValueError,match='real 0.992'):
        synthetic.donor_wave(pd.Series({'path':'short.wav','start_s':0}))
