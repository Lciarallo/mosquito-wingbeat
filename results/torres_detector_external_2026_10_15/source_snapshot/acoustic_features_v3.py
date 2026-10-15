"""Inductive 1-second frontend: retained timbre, local noise floor, 100/200 ms STFT.

Only DC cepstrum is suppressed. Scalar log-mel centring within each window
preserves relative spectral shape; it is not cross-recording or batch CMN.
Arbitrary microphone response cannot be disentangled from source timbre using
one unlabelled window alone. Device weighting and PCM isolation remain required.
"""
import numpy as np
from scipy import signal
from scipy.fft import dct
from scipy.ndimage import median_filter

SR = 16000
SHORT_NPERSEG, SHORT_HOP = 1600, 160
LONG_NPERSEG, LONG_HOP = 3200, 320
NOISE_MEDIAN_HZ = 150
PSD_GRID = np.arange(150, 3951, 50)
CONTRAST_GRID = np.arange(200, 3901, 50)
F0_GRID = np.arange(200, 901, 5)
FEATURE_NAMES = (
    [f'psd_local_floor_{f}' for f in PSD_GRID]
    + [f'contrast_short_mean_{f}' for f in CONTRAST_GRID]
    + [f'contrast_short_std_{f}' for f in CONTRAST_GRID]
    + [f'mfcc_centred_mean_{i}' for i in range(13)]
    + [f'mfcc_short_std_{i}' for i in range(13)]
    + [f'mfcc_short_delta_abs_{i}' for i in range(13)]
    + [f'mfcc_short_delta_std_{i}' for i in range(13)]
    + [f'shs_long_peak{k}_{v}' for k in range(3) for v in ['f0', 'strength', 'energy_h2_h1', 'energy_h3_h1']]
    + ['shs_short_f0_median', 'shs_short_f0_std', 'shs_short_f0_q10', 'shs_short_f0_q90',
       'shs_long_entropy', 'harmonic_energy_fraction', 'shs_long_spread',
       'shs_long_top2_ratio', 'spectral_flatness', 'spectral_centroid',
       'spectral_bandwidth', 'harmonic_salience',
       'energy_500_1600_fraction', 'energy_500_1600_vs_200_500',
       'shs_long_f0_median', 'shs_long_f0_std', 'shs_short_long_disagreement']
)


def _mel_bank(f):
    mel=lambda x:2595*np.log10(1+x/700)
    bounds=700*(10**(np.linspace(mel(100),mel(3900),34)/2595)-1)
    bank=np.maximum(0,np.minimum(
        (f[None,:]-bounds[:-2,None])/np.diff(bounds)[:-1,None],
        (bounds[2:,None]-f[None,:])/np.diff(bounds)[1:,None]))
    return bank/np.maximum(bank.sum(1,keepdims=True),1e-12)


SHORT_F = np.fft.rfftfreq(SHORT_NPERSEG,1/SR)
LONG_F = np.fft.rfftfreq(LONG_NPERSEG,1/SR)
MEL_BANK_SHORT = _mel_bank(SHORT_F)


def stft_power(wave,nperseg,hop):
    f,t,z=signal.stft(wave,fs=SR,nperseg=nperseg,noverlap=nperseg-hop,
                      boundary=None,padded=False)
    return f,t,np.abs(z)**2


def local_noise_floor(power,frequency_spacing):
    # 31 bins at 5 Hz span 150 Hz. 15 bins at 10 Hz are the nominal
    # 150-Hz short-STFT neighbourhood. No slope regression or batch scaling.
    width=int(round(NOISE_MEDIAN_HZ/frequency_spacing))
    if width%2==0:width+=1
    size=width if np.ndim(power)==1 else (width,1)
    floor=median_filter(power,size=size,mode='nearest')
    return floor,np.maximum(power-floor,0)


def centred_cepstrum(denoised_short_power):
    mel_db=10*np.log10(np.maximum(MEL_BANK_SHORT@denoised_short_power,1e-12))
    # A scalar offset per time frame within THIS one-second window; unlike
    # coefficient-wise temporal subtraction, it retains c1/c2 and formant tilt.
    mel_db-=np.median(mel_db,axis=0,keepdims=True)
    cep=dct(mel_db,type=2,axis=0,norm='ortho')[:13]
    means=cep.mean(1);means[0]=0.
    delta=np.diff(cep,axis=1)
    return means,cep.std(1),np.abs(delta).mean(1),delta.std(1)


def harmonic_comb(f,power,floor):
    evidence=np.log1p(np.maximum(np.clip(power/np.maximum(floor,1e-12),0,50)-1,0))
    positions=F0_GRID[None,:]*np.arange(1,5)[:,None]
    right=np.clip(np.searchsorted(f,positions),1,len(f)-1)
    fraction=(positions-f[right-1])/(f[right]-f[right-1])
    components=(1-fraction[:,:,None])*evidence[right-1]+fraction[:,:,None]*evidence[right]
    shs=np.maximum(np.sum(components/np.arange(1,5)[:,None,None],axis=0)
                   -.5*np.maximum(components[1]-components[0],0),0)
    return shs


def harmonic_energy_ratios(f,denoised_power,f0):
    energies=np.array([denoised_power[np.abs(f-f0*h)<=15].sum() for h in (1,2,3)])
    return energies[1]/max(energies[0],1e-12),energies[2]/max(energies[0],1e-12)


def window_features(raw_wave):
    w=np.asarray(raw_wave,dtype=np.float64)
    if w.ndim!=1 or len(w)!=SR or not np.isfinite(w).all():
        raise ValueError('Expected exactly one finite mono 1 s window; no padding.')
    w=w-w.mean();w=w/max(float(np.sqrt(np.mean(w*w))),1e-8)
    f,_,long=stft_power(w,LONG_NPERSEG,LONG_HOP)
    fs,ts,short=stft_power(w,SHORT_NPERSEG,SHORT_HOP)
    long_floor,_=local_noise_floor(long,5.)
    short_floor,short_clean=local_noise_floor(short,10.)
    mean_power=long.mean(1);_,denoised=local_noise_floor(mean_power,5.)
    psd=10*np.log10(np.maximum(np.interp(PSD_GRID,f,denoised),1e-12))
    psd-=np.median(psd)  # Gain offset only; relative harmonic levels survive.
    contrast=10*np.log10(np.maximum(short,1e-12)/np.maximum(short_floor,1e-12))
    cep=centred_cepstrum(short_clean)
    long_shs=harmonic_comb(f,long,long_floor);short_shs=harmonic_comb(fs,short,short_floor)
    profile=long_shs.mean(1)
    peaks,_=signal.find_peaks(profile,distance=5)
    if not len(peaks):peaks=np.array([int(profile.argmax())])
    ranked=sorted(peaks,key=lambda i:(-profile[i],int(i)))[:3]
    peak_features=[]
    for i in ranked:
        r2,r3=harmonic_energy_ratios(f,denoised,F0_GRID[i])
        peak_features.extend([F0_GRID[i],profile[i],min(r2,100.),min(r3,100.)])
    peak_features.extend([0.]*(12-len(peak_features)))
    short_track=F0_GRID[short_shs.argmax(0)];long_track=F0_GRID[long_shs.argmax(0)]
    q=profile/max(float(profile.sum()),1e-12)
    band=(f>=150)&(f<=3900);spectral=mean_power[band]/max(float(mean_power[band].sum()),1e-12)
    centroid=float(np.dot(f[band],spectral));bandwidth=float(np.sqrt(np.dot((f[band]-centroid)**2,spectral)))
    mask=np.zeros(len(f),bool)
    for i in ranked:
        for h in (1,2,3,4):mask|=np.abs(f-F0_GRID[i]*h)<=15
    energy_fraction=float(denoised[mask&band].sum()/max(float(denoised[band].sum()),1e-12))
    salience=float(np.clip(.1+profile.max()/3*np.sqrt(energy_fraction),.1,3.))
    mid=denoised[(f>=500)&(f<=1600)].sum();low=denoised[(f>=200)&(f<500)].sum()
    whole=denoised[(f>=200)&(f<=3900)].sum()
    # Match each short frame to the long-STFT track for disagreement descriptor.
    long_times=np.arange(len(long_track))*LONG_HOP/SR+LONG_NPERSEG/(2*SR)
    aligned=np.interp(ts,long_times,long_track)
    summary=[np.median(short_track),np.std(short_track),*np.quantile(short_track,[.1,.9]),
             -np.sum(q*np.log(np.maximum(q,1e-12)))/np.log(len(q)),energy_fraction,
             np.sqrt(np.dot((F0_GRID-np.dot(F0_GRID,q))**2,q)),
             profile[ranked[1]]/max(profile[ranked[0]],.1) if len(ranked)>1 else 0.,
             np.exp(np.log(np.maximum(spectral,1e-12)).mean())/max(spectral.mean(),1e-12),
             centroid,bandwidth,salience,mid/max(float(whole),1e-12),
             np.log1p(mid/max(float(low),1e-12)),np.median(long_track),np.std(long_track),
             np.mean(np.abs(short_track-aligned))]
    features=np.r_[psd,np.interp(CONTRAST_GRID,fs,contrast.mean(1)),
                   np.interp(CONTRAST_GRID,fs,contrast.std(1)),*cep,peak_features,summary].astype(np.float32)
    if len(features)!=len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError('Nonfinite or misaligned features.')
    return features,salience


def extract_features(windows):
    rows=[window_features(w) for w in windows]
    return np.stack([r[0] for r in rows]),np.array([r[1] for r in rows])
