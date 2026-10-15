"""Fixed offline filters for sensitivity analysis, without changing audio duration."""
import numpy as np
from scipy.signal import butter,sosfiltfilt
METHODS={
 'original':{'description':'Existing model frontend only'},
 'highpass_120':{'description':'Butterworth highpass 120Hz, order4, forward/backward without padding'},
 'bandpass_150_3500':{'description':'Butterworth bandpass 150..3500Hz, order4, forward/backward without padding'},
 'bandpass_250_2500':{'description':'Butterworth bandpass 250..2500Hz, order4, forward/backward without padding'},
 'spectral_mild':{'description':'Within-window spectral power subtraction, noise q20%, alpha0.75, gain floor0.5'},
 'bandpass_spectral':{'description':'150..3500Hz bandpass followed by mild spectral subtraction'},
}

def spectral_mild(waves):
 """Estimate stationary power within each window; this can suppress flight tones.

FFT1024/hop256, overlap-add with a nonzero-ended Hann window. No padding;
any uncovered final samples keep their original values. No noise-only labels,
external profiles, duration extension, pitch shifting or gain amplification.
 """
 x=np.asarray(waves,dtype='float64');size,hop=1024,256
 win=np.hanning(size+1)[:-1]+.001
 frames=np.lib.stride_tricks.sliding_window_view(x,size,axis=-1)[...,::hop,:]
 z=np.fft.rfft(frames*win,axis=-1);power=np.abs(z)**2
 noise=np.quantile(power,.2,axis=-2,keepdims=True)
 gain=np.sqrt(np.maximum(1-.75*noise/np.maximum(power,1e-20),.25))
 restored=np.fft.irfft(z*gain,n=size,axis=-1)
 out=np.zeros_like(x);norm=np.zeros(x.shape[-1])
 for i in range(frames.shape[-2]):
  start=i*hop;out[...,start:start+size]+=restored[...,i,:]*win;norm[start:start+size]+=win**2
 covered=norm>0;out[...,covered]/=norm[covered];out[...,~covered]=x[...,~covered]
 return out.astype('float32')

def apply(waves,method):
 if method not in METHODS:raise ValueError(method)
 x=np.asarray(waves,dtype='float32')
 if method=='original':return x.copy()
 if method=='highpass_120':sos=butter(4,120,btype='highpass',fs=16000,output='sos')
 elif method in ['bandpass_150_3500','bandpass_spectral']:sos=butter(4,[150,3500],btype='bandpass',fs=16000,output='sos')
 elif method=='bandpass_250_2500':sos=butter(4,[250,2500],btype='bandpass',fs=16000,output='sos')
 else:return spectral_mild(x)
 y=sosfiltfilt(sos,x,axis=-1,padtype=None,padlen=0).astype('float32')
 return spectral_mild(y) if method=='bandpass_spectral' else y
