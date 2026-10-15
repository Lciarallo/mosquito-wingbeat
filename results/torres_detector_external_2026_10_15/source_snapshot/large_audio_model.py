"""Computer-only residual spectrogram CNN; no Arduino export or dependencies."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SR, SAMPLES, FFT, HOP, MELS = 16000, 15872, 2048, 256, 128
VERSION = 'large-residual-logmel-contrast-multitask-1'


def mel_bank():
    frequencies = np.fft.rfftfreq(FFT, 1/SR)
    to_mel = lambda hz: 2595*np.log10(1+hz/700)
    to_hz = lambda value: 700*(10**(value/2595)-1)
    edges = to_hz(np.linspace(to_mel(80),to_mel(4000),MELS+2))
    bank = np.zeros((MELS,len(frequencies)),dtype='float32')
    for i in range(MELS):
        bank[i]=np.maximum(0,np.minimum((frequencies-edges[i])/(edges[i+1]-edges[i]),
                                      (edges[i+2]-frequencies)/(edges[i+2]-edges[i+1])))
        bank[i]/=max(bank[i].sum(),1e-12)
    return bank


class Spectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('window',torch.hann_window(FFT))
        self.register_buffer('bank',torch.tensor(mel_bank()))

    def forward(self, wave):
        # Exactly the same per-window normalization for all sources and inference.
        wave=wave.float()-wave.float().mean(dim=-1,keepdim=True)
        wave=wave/wave.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(1e-6)
        spectrum=torch.stft(wave,n_fft=FFT,hop_length=HOP,window=self.window,
                            center=False,return_complex=True)/self.window.sum()
        power=spectrum.abs().square()
        db=10*torch.log10((self.bank@power).clamp_min(1e-12))
        db=(db-db.amax(dim=(-2,-1),keepdim=True)).clamp(-80,0)
        # Preserve absolute frequency positions; second channel removes broad
        # spectral coloration without purporting to remove environmental noise.
        smooth=F.avg_pool2d(db.unsqueeze(1),(15,1),stride=1,padding=(7,0),count_include_pad=False)
        contrast=(db.unsqueeze(1)-smooth).clamp(-40,40)/20
        logmel=(db-db.mean(dim=(-2,-1),keepdim=True))/20
        return torch.cat([logmel.unsqueeze(1),contrast],dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, pool=(2,2)):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(incoming,outgoing,3,padding=1,bias=False),
            nn.BatchNorm2d(outgoing),nn.ReLU(),nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False),
            nn.BatchNorm2d(outgoing))
        self.skip=nn.Sequential(nn.Conv2d(incoming,outgoing,1,bias=False),nn.BatchNorm2d(outgoing)) if incoming!=outgoing else nn.Identity()
        self.pool=nn.AvgPool2d(pool)

    def forward(self,x):
        return self.pool(F.relu(self.body(x)+self.skip(x)))


class LargeAudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend=Spectrogram()
        self.encoder=nn.Sequential(ResidualBlock(2,32),ResidualBlock(32,64),
            ResidualBlock(64,128),ResidualBlock(128,256,(2,1)))
        self.embedding=nn.Sequential(nn.AdaptiveAvgPool2d((8,2)),nn.Flatten(),
            nn.Linear(256*8*2,512),nn.ReLU(),nn.Dropout(.4))
        self.species=nn.Linear(512,20)
        self.presence=nn.Linear(512,1)

    def forward(self,wave,augment=False):
        x=self.frontend(wave)
        if augment:
            # Frequency tilt and small masks: no pitch shifting of wingbeats.
            tilt=(torch.rand(len(x),1,1,1,device=x.device)-.5)*.5
            grid=torch.linspace(-1,1,MELS,device=x.device).view(1,1,-1,1)
            x=x+tilt*grid
            fm=torch.arange(MELS,device=x.device).view(1,-1,1)
            start=torch.randint(0,MELS-5,(len(x),1,1),device=x.device)
            width=torch.randint(0,5,(len(x),1,1),device=x.device)
            mask=(fm>=start)&(fm<start+width)
            tm=torch.arange(x.shape[-1],device=x.device).view(1,1,-1)
            start=torch.randint(0,x.shape[-1]-6,(len(x),1,1),device=x.device)
            width=torch.randint(0,6,(len(x),1,1),device=x.device)
            mask=mask|((tm>=start)&(tm<start+width))
            x=x.masked_fill(mask.unsqueeze(1),0)
        z=self.embedding(self.encoder(x))
        return self.species(z),self.presence(z).squeeze(-1)


def load_model(checkpoint,device='cpu'):
    saved=torch.load(checkpoint,map_location=device,weights_only=True)
    model=LargeAudioCNN().to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model,saved


def predict(model,waves,device,batch=96):
    logits,gates=[],[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(waves),batch):
            value=torch.as_tensor(np.array(waves[start:start+batch],dtype='float32',copy=True),device=device)
            p,g=model(value)
            logits.append(p.cpu().numpy());gates.append(g.cpu().numpy())
    return np.concatenate(logits),np.concatenate(gates)
