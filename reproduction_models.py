"""Computer-only binary models adapted from acoustic mosquito papers.

The architectures are reimplemented in PyTorch, not claimed to reproduce the
published metrics. No original-recording group is split into independent folds.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mel_bank(bands=60, fft=1024, sr=8000):
    # Slaney scale and area normalization, matching the authors' librosa default.
    def to_mel(hz):
        hz=np.asarray(hz);return np.where(hz<1000,hz/(200/3),15+np.log(np.maximum(hz,1)/1000)/(np.log(6.4)/27))
    def to_hz(m):
        return np.where(m<15,m*(200/3),1000*np.exp((m-15)*(np.log(6.4)/27)))
    edges=to_hz(np.linspace(0,to_mel(sr/2),bands+2));freq=np.fft.rfftfreq(fft,1/sr)
    bank=np.maximum(0,np.minimum((freq[None,:]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),
         (edges[2:,None]-freq[None,:])/(edges[2:,None]-edges[1:-1,None])))
    return (bank*2/(edges[2:,None]-edges[:-2,None])).astype('float32')


def normalize(wave):
    wave=wave-wave.mean(-1,keepdim=True)
    return wave/wave.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)


class LogMel(nn.Module):
    def __init__(self,bands=60,hop=256,contrast=False):
        super().__init__();self.hop=hop;self.contrast=contrast
        self.register_buffer('window',torch.hann_window(1024))
        self.register_buffer('bank',torch.tensor(mel_bank(bands)))

    def forward(self,wave):
        w=normalize(wave.float())
        spec=torch.stft(w,n_fft=1024,hop_length=self.hop,window=self.window,
                        center=True,pad_mode='reflect',return_complex=True).abs().square()
        db=10*torch.log10((self.bank@spec).clamp_min(1e-12))
        db=(db-db.amax((-2,-1),keepdim=True)).clamp(-80,0)
        if not self.contrast:
            # Fixed spectrogram image resolution; no audio padding/repetition.
            return F.interpolate((db/80+1).unsqueeze(1),size=(60,40),mode='bilinear',align_corners=False)
        smooth=F.avg_pool2d(db.unsqueeze(1),(13,1),stride=1,padding=(6,0),count_include_pad=False)
        return torch.cat([(db-db.mean((-2,-1),keepdim=True)).unsqueeze(1)/20,
                          (db.unsqueeze(1)-smooth).clamp(-40,40)/20],1)


class FernandesBinary(nn.Module):
    """32/64/64 valid 3x3 convolutions and 256-unit dense layer from binary.py.

Changes: single BCE output, explicit axis order, and spectrogram time resizing
for real-duration 0.3/0.992s input rather than discarding short recordings.
"""
    def __init__(self):
        super().__init__();self.frontend=LogMel()
        self.encoder=nn.Sequential(nn.Conv2d(1,32,3),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,64,3),nn.ReLU())
        self.project=nn.Sequential(nn.Flatten(),nn.Linear(64*11*6,256),nn.ReLU())
        self.head=nn.Linear(256,1)

    def forward(self,wave):
        emb=self.project(self.encoder(self.frontend(wave)))
        return self.head(emb).squeeze(1),emb


class MosquitoSongBinary(nn.Module):
    """MosquitoSong+ Figure 2 convolutions, strides, pool and dense dimensions.

Output changed from species/sex softmax to one binary BCE logit. 300ms input.
"""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(1,32,100,stride=4),nn.ReLU(),
            nn.Conv1d(32,32,64,stride=4),nn.ReLU(),nn.Conv1d(32,64,64,stride=3),nn.ReLU(),
            nn.MaxPool1d(3,stride=3),nn.Dropout(.5))
        self.project=nn.Sequential(nn.Flatten(),nn.Linear(64*7,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.head=nn.Linear(128,1)

    def forward(self,wave):
        if wave.shape[-1]!=2400:raise ValueError('300ms real audio at 8kHz required.')
        emb=self.project(self.encoder(normalize(wave).unsqueeze(1)))
        return self.head(emb).squeeze(1),emb


class ResidualBlock(nn.Module):
    def __init__(self,incoming,outgoing,pool):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(incoming,outgoing,3,padding=1,bias=False),nn.BatchNorm2d(outgoing),nn.ReLU(),
            nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False),nn.BatchNorm2d(outgoing))
        self.skip=nn.Sequential(nn.Conv2d(incoming,outgoing,1,bias=False),nn.BatchNorm2d(outgoing))
        self.pool=nn.AvgPool2d(pool)

    def forward(self,x):return self.pool(F.relu(self.body(x)+self.skip(x)))


class DomainResidualBinary(nn.Module):
    """Residual spectrogram network with optional species-conditional alignment.

Inspired by residual acoustic CNNs and DR-BioL; not their exact architecture.
"""
    def __init__(self):
        super().__init__();self.frontend=LogMel(bands=96,hop=64,contrast=True)
        self.encoder=nn.Sequential(ResidualBlock(2,32,(2,2)),ResidualBlock(32,64,(2,2)),
            ResidualBlock(64,128,(2,2)),ResidualBlock(128,256,(2,1)))
        self.project=nn.Sequential(nn.AdaptiveAvgPool2d((6,2)),nn.Flatten(),nn.Linear(256*6*2,512),
            nn.LayerNorm(512),nn.ReLU(),nn.Dropout(.3))
        self.head=nn.Linear(512,1)

    def forward(self,wave):
        emb=self.project(self.encoder(self.frontend(wave)))
        return self.head(emb).squeeze(1),emb


def cross_domain_loss(embedding,species,domain):
    """Contrast same known species across domains, without merging all negatives."""
    z=F.normalize(embedding,dim=1);s=z@z.T/.15
    eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    positive=species[:,None].eq(species[None,:])&~domain[:,None].eq(domain[None,:])&~eye
    valid=positive.any(1)
    if not valid.any():return embedding.sum()*0
    logp=s-s.masked_fill(eye,-1e4).logsumexp(1,keepdim=True)
    contrast=-(logp.masked_fill(~positive,0).sum(1)/positive.sum(1).clamp_min(1))[valid].mean()
    alignment=[]
    for label in species.unique():
        means=[z[species.eq(label)&domain.eq(d)].mean(0) for d in domain[species.eq(label)].unique()
               if int((species.eq(label)&domain.eq(d)).sum())>=2]
        if len(means)>1:
            means=torch.stack(means);alignment.append((means-means.mean(0)).square().mean())
    return .05*contrast+(torch.stack(alignment).mean() if alignment else embedding.sum()*0)


def build_model(name):
    if name.startswith('fernandes'):return FernandesBinary()
    if name=='mosquitosong_binary':return MosquitoSongBinary()
    if name.startswith('residual'):return DomainResidualBinary()
    raise ValueError(name)
