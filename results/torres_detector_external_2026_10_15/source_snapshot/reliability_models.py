"""PCEN and two-stage neural classification; always computer-only experiments."""
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from reproduction_models import mel_bank,normalize,ResidualBlock,DomainResidualBinary,MosquitoSongBinary


class PCENFrontend(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('window',torch.hann_window(1024))
        self.register_buffer('bank',torch.tensor(mel_bank(96)));self.smoothing=1-np.exp(-.008/.16)

    def forward(self,wave):
        spec=torch.stft(normalize(wave),1024,hop_length=64,window=self.window,center=True,pad_mode='reflect',return_complex=True).abs().square()
        energy=(self.bank@spec).clamp_min(1e-8)
        # Exact causal EMA, with M_0 = E_0, vectorized without audio extension.
        decay=(1-self.smoothing)**torch.arange(energy.shape[-1],device=wave.device,dtype=energy.dtype)
        smooth=decay*((1-self.smoothing)*energy[...,:1]+self.smoothing*(energy/decay).cumsum(-1))
        pcen=(energy/(1e-6+smooth).pow(.8)+2).sqrt()-np.sqrt(2)
        pcen=torch.log1p(pcen.clamp_min(0));pcen=(pcen-pcen.mean((-2,-1),keepdim=True))/pcen.std((-2,-1),keepdim=True).clamp_min(.05)
        db=10*energy.log10();db=(db-db.amax((-2,-1),keepdim=True)).clamp(-80,0)
        local=F.avg_pool2d(db[:,None],(13,1),stride=1,padding=(6,0),count_include_pad=False)
        return torch.cat([pcen[:,None],(db[:,None]+80)/80,(db[:,None]-local).clamp(-40,40)/20],1)


class JointDetector(nn.Module):
    def __init__(self,name,feature_mean,feature_scale):
        super().__init__();self.name=name;self.hybrid=name=='pcen_hybrid_large'
        if name.startswith('pcen'):
            self.frontend=PCENFrontend()
            self.encoder=nn.Sequential(ResidualBlock(3,64,(2,2)),ResidualBlock(64,128,(2,2)),
                ResidualBlock(128,256,(2,2)),ResidualBlock(256,512,(2,1)))
            self.project=nn.Sequential(nn.AdaptiveAvgPool2d((6,2)),nn.Flatten(),nn.Linear(512*6*2,768),
                nn.LayerNorm(768),nn.ReLU(),nn.Dropout(.4));size=768
        else:
            self.base=MosquitoSongBinary() if name=='raw_expanded' else DomainResidualBinary()
            size=128 if name=='raw_expanded' else 512
        self.register_buffer('feature_mean',torch.tensor(feature_mean,dtype=torch.float32))
        self.register_buffer('feature_scale',torch.tensor(feature_scale,dtype=torch.float32))
        if self.hybrid:
            self.physical=nn.Sequential(nn.Linear(138,256),nn.LayerNorm(256),nn.ReLU(),nn.Dropout(.5),nn.Linear(256,128),nn.ReLU())
            size+=128
        self.presence=nn.Linear(size,1);self.species=nn.Linear(size,1)

    def forward(self,wave,features):
        if self.name.startswith('pcen'):emb=self.project(self.encoder(self.frontend(wave)))
        else:_,emb=self.base(wave)
        if self.hybrid:
            x=((features-self.feature_mean)/self.feature_scale).clamp(-8,8)
            if self.training:x=x*(torch.rand(len(x),1,device=x.device)>.5)
            emb=torch.cat([emb,self.physical(x)],1)
        zp=self.presence(emb).squeeze(1);zs=self.species(emb).squeeze(1)
        probability=(zp.sigmoid()*zs.sigmoid()).clamp(1e-6,1-1e-6)
        return torch.logit(probability),zp,zs


def channel_augmentation(wave):
    """Simulation of microphone coloration; no held-out/private source is mixed."""
    w=normalize(wave);spectrum=torch.fft.rfft(w)
    response=F.interpolate(torch.empty(len(w),1,9,device=w.device).uniform_(-.7,.7),size=spectrum.shape[1],mode='linear',align_corners=True).exp().squeeze(1)
    w=torch.fft.irfft(spectrum*response,n=wave.shape[1]);return normalize(w)
