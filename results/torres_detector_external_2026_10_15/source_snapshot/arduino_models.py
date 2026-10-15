"""Small neural binary model with source-disjoint epoch selection."""
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
SEED=42

class TinyPresenceMLP:
    """One hidden layer, selected epoch using disjoint source validation only."""
    def get_params(self):
        return dict(hidden=16,max_epochs=80,patience=10,lr=.001,weight_decay=.001,batch=256,seed=SEED)
    def fit(self,x,y,sample_weight,validation=None,fixed_epochs=None):
        torch.set_num_threads(4);torch.manual_seed(SEED)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler=StandardScaler().fit(x)
        xx=self.scaler.transform(x).astype("float32")
        network=nn.Sequential(nn.Linear(x.shape[1],16),nn.ReLU(),nn.Linear(16,1)).to(device)
        optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.001)
        dataset=torch.utils.data.TensorDataset(torch.from_numpy(xx),torch.from_numpy(y.astype("float32")))
        generator=torch.Generator().manual_seed(SEED)
        sampler=torch.utils.data.WeightedRandomSampler(torch.from_numpy(sample_weight),len(y),replacement=True,generator=generator)
        loader=torch.utils.data.DataLoader(dataset,batch_size=256,sampler=sampler)
        best_loss=float("inf");stale=0;best_state=None;self.history=[]
        if validation is not None:
            vx,vy,vw=validation
            vx=torch.from_numpy(self.scaler.transform(vx).astype("float32")).to(device)
        epochs=fixed_epochs or 80;self.best_epoch=epochs
        for epoch in range(1,epochs+1):
            network.train()
            for a,b in loader:
                optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.binary_cross_entropy_with_logits(network(a.to(device)).squeeze(-1),b.to(device))
                loss.backward();optimizer.step()
            if validation is not None:
                network.eval()
                with torch.no_grad():
                    logits=network(vx).squeeze(-1).cpu().numpy()
                val_loss=float(np.average(np.logaddexp(0,logits)-vy*logits,weights=vw))
                self.history.append(dict(epoch=epoch,source_weighted_validation_loss=val_loss))
                if val_loss<best_loss-1e-4:
                    best_loss=val_loss;self.best_epoch=epoch;stale=0
                    best_state={key:value.detach().clone() for key,value in network.state_dict().items()}
                else: stale+=1
                if stale>=10: break
        if best_state is not None: network.load_state_dict(best_state)
        self.w1=network[0].weight.detach().cpu().numpy();self.b1=network[0].bias.detach().cpu().numpy()
        self.w2=network[2].weight.detach().cpu().numpy()[0];self.b2=float(network[2].bias.detach().cpu().numpy()[0])
        return self
    def decision_function(self,x):
        hidden=np.maximum(0,self.scaler.transform(x).astype("float32")@self.w1.T+self.b1)
        return hidden@self.w2+self.b2
