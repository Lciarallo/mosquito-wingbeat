"""Small multiclass networks, trained with source-disjoint epoch selection."""
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


class TinySpeciesMLP:
    def __init__(self, hidden=(64,), seed=42):
        self.hidden = tuple(hidden)
        self.seed = seed

    def get_params(self):
        return dict(hidden=self.hidden, seed=self.seed, max_epochs=120,
                    patience=12, lr=.001, weight_decay=.001, batch=256)

    def fit(self, x, y, sample_weight, validation=None, fixed_epochs=None):
        torch.set_num_threads(4)
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler().fit(x)
        xx = self.scaler.transform(x).astype("float32")
        dimensions = (x.shape[1],) + self.hidden + (20,)
        layers = []
        for i, (a, b) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dimensions)-2:
                layers.append(nn.ReLU())
        network = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.001)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(xx), torch.from_numpy(y.astype("int64")))
        generator = torch.Generator().manual_seed(self.seed)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(sample_weight), len(y), replacement=True, generator=generator)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, sampler=sampler)
        best_loss, stale, best_state = float("inf"), 0, None
        self.history = []
        if validation is not None:
            vx, vy, vw = validation
            vx = torch.from_numpy(self.scaler.transform(vx).astype("float32")).to(device)
        epochs = fixed_epochs or 120
        self.best_epoch = epochs
        for epoch in range(1, epochs+1):
            network.train()
            for a, b in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(network(a.to(device)), b.to(device))
                loss.backward()
                optimizer.step()
            if validation is not None:
                network.eval()
                with torch.no_grad():
                    losses = nn.functional.cross_entropy(
                        network(vx), torch.from_numpy(vy.astype("int64")).to(device),
                        reduction="none").cpu().numpy()
                val_loss = float(np.average(losses, weights=vw))
                self.history.append(dict(epoch=epoch,source_class_weighted_validation_loss=val_loss))
                if val_loss < best_loss-1e-4:
                    best_loss, stale, self.best_epoch = val_loss, 0, epoch
                    best_state = {k:v.detach().clone() for k,v in network.state_dict().items()}
                else:
                    stale += 1
                if stale >= 12:
                    break
        if best_state is not None:
            network.load_state_dict(best_state)
        self.layer_weights = [layer.weight.detach().cpu().numpy() for layer in network if isinstance(layer,nn.Linear)]
        self.layer_biases = [layer.bias.detach().cpu().numpy() for layer in network if isinstance(layer,nn.Linear)]
        return self

    def decision_function(self, x):
        value = self.scaler.transform(x).astype("float32")
        for i, (w, b) in enumerate(zip(self.layer_weights,self.layer_biases)):
            value = value@w.T+b
            if i < len(self.layer_weights)-1:
                value = np.maximum(0,value)
        return value


def fused_layers(model):
    """Export preprocessing into the first layer, without a scaler on the MCU."""
    if isinstance(model,TinySpeciesMLP):
        scaler = model.scaler
        weights = [w.copy() for w in model.layer_weights]
        biases = [b.copy() for b in model.layer_biases]
    else:
        scaler, linear = model.steps[0][1], model.steps[-1][1]
        weights, biases = [linear.coef_.copy()], [linear.intercept_.copy()]
    weights[0] = weights[0]/scaler.scale_[None]
    biases[0] = biases[0]-weights[0]@scaler.mean_
    return [w.astype("float32") for w in weights], [b.astype("float32") for b in biases]


def portable_logits(bundle, x):
    value = np.asarray(x,dtype="float32")
    for i, (w,b) in enumerate(zip(bundle["weights"],bundle["biases"])):
        value = value@w.T+b
        if i < len(bundle["weights"])-1:
            value = np.maximum(0,value)
    return value


def probabilities(bundle, x):
    from scipy.special import softmax
    return softmax(portable_logits(bundle,x)/np.float32(bundle["temperature"]),axis=1)


def confirm_species(candidates, paths, starts):
    """Causal 2/3 vote requiring the current candidate and contiguous endpoints."""
    out = np.full(len(candidates),-1,dtype="int32")
    eligible = np.zeros(len(candidates),dtype=bool)
    history, previous_path, previous_start = [], None, None
    for i,(candidate,path,start) in enumerate(zip(candidates,paths,starts)):
        if path != previous_path or previous_start is None or abs(start-previous_start-1.) > 1e-5:
            history = []
        history = (history+[int(candidate)])[-3:]
        eligible[i] = len(history)==3
        if eligible[i] and candidate>=0 and history.count(int(candidate))>=2:
            out[i] = candidate
        previous_path, previous_start = path, start
    return out, eligible
