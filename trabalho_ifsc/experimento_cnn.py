"""CNN didática para o trabalho IFSC: treino real e avaliação retrospectiva.

Executar da raiz: .venv/bin/python trabalho_ifsc/experimento_cnn.py
Não altera modelos anteriores, firmware ou instaladores. Recusa sobrescrever.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from large_audio_model import Spectrogram


class CNNDidatica(nn.Module):
    """Quatro convoluções aprendidas; entrada 2 x 64 x 55; 20 espécies."""
    def __init__(self, classes: int = 20):
        super().__init__()
        layers = []
        for incoming, outgoing in [(2, 16), (16, 32), (32, 64), (64, 96)]:
            layers += [nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                       nn.BatchNorm2d(outgoing), nn.ReLU(), nn.AvgPool2d(2)]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
                                  nn.Linear(384, 128), nn.ReLU(), nn.Dropout(.3),
                                  nn.Linear(128, classes))

    def forward(self, x):
        return self.head(self.encoder(x))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def frontend(waves, device):
    """Transformação fixa, individual, sem estatísticas aprendidas no teste."""
    module = Spectrogram().to(device).eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(waves), 128):
            wave = torch.tensor(np.array(waves[start:start+128]), device=device)
            values.append(F.avg_pool2d(module(wave), (2, 1)).cpu().numpy())
    return np.concatenate(values)


def predict(model, features, device):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            x = torch.tensor(features[start:start+256], device=device)
            result.append(model(x).cpu().numpy())
    return softmax(np.concatenate(result), axis=1)


def balanced_recording_weights(frame):
    # Uma espécie = uma massa; uma gravação dessa espécie = uma massa.
    groups = frame.groupby('label')['group'].nunique()
    windows = frame.groupby('group').size()
    w = 1 / (frame.label.map(groups).to_numpy() * frame.group.map(windows).to_numpy())
    return w / w.mean()


def evaluate(frame, probabilities, classes, model_name, out):
    y = frame.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    table = pd.DataFrame(probabilities, columns=classes)
    table['group'] = frame.group.to_numpy()
    gp = table.groupby('group')[classes].mean()
    labels = frame.groupby('group').label.first().reindex(gp.index)
    gy = labels.map({c: i for i, c in enumerate(classes)}).to_numpy()
    pred = gp.to_numpy().argmax(axis=1)
    record = pd.DataFrame({'group': gp.index, 'truth': labels,
                           'prediction': [classes[i] for i in pred],
                           'confidence': gp.to_numpy().max(axis=1)})
    record.to_csv(out / f'{model_name}_recordings.csv', index=False)
    cm = confusion_matrix(gy, pred, labels=np.arange(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / f'{model_name}_confusion.csv')
    return dict(model=model_name, windows=len(frame), recordings=len(gp),
                classes_observed=int(labels.nunique()),
                correct_recordings=int((pred == gy).sum()),
                window_accuracy=float(accuracy_score(y, probabilities.argmax(axis=1))),
                recording_accuracy=float(accuracy_score(gy, pred)),
                recording_balanced_accuracy=float(balanced_accuracy_score(gy, pred)),
                recording_macro_f1=float(f1_score(gy, pred, average='macro', zero_division=0)))


def train(output, epochs=30):
    if output.exists():
        raise FileExistsError(f'Escolha outra pasta; resultados preservados: {output}')
    output.mkdir(parents=True)
    seed = 20261011
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Mesma seleção de registros do corpus Dryad; os splits são históricos.
    all_frame = pd.read_csv(ROOT / 'results/expanded/segments.csv', low_memory=False)
    cache_frame = pd.read_csv(ROOT / 'results/segments.csv', low_memory=False)
    assert all_frame.iloc[:len(cache_frame)].path.tolist() == cache_frame.path.tolist()
    assert np.allclose(all_frame.iloc[:len(cache_frame)].start_s, cache_frame.start_s)
    assert all_frame.iloc[:len(cache_frame)].content_sha256.tolist() == cache_frame.content_sha256.tolist()
    selection = all_frame.domain.eq('dryad') & all_frame.label.ne('noise') & all_frame.quality_ok
    frame = all_frame.loc[selection].copy()
    assert frame.groupby('group').split.nunique().max() == 1
    assert frame.groupby('path').split.nunique().max() == 1
    assert frame.groupby('content_sha256').split.nunique().max() == 1
    assert frame.groupby('group').label.nunique().max() == 1
    classes = sorted(frame.label.unique().tolist())
    labels = frame.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    source = np.load(ROOT / 'data/cache/large_cnn/waves.npy', mmap_mode='r')
    assert len(source) == len(all_frame)
    features = frontend(source[frame.index], device)
    splits = {name: np.flatnonzero(frame.split.eq(name)) for name in ['train', 'validation', 'calibration', 'test']}
    audit = {name: dict(windows=len(idx), recordings=int(frame.iloc[idx].group.nunique()),
                       classes=int(frame.iloc[idx].label.nunique())) for name, idx in splits.items()}
    frame[['path', 'group', 'label', 'split', 'start_s', 'content_sha256']].to_csv(output / 'partitions.csv', index=False)
    model = CNNDidatica(len(classes)).to(device)
    protocol = dict(seed=seed, architecture='4 Conv2D+BN+ReLU+AvgPool; dense128; softmax20',
                    parameters=sum(p.numel() for p in model.parameters()), device=str(device),
                    sample_rate=16000, samples=15872, seconds=.992,
                    fft=2048, hop=256, input_shape=list(features.shape[1:]),
                    frequency_range_hz=[80, 4000],
                    input='log-mel e contraste; 128 bandas agrupadas em 64; normalização por janela',
                    splits=audit, epochs_max=epochs, patience=6, batch=128,
                    optimizer='AdamW, lr=0.001, weight_decay=0.01',
                    selection='menor entropia cruzada de validação, equilibrada por espécie e gravação',
                    class_recording_balancing='amostragem inversa de janelas/gravação e gravações/espécie',
                    calibration='reservada, não usada: sem limiar ou rejeição neste experimento',
                    split_csv_sha256=sha(ROOT / 'results/expanded/segments.csv'),
                    source_code_sha256=sha(__file__), frontend_code_sha256=sha(ROOT / 'large_audio_model.py'),
                    historical_test_previously_inspected=True,
                    interpretation='avaliação interna retrospectiva; grupos não garantem indivíduo/sessão independente',
                    presence_evaluated=False, firmware_changed=False, field_validated=False)
    save_json(output / 'protocol.json', protocol)
    tr, va = splits['train'], splits['validation']
    train_x = torch.tensor(features[tr], device=device)
    train_y = torch.tensor(labels[tr], device=device)
    weights = torch.tensor(balanced_recording_weights(frame.iloc[tr]), device=device, dtype=torch.float32)
    vw = balanced_recording_weights(frame.iloc[va])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best_loss, stale, history, best = float('inf'), 0, [], None
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.multinomial(weights, len(tr), replacement=True)
        losses = []
        for start in range(0, len(order), 128):
            idx = order[start:start+128]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        p = predict(model, features[va], device)
        vloss = float(np.average(-np.log(p[np.arange(len(va)), labels[va]].clip(1e-12)), weights=vw))
        entry = dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_loss=vloss,
                     validation_window_accuracy=float((p.argmax(axis=1) == labels[va]).mean()))
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if vloss < best_loss - 1e-4:
            best_loss, stale, best_epoch = vloss, 0, epoch
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(best)
    torch.save(dict(state_dict=best, classes=classes, protocol=protocol, best_epoch=best_epoch), output / 'cnn.pt')
    save_json(output / 'history.json', history)
    # Baseline linear explícito e fixo, sem otimização consultando o teste.
    summary = np.concatenate([features.mean(axis=3).reshape(len(features), -1),
                              features.std(axis=3).reshape(len(features), -1)], axis=1)
    scaler = StandardScaler().fit(summary[tr])
    baseline = LogisticRegression(C=1, max_iter=2000, random_state=seed)
    baseline.fit(scaler.transform(summary[tr]), labels[tr], sample_weight=balanced_recording_weights(frame.iloc[tr]))
    import joblib
    joblib.dump(make_pipeline(scaler, baseline), output / 'baseline.joblib')
    save_json(output / 'freeze.json', dict(cnn_sha256=sha(output / 'cnn.pt'),
                                         baseline_sha256=sha(output / 'baseline.joblib'),
                                         frozen_before_test_inference=True,
                                         best_epoch=best_epoch, training_seconds=time.time() - started))
    te = splits['test']
    tp = predict(model, features[te], device)
    bp = baseline.predict_proba(scaler.transform(summary[te]))
    metrics = [evaluate(frame.iloc[te], tp, classes, 'cnn', output),
               evaluate(frame.iloc[te], bp, classes, 'linear', output)]
    save_json(output / 'metrics.json', metrics)
    pd.DataFrame(metrics).to_csv(output / 'metrics.csv', index=False)
    save_json(output / 'audit.json', dict(group_split_disjoint=True, path_split_disjoint=True,
                                         exact_window_hash_split_disjoint=True,
                                         cache_row_alignment_verified=True,
                                         weights_sha256=sha(output / 'cnn.pt'),
                                         calibration_not_used=True,
                                         independent_field_validation=False))
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    return metrics


def infer(audio_path, checkpoint):
    import soundfile as sf
    from scipy.signal import resample_poly
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = CNNDidatica(len(saved['classes']))
    model.load_state_dict(saved['state_dict'])
    wave, sr = sf.read(audio_path, dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    if sr != 16000:
        div = np.gcd(sr, 16000)
        wave = resample_poly(wave, 16000 // div, sr // div)
    size = 15872
    chunks = np.asarray([wave[s:s+size] for s in range(0, len(wave)-size+1, size)])
    if not len(chunks):
        raise ValueError('Áudio precisa ter ao menos 0,992 s; não repetimos nem completamos o sinal.')
    probs = predict(model, frontend(chunks, 'cpu'), 'cpu').mean(axis=0)
    ranking = np.argsort(probs)[-3:][::-1]
    return dict(audio=str(audio_path), windows=len(chunks),
                forced_choice=True, presence_not_tested=True,
                warning='Sempre escolhe espécie, inclusive para ruído. Escore não é confiança clínica ou de campo.',
                top3=[dict(species=saved['classes'][i], score=float(probs[i])) for i in ranking])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ifsc_cnn_2026_10_11')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'results/ifsc_cnn_2026_10_11/cnn.pt')
    args = parser.parse_args()
    if args.audio:
        print(json.dumps(infer(args.audio, args.checkpoint), ensure_ascii=False, indent=2))
    else:
        train(args.output, args.epochs)
