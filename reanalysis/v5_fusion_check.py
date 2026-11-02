"""Fusão não transdutiva: média simples dos logits (sem padronizar no lote de teste)."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from reanalysis import v5_eval as ev
meta, reps, scores, lodo_scores = ev.load()
y = (meta.target.to_numpy() == 2).astype(int)
origin = meta.origin.to_numpy().astype(object)
groups = meta.cluster_merged.astype(str).to_numpy()
out = {}
for held in ev.LODO:
    tr, te = np.flatnonzero(origin != held), np.flatnonzero(origin == held)
    a0 = ev.fit_score(reps['A0_perch'], y, origin, tr, te)
    h1b = ev.fit_score(reps['H1b_optico_embedding_aug'], y, origin, tr, te)
    h1c = lodo_scores['h1c_optico'][held][te]
    for name, s in [('A0+H1b', (a0 + h1b) / 2), ('A0+H1b+H1c', (a0 + h1b + h1c) / 3), ('H1b+H1c', (h1b + h1c) / 2)]:
        out.setdefault(name, {})[held] = ev.summarize(y[te], s, groups[te])
for k, v in out.items():
    print(k, round(np.mean([v[h]['auroc'] for h in ev.LODO]), 3), {h: v[h]['auroc'] for h in ev.LODO})
json.dump(out, open('reanalysis/outputs/v5/fusao_logits.json', 'w'), indent=1, ensure_ascii=False)
