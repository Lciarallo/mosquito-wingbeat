"""v5 (exploratória): avaliação LODO de todas as hipóteses (reports/plano_v5_exploratorio.md).

Tarefa: *Ae. aegypti* x outros mosquitos. Cada corpus acústico (Abuzz, HumBug, MosquitoSong, Funchal) fica fora
por vez; o AUROC é medido dentro dele, com IC95% por bootstrap de clusters. Cabeça comum para embeddings:
padronização, PCA-128 se a dimensão passar de 128, logística C = 0,1, pesos iguais por classe e fonte.

Uso: python -m reanalysis.v5_eval {main,fewshot}
"""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/v5"
OUT = ROOT / "reanalysis/outputs/v5"
LODO = ["abuzz", "humbug", "mosquitosong", "funchal"]
SEED = 20261102


def load():
    from reanalysis import v4_train as v4
    meta = pd.read_csv(CACHE / "acoustic_meta.csv")
    frame, arrays = v4.load_development()
    mos = frame.target.to_numpy() > 0
    n_dev = int(mos.sum())
    if not (frame.path[mos].to_numpy() == meta.path[:n_dev].to_numpy()).all():
        raise AssertionError("cache acústico desalinhado do desenvolvimento da v4")
    reps, scores = {}, {}
    perch = np.zeros((len(meta), 1536), np.float32)
    perch[:n_dev] = arrays["perch_white"][mos]
    with np.load(CACHE / "emb_perch_funchal.npz") as z:
        perch[z["rows"]] = z["embedding"]
    if not np.abs(perch).sum(1).all():
        raise AssertionError("Perch ausente em alguma linha")
    reps["A0_perch"] = perch
    for name, dim in [("ast", 768), ("clap", 512), ("biolingual", 512)]:
        path = CACHE / f"emb_{name}.npz"
        if path.exists():
            with np.load(path) as z:
                reps[f"H2_{name}"] = z["embedding"][:, :dim]
                if name != "ast":
                    sims = z["embedding"][:, dim:]
                    scores[f"H2_{name}_zeroshot"] = sims[:, 0] - sims[:, 1:].mean(1)
    path = CACHE / "h1_optical_applied.npz"
    if path.exists():
        with np.load(path) as z:
            for tag in ["aug", "noaug"]:
                scores[f"H1a_optico_direto_{tag}"] = z[f"h1a_{tag}"]
                reps[f"H1b_optico_embedding_{tag}"] = z[f"h1b_{tag}"]
    lodo_scores = {}
    path = CACHE / "h1_lodo_scores.npz"
    if path.exists():
        with np.load(path) as z:
            for key in z.files:
                name, held = key.split("|")
                lodo_scores.setdefault(name, {})[held] = z[key]
    return meta, reps, scores, lodo_scores


def source_weights(origin, y):
    w = np.ones(len(y))
    for c in np.unique(y):
        mc = y == c
        S = np.unique(origin[mc])
        for s in S:
            k = mc & (origin == s)
            w[k] = 1 / k.sum() / len(S)
    return w / w.mean()


def head(dim):
    steps = [StandardScaler()] + ([PCA(128, random_state=SEED)] if dim > 128 else [])
    return make_pipeline(*steps, LogisticRegression(C=.1, max_iter=5000))


def fit_score(X, y, origin, tr, te):
    m = head(X.shape[1])
    m.fit(X[tr], y[tr], logisticregression__sample_weight=source_weights(origin[tr], y[tr]))
    return m.decision_function(X[te])


def threshold_from_training(X, y, origin, groups, tr):
    """Limiar para FPR <= 2% em outros mosquitos, escolhido em predições fora da dobra (por cluster) no treino."""
    oof = np.zeros(len(tr))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(tr, y[tr], groups[tr]):
        oof[b] = fit_score(X, y, origin, tr[a], tr[b])
    return float(np.quantile(oof[y[tr] == 0], .98))


def boot_ci(y, s, groups, reps=1000):
    rng = np.random.default_rng(SEED)
    ug = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(reps):
        ix = np.concatenate([rows[g] for g in rng.choice(ug, len(ug))])
        if 0 < y[ix].sum() < len(ix):
            vals.append(roc_auc_score(y[ix], s[ix]))
    return [round(float(np.quantile(vals, .025)), 3), round(float(np.quantile(vals, .975)), 3)]


def summarize(y, s, groups, thr=None):
    out = dict(auroc=round(float(roc_auc_score(y, s)), 3), ic95=boot_ci(y, s, groups))
    if thr is not None:
        out.update(recall=round(float((s[y == 1] > thr).mean()), 3), fpr_outros=round(float((s[y == 0] > thr).mean()), 3))
    return out


def mmd_head(X, y, origin, tr, te, lam=1.0, steps=1500, seed=SEED):
    """H5: MLP 1536->256->1 com BCE ponderada + MMD (RBF multi-escala) entre corpora dentro de cada classe."""
    import torch
    torch.manual_seed(seed)
    sc = StandardScaler().fit(X[tr])
    Xt = torch.tensor(sc.transform(X[tr]), dtype=torch.float32)
    Xe = torch.tensor(sc.transform(X[te]), dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.float32)
    o = origin[tr]
    groups = {(c, s): np.flatnonzero((y[tr] == c) & (o == s)) for c in [0, 1] for s in np.unique(o) if ((y[tr] == c) & (o == s)).sum() >= 4}
    body = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(.2))
    out = torch.nn.Linear(256, 1)
    opt = torch.optim.AdamW(list(body.parameters()) + list(out.parameters()), 1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(seed)

    def mmd(a, b):
        z = torch.cat([a, b])
        d = torch.cdist(z, z) ** 2
        bw = d.detach().median() + 1e-6
        k = sum(torch.exp(-d / (bw * f)) for f in [.5, 1, 2])
        n = len(a)
        return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()

    for _ in range(steps):
        pick = {g: rng.choice(ix, min(32, len(ix)), replace=len(ix) < 32) for g, ix in groups.items()}
        idx = np.concatenate(list(pick.values()))
        h = body(Xt[idx])
        logit = out(h)[:, 0]
        w = torch.ones(len(idx))  # cada grupo (classe, fonte) entra com 32 exemplos: já balanceado
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[idx], weight=w)
        pos = 0
        hs = {}
        for g, ix in pick.items():
            hs[g] = h[pos:pos + len(ix)]
            pos += len(ix)
        reg = 0
        for c in [0, 1]:
            srcs = [g for g in hs if g[0] == c]
            for i in range(len(srcs)):
                for j in range(i + 1, len(srcs)):
                    reg = reg + mmd(hs[srcs[i]], hs[srcs[j]])
        loss = loss + lam * reg
        opt.zero_grad(), loss.backward(), opt.step()
    body.eval()
    with torch.no_grad():
        return out(body(Xe))[:, 0].numpy()


def evaluate_rep(name, X, y, origin, groups, held):
    tr = np.flatnonzero(origin != held)
    te = np.flatnonzero(origin == held)
    s = fit_score(X, y, origin, tr, te)
    thr = threshold_from_training(X, y, origin, groups, tr)
    return name, held, summarize(y[te], s, groups[te], thr), s


def main():
    meta, reps, scores, lodo_scores = load()
    y = (meta.target.to_numpy() == 2).astype(int)
    origin = meta.origin.to_numpy().astype(object)
    groups = meta.cluster_merged.astype(str).to_numpy()
    jobs = [(n, X, held) for n, X in reps.items() for held in LODO]
    res = Parallel(n_jobs=8)(delayed(evaluate_rep)(n, X, y, origin, groups, held) for n, X, held in jobs)
    table, test_scores = {}, {}
    for name, held, summ, s in res:
        table.setdefault(name, {})[held] = summ
        test_scores.setdefault(name, {})[held] = s
    # pontuações sem cabeça: limiar = quantil 98% dos outros mosquitos dos corpora de treino
    for name, s_all in scores.items():
        for held in LODO:
            te, tr = origin == held, origin != held
            thr = float(np.quantile(s_all[tr & (y == 0)], .98))
            table.setdefault(name, {})[held] = summarize(y[te], s_all[te], groups[te], thr)
            test_scores.setdefault(name, {})[held] = s_all[te]
    for name, per_held in lodo_scores.items():
        label = {"c0_zero": "C0_cnn_do_zero_acustico", "h1c_optico": "H1c_cnn_optica_afinada"}[name]
        for held, s_all in per_held.items():
            te = origin == held
            table.setdefault(label, {})[held] = summarize(y[te], s_all[te], groups[te])
            test_scores.setdefault(label, {})[held] = s_all[te]
    # H5: Perch + MMD
    for held in LODO:
        tr, te = np.flatnonzero(origin != held), np.flatnonzero(origin == held)
        s = mmd_head(reps["A0_perch"], y, origin, tr, te)
        table.setdefault("H5_perch_mmd", {})[held] = summarize(y[te], s, groups[te])
        test_scores.setdefault("H5_perch_mmd", {})[held] = s
    # Diagnóstico do H5: a mesma MLP sem o termo de MMD (lambda = 0)
    diag_mlp = {}
    for held in LODO:
        tr, te = np.flatnonzero(origin != held), np.flatnonzero(origin == held)
        diag_mlp[held] = summarize(y[te], mmd_head(reps["A0_perch"], y, origin, tr, te, lam=0.0), groups[te])
    print("MLP sem MMD:", {h: v["auroc"] for h, v in diag_mlp.items()})
    # Diagnóstico (não é hipótese): quanto o Funchal no treino muda o A0 nos outros corpora
    for held in ["abuzz", "humbug", "mosquitosong"]:
        tr = np.flatnonzero((origin != held) & (origin != "funchal"))
        te = np.flatnonzero(origin == held)
        s = fit_score(reps["A0_perch"], y, origin, tr, te)
        table.setdefault("diag_A0_sem_funchal_no_treino", {})[held] = summarize(y[te], s, groups[te])
    diag = table.pop("diag_A0_sem_funchal_no_treino")
    summary = {}
    base = {h: table["A0_perch"][h]["auroc"] for h in LODO}
    for name, per in table.items():
        mean = float(np.mean([per[h]["auroc"] for h in LODO]))
        wins = sum(per[h]["auroc"] > base[h] for h in LODO)
        summary[name] = dict(auroc_medio=round(mean, 3), ganho_sobre_A0=round(mean - np.mean(list(base.values())), 3),
                             melhor_que_A0_em=f"{wins}/4",
                             promissor=bool(mean >= np.mean(list(base.values())) + .05 and wins >= 3))
    order = sorted(summary, key=lambda k: -summary[k]["auroc_medio"])
    # H3: fusão por média de postos (dentro do corpus, sem rótulo) das 3 melhores representações.
    # Aninhada: para cada corpus fora, o ranking usa só o AUROC nos outros 3 corpora.
    # A versão pós-hoc (ranking com os 4 corpora) fica só como referência otimista.
    cands = [k for k in table if not k.startswith("H3") and all(h in table[k] for h in LODO)]
    fusions = {"H3_fusao_top3_aninhada": {}, "H3_fusao_top3_poshoc": {}}
    for held in LODO:
        te = origin == held
        nested = sorted(cands, key=lambda k: -np.mean([table[k][h]["auroc"] for h in LODO if h != held]))[:3]
        post = [k for k in order if k in cands][:3]
        for label, comps in [("H3_fusao_top3_aninhada", nested), ("H3_fusao_top3_poshoc", post)]:
            ranks = [pd.Series(test_scores[k][held]).rank(pct=True).to_numpy() for k in comps]
            table.setdefault(label, {})[held] = dict(summarize(y[te], np.mean(ranks, 0), groups[te]), componentes=comps)
    for label in fusions:
        per = table[label]
        mean = float(np.mean([per[h]["auroc"] for h in LODO]))
        wins = sum(per[h]["auroc"] > base[h] for h in LODO)
        summary[label] = dict(auroc_medio=round(mean, 3), ganho_sobre_A0=round(mean - np.mean(list(base.values())), 3),
                              melhor_que_A0_em=f"{wins}/4",
                              promissor=bool(label.endswith("aninhada") and mean >= np.mean(list(base.values())) + .05 and wins >= 3))
    n = {h: dict(aedes=int(((origin == h) & (y == 1)).sum()), outros=int(((origin == h) & (y == 0)).sum()),
                 clusters_aedes=int(len(np.unique(groups[(origin == h) & (y == 1)])))) for h in LODO}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lodo.json").write_text(json.dumps(dict(n=n, resumo=summary, por_corpus=table,
                                                   diagnostico_A0_sem_funchal_no_treino=diag,
                                                   diagnostico_H5_mlp_sem_mmd=diag_mlp), indent=1, ensure_ascii=False) + "\n")
    print("A0 sem Funchal no treino:", {h: v["auroc"] for h, v in diag.items()})
    for k in sorted(summary, key=lambda k: -summary[k]["auroc_medio"]):
        print(f"{k:34s} {summary[k]['auroc_medio']:.3f} ({summary[k]['ganho_sobre_A0']:+.3f}, {summary[k]['melhor_que_A0_em']})  "
              + "  ".join(f"{h}={table[k][h]['auroc']:.3f}" for h in LODO))


def fewshot_run(Xs, y, origin, groups, held, add, test, seed):
    """Xs: uma ou mais representações; com mais de uma, a pontuação é a média dos logits das cabeças."""
    tr = np.flatnonzero((origin != held) | add)
    te = np.flatnonzero(test)
    return float(roc_auc_score(y[te], np.mean([fit_score(X, y, origin, tr, te) for X in Xs], 0)))


def fewshot(names, draws=20):
    meta, reps, _, _ = load()
    y = (meta.target.to_numpy() == 2).astype(int)
    origin = meta.origin.to_numpy().astype(object)
    groups = meta.cluster_merged.astype(str).to_numpy()
    rng = np.random.default_rng(SEED + 7)
    jobs, info = [], []
    for held in LODO:
        t = origin == held
        ca = np.unique(groups[t & (y == 1)])
        co = np.setdiff1d(np.unique(groups[t & (y == 0)]), ca)
        for m in range(1, len(ca) - 1):
            for d in range(draws):
                pa = rng.choice(ca, m, replace=False)
                po = rng.choice(co, int(round(len(co) * m / len(ca))), replace=False) if len(co) else np.array([], object)
                add = t & np.isin(groups, np.r_[pa, po])
                test = t & ~add
                if y[test].sum() == 0 or y[test].sum() == test.sum():
                    continue
                for name in names:
                    for with_t in [False, True]:
                        jobs.append((name, held, add if with_t else np.zeros_like(add), test))
                        info.append(dict(rep=name, alvo=held, m=m, com_alvo=with_t, sorteio=d))
    aucs = Parallel(n_jobs=12)(delayed(fewshot_run)([reps[k] for k in n.split("+")], y, origin, groups, h, a, t, SEED)
                               for n, h, a, t in jobs)
    rows = [dict(i, auroc=round(a, 4)) for i, a in zip(info, aucs)]
    df = pd.DataFrame(rows)
    piv = df.groupby(["rep", "alvo", "m", "com_alvo"]).auroc.mean().unstack("com_alvo").round(3)
    piv.columns = ["sem_alvo", "com_alvo"]
    print(piv)
    (OUT / "fewshot.json").write_text(json.dumps(dict(resumo=piv.reset_index().to_dict("records"), execucoes=rows),
                                                 indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["main", "fewshot"])
    parser.add_argument("--reps", nargs="*", default=["A0_perch"])
    a = parser.parse_args()
    main() if a.stage == "main" else fewshot(a.reps)
