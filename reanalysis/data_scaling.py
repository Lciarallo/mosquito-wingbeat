"""Mais dados ajudam? Três análises retrospectivas, sem tocar no teste reservado da v3.

  decompose  De onde vem o AUROC 0,86 do SVM da apresentação (receita Torres + Funchal).
  sources    Deixa um corpus fora e treina com 1, 2 ou 3 dos outros; mede Aedes x outros mosquitos
             dentro do corpus deixado de fora (a fonte não pode servir de atalho).
  clusters   Dados da v4 (desenvolvimento): curva de aprendizado variando o número de clusters
             de Aedes no treino versus o número de arquivos por cluster.

Uso: python -m reanalysis.data_scaling {decompose,sources,clusters,all}
Saída: reanalysis/outputs/data_scaling/*.json
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "apresentacao/experimento_mistura/pool.npz"
OUT = ROOT / "reanalysis/outputs/data_scaling"
SEED = 20261101


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=1, ensure_ascii=False, default=float) + "\n")


def load_pool(as_presented=False):
    """Pool da apresentação, sem UFRGS (ligado ao teste reservado da v3); Funchal agrupado por aparelho.

    as_presented=True repete o agrupamento de svm_final.py, em que '48khz' contém '8khz' e as pastas de 48 kHz
    caem no grupo de 8 kHz (o Funchal fica com 2 grupos, não 3)."""
    z = np.load(POOL, allow_pickle=True)
    X, y, src, grp = z["X"], z["y"], z["src"].astype(object), z["grp"].astype(object)
    f = src == "funchal"
    if as_presented:
        grp[f] = ["funchal:" + ("8k" if "8khz" in g else "44k" if "44.1khz" in g else "48k") for g in grp[f]]
    else:
        grp[f] = ["funchal:" + ("48k" if "48khz" in g else "44k" if "44.1khz" in g else "8k") for g in grp[f]]
    keep = src != "ufrgs"
    return X[keep], y[keep], src[keep], grp[keep]


def zscore_by_source(X, src, mask):
    Xn = X.copy()
    for s in np.unique(src[mask]):
        k = mask & (src == s)
        Xn[k] = (X[k] - X[k].mean(0)) / (X[k].std(0) + 1e-6)
    return Xn


def source_weights(idx, yy, src):
    """Mesmo peso por classe e por fonte dentro da classe (receita da apresentação)."""
    ss = src[idx]
    w = np.ones(len(idx))
    for c in np.unique(yy):
        mc = yy == c
        sources = np.unique(ss[mc])
        for s in sources:
            k = mc & (ss == s)
            w[k] = 1 / k.sum() / len(sources)
    return w / w.mean()


def cluster_bootstrap_auc(y, score, groups, reps=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ug}
    values = []
    for _ in range(reps):
        ix = np.concatenate([rows[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[ix])) == 2:
            values.append(roc_auc_score(y[ix], score[ix]))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values else None


# ---------------------------------------------------------------- 1. decomposição do SVM final

def decompose():
    X0, y, src, grp = load_pool(as_presented=True)
    mask = ((y == 2) & np.isin(src, ["torres", "funchal"])) | (y < 2)
    X = zscore_by_source(X0, src, mask)
    svm = lambda: SVC(C=1.0, gamma="scale", probability=True, random_state=11)
    idx = np.flatnonzero(mask)
    pa = np.full(len(y), np.nan)
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=11).split(X[idx], y[idx], grp[idx]):
        tr, te = idx[a], idx[b]
        y1 = (y[tr] > 0).astype(int)
        m1 = svm().fit(X[tr], y1, sample_weight=source_weights(tr, y1, src))
        mos = tr[y[tr] > 0]
        y2 = (y[mos] == 2).astype(int)
        m2 = svm().fit(X[mos], y2, sample_weight=source_weights(mos, y2, src))
        pa[te] = m1.predict_proba(X[te])[:, 1] * m2.predict_proba(X[te])[:, 1]
    i = idx
    ae, s, yy = y[i] == 2, src[i], y[i]

    def auc(pos, neg):
        k = pos | neg
        return round(float(roc_auc_score(ae[k], pa[i][k])), 3)

    neg = ~ae
    out = {
        "nota": "Reproduz apresentacao/experimento_mistura/svm_final.py sem a permutação de bandas; pool sem UFRGS.",
        "auroc_agregado": auc(ae, neg),
        "auroc_torres_vs_todos_negativos": auc(ae & (s == "torres"), neg),
        "auroc_funchal_vs_todos_negativos": auc(ae & (s == "funchal"), neg),
        "auroc_funchal_aedes_vs_funchal_culex": auc(ae & (s == "funchal"), neg & (s == "funchal")),
        "auroc_funchal_aedes_vs_negativos_de_outras_fontes": auc(ae & (s == "funchal"), neg & (s != "funchal")),
        "auroc_sem_torres": auc(ae & (s != "torres"), neg),
        "deteccao_limiar_0_5": {k: f"{int((pa[i][ae & (s == k)] > .5).sum())}/{int((ae & (s == k)).sum())}"
                                 for k in ["torres", "funchal"]},
        "alarme_limiar_0_5_por_fonte_negativa": {k: f"{int((pa[i][neg & (s == k)] > .5).sum())}/{int((neg & (s == k)).sum())}"
                                                  for k in np.unique(s[neg])},
        "funchal_por_aparelho": {},
    }
    for g in ["funchal:8k", "funchal:44k", "funchal:48k"]:
        k = grp[i] == g
        if len(np.unique(ae[k])) == 2:
            out["funchal_por_aparelho"][g] = dict(auroc=round(float(roc_auc_score(ae[k], pa[i][k])), 3),
                                                   aedes=int(ae[k].sum()), culex=int((~ae[k]).sum()))
        else:
            out["funchal_por_aparelho"][g] = dict(aedes=int(ae[k].sum()), culex=int((~ae[k]).sum()),
                                                   nota="uma classe só neste aparelho")
    out["prevalencia"] = prevalence_shift(X0, y, src, grp, mask)
    write("decomposicao_svm_final.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def prevalence_shift(X0, y, src, grp, mask):
    """A normalização por fonte usa as estatísticas do lote. Se a proporção de Aedes no lote do Funchal
    mudar, a mesma gravação recebe outro escore. Treino fixo (sem Funchal); só o lote de normalização muda."""
    rng = np.random.default_rng(SEED)
    f = src == "funchal"
    tr = np.flatnonzero(mask & ~f)
    Xtr = zscore_by_source(X0, src, mask & ~f)
    mos = tr[y[tr] > 0]
    y2 = (y[mos] == 2).astype(int)
    model = SVC(C=1.0, gamma="scale", probability=True, random_state=11).fit(Xtr[mos], y2, sample_weight=source_weights(mos, y2, src))
    fa, fc = np.flatnonzero(f & (y == 2)), np.flatnonzero(f & (y == 1))
    out = {}
    for p in [.1, .3, .63, .9]:
        n = 1000
        batch = np.r_[rng.choice(fa, int(n * p)), rng.choice(fc, n - int(n * p))]
        mu, sd = X0[batch].mean(0), X0[batch].std(0) + 1e-6
        Z = (X0[f] - mu) / sd
        score = model.predict_proba(Z)[:, 1]
        a = y[f] == 2
        out[f"aedes_no_lote_{int(p * 100)}%"] = dict(deteccao_aedes=round(float((score[a] > .5).mean()), 3),
                                                     culex_chamado_aedes=round(float((score[~a] > .5).mean()), 3),
                                                     auroc=round(float(roc_auc_score(a, score)), 3))
    return out


# ---------------------------------------------------------------- 2. número de corpora no treino

TARGETS = ["dryad", "humbug", "mosquitosong", "funchal"]


def model_factory(kind, seed):
    if kind == "logistica":
        return make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000))
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale"))
    return HistGradientBoostingClassifier(learning_rate=.05, max_iter=300, max_leaf_nodes=15,
                                          l2_regularization=1.0, min_samples_leaf=12, random_state=seed)


def score(model, X):
    return model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]


def one_source_run(X, y, src, grp, target, train_sources, kind, normalize):
    mos = y > 0
    Xn = zscore_by_source(X, src, mos) if normalize else X
    tr = np.flatnonzero(mos & np.isin(src, train_sources))
    te = np.flatnonzero(mos & (src == target))
    yt = (y[tr] == 2).astype(int)
    m = model_factory(kind, SEED)
    w = source_weights(tr, yt, src)
    if kind == "logistica" or kind == "svm":
        m.fit(Xn[tr], yt, **{m.steps[-1][0] + "__sample_weight": w})
    else:
        m.fit(Xn[tr], yt, sample_weight=w)
    s = score(m, Xn[te])
    a = y[te] == 2
    return float(roc_auc_score(a, s)), s, a, grp[te]


def sources():
    X, y, src, grp = load_pool()
    jobs = []
    for target in TARGETS:
        others = [t for t in TARGETS if t != target]
        for k in [1, 2, 3]:
            for subset in combinations(others, k):
                for kind in ["logistica", "svm", "hgb"]:
                    for normalize in [True, False]:
                        jobs.append((target, subset, kind, normalize))
    runs = Parallel(n_jobs=12)(delayed(one_source_run)(X, y, src, grp, t, list(sub), kind, nz) for t, sub, kind, nz in jobs)
    rows = []
    for (t, sub, kind, nz), (auc, s, a, g) in zip(jobs, runs):
        row = dict(alvo=t, treino=list(sub), k=len(sub), modelo=kind, normalizacao_por_fonte=nz, auroc=round(auc, 3))
        if len(sub) == 3:
            row["ic95_cluster"] = cluster_bootstrap_auc(a, s, g)
        rows.append(row)
    summary = {}
    for kind in ["logistica", "svm", "hgb"]:
        for nz in [True, False]:
            key = f"{kind} | {'norm. por fonte' if nz else 'sem norm.'}"
            summary[key] = {t: {k: round(float(np.mean([r["auroc"] for r in rows if r["alvo"] == t and r["k"] == k
                                                         and r["modelo"] == kind and r["normalizacao_por_fonte"] == nz])), 3)
                                for k in [1, 2, 3]} for t in TARGETS}
    clusters = {t: dict(aedes=int(len(np.unique(grp[(src == t) & (y == 2)]))), outros=int(len(np.unique(grp[(src == t) & (y == 1)]))))
                for t in TARGETS}
    write("numero_de_corpora.json", dict(
        nota="Pool da apresentação (80 atributos HTK por arquivo), sem UFRGS e sem Torres (só Aedes, não permite AUROC dentro da fonte). "
             "AUROC = Aedes x outros mosquitos DENTRO do corpus deixado de fora; média sobre os subconjuntos de tamanho k.",
        clusters_por_corpus=clusters, resumo=summary, execucoes=rows))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- 3. clusters x arquivos por cluster (v4)

def cluster_run(frame_y, origin, cluster, Xrep, kind, target, keep_rows, seed):
    tr = np.flatnonzero(keep_rows & (origin != target))
    te = np.flatnonzero((origin == target) & (frame_y > 0))
    yt = (frame_y[tr] == 2).astype(int)
    if yt.sum() == 0:
        return np.nan
    w = source_weights(tr, yt, origin)
    if kind == "logistica_perch":
        m = make_pipeline(StandardScaler(), PCA(128, random_state=seed), LogisticRegression(C=.1, max_iter=3000))
        m.fit(Xrep[tr], yt, logisticregression__sample_weight=w)
        s = m.decision_function(Xrep[te])
    else:
        m = HistGradientBoostingClassifier(learning_rate=.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0,
                                           min_samples_leaf=12, random_state=seed)
        m.fit(Xrep[tr], yt, sample_weight=w)
        s = m.predict_proba(Xrep[te])[:, 1]
    return float(roc_auc_score(frame_y[te] == 2, s))


def clusters(draws=12):
    from reanalysis import v4_train as v4
    frame, arrays = v4.load_development()  # o teste reservado da v3 sai aqui, antes de qualquer cálculo
    mos = frame.target.to_numpy() > 0
    frame = frame.loc[mos].reset_index(drop=True)
    arrays = {k: v[mos] for k, v in arrays.items()}
    y = frame.target.to_numpy()
    origin = frame.origin.to_numpy().astype(object)
    cl = frame.cluster_merged.astype(str).to_numpy()
    reps = {"logistica_perch": arrays["perch_white"], "hgb_bandas": arrays["band_tabular"]}
    rng = np.random.default_rng(SEED)
    jobs, meta = [], []
    for target in ["abuzz", "humbug", "mosquitosong"]:
        aedes_tr = (y == 2) & (origin != target)
        pool = np.unique(cl[aedes_tr])
        n_max = len(pool)
        for n in range(1, n_max + 1):
            for d in range(draws if n < n_max else 1):
                chosen = rng.choice(pool, n, replace=False)
                keep = (y == 1) | (aedes_tr & np.isin(cl, chosen))
                n_files = int((keep & aedes_tr).sum())
                for kind in reps:
                    jobs.append((kind, target, keep, d))
                    meta.append(dict(curva="clusters", alvo=target, n_clusters=n, n_arquivos_aedes=n_files, modelo=kind))
        aedes_idx = np.flatnonzero(aedes_tr)
        for frac in [.05, .1, .25, .5, 1.]:
            for d in range(draws if frac < 1 else 1):
                pick = []
                for c in pool:
                    rows = aedes_idx[cl[aedes_idx] == c]
                    pick.extend(rng.choice(rows, max(1, int(round(frac * len(rows)))), replace=False))
                keep = (y == 1).copy()
                keep[np.array(pick)] = True
                for kind in reps:
                    jobs.append((kind, target, keep, d))
                    meta.append(dict(curva="arquivos", alvo=target, n_clusters=n_max, fracao_arquivos=frac,
                                     n_arquivos_aedes=len(pick), modelo=kind))
    aucs = Parallel(n_jobs=14, verbose=0)(delayed(cluster_run)(y, origin, cl, reps[kind], kind, t, keep, SEED + d)
                                           for kind, t, keep, d in jobs)
    rows = [dict(m, auroc=round(a, 4)) for m, a in zip(meta, aucs)]
    summary = {}
    for curve, key in [("clusters", "n_clusters"), ("arquivos", "fracao_arquivos")]:
        for kind in reps:
            for t in ["abuzz", "humbug", "mosquitosong"]:
                sel = [r for r in rows if r["curva"] == curve and r["modelo"] == kind and r["alvo"] == t]
                levels = sorted({r[key] for r in sel})
                summary[f"{curve} | {kind} | {t}"] = {
                    str(v): dict(auroc_media=round(float(np.nanmean([r["auroc"] for r in sel if r[key] == v])), 3),
                                 auroc_dp=round(float(np.nanstd([r["auroc"] for r in sel if r[key] == v])), 3),
                                 arquivos_aedes=int(np.mean([r["n_arquivos_aedes"] for r in sel if r[key] == v])))
                    for v in levels}
    write("clusters_vs_arquivos.json", dict(
        nota="Desenvolvimento da v4 (teste reservado da v3 removido por v4_train.load_development). Binário Aedes x outros "
             "mosquitos; o corpus-alvo fica fora do treino e o AUROC é medido dentro dele. 'clusters': sorteia n clusters de "
             "Aedes do treino (todos os arquivos deles). 'arquivos': todos os clusters, só uma fração dos arquivos de cada. "
             f"{draws} sorteios por nível.",
        resumo=summary, execucoes=rows))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- 4. dados do próprio domínio

def indomain_run(y, origin, cl, Xrep, kind, target, add_rows, test_rows, seed):
    tr = np.flatnonzero(((origin != target) | add_rows) & (y > 0))
    te = np.flatnonzero(test_rows)
    yt = (y[tr] == 2).astype(int)
    w = source_weights(tr, yt, origin)
    if kind == "logistica_perch":
        m = make_pipeline(StandardScaler(), PCA(128, random_state=seed), LogisticRegression(C=.1, max_iter=3000))
        m.fit(Xrep[tr], yt, logisticregression__sample_weight=w)
        s = m.decision_function(Xrep[te])
    else:
        m = HistGradientBoostingClassifier(learning_rate=.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0,
                                           min_samples_leaf=12, random_state=seed)
        m.fit(Xrep[tr], yt, sample_weight=w)
        s = m.predict_proba(Xrep[te])[:, 1]
    return float(roc_auc_score(y[te] == 2, s))


def indomain(draws=20):
    """Quanto vale ter clusters rotulados do próprio domínio no treino? Para cada corpus-alvo, m clusters de Aedes
    e a mesma fração dos clusters de outros mosquitos do alvo entram no treino; o teste são os clusters restantes
    do alvo (disjuntos). m = 0 é o LODO puro, avaliado nos mesmos clusters de teste do sorteio."""
    from reanalysis import v4_train as v4
    frame, arrays = v4.load_development()
    mos = frame.target.to_numpy() > 0
    frame = frame.loc[mos].reset_index(drop=True)
    arrays = {k: v[mos] for k, v in arrays.items()}
    y = frame.target.to_numpy()
    origin = frame.origin.to_numpy().astype(object)
    cl = frame.cluster_merged.astype(str).to_numpy()
    reps = {"logistica_perch": arrays["perch_white"], "hgb_bandas": arrays["band_tabular"]}
    rng = np.random.default_rng(SEED + 7)
    jobs, meta = [], []
    for target in ["abuzz", "humbug", "mosquitosong"]:
        t = origin == target
        ca = np.unique(cl[t & (y == 2)])
        co = np.setdiff1d(np.unique(cl[t & (y == 1)]), ca)
        for m in range(1, len(ca) - 1):
            for d in range(draws):
                pa = rng.choice(ca, m, replace=False)
                po = rng.choice(co, int(round(len(co) * m / len(ca))), replace=False)
                add = t & np.isin(cl, np.r_[pa, po])
                test = t & ~np.isin(cl, np.r_[pa, po])
                if len(np.unique(y[test])) < 2:
                    continue
                for kind in reps:
                    for with_target in ([False, True] if m else [False]):
                        jobs.append((kind, target, add if with_target else np.zeros_like(add), test, d))
                        meta.append(dict(alvo=target, m_clusters_aedes_do_alvo=m if with_target else 0, sorteio_m=m,
                                         clusters_aedes_teste=int(len(np.unique(cl[test & (y == 2)]))), modelo=kind))
    aucs = Parallel(n_jobs=14)(delayed(indomain_run)(y, origin, cl, reps[k], k, t, add, test, SEED + d)
                               for k, t, add, test, d in jobs)
    rows = [dict(mm, auroc=round(a, 4)) for mm, a in zip(meta, aucs)]
    summary = {}
    for kind in reps:
        for target in ["abuzz", "humbug", "mosquitosong"]:
            out = {}
            for m in sorted({r["sorteio_m"] for r in rows if r["alvo"] == target}):
                if m == 0:
                    continue
                base = [r["auroc"] for r in rows if r["alvo"] == target and r["modelo"] == kind and r["sorteio_m"] == m and r["m_clusters_aedes_do_alvo"] == 0]
                inn = [r["auroc"] for r in rows if r["alvo"] == target and r["modelo"] == kind and r["sorteio_m"] == m and r["m_clusters_aedes_do_alvo"] == m]
                out[str(m)] = dict(sem_alvo=round(float(np.mean(base)), 3), com_alvo=round(float(np.mean(inn)), 3),
                                   ganho=round(float(np.mean(np.array(inn) - np.array(base))), 3),
                                   ganho_positivo_em=f"{int((np.array(inn) > np.array(base)).sum())}/{len(inn)}")
            summary[f"{kind} | {target}"] = out
    write("dados_do_dominio.json", dict(
        nota="Desenvolvimento da v4, binário Aedes x outros mosquitos. Em cada sorteio, os mesmos clusters de teste do alvo "
             "são avaliados sem e com m clusters de Aedes do alvo (e a fração correspondente de outros mosquitos) no treino.",
        resumo=summary, execucoes=rows))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["decompose", "sources", "clusters", "indomain", "all"])
    stage = parser.parse_args().stage
    if stage in ("decompose", "all"):
        decompose()
    if stage in ("sources", "all"):
        sources()
    if stage in ("clusters", "all"):
        clusters()
    if stage in ("indomain", "all"):
        indomain()
