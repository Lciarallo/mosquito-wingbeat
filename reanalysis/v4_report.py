"""Publish compact, auditable v4 outputs and reports/resultados_v4.md from results/v4 (no recomputation)."""
import importlib.metadata
import json
import shutil

import numpy as np
import pandas as pd

from reanalysis import v4_train as v4

NAMES = {"A0_perch_white": "A0 Perch normalizado (referência v3)", "A1_perch_coral": "A1 Perch + CORAL",
         "A2_perch_dann": "A2 Perch + DANN", "B1_band_tabular": "B1 bandas ≥ 300 Hz", "B2_wingbeat": "B2 batimento",
         "C1_aug_band_tabular": "C1 bandas ≥ 300 Hz + aumento", "C2_aug_perch_coral": "C2 Perch + CORAL + aumento",
         "D1_perch_ft": "D1 Perch afinado na GPU (emenda A1)"}


def pct(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f}%"


def ci(result, key):
    c = result.get("ci95", {}).get(key)
    if not c:
        return ""
    if c[0] == c[1]:
        return " (bootstrap degenerado)"
    return f" [{100 * c[0]:.1f}–{100 * c[1]:.1f}]"


def auroc(result):
    p = result["point"].get("auroc")
    c = result.get("ci95", {}).get("auroc")
    return "—" if p is None else f"{p:.3f}" + (f" [{c[0]:.2f}–{c[1]:.2f}]" if c else "")


def main():
    v4.check_freeze()
    out, pub = v4.OUT, v4.PUBLIC
    pub.mkdir(parents=True, exist_ok=True)
    dev = json.loads((out / "development_metrics.json").read_text())
    sel = json.loads((out / "selection.json").read_text())
    diag = json.loads((out / "diagnostics.json").read_text())
    few = pd.read_csv(out / "fewshot_calibration.csv")
    occl = pd.read_csv(out / "band_occlusion.csv")
    sex = pd.read_csv(out / "sex_strata.csv")
    final = json.loads((out / "final" / "final_freeze.json").read_text())
    for name in ["development_metrics.json", "selection.json", "diagnostics.json", "fewshot_calibration.csv",
                 "band_occlusion.csv", "sex_strata.csv"]:
        shutil.copyfile(out / name, pub / name)
    shutil.copyfile(out / "final" / "final_freeze.json", pub / "final_freeze.json")
    preds, parts = [], {}
    for tag in dev:
        directory = out / "development" / tag
        parts[tag] = json.loads((directory / "partitions.json").read_text())
        for name in v4.CANDIDATES + v4.DIAGNOSTIC:
            preds.append(pd.read_csv(directory / f"{name}.csv").drop(columns=["path"], errors="ignore"))
    pd.concat(preds).to_csv(pub / "development_predictions.csv.gz", index=False)
    v4.write_json(pub / "partitions.json", parts)
    rows = []
    for tag, models in dev.items():
        for key, r in models.items():
            p = r["point"]
            rows.append(dict(evaluation=tag, model=key, **{k: p[k] for k in ["n_files", "n_clusters", "aedes_n", "other_n", "noise_n",
                             "aedes_tp", "other_fp", "noise_fp", "alerts", "uncertain", "aedes_recall", "fpr_other",
                             "fpr_noise", "aedes_precision", "auroc"]},
                             aedes_clusters=p["class_clusters"]["2"],
                             **{f"{k}_ci95": str(v) for k, v in r.get("ci95", {}).items()}))
    pd.DataFrame(rows).to_csv(pub / "development_summary.csv", index=False)
    identity = dict(frozen=json.loads(v4.FREEZE.read_text()), report_code_sha256=v4.digest(v4.ROOT / "reanalysis/v4_report.py"),
                    augmented_cache_meta=json.loads((v4.AUG / "meta.json").read_text()),
                    packages={k: importlib.metadata.version(k) for k in ["numpy", "scipy", "pandas", "scikit-learn", "torch", "onnxruntime"]})
    v4.write_json(pub / "identity.json", identity)

    best = {c: max((r for r in sel["candidates"] if r["candidate"] == c), key=lambda r: r["auroc"]) for c in v4.CANDIDATES}
    L = ["# v4: resultados do desenvolvimento e estado do teste novo", "",
         "Executado em 22/10/2026 conforme `reanalysis/protocol_v4.json` (congelado em `protocol_v4_freeze.json`, com as emendas A1 — fine-tuning do Perch na GPU — e A2 — correção numérica —, todas congeladas antes de qualquer treino). "
         "Usa só os papéis de desenvolvimento da v3; o teste reservado da v3 não foi lido. Saídas auditáveis em `reanalysis/outputs/v4/`.", "",
         "## Resposta direta", "",
         "**Os resultados melhoraram no teste novo, com IC por cluster? Inconclusivo.** Não existe teste novo que cumpra os critérios pré-registrados "
         "(`reports/datasets_candidatos_v4.md`), e o desenvolvimento reutiliza os mesmos 16 clusters com Aedes já explorados na v3. "
         f"No desenvolvimento, {'houve' if sel['eligible'] else '**nenhum candidato atingiu**'} as metas por corpus "
         f"(recall ≥ 20%, FPR ≤ 2% em outros mosquitos, ≤ 1% em ruído){' (' + sel['eligibility_level'] + ')' if sel['eligible'] else ''}. "
         f"Congelado para o teste futuro: **{NAMES[sel['candidate']]}**, {'com' if sel['gated'] else 'sem'} portão"
         f"{'' if sel['eligible'] else ', apenas como referência descritiva (maior AUROC médio no LODO)'}.", "",
         "## LODO de mosquitos: cada corpus fora do treino", "",
         "Recall / FPR em outros mosquitos, por corpus, sem calibração local (k = 0). IC95% por bootstrap de clusters entre colchetes no AUROC.", "",
         "| Candidato | Portão | Abuzz recall / FPR | HumBug recall / FPR | MosquitoSong recall / FPR | AUROC Abuzz | AUROC HumBug | AUROC MosquitoSong |",
         "|---|---|---|---|---|---|---|---|"]
    for name in v4.CANDIDATES:
        for gated in [False, True]:
            key = f"{name}|{'gated' if gated else 'ungated'}"
            cells = [f"{pct(dev[f'LODO_{c}'][key]['point']['aedes_recall'])} / {pct(dev[f'LODO_{c}'][key]['point']['fpr_other'])}" for c in v4.LODO_MOSQUITO]
            aucs = [auroc(dev[f"LODO_{c}"][key]) for c in v4.LODO_MOSQUITO]
            L.append(f"| {NAMES[name]} | {'sim' if gated else 'não'} | " + " | ".join(cells + aucs) + " |")
    L += ["", "Clusters com Aedes avaliados: " + ", ".join(
        f"{c} {dev[f'LODO_{c}']['A0_perch_white|ungated']['point']['class_clusters']['2']}" for c in v4.LODO_MOSQUITO) +
        ". Com tão poucos clusters, diferenças de AUROC entre candidatos têm IC largo e se sobrepõem.", "",
        "## FPR em ruído de fonte nunca vista", "", "| Candidato | Portão | HumBug (ruído do próprio corpus) | ESC-50 fora | Pete fora |", "|---|---|---|---|---|"]
    for name in v4.CANDIDATES:
        for gated in [False, True]:
            key = f"{name}|{'gated' if gated else 'ungated'}"
            cells = []
            for tag in ["LODO_humbug", "LODO_esc50", "LODO_pete_noise"]:
                r = dev[tag][key]
                p = r["point"]
                cells.append(f"{p['noise_fp']}/{p['noise_n']} = {pct(p['fpr_noise'])}{ci(r, 'fpr_noise')}")
            L.append(f"| {NAMES[name]} | {'sim' if gated else 'não'} | " + " | ".join(cells) + " |")
    L += ["", "## Custo de calibração local", "",
          "Mediana de 5 repetições; só os mapas de probabilidade e o limiar são recalibrados com k arquivos por classe do corpus fora do treino; "
          "clusters tocados saem da avaliação. Recall / FPR em outros.", "",
          "| Candidato | Portão | k | Abuzz | HumBug | MosquitoSong |", "|---|---|---|---|---|---|"]
    av = few[few.available.astype(bool)]
    for name in v4.CANDIDATES:
        for gated in [False, True]:
            for k in [5, 10, 20]:
                cells = []
                for c in v4.LODO_MOSQUITO:
                    s = av[(av.candidate == name) & (av.corpus == c) & (av.k == k) & (av.gated.astype(bool) == gated)]
                    cells.append("indisponível" if s.empty else f"{pct(s.aedes_recall.median())} / {pct(s.fpr_other.median())} (n={len(s)})")
                L.append(f"| {NAMES[name]} | {'sim' if gated else 'não'} | {k} | " + " | ".join(cells) + " |")
    unavailable = few[~few.available.astype(bool)].groupby(["corpus", "k"]).reason.first()
    if len(unavailable):
        L += ["", "Braços indisponíveis: " + "; ".join(f"{c} k={k}: {r}" for (c, k), r in unavailable.items()) + "."]
    a0 = next(r for r in sel["candidates"] if r["candidate"] == "A0_perch_white" and not r["gated"])
    d1 = next(r for r in sel["candidates"] if r["candidate"] == "D1_perch_ft" and not r["gated"])
    a1 = next(r for r in sel["candidates"] if r["candidate"] == "A1_perch_coral" and not r["gated"])
    L += ["", "## Leitura", "",
          f"- **Nenhuma hipótese tornou o detector utilizável.** Sem portão, o recall fica muito abaixo de 20% em quase todos os corpora; com portão, quase nenhum alerta é emitido (o mesmo padrão do 0/197 da v3). A calibração local com 5, 10 ou 20 arquivos não levou nenhum candidato às metas.",
          f"- **O fine-tuning na GPU (D1) não superou o Perch congelado (A0):** AUROC médio {d1['auroc']:.3f} contra {a0['auroc']:.3f}, com IC sobrepostos.",
          f"- **CORAL piorou o ranking** (AUROC médio {a1['auroc']:.3f}). Alinhar a distribuição de cada corpus também apaga diferenças reais de composição de espécies entre corpora.",
          "- **O vencedor formal (B1) é indistinguível do A0** dentro dos IC por cluster; foi congelado só porque a regra exige um candidato descritivo.",
          "- **A origem segue decodificável** em todas as representações não alinhadas, inclusive nos 9 descritores físicos do batimento. Restringir a banda reduz pouco.",
          "- Diagnóstico consistente com o plano: com 16 clusters de Aedes, o problema é a diversidade de dados, não a arquitetura.", ""]
    probe = diag["source_probe"]
    L += ["", "## A origem ainda é decodificável?", "",
          "Classificador linear prevendo o corpus de origem (Abuzz, HumBug, MosquitoSong) entre arquivos de mosquito, 5 dobras por cluster. Acaso = 33,3%.", "",
          "| Representação | Acurácia balanceada |", "|---|---|"]
    L += [f"| {k} | {100 * v['balanced_accuracy']:.1f}% |" for k, v in probe.items()]
    L += ["", "CORAL usa a identidade da origem como lote não rotulado. O valor **abaixo do acaso** com CORAL é um artefato esperado: cada origem é centralizada na mesma média, e os clusters de validação ficam deslocados para o lado oposto da média de treino da própria origem. Ele mostra que a informação de primeira e segunda ordem foi removida, não que a origem deixou de influenciar o modelo.", "",
          "Alinhamento por grupo (MMD entre origens, só 'outro mosquito', menor = mais alinhado):", "", "| Par | Perch normalizado (PCA-128) | Perch + CORAL |", "|---|---|---|"]
    ga = diag["group_alignment"]
    for pair in ga["perch_white_pca128"]:
        L.append(f"| {pair.replace('~', ' × ')} | {ga['perch_white_pca128'][pair]:.4f} | {ga['coral(perch_white)'][pair]:.4f} |")
    o = occl.groupby(["model", "band"], sort=False)[["mean_abs_delta_p_aedes"]].mean().reset_index()
    L += ["", "## Oclusão de bandas (média nos 3 LODO)", "", "Variação média absoluta de p(Aedes) ao substituir a faixa pela média do ajuste.", "",
          "| Faixa | " + " | ".join(o.model.unique()) + " |", "|---|" + "---|" * o.model.nunique()]
    for band in o.band.unique():
        L.append(f"| {band} | " + " | ".join(
            (lambda s: f"{s.iloc[0]:.4f}" if len(s) else "—")(o[(o.band == band) & (o.model == m)].mean_abs_delta_p_aedes)
            for m in o.model.unique()) + " |")
    s0 = sex[~sex.gated.astype(bool)]
    L += ["", "## MosquitoSong por sexo (descritivo, sem portão)", "", "| Candidato | Sexo | Aedes detectados | Outros chamados Aedes | AUROC |", "|---|---|---|---|---|"]
    for _, r in s0.iterrows():
        L.append(f"| {NAMES[r.candidate]} | {r.sex} | {int(r.aedes_tp)}/{int(r.aedes_n)} | {int(r.other_fp)}/{int(r.other_n)} | "
                 f"{'—' if pd.isna(r.auroc) else f'{r.auroc:.3f}'} |")
    L += ["", "Cada sexo tem só 2 indivíduos de *Ae. aegypti* no MosquitoSong.", "",
          "## Seleção e congelamento", "", f"Regra: `protocol_v4.json` › `selection`. Resultado: {sel['reason']}.", "",
          f"Candidato congelado: `{final['selection']}` ({'com' if final['gated'] else 'sem'} portão), reajustado em todo o desenvolvimento. "
          f"SHA-256 dos pesos: " + ", ".join(f"`{k}` `{v[:16]}…`" for k, v in final["model_sha256"].items()) +
          f". Congelado em {final['frozen_utc']}, antes de existir qualquer dado do teste novo.", "",
          "## Limites", "",
          "- Sem teste novo, nenhum número desta página sustenta afirmação de melhora ou de generalização.",
          "- 16 clusters com Aedes no desenvolvimento, 2 a 4 deles na calibração de cada dobra; o limiar é escolhido com pouquíssimos positivos independentes.",
          "- Bootstrap por cluster com modelos fixos; não inclui a variabilidade de retreino. Zero eventos geram intervalos degenerados.",
          "- CORAL e A1/C2 são transdutivos: exigem um lote não rotulado do aparelho/corpus avaliado.",
          "- A normalização espectral por gravação (herdada da v3) já remove boa parte de uma equalização estática; o efeito do aumento vem sobretudo de ruído e reverberação externos.",
          "- O fine-tuning do Perch foi executado pela emenda A1 (GPU AMD RX 9070 XT, ROCm 6.4), com a correção numérica da emenda A2. Kernels de GPU não são bit a bit determinísticos; as seeds fixam inicialização e ordem dos dados (as perdas da dobra HumBug repetiram exatamente ao ser refeita).",
          "- Sexo e temperatura não entraram como variáveis do modelo: só existem no MosquitoSong e em parte do HumBug.",
          "- Resultados da v2/v3 permanecem como estão; esta é uma rodada nova, não uma correção delas.", "",
          "## Reprodução", "", "```bash",
          "python -m reanalysis.v4_augment        # atributos da cópia aumentada (OpenSLR 28 em data/external/openslr28/)",
          "python -m reanalysis.v4_train dev      # LODO, ablação do portão, calibração local",
          "python -m reanalysis.v4_train select",
          "python -m reanalysis.v4_train diagnostics",
          "python -m reanalysis.v4_train final    # congela o candidato antes do teste novo",
          "python -m reanalysis.v4_report", "```", ""]
    (v4.ROOT / "reports/resultados_v4.md").write_text("\n".join(L))
    print("ok", sel["candidate"], sel["eligible"])


if __name__ == "__main__":
    main()
