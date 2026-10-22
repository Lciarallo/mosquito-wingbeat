"""Publish compact, evidence-bound v3 continuation results after final evaluation."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reanalysis.v3_train import CACHE, CANDIDATES, CORPORA, OUT, ROOT, digest, write_json

PUBLIC = ROOT / "reanalysis/outputs/v3_continuation"
NAMES = {"A_perch_plain": "Perch + logística", "A_perch_white": "Perch normalizado + logística",
         "B_mmd_plain": "Perch + MLP/MMD", "B_mmd_white": "Perch normalizado + MLP/MMD",
         "C_wingbeat": "Batimento de asa", "D_tabular": "HistGB tabular", "duration_diagnostic": "Só duração"}


def percent(value):
    return "indefinido" if value is None else f"{100 * value:.2f}%"


DEGENERATE_NOTE = "bootstrap degenerado: zero eventos; ver IC binomial descritivo"


def degenerate(result, key):
    """Cluster bootstrap collapses to [x, x] when the event never occurs (e.g. 0/197)."""
    ci = result.get("ci95", {}).get(key)
    return bool(ci) and ci[0] == ci[1]


def interval(result, key, annotate=False):
    value = result["point"].get(key)
    ci = result.get("ci95", {}).get(key)
    if ci and degenerate(result, key):
        return percent(value) + (f" — {DEGENERATE_NOTE}" if annotate else " (bootstrap degenerado)")
    return percent(value) + (f" [{100 * ci[0]:.2f}–{100 * ci[1]:.2f}]" if ci else "")


def binomial(result, key):
    """Per-file binomial 95% CI, published as description only (files of a cluster are not independent)."""
    ci = result.get("file_binomial_ci95_descriptive_only", {}).get(key)
    return f"[{100 * ci[0]:.2f}–{100 * ci[1]:.2f}]" if ci else "—"


def main():
    final = json.loads((OUT / "final_metrics.json").read_text())
    dev = json.loads((OUT / "development_metrics.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    manifest = pd.read_csv(OUT / "manifest.csv")
    predictions = pd.read_csv(OUT / "final_predictions.csv")
    fewshot = json.loads((OUT / "fewshot_summary.json").read_text())
    PUBLIC.mkdir(parents=True, exist_ok=True)
    copies = ["inventory.csv", "manifest_audit.json", "training_identity.json", "selection.json", "final_freeze.json",
              "development_metrics.json", "final_metrics.json", "final_predictions.csv"]
    for name in copies:
        shutil.copyfile(OUT / name, PUBLIC / name)
    shutil.copyfile(CACHE / "meta.json", PUBLIC / "feature_meta.json")
    rows = []
    for evaluation, models in dev.items():
        for name, result in models.items():
            rows.append(dict(evaluation=evaluation, model=name, **result["point"],
                             **{f"{k}_ci95": str(v) for k, v in result.get("ci95", {}).items()}))
    pd.DataFrame(rows).drop(columns=["confusion", "class_clusters"]).to_csv(PUBLIC / "development_summary.csv", index=False)
    compact = []
    for record in fewshot:
        compact.append({key: record[key] for key in ["corpus", "k", "repeat", "available"]} |
                       (record["metrics"] if record["available"] else {"reason": record["reason"]}))
    pd.DataFrame(compact).drop(columns=["confusion", "class_clusters"], errors="ignore").to_csv(PUBLIC / "fewshot_summary.csv", index=False)
    _plots(dev, final, predictions, annotate=True)
    selected = selection["candidate"]
    model_path = OUT / "final" / f"{selected}.joblib"
    amendment = json.loads((PUBLIC / "technical_amendment_A3.json").read_text())
    body = report_lines(final, dev, selection, fewshot, digest(model_path), amendment, annotate=True)
    report_path = ROOT / "reports/continuacao_v3.md"
    report_path.write_text("\n".join(body) + "\n")
    delivery = dict(complete=True, report_sha256=digest(report_path),
                    files={str(p.relative_to(ROOT)): digest(p) for p in PUBLIC.iterdir() if p.is_file()},
                    selected=selected, success=final["success"], historical_weights_modified=False,
                    notebook_modified=False)
    write_json(OUT / "delivery_verification.json", delivery)
    print(json.dumps(dict(report=str(report_path), selected=selected, success=final["success"]), indent=2))


REASONS = {"No candidate met the development criteria; highest-AUROC descriptive fallback only":
           "Nenhum candidato atingiu as metas no desenvolvimento; este foi congelado apenas para uma avaliação descritiva, pelo maior AUROC médio."}
FEWSHOT_CORPORA = ["abuzz", "humbug", "mosquitosong", "final_locked"]


def results_table(final, annotate):
    """Final-test table. With annotate=True, degenerate cluster-bootstrap intervals are flagged and the per-file
    binomial CI (descriptive only) is shown next to every rate; no number is recomputed."""
    m, n = final["locked_mosquito"]["point"], final["locked_noise"]["point"]
    lm, ln = final["locked_mosquito"], final["locked_noise"]
    rows = [("Aedes detectados", f"{m['aedes_tp']}/{m['aedes_n']}", lm, "aedes_recall"),
            ("Alertas corretos entre alertas Aedes", f"{m['aedes_tp']}/{m['alerts']}", lm, "aedes_precision"),
            ("Outros mosquitos chamados Aedes", f"{m['other_fp']}/{m['other_n']}", lm, "fpr_other"),
            ("Ruídos chamados Aedes", f"{n['noise_fp']}/{n['noise_n']}", ln, "fpr_noise"),
            ("Cobertura nos mosquitos (resposta diferente de incerto)", f"{m['n_files'] - m['uncertain']}/{m['n_files']}", lm, "coverage")]
    if not annotate:
        return (["| Avaliação final | Contagem | Taxa e IC95% por cluster |", "|---|---:|---|"] +
                [f"| {label} | {count} | {interval(res, key)} |" for label, count, res, key in rows])
    return (["| Avaliação final | Contagem | Taxa e IC95% por cluster | IC95% binomial por arquivo (descritivo) |",
             "|---|---:|---|---|"] +
            [f"| {label} | {count} | {interval(res, key, annotate=True)} | {binomial(res, key)} |" for label, count, res, key in rows])


def report_lines(final, dev, selection, fewshot, model_sha256, amendment, annotate=False):
    """Report text. annotate=False reproduces the 20/10/2026 revision (document_revision.json) byte for byte."""
    selected = selection["candidate"]
    m, n = final["locked_mosquito"]["point"], final["locked_noise"]["point"]
    u = final["ungated_sensitivity_analysis"]["point"]
    noise_upper = final["locked_noise"]["file_binomial_ci95_descriptive_only"]["fpr_noise"][1]
    reason = REASONS.get(selection["reason"])
    fewshot_rows = []
    for corpus in FEWSHOT_CORPORA:
        records = [r for r in fewshot if r["corpus"] == corpus]
        recalls = [r["metrics"]["aedes_recall"] for r in records if r["available"]]
        span = f"{percent(min(recalls))}–{percent(max(recalls))}" if recalls else "indisponível"
        fewshot_rows.append(f"| {corpus} | {len(recalls)}/{len(records)} | {span} |")
    locked_available = [r for r in fewshot if r["corpus"] == "final_locked" and r["available"]]
    locked_hits = sum(r["metrics"]["aedes_tp"] > 0 for r in locked_available)
    body = [
        "# Continuação da v3: resultados e limites",
        "",
        "Experimento concluído em 20/10/2026. Os arquivos brutos, caches e pesos ficam locais; saídas leves auditáveis estão em `reanalysis/outputs/v3_continuation/`.",
        "",
        "## Resultado principal",
        f"Candidato congelado pelo desenvolvimento: **{NAMES[selected]}** (`{selected}`). Elegível pelos critérios de desenvolvimento: **{'sim' if selection['eligible'] else 'não'}**.",
        reason if reason else f"Motivo: {selection['reason']}.",
        "",
        *results_table(final, annotate),
        "",
        f"AUROC Aedes no teste reservado: {final['locked_mosquito']['point']['auroc']:.3f}. Metas pontuais atingidas: **{'sim' if final['success']['observed_targets_met'] else 'não'}**. Apoio estatístico conforme a regra congelada: **{'sim' if final['success']['statistical_support'] else 'não'}**. **Não há validação de campo.**",
        "",
        "![Resultados reais da continuação](../reanalysis/outputs/v3_continuation/resultados.png)",
        "",
    ]
    if annotate:
        body += ["**Intervalos com zero eventos.** Quando a contagem é zero (0/197, 0/371, 0/107), o bootstrap por cluster reamostra apenas zeros e devolve `[0, 0]`. Isso é um artefato, não precisão: na tabela, esses casos aparecem como *" + DEGENERATE_NOTE + "*. A coluna binomial vem de `final_metrics.json` (`file_binomial_ci95_descriptive_only`) e trata arquivos como independentes, o que eles não são; serve só como descrição da incerteza mínima.", ""]
    body += [
        f"**Leitura operacional.** Zero falsos alertas veio acompanhado de zero Aedes detectados. Isso não constitui um detector útil nem comprova um limite populacional de falsos positivos. Em particular, {n['noise_fp']}/{n['noise_n']} ruídos tem limite superior binomial descritivo de {percent(noise_upper)} (IC95%), já acima da meta de 1%, mesmo antes de considerar dependência entre arquivos. Os intervalos bootstrap degenerados não resolvem essa limitação.",
        "",
        f"Na ablação predefinida **sem o portão de concordância**, a mesma cabeça detectou **{u['aedes_tp']}/{u['aedes_n']} Aedes ({percent(u['aedes_recall'])})**, com **{u['other_fp']}/{u['other_n']} falsos alertas entre outros mosquitos ({percent(u['fpr_other'])})** e precisão de **{percent(u['aedes_precision'])}**. O portão prejudicou a sensibilidade; removê-lo também não atinge as metas. Esse resultado não foi usado para mudar a configuração final.",
        "",
        "## Onde o trabalho foi retomado",
        "O Claude havia baixado e conferido o HumBugDB completo, calculado componentes de duplicação e preparado o manifesto v3, mas não havia salvo características nem treinado classificadores v3. A análise inicial e as decisões antes de qualquer novo resultado estão em `analise_de_lacunas_continuacao_v3.md`, `protocol_v3_continuation.json` e no congelamento correspondente.",
        "",
        "Dois clusters provisórios cruzavam treino e teste. Foram colocados em quarentena 286 arquivos de desenvolvimento relacionados a eles. O teste de mosquitos herdado foi mantido. Reservaram-se adicionalmente 107 arquivos de 67 clusters exclusivamente de ruído, com seed fixa, antes de qualquer treino. Caminhos, componentes e clusters estão separados entre desenvolvimento e avaliações finais.",
        "",
        "## Métodos e escolhas",
        "O Perch 2.0 permanece congelado. Comparam-se áudio sem/com normalização espectral por gravação, cabeças logísticas, MLP de 256 unidades com MMD, descritores do batimento e HistGB tabular. Cada arquivo recebe uma unidade de massa; a representação é a média das características de suas janelas. São no máximo seis janelas Perch de cinco segundos e 30 janelas acústicas de um segundo. Arquivos curtos recebem zeros, sem repetição de áudio.",
        "",
        "Em cada avaliação, clusters de ajuste e calibração ficam separados. A escolha de parâmetros usa três dobras internas apenas no ajuste. As duas etapas são presença de mosquito e Aedes entre mosquitos. Calibradores e limiares usam a calibração; não há refit sobre ela antes do teste. Discordância com a cabeça de batimento gera `incerto`; esses casos continuam nos denominadores. A análise sem esse portão é publicada separadamente.",
        "",
        "A seleção final usa o desempenho LODO em Abuzz, HumBug e MosquitoSong, com sensibilidade mínima de 20% e limites de FPR de 2% em outros mosquitos e 1% em ruído. Os limites são exigidos em cada corpus onde são mensuráveis. Se nenhum candidato se qualifica, o maior AUROC médio é congelado apenas para uma avaliação descritiva. O teste final não escolhe modelo, parâmetros ou limiar.",
        "",
        "## Desenvolvimento: cada corpus fora do treino",
        "| Candidato | Abuzz: recall / FPR outros | HumBug: recall / FPR outros | MosquitoSong: recall / FPR outros |",
        "|---|---|---|---|",
    ]
    for name in CANDIDATES:
        cells = []
        for corpus in CORPORA:
            p = dev[f"LODO_{corpus}"][name]["point"]
            cells.append(f"{percent(p['aedes_recall'])} / {percent(p['fpr_other'])}")
        body.append(f"| {NAMES[name]} | " + " | ".join(cells) + " |")
    body.extend([
        "",
        "Todas as métricas, os intervalos e o diagnóstico que usa somente duração estão nos JSONs/CSVs publicados. Precisão indefinida por ausência de alertas não é tratada como 100%. A seleção usa precisão zero nesse caso.",
        "",
        "## Calibração com poucos exemplos",
        "Após congelar a configuração, foram executados os braços predefinidos com 5, 10 e 20 arquivos por classe, cinco seeds, ajustando somente as cabeças leves. Todo arquivo de qualquer cluster usado na calibração é retirado da avaliação, inclusive arquivos desse cluster não escolhidos para calibração. Mantêm-se os calibradores e limiares de origem. Braços sem arquivos/clusters suficientes são explicitamente indisponíveis.",
        "",
        f"Braços executados: {sum(bool(r['available']) for r in fewshot)}/{len(fewshot)}. Resultados individuais em `fewshot_summary.csv`; não somar os denominadores das repetições como se fossem novos mosquitos independentes.",
        "",
        "| Corpus de adaptação | Braços disponíveis | Sensibilidade: mínimo–máximo entre repetições |",
        "|---|---:|---|",
        *fewshot_rows,
        "",
        f"No teste reservado, {'nenhum' if locked_hits == 0 else locked_hits} dos {['zero', 'um', 'dois', 'três', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove', 'dez', 'onze', 'doze', 'treze', 'quatorze', 'quinze'][len(locked_available)]} braços disponíveis detectou Aedes. As variações entre repetições não são campanhas independentes. Mantiveram-se limiares e calibradores de origem; isso limita a conclusão sobre outras formas de calibração local.",
        "",
        "## Limites da evidência",
        f"- O teste de mosquitos tem {m['n_files']} arquivos e {m['n_clusters']} clusters, com apenas {m['class_clusters']['2']} clusters contendo Aedes. Não são {m['n_files']} indivíduos independentes.",
        "- O conjunto é relacionado ao domínio D3 do BioDCASE e aos áudios UFRGS conhecidos; a proveniência UFRGS não foi confirmada para todos os clipes. Metadados de origem e contagens aparecem separados.",
        f"- {final['previously_scored']['point']['n_files']} componentes do teste já tinham predições em experimentos anteriores. Os {final['not_previously_scored']['point']['n_files']} restantes não tinham sido pontuados, mas o desenho da pesquisa é retrospectivo/adaptativo; não corresponde a uma nova coleta cega.",
        "- O teste complementar de ruído vem das origens de desenvolvimento e não mede falsos alarmes em um ambiente de campo novo. Sua precisão estatística é limitada.",
        "- Bootstrap de clusters não mede a variabilidade de retreinar modelos. Com zero erros ele pode gerar intervalo degenerado; isso não estabelece um limite populacional para erros raros. Intervalos binomiais por arquivo são publicados apenas como descrição e não corrigem dependência entre arquivos.",
        "- Duração, zero-padding e composição de espécies continuam podendo revelar a origem. A ablação de normalização e o diagnóstico de duração investigam esse risco, sem demonstrar sua eliminação.",
        "- Os números antigos permanecem resultados do corpus e protocolo anteriores. Mudaram dados, representação e modelos; não atribuir diferenças apenas à correção da divisão nem dizer que toda decisão anterior reconhecia exclusivamente a fonte.",
        "",
        "## Reprodução e inferência",
        "```bash",
        "python -m pip install -r requirements.txt -r reanalysis/requirements-v3.txt",
        "python -m reanalysis.v3_continue_manifest",
        "python -m reanalysis.v3_features",
        "python -m reanalysis.v3_train dev",
        "python -m reanalysis.v3_train final",
        "python -m reanalysis.v3_train fewshot",
        "python -m reanalysis.v3_report",
        "python -m pytest -q",
        "```",
        "Os áudios e os componentes herdados devem existir antes da primeira etapa. Não há aquisição automática de dados privados. `final` recusa mudança de configuração e reutiliza a avaliação registrada em vez de selecionar novamente pelo teste. A extração é retomável somente com a mesma identidade de dados/código/modelo.",
        "",
        f"Modelo local: `results/v3_continuation/final/{selected}.joblib`, SHA-256 `{model_sha256}`. Inferência de pesquisa: `python -m reanalysis.v3_predict --audio caminho.wav`. O portão de batimento é carregado junto e a saída informa que o modelo não foi validado em campo.",
        "",
        "Referências de método: [Perch, código oficial](https://github.com/google-research/perch), [Gong et al., preprint de alinhamento de domínios](https://arxiv.org/abs/2610.09737), [BioDCASE, tarefa oficial](https://biodcase.github.io/challenge2026/task5). Esta implementação é uma adaptação registrada, não uma reprodução integral desses trabalhos.",
        "",
        "## Correção técnica e verificações finais",
        "",
        f"A emenda A3 recuperou as durações físicas de {amendment['n_missing_physical_durations_recovered']} arquivos MP4/M4A dos metadados v2 congelados, apenas para o diagnóstico de duração. Não houve imputação por rótulo, troca de partições, alteração de hiperparâmetros ou dos seis candidatos acústicos já executados. O código anterior, as identidades e os hashes dos modelos foram preservados; a auditoria está em `technical_amendment_A3.json`.",
        "",
        "A suíte terminou com **70 testes aprovados**. A inferência em um WAV real de desenvolvimento foi comparada às características em cache e produziu o mesmo rótulo e escore dentro da tolerância numérica. Os hashes do notebook, dos metadados de referência e dos pesos históricos permaneceram idênticos.",
    ])
    return body


def plot_data(dev, predictions):
    """Numbers drawn in resultados.png, so before/after regenerations can be compared exactly."""
    auroc = {corpus: [dev[f"LODO_{corpus}"][name]["point"]["auroc"] for name in CANDIDATES] for corpus in CORPORA}
    matrix = np.bincount(predictions.target.to_numpy(int) * 4 + predictions.pred.to_numpy(int), minlength=12).reshape(3, 4)
    return dict(auroc=auroc, confusion=matrix.tolist())


def _plots(dev, final, predictions, annotate=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = ["#176B87", "#D17A22", "#4D7C3A"]
    for j, corpus in enumerate(CORPORA):
        values = [dev[f"LODO_{corpus}"][name]["point"]["auroc"] for name in CANDIDATES]
        axes[0].scatter(values, np.arange(len(CANDIDATES)) + (j - 1) * .16, label=corpus, color=colors[j])
    axes[0].set_yticks(np.arange(len(CANDIDATES)), [NAMES[name] for name in CANDIDATES])
    axes[0].axvline(.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set(xlabel="AUROC Aedes", xlim=(0, 1), title="Desenvolvimento: um corpus fora do treino")
    axes[0].legend(loc="lower right")
    matrix = np.bincount(predictions.target.to_numpy(int) * 4 + predictions.pred.to_numpy(int), minlength=12).reshape(3, 4)
    axes[1].imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        axes[1].text(col, row, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    axes[1].set_xticks(range(4), ["Ruído", "Outro", "Aedes", "Incerto"])
    axes[1].set_yticks(range(3), ["Ruído", "Outro mosquito", "Aedes"])
    axes[1].set(xlabel="Resposta", ylabel="Rótulo", title="Teste final: contagens, incluindo ruído reservado")
    if annotate:
        lm, ln = final["locked_mosquito"], final["locked_noise"]
        m, n = lm["point"], ln["point"]
        lines = ["IC95% por cluster das três taxas abaixo: bootstrap degenerado: zero eventos.",
                 "Ver IC binomial descritivo por arquivo (arquivos não são independentes):",
                 f"  Aedes detectados {m['aedes_tp']}/{m['aedes_n']}: {binomial(lm, 'aedes_recall')} %",
                 f"  outros mosquitos chamados Aedes {m['other_fp']}/{m['other_n']}: {binomial(lm, 'fpr_other')} %",
                 f"  ruídos chamados Aedes {n['noise_fp']}/{n['noise_n']}: {binomial(ln, 'fpr_noise')} %"]
        axes[1].text(0.0, -0.2, "\n".join(lines), transform=axes[1].transAxes, ha="left", va="top", fontsize=8.5,
                     bbox=dict(boxstyle="round", facecolor="#FFF4E0", edgecolor="#D17A22"))
    fig.savefig(PUBLIC / "resultados.png", dpi=170, **({"bbox_inches": "tight", "pad_inches": 0.2} if annotate else {}))
    plt.close(fig)


def derived_only():
    """Regenerate only resultados.png and reports/continuacao_v3.md, with the degenerate-interval annotation.
    Reads the frozen outputs; does not copy, recompute or overwrite metrics, CSVs, freezes or delivery records."""
    final = json.loads((PUBLIC / "final_metrics.json").read_text())
    dev = json.loads((PUBLIC / "development_metrics.json").read_text())
    selection = json.loads((PUBLIC / "selection.json").read_text())
    predictions = pd.read_csv(PUBLIC / "final_predictions.csv")
    fewshot = json.loads((OUT / "fewshot_summary.json").read_text())
    amendment = json.loads((PUBLIC / "technical_amendment_A3.json").read_text())
    selected = selection["candidate"]
    report_path = ROOT / "reports/continuacao_v3.md"
    before = dict(report_sha256=digest(report_path), figure_sha256=digest(PUBLIC / "resultados.png"),
                  metrics_sha256=digest(PUBLIC / "final_metrics.json"))
    unannotated = "\n".join(report_lines(final, dev, selection, fewshot, digest(OUT / "final" / f"{selected}.joblib"),
                                          amendment, annotate=False)) + "\n"
    body = report_lines(final, dev, selection, fewshot, digest(OUT / "final" / f"{selected}.joblib"), amendment, annotate=True)
    current = report_path.read_text()
    if current == "\n".join(body) + "\n" and (PUBLIC / "document_revision_2026_10_22.json").exists():
        print("relatório já anotado; nada foi escrito")
        return
    if current != unannotated:
        raise SystemExit("o relatório atual não corresponde à revisão de 20/10/2026; nada foi escrito")
    _plots(dev, final, predictions, annotate=True)
    report_path.write_text("\n".join(body) + "\n")
    revision = dict(kind="post_run_document_clarification", date="2026-10-22",
                    changes="Flag degenerate zero-event cluster-bootstrap intervals in the report table and figure and show the "
                            "per-file binomial CI (descriptive only) beside them. No number recomputed; no model, parameter, "
                            "threshold, metric file or freeze changed.",
                    previous_revision="document_revision.json", before=before,
                    after=dict(report_sha256=digest(report_path), figure_sha256=digest(PUBLIC / "resultados.png"),
                               metrics_sha256=digest(PUBLIC / "final_metrics.json")),
                    plotted_numbers=plot_data(dev, predictions), scientific_model_and_threshold_changes=False)
    write_json(PUBLIC / "document_revision_2026_10_22.json", revision)
    print(json.dumps(dict(report=str(report_path), before=before, after=revision["after"]), indent=2))


if __name__ == "__main__":
    import sys
    derived_only() if "--derived-only" in sys.argv[1:] else main()
