"""Build a Portuguese scientific report directly from verified experiment outputs."""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
ASSETS = REPORTS / "assets"
PDF_DIR = ROOT / "output/pdf"
for directory in (REPORTS, ASSETS, PDF_DIR):
    directory.mkdir(parents=True, exist_ok=True)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def pct(value, places=1):
    return f"{100 * value:.{places}f}%".replace(".", ",")

def integer(value):
    return f"{int(value):,}".replace(",", ".")

def decimal(value, places=1):
    return f"{value:.{places}f}".replace(".", ",")

manifest = json.loads((RESULTS / "run_manifest.json").read_text())
execution = json.loads((RESULTS / "notebook_execution.json").read_text())
int8 = json.loads((RESULTS / "int8_audit.json").read_text())
metadata = json.loads((RESULTS / "metadata_diagnostic.json").read_text())
noise_source = json.loads((ROOT / "data/noise_manifest.json").read_text())
species = pd.read_csv(RESULTS / "species_metrics.csv")
tiny = pd.read_csv(RESULTS / "tinyml_metrics.csv")
transfer = pd.read_csv(RESULTS / "device_transfer_metrics.csv")
noise = pd.read_csv(RESULTS / "noise_robustness.csv")
selective = pd.read_csv(RESULTS / "selective_classification.csv")
inventory = pd.read_csv(RESULTS / "inventory.csv")
segments = pd.read_csv(RESULTS / "segments.csv")
best = species.loc[species.recording_macro_recall.idxmax()]
improvements = pd.read_csv(RESULTS / "improvements/species_metrics.csv")
improved_best = improvements.loc[improvements.recording_macro_recall.idxmax()]
paired = pd.read_csv(RESULTS / "improvements/paired_group_differences.csv")
arduino = pd.read_csv(RESULTS / "arduino/metrics.csv")
arduino_selected = arduino.loc[arduino.model.eq("Escolha por validação interna")]
arduino_default = arduino_selected.loc[arduino_selected["mode"].eq("Calibração FPR 10%")].iloc[0]
arduino_protocol = json.loads((RESULTS / "arduino/protocol.json").read_text())
native_audit = json.loads((RESULTS / "arduino/native_audit.json").read_text())
compile_audit = json.loads((RESULTS / "arduino/compile_audit.json").read_text())
importance = pd.read_csv(RESULTS / "arduino/permutation_importance.csv")
persistence = pd.read_csv(RESULTS / "arduino/persistence_metrics.csv")
arduino_noise = pd.read_csv(RESULTS / "arduino/noise_probe.csv")
species_board = RESULTS / "arduino_species"
compact_species = pd.read_csv(species_board / "metrics.csv")
compact_selected = compact_species.loc[compact_species.model.eq("Escolha por validação interna")].iloc[0]
compact_protocol = json.loads((species_board / "protocol.json").read_text())
compact_fold = compact_protocol["default_export_fold"]
compact_default = pd.read_csv(species_board / "fold_metrics.csv").query("fold == @compact_fold").iloc[0]
compact_selective = pd.read_csv(species_board / "default_selective_metrics.csv")
compact_confirmed = compact_selective.iloc[-1]
compact_class_selective = pd.read_csv(species_board / "selective_per_class.csv")
compact_native = json.loads((species_board / "native_audit.json").read_text())
compact_compile = json.loads((species_board / "compile_audit.json").read_text())
frequency = species.loc[species.model.eq("F0: log-verossimilhança")].iloc[0]
date = datetime.now(ZoneInfo("America/Sao_Paulo"))
base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

# Independent recomputation, without fitting or selecting any new model.
primary = segments.loc[(segments.fold_species >= 0) & segments.label.ne("noise")]
class_names = np.asarray(manifest["species"])
targets = np.searchsorted(class_names, primary.label.to_numpy())
recording_files = [f"predictions_classical_{i}.csv" for i in range(1, 7)] + [
    "predictions_species_cnn1d.csv", "predictions_species_cnn2d_noise.csv",
]
checks = []
with np.load(RESULTS / "species_oof_probabilities.npz", allow_pickle=False) as probabilities:
    for index, row in species.iterrows():
        values = probabilities[f"model_{index}"]
        assert values.shape == (len(primary), len(class_names))
        assert np.isfinite(values).all() and np.allclose(values.sum(axis=1), 1, atol=1e-5)
        prediction = values.argmax(axis=1)
        recomputed = dict(
            accuracy=accuracy_score(targets, prediction),
            balanced_accuracy=balanced_accuracy_score(targets, prediction),
            macro_f1=f1_score(targets, prediction, average="macro", zero_division=0),
        )
        for metric, value in recomputed.items():
            assert abs(value - row[metric]) < 1e-10, (row.model, metric)
        recording = pd.read_csv(RESULTS / recording_files[index])
        value = balanced_accuracy_score(recording.true, recording.predicted)
        assert abs(value - row.recording_macro_recall) < 1e-10
        assert recording.group.is_unique and len(recording) == len(primary.group.unique())
        checks.append(dict(model=row.model, window_metrics=recomputed, recording_macro_recall=value))

for column in ["fold_species", "fold_tinyml"]:
    assert segments.groupby("group")[column].nunique().max() == 1
    assert segments.groupby("content_sha256")[column].nunique().max() == 1
assert len(species) == 8 and len(primary) == manifest["species_windows"]
assert manifest["verified_archives"] == 20 and execution["error_outputs"] == 0
assert int8["c_python_matching_predictions"] == 400
with np.load(RESULTS / "improvements/species_oof.npz", allow_pickle=False) as probabilities:
    for i,row in improvements.iloc[1:].reset_index(drop=True).iterrows():
        values=probabilities[f"model_{i}"]
        assert values.shape==(len(primary),len(class_names))
        predicted=values.argmax(axis=1)
        for metric,value in dict(accuracy=accuracy_score(targets,predicted),
                                balanced_accuracy=balanced_accuracy_score(targets,predicted),
                                macro_f1=f1_score(targets,predicted,average="macro")).items():
            assert abs(value-row[metric])<1e-10
        suffix=i if i<4 else "ensemble"
        groups=pd.read_csv(RESULTS/f"improvements/species_groups_{suffix}.csv")
        assert abs(balanced_accuracy_score(groups.true,groups.predicted)-row.recording_macro_recall)<1e-10
arduino_predictions=pd.read_csv(RESULTS / "arduino/predictions.csv.gz")
for row in arduino.itertuples():
    frame=arduino_predictions.loc[arduino_predictions.model.eq(row.model)&arduino_predictions["mode"].eq(row.mode)]
    assert len(frame)==len(segments)
    positive=frame.label.ne("noise")
    assert abs(frame.loc[positive,"detected"].mean()-row.window_recall)<1e-10
    assert abs(frame.loc[~positive,"detected"].mean()-row.window_false_positive_rate)<1e-10
    for mask,metric in [(positive,"source_mean_recall"),(~positive,"source_mean_false_positive_rate")]:
        assert abs(frame.loc[mask].groupby("group").detected.mean().mean()-getattr(row,metric))<1e-10
assert native_audit["matching_decisions"]==native_audit["tested_windows"]==532
assert compile_audit["compile_succeeded"] and not compile_audit["physical_board_tested"]
with np.load(species_board / "oof_probabilities.npz",allow_pickle=False) as saved:
    compact_values = saved["probabilities"].copy()
    compact_indices = saved["segment_index"].copy()
    compact_classes = saved["classes"].tolist()
assert compact_classes==compact_protocol["classes"]
assert set(compact_indices)==set(segments.index) and len(np.unique(compact_indices))==len(segments)
assert np.isfinite(compact_values).all() and np.allclose(compact_values.sum(axis=1),1,atol=1e-5)
compact_frame = segments.iloc[compact_indices].copy()
compact_positive = compact_frame.label.ne("noise").to_numpy()
compact_target = compact_frame.label.map({c:i for i,c in enumerate(compact_classes)}).fillna(-1).to_numpy().astype(int)
for metric,value in dict(accuracy=accuracy_score(compact_target[compact_positive],compact_values[compact_positive].argmax(axis=1)),
                        balanced_accuracy=balanced_accuracy_score(compact_target[compact_positive],compact_values[compact_positive].argmax(axis=1)),
                        macro_f1=f1_score(compact_target[compact_positive],compact_values[compact_positive].argmax(axis=1),average="macro")).items():
    assert abs(value-compact_selected[metric])<1e-10
compact_groups = pd.DataFrame(compact_values[compact_positive],index=compact_frame.loc[compact_positive,"group"].to_numpy()).groupby(level=0).mean()
compact_group_target = compact_frame.loc[compact_positive].groupby("group").label.first().map({c:i for i,c in enumerate(compact_classes)}).reindex(compact_groups.index)
assert abs(balanced_accuracy_score(compact_group_target,compact_groups.to_numpy().argmax(axis=1))-compact_selected.recording_macro_recall)<1e-10
compact_predictions = pd.read_csv(species_board / "predictions.csv.gz")
compact_default_predictions = compact_predictions.loc[compact_predictions.fold.eq(compact_fold)]
for row in compact_selective.itertuples():
    current = compact_default_predictions if row.rule.startswith("Uma janela:") else compact_default_predictions.loc[compact_default_predictions.contiguous_endpoint]
    column = "confirmed_candidate" if row.rule.startswith("2/3") else "eligible_candidate"
    labels = current.label.map({c:i for i,c in enumerate(compact_classes)}).fillna(-1).to_numpy().astype(int)
    emitted = current[column].to_numpy()>=0
    positive_mask = labels>=0
    assert int(emitted.sum())==row.accepted_windows
    assert abs(emitted[positive_mask].mean()-row.positive_coverage)<1e-10
    assert abs((current[column].to_numpy()[emitted]==labels[emitted]).mean()-row.accepted_accuracy_including_noise)<1e-10
assert compact_native["feature_only_matching_threshold_decisions"]==compact_native["feature_only_test_windows"]
assert compact_native["complete_pcm_matching_threshold_decisions"]==compact_native["complete_pcm_test_windows"]
assert compact_compile["compile_succeeded"] and not compact_compile["physical_board_tested"]
for row in compact_class_selective.itertuples():
    current = compact_default_predictions
    column = "eligible_candidate"
    if "2/3" in row.scope:
        current = current.loc[current.contiguous_endpoint]
        column = "confirmed_candidate"
    c = compact_classes.index(row.species)
    emitted = current.loc[current[column].eq(c)]
    correct_count = int(emitted.label.eq(row.species).sum())
    assert len(emitted)==row.emitted_as_species and correct_count==row.correct_identifications
    assert int(current.label.eq(row.species).sum())==row.positive_endpoints
    if len(emitted):
        assert abs(correct_count/len(emitted)-row.emitted_species_precision)<1e-10
    else:
        assert pd.isna(row.emitted_species_precision)
recording = pd.read_csv(RESULTS / "predictions_classical_6.csv")
confusions = (recording.loc[recording.true_label.ne(recording.predicted_label)]
              .groupby(["true_label", "predicted_label"]).size()
              .sort_values(ascending=False).head(6))

BLUE, ORANGE, INK, MUTED = "#2563eb", "#d97706", "#17243b", "#526174"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 240})
labels = {
    "Majoritária": "Classe majoritária", "F0: soma de densidades": "F0: soma de densidades",
    "F0: log-verossimilhança": "F0: log-verossimilhança", "PSD + SVM linear": "PSD + SVM",
    "MFCC + regressão logística": "MFCC + regressão logística",
    "MFCC/PSD/F0 + ExtraTrees": "MFCC/PSD/F0 + ExtraTrees",
    "CNN1D compacta": "CNN1D compacta", "CNN2D + ruído real no treino": "CNN2D + ruído real",
}
order = species.sort_values("recording_macro_recall")
fig, ax = plt.subplots(figsize=(8.0, 4.4))
y = np.arange(len(order))
ax.barh(y, order.recording_macro_recall * 100,
        color=[BLUE if name == best.model else "#91a8c4" for name in order.model])
ax.errorbar(order.recording_macro_recall * 100, y,
            xerr=np.vstack([order.recording_macro_recall-order.ci_low,
                            order.ci_high-order.recording_macro_recall]) * 100,
            fmt="none", ecolor=INK, capsize=3, linewidth=1)
ax.set(yticks=y, yticklabels=[labels[n] for n in order.model], xlim=(0, 90),
       xlabel="Macro recall por grupo (%)")
ax.xaxis.grid(True, alpha=.15); ax.set_axisbelow(True)
for position, value in enumerate(order.recording_macro_recall):
    ax.text(value*100 + 5.0, position, pct(value), va="center", fontsize=9, color=INK)
fig.tight_layout()
benchmark_figure = ASSETS / "benchmark_por_grupo.png"
fig.savefig(benchmark_figure, bbox_inches="tight"); plt.close(fig)

means = noise.groupby(["model", "noise", "added_snr_db"]).balanced_accuracy.mean()
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
for ax, kind in zip(axes, ["ruído real reservado", "ruído gaussiano"]):
    for name, color, short in [(best.model, BLUE, "ExtraTrees"),
                               ("CNN2D + ruído real no treino", ORANGE, "CNN2D + ruído")]:
        curve = means.loc[(name, kind)].sort_index()
        ax.plot(curve.index, curve.to_numpy()*100, "o-", color=color, label=short)
    ax.set(title=kind.capitalize(), xlabel="SNR adicionado (dB)", xticks=[0, 10, 20], ylim=(0, 60))
    ax.grid(alpha=.15); ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("Acurácia balanceada por janela (%)")
fig.tight_layout()
noise_figure = ASSETS / "robustez_ruido.png"
fig.savefig(noise_figure, bbox_inches="tight"); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(8,3.6))
names={"ExtraTrees original (referência)":"ExtraTrees inicial","SVM RBF":"SVM RBF",
       "ExtraTrees + contraste/dinâmica":"ExtraTrees + contraste","HistGradientBoosting + contraste/dinâmica":"Boosting + contraste",
       "MLP compacta + contraste/dinâmica":"MLP + contraste","Ensemble fixo (RBF + árvores + boosting)":"Ensemble"}
for ax,column,title_plot in zip(axes,["balanced_accuracy","recording_macro_recall"],["Balanceada por janela","Macro recall por grupo"]):
    ax.barh(np.arange(len(improvements)),improvements[column]*100,color=["#91a8c4"]+[BLUE]*5)
    ax.set(yticks=np.arange(len(improvements)),yticklabels=[names[name] for name in improvements.model],xlim=(0,100),title=title_plot,xlabel="%")
    for i,value in enumerate(improvements[column]):ax.text(value*100+1,i,pct(value),va="center",fontsize=8)
    ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
fig.tight_layout();improvements_figure=ASSETS/"melhorias_especies.png"
fig.savefig(improvements_figure,bbox_inches="tight");plt.close(fig)

summary=importance.groupby("family").source_weighted_auc_drop.mean().sort_values()
fig,ax=plt.subplots(figsize=(8,3.2));ax.barh(summary.index,summary.values,color=BLUE)
ax.set_xlabel("Queda média de ROC AUC ponderada por fonte");ax.set_xlim(0,.24)
ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
for i,value in enumerate(summary):ax.text(value+.004,i,f"{value:.3f}".replace(".",","),va="center",fontsize=9)
fig.tight_layout();importance_figure=ASSETS/"explicabilidade_detector.png"
fig.savefig(importance_figure,bbox_inches="tight");plt.close(fig)

def font_file(pattern):
    return subprocess.check_output(["fc-match", "-f", "%{file}", pattern], text=True).strip()

pdfmetrics.registerFont(TTFont("ReportSans", font_file("DejaVuSans")))
pdfmetrics.registerFont(TTFont("ReportSans-Bold", font_file("DejaVuSans:style=Bold")))
pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSans-Bold",
                              italic="ReportSans", boldItalic="ReportSans-Bold")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportBody", fontName="ReportSans", fontSize=9.5, leading=14,
                         textColor=colors.HexColor(INK), spaceAfter=8, alignment=TA_LEFT))
styles.add(ParagraphStyle("ReportTitle", fontName="ReportSans-Bold", fontSize=27, leading=33,
                         textColor=colors.HexColor(INK), spaceAfter=16))
styles.add(ParagraphStyle("ReportHeading", fontName="ReportSans-Bold", fontSize=17, leading=22,
                         textColor=colors.HexColor(INK), spaceAfter=13))
styles.add(ParagraphStyle("ReportSubhead", fontName="ReportSans-Bold", fontSize=11, leading=16,
                         textColor=colors.HexColor(BLUE), spaceBefore=5, spaceAfter=8))
styles.add(ParagraphStyle("ReportSmall", fontName="ReportSans", fontSize=8, leading=11.5,
                         textColor=colors.HexColor(MUTED), spaceAfter=7))
styles.add(ParagraphStyle("ReportTable", fontName="ReportSans", fontSize=8, leading=11,
                         textColor=colors.HexColor(INK)))
styles.add(ParagraphStyle("ReportTableHead", fontName="ReportSans-Bold", fontSize=8, leading=11,
                         textColor=colors.white))
story, markdown = [], []
PAGE_WIDTH, PAGE_HEIGHT = A4
CONTENT_WIDTH = PAGE_WIDTH - 40 * mm

def rich(text):
    value = html.escape(str(text), quote=False)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\[([^\]]+)\]\(([^\s)]+)\)",
                   r'<link href="\2" color="#2563eb">\1</link>', value)
    return value

def paragraph(text, small=False):
    markdown.append(str(text) + "\n")
    story.append(Paragraph(rich(text), styles["ReportSmall" if small else "ReportBody"]))

def heading(text, level=2):
    markdown.append("#"*level + " " + text + "\n")
    story.append(Paragraph(rich(text), styles["ReportHeading" if level==2 else "ReportSubhead"]))

def bullets(items):
    for item in items:
        markdown.append("- " + item)
        story.append(Paragraph("&#8226; " + rich(item), styles["ReportBody"]))
    markdown.append("")

def table(headers, rows, widths=None):
    markdown.append("| " + " | ".join(map(str, headers)) + " |")
    markdown.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        markdown.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
    markdown.append("")
    cells = [[Paragraph(rich(v), styles["ReportTableHead"]) for v in headers]]
    cells += [[Paragraph(rich(v), styles["ReportTable"]) for v in row] for row in rows]
    widths = [CONTENT_WIDTH*w for w in widths] if widths else [CONTENT_WIDTH/len(headers)]*len(headers)
    item = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    item.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor(INK)),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f4f9"), colors.white]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor(INK)),
    ]))
    story.extend([item, Spacer(1, 9)])

def figure(path, caption, max_height=100*mm):
    width, height = PILImage.open(path).size
    scale = min(CONTENT_WIDTH/width, max_height/height)
    flow = Image(str(path), width=width*scale, height=height*scale)
    flow.hAlign="CENTER"
    story.append(KeepTogether([flow, Spacer(1, 5), Paragraph(rich(caption), styles["ReportSmall"])]))
    rel = Path(path).relative_to(ROOT)
    markdown.extend([f"![{caption}](../{rel.as_posix()})\n", caption + "\n"])

def new_page(title):
    story.append(PageBreak())
    heading(title)

title = "Classificação acústica de mosquitos"
markdown.append("# " + title + "\n")
story.append(Spacer(1, 20*mm))
story.append(Paragraph(title, styles["ReportTitle"]))
paragraph("Relatório técnico de métodos, resultados, explicabilidade e comparação com a literatura")
paragraph(f"Emitido em {date:%d/%m/%Y}. Execução analisada: {manifest['generated_utc'][:10]}. "
          "Projeto: mosquito-wingbeat. Idioma: português.", small=True)
heading("Resumo executivo", 3)
paragraph(f"O estudo reúne a leitura de três artigos, o download completo de um acervo de celulares "
          f"e a comparação de oito classificadores. O benchmark principal inclui **20 espécies, "
          f"{integer(manifest['species_windows'])} janelas de um segundo e {manifest['species_groups']} grupos**. "
          "As janelas de um mesmo grupo são reservadas juntas na avaliação.")
paragraph(f"A atualização melhorou o macro recall por grupo de {pct(best.recording_macro_recall)} para "
          f"**{pct(improved_best.recording_macro_recall)} com boosting e contraste/dinâmica**, com "
          f"**{pct(improved_best.balanced_accuracy)} de acurácia balanceada por janela**. "
          "As CNNs e MLPs próprias também foram comparadas. A reanálise é exploratória e usa o mesmo corpus.")
paragraph(f"Para Arduino, foi implementada uma rede de 16 unidades com frontend incremental e sketch "
          f"compilado para Nano 33 BLE Sense. No modo padrão, detectou {pct(arduino_default.window_recall)} "
          f"dos trechos positivos e marcou {pct(arduino_default.window_false_positive_rate)} dos trechos "
          "de ruído como candidatos. O sistema ainda erraria bastante; não foi testado numa placa física.")
table(["Questão", "Conclusão sustentada"], [
    ["Outros métodos melhoraram a classificação?", "Sim. Boosting + contraste teve 79,4% de macro recall por grupo; a comparação inicial está preservada."],
    ["Houve comparação com redes neurais?", "Sim. CNNs, MLP de espécies e rede binária de 16 unidades para Arduino."],
    ["Existe explicabilidade?", "Parcial: sinais, erros, sensibilidade e permutação de famílias no detector compacto."],
    ["O estudo superou os artigos?", "Superioridade não demonstrada. Dados, métricas e protocolos diferem."],
], [.38, .62])
paragraph("Este documento sintetiza experimentos já executados. A geração do relatório não treinou novos "
          "modelos; recalculou as métricas salvas para conferir a consistência dos números.", small=True)
paragraph(f"O firmware de espécies para Nano 33 BLE Sense usa dois modelos e saída incerta. "
          f"O modelo exportado acertou {pct(compact_default.accuracy)} por janela na escolha forçada. "
          f"Com rejeição e confirmação, acertou {pct(compact_confirmed.accepted_positive_class_accuracy)} "
          f"entre {integer(compact_confirmed.accepted_positive_windows)} emissões, cobrindo apenas "
          f"{pct(compact_confirmed.positive_coverage)} dos trechos positivos contíguos. "
          "Código compilado e comparado com Python, sem teste físico.")
paragraph("Seções 1-11: análise inicial de referência. Seções 12-15: melhorias e presença. "
          "Seções 16-17: identificação de espécies, rejeição e firmware para a placa.", small=True)
paragraph("Repositório: [Lciarallo/mosquito-wingbeat](https://github.com/Lciarallo/mosquito-wingbeat).", small=True)

new_page("1. Dados, origem e auditoria")
paragraph("O corpus principal corresponde ao DOI **10.5061/dryad.98d7s**, associado ao trabalho de "
          "Mukundarajan et al. [1]. Foram baixados os 20 arquivos RAR completos. As rotas automáticas do "
          "Dryad recusaram o download; foi utilizado o espelho Zenodo 4964774 do mesmo DOI. Nomes, tamanhos "
          "e MD5 foram comparados com os metadados oficiais Dryad; também foram calculados SHA-256 [2].")
table(["Item", "Valor verificado"], [
    ["Arquivos compactados", "20 de 20 íntegros"],
    ["Volume compactado", integer(manifest['download_total_bytes']) + " bytes (1,233 GB decimais)"],
    ["Arquivos inventariados", integer(manifest['inventory_files'])],
    ["Arquivos candidatos após auditoria", integer(manifest['usable_candidate_files'])],
    ["Janelas analisadas, incluindo ruído", integer(manifest['analyzed_windows'])],
    ["Benchmark de espécies", integer(manifest['species_windows']) + " janelas / 569 grupos"],
    ["Classe de ruído", "528 janelas / 44 grupos de fontes"],
], [.52, .48])
heading("Regras de exclusão de arquivos", 3)
reasons = {
    "included": "Incluídos como candidatos", "sierrensis_uncurated_or_copy": "A. sierrensis fora do recorte curado / cópias",
    "compressed_original_has_wav": "Original comprimido com WAV correspondente", "exact_duplicate": "Duplicatas exatas",
    "non_audio": "Arquivos sem áudio", "shorter_than_window": "Áudio com duração menor que 1 s",
    "derived_excerpt_unmapped": "Recortes derivados sem relação suficientemente mapeada",
    "conflicting_exact_labels": "Conteúdo idêntico com rótulos conflitantes",
}
table(["Categoria", "Arquivos"], [[reasons[k], integer(v)] for k,v in manifest['exclusions'].items()], [.83, .17])
paragraph("Na etapa de janelas, 67 trechos duplicados ou conflitantes foram removidos e foi identificada "
          "uma forma de onda com rótulos conflitantes. Grupos com formas de onda idênticas verificáveis "
          "foram conectados antes da divisão. As regras reduzem repetições conhecidas, sem garantir "
          "independência entre indivíduo, colônia ou sessão.")
paragraph("Ruídos complementares vieram do repositório dos autores TinyML [3], com commit e SHA-256 "
          "fixados. Os rótulos de espécie vêm do acervo; não houve anotação independente da atividade "
          "de mosquito em cada segundo. Arquivos e rótulos de ruído também podem ocorrer no corpus original.", small=True)

new_page("2. Pré-processamento e representações")
paragraph("A preparação padroniza formatos e escalas, preservando uma faixa espectral compatível com "
          "gravações antigas a 8 kHz. Reamostrar para 16 kHz não cria informação acima da frequência "
          "originalmente capturada. O processo não aplica um algoritmo avançado de remoção de ruído.")
table(["Etapa", "Implementação"], [
    ["Leitura / conversão", "SoundFile; ffmpeg como alternativa para formatos não decodificados diretamente."],
    ["Canais e taxa", "Média dos canais para mono; reamostragem polifásica com antialiasing quando necessário, alvo 16 kHz."],
    ["Segmentação", "Janelas de 1 s sem sobreposição; até 60 por arquivo, distribuídas ao longo da gravação."],
    ["Amplitude", "Remoção da média (DC) e normalização RMS; descarte de janelas praticamente silenciosas."],
    ["Integridade", "Verificação de valores finitos, hashes e relações de repetição verificáveis."],
], [.27, .73])
heading("Características extraídas", 3)
table(["Representação", "Informação utilizada"], [
    ["MFCC", "Média e desvio de 13 coeficientes: 26 características da distribuição espectral."],
    ["PSD de Welch", "Potência em escala logarítmica; 77 posições de 150 a 3.950 Hz, passo de 50 Hz."],
    ["Frequência / harmônicos", "Histogramas de picos de 200 a 900 Hz; mediana, dispersão, quantis e escore harmônico."],
    ["Resumo espectral", "Centroide, planicidade e entropia, além das estimativas de frequência."],
    ["Espectrograma Mel", "40 bandas entre 100 e 3.900 Hz; FFT de 512 amostras e avanço de 256 (32/16 ms)."],
], [.27, .73])
paragraph("A representação combinada contém 112 características. A estimativa chamada F0 no código "
          "é uma aproximação baseada em picos/harmônicos; não constitui uma medição independente "
          "da frequência fundamental física em todo trecho.")
paragraph("O ganho apresentado adiante pertence à combinação entre representação, modelo e protocolo. "
          "Não foi feita uma ablação controlada para medir separadamente o efeito de normalização, "
          "remoção de DC, faixa espectral ou cada família de características.")

new_page("3. Avaliação e definição das métricas")
paragraph("A avaliação principal usa três dobras estratificadas na tabela de grupos, com todas as "
          "espécies presentes em treino e teste. Grupos são definidos por hashes/relações de conteúdo "
          "verificáveis; trechos das fontes de ruído ficam juntos. São verificadas as separações por "
          "grupo e por forma de onda idêntica. Isso não equivale a reservar automaticamente toda uma "
          "colônia ou população.")
bullets([
    "As previsões fora da dobra são reunidas para calcular os resultados finais; as janelas de teste não treinam seu modelo.",
    "Padronização das características dos modelos lineares é ajustada somente no treino.",
    "Pesos por espécie e grupo reduzem a dominância de classes frequentes e gravações longas. Nas CNNs, amostragem ponderada implementa esse balanceamento.",
    "As CNNs reservam grupos de validação dentro do treino externo e selecionam checkpoints pela perda de validação, com até 25 épocas e paciência de 5.",
    "A semente é 42. As solicitações de determinismo não garantem identidade bit a bit em todos os kernels GPU.",
])
table(["Métrica", "Definição e interpretação"], [
    ["Acurácia geral por janela", "Fração de trechos corretamente classificados; pode ser dominada por classes mais frequentes."],
    ["Acurácia balanceada por janela", "Média do recall das espécies, dando peso igual a cada espécie."],
    ["Macro F1 por janela", "Média do F1 das espécies, combinando precisão e recall."],
    ["Macro recall por grupo", "Média dos escores das janelas em cada grupo, escolha da classe e média do recall entre espécies."],
    ["Intervalo bootstrap", "1.000 reamostragens de grupos dentro das espécies; intervalo percentil de 95%, descritivo e condicionado às previsões salvas."],
], [.31, .69])
paragraph("Agregar várias janelas pode melhorar o resultado por grupo. Portanto, 68,3% por grupo "
          "não significa 68,3% de acertos em todos os segundos de áudio. Os intervalos não incorporam "
          "toda a incerteza de seleção de modelo, repetição do treino ou coleta de novas populações.")
paragraph("Há uma comparação exploratória entre oito métodos, sem teste pareado de significância "
          "das diferenças e sem busca exaustiva de hiperparâmetros. Escolher o maior resultado observado "
          "não fornece uma estimativa imparcial para um produto futuro.")

new_page("4. Resultados iniciais para 20 espécies")
table(["Método", "Acurácia geral", "Balanceada / janela", "Macro F1", "Recall / grupo"], [
    [labels[row.model], pct(row.accuracy), pct(row.balanced_accuracy), pct(row.macro_f1), pct(row.recording_macro_recall)]
    for row in species.itertuples()
], [.38, .145, .17, .135, .17])
paragraph(f"ExtraTrees teve **{pct(best.accuracy)} de acurácia geral**, {pct(best.balanced_accuracy)} "
          f"de acurácia balanceada por janela e {pct(best.recording_macro_recall)} de macro recall por grupo. "
          "Esses valores representam três métricas diferentes e devem ser informados com seus nomes.")
paragraph(f"Em relação ao melhor método baseado somente em frequência (log-verossimilhança), o ganho "
          f"observado foi de **{decimal(100*(best.balanced_accuracy-frequency.balanced_accuracy))} pontos percentuais "
          f"por janela** e **{decimal(100*(best.recording_macro_recall-frequency.recording_macro_recall))} por grupo**. "
          "A diferença por grupo é modesta e não recebeu um teste inferencial específico.")
figure(benchmark_figure, "Figura 1. Macro recall por grupo. Barras de erro: intervalos bootstrap descritivos de 95%.", 86*mm)
paragraph("Fonte: results/species_metrics.csv e previsões fora da dobra. Os tempos de treinamento e "
          "intervalos de todos os métodos estão no CSV. Esta geração do relatório reutiliza os resultados registrados.", small=True)

new_page("5. Redes neurais e tarefa TinyML")
paragraph("Foram comparadas duas arquiteturas compactas próprias para 20 espécies. Ambas recebem "
          "um espectrograma Mel calculado a partir do sinal; a CNN1D faz convoluções ao longo do tempo "
          "usando as bandas como canais. A CNN2D opera no plano tempo-frequência. Não foram reproduzidas "
          "as redes completas DenseNet121, EON ou os pesos dos artigos.")
table(["Rede", "Arquitetura e treinamento"], [
    ["CNN1D", "Conv1D com 24/32 filtros, pooling, pooling global, dropout 0,5 e camada de saída."],
    ["CNN2D", "Conv2D com 16/24 filtros, pooling, pooling adaptativo 4 x 4 e camada densa de 96 unidades."],
    ["Treino", "Adam, LR 0,001, batch 128; no máximo 25 épocas; seleção por validação interna."],
    ["Ruído no treino", "Na CNN2D e CNN TinyML: mistura em cerca de 50% dos exemplos, SNR sorteado entre 5 e 25 dB, somente fontes de treino."],
], [.24, .76])
paragraph("No benchmark principal, ExtraTrees superou numericamente as CNNs implementadas. O resultado "
          "vale para estas arquiteturas, representações e orçamento de treinamento; não estabelece "
          "superioridade geral de árvores sobre redes neurais.")
heading("Tarefa própria de quatro classes", 3)
paragraph("As classes são Aedes aegypti, Aedes albopictus, outros mosquitos e ruído. Foram usados "
          "24.405 trechos e 613 grupos, incluindo 44 grupos de ruído. A seleção é mais ampla que "
          "a do artigo TinyML e as divisões por grupo são próprias.")
table(["Modelo", "Acurácia geral", "Balanceada / janela", "Recall / grupo"], [
    [row.model, pct(row.accuracy), pct(row.balanced_accuracy), pct(row.recording_macro_recall)]
    for row in tiny.itertuples()
], [.40, .20, .20, .20])
figure(RESULTS / "figures/tinyml_confusion.png", "Figura 2. Confusão por classe na tarefa TinyML própria, normalizada pela classe verdadeira.", 56*mm)

new_page("6. Generalização entre aparelhos e ruído")
paragraph("O teste entre aparelhos reserva um celular inferido das pastas e treina nos demais, "
          "com ExtraTrees. São consideradas espécies presentes no teste e com suporte de treino suficiente. "
          "Como as classes mudam entre testes, os percentuais não formam um ranking de aparelhos.")
table(["Aparelho reservado", "Classes", "Grupos", "Recall / grupo"], [
    [row.phone, str(row.n_classes), str(row.n_groups), pct(row.recording_macro_recall)]
    for row in transfer.itertuples()
], [.43, .15, .17, .25])
paragraph("Nos testes com pelo menos sete classes, o macro recall por grupo variou de 19,9% a 49,6%. "
          "O resultado de 100% no iPhone envolve apenas duas classes e 11 grupos, constituindo outra "
          "tarefa. A associação entre aparelho e classe limita a interpretação biológica dos ganhos.")
paragraph(f"Um diagnóstico sem áudio, usando somente aparelho e taxa de amostragem declarada, obteve "
          f"**{pct(metadata['balanced_accuracy'])} de acurácia balanceada por janela**, frente a cerca de 5% "
          "da classe majoritária. Isso evidencia associação entre coleta e rótulos; não mede quanto "
          "cada modelo acústico depende desse fator.")
figure(noise_figure, "Figura 3. Teste de ruído em 160 janelas balanceadas por dobra (480 por condição), com fontes reais reservadas e ruído gaussiano. Médias das três dobras.", 65*mm)
paragraph("SNR de 0 dB corresponde a potências semelhantes de sinal e ruído adicionado. Nessa condição "
          "com ruído real, ExtraTrees obteve 21,0% e CNN2D 23,1% de acurácia balanceada. A pequena vantagem "
          "da CNN nessa sonda não foi testada estatisticamente; com ruído gaussiano, a ordem se inverte.", small=True)

new_page("7. Explicabilidade: evidências e lacunas")
paragraph("A explicabilidade implementada é **parcial e predominantemente descritiva**. Há visualização "
          "dos sinais, espectros e espectrogramas; sobreposição de distribuições de frequência; matrizes "
          "de confusão; e testes de sensibilidade a ruído e aparelho. Essas análises ajudam a entender "
          "os erros e os limites do sistema.")
table(["Análise disponível", "O que permite concluir"], [
    ["Espectros / Mel / picos", "Mostram componentes presentes no sinal e variabilidade entre trechos, sem identificar sua causa biológica."],
    ["Sobreposição de frequências", "Identifica semelhanças descritivas entre distribuições. A medida Bhattacharyya usa todas as gravações e não entra no treino."],
    ["Matrizes de confusão", "Mostram espécies confundidas em previsões fora da dobra, sem explicar a característica responsável."],
    ["Ruído e aparelho", "Revelam sensibilidade e associações de coleta; não são atribuições causais das decisões."],
], [.32, .68])
heading("Exemplos de erros por grupo do ExtraTrees", 3)
table(["Espécie verdadeira", "Predição", "Grupos"], [
    [true, predicted, str(count)] for (true, predicted), count in confusions.items()
], [.44, .44, .12])
paragraph("A tabela lista as maiores contagens de erros; espécies com mais grupos podem aparecer "
          "mais vezes. Não se trata de ranking normalizado de dificuldade. Os rótulos e a agregação "
          "seguem o protocolo de avaliação do acervo.", small=True)
paragraph("Na etapa inicial, não havia SHAP, LIME, importância por permutação ou Grad-CAM. "
          "**Esta atualização acrescenta permutação de famílias para o detector compacto (seção 14)**. "
          "Continuam ausentes SHAP/Grad-CAM e avaliação de fidelidade de uma explicação local; "
          "não são demonstrados mecanismos biológicos causais.")
paragraph("Uma próxima etapa pode avaliar importância agrupada das famílias MFCC/PSD/frequência "
          "em dados reservados, ablações controladas e mapas de relevância das CNNs, com testes de "
          "estabilidade. Essas propostas são trabalho futuro.")

new_page("8. Quantização INT8 e rejeição por escore")
paragraph("A exportação compacta usa a regressão logística de 20 espécies com 26 características MFCC. "
          "Os pesos são INT8, os produtos acumulam em INT32 e escalas/vieses usam ponto flutuante. "
          "É uma quantização do classificador linear, não da CNN nem do frontend completo.")
table(["Versão", "Acurácia geral", "Balanceada / janela", "Macro F1"], [
    ["Logística original (float)", pct(int8['original_float_metrics']['accuracy']),
     pct(int8['original_float_metrics']['balanced_accuracy'], 3), pct(int8['original_float_metrics']['macro_f1'])],
    ["Pesos INT8 / acumulador INT32", pct(int8['int8_metrics']['accuracy']),
     pct(int8['int8_metrics']['balanced_accuracy'], 3), pct(int8['int8_metrics']['macro_f1'])],
], [.40, .20, .22, .18])
paragraph(f"O payload numérico por dobra tem **{int8['payload_bytes']} bytes**. A concordância das "
          f"previsões float/INT8 foi **{pct(int8['prediction_agreement'], 2)}**. O código C da dobra 0 foi "
          f"compilado e coincidiu com o Python em **{int8['c_python_matching_predictions']}/400** entradas "
          "de teste. Esse teste valida a equivalência numérica da amostra, sem medir desempenho no microcontrolador.")
paragraph("Os 892 bytes excluem cálculo de MFCC, firmware, runtime, buffers e RAM. Não houve medição "
          "de flash total, latência, energia ou bateria. Portanto, o valor não é diretamente comparável "
          "à memória de um pipeline EON do artigo TinyML.")
heading("Abstenção exploratória da regressão logística", 3)
table(["Limiar do escore", "Cobertura", "Trechos aceitos", "Acurácia nos aceitos"], [
    [str(row.threshold).replace(".", ","), pct(row.coverage), integer(row.accepted_windows), pct(row.accepted_accuracy)]
    for row in selective.itertuples()
], [.23, .20, .26, .31])
paragraph("Escores maiores selecionam menos trechos e aumentam a acurácia geral nos aceitos. "
          "No limiar 0,9, a cobertura é apenas 7,2%. O resultado não é desempenho sobre todo o acervo. "
          "Os escores não foram calibrados e não houve conjunto desconhecido externo para validar "
          "rejeição de novas espécies ou ruídos. Os limiares não são recomendações de produção.")

new_page("9. Comparação com os estudos publicados")
paragraph("Os resultados dos artigos e os resultados locais respondem a perguntas diferentes. "
          "Uma porcentagem maior com outra unidade de avaliação não comprova superioridade. "
          "A comparação abaixo foi conferida nos PDFs consultados, mantendo o contexto das métricas.")
table(["Trabalho", "Resultado publicado", "Diferença relevante"], [
    ["Mukundarajan et al., 2017 [1]", "Cerca de 35% sem localização e 65% com informação geográfica, média das espécies no protocolo do artigo.",
     "Referências curadas de frequência de fêmeas e filtro geográfico. A avaliação local usa janelas/grupos próprios e não reproduz a matriz de países."],
    ["Altayeb et al., 2022 [3]", "93,8% de acurácia geral de teste em quatro classes.",
     "Subseleção própria de gravações, segmentação sobreposta, frontend e CNN/EON diferentes. A tarefa local usa mais fontes e separação por grupos."],
    ["Fanioudakis et al., 2018 [4]", "96% de acurácia com DenseNet121, seis espécies e holdout aleatório de 20% dos eventos.",
     "Outro corpus: 279.566 eventos ópticos a 8 kHz, não gravações de microfone Dryad. Esse dataset não foi baixado nem treinado aqui."],
], [.26, .32, .42])
heading("O que foi demonstrado", 3)
paragraph("No protocolo local, ExtraTrees apresentou melhor resultado observado que os métodos "
          "de frequência, SVM, regressão logística e CNNs compactas testados. Há evidência de "
          "melhoria da classificação dentro dessa comparação exploratória.")
heading("O que ainda precisa ser demonstrado", 3)
paragraph("Não há uma reprodução dos três protocolos originais, um teste no corpus óptico, "
          "uma equivalência com a rede Edge Impulse ou um teste estatístico pareado de superioridade. "
          "O resultado de 68,3% por grupo não deve ser apresentado como superação dos 65%, 93,8% ou "
          "96% publicados.")
paragraph("MosquitoSong+ [5] motivou o uso de ruído real no treino e o teste com fontes reservadas. "
          "A extensão local usa outra arquitetura, tarefa e acervo; não reproduz o desempenho do estudo.")

new_page("10. Limitações e próximos passos")
bullets([
    "Rótulos temporais fracos: a espécie no arquivo não garante mosquito ativo em cada janela. Ruído e silêncio residual podem participar das previsões.",
    "Dependências não resolvidas: indivíduos, colônias, ambiente e sessões não têm identificação suficiente para garantir independência completa.",
    "Cobertura desigual: espécies têm diferentes números de fontes e durações; algumas estão concentradas em poucas sessões/aparelhos.",
    "Confundimento da coleta: aparelho/taxa de amostragem já predizem parte dos rótulos sem usar o áudio.",
    "Seleção exploratória: comparação de múltiplos modelos e poucas sementes, sem otimização abrangente, teste pareado ou validação externa independente.",
    "Explicabilidade incompleta: faltam atribuições de características e avaliação da fidelidade das explicações.",
    "Implantação não medida: exportar C não valida latência, memória total, consumo ou robustez em hardware de campo.",
])
heading("Plano de avanço, em ordem de prioridade", 3)
table(["Prioridade", "Trabalho futuro e critério de avaliação"], [
    ["1. Melhorar rótulos", "Anotar atividade de mosquito e qualidade em uma amostra reservada; verificar identificação dos espécimes e exclusões."],
    ["2. Validar externamente", "Reservar novas sessões, populações e aparelhos antes do treino, com metadados de sexo/temperatura quando disponíveis."],
    ["3. Isolar contribuições", "Executar ablações de pré-processamento e famílias de características nas mesmas divisões; repetir sementes."],
    ["4. Explicar decisões", "Importância por permutação agrupada, SHAP para árvores e Grad-CAM/oclusão para CNNs, com testes de fidelidade e estabilidade."],
    ["5. Comparar justamente", "Reproduzir protocolos dos artigos e comparar os mesmos exemplos/métricas; usar teste pareado por unidade independente."],
    ["6. Medir implantação", "Executar frontend e modelo no dispositivo alvo; medir RAM/flash, latência, energia e rejeição de classes desconhecidas."],
], [.28, .72])
paragraph("Na etapa inicial, ExtraTrees foi o melhor entre os oito métodos implementados. "
          "As extensões das seções 12-15 melhoraram o resultado local e acrescentaram um protótipo "
          "Arduino. Transferência e ruído continuam impondo limitações antes de uso em campo.")

new_page("11. Reprodutibilidade, auditoria e referências")
paragraph(f"A execução registrada contém **{execution['total_cells']} células, "
          f"{execution['executed_code_cells']} células de código executadas e {execution['error_outputs']} erros**. "
          f"A última execução levou {decimal(execution['elapsed_seconds'])} s, reutilizando caches e checkpoints "
          "de CNN quando as assinaturas coincidiram; esse tempo não é o treinamento completo do projeto.")
paragraph(f"Ambiente: Python 3.14.7; NumPy {manifest['versions']['numpy']}; SciPy {manifest['versions']['scipy']}; "
          f"scikit-learn {manifest['versions']['scikit-learn']}; PyTorch {manifest['versions']['torch']}; "
          f"GPU {manifest['gpu']}. O erro absoluto máximo entre os frontends SciPy/Torch foi "
          f"{manifest['frontend_maximum_absolute_error']:.2e}, abaixo do limite de 1e-4.", small=True)
paragraph("A geração deste documento recalculou acurácia geral, acurácia balanceada e macro F1 "
          "dos oito classificadores a partir das probabilidades fora da dobra e conferiu macro recall "
          "nas previsões por grupo, com tolerância de 1e-10. Também verificou a consistência das dobras "
          "por grupo e por hash de forma de onda. Hashes dos insumos e saídas estão em reports/report_manifest.json.", small=True)
paragraph("Para reproduzir: instalar requirements.txt, seguir README.md e executar o notebook. "
          "Para gerar somente o relatório a partir dos resultados existentes: instalar requirements-report.txt "
          "e executar build_report.py. A máquina precisa de fontconfig e da fonte DejaVu Sans.", small=True)
paragraph(f"Código de análise de referência: {base_commit[:12]}. Fonte dos resultados: results/. "
          "Os áudios completos ficam em data/raw/ e são baixados pelo notebook em um clone novo. "
          "Os PDFs originais dos artigos ficam somente na cópia local.", small=True)
heading("Referências", 3)
references = [
    "[1] Mukundarajan H. et al. (2017). Using mobile phones as acoustic sensors for high-throughput mosquito surveillance. eLife 6:e27854. [DOI](https://doi.org/10.7554/eLife.27854).",
    "[2] Dataset 10.5061/dryad.98d7s. [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.98d7s); [espelho Zenodo 4964774](https://zenodo.org/records/4964774). Metadados e hashes: data/.",
    "[3] Altayeb M., Zennaro M., Rovai M. (2022). Classifying mosquito wingbeat sound using TinyML. GoodIT, 132-137. [DOI](https://doi.org/10.1145/3524458.3547258); [dados/código](https://github.com/Mjrovai/wingbeat-mosquito-tinyml).",
    "[4] Fanioudakis E., Geismar M., Potamitis I. (2018). Mosquito wingbeat analysis and classification using deep learning. EUSIPCO, 2410-2414. [DOI](https://doi.org/10.23919/EUSIPCO.2018.8553542).",
    "[5] Supratak A. et al. (2024). MosquitoSong+: A noise-robust deep learning model for mosquito classification from wingbeat sounds. PLOS ONE 19:e0310121. [DOI](https://doi.org/10.1371/journal.pone.0310121).",
    "[6] Documentação dos métodos: [SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html), [scikit-learn avaliação](https://scikit-learn.org/stable/modules/model_evaluation.html), [PyTorch](https://docs.pytorch.org/docs/stable/index.html).",
]
for reference in references:
    paragraph(reference, small=True)
paragraph("Licenças e direitos: o dataset Dryad/Zenodo declara CC0; ruídos e artigos mantêm os "
          "termos de suas fontes. O relatório publica a análise e as citações, sem reproduzir "
          "os textos integrais dos artigos.", small=True)

new_page("12. Melhorias executadas para 20 espécies")
paragraph("Foram mantidas as mesmas 23.877 janelas, 569 grupos e três dobras externas. "
          "A extensão acrescenta 181 características: contraste espectral local em 75 frequências "
          "(média/desvio), dinâmica dos MFCC e rastreamento de picos. Árvores, boosting e MLP usam "
          "293 entradas; SVM RBF usa as 112 originais. Mudam representação, modelo e hiperparâmetros; "
          "o ganho não foi isolado por ablação de cada componente.")
table(["Método", "Geral / janela", "Balanceada / janela", "Recall / grupo"], [
    [names[row.model],pct(row.accuracy),pct(row.balanced_accuracy),pct(row.recording_macro_recall)]
    for row in improvements.itertuples()
], [.43,.19,.19,.19])
paragraph(f"O boosting obteve {pct(improved_best.accuracy)} de acurácia geral e "
          f"{pct(improved_best.recording_macro_recall)} de macro recall por grupo, ganho de "
          f"**{decimal(100*(improved_best.recording_macro_recall-best.recording_macro_recall))} pontos "
          "percentuais por grupo** sobre ExtraTrees inicial. O ensemble teve a maior acurácia "
          "balanceada por janela (59,0%), mas macro recall por grupo inferior ao boosting.")
figure(improvements_figure,"Figura 4. Etapa inicial e novos métodos nas mesmas dobras. São métricas diferentes, não taxas de detecção de mosquito.",75*mm)
gain=paired.loc[paired.model.eq(improved_best.model)].iloc[0]
paragraph(f"Bootstrap pareado: o intervalo descritivo do ganho de macro recall por grupo do boosting "
          f"foi {decimal(gain.descriptive_ci_low*100)} a {decimal(gain.descriptive_ci_high*100)} pontos "
          "percentuais, 2.000 reamostragens dentro das espécies. É condicionado às previsões salvas, "
          "sem teste inferencial confirmatório, reotimização ou validação externa. A análise é adaptativa "
          "sobre o corpus já utilizado; não estabelece superioridade sobre os artigos.",small=True)
paragraph("A nova MLP 293-128-64-20 teve seleção por perda em grupos internos, até 70 épocas/paciência 10, "
          "e ficou abaixo do boosting. Comparações neurais incluem essa MLP e as CNNs da etapa inicial.",small=True)

new_page("13. Detecção binária e erros do aparelho")
paragraph("Para a intenção de detectar sons em Arduino, foi criada uma tarefa binária independente "
          "da identificação de espécie. O frontend incremental usa Hann/FFT de 512 amostras a 16 kHz, "
          "31 frames por decisão (0,992 s) e 68 características: energia relativa em 31 bandas, "
          "dispersão temporal, flatness e pico de 219-875 Hz. Não armazena um segundo inteiro nem "
          "usa normalização dependente de áudio futuro.")
paragraph("Logística, árvore, floresta e rede de 16 unidades foram comparadas. A seleção usa ROC AUC "
          "ponderada por fonte numa divisão interna de grupos, separada da calibração. A rede foi "
          "escolhida nas três dobras; número de épocas e padronização também respeitam essas divisões. "
          "As saídas são calibradas com pesos iguais para fontes positivas/negativas (mistura 50/50), "
          "que não é a prevalência real do ambiente.")
table(["Modo", "Recall positivo / janela", "Falso positivo / janela", "Recall / fonte", "Falso positivo / fonte"], [
    [row.mode,pct(row.window_recall),pct(row.window_false_positive_rate),pct(row.source_mean_recall),pct(row.source_mean_false_positive_rate)]
    for row in arduino_selected.itertuples()
], [.32,.17,.17,.17,.17])
paragraph(f"**Sim, o aparelho ainda erraria bastante:** no modo moderado padrão, perderia "
          f"{pct(1-arduino_default.window_recall)} dos trechos positivos e marcaria "
          f"{pct(arduino_default.window_false_positive_rate)} dos trechos de ruído como candidatos "
          "antes da confirmação. Há 23.877 trechos positivos de 569 fontes e 528 trechos negativos de "
          "44 fontes. Os rótulos são por arquivo, sem anotação de cada mosquito ativo ou voo.")
paragraph(f"No modo padrão, intervalos bootstrap descritivos por fonte: recall "
          f"{pct(arduino_default.source_recall_ci_low)}-{pct(arduino_default.source_recall_ci_high)}; "
          f"falsos positivos {pct(arduino_default.source_false_positive_rate_ci_low)}-"
          f"{pct(arduino_default.source_false_positive_rate_ci_high)}. As taxas por fonte dão peso igual "
          "a gravações e diferem das taxas por janela, dominadas por fontes longas.",small=True)
paragraph("As metas de 5%/10% na calibração não foram garantidas no teste. Na comparação no limiar "
          "0,5, a rede aumentou recall de 75,9% (logística) para 82,7%, mas falsos positivos passaram "
          "de 33,1% para 35,2%. Isso mostra a troca entre sensibilidade e alarmes; não é uma melhora "
          "uniforme de todas as métricas. A árvore no limiar conservador não aceitou nenhum trecho.")

new_page("14. Explicabilidade e ruído do detector")
paragraph("A atualização executou importância por permutação em cinco famílias de características: "
          "cada família é embaralhada conjuntamente, cinco repetições por dobra, nos dados de teste "
          "reservados. A métrica é a queda de ROC AUC ponderada por fonte. Relações internas da família "
          "são preservadas; as relações entre famílias podem resultar em entradas pouco usuais.")
figure(importance_figure,"Figura 5. Sensibilidade preditiva média do detector compacto após permutar famílias. Sem atribuição causal ou intervalo inferencial.",76*mm)
paragraph("As bandas de 1.125-4.000 Hz tiveram a maior queda média, cerca de 0,175 de ROC AUC; "
          "as bandas de 125-1.125 Hz, 0,091. Isso mostra dependência preditiva do espectro, não prova "
          "que essas bandas representam exclusivamente asas ou que eliminam o efeito do aparelho. "
          "SHAP, LIME, Grad-CAM e fidelidade de explicações locais continuam ausentes.")
heading("Confirmação temporal",3)
paragraph("A regra causal aceita duas de três janelas completas, reiniciando em lacunas ou mudança "
          "de arquivo. O corpus foi amostrado com no máximo 60 janelas por arquivo, muitas espaçadas. "
          "A comparação nos mesmos endpoints contíguos teve 18.180 trechos positivos e 244 negativos, "
          "mas só duas fontes de ruído. No modo padrão, o falso positivo por janela mudou de 44,3% "
          "para 43,9%. A confirmação não demonstrou controle confiável de ruídos persistentes; "
          "não pode ser apresentada como solução comprovada para alarmes contínuos.")
heading("Sonda com ruído real reservado",3)
curve=arduino_noise.loc[arduino_noise["mode"].eq("Calibração FPR 10%")]
table(["Condição", "Recall dos trechos positivos"], [
    ["Sem ruído adicionado",pct(curve.loc[curve.noise.eq("Sem ruído adicionado"),"positive_window_recall"].mean())],
] + [[f"Ruído real: SNR {snr} dB",pct(curve.loc[curve.added_snr_db.eq(snr),"positive_window_recall"].mean())] for snr in [20,10,0]], [.65,.35])
paragraph("São 480 trechos, oito por espécie/dobra, com fontes de ruído exclusivas do teste. "
          "A sonda mede sensibilidade a mistura em níveis definidos; não mede desempenho em campo, "
          "distância de alcance ou alarmes por hora.",small=True)

new_page("15. Firmware Arduino e avanço necessário")
paragraph("O sketch inclui captura PDM, fila de áudio, frontend, rede neural e confirmação, com "
          "avisos e reset quando perde amostras. O alvo inicial é Nano 33 BLE Sense / Sense Rev2, "
          "com microfone integrado. O Uno R3/Nano clássico não tem memória suficiente para este "
          "firmware; outra placa exige adaptador de captura e compilação próprios.")
table(["Verificação", "Resultado / limite"], [
    ["Modelo compacto", "68-16-1 com ReLU, 1.121 parâmetros; aproximadamente 4.504 bytes numéricos incluindo constantes."],
    ["Frontend incremental", f"Objeto C++ nativo: {integer(native_audit['streaming_frontend_object_bytes'])} bytes, mais buffers/componentes no sketch."],
    ["Compilação Nano 33 BLE", f"CLI 1.5.1/core 4.6.0; {integer(compile_audit['program_storage_bytes'])} bytes de programa / {integer(compile_audit['global_static_memory_bytes'])} bytes globais."],
    ["Equivalência PCM-C++/Python", f"{native_audit['matching_decisions']}/{native_audit['tested_windows']} decisões iguais; maior erro de característica {native_audit['max_absolute_feature_error']:.2e}, de escore {native_audit['max_absolute_score_error']:.2e}."],
    ["Lógica de confirmação", "Testes causais e reset por lacunas passaram. Isso não comprova benefício em campo."],
    ["Placa física", "Sem upload, teste do microfone, latência, consumo, bateria ou pico de RAM medidos."],
], [.30,.70])
paragraph("A compilação informa programa e RAM estática, excluindo pico de pilha/heap. O teste nativo "
          "usa 400 trechos positivos e todos os 132 negativos de teste da dobra 0. A exportação padrão "
          "é dessa dobra; o benchmark agregado aplica cada modelo à sua própria dobra. Não houve "
          "retreino e validação de um modelo de produção.",small=True)
paragraph("A primeira confirmação exige três janelas completas, ou 2,976 s de observação, "
          "além do tempo computacional não medido. Passagens muito curtas podem ser perdidas; "
          "não foi validada a detecção de cada voo isolado.",small=True)
heading("Para reduzir erros de forma verificável",3)
bullets([
    "Coletar áudio com a placa e microfone escolhidos no local de uso, incluindo distância, ganho e qualidade de captura.",
    "Anotar mosquito realmente ativo/inativo e gravar horas sem mosquito com ventilador, fala, chuva, máquinas e outros insetos.",
    "Reservar dias e locais inteiros antes do ajuste; treinar com ruídos locais, recalibrar limiares e testar classes desconhecidas.",
    "Medir sensibilidade por evento, falsos alarmes por hora, latência e perdas de amostras no hardware. Estes resultados requerem a coleta física que ainda não existe.",
])
paragraph("Instalação e modos de operação: firmware/README.md. Arquivos de auditoria: "
          "results/arduino/. Documentação oficial: [Nano 33 BLE Sense Rev2](https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2), "
          "[Uno R3](https://docs.arduino.cc/hardware/uno-rev3).",small=True)

new_page("16. Identificação de espécies no Nano 33 BLE Sense")
paragraph("A versão MosquitoSpecies acrescenta identificação de 20 espécies ao microfone PDM "
          "integrado. Reutiliza o frontend de 68 características e o detector de presença, mas "
          "inclui um classificador multiclasse. Foram treinados logística, MLP de 64 unidades "
          "e MLP de 128/64 unidades, com treino, seleção interna, calibração e teste separados "
          "por grupo e conteúdo de forma de onda.")
table(["Candidato", "Geral / janela", "Balanceada / janela", "Recall / grupo"], [
    [row.model.replace("→","-"),pct(row.accuracy),pct(row.balanced_accuracy),pct(row.recording_macro_recall)]
    for row in compact_species.itertuples()
], [.43,.19,.19,.19])
paragraph(f"Forçando uma classe em cada trecho, a escolha interna obteve **{pct(compact_selected.accuracy)} "
          f"de acurácia geral por janela** e {pct(compact_selected.balanced_accuracy)} balanceada. "
          "Ficou abaixo dos classificadores maiores de computador. O frontend, conjunto de ajuste "
          "e seleção são diferentes; não é uma ablação isolada de tamanho da rede.")
paragraph("Aedes mediovittatus tem apenas três grupos: um por teste, calibração e ajuste. "
          "O único grupo de ajuste permanece no treino interno; não há validação interna "
          "independente dessa classe. A seleção usa as demais classes disponíveis. As fontes "
          "de calibração são as mesmas do detector binário, evitando que este treine nelas.",small=True)
paragraph("Uma temperatura de softmax e o limiar de abstenção são ajustados em fontes positivas "
          "de calibração, com pesos por fonte/espécie. A meta pré-definida exige 80% de acerto "
          "ponderado entre aceitos, cobertura ponderada de ao menos 10% e 15 fontes. "
          "Se a meta não é atingida, a dobra rejeita todas as identificações.")
paragraph(f"A exportação usa a **dobra {compact_fold}**, primeira que cumpriu a meta de calibração, "
          "sem escolher pela acurácia de teste. As outras duas rejeitam todas as identificações. "
          f"Esse modelo é 68-64-20, com 5.716 parâmetros e 22.872 bytes numéricos incluindo temperatura/limiar. "
          f"No seu teste, a escolha forçada acertou **{pct(compact_default.accuracy)}** das "
          f"{integer(compact_default.positive_windows)} janelas positivas, com "
          f"{pct(compact_default.balanced_accuracy)} de acurácia balanceada.")
paragraph("Os escores não são garantia de que uma previsão está correta. O rótulo vem do arquivo; "
          "não foi anotado cada voo. Uma espécie desconhecida pode receber o nome de uma classe "
          "conhecida. Não há demonstração de desempenho superior aos artigos ou em campo.",small=True)

new_page("17. Rejeição, firmware de espécies e instalação")
paragraph("O aparelho só emite IDENTIFICACAO_PROVISORIA se a presença passa pelo limiar, "
          "a classe passa pelo seu limiar e a mesma espécie é elegível em duas das três janelas. "
          "A janela atual também precisa ser elegível. INCERTO mostra a candidata, mas não "
          "emite identificação; SEM_EVIDENCIA e AUDIO_INVALIDO indicam outras condições. "
          "Falhas de captura reiniciam o histórico. A primeira emissão exige 2,976 s de áudio.")
table(["Dobra exportada", "Identificações", "Acerto entre emitidas", "Cobertura positiva"], [
    ["Uma janela / todos os trechos",integer(compact_selective.iloc[0].accepted_positive_windows),
     pct(compact_selective.iloc[0].accepted_positive_class_accuracy),pct(compact_selective.iloc[0].positive_coverage)],
    ["Uma janela / endpoints contíguos",integer(compact_selective.iloc[1].accepted_positive_windows),
     pct(compact_selective.iloc[1].accepted_positive_class_accuracy),pct(compact_selective.iloc[1].positive_coverage)],
    ["2/3 / mesmos endpoints",integer(compact_confirmed.accepted_positive_windows),
     pct(compact_confirmed.accepted_positive_class_accuracy),pct(compact_confirmed.positive_coverage)],
], [.40,.18,.21,.21])
paragraph(f"A confirmação acertou **{pct(compact_confirmed.accepted_positive_class_accuracy)} entre "
          f"{integer(compact_confirmed.accepted_positive_windows)} identificações emitidas**, mas cobriu "
          f"somente **{pct(compact_confirmed.positive_coverage)}** dos "
          f"{integer(compact_confirmed.positive_windows)} endpoints positivos contíguos. "
          "Esse acerto condicional não descreve todos os sons: a maioria continua rejeitada. "
          "O modelo não emitiu espécie nos 56 trechos negativos reservados dessa dobra, mas não "
          "há trechos negativos contíguos elegíveis nela para testar confirmação. Zero observado "
          "antes da confirmação não garante zero alarmes de campo.")
paragraph("O acerto também varia entre nomes emitidos: Aedes aegypti teve 21 corretas "
          "em 22 emissões confirmadas; Aedes albopictus, cinco em cinco. São poucos trechos "
          "correlacionados, não indivíduos independentes. Culex quinquefasciatus não teve "
          "nenhuma emissão aceita, portanto não há acerto condicional estimável para essa "
          "classe. Suporte e cobertura por classe estão em selective_per_class.csv.",small=True)
table(["Verificação", "Resultado"], [
    ["Alvo e compilação",f"Nano 33 BLE Sense / Sense Rev2; CLI 1.5.1/core 4.6.0; {integer(compact_compile['program_storage_bytes'])} bytes de programa; {integer(compact_compile['global_static_memory_bytes'])} bytes globais."],
    ["C++ / Python",f"{compact_native['feature_only_matching_threshold_decisions']}/{compact_native['feature_only_test_windows']} decisões com features iguais; {compact_native['complete_pcm_matching_threshold_decisions']}/{compact_native['complete_pcm_test_windows']} no caminho PCM completo."],
    ["Confirmação e checkpoint", "Warmup, discordância, janela incerta, reset e restauração de pesos/calibração passaram."],
    ["Placa física", "Sem upload, teste do microfone, latência total, distância, autonomia ou pico de RAM medidos."],
], [.30,.70])
paragraph("Windows 10/11 x64: baixe MosquitoWingbeat-Windows.exe em "
          "[GitHub Releases](https://github.com/Lciarallo/mosquito-wingbeat/releases/latest). "
          "Abra, conecte a Sense/Sense Rev2, clique em Buscar minha placa e em Instalar no Arduino. "
          "Os resultados aparecem na janela; dispensa Python, IDE e terminal. O executável abriu "
          "a interface e compilou o firmware em Windows no GitHub Actions, sem gravar placa física. "
          "No Linux/macOS com Python 3.9+, extraia MosquitoSpecies.zip e execute "
          "bash flash_arduino.sh --monitor. Alternativa: Arduino IDE/core 4.6.0, alvo Nano 33 BLE, "
          "serial 115200 baud. Primeira preparação: internet e cerca de 1 GB livre. "
          "WINDOWS.md e LEIA_PRIMEIRO.md trazem o passo a passo.",small=True)
paragraph("A memória do compilador exclui pico de pilha/heap. Os guardas de silêncio/clipping "
          "e o tempo computacional ainda precisam de medição física. A melhoria prioritária "
          "continua sendo áudio anotado do próprio Arduino, ruídos locais, mais fontes das "
          "espécies raras e teste em dias/locais reservados. Auditorias: results/arduino_species/.",small=True)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d8e1eb"))
    canvas.line(20*mm, 17*mm, PAGE_WIDTH-20*mm, 17*mm)
    canvas.setFont("ReportSans", 7.2)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(20*mm, 12*mm, "Mosquito-wingbeat | Relatório técnico | " + date.strftime("%d/%m/%Y"))
    canvas.drawRightString(PAGE_WIDTH-20*mm, 12*mm, f"Página {doc.page}")
    if doc.page > 1:
        canvas.setFont("ReportSans", 7)
        canvas.drawString(20*mm, PAGE_HEIGHT-12*mm, "CLASSIFICAÇÃO ACÚSTICA DE MOSQUITOS")
    canvas.restoreState()

pdf_path = PDF_DIR / "Relatorio_Classificacao_Acustica_Mosquitos.pdf"
md_path = REPORTS / "relatorio_classificacao_mosquitos.md"
doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=22*mm, bottomMargin=23*mm,
                        leftMargin=20*mm, rightMargin=20*mm,
                        title=title, author="Projeto mosquito-wingbeat",
                        subject="Métodos, resultados, explicabilidade e comparação com estudos de classificação acústica")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
md_path.write_text("\n".join(markdown), encoding="utf-8")
pdf = PdfReader(pdf_path)
text = "\n".join(page.extract_text() or "" for page in pdf.pages)
assert all(len((page.extract_text() or "").strip()) > 100 for page in pdf.pages)
assert "SHAP" in text and "Superioridade não demonstrada" in text
assert "68,3%" in text and "400/400" in text and "93,8%" in text

input_paths = [RESULTS/p for p in [
    "run_manifest.json", "notebook_execution.json", "species_metrics.csv", "tinyml_metrics.csv",
    "int8_audit.json", "device_transfer_metrics.csv", "noise_robustness.csv", "selective_classification.csv",
    "metadata_diagnostic.json", "inventory.csv", "segments.csv", "species_oof_probabilities.npz",
] + recording_files]
input_paths += [ROOT/"data/noise_manifest.json", RESULTS/"figures/tinyml_confusion.png"]
input_paths += [p for folder in [RESULTS/"improvements",RESULTS/"arduino",species_board] for p in folder.glob("*") if p.is_file()]
input_paths += [p for p in (ROOT/"firmware").rglob("*") if p.is_file()]
input_paths += [ROOT/p for p in ["install_arduino.py","flash_arduino.sh","Instalar_no_Windows.cmd",
    "windows_installer.py","scripts/build_windows.py","requirements-windows.txt","WINDOWS.md",
    ".github/workflows/windows-installer.yml",".gitattributes"]]
output_paths = [pdf_path, md_path, benchmark_figure, noise_figure, improvements_figure, importance_figure]
report_audit = dict(
    generated_at=date.isoformat(), analysis_base_commit=base_commit,
    source_run_utc=manifest["generated_utc"], pages=len(pdf.pages),
    new_models_trained=False, model_metrics_independently_recomputed=checks,
    update_contains_new_executed_experiments=True,improvement_models_metrics_verified=5,
    arduino_operating_rows_independently_verified=len(arduino),
    arduino_species_selected_metrics_independently_verified=True,
    arduino_species_default_operating_rows_independently_verified=len(compact_selective),
    arduino_species_selective_class_rows_independently_verified=len(compact_class_selective),
    group_fold_consistency_asserted=True, identical_waveform_fold_consistency_asserted=True,
    metric_absolute_tolerance=1e-10, pdf_text_checked=True,
    reportlab_version=importlib.metadata.version("reportlab"),
    noise_commit=noise_source["commit"],
    inputs={str(p.relative_to(ROOT)):sha256(p) for p in input_paths},
    outputs={str(p.relative_to(ROOT)):sha256(p) for p in output_paths},
)
(REPORTS/"report_manifest.json").write_text(json.dumps(report_audit, indent=2, ensure_ascii=False))
print(json.dumps(dict(pdf=str(pdf_path), markdown=str(md_path), pages=len(pdf.pages),
                     metrics_verified=len(checks), new_models_trained=False), indent=2, ensure_ascii=False))
