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
paragraph(f"O melhor resultado observado foi **{best.model}**, com **{pct(best.balanced_accuracy)} de "
          f"acurácia balanceada por janela** e **{pct(best.recording_macro_recall)} de macro recall por grupo**. "
          f"O intervalo bootstrap descritivo por grupo foi {pct(best.ci_low)} a {pct(best.ci_high)}. "
          "As CNNs compactas testadas ficaram abaixo desse resultado no benchmark principal.")
table(["Questão", "Conclusão sustentada"], [
    ["Outros métodos melhoraram a classificação?", "Sim, no protocolo local. ExtraTrees teve o maior resultado observado."],
    ["Houve comparação com redes neurais?", "Sim. CNN1D, CNN2D com ruído e uma tarefa TinyML de quatro classes."],
    ["Existe explicabilidade?", "Parcial: sinais, sobreposição, erros e sensibilidade. Faltam atribuições das decisões."],
    ["O estudo superou os artigos?", "Superioridade não demonstrada. Dados, métricas e protocolos diferem."],
], [.38, .62])
paragraph("Este documento sintetiza experimentos já executados. A geração do relatório não treinou novos "
          "modelos; recalculou as métricas salvas para conferir a consistência dos números.", small=True)
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

new_page("4. Resultados para 20 espécies")
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
paragraph("**Ainda não foram implementados SHAP, LIME, importância por permutação ou Grad-CAM**. "
          "Também não foi medida a fidelidade ou estabilidade de uma explicação local. O relatório "
          "não atribui uma previsão específica a determinada banda, MFCC ou mecanismo biológico.")
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
paragraph("A conclusão atual é exploratória: a combinação de características acústicas com ExtraTrees "
          "foi a melhor entre os métodos implementados nesse acervo, e os testes de transferência e "
          "ruído mostram limitações práticas que precisam ser resolvidas antes de uso em campo.")

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
output_paths = [pdf_path, md_path, benchmark_figure, noise_figure]
report_audit = dict(
    generated_at=date.isoformat(), analysis_base_commit=base_commit,
    source_run_utc=manifest["generated_utc"], pages=len(pdf.pages),
    new_models_trained=False, model_metrics_independently_recomputed=checks,
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
