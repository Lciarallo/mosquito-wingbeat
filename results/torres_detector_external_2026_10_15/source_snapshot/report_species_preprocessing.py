"""Render the measured preprocessing comparison, with paired recording errors."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from species_preprocessing import SpeciesFrontend, VARIANTS
from torres_lstm import MelSequence, write_json


ROOT = Path(__file__).resolve().parent
LABELS = dict(original="Original · 32 mel", hires_logmel="FFT 2048 · 64 mel",
              hires_contrast="64 mel + contraste", hires_pcen="64 mel + PCEN")


def read_json(path):
    return json.loads(Path(path).read_text())


def pct(value):
    return "não estimável" if value is None else f"{value * 100:.2f}%".replace(".", ",")


def paired_comparison(directory, candidate, reference, frame):
    new = pd.read_csv(directory / f"{candidate}_recordings.csv").set_index("group").sort_index()
    old = pd.read_csv(directory / f"{reference}_recordings.csv").set_index("group").sort_index()
    pd.testing.assert_series_equal(new.target, old.target)
    a, b = new.prediction.eq(new.target), old.prediction.eq(old.target)
    block = frame.groupby("group").pcm_block.first().reindex(new.index)
    table = pd.DataFrame(dict(difference=a.astype(int) - b.astype(int), block=block, count=1))
    clusters = table.groupby("block")[["difference", "count"]].sum().to_numpy()
    rng = np.random.default_rng(20261015)
    indices = rng.integers(0, len(clusters), size=(2000, len(clusters)))
    summed = clusters[indices].sum(1)
    interval = np.quantile(summed[:, 0] / summed[:, 1], [.025, .975]).tolist()
    return dict(candidate=candidate, reference=reference, total=len(a),
                corrected_errors=int((a & ~b).sum()), introduced_errors=int((~a & b).sum()),
                both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum()),
                accuracy_difference=float((a.astype(int) - b.astype(int)).mean()),
                pcm_blocks=len(clusters), exploratory_cluster_bootstrap_95_interval=interval)


def make_figures(directory, rows, frame, selected):
    figure_path = directory / "figures"
    figure_path.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    f, ax = plt.subplots(figsize=(12, 5.6))
    for i, (model, color, label) in enumerate([("linear", "#82918A", "Referência linear"), ("bilstm", "#278333", "Bi-LSTM")]):
        values = [next(r for r in rows if r["variant"] == v and r["model"] == model) for v in LABELS]
        positions = np.arange(4) + (i - .5) * .35
        bars = ax.bar(positions, [r["test"]["accuracy"] * 100 for r in values], .33, color=color, label=label)
        for bar, row in zip(bars, values):
            mark = " ★" if row["id"] == selected else ""
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    pct(row["test"]["accuracy"]) + mark, ha="center", fontsize=9)
    ax.set(xticks=range(4), xticklabels=list(LABELS.values()), ylabel="Acerto em todas as gravações (%)",
           ylim=(0, 100), title="20 espécies · mesmas 190 gravações de teste")
    ax.grid(axis="y", alpha=.2);ax.set_axisbelow(True);ax.legend()
    f.text(.5, .015, "★ Candidato escolhido pela perda de validação, antes das novas previsões de teste.",
           ha="center", fontsize=9)
    f.tight_layout(rect=(0, .035, 1, 1));f.savefig(figure_path / "comparacao.png", dpi=180);plt.close(f)

    # Example is fixed by metadata order among training rows, not selected by result.
    sample = frame.loc[frame.split.eq("train")].iloc[0]
    waves = np.load(ROOT / "data/cache/large_cnn/waves.npy", mmap_mode="r")
    wave = torch.tensor(np.array(waves[int(sample.wave_index):int(sample.wave_index) + 1]))
    f, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, variant in zip(axes.flat, LABELS):
        frontend = MelSequence() if variant == "original" else SpeciesFrontend(variant)
        with torch.inference_mode():
            features = frontend(wave)[0].numpy()
        c = frontend.config
        mel_edges = np.linspace(2595 * np.log10(1 + 80 / 700), 2595 * np.log10(1 + 4000 / 700), c["bands"] + 2)
        frequencies = 700 * (10 ** (mel_edges[1:-1] / 2595) - 1)
        times = (np.arange(len(features)) * 256 + c["n_fft"] / 2) / 16000
        plot = ax.pcolormesh(times, frequencies, features.T, shading="nearest", cmap="magma")
        ax.set(title=LABELS[variant], xlabel="Tempo (s)", ylabel="Frequência (Hz)", ylim=(80, 4000))
        f.colorbar(plot, ax=ax, label="Energia PCEN" if variant == "hires_pcen" else "Atributo em escala log")
    f.suptitle(f"Mesma janela de treino · {sample.label}")
    f.tight_layout();f.savefig(figure_path / "espectrogramas.png", dpi=180);plt.close(f)
    write_json(directory / "spectrogram_example.json", dict(group=str(sample.group), path=sample.path,
        split="train", wave_index=int(sample.wave_index), start_s=float(sample.start_s)))

    frozen = read_json(directory / "freeze.json")
    neural_variant = frozen["selected_neural_candidate"].removesuffix("_bilstm")
    history_path = (ROOT / read_json(directory / "protocol.json")["reference"] if neural_variant == "original"
                    else directory / neural_variant) / "bilstm_history.json"
    history = pd.DataFrame(read_json(history_path))
    f, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history.epoch, history.train_loss, label="Treino: janelas", color="#278333")
    ax.plot(history.epoch, history.validation_recording_loss, label="Validação: gravações", color="#DDA600")
    chosen = next(c for c in frozen["candidates"] if c["id"] == frozen["selected_neural_candidate"])
    best_epoch = chosen.get("best_epoch", int(history.loc[history.validation_recording_loss.idxmin(), "epoch"]))
    ax.axvline(best_epoch, ls="--", color="#087C83", label="Checkpoint escolhido")
    ax.set(title="Rede selecionada na validação · " + LABELS[neural_variant], xlabel="Época", ylabel="Entropia cruzada")
    ax.legend();ax.grid(alpha=.2)
    f.tight_layout();f.savefig(figure_path / "treino_validacao.png", dpi=180);plt.close(f)
    return figure_path


def report(directory, report_path):
    directory, report_path = Path(directory).resolve(), Path(report_path).resolve()
    verification = read_json(directory / "verification.json")
    assert verification["complete"]
    rows, protocol, frozen = [read_json(directory / name) for name in ["metrics.json", "protocol.json", "freeze.json"]]
    frame = pd.read_csv(directory / "partitions.csv")
    selected = frozen["selected_candidate"]
    neural = frozen["selected_neural_candidate"]
    figures = make_figures(directory, rows, frame, selected)
    paired = [paired_comparison(directory, c["id"], "original_" + c["model"], frame)
              for c in rows if c["variant"] != "original"]
    write_json(directory / "paired_comparisons.json", paired)
    neural_variant = neural.removesuffix("_bilstm")
    neural_path = ROOT / protocol["reference"] if neural_variant == "original" else directory / neural_variant
    mc = read_json(neural_path / "bilstm_metrics.json")
    unanimous = mc["selective"][0]
    log = ROOT / "tmp/species_preprocessing_pytest_final_2026_10_15.log"
    matches = re.findall(r"(\d+) passed", log.read_text()) if log.exists() else []
    tests = int(matches[-1]) if matches else None
    original = next(r for r in rows if r["id"] == "original_bilstm")
    original_linear = next(r for r in rows if r["id"] == "original_linear")
    selected_linear = min((r for r in rows if r["model"] == "linear"), key=lambda r: r["validation_recording_loss"])
    linear_delta = (selected_linear["test"]["accuracy"] - original_linear["test"]["accuracy"]) * 100
    new_neural = next(r for r in rows if r["id"] == neural)
    delta = (new_neural["test"]["accuracy"] - original["test"]["accuracy"]) * 100
    delta_display = f"{delta:+.2f}".replace(".", ",")
    linear_delta_display = f"{linear_delta:+.2f}".replace(".", ",")
    selected_row = next(r for r in rows if r["id"] == selected)
    hires_neural = next(r for r in rows if r["id"] == "hires_logmel_bilstm")
    result = dict(selected_candidate=selected, selected_neural_candidate=neural,
        selected_linear_candidate=selected_linear["id"], linear_accuracy_gain_percentage_points=linear_delta,
        neural_accuracy_gain_percentage_points=delta, selective=unanimous, paired=paired,
        pytest_passed=tests, verification=verification)
    write_json(directory / "summary.json", result)

    lines = ["# Pré-processamento das 20 espécies: comparação medida", "",
        "**Luiz Eduardo do Nascimento Ciarallo** · 15/10/2026", "",
        "Foram treinadas três novas Bi-LSTMs e três referências lineares, aplicando cada frontend em treino, validação e teste. "
        "A rodada anterior foi reexecutada com seus pesos congelados para confirmar a comparação nas mesmas 190 gravações.", "",
        "## Resultado principal", "",
        f"Na família linear, o frontend escolhido pela menor perda de validação foi **{selected_linear['variant']}**: "
        f"**{selected_linear['test']['correct']}/190 ({pct(selected_linear['test']['accuracy'])})**, "
        f"contra **{original_linear['test']['correct']}/190 ({pct(original_linear['test']['accuracy'])})** anteriormente. "
        f"Ganho de **{linear_delta_display} pontos percentuais**.", "",
        f"A rede selecionada somente pela validação foi **{neural}**: "
        f"**{new_neural['test']['correct']}/190 ({pct(new_neural['test']['accuracy'])})**, "
        f"contra **{original['test']['correct']}/190 ({pct(original['test']['accuracy'])})** da Bi-LSTM anterior. "
        f"Diferença: **{delta_display} pontos percentuais**.", "",
        f"Entre todas as famílias, o menor log loss de validação selecionou **{selected}**, "
        f"com **{selected_row['test']['correct']}/190 ({pct(selected_row['test']['accuracy'])})** no teste. "
        "A tabela mantém todos os candidatos; o teste não foi usado para mudar essa escolha.", "",
        "| Pré-processamento | Modelo | Perda de validação | Acertos/190 | Acerto | F1 macro |",
        "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {LABELS[row['variant']]} | {row['model']} | {row['validation_recording_loss']:.4f} | "
                     f"{row['test']['correct']} | {pct(row['test']['accuracy'])} | {pct(row['test']['macro_f1'])} |")
    lines += ["", "O candidato Bi-LSTM com FFT 2.048 e 64 bandas, sem compensação adicional, obteve "
        f"{hires_neural['test']['correct']}/190 acertos ({pct(hires_neural['test']['accuracy'])}) e F1 macro de "
        f"{pct(hires_neural['test']['macro_f1'])}, contra {pct(original['test']['macro_f1'])} de F1 macro anteriormente. "
        "A tabela permite comparar esse candidato com a escolha de validação. "
        "Os resultados observados no teste não foram usados para trocar o modelo escolhido.", "",
        "## O que mudou no sinal", "",
        "| Componente | Rodada anterior | Novos candidatos |", "|---|---|---|",
        "| Áudio | Mono, 16 kHz, 0,992 s | Igual |",
        "| FFT / janela Hann | 1.024 pontos / 64 ms | 2.048 pontos / 128 ms |",
        "| Espaçamento entre bins | 15,625 Hz | 7,8125 Hz |",
        "| Salto temporal | 256 amostras / 16 ms | Igual |",
        "| Mel | 32 bandas de 80 a 4.000 Hz | 64 bandas na mesma faixa |",
        "| Entrada da rede | 59 passos × 32 bandas | 55 passos × 64 bandas |", "",
        "- **Maior resolução:** amplia a representação dos tons e dos harmônicos sem deslocar suas frequências.",
        "- **Contraste:** subtrai metade da envoltória local, calculada em nove bandas vizinhas. Compensa parte da coloração espectral sem estimar um perfil de ruído por espécie.",
        "- **PCEN:** controle de ganho por banda, suavização temporal de 0,4 s, alpha 0,98, delta 2, expoente 0,5 e epsilon 10⁻⁶. "
        "O estado começa no primeiro frame de cada janela e não atravessa gravações.",
        "- **Normalização:** remoção de média e RMS por janela; média/desvio por banda ajustados só no treino, com piso de desvio 0,001.", "",
        "As novas redes mantêm duas camadas de 256 unidades, dropout 0,20, Adam 0,0003, lote 128, clipping 3, "
        "máximo de 80 épocas e paciência 15. A camada de entrada recebe 64 atributos, portanto o número de parâmetros "
        "também muda em relação ao modelo anterior de 32 bandas. Todos os candidatos novos usam a mesma arquitetura de 64 entradas.", "",
        "## Isolamento e seleção", "",
        "As partições são cópias exatas das partições auditadas da rodada Torres: 209 gravações de treino, "
        "59 de validação e 190 de teste, estas distribuídas em 174 blocos PCM. O mesmo bloco não atravessa as três partições. "
        "A ponderação equilibra classes e gravações; MosquitoSong permanece fora do treino.", "",
        "`freeze.json` registra a escolha de frontend/modelo antes das novas predições de teste. "
        "Os pesos, scalers, caches e scripts executados têm hashes verificados. A reexecução da referência "
        "recuperou as mesmas matrizes de confusão da rodada anterior.", "",
        "## Erros pareados por gravação", "",
        "Cada linha compara com o modelo da mesma família na representação original.", "",
        "| Candidato | Erros anteriores corrigidos | Novos erros introduzidos |", "|---|---:|---:|"]
    for row in paired:
        lines.append(f"| {row['candidate']} | {row['corrected_errors']} | {row['introduced_errors']} |")
    lines += ["", "`paired_comparisons.json` também contém intervalos exploratórios por bootstrap de 2.000 reamostragens "
        "dos 174 blocos PCM, preservando a relação entre gravações do mesmo bloco.", "",
        "## Monte Carlo dropout da rede escolhida", "",
        f"Com 100 passagens e exigência de unanimidade (UC = 0), a rede aceita **{unanimous['accepted']}/190** gravações "
        f"(**{pct(unanimous['coverage'])} de cobertura**), com **{unanimous['correct']} acertos entre as aceitas** "
        f"(**{pct(unanimous['accuracy'])} de acerto condicional**). "
        "A tabela principal usa todas as gravações, sem excluir as incertas.", "",
        "## Gráficos", "",
        f"![Comparação](../{figures.relative_to(ROOT)}/comparacao.png)", "",
        f"![Mesma janela nos quatro frontends](../{figures.relative_to(ROOT)}/espectrogramas.png)", "",
        f"![Treino e validação da rede selecionada](../{figures.relative_to(ROOT)}/treino_validacao.png)", "",
        "A janela dos espectrogramas é a primeira janela de treino na ordem dos metadados; sua seleção não dependeu da previsão.", "",
        "## Executar e conferir", "", "```bash",
        ".venv/bin/python evaluate_species_preprocessing.py --output results/nova_rodada_preprocessamento",
        ".venv/bin/python evaluate_species_preprocessing.py --verify-only results/nova_rodada_preprocessamento",
        ".venv/bin/python report_species_preprocessing.py results/nova_rodada_preprocessamento",
        ".venv/bin/python -m pytest -q", "```", "",
        "O executor preserva diretórios existentes e usa os dados/cache e a rodada Torres como referências de entrada.", "",
        "Inferência da rede escolhida com o frontend correspondente:", "", "```bash",
        f".venv/bin/python predict_torres_2025.py {neural_path.relative_to(ROOT)} --model bilstm --audio caminho/gravacao.wav", "```", "",
        "Inferência da referência linear escolhida na validação, com o mesmo frontend:", "", "```bash",
        f".venv/bin/python predict_torres_2025.py {(ROOT / protocol['reference'] if selected_linear['variant'] == 'original' else directory / selected_linear['variant']).relative_to(ROOT)} --model linear --audio caminho/gravacao.wav", "```", "",
        f"Validação de software: **{tests} testes aprovados**, incluindo os 10 novos contratos de pré-processamento. "
        "`verification.json` valida os oito candidatos, o isolamento, a normalização ajustada no treino e os artefatos congelados.", "",
        "O README histórico protegido pelos manifests foi preservado. As novas entradas de documentação estão em `README_TORRES_2025.md`.", "",
        "## Referências técnicas", "",
        "- Mukundarajan et al.: frequência fundamental e harmônicos do voo. https://elifesciences.org/articles/27854",
        "- Wang et al.: PCEN e compressão dinâmica por banda. "
        "https://research.google/pubs/trainable-frontend-for-robust-and-far-field-keyword-spotting/", "",
        "## Limitações e interpretação", "",
        "- Este é o teste histórico já examinado em outras rodadas, com gravações relacionadas dentro do próprio teste; a comparação é retrospectiva.",
        "- A comparação original versus novos frontends altera FFT, número de bandas e dimensão de entrada. "
        "A comparação entre os três candidatos novos mantém essas três escolhas e muda apenas a representação de energia.",
        "- PCEN e contraste podem atenuar componentes informativos; o ganho precisa ser medido e não decorre automaticamente de um áudio parecer mais limpo.",
        "- A tarefa força uma das 20 espécies de mosquito e não mede falsos alarmes em ruído. A detecção de presença/ternária mantém sua avaliação própria.",
        "- Concordância do dropout indica estabilidade da rede e não garante correção.", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/preprocessamento_especies_2026_10_15.md")
    args = parser.parse_args()
    report(args.directory, args.report)
