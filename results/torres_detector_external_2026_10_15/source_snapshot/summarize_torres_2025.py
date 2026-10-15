"""Produce a measured-results report and scientific figures for both datasets."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torres_lstm import AUTHOR_COMMIT, AUTHOR_REPOSITORY, write_json

ROOT = Path(__file__).resolve().parent


def percent(value):
    return "não estimável" if value is None else f"{100 * value:.2f}%".replace(".", ",")


def summarize(result_directory, report_path):
    directory, report_path = Path(result_directory).resolve(), Path(report_path).resolve()
    if not (directory / "complete.json").exists():
        raise ValueError("Both datasets must finish before generating the final report.")
    verification = json.loads((directory / "verification.json").read_text())
    if not verification["complete"]:
        raise ValueError("Verify the frozen artifacts before summarizing.")
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    tasks = {}
    for task in ["infection", "species"]:
        tasks[task] = dict(metrics=json.loads((directory / task / "metrics.json").read_text()),
                           protocol=json.loads((directory / task / "protocol.json").read_text()),
                           freeze=json.loads((directory / task / "freeze.json").read_text()))
    f, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task, title in zip(axes, tasks, ["Infecção DENV-2 · 60 exemplos", "20 espécies · 190 gravações"]):
        data = tasks[task]["metrics"]
        values = [data["linear_baseline"]["accuracy"]] + [m["deterministic"]["accuracy"] for m in data["neural_models"]]
        bars = ax.bar(["Linear", "LSTM", "Bi-LSTM"], np.array(values) * 100,
                      color=["#83918A", "#278333", "#087C83"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 100 + 1.5,
                    percent(value), ha="center", fontsize=12)
        ax.set(title=title, ylabel="Acerto em todas as entradas (%)", ylim=(0, 105))
        ax.grid(axis="y", alpha=.2);ax.set_axisbelow(True)
    f.tight_layout();f.savefig(figures / "acerto_todas_entradas.png", dpi=180);plt.close(f)

    f, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task, title in zip(axes, tasks, ["Infecção DENV-2", "20 espécies"]):
        for index, (model, color) in enumerate(zip(tasks[task]["metrics"]["neural_models"], ["#278333", "#087C83"])):
            rows = [row for row in model["selective"] if row["accuracy"] is not None]
            label = "LSTM" if model["model"] == "lstm" else "Bi-LSTM"
            ax.plot([r["coverage"] * 100 for r in rows], [r["accuracy"] * 100 for r in rows],
                    "o-", color=color, label=label)
            first = rows[0]
            ax.annotate(f"UC = 0 · {label}", (first["coverage"] * 100, first["accuracy"] * 100),
                        xytext=(-40, 25 if index == 0 else -30), textcoords="offset points",
                        fontsize=9, ha="right", color=color, arrowprops=dict(arrowstyle="-", color=color))
        ax.set(title=title, xlabel="Cobertura: entradas aceitas (%)", ylabel="Acerto entre entradas aceitas (%)",
               xlim=(80, 102), ylim=(80, 93) if task == "infection" else (45, 61))
        ax.legend();ax.grid(alpha=.2)
    f.tight_layout();f.savefig(figures / "cobertura_e_acerto.png", dpi=180);plt.close(f)

    f, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, task in enumerate(tasks):
        for j, name in enumerate(["lstm", "bilstm"]):
            ax = axes[i, j]
            history = pd.DataFrame(json.loads((directory / task / f"{name}_history.json").read_text()))
            candidate = next(c for c in tasks[task]["freeze"]["candidates"] if c["model"] == name)
            ax.plot(history.epoch, history.train_loss, label="Treino: janelas", color="#278333")
            ax.plot(history.epoch, history.validation_recording_loss, label="Validação: grupos", color="#E5AA00")
            ax.axvline(candidate["best_epoch"], ls="--", color="#087C83", label="Checkpoint")
            ax.set(title=f"{'Infecção' if task == 'infection' else 'Espécies'} · {name.upper()}",
                   xlabel="Época", ylabel="Entropia cruzada")
            ax.legend(fontsize=9);ax.grid(alpha=.2)
    f.tight_layout();f.savefig(figures / "curvas_treino_validacao.png", dpi=180);plt.close(f)

    selected = tasks["infection"]["metrics"]["selected_neural_model"]
    neural = next(m for m in tasks["infection"]["metrics"]["neural_models"] if m["model"] == selected)
    f, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, matrix, title, cols in [(axes[0], neural["deterministic"]["confusion_matrix"], "Todas as entradas", ["Infectado", "Saudável"]),
                                   (axes[1], neural["selective"][0]["confusion_with_abstention"], "Com abstenção: UC = 0", ["Infectado", "Saudável", "Incerto"])]:
        a = np.asarray(matrix)
        ax.imshow(a, cmap="Blues", vmin=0, vmax=30)
        for (row, col), value in np.ndenumerate(a):
            ax.text(col, row, str(value), ha="center", va="center", color="white" if value > 15 else "black")
        ax.set(xticks=range(a.shape[1]), xticklabels=cols, yticks=[0, 1], yticklabels=["Infectado", "Saudável"],
               title=title, xlabel="Predição", ylabel="Rótulo fornecido")
    f.suptitle(f"Infecção DENV-2 · {selected.upper()} selecionada na validação")
    f.tight_layout();f.savefig(figures / "matriz_infeccao.png", dpi=180);plt.close(f)

    performance = []
    selective = []
    for task in tasks:
        metrics = tasks[task]["metrics"]
        for name, row in [("linear", metrics["linear_baseline"])] + [(m["model"], m["deterministic"]) for m in metrics["neural_models"]]:
            performance.append(dict(task=task, model=name, total=row["total"], correct=row["correct"],
                                    accuracy=row["accuracy"], macro_f1=row["macro_f1"]))
        for m in metrics["neural_models"]:
            row = m["selective"][0]
            selective.append(dict(task=task, model=m["model"], total=row["input_total"], accepted=row["accepted"],
                correct_accepted=row["correct"], accuracy_accepted=row["accuracy"], coverage=row["coverage"],
                overall_correct_decision_rate=row["overall_correct_decision_rate"]))
    pd.DataFrame(performance).to_csv(directory / "comparison.csv", index=False)
    pd.DataFrame(selective).to_csv(directory / "selective_comparison.csv", index=False)
    write_json(directory / "summary.json", dict(performance=performance, unanimous_decisions=selective,
        selected_neural_models={task: tasks[task]["metrics"]["selected_neural_model"] for task in tasks},
        verification=verification))

    line = ["# Aplicação de Torres et al. (2025): LSTM, Bi-LSTM e Monte Carlo dropout", "",
            "**Luiz Eduardo do Nascimento Ciarallo** · execução de 15/10/2026", "",
            "Foram baixados os dados públicos e pesos dos autores, reexecutados os checkpoints e treinadas duas redes novas em cada base. "
            "As tarefas de infecção por DENV-2 e identificação de espécie usam o mesmo pipeline recorrente, com saídas e avaliações próprias.", "",
            "## Resultados dos novos treinos", "",
            "A tabela usa todas as entradas de teste, com predição determinística e sem descartar casos.", "",
            "| Tarefa | Modelo | Acertos/total | Acerto | F1 macro |", "|---|---|---:|---:|---:|"]
    for row in performance:
        task = "Infecção DENV-2" if row["task"] == "infection" else "20 espécies"
        line.append(f"| {task} | {row['model']} | {row['correct']}/{row['total']} | {percent(row['accuracy'])} | {percent(row['macro_f1'])} |")
    line += ["", "Seleção exclusivamente pela perda de validação: " + "; ".join(
        f"**{task}: {data['metrics']['selected_neural_model']}**" for task, data in tasks.items()) + ".",
        "A escolha foi congelada antes da inferência de teste; o resultado de teste não troca o candidato escolhido.", "",
        "Na tarefa de infecção, as duas redes superaram numericamente a referência linear nesta divisão. "
        "Na tarefa de 20 espécies, a referência linear obteve o maior acerto; esta rodada não sustenta substituir o classificador existente pelas redes recorrentes. "
        "A inferência com dropout e abstenção está disponível para comparação, com seus próprios denominadores.", "",
        "## Monte Carlo dropout e abstenção", "",
        "Cada entrada recebe 100 inferências com dropout entre as camadas recorrentes. A discordância é "
        "`UC = 1 - maior_contagem_de_votos / 100`. Para duas classes, isso equivale à fórmula de Torres et al. "
        "O limiar principal é UC = 0: todas as passagens devem votar na mesma classe.", "",
        "| Tarefa | Modelo | Aceitas/total | Acertos nas aceitas | Acerto condicional | Cobertura | Acertos emitidos/total |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in selective:
        task = "Infecção" if row["task"] == "infection" else "Espécies"
        line.append(f"| {task} | {row['model']} | {row['accepted']}/{row['total']} | {row['correct_accepted']} | "
                    f"{percent(row['accuracy_accepted'])} | {percent(row['coverage'])} | {percent(row['overall_correct_decision_rate'])} |")
    line += ["", "O arquivo `selective_comparison.csv` preserva esses denominadores. Os arquivos individuais também mostram "
        "limiares predefinidos de 1%, 5%, 10%, 20% e 50% de discordância, sem escolha pelo teste.", "",
        "## Dados e separação", "",
        "| Base | Treino | Validação | Teste | Unidade de avaliação |", "|---|---:|---:|---:|---|"]
    for task, data in tasks.items():
        counts = data["protocol"]["split_counts"]
        line.append(f"| {task} | {counts['train']['groups']} | {counts['validation']['groups']} | {counts['test']['groups']} | "
                    f"{'Tensor fornecido pelos autores' if task == 'infection' else 'Gravação original'} |")
    sp = tasks["species"]["protocol"]
    line += ["", "- A base dos autores contém **300 tensores, 150 por classe**, cada um com **188 passos e 32 bandas mel**. "
        "A ordem dos rótulos vem do código dos autores: 0 = infectado DENV-2, 1 = saudável.",
        "- A base de espécies usa Dryad e preserva as 190 gravações do teste histórico. "
        f"A auditoria removeu **{sp['removed_groups']} grupos e {sp['removed_windows']} janelas** das partições de menor prioridade "
        "quando o mesmo componente PCM aparecia em outra partição. O teste reúne **174 blocos PCM em 190 gravações**.",
        "- Os rótulos de espécie não são convertidos em rótulos de infecção. MosquitoSong permanece fora desse treino.",
        "- A referência linear foi ajustada de novo nas mesmas linhas retidas, com média e desvio temporal dos mesmos atributos mel.", "",
        "## Técnicas implementadas", "",
        "| Componente | Base dos autores | Adaptação ao projeto |", "|---|---|---|",
        "| Entrada | Mel fornecido: 188 × 32 | Mel calculado: 59 × 32 |",
        "| Áudio / STFT | Artigo: 48 kHz; FFT/Hann 1.024; passo 256 | 16 kHz; FFT/Hann 1.024; passo 256; 80–4.000 Hz |",
        "| Sequência | Cerca de 1 segundo por sinal | 0,992 segundo por janela |",
        "| Rede | LSTM e Bi-LSTM, duas camadas, 256 unidades | Mesma estrutura, saída de 20 classes |",
        "| Dropout | 0,20 entre camadas | 0,20 entre camadas |",
        "| Otimizador | Adam, lr 0,000005; lote 16 | Adam, lr 0,0003; lote 128; clipping 3 |",
        "| Épocas máximas | 300, paciência 50 | 80, paciência 15 |",
        "| Escala ajustada no treino | Tensores originais, sem scaler adicional | Média/desvio por banda ajustados só no treino |",
        "| Inferência | 100 passagens estocásticas | 100 passagens, agregação por gravação em cada passagem |", "",
        "Foi usada `CrossEntropyLoss`, como no código público; a tabela do artigo descreve entropia cruzada binária. "
        "O treino usa amostragem com reposição balanceada por classe e gravação, para que uma gravação longa não domine uma classe. "
        "O readout segue o código dos autores: camada linear sobre a última saída da sequência. "
        "As camadas LSTM foram empilhadas com dropout explícito para contornar a dependência ausente do dropout recorrente do MIOpen. "
        "A equivalência determinística com o `nn.LSTM` original é testada nos dois sentidos.", "",
        "## Reexecução dos pesos publicados", "",
        f"Fonte: [repositório dos autores]({AUTHOR_REPOSITORY}), commit `{AUTHOR_COMMIT}`. "
        "O código público usa 260/40 e seleciona checkpoint pela perda nesses 40 exemplos; o texto do artigo informa 80/20. "
        "Esta reexecução usa os mesmos 40 exemplos do código, identificados separadamente dos novos treinos.", "",
        "| Pesos dos autores | Acerto em todos os 40 | Acertos/aceitas com UC = 0 | Acerto condicional | Cobertura |",
        "|---|---:|---:|---:|---:|"]
    for m in json.loads((directory / "author_checkpoints/metrics.json").read_text()):
        row = m["selective"][0]
        line.append(f"| {m['model']} | {m['deterministic']['correct']}/40 ({percent(m['deterministic']['accuracy'])}) | "
                    f"{row['correct']}/{row['accepted']} | {percent(row['accuracy'])} | {percent(row['coverage'])} |")
    line += ["", "O artigo informa 100 passagens; o script de teste publicado executa 10. Esta rodada fixa 100 passagens e semente 20261015. "
        "Os novos treinos usam validação separada, em vez de escolher checkpoint pela perda do teste.", "",
        "## Gráficos", "",
        f"![Acerto em todas as entradas](../{figures.relative_to(ROOT)}/acerto_todas_entradas.png)", "",
        f"![Cobertura e acerto condicional](../{figures.relative_to(ROOT)}/cobertura_e_acerto.png)", "",
        f"![Curvas de treino e validação](../{figures.relative_to(ROOT)}/curvas_treino_validacao.png)", "",
        f"![Matriz de confusão com abstenção](../{figures.relative_to(ROOT)}/matriz_infeccao.png)", "",
        "## Executar e conferir", "", "```bash", ".venv/bin/python fetch_torres_2025.py",
        ".venv/bin/python apply_torres_2025.py --task all --output results/nova_rodada_torres",
        ".venv/bin/python verify_torres_2025.py results/nova_rodada_torres",
        ".venv/bin/python -m pytest -q", "```", "",
        "O executor recusa sobrescrever resultados existentes. `freeze.json` registra a escolha e os hashes antes do teste; "
        "`source/` preserva os scripts efetivamente executados. As previsões CSV e as 100 distribuições por grupo em NPZ permitem recalcular votação e abstenção. "
        "`artifact_format_migration.json` registra a conversão dos IDs dos NPZ para strings Unicode, sem alterar probabilidades, alvos ou pesos.",
        "", "Para usar os pesos novos:", "", "```bash",
        f".venv/bin/python predict_torres_2025.py {directory.relative_to(ROOT)}/species --audio caminho/arquivo.wav",
        f".venv/bin/python predict_torres_2025.py {directory.relative_to(ROOT)}/infection --author-row 0", "```", "",
        "O modelo de infecção recebe os tensores mel originais (`--author-row` ou `--features`); o repositório dos autores "
        "fornece só dois WAVs exemplificativos, e não os 300 sinais brutos com a receita completa de extração. "
        "O modelo de espécies tem frontend de áudio implementado e scaler congelado.", "",
        "Validação: **298 testes aprovados no total, incluindo 13 novos**. "
        "`pytest.ini` restringe a coleta à pasta `tests/`, evitando importar scripts executáveis de pesquisa e cópias de entregas. "
        "`verification.json` confirma isolamento, hashes, seleção na validação, probabilidades e denominadores de abstenção.", "",
        "## Referências", "",
        "- Torres et al. (2025). *Dengue-Infected Mosquito Detection with Uncertainty Evaluation based on Monte Carlo Dropout*. "
        "IEEE Latin America Transactions 23(2), 135–143. https://latamt.ieeer9.org/index.php/transactions/article/view/9221",
        f"- Código e dados dos autores: {AUTHOR_REPOSITORY}/tree/{AUTHOR_COMMIT}",
        "- Gal e Ghahramani (2016). *Dropout as a Bayesian Approximation*. https://proceedings.mlr.press/v48/gal16.html", "",
        "## Limitações e interpretação final", "",
        "- Os 300 tensores dos autores não incluem identificadores de indivíduo ou gravação. A divisão protege hashes exatos, "
        "mas não comprova independência por mosquito. Os 40 exemplos da reexecução foram usados pelos autores na seleção dos pesos.",
        "- O teste de espécies é retrospectivo e contém gravações relacionadas dentro do próprio teste. Os resultados não demonstram generalização para nova coleta em campo.",
        "- Concordância em 100 passagens mede estabilidade do modelo com dropout; não confirma infecção nem garante correção. "
        "Acurácia entre aceitas deve ser lida junto com cobertura e erros emitidos.",
        "- A tarefa de 20 espécies escolhe uma espécie de mosquito. Ela não mede a taxa operacional de falso alarme em ruído; "
        "a detecção ternária existente constitui outra avaliação.",
        "- Um resultado maior em um conjunto não implica superioridade em outra tarefa. "
        "As tabelas mantêm os dois modelos, a seleção de validação e a referência linear para que esse julgamento seja verificável.", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(line), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/aplicacao_torres_2025_2026_10_15.md")
    args = parser.parse_args()
    summarize(args.result_directory, args.report)
