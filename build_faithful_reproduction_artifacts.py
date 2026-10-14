"""Build an auditable comparison of actual author-code executions and diagnostics."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/faithful_reproduction_2026_10_03"
NAME = "reproducao_estudos_originais_2026_10_03"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(x):
    return "—" if x is None or pd.isna(x) else f"{100*x:.2f}%"


def main():
    dr = load(OUT/"dr_biol/metrics.json")
    ms = load(OUT/"mosquitosong/metrics.json")
    ext = load(OUT/"mosquitosong/external_metrics.json")
    executions = {n:load(OUT/n/"execution.json") for n in ["dr_biol","mosquitosong"]}
    assert all(e["status"]=="complete" and not e["training_rerun"] for e in executions.values())
    rows = []
    binary_rows = []
    reference = {"DR_BioL_CNN":(.85345,.9732,.9002),"DAT_CNN":(.79583,.9607,.8481)}
    for name in reference:
        part = [r for r in dr if r["model"]==name]
        row = dict(study="DR-BioL",model=name,task="8 species",test_examples=10256,
                   repeated_evaluation_draws=len(part),training_rerun=False,
                   published_accuracy=reference[name][0], published_roc_auc_micro=reference[name][1],
                   published_average_precision_micro=reference[name][2])
        for key in ["accuracy","roc_auc_micro","average_precision_micro","macro_f1"]:
            row[key] = float(np.mean([r[key] for r in part]))
            row[key+"_dropout_sd"] = float(np.std([r[key] for r in part],ddof=1))
        row["accuracy_delta_pp"] = 100*(row["accuracy"]-row["published_accuracy"])
        rows.append(row)
        binary_rows.append(dict(model=name,task="derived Ae aegypti vs 7 species, original argmax",
                                **{k:float(np.mean([r["binary"][k] for r in part])) for k in ["precision","recall","specificity"]}))
    all_y, all_p = [], []
    for fold in range(1,11):
        with np.load(OUT/"mosquitosong"/("fold%d_gain3.npz"%fold)) as a:
            all_y.extend(a["y_true"]); all_p.extend(a["y_pred"])
    y,p = np.asarray(all_y),np.asarray(all_p)
    row = dict(study="MosquitoSong+",model="MosquitoSong+ original notebook",task="4 species x 2 sexes",
               test_examples=len(y),unique_base_epochs_before_6_simulations=6000,
               repeated_evaluation_draws=10,training_rerun=False,
               published_accuracy=None,published_accuracy_exp1=.918,published_accuracy_exp3=.933,
               accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,average="macro")),
               notebook_mean_macro_f1=float(np.mean([r["macro_f1"] for r in ms if r["gain_index"]==3])),
               exact_paper_experiment_mapping="Not specified in release; Exp3 additionally uses data/noise absent from release.")
    rows.append(row)
    yp,pp = y<2,p<2
    binary_rows.append(dict(model="MosquitoSong+ original notebook",task="derived Ae aegypti F or M vs remaining 6 labels",
                            precision=float(precision_score(yp,pp)),recall=float(recall_score(yp,pp)),
                            specificity=float(recall_score(~yp,~pp))))
    table = pd.DataFrame(rows)
    table.to_csv(OUT/"paper_comparison.csv",index=False)
    pd.DataFrame(binary_rows).to_csv(OUT/"native_binary_metrics.csv",index=False)

    # Paired, frozen retrospective comparison on exactly the previous 641 intervals.
    prior_dir = ROOT/"results/cross_domain_strategies_2026_10_03"
    prior = pd.read_csv(prior_dir/"fresh_metrics.csv")
    prior = prior[prior.scope.eq("all_fresh")].copy()
    drive = pd.read_csv(prior_dir/"drive_positive_control.csv").set_index("model")
    comparison = []
    for _,r in prior.iterrows():
        comparison.append(dict(model=r.model,precision=r.aedes_precision,recall=r.aedes_recall,
                         specificity=r.specificity,balanced_accuracy=r.balanced_accuracy,
                         drive_tp=int(drive.loc[r.model,"true_positive"]),drive_n=107,
                         original_author_model=False,task="binary, unchanged frozen threshold"))
    ensemble = [r for r in ext if r["fold"]=="all_folds_mean_derived"]
    er = next(r for r in ensemble if r["scope"]=="external32_retrospective")
    ed = next(r for r in ensemble if r["scope"]=="drive_positive_only")
    comparison.append(dict(model="MosquitoSong+ ensemble derivado (10 pesos originais)",precision=er["precision"],
                       recall=er["recall"],specificity=er["specificity"],balanced_accuracy=er["balanced_accuracy"],
                       drive_tp=ed["tp"],drive_n=ed["n"],original_author_model=True,
                       task="binary derived from 8 labels; fixed 3 subwindow / 10 fold averages"))
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(OUT/"external_same_intervals_comparison.csv",index=False)
    pd.DataFrame(ext).to_csv(OUT/"mosquitosong/external_metrics.csv",index=False)

    original_csv = ROOT/"tmp/reproductions_2026_10_01/fernandes/results/02_binary/binary.csv"
    fdata = pd.read_csv(original_csv)
    fernandes = dict(status="author_results_csv_recomputed_only",new_training_or_inference=False,
                     available_trained_weights=False,available_manually_curated_dataset=False,
                     source="https://github.com/ComputerNetworks-UFRGS/CNN_Aedes_Aegypti_Classifier",
                     csv_sha256=sha(original_csv),folds=len(fdata),
                     means=fdata.mean().to_dict(),std=fdata.std(ddof=0).to_dict(),
                     close_to_published_within_0_01_percentage_point=True,
                     exact_published_rounding_match=False,
                     published_means=dict(accuracy=.9765,precision=.9063,recall=.8849,f1_score=.8929),
                     deltas_percentage_points={k:100*(fdata[k].mean()-v) for k,v in
                         dict(accuracy=.9765,precision=.9063,recall=.8849,f1_score=.8929).items()},
                     missing_artifacts="Manually trimmed/resampled waveforms or exact feature NPZ; trained checkpoint(s). Public repository and releases checked.")
    (OUT/"fernandes_reference_recalculation.json").write_text(json.dumps(fernandes,indent=2))
    audit = load(OUT/"mosquitosong/split_audit.json")
    overlap = pd.DataFrame([dict(fold=r["fold"],shared_train_test_recordings=r["shared_recordings"]["train_test"],
                        overlapping_train_test_window_pairs=r["cross_split_overlapping_window_pairs"]["train_test"],
                        shared_train_val_recordings=r["shared_recordings"]["train_val"],
                        test_epochs=r["windows"]["test"]) for r in audit["folds"]])
    overlap.to_csv(OUT/"mosquitosong/split_audit.csv",index=False)

    assets=ROOT/"reports/assets"
    assets.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    ax=axes[0]
    x=np.arange(2)
    published=np.array([reference[k][0] for k in reference])*100
    reproduced=table.iloc[:2].accuracy.to_numpy()*100
    ax.bar(x-.17,published,width=.34,color="#8897a7",label="Artigo (média de treinos)")
    ax.bar(x+.17,reproduced,width=.34,color="#228b83",label="Pesos originais (5 avaliações)")
    for i,v in enumerate(reproduced):ax.text(i+.17,v+.6,f"{v:.2f}",ha="center",fontsize=9)
    ax.set_xticks(x,["DR-BioL","DAT"])
    ax.set_ylim(0,100);ax.set_ylabel("Acurácia (%)");ax.set_title("Mesmo teste oficial: 8 espécies");ax.legend(fontsize=8)
    ax=axes[1]
    ax.bar([0,1,2],[91.8,92.7777777778,93.3],color=["#8897a7","#228b83","#d69a47"])
    ax.set_xticks([0,1,2],["Artigo\nExp. 1","Notebook\ndisponível","Artigo\nExp. 3"])
    for i,v in enumerate([91.8,92.7777777778,93.3]):ax.text(i,v+.7,f"{v:.2f}",ha="center",fontsize=9)
    ax.set_ylim(0,100);ax.set_title("MosquitoSong+: 4 espécies × 2 sexos")
    fig.suptitle("Avaliações originais reproduzidas; tarefas distintas entre painéis",fontsize=12)
    fig.tight_layout();fig.savefig(assets/"faithful_author_accuracy.png",dpi=170);plt.close(fig)
    selected_order=["ridge_contrast_swarm","svm_multipitch","trees_multipitch",
                    "prior_dryad+humbug__svm_rbf","MosquitoSong+ ensemble derivado (10 pesos originais)"]
    selected=comparison.set_index("model").loc[selected_order]
    labels=["CNN contraste + mistura","SVM múltiplos picos","ExtraTrees múltiplos picos","SVM Dryad + HumBug","Pesos MosquitoSong+\nensemble derivado"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));x=np.arange(len(selected))
    axes[0].barh(x-.16,selected.recall*100,height=.32,color="#228b83",label="Sensibilidade a Aedes")
    axes[0].barh(x+.16,selected.specificity*100,height=.32,color="#8897a7",label="Especificidade")
    axes[0].set_yticks(x,labels);axes[0].invert_yaxis();axes[0].set_xlim(0,110);axes[0].legend(fontsize=8)
    axes[0].set_title("Mesmos 641 trechos externos (diagnóstico)");axes[0].set_xlabel("%")
    axes[1].barh(x,selected.drive_tp/107*100,color="#d69a47")
    axes[1].set_yticks(x,[]);axes[1].invert_yaxis();axes[1].set_xlim(0,100)
    for i,v in enumerate(selected.drive_tp):axes[1].text(v/107*100+1,i,f"{v}/107",va="center",fontsize=9)
    axes[1].set_title("Drive: controle só positivo");axes[1].set_xlabel("Trechos reconhecidos como Aedes (%)")
    fig.suptitle("Esta comparação externa usa uma agregação adicional declarada para MosquitoSong+",fontsize=11)
    fig.tight_layout();fig.savefig(assets/"faithful_external_comparison.png",dpi=170);plt.close(fig)

    binary_table="| Modelo original | Precisão Aedes | Sensibilidade Aedes | Especificidade |\n|---|---:|---:|---:|\n"+"\n".join(
        f"| {r['model']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['specificity'])} |" for r in binary_rows)
    external_table="| Modelo congelado | Precisão | Sensibilidade Aedes | Especificidade | Acerto balanceado | Drive |\n|---|---:|---:|---:|---:|---:|\n"+"\n".join(
        f"| {r.model} | {pct(r.precision)} | {pct(r.recall)} | {pct(r.specificity)} | {pct(r.balanced_accuracy)} | {r.drive_tp}/107 |" for _,r in comparison.iterrows())
    folds=[r for r in ext if isinstance(r["fold"],int) and r["protocol"]=="common_992ms_three_subwindows"]
    drive_folds=[r["tp"] for r in folds if r["scope"]=="drive_positive_only"]
    external_folds=[r["recall"] for r in folds if r["scope"]=="external32_retrospective"]
    report=f"""# Reexecução dos estudos com os artefatos originais — 03/10/2026

Reexecutei a **avaliação** de DR-BioL/DAT e o notebook oficial MosquitoSong+, com as arquiteturas, pesos e entradas disponibilizados pelos autores. Os resultados internos ficaram próximos dos artigos. **Não refiz os treinamentos dos artigos, suas dez repetições de treino ou todos os experimentos.** A avaliação original e os diagnósticos externos são apresentados separadamente. Não há reprodução integral de todos os estudos, nem equivalência bit a bit de ambientes sem versões publicadas.

## Comparação com os resultados publicados

| Estudo e tarefa | Publicado | Reexecução realizada | Diferença |
|---|---:|---:|---:|
| DR-BioL, 8 espécies, teste oficial | 85,345% | {pct(rows[0]['accuracy'])} | {rows[0]['accuracy_delta_pp']:+.3f} p.p. |
| DAT, mesmas 8 espécies e teste | 79,583% | {pct(rows[1]['accuracy'])} | {rows[1]['accuracy_delta_pp']:+.3f} p.p. |
| MosquitoSong+, espécie e sexo, Exp. 1 / Exp. 3 | 91,8% / 93,3% | {pct(row['accuracy'])}, notebook disponível | +0,978 / −0,522 p.p. |
| Fernandes, binário | 97,65%; sensibilidade 88,49% | CSV dos autores recalculado: 97,654%; 88,496% | Conferência numérica; sem novo experimento |

O DR-BioL/DAT foi comparado à Tabela 4 do [preprint v1](https://arxiv.org/html/2510.00346v1), usando o [código original](https://github.com/Yuanbo2020/DR-BioL) e o [dataset oficial](https://zenodo.org/records/17287037). AUC micro: DR-BioL {rows[0]['roc_auc_micro']:.5f} contra 0,9732; DAT {rows[1]['roc_auc_micro']:.5f} contra 0,9607. AP micro: {rows[0]['average_precision_micro']:.5f} contra 0,9002; {rows[1]['average_precision_micro']:.5f} contra 0,8481.

No [artigo MosquitoSong+](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121), 91,8% corresponde ao experimento 1 e 93,3% ao experimento 3. O [repositório disponibilizado](https://github.com/akaraspt/mosquitosongp) não identifica de forma inequívoca qual experimento corresponde ao conjunto dos dez H5. O experimento 3 também usa dados e ruídos adicionais que não constam nesse pacote. Por isso, não afirmo ter reproduzido exatamente a linha de 93,3% do artigo. A comparação mostra as duas referências, sem escolher a mais conveniente. O macro F1 combinado reproduzido foi {row['macro_f1']:.5f}; a média entre folds do notebook foi {row['notebook_mean_macro_f1']:.5f} antes do arredondamento.

![Resultados internos](assets/faithful_author_accuracy.png)

## O que permaneceu fiel aos artefatos

**DR-BioL/DAT:** todos os 10.256 exemplos do teste no pickle oficial; Mel pré-calculado 200 × 64; oito classes; batch 64; pesos originais carregados com `strict=True`; saída por argmax. Preservei `MC_dropout=True`, que mantém dropout na última camada mesmo com `eval()`. Fiz cinco avaliações com sementes 0–4 do mesmo checkpoint, não cinco treinamentos. Desvio entre essas avaliações: {rows[0]['accuracy_dropout_sd']*100:.3f} p.p. para DR-BioL e {rows[1]['accuracy_dropout_sd']*100:.3f} p.p. para DAT. O arquivo ZIP de 1,4 GB confere com o MD5 oficial `4d3fe3742d104a022bbea013b4bd290e`.

O `Evaluation.py` foi executado com seu cálculo e gráficos originais. Para DAT, somente a escolha da variável `model_name` foi trocada para a alternativa indicada no próprio arquivo. A compatibilidade de sklearn exigiu aceitar o argumento posicional de `compute_class_weight` como argumento nomeado; a fórmula permanece a da biblioteca. O código e os pesos originais no checkout não foram alterados. O ambiente local foi PyTorch {executions['dr_biol']['torch']}, Python {executions['dr_biol']['python']} e GPU {executions['dr_biol']['device']}; o repositório não fixa todas as versões do ambiente original.

**MosquitoSong+:** executei todas as células de código originais, sem editar seu texto; dez H5; 1.324 WAVs e 12 ruídos; resampling por librosa; 300 ms com 150 ms de passo e jitter original; partições e limites originais; simulação de ganho, atraso, dois microfones e cancelamento; três ganhos e suas combinações; relatórios e CSVs oficiais. Instrumentei somente o registro dos argumentos de `classification_report`, devolvendo seu resultado original. Foram 600 epochs base por fold, seis versões simuladas de cada epoch e 36.000 previsões combinadas entre folds. Elas não representam 36.000 gravações independentes.

TensorFlow **2.4.0** foi instalado em um ambiente separado, com NumPy 1.19.5, librosa 0.8.1, h5py 2.10.0 e pandas 1.3.5. O README sugere Python 3.7; usei 3.8.20 e execução em CPU. O README não fixa as versões das demais bibliotecas. O arquivo completo de versões e os hashes estão nos resultados. Preservar código, pesos e dados permite reexecutar esta avaliação; não elimina a diferença do ambiente nem completa os artefatos ausentes.

## Resposta binária derivada dos modelos originais

Sem novo treino ou limiar ajustado: DR-BioL/DAT é Aedes quando o argmax original é a classe 2; MosquitoSong+ é Aedes quando o argmax é a classe 0 ou 1, feminino ou masculino. As métricas abaixo pertencem **aos respectivos testes nativos**, não ao Arduino nem ao Drive. As classes negativas são outras espécies de mosquito; estes números não avaliam a ausência de mosquito.

{binary_table}

Isso mostra que um modelo original consegue identificar muitos Aedes **no teste distribuído pelos autores**. Não demonstra a mesma sensibilidade num sensor, ambiente, enxame ou gravação diferente.

## Auditoria das partições

No notebook MosquitoSong+, executei a função original `split_data` e rastreei cada janela de volta ao WAV e ao intervalo de amostras que a originaram. Em todos os folds há gravações presentes tanto nos arrays chamados treino quanto nos de teste: **{int(overlap.shared_train_test_recordings.min())}–{int(overlap.shared_train_test_recordings.max())} gravações por fold**, com **{int(overlap.overlapping_train_test_window_pairs.min())}–{int(overlap.overlapping_train_test_window_pairs.max())} pares de janelas com amostras realmente sobrepostas** entre essas partições. Também há sobreposição com validação. Os dez folds completos estão em `mosquitosong/split_audit.csv`.

É evidência de dependência nas partições construídas pelo notebook público, não uma acusação de fraude nem a prova de quanto a acurácia foi inflada. Como os H5 não trazem um manifesto de quais gravações foram usadas no treino, não posso confirmar retrospectivamente se cada peso foi treinado exatamente com esses arrays. Também não há identificação completa dos mosquitos individuais. Uma validação rigorosa deve separar gravações e, quando possível, indivíduos/sessões **antes** de criar janelas.

No DR-BioL, a auditoria por hash dos vetores Mel não encontrou entradas idênticas entre treino, validação e teste. O pickle não fornece identificadores das gravações/sessões originais. Logo, **zero duplicatas exatas não prova independência entre gravações**. A avaliação executada é a mistura D1–D4 do teste oficial; não reproduzi o experimento separado de domínio nunca visto da Tabela 1.

## Comparação externa com os nossos modelos

Mantive os 32 arquivos e 641 intervalos de 0,992 s do teste anterior, sendo 493 positivos e 148 negativos pelo rótulo do arquivo. **É uma comparação retrospectiva em arquivos já avaliados**, sem novo treino, escolha por resultado ou ajuste de limiar. Não apresento este conjunto como recém-descoberto ou novo teste cego da nossa linha de pesquisa. Não há anotação temporal de presença de voo em cada intervalo, nem garantia de novos indivíduos por identificadores; são limites desta comparação.

O modelo MosquitoSong+ requer 300 ms. Declarei antes da avaliação a média dos oito escores de três sub-janelas reais de 300 ms em cada intervalo comum, e um ensemble dos dez pesos sem selecionar o melhor fold. Essa agregação é **uma extensão nossa**. Também mantive resultados separados no protocolo de janelas de 300 ms, com jitter/passo originais. Em gravações mono reais não apliquei ganho artificial nem inventei cancelamento físico entre dois microfones; este diagnóstico não reproduz o sistema de aquisição de dois microfones do estudo.

{external_table}

Os folds originais, individualmente, ficaram entre {pct(min(external_folds))} e {pct(max(external_folds))} de sensibilidade nos mesmos intervalos externos; no Drive, entre {min(drive_folds)} e {max(drive_folds)} acertos de 107. O ensemble derivado reconheceu **{ed['tp']}/107** no Drive. Há 105 intervalos válidos e dois rejeitados pelo critério de qualidade comum. O Drive foi rotulado como enxame de Aedes pelo usuário e tem somente a classe positiva: permite medir a proporção reconhecida sob essa referência, **não** especificidade, precisão informativa, taxa de falsos alarmes ou confiabilidade geral. Os denominadores de 300 ms são diferentes e constam nos arquivos; não os misturei com os 107 intervalos.

![Diagnóstico externo](assets/faithful_external_comparison.png)

A diferença de aquisição e pré-processamento limita o diagnóstico externo dos pesos MosquitoSong+. Ainda assim, os resultados não sustentam seu uso direto nos nossos áudios mono. Os nossos candidatos tampouco alcançaram sensibilidade de 60% nesses mesmos intervalos. **Não há evidência suficiente para liberar o modelo no cenário do usuário.**

Não testei DR-BioL no Drive sob a etiqueta de reprodução exata: o repositório entrega Mel pronto e não entrega o extrator de WAV completo que produziu esse pickle. Escolher parâmetros aproximados e chamar o resultado de original repetiria o problema das adaptações anteriores. Também não comparei diretamente a acurácia multiclasses nativa de 85% com a sensibilidade binária externa dos nossos modelos.

## Estudos que não puderam ser reproduzidos integralmente

**Fernandes:** o [artigo](https://arxiv.org/pdf/2008.09024) descreve cortes manuais de ruído/ausência de mosquito, conversão e resampling com SoX. O código usa janela de **1,248 s**, espectrograma transposto/achatado e reorganizado, duas saídas sigmoid, RMSprop e dez épocas; nossas adaptações anteriores diferem desses detalhes. O checkout público e seus releases não forneceram o pacote de pesos nem a base curada/NPZ. Recalculei o CSV original de dez folds: as médias ficam a menos de 0,01 ponto percentual das publicadas, com pequenas diferenças na última casa decimal. Isso não é novo treino nem teste. Não substituí a base curada por áudios brutos e não chamei isso de réplica exata.

**Altayeb/TinyML:** baixei o [repositório dos autores](https://github.com/Mjrovai/wingbeat-mosquito-tinyml), que contém a base, o artigo e notebooks. O notebook de treinamento depende de vetores exportados de um projeto Edge Impulse; os endpoints de X/Y retornaram HTTP 401 em acesso anônimo. Não obtive um export público versionado com os vetores/partições/pesos necessários para repetir a avaliação específica do estudo. Não gerei modelos, bibliotecas ou firmware para Arduino.

As estratégias anteriores continuam rotuladas como adaptações. A reprodução da avaliação original não deve ser confundida com refazer todos os treinos, todos os estudos ou demonstrar generalização.

## Artefatos e execução

Resultados em `results/faithful_reproduction_2026_10_03/`: tabela artigo/reexecução, métricas binárias nativas, previsões originais por fold/semente, seis CSVs oficiais, notebook original executado, auditorias, protocolos, versões e hashes. O notebook deste relatório foi executado e o HTML gerado. O áudio privado não foi incorporado aos artefatos públicos. Firmware, instaladores e modelos anteriores permanecem intactos.

Com os checkouts fixados nas revisões dos manifestos, a base oficial baixada e os WAVs/H5 nas pastas descritas pelo README original:

```bash
.venv/bin/python fetch_faithful_dr_biol.py
.venv/bin/python run_author_dr_biol.py
TF_CPP_MIN_LOG_LEVEL=2 MPLBACKEND=Agg tmp/faithful_reproduction_2026_10_03/env_tf24/bin/python run_author_mosquitosong.py
tmp/faithful_reproduction_2026_10_03/env_tf24/bin/python audit_author_mosquitosong.py
tmp/faithful_reproduction_2026_10_03/env_tf24/bin/python evaluate_author_mosquitosong_external.py
.venv/bin/python build_faithful_reproduction_artifacts.py
```

A criação do ambiente isolado e o preparo dos 641/107 intervalos estão documentados em `results/faithful_reproduction_2026_10_03/REPRODUCING.md`. Os comandos de avaliação não treinam, não escolhem limiares e não compilam Arduino.
"""
    report_path=ROOT/"reports"/(NAME+".md")
    report_path.write_text(report)

    notebook=nbf.v4.new_notebook()
    notebook.cells=[nbf.v4.new_markdown_cell(report.replace("assets/","reports/assets/"))]
    notebook.cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json, numpy as np, pandas as pd
from IPython.display import display, Image
out = Path('results/faithful_reproduction_2026_10_03')
display(pd.read_csv(out/'paper_comparison.csv'))
display(pd.read_csv(out/'native_binary_metrics.csv'))
display(Image(filename='reports/assets/faithful_author_accuracy.png'))
"""))
    notebook.cells.append(nbf.v4.new_code_cell("""display(pd.read_csv(out/'external_same_intervals_comparison.csv'))
display(Image(filename='reports/assets/faithful_external_comparison.png'))
display(pd.read_csv(out/'mosquitosong/split_audit.csv'))
"""))
    notebook.cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import accuracy_score, f1_score
labels, predicted = [], []
for fold in range(1,11):
    data = np.load(out/'mosquitosong'/f'fold{fold}_gain3.npz')
    labels.extend(data['y_true']); predicted.extend(data['y_pred'])
print('Recalculado dos resultados originais: n=', len(labels),
      'accuracy=', accuracy_score(labels,predicted),
      'macro_f1=', f1_score(labels,predicted,average='macro'))
for study in ['dr_biol','mosquitosong']:
    e = json.loads((out/study/'execution.json').read_text())
    print(study, e['status'], 'novo treino:', e['training_rerun'], 'commit:',e['source_commit'])
print(json.loads((out/'dr_biol/data_audit.json').read_text())['identical_feature_overlap'])
"""))
    notebook.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
    path=ROOT/(NAME+".ipynb")
    NotebookClient(notebook,timeout=120,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
    nbf.write(notebook,path)
    html,_=HTMLExporter().from_notebook_node(notebook)
    (ROOT/(NAME+".html")).write_text(html)
    source_paths=[ROOT/n for n in ["fetch_faithful_dr_biol.py","run_author_dr_biol.py","run_author_mosquitosong.py",
                   "audit_author_mosquitosong.py","evaluate_author_mosquitosong_external.py","build_faithful_reproduction_artifacts.py"]]
    verification=dict(author_repositories_unmodified=True,author_weights_unmodified=True,
                      training_rerun=False,arduino_compiled=False,private_raw_audio_exported=False,
                      notebook_code_cells=3,notebook_errors=0,
                      source_sha256={p.name:sha(p) for p in source_paths},
                      artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in [report_path,path,ROOT/(NAME+".html")]})
    for relative,expected in load(prior_dir/"device_files_sha256.json").items():
        assert sha(ROOT/relative)==expected,relative
    for relative,expected in load(prior_dir/"original_weights_sha256.json").items():
        assert sha(ROOT/relative)==expected,relative
    verification["device_files_and_previous_models_sha256_match"]=True
    (OUT/"artifact_verification.json").write_text(json.dumps(verification,indent=2))
    decision=dict(evaluation_stage_completed=True,all_studies_fully_reproduced=False,
                  original_training_repeated=False,can_release_to_user_device=False,
                  arduino_compiled=False,previous_models_modified=False,
                  DR_BioL_external_validation="Not performed: complete original WAV-to-pickle frontend absent",
                  MosquitoSong_user_scenario="Failed frozen mono diagnostics; two-microphone acquisition not reproduced",
                  generalization_proven=False,
                  reasons=["Native results do not test the user's acquisition setup",
                           "Published notebook partitions share recordings and waveform samples",
                           "Original training manifests and complete environments are not supplied",
                           "External same-interval Aedes recall below 60%; external labels are per recording"],
                  evidence={"original_data_checksum_verified":True,"source_code_and_weights_verified":True,
                            "no_threshold_or_fold_selection_by_test":True,
                            "external_test_retrospective_not_new_blind_acquisition":True,
                            "private_control_positive_only":True})
    (OUT/"reliability_decision.json").write_text(json.dumps(decision,indent=2))
    print(table[["model","accuracy"]].to_string(index=False))
    print("created",report_path,path,flush=True)


if __name__ == "__main__":
    main()
