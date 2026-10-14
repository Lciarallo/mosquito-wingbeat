"""Build an executed, read-only notebook and the honest comparison report."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from train_cross_domain_strategies import ROOT, OUT
from train_reproduction import sha


def main():
    validation = pd.DataFrame(json.loads((OUT / 'validation_search.json').read_text()))
    selection = json.loads((OUT / 'selection.json').read_text())
    decision = json.loads((OUT / 'reliability_decision.json').read_text())
    fresh = pd.read_csv(OUT / 'fresh_metrics.csv'); fresh = fresh[fresh.scope.eq('all_fresh')]
    private = pd.read_csv(OUT / 'drive_positive_control.csv')
    historical = pd.read_csv(OUT / 'historical_metrics.csv')
    intervals = json.loads((OUT / 'confidence_intervals.json').read_text())
    comparisons = json.loads((OUT / 'paired_comparisons.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    audit = pd.read_csv(OUT / 'novelty_audit.csv')
    files = pd.read_csv(OUT / 'file_metrics.csv')
    selected = selection['selected_model']; row = fresh[fresh.model.eq(selected)].iloc[0]
    control = fresh[fresh.model.eq('ridge_control')].iloc[0]
    drive = private[private.model.eq(selected)].iloc[0]
    table = fresh[['model', 'selected', 'files', 'aedes_precision', 'aedes_recall', 'aedes_f1',
                   'specificity', 'macro_f1', 'balanced_accuracy', 'roc_auc']].copy()
    joined = table.merge(private[['model', 'true_positive', 'known_windows']], on='model')
    joined.to_csv(OUT / 'comparison.csv', index=False)
    assets = ROOT / 'reports/assets'
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ix = np.arange(len(joined)); width = .25
    for offset, metric, title in [(-1, 'aedes_precision', 'Precisão'), (0, 'aedes_recall', 'Sensibilidade'), (1, 'specificity', 'Especificidade')]:
        ax.bar(ix + offset * width, joined[metric].fillna(0) * 100, width, label=title)
    ax.axhline(60, color='black', ls='--', lw=1); ax.set_ylim(0, 105)
    ax.set_ylabel('% — teste acústico inédito'); ax.set_xticks(ix, joined.model.str.replace('prior_', '', regex=False), rotation=30, ha='right')
    ax.legend(); ax.grid(axis='y', alpha=.2); fig.tight_layout()
    fig.savefig(assets / 'cross_domain_strategy_comparison.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    matrix = np.asarray(json.loads(row.confusion.replace("'", '"'))) if isinstance(row.confusion, str) else np.asarray(row.confusion)
    axes[0].imshow(matrix, cmap='Blues'); axes[0].set_xticks([0, 1, 2], ['Não Aedes', 'Aedes', 'Inválido'])
    axes[0].set_yticks([0, 1], ['Não Aedes', 'Aedes']); axes[0].set_title(selected + ': teste novo')
    for i in range(2):
        for j in range(3): axes[0].text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i,j] > matrix.max()/2 else 'black')
    for name in ['ridge_control', 'ridge_contrast', 'ridge_contrast_swarm']:
        history = pd.DataFrame(json.loads((OUT / (name + '_history.json')).read_text()))
        axes[1].plot(history.epoch, history.early_score, marker='.', label=name)
    axes[1].set(xlabel='Época', ylabel='Menor AUC ponderada entre fontes', title='Validação, sem consulta ao teste'); axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(assets / 'cross_domain_strategy_confusion.png', dpi=170); plt.close(fig)
    cirows = [dict(model=name, metric=metric, **bounds) for name, info in intervals.items() for metric, bounds in info['metrics'].items()]
    pd.DataFrame(cirows).to_csv(OUT / 'confidence_intervals.csv', index=False)
    paired = [dict(baseline=name, metric=metric, **bounds) for name, data in comparisons.items() for metric, bounds in data.items()]
    pd.DataFrame(paired).to_csv(OUT / 'paired_comparisons.csv', index=False)
    taxonomy = fresh.iloc[0]  # counts same for every model
    best_drive = private.sort_values('aedes_recall', ascending=False).iloc[0]
    conclusion = ('Os critérios desta amostra foram atingidos; isso não confirma confiabilidade na caixa/Arduino, onde faltam negativos independentes.'
                  if decision['reliable'] else 'Nenhum resultado autoriza declarar este detector confiável. O candidato selecionado falhou nos critérios registrados abaixo.')
    report = f'''# Contraste entre bases, múltiplas frequências e enxames simulados — 03/10/2026

{conclusion}

O candidato escolhido antes do teste foi **{selected}**. No teste novo: precisão **{100*row.aedes_precision:.1f}%**, sensibilidade **{100*row.aedes_recall:.1f}%**, especificidade **{100*row.specificity:.1f}%**, macro F1 **{100*row.macro_f1:.1f}%**. No Drive: **{int(drive.true_positive)}/{int(drive.known_windows)}** janelas detectadas. A rede de controle detectou **{100*control.aedes_recall:.1f}%** dos Aedes no mesmo teste. Consulte a comparação pareada e seu intervalo antes de atribuir melhora às estratégias.

## Estratégias implementadas

1. Representação espectral preservando várias linhas: STFT em quadros reais, 2048 amostras/128 de passo a 8 kHz, banda 200–3500 Hz. Contrastes locais em frequência e tempo reduzem componentes suaves; não são prova de invariância a microfone/ruído. A representação é redimensionada; o áudio não é repetido nem preenchido.
2. Contraste supervisionado entre a mesma espécie/táxon em Dryad e HumBug, exigindo grupos de gravação diferentes. O amostrador garante Aedes em ambas as fontes e pares de quatro outros táxons comuns em cada lote. Não aproxima todas as espécies negativas como se fossem uma espécie só. Perda auxiliar de espécie e alinhamento MMD RBF condicionado ao táxon. É uma adaptação inspirada em DR-BioL, não reprodução integral do artigo.
3. Misturas de enxame apenas no treino: até 50% de superposições de gravações da mesma espécie e grupos diferentes, com variação de ganho e coloração. A mistura conserva o rótulo de espécie; não mistura negativos de espécies diferentes e lhes atribui o rótulo do primeiro mosquito. São simulações, não novos indivíduos e não prova de equivalência a enxame real.
4. Trinta e quatro descritores com quatro picos locais e três candidatos de suporte harmônico por quadro, variabilidade temporal, entropia e contagem de picos. SVM e ExtraTrees foram treinados nesses descritores para comparar ML clássico com as redes. Os picos podem mudar de ordem entre quadros; não são trajetórias identificadas de cada indivíduo ou estimativas validadas de fundamental.

Não foram inventadas temperaturas, sexos ou identidades biológicas ausentes. Relações harmônicas não são declaradas invariantes à temperatura ou ao microfone. Separação real de cada mosquito em um único microfone não foi demonstrada.

## Dados e escolha dos modelos

Somente Dryad e HumBug entram no ajuste: {protocol['training_windows']} janelas válidas e {protocol['training_groups']} grupos. Esses grupos não necessariamente equivalem a indivíduos. Mantivemos as divisões existentes por grupo e bytes, com validação e calibração separadas. MosquitoSong, UFRGS, CDMSC e Drive não entram no treino, perdas, normalização aprendida ou limites.

Três redes de 5,65 milhões de parâmetros: controle espectral, contraste e contraste+enxame. Mesmo frontend, arquitetura, amostragem e ruído real de treino; a perda auxiliar de espécie integra o pacote de contraste. Portanto não isolamos cada uma das perdas individualmente. Seeds diferentes são uma limitação desta comparação; não fizemos replicações por seed.

Até 24 épocas, 80 lotes de 96 por época; parada antecipada pela menor AUC ponderada entre fontes na validação. Limite final escolhido somente na calibração para equilibrar o pior recall por fonte/classe com pesos por grupo. Escolha final entre cinco candidatos pela validação, no limite realmente usado. Scores brutos não são probabilidades de acerto. Não ajustamos qualquer parâmetro pelo Drive ou pelo teste novo.

{validation[['name','best_epoch','cut','validation_precision','validation_recall','validation_macro_f1','selection_score']].to_markdown(index=False,floatfmt='.3f')}

## Teste adquirido depois do congelamento

Seleção fixa de 16 arquivos Aedes aegypti e quatro de cada uma das quatro espécies negativas, todos na fonte acústica D3. Total planejado: {len(audit)} arquivos; mantidos: {int(row.files)}; excluídos por fragmentos PCM não constantes já conhecidos: {int(audit.excluded.sum())}. CRC, bytes e fragmentos decodificados comparados com as bases e testes anteriores. Coincidências constantes/silenciosas foram registradas separadamente e não chamadas de duplicação de voo.

São {int(row.known_windows)} janelas rotuladas, {int(row.true_positive+row.false_negative+row.positive_invalid)} positivas e {int(row.true_negative+row.false_positive+row.negative_invalid)} negativas. O número de janelas não é número de mosquitos. CDMSC não disponibiliza aqui identidade original de indivíduo/sessão; demonstramos novidade de arquivos/fragmentos, não de indivíduos. Rótulos de arquivo não anotam presença de voo em cada janela.

{joined.to_markdown(index=False,floatfmt='.3f')}

Modelos `prior_` são comparações antigas congeladas, não candidatos da seleção atual. A antiga SVM Dryad+HumBug teve bom resultado relativo no Drive; sua comparação aqui é exploratória devido à escolha anterior informada por esse áudio. A SVM HumBug foi o candidato escolhido por validação na rodada anterior.

## Intervalos e comparação pareada

Bootstrap de 2000 reamostragens por arquivo, estratificado pela classe. Todas as previsões usam os mesmos arquivos; as diferenças pareadas reamostram os mesmos grupos. Esses intervalos não medem independência biológica desconhecida e não compensam pouca diversidade de sessões.

{pd.DataFrame(cirows).query('model == @selected').to_markdown(index=False,floatfmt='.3f')}

{pd.DataFrame(paired).to_markdown(index=False,floatfmt='.3f')}

## Áudio privado

O usuário identifica o Drive como enxame de Aedes na caixa de vidro. Mantivemos 107 janelas no denominador, incluindo duas inválidas. Sensibilidade nesse áudio não é acurácia binária: faltam negativos para medir falsos positivos. Melhor candidato no Drive, observado depois do congelamento: {best_drive.model}, {int(best_drive.true_positive)}/107; não foi promovido por esse resultado. Nenhuma janela privada foi treinada nem distribuída.

## Regressões históricas

As partições antigas de teste/challenge foram reavaliadas apenas depois do congelamento. São resultados retrospectivos: decisões gerais desta pesquisa já foram informadas por avaliações anteriores.

{historical[['model','scope','aedes_precision','aedes_recall','specificity','macro_f1','balanced_accuracy']].to_markdown(index=False,floatfmt='.3f')}

## Critérios e conclusão

Exigimos precisão, sensibilidade, especificidade e macro F1 de pelo menos 60%, inclusive limites inferiores de 95% por arquivo. Falhas do candidato escolhido: {', '.join(decision['failures']) or 'nenhuma neste teste'}.

Um resultado favorável em arquivo novo não comprova robustez a outra população, caixa, sexo ou temperatura. Precisamos de múltiplas sessões de Aedes e outras espécies com a mesma captação, além de controle da caixa vazia, para validar o domínio de uso. Firmware, instalador e checkpoints anteriores foram preservados; não compilamos nem comprimimos nada para Arduino.

![Comparação no teste novo](assets/cross_domain_strategy_comparison.png)

![Confusão e treinamento](assets/cross_domain_strategy_confusion.png)

## Fontes e reprodução

[DR-BioL](https://arxiv.org/html/2510.00346v1): inspiração para contraste e MMD por espécie; arquitetura e alguns objetivos diferem. [Enxames sintéticos, prova de conceito](https://arxiv.org/abs/2512.12365): motivação para simulação, não validação do nosso enxame real. [Sexo e temperatura](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0343060): justificam manter explícitas as variáveis ausentes. [BioDCASE](https://zenodo.org/records/19095788): origem dos novos arquivos acústicos.

Código: `cross_domain_strategy_models.py`, `train_cross_domain_strategies.py`, `fetch_cross_domain_strategy_test.py`, `evaluate_cross_domain_strategies.py`. Resultados e hashes em `results/cross_domain_strategies_2026_10_03/`. Os scripts de treino recusam sobrescrever um experimento congelado; uma nova busca precisa de nova rodada. O notebook apenas lê resultados.
'''
    report_path = ROOT / 'reports/estrategias_contrastivas_enxames_2026_10_03.md'; report_path.write_text(report)
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(report.replace('(assets/', '(reports/assets/')),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\np=Path('results/cross_domain_strategies_2026_10_03')\nfor name in ['comparison','historical_metrics','confidence_intervals','paired_comparisons']:\n    print(name)\n    display(pd.read_csv(p/(name+'.csv')))"),
        nbformat.v4.new_code_cell("from IPython.display import Image,display\ndisplay(Image('reports/assets/cross_domain_strategy_comparison.png'))\ndisplay(Image('reports/assets/cross_domain_strategy_confusion.png'))"),
        nbformat.v4.new_code_cell("decision=json.loads((p/'reliability_decision.json').read_text())\ndisplay(decision)\naudit=pd.read_csv(p/'novelty_audit.csv')\nassert (audit.loc[~audit.excluded,'nonconstant_pcm_matches']==0).all()\nprivate=pd.read_csv(p/'drive_positive_control.csv')\nassert (private.known_windows==107).all()\nassert not decision['arduino_compiled']")],
        metadata={'kernelspec': {'name':'python3','display_name':'Python 3','language':'python'}})
    NotebookClient(nb, timeout=120, resources={'metadata': {'path': str(ROOT)}}).execute()
    path = ROOT / 'estrategias_contrastivas_enxames_2026_10_03.ipynb'; nbformat.write(nb, path)
    html, _ = HTMLExporter().from_notebook_node(nb); path.with_suffix('.html').write_text(html)
    # Refresh the landing documentation without rewriting historical reports.
    readme = ROOT / 'README.md'; previous = readme.read_text()
    update = f'''## Contraste entre bases e enxames simulados — 03/10/2026

**Experimental; não incorporado ao Arduino.** Cinco candidatos novos: três redes de 5,65M parâmetros, SVM e ExtraTrees com descritores de múltiplos picos. Escolhido pela validação: `{selected}`. Teste acústico adquirido depois do congelamento: precisão {100*row.aedes_precision:.1f}%, sensibilidade {100*row.aedes_recall:.1f}%, especificidade {100*row.specificity:.1f}%; Drive {int(drive.true_positive)}/107. Consulte os intervalos e as limitações antes de interpretar confiabilidade.

[Relatório](reports/estrategias_contrastivas_enxames_2026_10_03.md) · [Notebook executado](estrategias_contrastivas_enxames_2026_10_03.ipynb) · [HTML](estrategias_contrastivas_enxames_2026_10_03.html) · [Decisão](results/cross_domain_strategies_2026_10_03/reliability_decision.json).

'''
    if '## Contraste entre bases e enxames simulados — 03/10/2026' not in previous:
        readme.write_text(previous.replace('# Estudo de classificação acústica de mosquitos\n\n', '# Estudo de classificação acústica de mosquitos\n\n' + update, 1))
    verification = dict(report_sha256=sha(report_path), notebook_sha256=sha(path), html_sha256=sha(path.with_suffix('.html')),
                        executed_code_cells=3, arduino_compiled=False)
    (OUT / 'artifact_verification.json').write_text(json.dumps(verification, indent=2))
    print('Report, executed notebook and HTML created', flush=True)


if __name__ == '__main__': main()
