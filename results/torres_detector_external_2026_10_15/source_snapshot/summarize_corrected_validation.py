"""Generate the corrected scientific report and a read-only notebook from frozen results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nbformat
from corrected_validation import ROOT,OUT,sha


def pct(value):
    return '—' if pd.isna(value) else f'{100*value:.1f}%'


def table(columns,rows):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    names={'tree_lightgbm':'LightGBM','tree_hist_gradient_boosting':'HistGB','tree_extra_trees':'ExtraTrees novo',
           'tree_random_forest':'Random Forest','committee_soft_voting':'Soft Voting','committee_stacking':'Stacking corrigido',
           'deep_hybrid_net':'Rede profunda preservada','super_committee':'Super-Comitê recalibrado','swarm_corrected':'Ensemble com CDF fixa',
           'baseline_svm_rbf':'SVM anterior','baseline_extra_trees':'ExtraTrees anterior','baseline_gradient_boosting':'Boosting anterior'}
    hist=pd.read_csv(OUT/'historical_metrics.csv');priv=pd.read_csv(OUT/'private_audio_aggregate.csv')
    synth=pd.read_csv(OUT/'synthetic_file_metrics_exploratory.csv');d5=pd.read_csv(OUT/'d5_test_metrics.csv');fresh=pd.read_csv(OUT/'fresh_metrics.csv')
    freeze=json.loads((OUT/'freeze.json').read_text());selected=json.loads((OUT/'fresh_selection.json').read_text())
    noise=pd.read_csv(OUT/'heldout_noise_metrics.csv');batch=json.loads((OUT/'batch_invariance_audit.json').read_text())
    folds=json.loads((OUT/'models_trees/fold_audit.json').read_text());old_hash=json.loads((OUT/'original_weights_sha256.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in old_hash.items())
    # Recording-cluster bootstrap, with explicit refusal when a class has too few source groups.
    predictions=pd.read_csv(OUT/'historical_predictions.csv.gz');thresholds={}
    import joblib
    bundle=joblib.load(OUT/'models_committee/corrected_bundle.joblib');thresholds=bundle['thresholds']
    intervals=[];rng=np.random.default_rng(20261006)
    for domain,frame in predictions.groupby('domain'):
        pos=frame.label.eq('Aedes aegypti');neg=~frame.label.isin(['Aedes aegypti','unknown_mosquito'])
        for model in names:
            high=thresholds[model][1];emission=frame[model].ge(high)&frame.quality_ok
            for target,mask in [('recall',pos),('fpr',neg)]:
                sub=pd.DataFrame({'group':frame.loc[mask,'group'],'n':1,'hit':emission[mask].astype(int)}).groupby('group').sum()
                row=dict(model=model,domain=domain,metric=target,source_groups=len(sub),windows=int(sub.n.sum()))
                if len(sub)<2:
                    row.update(low95=None,high95=None,note='Insufficient independent source groups; no interval.')
                else:
                    idx=rng.integers(0,len(sub),size=(2000,len(sub)));v=sub.to_numpy();boot=v[idx,1].sum(1)/v[idx,0].sum(1)
                    row.update(low95=float(np.quantile(boot,.025)),high95=float(np.quantile(boot,.975)),note='Recording-cluster bootstrap; exploratory, source identities may be incomplete.')
                intervals.append(row)
    pd.DataFrame(intervals).to_csv(OUT/'recording_cluster_intervals.csv',index=False)
    decision=dict(status='EXPERIMENTAL_CROSS_SOURCE_VALIDATION_FAILURE',models_released=False,
                  original_weights_preserved=True,arduino_compiled=False,
                  reasons=['Swarm ensemble remains near chance on Abuzz-derived historical mixtures.',
                           'New optical test shows 54.5–61.25% balanced accuracy after separate calibration.',
                           'Fresh non-Aedes recordings still produce false Aedes emissions.',
                           'No unused positive HumBug groups remained; fresh acoustic sensitivity is unmeasured.'])
    (OUT/'reliability_decision.json').write_text(json.dumps(decision,indent=2))
    body=['''# Validação corrigida dos modelos de mosquito — 01/10/2026

**Correções implementadas e experimento executado. O desempenho corrigido não sustenta a conclusão anterior de alta confiabilidade.** Esta é uma avaliação dos modelos no computador. Não houve trabalho de firmware, compilação, adaptação ou compactação para Arduino.

## O que foi corrigido

- As cinco dobras do stacking agora usam `StratifiedGroupKFold`: todas as janelas de uma mesma gravação ficam na mesma dobra. São 17.391 janelas em 355 grupos, sem grupo compartilhado entre treino e validação OOF. Antes, 99,95% das janelas OOF compartilhavam o grupo com o treino da dobra.
- O meta-modelo recebe probabilidades brutas das árvores tanto no OOF quanto na inferência. A versão anterior era treinada com probabilidades brutas e testada com probabilidades recalibradas, em escalas diferentes.
- O ensemble substitui os ranks do lote por distribuições empíricas ponderadas e congeladas no treino. O limiar é escolhido exclusivamente na calibração principal. Gravações inéditas não alteram a normalização nem o limiar.
- O Super-Comitê recebe limiares derivados da calibração principal, em vez do par manual 0,35/0,65. Sua meta de até 5% de falsos positivos é ponderada por origem e espécie na calibração; não é garantia de 5% no teste.
- As 212 janelas HumBug de espécie desconhecida continuam sem rótulo binário; não viram negativos.
- Os 436 controles de ruído que repetiam o treino foram excluídos. Foram avaliados 971 trechos das partições de teste e, depois do congelamento, novos ruídos baixados da internet.
- No domínio óptico D5, a calibração usa 400 arquivos da lista oficial de validação; o teste usa outros 400 da lista oficial de teste. Os 400 arquivos examinados anteriormente foram excluídos de ambos. Não há IDs, conteúdo binário ou PCM idênticos entre calibração e teste. Isso comprova separação de clipes, não separação por indivíduo: a identidade de cada mosquito não é fornecida nesse recorte.

As quatro árvores foram retreinadas com os mesmos hiperparâmetros e partições principais; o stacking foi reconstruído com o OOF corrigido. A rede profunda e o SVM existentes foram preservados e reavaliados: o problema identificado no OOF não exigia retreinar esses modelos. Os 12 checkpoints anteriores permaneceram idênticos por SHA-256. Todos os resultados novos ficam em `results/corrected_2026_10_01/`; predições do áudio privado ficam em `tmp/`.

**Limitação do frontend preservado:** a rede profunda do outro modelo repete internamente entradas curtas de 0,3 s até 0,992 s, conforme sua implementação anterior. Mantivemos esse comportamento para avaliar os mesmos pesos, sem apresentá-lo como uma melhoria de pré-processamento. Isso não acrescenta informação acústica e pode alterar os padrões temporais; os resultados D5 e os diagnósticos de arquivos curtos devem ser lidos com essa ressalva. Uma arquitetura que aceite a duração real precisa de treinamento e novo teste após congelamento. Os novos áudios acústicos desta rodada e o áudio privado foram avaliados em janelas completas de 0,992 s, sem essa extensão interna.
''']
    body.append('Congelamento dos modelos e limiares: **'+freeze['frozen_utc']+'**. Seleção dos novos áudios: **'+selected['selection_utc']+'**, posterior ao congelamento. Não houve reajuste após examinar os testes.')
    body.append('''## Comparação nas mesmas bases históricas

As bases abaixo já haviam sido examinadas, portanto são comparações históricas. Cada célula mostra **sensibilidade de emissão / falsos positivos emitidos**. Os limiares estão congelados; incertos e inválidos permanecem no denominador e não contam como acertos. As acurácias balanceadas forçadas, scores e coberturas estão nos CSVs.
''')
    rows=[]
    for model,label in names.items():
        row=[label]
        for domain in ['dryad','humbug','ufrgs']:
            r=hist[(hist.model==model)&(hist.domain==domain)].iloc[0];row.append(pct(r.emitted_aedes_recall)+' / '+pct(r.emitted_false_positive_rate))
        rows.append(row)
    body.append(table(['Modelo','Dryad: sensibilidade / FP','HumBug: sensibilidade / FP','UFRGS: sensibilidade / FP'],rows))
    old=pd.read_csv(ROOT/'tmp/independent_retest_2026_10_01/corrected_historical_metrics.csv')
    old_stack=old[old.model.eq('committee_stacking')];new_stack=hist[hist.model.eq('committee_stacking')]
    body.append('''### Stacking: antes e depois

Comparação de acurácia balanceada no corte 0,5, preservando os rótulos desconhecidos em ambas as versões:
''')
    body.append(table(['Base','Anterior','Corrigido'],[[d,pct(old_stack[old_stack.domain.eq(d)].iloc[0].balanced_accuracy),pct(new_stack[new_stack.domain.eq(d)].iloc[0].balanced_accuracy)] for d in ['dryad','humbug','ufrgs']]))
    body.append('A queda do stacking acompanha duas correções: dobras por gravação e escala consistente das entradas do meta-modelo. Não atribuímos toda a diferença exclusivamente ao vazamento. As árvores isoladas e a rede mantêm suas características/hiperparâmetros; o Super-Comitê também mudou os limiares, então mudanças na emissão não significam que aprendeu novas características.')
    body.append('''## Enxames sintéticos: comparação exploratória

O código público da base usa recortes do **Abuzz**, já utilizado no projeto, inseridos em posições diferentes do arquivo. O JSON indica presença da espécie no arquivo inteiro; não identifica os instantes em que ela está audível. Por isso, esta rodada mede presença por arquivo, não acurácia de espécie por janela, e não trata essas misturas como validação externa independente.
''')
    body.append(table(['Modelo','Arquivos','Acurácia balanceada','Sensibilidade emitida','Falsos positivos emitidos'],[[names[r.model],r.windows,pct(r.balanced_accuracy),pct(r.emitted_aedes_recall),pct(r.emitted_false_positive_rate)] for r in synth.itertuples()]))
    body.append(f"O ensemble corrigido ficou em **{pct(synth[synth.model.eq('swarm_corrected')].iloc[0].balanced_accuracy)}**, frente aos 65,55% anteriores obtidos com ranks do próprio lote e limiar otimizado no próprio teste. Seu limiar corrigido é {bundle['ensemble'].calibrated_threshold:.6f}, escolhido na calibração acústica principal. Ele ainda não generaliza adequadamente a essas misturas. Aplicar uma média de scores por arquivo é uma regra fixa deste diagnóstico; não houve calibração nas misturas.")
    body.append(f"A estabilidade entre lotes foi verificada em {batch['rows']} janelas: diferença máxima {batch['max_difference_single']:.2e} ao processar cada janela sozinha e {batch['max_difference_by_file']:.2e} ao processar por arquivo. O defeito de dependência do lote foi resolvido, sem que isso garanta qualidade de classificação.")
    body.append('''## D5: novo teste óptico, com calibração separada

200 Aedes e 200 não-Aedes na calibração oficial; outros 200 Aedes e 200 não-Aedes no teste oficial. Seleção por ordenação de hashes com semente pré-fixada, sem consultar as predições. Os limiares específicos do D5 são uma adaptação de domínio feita na validação; não são limiares escolhidos no teste.
''')
    optical=d5[d5.calibration.eq('separate_D5_validation')]
    body.append(table(['Modelo','AUC','Acurácia balanceada','Sensibilidade','Falsos positivos'],[[names[r.model],f'{r.auc:.3f}',pct(r.balanced_accuracy),pct(r.emitted_aedes_recall),pct(r.emitted_false_positive_rate)] for r in optical.itertuples()]))
    body.append('O intervalo observado de acurácia balanceada foi **54,5% a 61,25%**, inferior aos 73,5% destacados anteriormente. Não é a mesma amostra antiga, portanto essa diferença também inclui mudança de amostra/distribuição. Os resultados com os limiares acústicos originais também estão em `d5_test_metrics.csv` e apresentam muitos falsos positivos. Esta avaliação óptica não mede diretamente desempenho em áudio de microfone.')
    body.append('''## Novos áudios baixados após o congelamento

Foram selecionados, sem consultar resultados, todos os 12 arquivos de ruído ambiental/ventiladores disponíveis no repositório MosquitoSong e os três primeiros grupos HumBug ainda não utilizados segundo a regra registrada. Os três são Anopheles arabiensis: IDs 206579, 206590 e 206597. Sua identidade de gravação não aparece nos grupos anteriores e seus arquivos não são cópias binárias dos áudios já baixados. São exemplos novos da mesma campanha HumBug, não um novo domínio de aquisição.

Os ruídos totalizam **161 janelas** e os três mosquitos **seis janelas**. São amostras pequenas e correlacionadas dentro de cada arquivo. As gravações de ventiladores por microfones diferentes também não devem ser tratadas como eventos independentes.

Não restaram grupos acústicos de Aedes aegypti inéditos no HumBug que cumprissem a duração mínima. Isso foi registrado antes da inferência, sem reutilizar positivos antigos. Logo, esta rodada externa acústica estima falsos alarmes, **não sensibilidade em Aedes inédito**. O novo teste D5 tem positivos, mas é óptico.
''')
    rows=[]
    for model,label in names.items():
        a=fresh[(fresh.model==model)&fresh.source.eq('fresh_noise')].iloc[0];b=fresh[(fresh.model==model)&fresh.source.eq('fresh_other_mosquito')].iloc[0]
        rows.append([label,f'{int(a.aedes_emissions)}/{int(a.windows)} ({pct(a.emitted_false_positive_rate)})',
                     f'{int(b.aedes_emissions)}/{int(b.windows)} ({pct(b.emitted_false_positive_rate)})',int(b.uncertain)])
    body.append(table(['Modelo','Falsos Aedes no ruído novo','Falsos Aedes em An. arabiensis novo','Incerto em An. arabiensis'],rows))
    body.append('O SVM não emitiu Aedes nas seis janelas novas de Anopheles, mas rejeitou quatro como incertas; isso não corresponde a seis classificações corretas confirmadas. O ensemble com CDF fixa emitiu Aedes em cinco das seis, apesar de detectar muitos trechos do áudio privado: a sensibilidade maior vem acompanhada de alarmes falsos.')
    body.append('''## Áudio privado: referência do usuário em todas as 107 janelas

Todo o arquivo foi informado como enxame de Aedes aegypti em caixa de vidro gravado pelo Arduino. Essa é a referência fornecida pelo usuário, sem verificação entomológica independente. Existem duas janelas inválidas, mantidas no total. A gravação já era conhecida antes desta rodada e não é um teste cego nem uma amostra de 107 gravações diferentes.
''')
    body.append(table(['Modelo','Aedes detectado /107','Não-Aedes','Incerto','Inválido'],[[names[r.model],f'{r.aedes_emissions}/107 ({pct(r.emitted_aedes_recall)})',r.false_negative_emissions,r.uncertain,r.invalid] for r in priv.itertuples()]))
    body.append('O ensemble corrigido detecta 70/107, mas sua taxa de falsos positivos na UFRGS chega a 69,9% e nos seis trechos inéditos de Anopheles a 83,3%. A detecção no áudio privado, isoladamente, não demonstra reconhecimento confiável da espécie. A rede preservada continua com 59/107; o stacking corrigido ficou com 18/107 e 82 incertos.')
    body.append('''## Ruído de teste e incerteza

Nos 971 trechos de ruído efetivamente reservados para teste, o Super-Comitê recalibrado emitiu 57 falsos Aedes (5,9%), a rede 70 (7,2%), o stacking corrigido quatro (0,4%) e o ensemble com CDF fixa 195 (20,1%). Uma taxa baixa de alarmes acompanhada de muitas rejeições/positivos perdidos não basta para declarar o detector confiável.

`recording_cluster_intervals.csv` inclui intervalos exploratórios por reamostragem de gravações, em vez de tratar janelas como observações independentes. Quando há menos de dois grupos de origem para uma classe, o intervalo é recusado. Identidades biológicas/sessões incompletas continuam sendo uma limitação. O áudio privado e os seis novos trechos de Anopheles não permitem estimar confiabilidade geral com precisão.

## Estado final e reprodução

**Estado: experimental; generalização insuficiente.** As falhas de avaliação identificadas foram corrigidas. Os resultados agora mostram limitações importantes nos modelos, particularmente no ensemble e no stacking. Não houve seleção de um vencedor por estes testes, reajuste de limiares após avaliação nem atualização de firmware.

Arquivos de entrada, modelos, seleção, hashes, horários e auditorias estão em `results/corrected_2026_10_01/`. Os modelos anteriores permanecem preservados. Há sete testes de regressão das falhas corrigidas e 20 testes prévios, totalizando 27 testes aprovados. O notebook `validacao_corrigida_2026_10_01.ipynb` apenas lê os resultados e não treina nem compila.

Em uma reprodução nova, execute primeiro `models_trees.py`, depois `models_committee.py` e `corrected_validation.py --phase fit`. Depois execute `corrected_validation.py --phase evaluate`, `fetch_corrected_fresh.py`, `evaluate_corrected_fresh.py` e `summarize_corrected_validation.py`. Use `.venv/bin/python` na raiz do projeto. O experimento já congelado bloqueia novos ajustes: outra rodada de treinamento precisa de um novo diretório/protocolo. Os avaliadores anteriores agora encaminham a execução ao protocolo corrigido, preservando os resultados antigos.

## Referências de metodologia e origem

- [StratifiedGroupKFold: preservação dos grupos nas dobras](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [Gerador original das misturas a partir de Abuzz](https://github.com/duydinhthai27/synthetic-swarm-mosquito/blob/main/mosquito_swarm.ipynb).
- [MosquitoSong: origem dos novos controles ambientais e de ventiladores](https://github.com/akaraspt/mosquitosongp).
- [HumBugDB: origem dos três novos arquivos de Anopheles](https://zenodo.org/records/4904800).
- [BioDCASE: arquivos e listas oficiais de validação/teste](https://zenodo.org/records/20478577).
''')
    report=ROOT/'reports/validacao_corrigida_2026_10_01.md';report.write_text('\n\n'.join(body))
    nb=nbformat.v4.new_notebook();nb.cells=[
        nbformat.v4.new_markdown_cell('# Validação corrigida dos modelos\n\nExperimento congelado; este notebook somente lê resultados. Nenhum treino, download ou compilação é executado.\n\nLeia [o relatório](reports/validacao_corrigida_2026_10_01.md). Estado: experimental, generalização insuficiente.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nROOT = Path.cwd()\nif not (ROOT / 'results/corrected_2026_10_01').exists():\n    ROOT = ROOT.parent\nO = ROOT / 'results/corrected_2026_10_01'\nfreeze = json.loads((O / 'freeze.json').read_text())\nfreeze"),
        nbformat.v4.new_markdown_cell('## Separação por gravação e estabilidade por lote'),
        nbformat.v4.new_code_cell("display(pd.read_json(O / 'models_trees/fold_audit.json'))\ndisplay(json.loads((O / 'batch_invariance_audit.json').read_text()))"),
        nbformat.v4.new_markdown_cell('## Comparações históricas: sensibilidade, falsos positivos e rejeições'),
        nbformat.v4.new_code_cell("historical = pd.read_csv(O / 'historical_metrics.csv')\nhistorical[['model','domain','windows','balanced_accuracy','emitted_aedes_recall','emitted_false_positive_rate','known_coverage']]"),
        nbformat.v4.new_markdown_cell('## Misturas sintéticas: resultado por arquivo, sem garantia de origem independente'),
        nbformat.v4.new_code_cell("pd.read_csv(O / 'synthetic_file_metrics_exploratory.csv')"),
        nbformat.v4.new_markdown_cell('## D5 óptico: limiar da validação separado do teste'),
        nbformat.v4.new_code_cell("d5 = pd.read_csv(O / 'd5_test_metrics.csv')\nd5[d5.calibration.eq('separate_D5_validation')][['model','windows','auc','balanced_accuracy','emitted_aedes_recall','emitted_false_positive_rate']]"),
        nbformat.v4.new_markdown_cell('## Áudios acústicos inéditos após congelamento\n\nSomente negativos: não permitem estimar sensibilidade em Aedes inédito. Há três gravações novas de Anopheles, com seis janelas, e 12 gravações novas de ruído, com 161 janelas.'),
        nbformat.v4.new_code_cell("pd.read_csv(O / 'fresh_metrics.csv')[['model','source','windows','aedes_emissions','emitted_false_positive_rate','uncertain']]"),
        nbformat.v4.new_markdown_cell('## Áudio privado\n\nReferência do usuário: Aedes em todo o áudio. Métrica de sensibilidade em uma gravação conhecida; não acurácia geral.'),
        nbformat.v4.new_code_cell("pd.read_csv(O / 'private_audio_aggregate.csv')[['model','windows','aedes_emissions','emitted_aedes_recall','false_negative_emissions','uncertain','invalid']]"),
        nbformat.v4.new_markdown_cell('## Incerteza por gravação e decisão de confiabilidade'),
        nbformat.v4.new_code_cell("display(pd.read_csv(O / 'recording_cluster_intervals.csv'))\ndisplay(json.loads((O / 'reliability_decision.json').read_text()))")]
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.write(nb,ROOT/'validacao_corrigida_2026_10_01.ipynb')
    print('Report and read-only notebook generated.',flush=True)


if __name__=='__main__':main()
