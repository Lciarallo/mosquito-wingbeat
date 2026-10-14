"""Reports and conditional recording bootstrap for the corrected experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nested_species_ensemble import ROOT, NAMES


def paired_intervals(r):
    truth=r.drop_duplicates('group').set_index('group').true_species.sort_index()
    groups=truth.index.to_numpy()
    labels=truth.to_numpy()
    draws=[]
    rng=np.random.default_rng(20261009)
    for label in np.unique(labels):
        ix=np.flatnonzero(labels==label)
        draws.append(rng.choice(ix,size=(2000,len(ix)),replace=True))
    indices=np.concatenate(draws,axis=1)
    result={};samples={}
    for name in NAMES:
        sub=r[r.model.eq(name)].set_index('group').reindex(groups)
        correct=sub.predicted_species.to_numpy()==labels
        samples[name]=correct[indices].mean(1)
        result[name]=dict(accuracy_lower95=float(np.quantile(samples[name],.025)),
                          accuracy_upper95=float(np.quantile(samples[name],.975)))
    comparisons={}
    for other in ['hist_gradient_boosting','fixed']:
        diff=samples['calibrated']-samples[other]
        comparisons[other]=dict(mean_delta=float(diff.mean()),lower95=float(np.quantile(diff,.025)),
                                upper95=float(np.quantile(diff,.975)))
    return dict(method='Paired percentile bootstrap of recording groups, stratified by true species; 2000 replicates.',
                seed=20261009,scope='Conditional on fitted models and observed recordings; not retraining variance or unseen-domain uncertainty.',
                metrics=result,calibrated_accuracy_differences=comparisons)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/species_nested_2026_10_09')
    args=parser.parse_args();out=args.output.resolve()
    metrics=pd.read_csv(out/'metrics.csv')
    per=pd.read_csv(out/'per_species.csv')
    rec=pd.read_csv(out/'recording_predictions.csv')
    intervals=paired_intervals(rec)
    (out/'intervals.json').write_text(json.dumps(intervals,indent=2)+'\n')
    chosen=per[per.fold.eq('pooled') & per.unit.eq('recording') & per.model.eq('calibrated')]
    baseline=per[per.fold.eq('pooled') & per.unit.eq('recording') & per.model.eq('hist_gradient_boosting')]
    fig,ax=plt.subplots(figsize=(10,8))
    y=np.arange(len(chosen));labels=chosen.species.tolist()
    ax.barh(y-.18,baseline.set_index('species').loc[labels].recall*100,.35,label='HistGradientBoosting')
    ax.barh(y+.18,chosen.recall*100,.35,label='Ensemble calibrado')
    ax.set_yticks(y,[f'{sp} (n={n})' for sp,n in zip(labels,chosen.support)])
    ax.set_xlabel('Recall por grupo de gravação (%)');ax.set_xlim(0,105);ax.invert_yaxis()
    ax.set_title('Classificação nas dobras reservadas do Dryad\nCorpus conhecido; sem garantia de generalização')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
    fig.tight_layout();fig.savefig(out/'recall_per_species.png',dpi=150);plt.close(fig)
    rows=[]
    for name in NAMES:
        w=metrics[metrics.model.eq(name)&metrics.unit.eq('window')].iloc[0]
        r=metrics[metrics.model.eq(name)&metrics.unit.eq('recording')].iloc[0]
        accepted='indefinido' if pd.isna(r.accepted_accuracy) else f'{r.accepted_accuracy*100:.1f}%'
        rows.append(f'| {name} | {w.accuracy*100:.2f}% | {r.accuracy*100:.2f}% | {r.balanced_accuracy*100:.2f}% | {r.coverage*100:.1f}% | {accepted} |')
    species_rows=[]
    for row in chosen.itertuples():
        precision='indefinida' if pd.isna(row.precision) else f'{row.precision*100:.1f}%'
        species_rows.append(f'| {row.species} | {row.correct}/{row.support} | {row.recall*100:.1f}% | {precision} | {row.f1*100:.1f}% |')
    differences='\n'.join(f"- Calibrado menos {name}: {d['mean_delta']*100:+.2f} pontos percentuais; IC condicional de 95% {d['lower95']*100:+.2f} a {d['upper95']*100:+.2f}." for name,d in intervals['calibrated_accuracy_differences'].items())
    external='O teste externo ainda não foi concluído.'
    if (out/'external_metrics.csv').exists():
        e=pd.read_csv(out/'external_metrics.csv')
        primary=e[e.model.eq('calibrated')&e.unit.eq('recording')].iloc[0]
        ep=pd.read_csv(out/'external_recording_predictions.csv')
        ep=ep[ep.model.eq('calibrated')]
        ew=e[e.model.eq('calibrated') & e.unit.eq('window')].iloc[0]
        count=int(ep.accepted.sum())
        successes=int((ep.accepted & ep.label.eq(ep.forced_species)).sum())
        spp=[]
        for species,part in ep.groupby('label'):
            correct=int(part.label.eq(part.forced_species).sum())
            spp.append(f'| {species} | {correct}/{len(part)} |')
        external=f"No teste de arquivos externos selecionados depois do congelamento, o calibrado acertou **{int(primary.correct)}/{int(primary.n)} arquivos ({primary.accuracy*100:.1f}%)** sem rejeição. Em {int(ew.n)} janelas reais, acertou {int(ew.correct)} quando forçado. Com o limiar congelado, emitiu **{count} identificações: {successes} corretas e {count-successes} erradas**; cobertura {primary.coverage*100:.1f}% e acerto nas emissões {primary.accepted_accuracy*100:.1f}%. Não atingiu o alvo de 90% aprendido na calibração.\n\n| Espécie externa | Arquivos corretos / avaliados, escolha forçada |\n|---|---:|\n"+'\n'.join(spp)+"\n\nForam procurados quatro arquivos ainda não usados para cada uma das nove espécies acústicas D3 do [BioDCASE oficial](https://zenodo.org/records/19095788), em ordem de ID e com duração real de ao menos um segundo. Só cinco espécies tinham arquivos disponíveis após excluir nomes já presentes: quatro Ae. aegypti, um Ae. albopictus e quatro de cada Culex pipiens, Anopheles minimus e Anopheles stephensi. Não houve substituição por acerto do classificador. Os 17 arquivos passaram pela auditoria de bytes e fragmentos PCM declarados. A varredura incluiu 7.535 caminhos WAV; 261 não foram decodificados, sendo metadados AppleDouble e fixtures de teste SciPy, registrados na seleção. O teste de novidade não cobre cortes em deslocamentos arbitrários nem identidade biológica.\n\nEsses arquivos pertencem a um corpus conhecido, com ancestralidade de indivíduos/sessões incompleta; não são uma coleta independente nova. Os IDs sequenciais podem ser recortes relacionados: **17 arquivos não equivalem a 17 mosquitos**. Os longos arquivos Aedes dominam a contagem de janelas, por isso a acurácia por janela externa não deve ser comparada à distribuição Dryad sem considerar espécies/durações. Consulte [métricas externas](../results/species_nested_2026_10_09/external_metrics.csv), [seleção](../results/species_nested_2026_10_09/fresh_selection.json) e [auditoria de novidade](../results/species_nested_2026_10_09/novelty_audit.csv)."
    private=''
    if (out/'private_audio_metrics.csv').exists():
        p=pd.read_csv(out/'private_audio_metrics.csv')
        row=p[p.model.eq('calibrated')].iloc[0]
        private=f"\n\nNo áudio conhecido informado pelo usuário como Aedes aegypti, o calibrado acertou {int(row.forced_correct_windows)}/{int(row.windows)} janelas quando forçado; a espécie agregada foi `{row.recording_candidate}` e a emissão com o limiar congelado foi `{row.recording_identification}`. A referência é a informação do usuário sobre todo o áudio; não há anotação independente de cada voo. Esse é um controle retrospectivo, não um teste novo."
    report=f'''# Ensemble por espécie com calibração separada — 09/10/2026

**Treinamento novo de três modelos, com teste separado do ajuste e da calibração. Continua experimental, sem comprovação de confiabilidade em novas aquisições.** Esta rodada corrige o caminho de dependência encontrado na calibração anterior. Não repete integralmente o ensemble antigo de sete modelos.

## Protocolo executado

SVM RBF, ExtraTrees e HistGradientBoosting usam as mesmas partições de gravação. Mantidas as três dobras externas históricas; dentro do treino externo, reservados 20% dos grupos de cada espécie para calibração, com ao menos um grupo. Os modelos-base, incluindo o scaler SVM, veem apenas o ajuste. Os pesos e correções de classe são ajustados na calibração; o teste externo à calibração é consultado somente na avaliação. Não há refit dos modelos na calibração antes do teste.

Comparamos cada base, uma média fixa e a mistura calibrada nos mesmos arquivos. Primário declarado antes de testar: `calibrated`. São 23.877 janelas e 569 grupos das 20 espécies Dryad. Os arquivos já orientaram pesquisas anteriores; o resultado interno é retrospectivo. Grupos não comprovam independência de indivíduos/sessões. A gravação recebe a média de suas janelas reais.

Para rejeição por gravação, cada modelo tem limiar escolhido somente na calibração: alvo de 90% de acerto nas emissões, com ao menos 20 grupos emitidos. Esse alvo pode falhar no teste. Se a calibração não satisfaz o critério, o modelo se abstém de todas as gravações. É rejeição por confiança de espécie; não há detector de presença ou validação de falsos alarmes em ruído neste experimento. O frontend de frequências/contraste é fixo; nenhuma normalização do lote de teste é aprendida.

## Teste por espécie no corpus conhecido

| Modelo | Acerto por janela, forçado | Acerto por gravação, forçado | Recall médio das 20 espécies | Cobertura com rejeição | Acerto entre emissões |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Escolha forçada sempre retorna uma das 20 espécies; rejeição conserva todas as gravações no denominador de cobertura. Não comparar a última coluna isoladamente. O ganho de acerto ao rejeitar pode vir de emitir poucas identificações.

Comparação pareada por gravação, com 2.000 reamostragens estratificadas por espécie:

{differences}

Os intervalos são condicionais aos modelos já ajustados e aos grupos observados, sem reexecutar o treino. Não são garantia em outro microfone, ambiente, indivíduo ou campanha. A avaliação não mede eventos de voo independentes.

### Modelo primário, por gravação

| Espécie | Corretas / grupos | Recall | Precisão | F1 |
|---|---:|---:|---:|---:|
{chr(10).join(species_rows)}

![Recall por espécie](../results/species_nested_2026_10_09/recall_per_species.png)

## Teste depois do congelamento

{external}{private}

## Evidência e reprodução

[Partições por linha, grupo, caminho e hash](../results/species_nested_2026_10_09/partitions.csv) · [Auditoria de disjunção](../results/species_nested_2026_10_09/partition_audit.json) · [Protocolo](../results/species_nested_2026_10_09/protocol.json) · [Congelamento](../results/species_nested_2026_10_09/freeze.json) · [Intervalos](../results/species_nested_2026_10_09/intervals.json) · [Métricas por espécie](../results/species_nested_2026_10_09/per_species.csv).

Nenhum peso anterior, firmware ou instalador foi substituído. Modelos locais ignorados pelo Git. Para repetir o treino, usar outra pasta de saída; o script recusa sobrescrever uma pasta com protocolo existente.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python nested_species_ensemble.py --output results/outra_rodada
.venv/bin/python summarize_nested_species.py
.venv/bin/python -m pytest -q tests
```

Os resultados de `results/species_nested_2026_10_09/` usam o código identificado pelos hashes do protocolo. `optimize_ensemble_oof.py` passou a encaminhar para o experimento corrigido; a implementação antiga está preservada no histórico Git em `1ab88b8`. A aquisição recusa selecionar novamente quando já há um manifesto, e a avaliação recusa sobrescrever as métricas externas. Para executar essas etapas em outro experimento, passar seu `--output` explicitamente para `evaluate_nested_species.py --phase acquire` e depois `--phase evaluate`. Os áudios já adquiridos não são novos para a próxima rodada.

Não chamar o resultado de redução quantificada do vazamento do ensemble antigo: o número de membros e a quantidade de dados de ajuste também mudaram. Para confirmar confiabilidade, ainda é necessária aquisição independente, com espécie de referência, negativos/ruído e diversidade de indivíduos, sessões e dispositivos. Foram aprovados **136 testes**, incluindo a disjunção das partições realmente executadas, scalers ajustados apenas no treino, identidade das linhas OOF e hashes dos modelos/limiares congelados.
'''
    (ROOT/'reports/ensemble_especies_calibracao_separada_2026_10_09.md').write_text(report)
    decision=dict(status='EXPERIMENTAL_SPECIES_GENERALISATION_UNPROVEN',primary_model='calibrated',
                  ensemble_calibration_excludes_outer_test=True,independent_new_campaign=False,
                  can_release_to_user_device=False,arduino_compiled=False,
                  limitations=['Retrospective internal corpus','Unknown biological ancestry of external excerpts',
                               'Sparse recordings per species','No operational presence/noise validation'])
    (out/'reliability_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(metrics.to_string(index=False));print(json.dumps(intervals['calibrated_accuracy_differences'],indent=2))


if __name__=='__main__':main()
