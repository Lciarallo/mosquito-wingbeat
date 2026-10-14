"""Immutable-input v1/v2/v3 audit and report; no fitting or threshold adjustment."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from improve_device import ROOT,file_hash,atomic_json
from nested_species_ensemble import now
from research_metrics_v2 import classification_metrics,bootstrap_metrics


def historical_new_file_predictions(new,out):
    """Only postfreeze_new has no saved historical predictions; evaluate fixed old weights."""
    from nested_species_ensemble import audio_probabilities as audio_v1
    import nested_species_ensemble_v2 as v2
    b1=joblib.load(ROOT/'results/species_nested_2026_10_09/models/final_bundle.joblib')
    b2=joblib.load(ROOT/'results/species_nested_2026_10_13/models/final_bundle.joblib')
    records=[]
    for r in new.itertuples():
        p,q,_=audio_v1(ROOT/r.path,b1);p=p['calibrated'];avg=p.mean(0);g,gp=v2.marginalize_genus(avg[None,:],b1['classes'])
        records.append(dict(group=r.group,version='v1',species=b1['classes'][avg.argmax()] if q.all() else 'INVALID',
            genus=g[gp.argmax(1)][0] if q.all() else 'INVALID',posthoc=b1['classes'][avg.argmax()].split()[0] if q.all() else 'INVALID',
            accepted_species=bool(q.all() and avg.max()>=b1['thresholds']['calibrated']['threshold']),accepted_genus=bool(q.all())))
        p,pp,s,q,_=v2.audio_probabilities(ROOT/r.path,b2)
        f=pd.DataFrame(dict(group=[r.group]*len(p),label=[r.label]*len(p)))
        old,_,_=v2.prediction_records(f,p,s,np.array(b2['classes']),b2['settings'],pp,q,'postfreeze_new')
        t=old.iloc[0]
        records.append(dict(group=r.group,version='v2',species=t.predicted_species if t.quality_ok else 'INVALID',
            genus=t.genus_marginalized if t.quality_ok else 'INVALID',posthoc=t.genus_posthoc if t.quality_ok else 'INVALID',
            accepted_species=bool(t.accepted_species),accepted_genus=bool(t.accepted_genus)))
    data=pd.DataFrame(records);data.to_csv(out/'historical_postfreeze_new_predictions.csv',index=False)
    return data


def main():
    from nested_species_ensemble_v3 import OUT,V2,verify_freeze,verify_history,SEED
    freeze=verify_freeze()
    if (OUT/'comparison_audit.json').exists():raise FileExistsError('Completed comparisons cannot be replaced.')
    evaluation=json.loads((OUT/'evaluation_audit.json').read_text())
    if evaluation['completed_utc']<=freeze['created_utc']:raise ValueError('Evaluation predates freeze.')
    internal=pd.read_csv(OUT/'recording_predictions.csv');external=pd.read_csv(OUT/'external_recording_predictions.csv')
    baseline=pd.read_csv(ROOT/'results/three_levels_2026_10_13/paired_recording_predictions.csv')
    frame=pd.read_csv(OUT/'segments_v3.csv');mapping=frame.groupby('group').pcm_block.first()
    rows=[];per_taxon=[];paired=[];intervals={};matrices=OUT/'confusion_matrices';matrices.mkdir(exist_ok=False)
    all_scopes=[('dryad',internal)]+[(scope,f) for scope,f in external.groupby('scope')]
    novel=external[external.scope.eq('postfreeze_new')]
    older_new=historical_new_file_predictions(novel,OUT) if len(novel) else pd.DataFrame()
    for scope,new in all_scopes:
        new=new[new.label.ne('noise')].sort_values('group').reset_index(drop=True)
        if new.empty:continue
        if new.group.duplicated().any():raise ValueError('Duplicated recording identities.')
        clusters=new.group.map(mapping).fillna(new.group).to_numpy(str)
        for task,true,column,acceptcol in [
            ('species',new.label,'predicted_species','accepted_species'),
            ('genus_marginalized',new.true_genus,'genus_marginalized','accepted_genus'),
            ('genus_posthoc',new.true_genus,'genus_posthoc',None)]:
            predictions={};accepted={}
            if scope=='postfreeze_new':
                key={'species':'species','genus_marginalized':'genus','genus_posthoc':'posthoc'}[task]
                for version in ['v1','v2']:
                    old=older_new[older_new.version.eq(version)].set_index('group').reindex(new.group)
                    predictions[version]=old[key].to_numpy()
                    accepted[version]=np.ones(len(new),bool) if acceptcol is None else old[acceptcol].to_numpy(bool)
            else:
                old=baseline[baseline.scope.eq(scope)&baseline.task.eq(task)].set_index('group').reindex(new.group)
                if old.label.isna().any() or not np.array_equal(old.label.to_numpy(),np.asarray(true)):
                    raise ValueError('Historical pairing/reference differs for '+scope+' '+task)
                for version,key in [('v1','v1'),('v2','v2')]:
                    predictions[version]=old['prediction_'+key].to_numpy()
                    accepted[version]=old['accepted_'+key].to_numpy(bool)
            predictions['v3']=np.where(new.quality_ok,new[column],'INVALID')
            accepted['v3']=new[acceptcol].to_numpy(bool) if acceptcol else new.quality_ok.to_numpy(bool)
            cis=bootstrap_metrics(true,predictions,accepted,seed=SEED,clusters=clusters)
            pair23=bootstrap_metrics(true,{k:predictions[k] for k in ['v2','v3']},
                                    {k:accepted[k] for k in ['v2','v3']},seed=SEED,clusters=clusters)
            intervals[scope+'|'+task]=dict(v1_reference=cis,v2_reference=pair23)
            for version,pred in predictions.items():
                m=classification_metrics(true,pred,accepted[version]);m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
                rows.append(dict(scope=scope,task=task,version=version,unit='recording',**m))
                labels=np.union1d(np.unique(true),np.unique(pred));cm=confusion_matrix(true,pred,labels=labels)
                pd.DataFrame(cm,index=labels,columns=labels).to_csv(matrices/f'{scope}_{task}_{version}.csv')
                for taxon in labels:
                    y=np.asarray(true);tp=int(((y==taxon)&(pred==taxon)).sum());support=int((y==taxon).sum());count=int((pred==taxon).sum())
                    per_taxon.append(dict(scope=scope,task=task,version=version,taxon=taxon,support=support,
                        correct=tp,predicted=count,recall=tp/support if support else None,precision=tp/count if count else None))
            for i,g in enumerate(new.group):
                paired.append(dict(scope=scope,task=task,group=g,pcm_block=clusters[i],label=np.asarray(true)[i],
                    **{'prediction_'+v:predictions[v][i] for v in predictions},**{'accepted_'+v:bool(accepted[v][i]) for v in accepted}))
            if acceptcol:
                gate=new[acceptcol+'_gated'].to_numpy(bool)
                m=classification_metrics(true,predictions['v3'],gate);m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
                rows.append(dict(scope=scope,task=task+'_presence_gate',version='v3',unit='recording',**m))
    for scope,f in all_scopes:
        y=np.where(f.label.eq('noise'),'noise','mosquito');pred=np.where(f.quality_ok,np.where(f.detected,'mosquito','noise'),'INVALID')
        m=classification_metrics(y,pred,f.quality_ok);m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
        positive=f.label.ne('noise').to_numpy()
        m.update(positive_groups=int(positive.sum()),negative_groups=int((~positive).sum()),
                 sensitivity=float(np.mean(pred[positive]=='mosquito')) if positive.any() else None,
                 specificity=float(np.mean(pred[~positive]=='noise')) if (~positive).any() else None,
                 precision_informative=bool(positive.any() and (~positive).any()))
        rows.append(dict(scope=scope,task='presence',version='v3',unit='recording',**m))
        labels=np.union1d(y,pred);pd.DataFrame(confusion_matrix(y,pred,labels=labels),index=labels,columns=labels).to_csv(matrices/f'{scope}_presence_v3.csv')
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'comparison_metrics.csv',index=False)
    taxa=pd.DataFrame(per_taxon);taxa.to_csv(OUT/'per_taxon.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_recording_predictions.csv',index=False)
    atomic_json(OUT/'bootstrap_intervals.json',intervals)
    atomic_json(OUT/'comparison_audit.json',dict(completed_utc=now(),v1_v2_saved_pairs_sha256=file_hash(ROOT/'results/three_levels_2026_10_13/paired_recording_predictions.csv'),
        exact_v2_v3_partitions=True,no_retuning=True,release_allowed=False,historical_files=verify_history()))
    generate_report(metrics,taxa,intervals,OUT,freeze)
    verify_freeze();verify_history()
    print(metrics[metrics.task.isin(['species','genus_marginalized'])][['scope','task','version','n','accuracy','balanced_accuracy','macro_f1_support_only','macro_f1_strict_union','coverage','accepted_accuracy']].to_string(index=False))


def generate_report(metrics,taxa,intervals,out,freeze):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=ROOT/'reports/ensemble_especies_v3_2026_10_14.md'
    if target.exists():raise FileExistsError('Never overwrite a completed report.')
    figures=out/'figures';figures.mkdir(exist_ok=False)
    protocol=json.loads((out/'protocol.json').read_text());parts=json.loads((out/'partition_audit.json').read_text())
    novel=json.loads((out/'fresh_selection_v3.json').read_text());tests=json.loads((out/'test_validation.json').read_text())
    calibrations={str(f):json.loads((out/f'calibration_{f}.json').read_text()) for f in [0,1,2,'final']}
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for ax,scope in zip(axes,['dryad','external']):
        for j,task in enumerate(['species','genus_marginalized']):
            for i,(version,color) in enumerate(zip(['v1','v2','v3'],['#8b929c','#b8794f','#197e64'])):
                row=metrics.query('scope==@scope and task==@task and version==@version').iloc[0]
                x=j+(i-1)*.24;y=100*row.accuracy;ax.bar(x,y,.23,color=color,label=version if j==0 else None)
                ci=intervals[scope+'|'+task]['v1_reference']['intervals'][version]['accuracy']
                ax.vlines(x,100*ci['low'],100*ci['high'],color='#303030')
                ax.hlines([100*ci['low'],100*ci['high']],x-.025,x+.025,color='#303030')
        ax.set_xticks([0,1],['Espécie','Gênero']);ax.set_ylim(0,105);ax.spines[['top','right']].set_visible(False)
        ax.set_title('Dryad: 569 gravações' if scope=='dryad' else 'BioDCASE D3: 17 arquivos')
    axes[0].set_ylabel('Acurácia por gravação (%)');axes[1].legend(frameon=False)
    fig.suptitle('v1 / v2 / v3 — IC95% bootstrap por blocos PCM')
    fig.tight_layout();fig.savefig(figures/'comparison_accuracy.png',dpi=180);fig.savefig(figures/'comparison_accuracy.pdf');plt.close(fig)
    pct=lambda x:'indefinido' if pd.isna(x) else f'{100*x:.2f}%'
    table=['| Base | Nível | Versão | n | Acurácia | Balanceada | F1 suporte | F1 união estrita | Cobertura | Acerto aceitas |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ['dryad','external','fresh','drive','postfreeze_new']:
        for task in ['species','genus_marginalized']:
            for _,r in metrics[(metrics.scope==scope)&(metrics.task==task)].iterrows():
                table.append(f'| {scope} | {task} | {r.version} | {int(r.n)} | {pct(r.accuracy)} | {pct(r.balanced_accuracy)} | {pct(r.macro_f1_support_only)} | {pct(r.macro_f1_strict_union)} | {pct(r.coverage)} | {pct(r.accepted_accuracy)} |')
    thresholds=['| Dobra | Limiar v2 de gênero | Piso v3 | Limiar v3 | Objetivo na calibração | Cobertura na calibração | Acerto na calibração |',
                '|---|---:|---:|---:|---:|---:|---:|']
    for fold,c in calibrations.items():
        old=json.loads((ROOT/'results/species_nested_2026_10_13'/f'calibration_{fold}.json').read_text())
        t=c['threshold_genus'];selected=next(r for r in t['curve'] if r['threshold']==t['threshold'])
        thresholds.append(f'| {fold} | {old["threshold_genus"]["threshold"]:.6f} | {t["minimum_confidence"]:.6f} | {t["threshold"]:.6f} | {t["objective"]:.4f} | {pct(selected["coverage"])} | {pct(t["calibration_accuracy"])} |')
    delta=['| Base | Nível | Δ acurácia v3−v2 (p.p.) | IC95% pareado | Δ F1 estrito | IC95% pareado |',
           '|---|---|---:|---|---:|---|']
    for scope in ['dryad','external','fresh','postfreeze_new']:
        for task in ['species','genus_marginalized']:
            key=scope+'|'+task
            if key not in intervals:continue
            bounds=intervals[key]['v2_reference'].get('paired_deltas',{}).get('v3',{})
            sub=metrics[(metrics.scope==scope)&metrics.task.eq(task)].set_index('version')
            vals=[]
            for metric in ['accuracy','macro_f1_strict_union']:
                difference=100*(sub.loc['v3',metric]-sub.loc['v2',metric]);q=bounds.get(metric,{})
                vals.extend([f'{difference:+.2f}',f'[{100*q["low"]:+.2f}; {100*q["high"]:+.2f}]' if q.get('low') is not None else 'indefinido'])
            delta.append('| '+scope+' | '+task+' | '+' | '.join(vals)+' |')
    aedes=['| Base | n Ae. aegypti | Acertos v1 | Acertos v2 | Acertos v3 |', '|---|---:|---:|---:|---:|']
    for scope in ['dryad','external','fresh','drive','postfreeze_new']:
        sub=taxa.query('scope==@scope and task=="species" and taxon=="Aedes aegypti"').set_index('version')
        if len(sub)==3:aedes.append(f'| {scope} | {int(sub.loc["v3","support"])} | {int(sub.loc["v1","correct"])} | {int(sub.loc["v2","correct"])} | {int(sub.loc["v3","correct"])} |')
    genus=['| Base | Versão | Acurácia gênero da espécie vencedora | Acurácia gênero marginalizado |', '|---|---|---:|---:|']
    for scope in metrics.scope.unique():
        for version in ['v1','v2','v3']:
            sub=metrics.query('scope==@scope and version==@version').set_index('task')
            if 'genus_posthoc' in sub.index:genus.append(f'| {scope} | {version} | {pct(sub.loc["genus_posthoc","accuracy"])} | {pct(sub.loc["genus_marginalized","accuracy"])} |')
    presence=metrics[metrics.task.eq('presence')]
    pt=['| Base | Positivos | Negativos | Sensibilidade | Especificidade | Balanceada |', '|---|---:|---:|---:|---:|---:|']
    for _,r in presence.iterrows():pt.append(f'| {r.scope} | {int(r.positive_groups)} | {int(r.negative_groups)} | {pct(r.sensitivity)} | {pct(r.specificity)} | {pct(r.balanced_accuracy)} |')
    drive=pd.read_csv(out/'external_recording_predictions.csv').query('scope=="drive"').iloc[0]
    external=metrics.query('scope=="external" and version=="v3"').set_index('task')
    target.write_text(f'''# Ensemble de espécies v3 — 14/10/2026

## Resultado e decisão

Rodada offline concluída: espécie {int(external.loc['species','correct'])}/17 e gênero {int(external.loc['genus_marginalized','correct'])}/17 no BioDCASE D3 pareado. O piso e a otimização do limiar de gênero foram implementados; não equivalem a confirmação biológica nem a rejeição universal de fontes desconhecidas. **Confiabilidade externa não estabelecida; sem liberação ou compilação para Arduino.** Não houve ajuste por nenhum resultado externo.

O motivo da rodada é a hipótese de que zerar c1/c2 e remover a inclinação do PSD sacrificou timbre na v2. O contraste conjunto v2/v3 não isola causalmente cada alteração; não prova que essa foi a única causa do colapso externo. Todos os 17 arquivos, os dois chamados `fresh` da v2 e o Drive já eram conhecidos: continuam retrospectivos nesta rodada. A nova aquisição posterior ao congelamento está separada como `postfreeze_new`.

## Preservação e partições

O diretório v2 real é `results/species_nested_2026_10_13/`; o nome com `species_nested_v2` citado na solicitação não existe e não foi criado nem renomeado. SHA-256 de todos os resultados anteriores, relatórios, firmware, caches, scripts anteriores, testes anteriores, README e MODEL_VALIDATION foram registrados antes das alterações. `verify_freeze` validou v1, v2 e v3; `historical_hashes_*.json` registra a verificação completa. Nenhum artefato histórico foi reescrito.

Reutilizamos exatamente as linhas de fit, calibração e teste da v2, mantendo 24.405 janelas, 613 grupos originais, 560 blocos PCM e 20 espécies. O agrupamento por `pcm_block` conserva o bloco de 54 grupos com rótulos fracos mistos. Bytes dos 818 caminhos de origem foram reconferidos contra `source_pcm_audit.csv`. Não se usaram estatísticas do teste para normalizar, balancear ou selecionar features. A auditoria PCM é limitada a identidade inteira/fragmentos alinhados de 1/.992/.3 s a 8 kHz, com versões de ganho; não prova ausência de derivados arbitrários nem independência de indivíduo/sessão.

v2 e v3 usam as mesmas partições, facilitando o pareamento. A v1 usou outro isolamento; suas colisões PCM atravessavam partições, como documentado na v2. Sua comparação não é uma ablação no mesmo protocolo.

## Frontend v3

- Uma janela mono real de 1 s a 16 kHz, centralizada e normalizada por seu próprio RMS. Não se usa CMN sobre uma gravação inteira ou estatísticas de outros arquivos. O c0 médio fica exatamente zero.
- Subtração de um offset escalar de mediana da envoltória log-mel em cada frame, exclusivamente dentro da janela; c1/c2 e os demais coeficientes médios são preservados, assim como variabilidade e deltas. A v2 zerava c0/c1/c2 e atenuava c3. O log-mel usa potência com piso de ruído local subtraído.
- PSD com subtração da mediana espectral móvel local: 31 bins de 5 Hz abrangem 150 Hz. A v3 não faz regressão/removal da inclinação linear nem escala IQR. Centralização final subtrai somente um offset comum de ganho. Razões h2/h1 e h3/h1 vêm da energia com piso subtraído em bandas ±15 Hz, e não de razões entre contrastes logarítmicos. São limitadas a 100 nas features para estabilizar fundamentais quase ausentes.
- STFT de 100 ms (1600 amostras, hop 160) rastreia o pente harmônico temporal; a de 200 ms (3200, hop 320) estima perfil/peaks com espaçamento de bins de 5 Hz. A grade de candidatos é 200–900 Hz. Espaçamento de 5 Hz não garante erro de estimação de 5 Hz, e hops sobrepostos não criam observações independentes.
- Saliência, três picos, fração de energia harmônica, razões de energia entre faixas e divergência entre tracks curto/longo completam as {len(protocol['feature_names'])} features. A faixa 500–1600 Hz é um descritor de energia, não uma regra que atribui automaticamente espécie Aedes.

Normalizar ganho e reduzir piso local não garante invariância a qualquer curva de microfone. Resposta de canal e timbre podem ser inseparáveis em uma janela isolada. Pesos espécie/aparelho continuam calculados somente no fit; espécies com um único aparelho declarado continuam confundidas. Testes verificam preservação de c1/c2, relações harmônicas, invariância ao ganho em tolerância numérica e independência bit a bit de outras linhas do lote.

## Ajuste, calibração e limiar

SVM RBF C=3 com StandardScaler somente no fit; ExtraTrees com 200 árvores/max_features=.5/min_samples_leaf=2; HistGradientBoosting com 120 iterações/learning_rate=.08/15 folhas/L2=1. HGB binário e Platt medem presença separadamente. Pesos de bloco PCM/classe/aparelho e MIL foram mantidos: voto por saliência vezes nitidez, top 25/50/100% escolhido somente na calibração. Temperatura, combinação com prior logarítmico e rejeição também usam somente essa partição. Sua reutilização pode gerar otimismo interno; nenhum estimador reentra no ajuste com a calibração.

O gênero é a soma exata das probabilidades calibradas de suas espécies. Seu piso é 1/4+.05=0.30. A busca maximiza acurácia aceita vezes raiz da cobertura, sujeita a acurácia de calibração ≥90% e ≥20 blocos PCM distintos aceitos. Candidatos e objetivo completo estão em `calibration_*.json`; ausência de candidato qualificado rejeita tudo com 1.000001. A regra não força um corte adicional quando todas as decisões já satisfazem o objetivo. O limiar pode ficar no piso, mas nunca em zero; isso não significa que a calibração transfere ao domínio externo.

{chr(10).join(thresholds)}

## Congelamento e cronologia

Todas as bases, transformações, temperaturas, pesos, regras MIL e limiares das três dobras e do ajuste final foram congelados em **{freeze['created_utc']} UTC**, antes da primeira predição de qualquer teste OOF ou externo. Hash de `freeze.json`: `{file_hash(out/'freeze.json')}`. `manifest.json` registra SHA-256 dos pesos, calibrações, protocolo e features. `evaluation_started.json` e `evaluation_audit.json` comprovam a ordem. A extração prévia de features de teste era determinística por janela, sem estatística de lote nem ajuste supervisionado. Modelos e inferência avaliam cada janela da mesma forma isolada ou em lote.

Após o congelamento foi feita busca na fonte oficial [BioDCASE/Zenodo](https://zenodo.org/records/19095788) e aquisição por ID/duração predefinidos. Foram aceitos {len(novel['accepted_files'])} arquivos ainda não usados; {len(novel['exclusions'])} exclusões registradas, sem reposição por previsões. Forks Apple/fixtures incompatíveis têm seus bytes indexados e erros de PCM explicitamente registrados, não são anunciados como PCM auditado. Arquivos novos são de um corpus familiar, sem identidade biológica/sessão confirmada: não uma campanha independente. Os dois arquivos `fresh` da v2 permanecem apenas comparadores conhecidos.

## Comparação por gravação

{chr(10).join(table)}

![Acurácia v1/v2/v3](../results/species_nested_v3_2026_10_14/figures/comparison_accuracy.png)

A tabela taxonômica é sem gate de presença, para manter a comparação histórica. Colunas/métricas adicionais com gate estão nos CSVs. Qualidade antiga exige todas as janelas válidas por arquivo; v2/v3 permitem votar com as janelas válidas selecionadas por MIL. Inválidos continuam nos denominadores e não contam como negativos corretos. Coberturas diferentes também decorrem dessa política; acerto nas emissões compara subconjuntos potencialmente distintos.

## Intervalos pareados

{chr(10).join(delta)}

`bootstrap_intervals.json` contém todos os IC95% e diferenças v3−v1/v3−v2. São 2.000 réplicas percentis, semente 20261014, com os mesmos sorteios de blocos PCM para os modelos, estratificadas por conjunto de rótulos verdadeiro do bloco. F1 usa a união de rótulos observados verdadeiros/preditos na réplica. Sem emissões, acerto aceito fica indefinido e conta-se a quantidade de réplicas válidas. Estratos únicos ou desfechos homogêneos podem produzir intervalos degenerados; largura zero não demonstra precisão populacional. O Drive é uma só gravação e não recebe IC independente por janelas. Comparações múltiplas são exploratórias.

F1 por suporte continua penalizando falsos positivos ausentes via falsos negativos das classes verdadeiras; F1 de união também inclui explicitamente os táxons espúrios com F1 zero. Não se interpretam os 17 arquivos de cinco espécies como confirmação das 20 classes.

## Aedes aegypti e gênero

{chr(10).join(aedes)}

{chr(10).join(genus)}

Todas as espécies têm métricas por táxon e matrizes de confusão em `per_taxon.csv` e `confusion_matrices/`; táxons ausentes não recebem sensibilidade inventada. Avaliar a hipótese de timbre exige futuras ablações internas, sem escolher a arquitetura pelos externos já examinados.

## Presença e enxame privado

{chr(10).join(pt)}

Externos sem ausência/ruído medem sensibilidade, não validam precisão/especificidade. O detector histórico de presença tinha outro protocolo/unidade e não foi anunciado como comparação pareada.

O Drive possui {drive.duration_s:.2f} s: {int(drive.complete_windows)} janelas completas e {drive.unanalysed_tail_s:.2f} s finais não analisados por não completar 1 s. {int(drive.invalid_windows)} janelas inválidas permaneceram no denominador. Na v3 houve {int(drive.forced_correct_windows)}/{int(drive.complete_windows)} escolhas corretas de espécie por janela, apenas diagnóstico descritivo. Candidata por gravação: **{drive.predicted_species}**; gênero: **{drive.genus_marginalized}**; espécie aceita: {drive.accepted_species}; gênero aceito: {drive.accepted_genus}; presença detectada: {drive.detected}. Referência do usuário: Aedes aegypti em todo o arquivo, sem anotação entomológica de eventos independente. O áudio privado não foi copiado ou publicado.

## Testes, artefatos e limites

Linha de base: 156 testes passaram. Suíte final: **{tests['summary']}**, `.venv/bin/python -m pytest -q tests`, status {tests['exit_code']}. Logs de avisos ficam preservados. Testes novos cobrem PCM, comportamento indutivo, ganho/timbre/energia harmônica, STFTs, critério de confiança, candidatos de calibração, congelamento anterior a OOF/externos, pares 17+2 e preservação histórica. Todos os resultados e figuras novos estão em `results/species_nested_v3_2026_10_14/`; somente este relatório novo foi escrito em `reports/`. README, MODEL_VALIDATION, relatórios e firmware anteriores permaneceram intactos.

Os critérios de MODEL_VALIDATION exigem métricas e limites inferiores ≥60% em aquisição adequada, inclusive entre mosquitos e táxons negativos. A amostra externa pequena, conhecida em parte, sem negativos suficientes e sem novos indivíduos/sessões confirmados não satisfaz esse conjunto de critérios. Mesmo melhora pontual não autoriza firmware. A próxima confirmação deve trazer outros aparelhos/ambientes, espécies/sexo/indivíduos conhecidos, controles de ruído, múltiplos enxames e anotação temporal, com modelos congelados e falsos alarmes por hora.

## Referências

- [SciPy STFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html): relação entre nperseg, sobreposição e eixos de frequência/tempo.
- [Rosenberg et al., 1994, normalização cepstral de canal](https://www.isca-archive.org/icslp_1994/rosenberg94_icslp.html): contexto de compensação de canal; a centralização escalar intra-janela aqui não é reprodução de CMN por gravação nem garantia de remoção completa de canal.
''')
    print('V3 REPORT',target,flush=True)


if __name__=='__main__':main()
