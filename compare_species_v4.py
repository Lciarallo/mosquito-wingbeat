"""Post-freeze v1/v2/v3/v4 paired audit. No fitting, selection or retuning."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from improve_device import ROOT,file_hash,atomic_json
from nested_species_ensemble import now
from research_metrics_v2 import classification_metrics,bootstrap_metrics


def historical_predictions_on_v4_new(new,out):
    from compare_species_v3 import historical_new_file_predictions
    from nested_species_ensemble_v3 import audio_probabilities
    from nested_species_ensemble_v2 import prediction_records
    records=historical_new_file_predictions(new,out).to_dict('records')
    b=joblib.load(ROOT/'results/species_nested_v3_2026_10_14/models/final_bundle.joblib')
    for item in new.itertuples():
        p,pp,s,q,_=audio_probabilities(ROOT/item.path,b)
        f=pd.DataFrame({'group':[item.group]*len(p),'label':[item.label]*len(p)})
        r,_,_=prediction_records(f,p,s,np.array(b['classes']),b['settings'],pp,q,'postfreeze_new_v4');r=r.iloc[0]
        records.append(dict(group=item.group,version='v3',species=r.predicted_species if r.quality_ok else 'INVALID',
            genus=r.genus_marginalized if r.quality_ok else 'INVALID',posthoc=r.genus_posthoc if r.quality_ok else 'INVALID',
            accepted_species=bool(r.accepted_species),accepted_genus=bool(r.accepted_genus)))
    data=pd.DataFrame(records);data.to_csv(out/'historical_postfreeze_new_v4_predictions.csv',index=False)
    return data


def main():
    from nested_species_ensemble_v4 import OUT,V3,SEED,verify_freeze,verify_history
    freeze=verify_freeze()
    if (OUT/'comparison_audit.json').exists():raise FileExistsError('Never replace a completed comparison.')
    done=json.loads((OUT/'evaluation_audit.json').read_text())
    if done['completed_utc']<=freeze['created_utc']:raise ValueError('Evaluation predates freeze.')
    internal=pd.read_csv(OUT/'recording_predictions.csv');external=pd.read_csv(OUT/'external_recording_predictions.csv')
    baseline=pd.read_csv(V3/'paired_recording_predictions.csv')
    frame=pd.read_csv(OUT/'segments_v4.csv');mapping=frame.groupby('group').pcm_block.first()
    all_scopes=[('dryad',internal)]+list(external.groupby('scope'))
    oldnew=historical_predictions_on_v4_new(external.query('scope=="postfreeze_new_v4"'),OUT)
    rows=[];taxa=[];pairs=[];intervals={};matrices=OUT/'confusion_matrices';matrices.mkdir(exist_ok=False)
    for scope,new in all_scopes:
        new=new[new.label.ne('noise')].sort_values('group').reset_index(drop=True)
        if new.empty:continue
        if new.group.duplicated().any():raise ValueError('Duplicate recording identity.')
        clusters=new.group.map(mapping).fillna(new.group).to_numpy(str)
        for task,truth,column,acceptcol in [
            ('species',new.label,'predicted_species','accepted_species'),
            ('genus_marginalized',new.true_genus,'genus_marginalized','accepted_genus'),
            ('genus_posthoc',new.true_genus,'genus_posthoc',None)]:
            predictions={};accepted={}
            if scope=='postfreeze_new_v4':
                key={'species':'species','genus_marginalized':'genus','genus_posthoc':'posthoc'}[task]
                for version in ['v1','v2','v3']:
                    old=oldnew[oldnew.version.eq(version)].set_index('group').reindex(new.group)
                    if old[key].isna().any():raise ValueError('Missing frozen historical new-file prediction.')
                    predictions[version]=old[key].to_numpy()
                    accepted[version]=old[acceptcol].to_numpy(bool) if acceptcol else np.ones(len(new),bool)
            else:
                old=baseline[baseline.scope.eq(scope)&baseline.task.eq(task)].set_index('group').reindex(new.group)
                if old.label.isna().any() or not np.array_equal(old.label.to_numpy(),np.asarray(truth)):
                    raise ValueError('Historical pairing/labels differ: '+scope+' '+task)
                for version in ['v1','v2','v3']:
                    predictions[version]=old['prediction_'+version].to_numpy()
                    accepted[version]=old['accepted_'+version].to_numpy(bool)
            predictions['v4']=np.where(new.quality_ok,new[column],'INVALID')
            accepted['v4']=new[acceptcol].to_numpy(bool) if acceptcol else new.quality_ok.to_numpy(bool)
            intervals[scope+'|'+task]=dict(
                v1_reference=bootstrap_metrics(truth,predictions,accepted,seed=SEED,clusters=clusters),
                v3_reference=bootstrap_metrics(truth,{v:predictions[v] for v in ['v3','v4']},
                                               {v:accepted[v] for v in ['v3','v4']},seed=SEED,clusters=clusters))
            for version,pred in predictions.items():
                m=classification_metrics(truth,pred,accepted[version]);m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
                rows.append(dict(scope=scope,task=task,version=version,unit='recording',**m))
                labels=np.union1d(np.unique(truth),np.unique(pred));cm=confusion_matrix(truth,pred,labels=labels)
                pd.DataFrame(cm,index=labels,columns=labels).to_csv(matrices/f'{scope}_{task}_{version}.csv')
                for taxon in labels:
                    y=np.asarray(truth);tp=int(((y==taxon)&(pred==taxon)).sum());support=int((y==taxon).sum());count=int((pred==taxon).sum())
                    taxa.append(dict(scope=scope,task=task,version=version,taxon=taxon,support=support,correct=tp,predicted=count,
                        recall=tp/support if support else None,precision=tp/count if count else None))
            for i,g in enumerate(new.group):
                pairs.append(dict(scope=scope,task=task,group=g,pcm_block=clusters[i],label=np.asarray(truth)[i],
                    **{'prediction_'+v:predictions[v][i] for v in predictions},**{'accepted_'+v:bool(accepted[v][i]) for v in accepted}))
            if acceptcol:
                m=classification_metrics(truth,predictions['v4'],new[acceptcol+'_gated'])
                m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
                rows.append(dict(scope=scope,task=task+'_presence_gate',version='v4',unit='recording',**m))
        m=classification_metrics(new.true_genus,np.where(new.quality_ok,new.genus_raw_sum,'INVALID'),new.quality_ok)
        m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
        rows.append(dict(scope=scope,task='genus_raw_sum',version='v4',unit='recording',**m))
    for scope,f in all_scopes:
        y=np.where(f.label.eq('noise'),'noise','mosquito');pred=np.where(f.quality_ok,np.where(f.detected,'mosquito','noise'),'INVALID')
        m=classification_metrics(y,pred,f.quality_ok);m['predicted_taxa_outside_support']='|'.join(m['predicted_taxa_outside_support'])
        positive=f.label.ne('noise').to_numpy()
        m.update(positive_groups=int(positive.sum()),negative_groups=int((~positive).sum()),
            sensitivity=float(np.mean(pred[positive]=='mosquito')) if positive.any() else None,
            specificity=float(np.mean(pred[~positive]=='noise')) if (~positive).any() else None,
            precision_informative=bool(positive.any() and (~positive).any()))
        rows.append(dict(scope=scope,task='presence',version='v4',unit='recording',**m))
    metrics=pd.DataFrame(rows);taxa=pd.DataFrame(taxa)
    metrics.to_csv(OUT/'comparison_metrics.csv',index=False);taxa.to_csv(OUT/'per_taxon.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'paired_recording_predictions.csv',index=False)
    atomic_json(OUT/'bootstrap_intervals.json',intervals)
    atomic_json(OUT/'comparison_audit.json',dict(completed_utc=now(),exact_v2_v3_v4_partitions=True,
        historical_pairs_sha256=file_hash(V3/'paired_recording_predictions.csv'),no_retuning=True,
        historical_files=verify_history(),release_allowed=False))
    generate_report(metrics,taxa,intervals,OUT,freeze)
    verify_freeze();verify_history()
    print(metrics[metrics.task.isin(['species','genus_marginalized'])].to_string(index=False))


def generate_report(metrics,taxa,intervals,out,freeze):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=ROOT/'reports/ensemble_especies_v4_2026_10_14.md'
    if target.exists():raise FileExistsError('Never overwrite a completed report.')
    figures=out/'figures';figures.mkdir(exist_ok=False)
    tests=json.loads((out/'test_validation.json').read_text())
    novel=json.loads((out/'fresh_selection_v4.json').read_text())
    history=json.loads((out/'historical_hashes_after_evaluation.json').read_text())
    pct=lambda x:'indefinido' if pd.isna(x) else f'{100*x:.2f}%'
    scopes=['dryad','external','fresh','postfreeze_new','postfreeze_new_v4','drive']
    table=['| Base | Nível | Versão | n | Acurácia | Balanceada | F1 suporte | F1 união estrita | Cobertura | Acerto aceitas |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope in scopes:
        for task in ['species','genus_marginalized']:
            for _,r in metrics.query('scope==@scope and task==@task').iterrows():
                table.append(f'| {scope} | {task} | {r.version} | {int(r.n)} | {pct(r.accuracy)} | {pct(r.balanced_accuracy)} | {pct(r.macro_f1_support_only)} | {pct(r.macro_f1_strict_union)} | {pct(r.coverage)} | {pct(r.accepted_accuracy)} |')
    curves=['| Dobra | MIL espécie | γ | τ | Máximo nulo | Limiar gênero | T mistura | Força prior |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for fold in ['0','1','2','final']:
        c=json.loads((out/f'calibration_{fold}.json').read_text());g=c['genus_parameters'];p=c['parameters']
        curves.append(f'| {fold} | {c["pooling"]["top_fraction"]:.2f} | {g["gamma"]:.2f} | {g["tau"]:.2f} | {g["uniform_null_max"]:.6f} | {c["threshold_genus"]["threshold"]:.6f} | {p["blend_temperature"]:.2f} | {p["prior_strength"]:.2f} |')
    delta=['| Base | Nível | Δ acurácia v4−v3 (p.p.) | IC95% pareado | Δ F1 união estrita (p.p.) | IC95% pareado |', '|---|---|---:|---|---:|---|']
    for scope in scopes:
        for task in ['species','genus_marginalized']:
            b=intervals[scope+'|'+task]['v3_reference'].get('paired_deltas',{}).get('v4',{})
            sub=metrics.query('scope==@scope and task==@task').set_index('version');values=[]
            for name in ['accuracy','macro_f1_strict_union']:
                bounds=b.get(name,{})
                values.extend([f'{100*(sub.loc["v4",name]-sub.loc["v3",name]):+.2f}',
                    f'[{100*bounds["low"]:+.2f}; {100*bounds["high"]:+.2f}]' if bounds.get('low') is not None else 'não estimável'])
            delta.append('| '+scope+' | '+task+' | '+' | '.join(values)+' |')
    aedes=['| Base | n Ae. aegypti | Acertos v1 | Acertos v2 | Acertos v3 | Acertos v4 |', '|---|---:|---:|---:|---:|---:|']
    genus=['| Base | Versão | Gênero pós-hoc | Soma exata v4 | Gênero principal |', '|---|---|---:|---:|---:|']
    for scope in scopes:
        a=taxa.query('scope==@scope and task=="species" and taxon=="Aedes aegypti"').set_index('version')
        if len(a)==4:aedes.append(f'| {scope} | {int(a.loc["v4","support"])} | '+' | '.join(str(int(a.loc[v,'correct'])) for v in ['v1','v2','v3','v4'])+' |')
        for version in ['v1','v2','v3','v4']:
            m=metrics.query('scope==@scope and version==@version').set_index('task')
            genus.append(f'| {scope} | {version} | {pct(m.loc["genus_posthoc","accuracy"])} | {pct(m.loc["genus_raw_sum","accuracy"]) if version=="v4" else "—"} | {pct(m.loc["genus_marginalized","accuracy"])} |')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,scope in zip(axes,['dryad','external']):
        for j,task in enumerate(['species','genus_marginalized']):
            for i,(version,color) in enumerate(zip(['v1','v2','v3','v4'],['#89909b','#b57950','#25816e','#455dc5'])):
                row=metrics.query('scope==@scope and task==@task and version==@version').iloc[0]
                x=j+(i-1.5)*.18;ax.bar(x,100*row.accuracy,.17,color=color,label=version if j==0 else None)
                ci=intervals[scope+'|'+task]['v1_reference']['intervals'][version]['accuracy']
                ax.vlines(x,100*ci['low'],100*ci['high'],color='#303030')
        ax.set_xticks([0,1],['Espécie','Gênero']);ax.set_ylim(0,105);ax.spines[['top','right']].set_visible(False)
        ax.set_title('Dryad — 569 gravações' if scope=='dryad' else 'D3 — 17 arquivos conhecidos')
    axes[0].set_ylabel('Acurácia por gravação (%)');axes[1].legend(frameon=False)
    fig.suptitle('v1 / v2 / v3 / v4 — IC95% por blocos PCM');fig.tight_layout()
    fig.savefig(figures/'comparison_accuracy.png',dpi=180);fig.savefig(figures/'comparison_accuracy.pdf');plt.close(fig)
    external=metrics.query('scope=="external" and version=="v4"').set_index('task')
    drive=pd.read_csv(out/'external_recording_predictions.csv').query('scope=="drive"').iloc[0]
    content=f'''# Ensemble de espécies v4 — 14/10/2026

## Resultado e decisão

Rodada offline concluída. Nos mesmos 17 arquivos BioDCASE D3: espécie **{int(external.loc['species','correct'])}/17**, gênero corrigido **{int(external.loc['genus_marginalized','correct'])}/17**, gênero pós-hoc **{int(external.loc['genus_posthoc','correct'])}/17**. Cobertura e acerto nas emissões estão abaixo; escolha forçada não equivale a detecção. **Confiabilidade externa não estabelecida; sem liberação para Arduino.** Nenhum hiperparâmetro foi alterado por resultados externos.

As quatro alterações são uma hipótese conjunta motivada por testes externos já examinados. Este contraste não isola causas: não prova causalmente que cada gargalo é um bug ou que sua correção recuperará Ae. aegypti. Soma exata de probabilidades não é matematicamente errada por existir mais espécies de Anopheles; seu prior e calibração podem ser inadequados à aplicação. A v4 muda esse prior e a regra de agregação, com seleção interna explícita.

## Preservação e isolamento

Os **{history['files']} arquivos históricos** registrados antes da implementação mantiveram SHA-256. verify_freeze verificou v1/v2/v3/v4; verify_history inclui resultados, relatórios, firmware e fontes anteriores. O diretório v2 real permanece `results/species_nested_2026_10_13/`. README.md e MODEL_VALIDATION.md também são históricos congelados: conforme a escolha expressa do usuário, permanecem intactos; as atualizações são README_v4.md e MODEL_VALIDATION_v4.md. Nenhum firmware, header ou instalador foi alterado/compilado.

Reutilizamos exatamente as partições v2/v3: 24.405 janelas, 613 grupos, 560 blocos PCM, 20 espécies. Fit, calibration e outer test são disjuntos por bloco, grupo, caminho e colisão auditada; scaler e pesos espécie/aparelho/bloco usam somente fit. As bases são SVM RBF, ExtraTrees e HistGradientBoosting; parâmetros-base mantidos. Nenhuma base é reajustada com calibration. A v1 tem outro isolamento e colisões já documentadas: não é ablação equivalente.

## Frontend e teste de piso estacionário

PSD usa `10log10(power/local_median_power)`, limitado a [−6,+45] dB, interpolado e centralizado pela mediana. Mel usa `MEL_BANK_SHORT @ (short/local_floor)`, log com piso 1e−6, mediana por frame e DCT, com c0=0 e c1/c2 preservados. Não se tira log de uma subtração linear truncada em zero. Energia harmônica com piso subtraído permanece apenas nos descritores físicos de energia; não alimenta os logs PSD/Mel. STFTs 100/200 ms mantêm resolução temporal e bins 10/5 Hz.

Toda transformação opera dentro de uma única janela real de 1 s. O teste SNR 25→12 dB verifica forma PSD/MFCC (cosseno >.97), RMS de alteração do PSD <3 dB, fundamental dentro de 5 Hz e saliência dentro de .1, em sinais de 400 e 625 Hz. Isso não é invariância absoluta a qualquer ruído: elevar ruído altera a razão sinal/piso e as amplitudes cepstrais. A demonstração de 400 Hz deu cossenos PSD≈.9922/MFCC≈.9965 e alteração relativa de amplitude do bloco MFCC≈39.7%. Em 625 Hz o cosseno MFCC≈.9776 reprova um critério mais forte >.98; mantivemos esse caso no teste e registramos a fronteira, sem alterar a fórmula para selecioná-lo favoravelmente. Não se anuncia a amplitude como invariável. Ganho escalar, silêncio e independência bit a bit de outras janelas têm testes próprios. Canal arbitrário, ruído colorido e enxames múltiplas espécies continuam limitações.

## Calibração de espécie, gênero e MIL

Pesos da mistura são ajustados por log-loss balanceada em calibration. Temperatura pós-mistura [0.75,1,1.25,1.5,2] e força [0,.25,.5,.75,1] de log-prior inverso de frequência são selecionadas exclusivamente em recordings de calibration. Frequência é contagem de blocos PCM distintos por espécie da própria calibration; viés absoluto ≤.8 (fator máximo exp(.8)). Objetivo: média de balanced accuracy e soft macro-F1 ponderado por espécie/bloco, em MIL provisório .35; desempate por log-loss balanceada e menor viés. Isso limita prevalência, sem atribuir privilégio manual a Aedes. Espécies com 1 bloco de calibração continuam frágeis.

Gênero corrigido usa `S_G=(sum(p_s**tau)/m_G**gamma)**(1/tau)` e normalização. γ∈[0,.25,.5,.75,1], τ∈[1,1.5,2,3]. Objetivo interno: `.5*(balanced_accuracy + exp(-balanced_log_loss)) - .05*KL(null || uniform_genera)`. O exponencial torna o termo de log-loss um escore crescente de qualidade; não se soma log-loss positiva a acurácia. Candidatos cuja distribuição de 20 espécies uniformes produza máximo ≥.30 são inelegíveis. γ=0,τ=1 recupera soma exata, mas seu nulo .60 viola a restrição deste catálogo. A distribuição normaliza a 1, porém **não é o posterior exato marginalizado** fora desse caso especial; o nome da coluna histórica é mantido para pareamento.

Limiar de gênero maximiza acurácia aceita×√cobertura, exigindo acurácia interna ≥90% e ≥20 blocos PCM aceitos, piso .30; sem candidato qualificado rejeita tudo. Espécie conserva a regra de rejeição interna anterior. Seleção de temperatura/prior/MIL/taxonomia/limiar reutiliza calibration e pode ser otimista; outer test e externos ficam excluídos de toda seleção.

MIL seleciona [.20,.35,.50,.65] em calibration; gravações com >4 janelas não podem selecionar todas. Peso de espécie por janela = saliência×(1−H/logK)×(.15+.85×presença calibrada). Ranking usa esse peso, qualidade exclui voto inválido e nitidez numericamente uniforme é zero. Até quatro janelas usam todas. Presença é treinada e calibrada separadamente e agregada sem autoponderação de presença; também tem seleção MIL interna e controle FPR interno. A regra reduz influência de fundos uniformes; **não garante** rejeitar ruído que o próprio modelo reconheça erroneamente com alta confiança.

{chr(10).join(curves)}

## Congelamento, aquisição e avaliação

Todos os modelos, fontes, transformações e limiares foram congelados em **{freeze['created_utc']} UTC**, SHA-256 de freeze.json `{file_hash(out/'freeze.json')}`, antes de qualquer predição outer OOF ou externa. Manifesto registra pesos, códigos, entradas e configurações; evaluation_started.json/evaluation_audit.json registram a sequência. Extração prévia determinística não ajusta estatísticas do teste.

17 externos, fresh da v2 (83/84), postfreeze_new da v3 (85/86) e Drive são conhecidos, retrospectivos. Depois do congelamento, busca oficial e seleção numérica determinística adquiriram **{len(novel['accepted_files'])} arquivos** separados em postfreeze_new_v4: {', '.join(Path(x['path']).name for x in novel['accepted_files'])}. SHA/CRC/tamanho/duração e PCM inteiro/fragmentos alinhados de 1/.992/.3 s a 8 kHz, com variantes de ganho, foram conferidos antes de predição, incluindo todos os WAVs anteriores em results. Erros de decodificação e limites estão em fresh_selection_v4.json. Não se substituiu candidato por resultado. Esse corpus familiar não estabelece novos indivíduos/sessões nem uma campanha independente; auditoria alinhada não exclui todos os derivados arbitrários.

## Comparação completa por gravação

{chr(10).join(table)}

![Comparação das quatro versões](../results/species_nested_v4_2026_10_14/figures/comparison_accuracy.png)

São métricas taxonômicas sem gate para parear com históricos; CSV também inclui gate de presença e soma bruta de gênero v4. Inválidos e rejeições permanecem nos denominadores; acerto aceito sem emissões fica indefinido. F1 suporte usa apenas táxons verdadeiros; F1 união inclui explicitamente os previstos fora do suporte. Coberturas diferentes representam subconjuntos distintos e não autorizam comparar precisão como se fossem as mesmas emissões.

## Diferenças pareadas

{chr(10).join(delta)}

IC95% percentis, 2.000 réplicas, seed 20261014, sorteios pareados por bloco PCM estratificados pelo conjunto de rótulos verdadeiro. A união para F1 é recalculada em cada réplica. Réplicas sem emissões não viram acerto zero; sua quantidade válida é registrada. Estratos pequenos e métricas constantes podem gerar intervalos degenerados, sem precisão populacional demonstrada. Drive tem uma gravação, sem IC por janelas. Todas as comparações são exploratórias, sem correção de multiplicidade.

## Ae. aegypti e as três regras de gênero

{chr(10).join(aedes)}

{chr(10).join(genus)}

O diagnóstico taxonômico de v3 favorecia Anopheles sob incerteza difusa; v4 mede separadamente soma exata, pós-hoc e agregação corrigida. Per_taxon.csv e matrizes trazem todos os táxons, incluindo falsos positivos fora do suporte. Acurácia total não substitui sensibilidade de Ae. aegypti nem prova efeito causal individual das quatro alterações.

## Presença, Drive e limites

O Drive tem {drive.duration_s:.2f} s, {int(drive.complete_windows)} janelas completas, {int(drive.invalid_windows)} inválidas e {drive.unanalysed_tail_s:.2f} s finais não completados/padronizados. MIL votou com {int(drive.selected_windows)} janelas selecionadas. Candidata v4 **{drive.predicted_species}**; gênero corrigido **{drive.genus_marginalized}**; presença {drive.detected}; espécie aceita {drive.accepted_species}; gênero aceito {drive.accepted_genus}. Diagnóstico por janela: {int(drive.forced_correct_windows)}/{int(drive.complete_windows)} escolhas corretas, sem IC independente. Referência do usuário: enxame Ae. aegypti, sem confirmação entomológica/anotação independente. O WAV privado não foi publicado.

Presença por recording está em comparison_metrics.csv, separada de gênero/espécie, com sensibilidade/especificidade e quantidade de positivos/negativos. Externos somente positivos não medem especificidade/precisão de presença. Os critérios de MODEL_VALIDATION exigem ≥60% e limites inferiores apropriados por táxon/entre mosquitos em aquisição adequada, além de confirmação independente no cenário pretendido. Esses requisitos de evidência não são satisfeitos por uma amostra pequena e familiar ou por um enxame isolado. Não houve distribuição embarcada.

## Testes e artefatos

Linha de base **177 passed**. Suíte final **{tests['summary']}**, status {tests['exit_code']}, sem alteração de testes anteriores. Os novos testes cobrem piso logarítmico/ruído estacionário, ganho, c0/c1/c2, bit-indução, colisões PCM, fórmula taxonômica e nulo abaixo do piso, MIL seletivo/presença, prior limitado, histórico, scaler-fit e cronologia congelamento/aquisição/avaliação. Logs preservam avisos conhecidos.

Fontes novas: acoustic_features_v4.py, nested_species_ensemble_v4.py, fetch_species_v4_fresh.py, compare_species_v4.py e tests/test_nested_species_v4.py. Resultados, curvas, previsões por janela/gravação, bootstrap, matrizes, figuras e manifestos ficam somente em results/species_nested_v4_2026_10_14/. Relatório histórico algum foi reescrito.
'''
    target.write_text(content)
    for name,title in [('README_v4.md','Atualização científica v4'),('MODEL_VALIDATION_v4.md','Adendo de validação v4')]:
        destination=ROOT/name
        if destination.exists():raise FileExistsError('Do not overwrite a versioned documentation update.')
        destination.write_text(f'# {title} — 14/10/2026\n\nOs documentos originais permanecem congelados, conforme a escolha do usuário.\n\n'+
            f'[Relatório completo](reports/ensemble_especies_v4_2026_10_14.md) · [Resultados auditados](results/species_nested_v4_2026_10_14/)\n\n'+
            '**Confiabilidade externa não estabelecida; versão offline, sem autorização para Arduino.**\n\n'+
            chr(10).join(table)+'\n\n'+chr(10).join(delta)+'\n\n'+chr(10).join(aedes)+'\n\n'+chr(10).join(genus)+'\n\n'+
            'Preservação histórica, limitações de SNR/taxonomia, isolamento PCM, seleção interna, congelamento pré-avaliação e aquisição inédita estão documentados no relatório.\n')


if __name__=='__main__':main()
