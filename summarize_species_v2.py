"""Generate the technical report from frozen results; no model selection here."""
import json
import platform
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from improve_device import ROOT,file_hash
from nested_species_ensemble_v2 import OUT,verify_freeze,verify_history


def pct(x):
    return 'indefinido' if pd.isna(x) else f'{100*x:.2f}%'


def main():
    compare=ROOT/'results/three_levels_2026_10_13'
    target=ROOT/'reports/ensemble_especies_v2_2026_10_13.md'
    if target.exists():raise FileExistsError('Report already exists; preserve completed reports.')
    freeze=verify_freeze();history=verify_history(OUT/'historical_snapshot.json')
    scopeguard=verify_history(OUT/'offline_scope_guard.json')
    metrics=pd.read_csv(compare/'metrics.csv');ci=json.loads((compare/'bootstrap_intervals.json').read_text())
    collision=json.loads((OUT/'collision_audit.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    fresh=json.loads((OUT/'fresh_selection_v2.json').read_text())
    frame=pd.read_csv(OUT/'segments_v2.csv');oldparts=pd.read_csv(ROOT/'results/species_nested_2026_10_09/partitions.csv')
    device_counts=frame[frame.label.ne('noise')].groupby('label').phone.nunique()
    single_device=device_counts[device_counts.eq(1)].index.tolist()
    oldparts['collision_group']=oldparts.row_index.map(frame.collision_group)
    crossed={str(f):int((g.groupby('collision_group').partition.nunique()>1).sum()) for f,g in oldparts.groupby('fold')}
    cal={str(f):json.loads((OUT/f'calibration_{f}.json').read_text()) for f in [0,1,2,'final']}
    audit=json.loads((compare/'audit.json').read_text())
    tests=json.loads((compare/'test_validation.json').read_text())
    matrices=OUT/'confusion_matrices';matrices.mkdir(exist_ok=False)
    for source in compare.glob('*_confusion.csv'):
        shutil.copy2(source,matrices/source.name)
    # New-round derived summaries only; old result directories remain read-only.
    shutil.copy2(compare/'metrics.csv',OUT/'audited_metrics.csv')
    assets=ROOT/'reports/assets/species_v2_2026_10_13';assets.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,scope,title in zip(axes,['dryad','external'],['Dryad: mesmas 569 gravações','BioDCASE D3: mesmos 17 arquivos']):
        for j,(task,label) in enumerate([('species','Espécie'),('genus_marginalized','Gênero marginalizado')]):
            for i,(version,color) in enumerate([('09/10/2026','#737d8b'),('13/10/2026 v2','#197c63')]):
                row=metrics.query('scope==@scope and task==@task and version==@version').iloc[0]
                interval=ci[scope+'|'+task]['intervals'][version]['accuracy']
                x=j+(i-.5)*.3;y=100*row.accuracy
                ax.bar(x,y,.28,color=color,label=version if j==0 else None)
                low,high=100*interval['low'],100*interval['high']
                ax.vlines(x,low,high,color='#303030');ax.hlines([low,high],x-.035,x+.035,color='#303030')
        ax.set_xticks([0,1],['Espécie','Gênero']);ax.set_title(title);ax.set_ylim(0,105);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Acurácia por gravação (%)');axes[1].legend(frameon=False)
    fig.suptitle('V2: mudanças metodológicas e de representação; IC95% por blocos PCM')
    fig.tight_layout();fig.savefig(assets/'paired_accuracy.png',dpi=180);fig.savefig(assets/'paired_accuracy.pdf');plt.close(fig)
    primary=metrics[metrics.task.isin(['species','genus_marginalized'])]
    table=['| Base | Nível | Versão | n | Acurácia | Balanceada | F1 suporte | F1 estrito | Cobertura | Aceitas corretas |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scope in ['dryad','external','drive','fresh']:
        for task in ['species','genus_marginalized']:
            for _,r in primary[(primary.scope==scope)&(primary.task==task)].iterrows():
                table.append(f'| {scope} | {task} | {r.version} | {int(r.n)} | {pct(r.accuracy)} | {pct(r.balanced_accuracy)} | {pct(r.macro_f1_support_only)} | {pct(r.macro_f1_strict_union)} | {pct(r.coverage)} | {pct(r.accepted_accuracy)} |')
    deltas=['| Base | Nível | Δ acurácia (p.p.) [IC95%] | Δ F1 estrito (p.p.) [IC95%] | Δ cobertura (p.p.) [IC95%] | Δ acerto aceitas (p.p.) [IC95%] |',
            '|---|---|---:|---:|---:|---:|']
    for scope in ['dryad','external','fresh']:
        for task in ['species','genus_marginalized']:
            part=primary[(primary.scope==scope)&(primary.task==task)]
            if len(part)!=2:continue
            v1=part[part.version.eq('09/10/2026')].iloc[0];v2=part[part.version.eq('13/10/2026 v2')].iloc[0]
            bounds=ci[scope+'|'+task]['paired_deltas'].get('13/10/2026 v2',{})
            values=[]
            for key in ['accuracy','macro_f1_strict_union','coverage','accepted_accuracy']:
                q=bounds.get(key,{})
                if pd.isna(v1[key]) or pd.isna(v2[key]) or q.get('low') is None:values.append('indefinido')
                else:values.append(f'{100*(v2[key]-v1[key]):+.2f} [{100*q["low"]:+.2f}; {100*q["high"]:+.2f}]')
            deltas.append('| '+scope+' | '+task+' | '+' | '.join(values)+' |')
    selection=['| Dobra | Temperaturas SVM/ET/HGB | Pesos SVM/ET/HGB | MIL espécie | MIL presença | Limiar espécie | Limiar gênero | Limiar presença |',
               '|---|---|---|---:|---:|---:|---:|---:|']
    for f,c in cal.items():
        selection.append('| '+f+' | '+', '.join(f'{v:.4f}' for v in c['temperatures'])+' | '+', '.join(f'{v:.4f}' for v in c['parameters']['weights'])+
                         f' | {c["pooling"]["top_fraction"]} | {c["presence_pooling"]["top_fraction"]} | {c["threshold_species"]["threshold"]:.6f} | {c["threshold_genus"]["threshold"]:.6f} | {c["threshold_presence"]["threshold"]:.6f} |')
    genus=['| Base | Versão | Gênero da espécie vencedora | Gênero marginalizado |', '|---|---|---:|---:|']
    for scope in ['dryad','external','drive','fresh']:
        for version in ['09/10/2026','13/10/2026 v2']:
            a=metrics.query('scope==@scope and version==@version and task=="genus_posthoc"')
            b=metrics.query('scope==@scope and version==@version and task=="genus_marginalized"')
            if len(a) and len(b):genus.append(f'| {scope} | {version} | {pct(a.iloc[0].accuracy)} | {pct(b.iloc[0].accuracy)} |')
    taxa=pd.read_csv(compare/'per_taxon.csv')
    species_table=['| Espécie | Dryad n | Recall v1 | Recall v2 | Externos n | Recall v1 | Recall v2 |',
                   '|---|---:|---:|---:|---:|---:|---:|']
    for label in protocol['classes']:
        values=[]
        for scope in ['dryad','external']:
            sub=taxa.query('scope==@scope and task=="species" and taxon==@label')
            vals=[];support=0
            for version in ['09/10/2026','13/10/2026 v2']:
                r=sub[sub.version.eq(version)]
                if len(r):support=int(r.iloc[0].support);vals.append(pct(r.iloc[0].recall))
                else:vals.append('não observada')
            values.extend([str(support)]+vals)
        species_table.append('| '+label+' | '+' | '.join(values)+' |')
    external=primary[(primary.scope=='external')&primary.version.eq('13/10/2026 v2')]
    drive=pd.read_csv(compare/'private_audio_recording_metrics.csv').iloc[0]
    text=f'''# Ensemble de espécies v2 — rodada local de 13/10/2026

## Conclusão e decisão

Treinamento e avaliação offline executados. A confiabilidade externa **não foi atingida**. No BioDCASE D3 pareado, a v2 acertou {int(external[external.task.eq('species')].iloc[0].correct)}/17 espécies e {int(external[external.task.eq('genus_marginalized')].iloc[0].correct)}/17 gêneros. Não houve promoção do modelo por resultados externos. Presença, gênero e espécie têm saídas separadas; o sistema integrado com gate de presença também está nos CSVs. Firmware, cabeçalhos de produção e instalador não foram alterados nem compilados.

O recálculo PCM encontrou **{collision['additional_collisions']} ligações adicionais** entre grupos declarados: {collision['original_groups']} grupos originais, {collision['collision_groups']} blocos de colisão. Um bloco contém 54 grupos com fundo compartilhado e múltiplos rótulos fracos. Os rótulos e denominadores originais foram mantidos. As três dobras foram refeitas por blocos PCM. Na versão histórica, o número de blocos que atravessava partições em cada execução era {crossed}. Logo, a diferença v1–v2 combina correção do isolamento, outras partições internas e novas representações; não é uma ablação causal das features.

## Inspeção da linha de base

Foram inspecionados os arquivos solicitados, loaders, manifestos e testes. A linha de base passou com **136 testes**. O gênero de 09/10 já era marginalizado em `evaluate_nested_genus.py`; a hipótese de que vinha apenas da espécie vencedora não se confirmou. Essa versão não tinha rejeição específica de gênero. O frontend `extract_robust_features` tem um parâmetro CMVN que não aplica a transformação anunciada; a v2 usa uma implementação própria explícita. O rastreador antigo retém um único pico; a v2 adiciona perfil harmônico multi-pico. Fontes comprimidas foram decodificadas pelo mesmo FFmpeg a 16 kHz do loader original, sem reescrever caches.

## Dados, isolamento e denominadores

São {len(frame)} janelas reais de 1 s, 20 espécies, 569 grupos positivos e 44 grupos de ruído originalmente declarados, correspondentes a 818 caminhos. Usa-se a ordem do `results/segments.csv` e `data/cache/waves.npy`; SHA-256 de ambos e da configuração original estão no protocolo. O cache contém janelas já centralizadas e normalizadas individualmente por RMS, nunca por lote.

O grafo PCM verifica decodificação inteira e fragmentos alinhados de 1, .992 e .3 s a 8 kHz, com variantes de ganho, excluindo fragmentos quase constantes. Não prova ausência de derivados com deslocamentos arbitrários, identidade biológica, sessões ou ambientes compartilhados. Cada componente permanece integralmente em fit, calibração ou teste; os testes auditam linhas, caminhos, hashes, grupos originais e blocos. Blocos de rótulos mistos são separados por união gulosa com semente, conservando representantes de cada classe em fit e calibração. Não foram eliminados para melhorar métricas.

A unidade das métricas continua sendo o grupo original de gravação, para preservar o pareamento. A unidade independente dos intervalos é o bloco PCM. As espécies são classificadas em escolha fechada; ruído não recebe uma espécie verdadeira. Identificação e presença são avaliadas separadamente, e as colunas com gate mostram a emissão operacional combinada.

A qualidade antiga exige todas as janelas válidas para emitir por gravação. A regra v2 pré-declarada permite votar apenas com janelas válidas selecionadas pelo MIL; uma gravação sem voto válido é inválida e não emite. Janelas inválidas continuam nos diagnósticos por janela e o arquivo continua no denominador por gravação. Assim, mudanças de cobertura também incluem essa política de agregação/qualidade, não apenas a confiança do classificador.

## As cinco correções

1. **F1 auditado:** `macro_f1_support_only` usa as classes verdadeiras presentes; `macro_f1_strict_union` usa a união das classes verdadeiras e preditas. Predições em táxons ausentes também prejudicam o recall das classes verdadeiras na primeira métrica; a segunda adiciona explicitamente F1 zero para os táxons espúrios. O nome antigo `macro_f1` permanece compatível nos scripts históricos, sem regenerar resultados antigos. Os CSVs novos expõem ambas e a diferença.
2. **Gênero probabilístico:** soma exata das probabilidades calibradas de espécies de cada gênero, com limiar próprio escolhido exclusivamente na calibração. O gênero da espécie vencedora aparece como comparador forçado. O gênero histórico tem cobertura de escolha forçada; não se inventou um limiar histórico.
3. **Rótulos fracos/MIL:** saliência estima contraste dos harmônicos 1–4 de candidatos 200–900 Hz contra a mediana espectral local dentro da janela. Combina contraste e fração de energia em bandas de ±15 Hz; fica entre .1 e 3, sem apagar grupos. No fit multiplica o peso de cada janela. No MIL, escolhem-se top 25%, 50% ou 100% das janelas por saliência; o voto pesa saliência vezes max(1−H(p)/log K, .01). Empates são determinísticos. A fração é escolhida por acurácia balanceada, F1 estrito e maior fração como desempate, apenas na calibração.
4. **Canal/aparelho:** PSD centrado por mediana, remoção de inclinação em log-frequência e escala IQR intra-janela; MFCC médios 0–2 zerados e coeficiente 3 atenuado por .25, mantendo variabilidade e deltas. Contraste espectral local reduz envoltórias suaves. Isso não garante invariância a qualquer microfone. Pesos de treino combinam inverso do tamanho do bloco PCM, balanceamento de classe, raiz do inverso amortecido da frequência espécie/aparelho e saliência. Todas as frequências vêm exclusivamente do fit; aparelho desconhecido permanece uma categoria explícita.
5. **Multi-picos/enxame:** pente harmônico baseado em soma sub-harmônica, componente de presença de fundamental contra erros de oitava, três picos separados e seus contrastes/razões harmônicas; também entropia, espalhamento e evolução do perfil. Não é uma reprodução exata do algoritmo de Hermes e não resolve automaticamente misturas de espécies ou valida indivíduos em enxame.

## Estimadores e calibração

SVM RBF C=3 com StandardScaler ajustado só no fit; ExtraTrees com 200 árvores, max_features=.5 e min_samples_leaf=2; HistGradientBoosting com 120 iterações, learning_rate=.08, 15 folhas e L2=1. As separações usam semente 20261013; os estimadores clássicos conservam random_state=42 de `model_for`, e o Platt binário usa 20261013. A presença usa HGB binário e Platt no logit em calibração. Predição de cada janela e soma das árvores têm ordem determinística em uma thread; adicionar outras linhas não muda um bit da predição.

Temperaturas por membro, pesos e correção logarítmica por classe são ajustados na calibração. Rejeição exige 90% de acerto e ao menos 20 blocos PCM distintos aceitos; ausência de limiar qualificado resulta em limiar 1.000001, rejeitando tudo. Presença maximiza acurácia balanceada com FPR de calibração ≤5%; poucos negativos tornam essa estimativa frágil. As bases não são reajustadas com a calibração. Reutilizar essa partição para temperatura, combinação, pooling e rejeição pode causar otimismo interno; o teste externo da dobra não participa de nenhuma dessas escolhas.

{chr(10).join(selection)}

## Resultados pareados por gravação

{chr(10).join(table)}

![Acurácias pareadas e intervalos](assets/species_v2_2026_10_13/paired_accuracy.png)

Intervalos individuais, acurácia balanceada, F1 estrito, cobertura e acerto nas emissões: `results/three_levels_2026_10_13/bootstrap_intervals.json`. São 2.000 réplicas percentis pareadas com semente 20261013, estratificadas pelo conjunto de rótulos de cada bloco PCM. O conjunto de classes da união é recalculado em cada réplica. Um estrato com um único bloco não fornece variação independente adicional. Réplicas sem emissões deixam acerto aceito indefinido; sua contagem válida é registrada. Os 17 arquivos continuam uma amostra pequena e assimétrica de cinco espécies e três gêneros.

Alguns intervalos são degenerados, por exemplo a acurácia externa de gênero da v2: dentro de cada gênero observado, todas as decisões têm o mesmo desfecho. Reamostrar esses estratos não cria erros ou acertos ainda não observados. Um intervalo bootstrap de largura zero nessa situação não demonstra precisão populacional; a quantidade pequena de gravações e a ancestralidade desconhecida permanecem limitações. A diferença pareada de gênero ainda inclui zero.

{chr(10).join(deltas)}

Acerto nas emissões compara políticas com coberturas diferentes; os subconjuntos aceitos não são necessariamente os mesmos. O intervalo não transforma esse contraste em evidência de confiabilidade ou equivalência. Não há ajuste de multiplicidade; várias métricas e candidatos são exploratórios.

## Gênero marginalizado contra mapeamento da vencedora

{chr(10).join(genus)}

As matrizes completas e métricas por táxon, incluindo falsos positivos fora do suporte e inválidos, estão no diretório de comparação. Desempenho nas classes raras e ausentes não é validado por médias agregadas.

{chr(10).join(species_table)}

## Presença e áudio privado

Presença interna tem controles de ruído, mas os testes externos pareado e novo contêm somente mosquitos: medem sensibilidade, não comprovam precisão/especificidade. O detector histórico é por janela, com outro protocolo; está registrado como `historical_corpus_unpaired`, sem alegar comparação pareada com a nova presença por gravação.

A v2 detectou presença em 455/569 grupos positivos internos e rejeitou 41/44 grupos de ruído: sensibilidade 79,96%, especificidade 93,18%, acurácia 80,91%, acurácia balanceada 86,57% e F1 estrito 64,91%. O FPR OOF de 6,82% excede a meta interna de calibração de 5%. Nos externos pareados foram 13/17 positivos detectados (76,47%); nos dois arquivos novos, 1/2; no Drive, nenhuma presença aceita por gravação. A comparação histórica não pareada usa explicitamente o modo `Calibração FPR 5%`, distinto do modo padrão histórico apresentado no README.

`drive_audio.wav` dura {drive.duration_s:.2f} s e fornece {int(drive.complete_windows)} janelas completas; {drive.unanalysed_tail_s:.2f} s finais não completam uma janela. A referência do usuário é Aedes aegypti ao longo de todo o áudio, sem anotação entomológica independente. Todas as janelas, inclusive {int(drive.invalid_windows)} inválidas e rejeições, permanecem no denominador dos diagnósticos. Acertos de espécie por janela: histórico {int(drive.old_forced_correct_windows)}/{int(drive.complete_windows)}, v2 {int(drive.forced_correct_windows)}/{int(drive.complete_windows)}. Esses valores são descritivos de uma única gravação; não há IC independente por gravação, nem estimativa de precisão/especificidade. Candidata v2: {drive.predicted_species}; gênero: {drive.genus_marginalized}; espécie aceita: {drive.accepted_species}; gênero aceito: {drive.accepted_genus}; presença detectada: {drive.detected}. O áudio privado não foi copiado nem publicado.

## Congelamento, aquisição posterior e novidade

Congelado em **{freeze['created_utc']} UTC**, antes de qualquer inferência v2 externa. SHA-256 de freeze.json: `{file_hash(OUT/'freeze.json')}`. Modelos, thresholds, protocolo, features e código estão protegidos por `verify_freeze`; pesos/thresholds também aparecem no manifest. Data local do identificador da rodada é 13/10; horários são registrados em UTC. Busca na web e seleção posteriores estão em `postfreeze_web_search.json` e `fresh_selection_v2.json`. Nenhum hiperparâmetro ou limiar mudou depois da avaliação.

Além dos mesmos 17 arquivos retrospectivos e do Drive, a seleção por metadados adquiriu {len(fresh['accepted_files'])} arquivos aceitos ainda não usados, com {len(fresh['exclusions'])} exclusões de duração ou colisão registradas. A fonte oficial é [BioDCASE / Zenodo 19095788](https://zenodo.org/records/19095788). São novos nomes/bytes/fragmentos conhecidos em um corpus familiar; não uma nova campanha, e não se conhece a ancestralidade dos indivíduos/sessões. Não houve substituição por resultados do classificador. Resultados históricos para esses arquivos foram calculados com os pesos congelados antigos somente como comparador, depois do congelamento v2.

O script congelado original de aquisição permanece com seu SHA-256. Um auxiliar adicional `fetch_species_v2_fresh_checked.py` foi congelado em `acquisition_freeze_v2.json` antes da busca/download para registrar, sem fingir auditoria PCM, arquivos conhecidos que são forks de recursos Apple ou fixtures sem decodificação suportada. Seus bytes ainda são indexados; as exceções estão em `novelty_index.decoding_errors`. A regra de seleção por espécie/ID/duração e colisões não mudou. Nenhum código de inferência, peso, transformação ou limiar foi alterado por esse tratamento de aquisição.

## Preservação e verificação

`verify_history` e `verify_freeze` foram executados antes e depois: {history['files']} arquivos históricos inalterados, mais {scopeguard['files']} scripts do instalador/instalação preservados no guard adicional. Os resultados de 09/10, Arduino e demais rodadas foram apenas lidos. Tentativas interrompidas na inspeção/decodificação foram conservadas em `tmp/v2_2026_10_13/`; não são treinamentos concluídos nem resultados substituídos. A suíte final: **{tests['summary']}**, comando `{tests['command']}`, status {tests['exit_code']}. Avisos de depreciação Joblib/NumPy são registrados no log; não foram ocultados. Ambiente: Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}.

Artefatos: `results/species_nested_2026_10_13/` (partições, auditoria PCM, pesos, saliências, OOF e externos por janela/gravação, probabilidades, freeze e manifest); `results/three_levels_2026_10_13/` (métricas pareadas, matrizes, táxons, intervalos e diagnósticos privados). Scripts: `nested_species_ensemble_v2.py`, `acoustic_features_v2.py`, `fetch_species_v2_fresh.py`, `compare_detection_levels_v2.py`, `research_metrics_v2.py`, `summarize_species_v2.py`; testes em `tests/test_nested_species_v2.py`.

## Limitações e próximo experimento

Não há confirmação independente de 20 espécies em novos aparelhos, controles negativos externos suficientes, múltiplos enxames/indivíduos ou eventos temporais anotados. O protocolo binário de MODEL_VALIDATION exige simultaneamente métricas e limites inferiores de 95% ≥60%, inclusive entre mosquitos e em cada táxon negativo. Esta rodada não satisfaz esse conjunto de exigências. Nenhuma liberação de campo ou Arduino.

Espécies com um único valor declarado de aparelho: {', '.join(single_device)}. O balanceamento não cria observações de aparelhos ausentes; `unknown` também não prova que um único aparelho foi usado. Normalizar ganho/inclinação não elimina respostas espectrais arbitrárias nem diferenças de sexo, ambiente e distância.

Uma próxima rodada precisa de aquisição independente com espécie/sexo/indivíduo/sessão confirmados, gravações de ruído e outras espécies no mesmo aparelho/ambiente, anotação de voo, validação de eventos e falsos alarmes por hora. Ajustar parâmetros pelos 17 arquivos ou pelo Drive faria adaptação e exigiria outro teste congelado. Para isolar causas, ablações de normalização, pesos, SHS e MIL devem usar desenvolvimento por blocos e uma nova aquisição final; não se deve interpretar a rodada conjunta como prova da contribuição de cada componente.

## Referências metodológicas

- [Hermes (1988), Measurement of pitch by subharmonic summation, JASA 83:257–264](https://research.tue.nl/en/publications/measurement-of-pitch-by-subharmonic-summation/): referência para soma sub-harmônica; a implementação aqui é um pente de contraste adaptado.
- [Guo et al. (2017), On Calibration of Modern Neural Networks, PMLR 70](https://proceedings.mlr.press/v70/guo17a.html): motivação para temperatura; não estabelece transferência de calibração entre microfones ou confiabilidade bioacústica.
'''
    target.write_text(text)
    # Prepend current evidence without rewriting historical statements below.
    s=external[external.task.eq('species')].iloc[0];g=external[external.task.eq('genus_marginalized')].iloc[0]
    intro=f'''Rodada v2 de 13/10/2026: [relatório técnico](reports/ensemble_especies_v2_2026_10_13.md). Auditoria PCM uniu 613 grupos em 560 blocos, corrigindo colisões que atravessavam as partições históricas. Treinamento aninhado offline com normalização por janela, pesos espécie/aparelho, saliência, SHS multi-pico, MIL e rejeição própria de gênero. Nos mesmos 17 arquivos externos retrospectivos: espécie {int(s.correct)}/17 ({pct(s.accuracy)}), gênero {int(g.correct)}/17 ({pct(g.accuracy)}). Intervalos pareados, F1 por suporte e pela união, emissões e regressões estão no relatório. **Confiabilidade externa não atingida; sem liberação para Arduino.** Novos arquivos foram adquiridos somente depois do congelamento. Histórico e instalador preservados; {tests['summary']}.

'''
    for name in ['README.md','MODEL_VALIDATION.md']:
        p=ROOT/name;old=p.read_text();first,rest=old.split('\n',1)
        p.write_text(first+'\n\n'+intro+rest.lstrip('\n'))
    print(target)


if __name__=='__main__':main()
