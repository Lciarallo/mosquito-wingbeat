"""Generate an honest comparison report from the completed frozen experiment."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from improve_device import atomic_json,file_hash
ROOT=Path(__file__).resolve().parent;O=ROOT/'results/multisource'
def main():
 m=pd.read_csv(O/'comparison.csv');a=json.loads((O/'training_audit.json').read_text());data=json.loads((O/'data_audit.json').read_text());selection=json.loads((O/'fresh_external/selection.json').read_text());private=json.loads((ROOT/'tmp/private-audio-test/multisource_cnn/analysis.json').read_text());hist=pd.read_json(O/'history.json')
 assert file_hash(O/'models/multisource_cnn.pt')==a['checkpoint_sha256']
 fresh=[]
 for item in selection['files']:
  for model in ['large_cnn','multisource_cnn']:
   x=json.loads((O/f"fresh_external/{item['id']}/{model}/analysis.json").read_text());assert x['model_unchanged']
   fresh.append(dict(id=item['id'],model=model,windows=x['complete_windows'],forced_correct=x['forced_correct_windows'],correct_identifications=x['correctly_identified_windows'],wrong_identifications=x['incorrectly_identified_windows'],presence_candidate_windows=x['presence_candidate_windows']))
 f=pd.DataFrame(fresh);f.to_csv(O/'fresh_external/summary.csv',index=False)
 fs=f.groupby('model').agg(files=('id','nunique'),windows=('windows','sum'),forced_correct=('forced_correct','sum'),correct_identifications=('correct_identifications','sum'),wrong_identifications=('wrong_identifications','sum'))
 fs['files_with_Aedes_identification']=f.assign(hit=f.correct_identifications.gt(0)).groupby('model').hit.sum();fs.to_csv(O/'fresh_external/aggregate.csv')
 decision=dict(status='NOT_RELIABLE_CROSS_SOURCE_AEDES_FAILURE',model_sha256=a['checkpoint_sha256'],additional_training_windows=4344,ufrgs_entirely_held_out=True,ufrgs_Aedes_forced_accuracy=float(m[(m.model=='multisource_cnn')&(m.domain=='ufrgs')].forced_window_accuracy.iloc[0]),ufrgs_correct_identifications=0,fresh_audio_files_after_freeze=5,fresh_files_with_Aedes_identification=int(fs.loc['multisource_cnn','files_with_Aedes_identification']),fresh_correct_identification_windows=int(fs.loc['multisource_cnn','correct_identifications']),fresh_wrong_identification_windows=int(fs.loc['multisource_cnn','wrong_identifications']),private_swarm_identifications=0,private_reference_user_reported=True,thresholds_not_changed_after_test=True,arduino_compile_allowed=False,arduino_adaptation_requires_later_explicit_authorization=True,biodcase_auxiliary_not_independent_ancestry_unresolved=True,reasons=['UFRGS Aedes domain remains largely misclassified','Private swarm still not identified','Coverage decreased on historical and fresh tests','Few species have new recording campaigns; not proof of all 20 classes'])
 atomic_json(O/'reliability_decision.json',decision)
 plt.rcParams.update({'font.size':10});fig,axs=plt.subplots(1,3,figsize=(15,4.6))
 for ax,metric,title in zip(axs,['forced_window_accuracy','known_coverage','accepted_known_accuracy'],['Acerto sem rejeição','Cobertura','Acerto nas emissões']):
  domains=['dryad','humbug','mosquitosong','ufrgs'];pos=np.arange(4)
  for delta,model,label in [(-.18,'large_cnn','CNN anterior'),(.18,'multisource_cnn','CNN multifonte')]:
   vals=m[m.model.eq(model)].set_index('domain').loc[domains,metric]*100
   bars=ax.bar(pos+delta,vals,width=.36,label=label);ax.bar_label(bars,fmt='%.1f',fontsize=8)
  ax.set(xticks=pos,xticklabels=['Dryad','HumBug','MosquitoSong*','UFRGS'],ylim=(0,112),ylabel='%',title=title);ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=8)
 fig.suptitle('Mais dados: ganhos parciais; UFRGS e enxame ainda falham. *Desafio retrospectivo.',fontsize=12);fig.tight_layout();fig.savefig(ROOT/'reports/assets/multisource_comparison.png',dpi=160);plt.close(fig)
 table='\n'.join(f'| {r.domain} | {r.model} | {r.known_windows} | {100*r.forced_window_accuracy:.1f}% | {100*r.known_coverage:.1f}% | {100*r.accepted_known_accuracy:.1f}% |' for r in m[m.evaluation_kind.eq('heldout')].itertuples())
 bt='\n'.join(f'| {r.domain} | {r.model} | {r.known_windows} | {100*r.forced_window_accuracy:.1f}% | {100*r.accepted_known_accuracy:.1f}% |' for r in m[m.evaluation_kind.eq('auxiliary')].itertuples())
 ft='\n'.join(f'| {r.id} | {r.model} | {r.windows} | {r.forced_correct} | {r.correct_identifications} | {r.wrong_identifications} |' for r in f.itertuples())
 report=f'''# Retreinamento com múltiplas fontes de áudio

**Execução concluída em 30/09/2026.** Os datasets adicionais foram baixados, auditados e usados em treinamento ou avaliação. A CNN melhorou em algumas tarefas e reduziu identificações erradas, mas **não ficou confiável em novos aparelhos/ambientes**. A UFRGS continua falhando e o enxame privado não foi identificado. Arduino, instalador e binários publicados não foram alterados.

## Dados baixados e uso efetivo

- **Figshare:** 2.988 WAV baixados e extraídos; 1.867 janelas de Aedes e 1.077 de rótulos taxonômicos não resolvidos entraram na adaptação. Culex pipiens & quinquefasciatus e Culiseta não foram inventados como espécies exatas. Todos os arquivos desse corpus ficaram no treino, pois não temos identidade da gravação longa/indivíduo para justificar um teste aleatório de recortes. Variantes e recortes não são indivíduos novos.
- **MosquitoSong:** todos os 1.324 WAV dos autores foram baixados; cortes e canais do mesmo grupo permaneceram juntos. 1.400 janelas foram usadas no treino, 987 na validação e 352 no desafio histórico de Aedes de cinco dias. Aedes de 21 dias é adaptação; as gravações de cinco dias antes examinadas continuam fora do treino. **Não é possível comprovar que os indivíduos de idades diferentes são distintos**: o resultado de MosquitoSong é retrospectivo e não valida transferência para indivíduos desconhecidos. Para as outras três espécies, machos ficaram no treino e fêmeas na validação, mantendo as idades e ambos os microfones do mesmo grupo juntos.
- **UFRGS:** 15 WAV baixados, todos fora do treino/validação/calibração: dez Aedes aegypti (271 janelas) e cinco referências **não-Aedes** (146 janelas). O nome `No_Aedes_aegypti` não identifica outra espécie nem comprova ausência de mosquito. Corrigimos a descrição inicial do catálogo que mencionava albopictus nessa seleção: não há essa pasta no repositório atual.
- **BioDCASE Development:** ZIP completo de 3.799.065.219 bytes baixado; MD5 `dbf35b9066e82cdee5461fca138f4986` conferido. Usado somente como teste auxiliar dos domínios D1–D4. De 4.238 arquivos do teste oficial nesses domínios, 3.893 foram retidos, com 36.963 janelas. Foram removidos 280 arquivos por identidade binária com treino ou duplicação no benchmark, e 65 menores que 0,3 s. A origem/identidade individual não está completamente mapeada; **não é um teste comprovadamente independente dos corpora anteriores**. D5 foi excluído porque a modalidade/origem de sensor não foi confirmada. O conjunto oficial Evaluation, sem rótulos públicos de espécie, não foi usado.
- **Synthetic Swarm:** 315 WAV e rótulos foram baixados. Não entraram na rodada: misturas derivam do Abuzz já usado, sem mapeamento completo dos ancestrais. Inserir uma mistura de origens de teste no treino contaminaria a avaliação. Mantivemos apenas a mistura sintética feita dentro do nosso treino, com fontes do treino.

Ao preparar as fontes adicionais foram detectados 42 arquivos binariamente duplicados e quatro curtos demais. O experimento reuniu 39.247 janelas, preservando todas as partições originais; 6.100 janelas foram adicionadas ao conjunto completo, **4.344 ao treino**. Treino válido total: **19.167 janelas**. [Contagens](../results/multisource/additional_counts.csv) · [Auditoria](../results/multisource/data_audit.json).

Fontes: [Figshare](https://figshare.com/articles/dataset/Mosquitoes_Bioacoustic_Features_-_A_Public_Dataset/11902125), [MosquitoSong](https://github.com/akaraspt/mosquitosongp), [UFRGS](https://github.com/ComputerNetworks-UFRGS/Aedes_Detector_Smartphone_App_Paper), [BioDCASE](https://zenodo.org/records/20478577), [Synthetic Swarm](https://github.com/duydinhthai27/synthetic-swarm-mosquito).

## Treinamento e controle

Mantivemos a CNN residual de **3.325.397 parâmetros** e ajustamos seus pesos a partir da versão grande anterior, sem substituí-la. Os lotes alternam 0,3 s (4.800 amostras reais) e 0,992 s (15.872); janelas longas podem ser cortadas para o lote curto, mas os áudios curtos não são repetidos, concatenados nem preenchidos. O frontend aceita esses comprimentos e usa o mesmo log-mel/contraste e normalização RMS. Isso permite aproveitar o Figshare sem fabricar continuidade acústica.

AdamW com taxa inicial 0,0002, amostragem equilibrando classe/domínio/origem, ruído e sobreposição da mesma espécie provenientes apenas do treino. Foram executadas **10 épocas**, com melhor perda de validação na **época {a['best_epoch']}**. O ajuste é exploratório; a validação já tinha sido usada na rodada anterior. Calibração ficou na partição antiga separada, com temperatura {a['calibration']['temperature']:.3f}, limiar de presença {a['calibration']['presence_threshold']:.3f} e de espécie {a['calibration']['species_threshold']:.3f}. Uma emissão temporal exige pelo menos 2,976 s reais e concordância causal; recortes curtos isolados têm classificação forçada avaliada, sem fabricar identificação temporal.

Pesos e limiares congelados em `{a['frozen_utc']}`, SHA-256 `{a['checkpoint_sha256']}`. Nenhum limiar foi ajustado com os testes abaixo. A UFRGS e o áudio privado não entraram no treino. Os nomes de arquivo/rótulos/domínios não são entradas da rede; somente o áudio é processado.

## Comparação pareada

As duas CNNs foram aplicadas às **mesmas janelas**, com o mesmo controle temporal. Cada versão conserva sua própria calibração. Acerto sem rejeição força uma espécie; cobertura é a proporção de janelas com emissão; acerto nas emissões exclui as rejeições. `nan` indica ausência de emissões, não acerto de 0%.

| Corpus | Modelo | Janelas conhecidas | Acerto sem rejeição | Cobertura | Acerto nas emissões |
|---|---|---:|---:|---:|---:|
{table}

A melhora em HumBug é pequena: 89,0% → 90,3%, enquanto a cobertura cai de 66,6% para 55,9%. No Dryad, o acerto sem rejeição permanece em cerca de 62,6%, com menor cobertura. **Mais precisão nas emissões não significa mais sons identificados.** Anopheles minimus no HumBug ainda tem recall zero em dez janelas de uma origem.

No MosquitoSong, 5,1% → 40,3% de acerto forçado, mas apenas três emissões, duas corretas e uma errada. Na UFRGS, 5,5% → 6,6% forçado; as dez emissões da nova versão são todas incorretas. A comparação não demonstra confiabilidade fora das fontes de treinamento. Nos cinco arquivos negativos da UFRGS, nenhuma versão emitiu Aedes; isso não valida as outras espécies emitidas.

Nas 212 janelas HumBug de mosquito sem rótulo resolvido para as 20 classes, emissões caíram de 34,9% para 11,3%. Não houve emissão de espécie nas 56 janelas de ruído Dryad ou 915 HumBug. Esses resultados amostrais não garantem ausência de falsos alarmes em campo.

![Comparação](assets/multisource_comparison.png)

### Benchmark BioDCASE auxiliar

| Domínio | Modelo | Janelas | Acerto sem rejeição | Acerto nas emissões |
|---|---|---:|---:|---:|
{bt}

Há ganhos nos quatro domínios, mas D2/D3 permanecem muito ruins. Janelas de uma gravação são correlacionadas. Não comparar esses números diretamente com a classificação por clipe do desafio: nosso recorte de duração, exclusões e protocolo temporal diferem. A verificação de duplicatas binárias não garante ausência de recortes irmãos, versões reamostradas ou sobreposição de indivíduos.

## Cinco áudios inéditos após o congelamento

Após concluir o treino, procuramos novamente a fonte pública HumBug e selecionamos, **antes da inferência**, os cinco primeiros IDs de Aedes de grupos distintos, duração ≥2,976 s e ano ≥2018, excluindo todas as origens dos 307 trechos anteriores. Downloads novos passaram em CRC32 e SHA-256. Não houve duplicatas binárias com o treinamento ou os áudios BioDCASE já avaliados; uma comparação das formas de onda reamostradas também não encontrou equivalências completas. Isso não prova ausência de todo recorte deslocado no benchmark. Os cinco são da mesma campanha/país e tipo de microfone presentes no HumBug: **novas gravações, não um domínio inteiramente novo**.

| ID | Modelo | Janelas | Aedes como primeira classe | Emissões corretas | Emissões erradas |
|---|---|---:|---:|---:|---:|
{ft}

Ambas as versões identificaram Aedes em **quatro de cinco arquivos**. Em 85 janelas, ambas acertaram 84 quando forçadas. Emissões corretas: CNN anterior **65/85**; nova **41/85**, sem emissões erradas em nenhuma versão. A nova ficou mais conservadora; este teste **não demonstra superioridade da nova CNN**. Não tratar 4/5 arquivos como estimativa da acurácia geral, nem 85 janelas como 85 mosquitos.

[Seleção e metadados](../results/multisource/fresh_external/selection.json) · [Resultados pareados](../results/multisource/fresh_external/summary.csv) · [Auditoria de novidade](../results/multisource/fresh_external/novelty_audit.json).

## Seu áudio privado de enxame

A referência de Aedes aegypti é a identificação informada pelo usuário, sem confirmação entomológica independente. Das 107 janelas: 99 SEM_EVIDENCIA, seis INCERTO e duas AUDIO_INVALIDO. Nenhuma identificação foi emitida, contra dez emissões erradas de Anopheles gambiae na CNN anterior. Aedes ficou em primeiro lugar em 46 janelas (antes 22), porém isso **não é identificação aceita**. O áudio permanece privado, fora do treino. Há redução de falsos rótulos, mas o enxame continua sem resolução de espécie.

## Decisão e reprodução

**Manter em avaliação offline.** Não substituir a versão anterior como se estivesse validada: ganhos são parciais, houve queda de cobertura e a generalização à UFRGS continua falhando. Antes de outro treinamento, será preciso definir como usar uma parte da UFRGS para adaptação sem chamar os mesmos arquivos de teste independente, e reservar outra campanha real de celular/microfone com espécie confirmada. Toda gravação usada para orientar novas decisões passa a ser histórica; a próxima rodada exige novamente áudio inédito após congelar pesos.

Scripts: `fetch_additional_audio.py`, `fetch_biodcase.py`, `prepare_multisource.py`, `train_multisource.py`, `prepare_biodcase_benchmark.py`, `evaluate_multisource.py`, `fetch_fresh_multisource_test.py`, `summarize_multisource.py`. O treinamento sobrescreve somente `results/multisource`; modelos anteriores são preservados. Inferência usa `test_large_audio.py --checkpoint results/multisource/models/multisource_cnn.pt ...`. Pesos e datasets ficam locais, ignorados pelo Git.

[Decisão](../results/multisource/reliability_decision.json) · [Comparação completa](../results/multisource/comparison.csv) · [Protocolo](../results/multisource/protocol.json) · [Notebook](../mosquito_wingbeat_estudo.ipynb). Nenhuma adaptação/compilação de Arduino, atualização de instalador, publicação de pesos ou execução de CI foi realizada.
'''
 (ROOT/'reports/retreinamento_multifonte.md').write_text(report)
 print(fs.to_string());print('Report and decision written')
if __name__=='__main__':main()
