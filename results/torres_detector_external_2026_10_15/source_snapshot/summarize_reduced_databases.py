import json
import numpy as np,pandas as pd,nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from compare_reduced_databases import ROOT,OUT
from train_reproduction import sha
m=pd.read_csv(OUT/'metrics.csv');u=pd.read_csv(OUT/'unseen_metrics.csv');d=pd.read_csv(OUT/'unseen_predictions.csv.gz');models=json.loads((OUT/'freeze.json').read_text());ci=[]
for name,a in d.groupby('model'):
 cut=__import__('joblib').load(OUT/'models'/f'{name}.joblib')['cut'];rng=np.random.default_rng(20261002)
 groups=a.assign(hit=a.quality_ok&(a.score>=cut),reject=a.quality_ok&(a.score<cut)).groupby(['label','group']).agg(n=('score','size'),hit=('hit','sum'),reject=('reject','sum'));values=[]
 for i in range(2000):
  sums=[]
  for label in [0,1]:
   g=groups.loc[label].to_numpy();sums.append(g[rng.integers(len(g),size=len(g))].sum(0))
  neg,pos=sums;values.append(.5*(pos[1]/pos[0]+neg[2]/neg[0]))
 ci.append(dict(model=name,balanced_accuracy_lower95=float(np.quantile(values,.025)),balanced_accuracy_upper95=float(np.quantile(values,.975))))
pd.DataFrame(ci).to_csv(OUT/'new_file_bootstrap.csv',index=False)
cols=['model','aedes_precision','aedes_recall','specificity','balanced_accuracy']
def table(f,cols):return f[cols].to_markdown(index=False,floatfmt='.3f')
text='''# Comparação com menos bases — 02/10/2026

Foram treinados nove modelos binários: Dryad, HumBug e ambas, cada configuração com regressão logística, SVM RBF e ExtraTrees. As mesmas 138 características acústicas e o mesmo preprocessamento foram usados para isolar a mudança de fonte e de classificador. Não são nove frontends ou reproduções integrais de artigos. Nenhum modelo foi adaptado ao Arduino.

Treino, validação e calibração conservaram grupos e hashes separados. Normalização foi ajustada apenas no treino; limites de decisão maximizaram a acurácia balanceada ponderada na calibração. A escolha do candidato foi feita pela validação, antes dos testes. Classes desconhecidas não viraram negativos. Grupos recebem pesos para evitar que gravações longas dominem o treino.

Os testes históricos e o Drive já haviam sido examinados: esta parte é retrospectiva, não confirmação cega. Depois do congelamento foram baixados 12 arquivos acústicos CDMSC D3 ainda não utilizados: quatro Aedes aegypti e dois de cada uma das quatro espécies negativas. Seleção numérica predeterminada, CRC e hash verificados. São 125 janelas positivas e 72 negativas. Identidades biológicas/sessões originais não estão disponíveis; rótulos de arquivo não anotam voo em cada janela. A amostra é pequena.

## Validação e escolha anterior ao teste

'''+table(m[m.scope.eq('validation')],cols)+f"\n\nCandidato escolhido: **{models['chosen_by_validation']}**. Comparar resultados de candidatos no teste é exploratório, não autoriza promover o vencedor sem nova validação.\n\n## Teste inédito, mesma fonte acústica para ambas as classes\n\n"+table(u,cols)+'''

## Áudio privado do Drive

O usuário identifica a gravação como enxame de Aedes na caixa de vidro. As 107 janelas foram mantidas no denominador, inclusive duas inválidas. A proporção reconhecida é sensibilidade nesse áudio, não acurácia de um detector binário; não há negativos para medir falsos positivos. O áudio não foi treinado, copiado para o relatório ou publicado.

'''+table(m[m.scope.eq('drive_positive_only')],['model','true_positive','known_windows','aedes_recall'])+'''

## Testes históricos por base

'''+table(m[m.scope.isin(['historical_dryad','historical_humbug','historical_ufrgs'])],['model','scope','aedes_recall','specificity','balanced_accuracy'])+'''

A SVM Dryad atinge 94,8% de acurácia balanceada em Dryad e aproximadamente 50% em HumBug. A SVM HumBug atinge 89,8% em HumBug e 48,9% em Dryad. Essa queda entre fontes reforça a dependência do domínio, mas não identifica isoladamente microfone, sexo ou ruído como causa. Misturar ambas ajuda nas duas fontes conhecidas, sem resolver a nova fonte nem o Drive.

Nenhum candidato atingiu simultaneamente sensibilidade e especificidade de 60% na amostra inédita. Melhor resultado no Drive: SVM Dryad+HumBug, 32/107 (29,9%); no teste inédito esse mesmo modelo detectou apenas 4% dos Aedes. Regressão Dryad detectou 41,6% dos Aedes inéditos, mas apenas 1/107 do Drive. Não foi demonstrada confiabilidade.

Os intervalos em new_file_bootstrap.csv reamostram arquivos por classe, 2000 vezes; não são intervalos por indivíduos biológicos. Arquivos novos foram verificados por nomes e bytes; esta rodada não fez auditoria completa de fragmentos PCM coincidentes. As divisões anteriores têm limitações de metadados e os testes históricos não são independentes das decisões de pesquisa anteriores.
'''
report=ROOT/'reports/comparacao_bases_reduzidas_2026_10_02.md';report.write_text(text)
nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text),nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\np=Path('results/reduced_databases_2026_10_02')\nfor name in ['validation','unseen_metrics','new_file_bootstrap']:\n    print(name)\n    display(pd.read_csv(p/(name+'.csv')))"),nbformat.v4.new_code_cell("m=pd.read_csv(p/'metrics.csv')\ndisplay(m[m.scope.eq('drive_positive_only')][['model','true_positive','known_windows','aedes_recall']])\nassert len(m.model.unique())==9\nassert (m[m.scope.eq('drive_positive_only')].known_windows==107).all()")],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
NotebookClient(nb,timeout=120,resources={'metadata':{'path':str(ROOT)}}).execute();path=ROOT/'comparacao_bases_reduzidas_2026_10_02.ipynb';nbformat.write(nb,path);html,_=HTMLExporter().from_notebook_node(nb);path.with_suffix('.html').write_text(html)
for p,h in models['hashes'].items():assert sha(ROOT/p)==h
print('Report, executed notebook, HTML and bootstrap verified')
