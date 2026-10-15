"""Render the completed simulation outputs without altering predictions."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIMARY = 'activity_gated_log_odds_cascade_event'
result = json.loads((HERE/'metrics.json').read_text())
parents = pd.read_csv(HERE/'parents.csv')
pred = pd.read_csv(HERE/'predictions.csv')
audit = json.loads((HERE/'frontend_audit.json').read_text())
conditions = ['clean_control','white_snr20','pink_snr10','pink_snr0','pink_snr_minus5',
              'attenuation18db','lowpass1200','passby_envelope']
labels = ['Controle','Branco +20 dB','Rosa +10 dB','Rosa 0 dB','Rosa −5 dB',
          'Atenuação 18 dB','Passa-baixa 1,2 kHz','Envelope de passagem']


def condition(suite, name, prefix='heldout_'):
    return result['conditions'][suite+'/'+name]['models'][prefix+PRIMARY]


def pct(x):
    return 'Não estimável' if x is None else f'{100*x:.2f}%'.replace('.',',')


metrics_rows = []
for key, item in result['conditions'].items():
    for model, m in item['models'].items():
        metrics_rows.append(dict(suite_condition=key,model=model,**{
            k:v for k,v in m.items() if not isinstance(v,(list,dict))}))
pd.DataFrame(metrics_rows).to_csv(HERE/'metrics_summary.csv',index=False)

# Plot only the primary held-out results, not fit-donor final-model diagnostics.
values = [condition('paired_perturbations',name) for name in conditions]
fig, axes = plt.subplots(3,1,figsize=(12,11),layout='constrained')
positions=np.arange(len(conditions))
axes[0].bar(positions,[100*v['aedes_recall'] for v in values],color='#246B8E')
axes[0].set(ylabel='Aedes detectados (%)',ylim=(0,100),title='Clipes pareados: 60 doadores positivos por condição')
axes[0].set_xticks(positions,labels,rotation=15,ha='right')
for i,v in enumerate(values): axes[0].text(i,100*v['aedes_recall']+2,f"{v['aedes_tp']}/60",ha='center',fontsize=9)
axes[1].bar(positions-.18,[100*v['noise_to_aedes_fpr'] for v in values],width=.36,label='Ruído → Aedes',color='#4D9970')
axes[1].bar(positions+.18,[100*v['other_to_aedes_fpr'] for v in values],width=.36,label='Outro mosquito → Aedes',color='#BF6C35')
axes[1].set(ylabel='Falsos positivos (%)',ylim=(0,8),title='120 doadores negativos de cada classe por condição')
axes[1].set_xticks(positions,labels,rotation=15,ha='right');axes[1].legend(frameon=False)
mixed=[condition('mixed_mosquitoes',f'aedes_to_other_{db}db') for db in [10,0,-10]]
axes[2].bar(np.arange(3),[100*v['aedes_recall'] for v in mixed],color='#246B8E')
axes[2].set_xticks(np.arange(3),['Aedes 10 dB acima do outro','Mesma potência RMS','Aedes 10 dB abaixo do outro'])
axes[2].set(ylabel='Aedes detectados (%)',ylim=(0,100),title='Mistura simultânea de dois mosquitos: 60 positivos por condição')
for i,v in enumerate(mixed):axes[2].text(i,100*v['aedes_recall']+2,f"{v['aedes_tp']}/60",ha='center',fontsize=10)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Teste de robustez semissintético — modelos congelados por dobra',fontsize=16)
fig.savefig(HERE/'robustness.png',dpi=180)
plt.close(fig)

text = '''# Dataset sintético e teste do detector ternário

Data: 14/10/2026. Modelo: `ternary_verified_2026_10_14_corrections`. Nenhum modelo foi retreinado nem teve limiares alterados.

## Resultado observado

Foram criados e avaliados **2.740 WAVs**. A cascata final classificou **0/160 negativos totalmente artificiais como Aedes**; 158 foram classificados como ruído e dois como outros mosquitos. No teste com doadores fora do treino de cada modelo, a detecção de Aedes caiu de **36/60 no controle** para **8/60 com ruído branco a +20 dB** e **2/60 com ruído rosa a +10 dB**.

O teste encontrou resistência aos falsos alarmes de Aedes nesses negativos e uma fragilidade de sensibilidade às perturbações simuladas. Esses resultados foram preservados sem escolher novos limiares ou regenerar exemplos por seus acertos.

## Análise de lacunas e desenho do teste

O resultado anterior agrega várias janelas por grupo. Faltava medir de forma controlada a alteração de decisão após acrescentar ruído, filtrar, atenuar ou misturar sinais. A simulação usa trechos reais anotados para preservar a referência de espécie; frequência de um oscilador, sozinha, não recebe rótulo de Aedes.

- **300 grupos doadores:** 60 Aedes, 120 outros mosquitos e 120 ruídos; correspondem a 297 blocos PCM.
- **2.400 clipes semissintéticos:** oito versões pareadas de cada trecho, incluindo o controle sem alteração.
- **180 misturas:** Aedes e outro mosquito simultâneos em relações de +10, 0 e −10 dB.
- **160 negativos inteiramente sintéticos:** oito perfis com 20 sementes cada.
- Áudio mono a 16 kHz. Doadores com **0,992 s reais**, sem preencher ou repetir trechos; negativos procedurais de quatro segundos.
- Famílias procedurais: ruído branco, rosa, marrom, harmônicos de rede elétrica, harmônicos mecânicos, chirp, cliques e zumbido mecânico modulado. São controles de engenharia, sem pretensão de reproduzir aparelhos específicos.

As receitas, sementes, doadores e WAVs foram registrados antes da inferência. A escolha de trechos foi aleatória dentro dos grupos selecionados, sem consultar scores. Os doadores negativos foram distribuídos por fonte/táxon. Todos os 60 grupos positivos do desenvolvimento participaram.

O SNR é a razão entre o RMS do **trecho inteiro** e o componente acrescentado; não mede o SNR isolado das asas. Componentes de mistura foram escalados antes de uma redução conjunta de pico quando necessária, preservando a razão entre eles. O filtro é Butterworth de quarta ordem, passa-baixa de 1,2 kHz; a passagem usa envelope gaussiano com piso de 0,05, centro de 0,50 s e desvio de 0,13 s.

## Isolamento e unidade de avaliação

A avaliação principal de derivados reais usa o modelo externo congelado da dobra que deixou **todos os componentes da mistura** fora do ajuste, incluindo seus blocos PCM. Os ruídos procedurais usam o modelo final congelado. Os parâmetros dos modelos permanecem os mesmos do experimento original.

Também foram calculadas previsões do modelo final nos derivados reais como diagnóstico secundário. Esse modelo foi ajustado nos grupos doadores; seus resultados não são uma estimativa independente. As colunas do CSV distinguem `heldout_` de `final_`.

Cada observação aqui é um clipe de 0,992 s, e não a agregação original de todas as janelas de um grupo. Por isso, **36/60 no controle não substitui nem contradiz os 47/60 da avaliação anterior por grupo**. A comparação controlada é entre cada perturbação e este controle de janela.

As 2.740 versões não representam 2.740 indivíduos independentes. Os resultados por condição compartilham os mesmos doadores. A prevalência de Aedes nas condições pareadas é de 20%, escolhida pelo desenho; precisão não deve ser transportada para a prevalência do corpus original ou do campo.

## Condições pareadas

Modelo principal: cascata congelada correspondente à dobra do doador. Os denominadores incluem os clipes tecnicamente inválidos, registrados como ausência de classificação.

| Condição | Aedes detectados | Sensibilidade | Ruído → Aedes | Outro → Aedes | Cobertura | Aedes no modelo final¹ |
|---|---:|---:|---:|---:|---:|---:|
'''
for label,name,m in zip(labels,conditions,values):
    f=condition('paired_perturbations',name,'final_')
    text+=f"| {label} | {m['aedes_tp']}/60 | {pct(m['aedes_recall'])} | {m['noise_fp']}/120 | {m['other_fp']}/120 | {pct(m['coverage'])} | {f['aedes_tp']}/60 |\n"
text+='''
¹ Diagnóstico com doadores presentes no treino do modelo final, sem valor de confirmação independente.

A atenuação produziu 48 clipes sem janela válida; o filtro e o envelope produziram mais um caso cada. Os 50 permaneceram nos denominadores, incluindo os positivos. O baixo FPR nas condições mais difíceis acompanha uma queda da detecção de positivos.

## Misturas de mosquitos

| Relação Aedes / outro, em RMS | Aedes detectados fora do treino | Sensibilidade | Modelo final¹ |
|---|---:|---:|---:|
'''
for db,m in zip([10,0,-10],mixed):
    f=condition('mixed_mosquitoes',f'aedes_to_other_{db}db','final_')
    text+=f"| {db:+d} dB | {m['aedes_tp']}/60 | {pct(m['aedes_recall'])} | {f['aedes_tp']}/60 |\n"
text+='''
Todas essas misturas contêm um componente com referência de Aedes. Elas medem detecção em coocorrência simulada; não permitem estimar especificidade nessa seção.

## Negativos totalmente procedurais

| Perfil | Ruído → Aedes | Ruído → qualquer mosquito |
|---|---:|---:|
'''
for name in ['white','pink','brown','mains_harmonics','fan_harmonics','chirp','clicks','modulated_fan']:
    m=condition('procedural_negatives',name,'final_');cm=m['confusion_rows_noise_other_aedes_columns_noise_other_aedes_invalid']
    text+=f"| {name} | {m['noise_fp']}/20 | {cm[0][1]+cm[0][2]}/20 |\n"
text+='''
As taxas descrevem estes geradores e estas sementes. Não são uma estimativa da variedade de ruídos reais.

## Conferências e testes

- **51 testes relacionados aprovados**, 135 avisos, em 4,02 s, incluindo mistura por componentes independentes, SNR, controle sem alteração, separação de blocos PCM, frontend idêntico à inferência e tratamento de áudio inválido.
- As características dos controles foram comparadas ao cache original. A diferença absoluta máxima foi `0,0001220703125`, a mediana foi zero e **nenhuma das 300 decisões da cascata mudou**. Ambos os caminhos deram 36/60 Aedes. Isso descarta divergência de frontend como explicação desses controles.
- A razão RMS registrada ficou a menos de `3,56e-15 dB` da receita antes da escrita FLOAT. Todos os WAVs tiveram duração e taxa de amostragem conferidas.
- Foram preservados hashes do gerador, do dataset, dos modelos, das previsões e dos resultados. O protocolo foi encerrado antes de iniciar a inferência.

O gerador novo e os testes foram adicionados sem alterar o frontend, os classificadores ou os limiares da rodada anterior. Este conjunto é um teste de robustez por simulação. Se for usado futuramente para ajustar o modelo, deixa de ser um teste reservado dessa nova versão.

## Artefatos e reprodução

- [Áudios WAV](audio/), [manifesto e hashes](manifest.csv), [doadores e linhagem](parents.csv), [protocolo](protocol.json).
- [Previsões de todos os modelos](predictions.csv), [métricas completas](metrics.json), [resumo CSV](metrics_summary.csv), [auditoria de frontend](frontend_audit.json).
- [Gráfico das condições](robustness.png).

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python evaluate_ternary_synthetic.py \\
  --run results/ternary_verified_2026_10_14_corrections \\
  --out results/ternary_synthetic_reproducao --phase all

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \\
  tests/test_ternary_synthetic.py tests/test_ternary_verified.py tests/test_ternary_detector.py
```

A reprodução exige o repositório, as entradas originais e os modelos congelados disponíveis localmente. O diretório de saída deve ser novo. Os dados e a inferência foram executados nesta entrega; não houve retreinamento.
'''
(HERE/'README.md').write_text(text)
public_text=text
for name in ['audio/','manifest.csv','parents.csv','protocol.json','predictions.csv','metrics.json',
             'metrics_summary.csv','frontend_audit.json','robustness.png']:
    public_text=public_text.replace(']('+name+')','](../results/'+HERE.name+'/'+name+')')
(ROOT/'reports/teste_sintetico_ternario_2026_10_14.md').write_text(public_text)
print('Report, CSV summary and figure generated.')
