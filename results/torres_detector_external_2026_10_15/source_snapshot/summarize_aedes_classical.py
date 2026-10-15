"""Generate honest paired comparison/report without changing any model."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aedes_classical import O,ROOT

def main():
    h=pd.read_csv(O/'historical_metrics.csv');cnn=pd.read_csv(ROOT/'results/aedes_binary/historical_metrics.csv');cnn['model']='cnn_binary'
    combined=pd.concat([cnn,h],ignore_index=True);combined.to_csv(O/'comparison.csv',index=False)
    fresh=pd.read_csv(O/'fresh_external/metrics.csv');audit=json.loads((O/'training_audit.json').read_text());private=json.loads((O/'private_audio_summary.json').read_text())
    names={'cnn_binary':'CNN binária','logistic':'Regressão logística','svm_rbf':'SVM RBF','extra_trees':'Extra Trees','gradient_boosting':'Gradient boosting'}
    lines=[]
    for r in combined.itertuples():
        fmt=lambda v:'Não avaliável' if pd.isna(v) else f'{v*100:.1f}%'
        lines.append(f'| {names[r.model]} | {r.domain} | {fmt(r.emitted_aedes_recall)} | {fmt(r.emitted_false_positive_rate)} | {fmt(r.emitted_aedes_precision)} | {fmt(r.known_coverage)} |')
    text='''# Comparação de outras técnicas de ML: Aedes aegypti

**Quatro famílias de modelos clássicos foram treinadas e comparadas à CNN binária congelada.** Há ganhos importantes em algumas fontes, mas nenhuma versão demonstrou confiabilidade suficiente para liberar o aparelho. Não houve adaptação ou compilação para Arduino.

## Métodos e controle do experimento

Regressão logística (C 0,1/1), SVM com kernel RBF (C 1/10), Extra Trees (300 árvores, folhas mínimas 2/8) e gradient boosting histográfico (200 iterações, 15/31 folhas). Todas usam as mesmas 17.391 janelas válidas de treino e as mesmas partições de validação e calibração do detector binário; UFRGS inteira e áudio privado ficam fora do treino. Não houve novo treino de CNN. Dados e testes históricos já influenciaram decisões de projeto: esta comparação é exploratória, não avaliação final cega.

Extraímos **122 características acústicas**: média e variabilidade de energia relativa em 31 bandas de 125 Hz, coeficientes cepstrais por DCT dessas bandas lineares (não MFCC), centroide e largura espectral, rolloff, planicidade, entropia, picos de frequência, relações de harmônicos e fluxo espectral. FFT 2.048, hop 256, 16 kHz; remoção da média e normalização RMS por janela, sem preenchimento ou repetição. Janelas reais de 0,3 e 0,992 s são mantidas. Essas características não contêm nomes de arquivos, espécie, aparelho ou duração; mesmo assim podem reter assinaturas do ambiente.

Amostragem/ponderação: classes binárias com igual massa; negativos equilibrados entre rótulos, origens e grupos. Mosquitos de espécie não resolvida não recebem rótulo negativo. StandardScaler ajustado somente no treino. Dois candidatos por família, escolha pela AUC ponderada da validação. Calibração logística em grupos separados e limiar visando no máximo 5% de falsos positivos **ponderados** na calibração, não uma promessa de 5% em outras fontes. Há faixa de rejeição e guarda de qualidade.

O gradient boosting foi selecionado pela validação (AUC ponderada 0,9182), antes de consultar a nova seleção externa. Essa escolha foi preservada apesar dos resultados posteriores. Ao otimizar a explicabilidade, o processo foi retomado a partir dos arquivos já congelados de regressão logística e SVM, sem retreiná-los nem alterar parâmetros/limiares.

## Comparação histórica, com rejeição

Detecção de Aedes = emissões corretas / todas as janelas positivas. Falso positivo = emissões de Aedes / todas as janelas negativas rotuladas. Precisão = emissões corretas de Aedes / todas as emissões de Aedes com referência binária conhecida. Cobertura inclui emissões positivas e negativas, excluindo as rejeições. Não confundir com acurácia global. Janelas correlacionadas não são mosquitos independentes.

| Modelo | Fonte | Detecção de Aedes | Falsos positivos | Precisão de Aedes emitido | Cobertura conhecida |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'''

Dryad: 360 janelas positivas e 7.541 negativas; HumBug: 154 positivas, 1.144 negativas e 212 de espécie não resolvida; UFRGS: 271 positivas e 146 negativas; desafio MosquitoSong: 352 positivas, sem negativos. A ausência de negativos em MosquitoSong não permite medir falsos positivos nem sustenta uma precisão de 100% em uso real. Os grupos UFRGS são categorias de sexo/pasta, não indivíduos identificados.

O SVM melhora o compromisso do HumBug: 149/154 Aedes aceitos corretamente, 59/1.144 falsos positivos, frente a 144/154 e 279/1.144 na CNN. Na UFRGS, Extra Trees aceita 220/271 Aedes, mas erra 43/146 negativos; a CNN aceita 61/271 com quatro falsos positivos. Maior sensibilidade não significa melhor confiabilidade universal. Percentuais dependem da prevalência de Aedes no conjunto e dos limiares; não compará-los como se fossem um único número de acurácia do aparelho.

![Detecção e falsos positivos por fonte](assets/aedes_classical_comparison.png)

## Novo teste após congelar todos os modelos

Busca online, seleção de metadados e download de cinco gravações HumBug posteriores ao congelamento de todos os modelos. Excluídos grupos de treinamento/avaliação e todos os WAV numéricos HumBug anteriormente baixados. Selecionadas as duas únicas origens positivas restantes com duração >=0,3 s e três negativas de espécies distintas, por ordem de metadados, antes de inferir. Áudios: Aedes 53 e 220411, outros IDs na seleção JSON. Verificados CRC e SHA-256; sem bytes iguais aos arquivos externos locais. Não concluída auditoria de sobreposição deslocada nem identidade individual entre fontes. Campanhas familiares, não domínio totalmente novo. O conjunto negativo inclui Ae. albopictus (ID 87), An. arabiensis (206567) e An. dirus (245).

O teste contém **somente oito janelas: três positivas e cinco negativas**. É insuficiente para afirmar desempenho estável.

| Modelo | Aedes aceitos corretamente / 3 | Falsos positivos / 5 | Rejeições |
|---|---:|---:|---:|
'''
    for r in fresh.itertuples():
        rejected=int(r.windows-round(r.known_coverage*r.windows))
        text+=f'| {names[r.model]} | {r.true_positive_emissions}/3 | {r.false_positive_emissions}/5 | {rejected} |\n'
    text+='''
SVM detectou as três janelas positivas, mas gerou um falso positivo; o resultado favorável em uma amostra tão pequena não autoriza mudar o vencedor pela validação nem liberar firmware. Os demais modelos também apresentaram erros.

## Seu áudio privado de enxame

107 janelas, com a espécie informada pelo usuário, sem confirmação independente ou anotação da presença do mosquito por instante. Não foi usado no treino ou ajuste de limiares. Somente resultados agregados são públicos; WAV e previsões por instante continuam locais e ignorados pelo Git.

| Modelo | Indicações de Aedes | Não-Aedes | Incerto | Inválido |
|---|---:|---:|---:|---:|
'''
    for r in private:
        s=r['states'];text+=f"| {names[r['model']]} | {s.get('AEDES_AEGYPTI',0)} | {s.get('NAO_AEDES_AEGYPTI',0)} | {s.get('INCERTO',0)} | {s.get('AUDIO_INVALIDO',0)} |\n"
    text+='''
Mais indicações de Aedes nesse arquivo não demonstram melhor acerto: os modelos com mais indicações também geram falsos positivos relevantes nos controles. Não declarar as janelas negativas como erros confirmados, pois não há anotação por instante.

## Explicabilidade

Importância por permutação calculada exclusivamente em uma seleção de 256 janelas de validação (128 de cada classe), com os mesmos exemplos e pesos para a AUC inicial e todas as permutações, três repetições. Arquivos explanation_*.csv mostram a queda de AUC ao embaralhar cada característica. Coeficientes da regressão logística são exportados em unidades padronizadas. As bandas e coeficientes são correlacionados: importância não comprova causalidade, um marcador biológico exclusivo ou estabilidade fora dessas fontes. Nenhum resultado explicativo foi usado para ajustar modelos ou consultar o teste final.

## Uso e decisão

```bash
.venv/bin/python aedes_classical.py gravacao.wav --model svm_rbf --output tmp/teste_svm
```

Opções: logistic, svm_rbf, extra_trees e gradient_boosting. Sem --model, usa o gradient boosting selecionado pela validação. Precisa dos arquivos locais results/aedes_classical/models/*.joblib, ignorados pelo Git. Não substitui automaticamente o algoritmo anterior nem publica novos pesos. Salva predictions.csv e summary.json; aceita pelo menos 0,3 s de som real.

**Manter todos em avaliação offline.** O foco binário e as características acústicas trouxeram ganhos parciais, mas há troca entre falsos positivos e perdas de Aedes. Próximo experimento deve incluir negativos acusticamente semelhantes e dados da caixa/microfone real, com indivíduos e dias novos reservados para teste. Novas decisões orientadas pelos testes desta rodada tornam esses áudios históricos; outro treinamento exige buscar outro áudio inédito após congelar.

Fontes primárias: [SVM](https://scikit-learn.org/stable/modules/svm.html), [árvores e boosting](https://scikit-learn.org/stable/modules/ensemble.html), [HumBugDB](https://zenodo.org/records/4904800). Resultados locais: results/aedes_classical/comparison.csv, training_audit.json, evaluation_audit.json e fresh_external/selection.json. Notebook: seção 26.
'''
    (ROOT/'reports/comparacao_ml_binario_aedes.md').write_text(text)
    (O/'reliability_decision.json').write_text(json.dumps(dict(status='EXPERIMENTAL_SENSITIVITY_FALSE_POSITIVE_TRADEOFF',validation_selected=audit['selected_by_validation'],fresh_test_too_small=True,arduino_compiled=False,default_model_replaced=False),indent=2))
    order=list(names);fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,domain in zip(axes,['humbug','ufrgs']):
        part=combined[combined.domain.eq(domain)].set_index('model').reindex(order);xx=np.arange(len(order))
        ax.bar(xx-.18,100*part.emitted_aedes_recall,width=.36,color='#167b68',label='Aedes detectados (%)')
        ax.bar(xx+.18,100*part.emitted_false_positive_rate,width=.36,color='#c74a45',label='Falsos positivos (%)')
        ax.set_xticks(xx,[names[n].replace(' ','\n') for n in order],fontsize=9);ax.set_title(domain.upper());ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Percentual de janelas na classe de referência');axes[1].legend(loc='upper right',fontsize=8)
    fig.suptitle('Detecção binária com rejeição: ganhos e falsos positivos\nTestes históricos; janelas não são indivíduos independentes',fontsize=12)
    fig.tight_layout();fig.savefig(ROOT/'reports/assets/aedes_classical_comparison.png',dpi=160);plt.close(fig)
    print('Report and comparison figure generated')
if __name__=='__main__':main()
