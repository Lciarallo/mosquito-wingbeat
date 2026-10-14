# Experimento TinyML + score Dryad: interpretação corrigida

Revisão de 08/10/2026 sobre o commit `1ab88b8`. **Análise retrospectiva de presença de mosquito versus ruído; não valida classificação por espécie.** As métricas abaixo foram transcritas do [JSON produzido pelo experimento](../results/mixed_dataset_experiment.json). O treinamento não foi repetido nesta revisão.

## O que foi implementado

`experiment_mixed_datasets.py` separa o corpus TinyML por nome de arquivo com `GroupShuffleSplit`. Treina HistGradientBoosting no TinyML como baseline. O modelo chamado misto é um RandomForest ajustado somente no treino TinyML, recebendo as 68 features e o score de uma rede de presença Dryad pré-treinada. Não há treinamento conjunto balanceado dos dois corpora, nem um controle com o mesmo estimador para isolar o efeito de combinar fontes.

O alvo avaliado é qualquer mosquito contra ruído. O script ajusta também um classificador de quatro classes, mas não exporta sua avaliação. Não se pode usar os F1 abaixo como resultado de *Ae. aegypti* versus *Ae. albopictus*, de gênero ou das 20 espécies.

## Resultados registrados

F1 abaixo é o F1 binário da classe positiva, conforme `f1_score(y_true, y_pred)` no código, não macro F1.

| Modelo | F1 positivo TinyML | Acurácia balanceada TinyML | AUC HumBug | AUC Drive |
|---|---:|---:|---:|---:|
| Rede Dryad pré-treinada | 93,69% | 85,15% | 0,416 | 0,276 |
| HistGradientBoosting TinyML | 95,90% | 86,25% | 0,668 | 0,986 |
| RandomForest TinyML + score Dryad | 96,47% | 83,22% | 0,537 | 0,879 |

| Modelo híbrido | Sensibilidade | Falso positivo nos negativos | Acurácia balanceada |
|---|---:|---:|---:|
| HumBug | 100,0% | 99,18% | 50,41% |
| Drive | 100,0% | 82,05% | 58,97% |

O F1 TinyML sobe modestamente, mas a acurácia balanceada cai. AUC e acurácia balanceada externas também não demonstram vantagem do híbrido sobre o baseline TinyML. Sua alta sensibilidade externa acompanha muitos falsos positivos. Não foi isolada a causa dos erros nem demonstrada robustez operacional.

## Correções em relação ao relatório original

- O híbrido tinha sido descrito com AUC 0,668 no HumBug e 0,986 no Drive; o JSON registra **0,537 e 0,879**, respectivamente. Os valores anteriores pertencem ao baseline TinyML.
- O [CSV histórico](../results/mixed_dataset_metrics.csv) também difere do JSON no AUC híbrido HumBug e chama o F1 positivo de `macro_f1`. O script salva JSON, mas não gera esse CSV.
- A linha “Multi-Dataset Calibrado”, com TPR 96,6% / FPR 9,0%, não aparece no JSON nem tem etapa correspondente de calibração/avaliação em `experiment_mixed_datasets.py`. Sua procedência e isolamento de teste não foram estabelecidos nesta revisão; foi retirada das conclusões validadas.
- Os rótulos temporais Drive são definidos no próprio código pelo intervalo 49,5–77,5 s, tratando todo o restante como negativo. Não foi identificado manifesto independente de anotação que justifique essa referência. A espécie é a informação do usuário; este áudio já foi analisado repetidamente e não é confirmação cega nova.
- Holdout por nome de arquivo não comprova ausência de derivados, indivíduos ou ruídos compartilhados com o treino da rede Dryad. O certificado das dobras Dryad não audita este experimento entre bases.

Os artefatos numéricos originais foram preservados para rastreabilidade. Para avaliar melhoria por espécie, é necessário um experimento com alvo taxonômico explícito, estimadores comparáveis, agrupamento por origem/indivíduo/sessão, calibração separada e teste confirmatório após congelamento. [Revisão completa da atualização Git e reexecuções no áudio](revisao_git_classificacao_especies_2026_10_08.md).
