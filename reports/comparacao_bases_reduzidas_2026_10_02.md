# Comparação com menos bases — 02/10/2026

Foram treinados nove modelos binários: Dryad, HumBug e ambas, cada configuração com regressão logística, SVM RBF e ExtraTrees. As mesmas 138 características acústicas e o mesmo preprocessamento foram usados para isolar a mudança de fonte e de classificador. Não são nove frontends ou reproduções integrais de artigos. Nenhum modelo foi adaptado ao Arduino.

Treino, validação e calibração conservaram grupos e hashes separados. Normalização foi ajustada apenas no treino; limites de decisão maximizaram a acurácia balanceada ponderada na calibração. A escolha do candidato foi feita pela validação, antes dos testes. Classes desconhecidas não viraram negativos. Grupos recebem pesos para evitar que gravações longas dominem o treino.

Os testes históricos e o Drive já haviam sido examinados: esta parte é retrospectiva, não confirmação cega. Depois do congelamento foram baixados 12 arquivos acústicos CDMSC D3 ainda não utilizados: quatro Aedes aegypti e dois de cada uma das quatro espécies negativas. Seleção numérica predeterminada, CRC e hash verificados. São 125 janelas positivas e 72 negativas. Identidades biológicas/sessões originais não estão disponíveis; rótulos de arquivo não anotam voo em cada janela. A amostra é pequena.

## Validação e escolha anterior ao teste

| model                     |   aedes_precision |   aedes_recall |   specificity |   balanced_accuracy |
|:--------------------------|------------------:|---------------:|--------------:|--------------------:|
| dryad__logistic           |             0.151 |          0.842 |         0.784 |               0.813 |
| dryad__svm_rbf            |             0.308 |          0.608 |         0.938 |               0.773 |
| dryad__extra_trees        |             0.293 |          0.483 |         0.947 |               0.715 |
| humbug__logistic          |             0.207 |          0.818 |         0.844 |               0.831 |
| humbug__svm_rbf           |             0.189 |          0.884 |         0.812 |               0.848 |
| humbug__extra_trees       |             0.160 |          0.793 |         0.794 |               0.794 |
| dryad+humbug__logistic    |             0.170 |          0.830 |         0.808 |               0.819 |
| dryad+humbug__svm_rbf     |             0.198 |          0.772 |         0.852 |               0.812 |
| dryad+humbug__extra_trees |             0.202 |          0.739 |         0.861 |               0.800 |

Candidato escolhido: **humbug__svm_rbf**. Comparar resultados de candidatos no teste é exploratório, não autoriza promover o vencedor sem nova validação.

## Teste inédito, mesma fonte acústica para ambas as classes

| model                     |   aedes_precision |   aedes_recall |   specificity |   balanced_accuracy |
|:--------------------------|------------------:|---------------:|--------------:|--------------------:|
| dryad+humbug__extra_trees |             0.250 |          0.008 |         0.958 |               0.483 |
| dryad+humbug__logistic    |             0.848 |          0.224 |         0.931 |               0.577 |
| dryad+humbug__svm_rbf     |             0.625 |          0.040 |         0.958 |               0.499 |
| dryad__extra_trees        |             0.000 |          0.000 |         0.986 |               0.493 |
| dryad__logistic           |             0.703 |          0.416 |         0.694 |               0.555 |
| dryad__svm_rbf            |             0.375 |          0.024 |         0.931 |               0.477 |
| humbug__extra_trees       |           nan     |          0.000 |         1.000 |               0.500 |
| humbug__logistic          |           nan     |          0.000 |         1.000 |               0.500 |
| humbug__svm_rbf           |           nan     |          0.000 |         1.000 |               0.500 |

## Áudio privado do Drive

O usuário identifica a gravação como enxame de Aedes na caixa de vidro. As 107 janelas foram mantidas no denominador, inclusive duas inválidas. A proporção reconhecida é sensibilidade nesse áudio, não acurácia de um detector binário; não há negativos para medir falsos positivos. O áudio não foi treinado, copiado para o relatório ou publicado.

| model                     |   true_positive |   known_windows |   aedes_recall |
|:--------------------------|----------------:|----------------:|---------------:|
| dryad+humbug__extra_trees |               2 |             107 |          0.019 |
| dryad+humbug__logistic    |               0 |             107 |          0.000 |
| dryad+humbug__svm_rbf     |              32 |             107 |          0.299 |
| dryad__extra_trees        |               0 |             107 |          0.000 |
| dryad__logistic           |               1 |             107 |          0.009 |
| dryad__svm_rbf            |               1 |             107 |          0.009 |
| humbug__extra_trees       |               0 |             107 |          0.000 |
| humbug__logistic          |               0 |             107 |          0.000 |
| humbug__svm_rbf           |              11 |             107 |          0.103 |

## Testes históricos por base

| model                     | scope             |   aedes_recall |   specificity |   balanced_accuracy |
|:--------------------------|:------------------|---------------:|--------------:|--------------------:|
| dryad__logistic           | historical_dryad  |          0.964 |         0.840 |               0.902 |
| dryad__logistic           | historical_humbug |          0.169 |         0.711 |               0.440 |
| dryad__logistic           | historical_ufrgs  |          0.472 |         0.890 |               0.681 |
| dryad__svm_rbf            | historical_dryad  |          0.947 |         0.948 |               0.948 |
| dryad__svm_rbf            | historical_humbug |          0.006 |         0.994 |               0.500 |
| dryad__svm_rbf            | historical_ufrgs  |          0.055 |         0.993 |               0.524 |
| dryad__extra_trees        | historical_dryad  |          0.939 |         0.950 |               0.944 |
| dryad__extra_trees        | historical_humbug |          0.000 |         0.998 |               0.499 |
| dryad__extra_trees        | historical_ufrgs  |          0.000 |         1.000 |               0.500 |
| humbug__logistic          | historical_dryad  |          0.000 |         0.979 |               0.490 |
| humbug__logistic          | historical_humbug |          0.877 |         0.898 |               0.887 |
| humbug__logistic          | historical_ufrgs  |          0.000 |         1.000 |               0.500 |
| humbug__svm_rbf           | historical_dryad  |          0.000 |         0.979 |               0.489 |
| humbug__svm_rbf           | historical_humbug |          0.909 |         0.887 |               0.898 |
| humbug__svm_rbf           | historical_ufrgs  |          0.000 |         0.973 |               0.486 |
| humbug__extra_trees       | historical_dryad  |          0.000 |         0.953 |               0.476 |
| humbug__extra_trees       | historical_humbug |          0.903 |         0.830 |               0.866 |
| humbug__extra_trees       | historical_ufrgs  |          0.000 |         0.897 |               0.449 |
| dryad+humbug__logistic    | historical_dryad  |          0.861 |         0.862 |               0.862 |
| dryad+humbug__logistic    | historical_humbug |          0.857 |         0.847 |               0.852 |
| dryad+humbug__logistic    | historical_ufrgs  |          0.225 |         0.932 |               0.578 |
| dryad+humbug__svm_rbf     | historical_dryad  |          0.972 |         0.927 |               0.950 |
| dryad+humbug__svm_rbf     | historical_humbug |          0.922 |         0.890 |               0.906 |
| dryad+humbug__svm_rbf     | historical_ufrgs  |          0.011 |         0.973 |               0.492 |
| dryad+humbug__extra_trees | historical_dryad  |          0.928 |         0.938 |               0.933 |
| dryad+humbug__extra_trees | historical_humbug |          0.890 |         0.918 |               0.904 |
| dryad+humbug__extra_trees | historical_ufrgs  |          0.007 |         0.911 |               0.459 |

A SVM Dryad atinge 94,8% de acurácia balanceada em Dryad e aproximadamente 50% em HumBug. A SVM HumBug atinge 89,8% em HumBug e 48,9% em Dryad. Essa queda entre fontes reforça a dependência do domínio, mas não identifica isoladamente microfone, sexo ou ruído como causa. Misturar ambas ajuda nas duas fontes conhecidas, sem resolver a nova fonte nem o Drive.

Nenhum candidato atingiu simultaneamente sensibilidade e especificidade de 60% na amostra inédita. Melhor resultado no Drive: SVM Dryad+HumBug, 32/107 (29,9%); no teste inédito esse mesmo modelo detectou apenas 4% dos Aedes. Regressão Dryad detectou 41,6% dos Aedes inéditos, mas apenas 1/107 do Drive. Não foi demonstrada confiabilidade.

Os intervalos em new_file_bootstrap.csv reamostram arquivos por classe, 2000 vezes; não são intervalos por indivíduos biológicos. Arquivos novos foram verificados por nomes e bytes; esta rodada não fez auditoria completa de fragmentos PCM coincidentes. As divisões anteriores têm limitações de metadados e os testes históricos não são independentes das decisões de pesquisa anteriores.
