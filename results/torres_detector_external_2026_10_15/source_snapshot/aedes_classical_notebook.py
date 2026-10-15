"""Read-only comparison section for the study notebook."""
import nbformat as nbf


def cells():
    return [nbf.v4.new_markdown_cell('''## 26. Outras técnicas de ML para Aedes aegypti

Regressão logística, SVM RBF, Extra Trees e gradient boosting com 122 características acústicas. Mesmos grupos de treino/validação/calibração da CNN binária; UFRGS e áudio privado fora do treino. Escolha pela AUC ponderada da validação, antes do novo teste externo. Escores e limiares separados por modelo; a comparação mostra sensibilidade e falsos positivos juntos.

As análises históricas são exploratórias. O novo teste tem somente cinco arquivos, três janelas positivas e cinco negativas; não comprova confiabilidade. Mais indicações no áudio de enxame não demonstram maior acerto. Nenhum firmware foi compilado. [Relatório completo](reports/comparacao_ml_binario_aedes.md).'''),
            nbf.v4.new_code_cell('''import json
from pathlib import Path
import pandas as pd
from IPython.display import Image, display
classical_audit = json.loads(Path('results/aedes_classical/training_audit.json').read_text())
print('Escolhido somente pela validação:', classical_audit['selected_by_validation'])
display(pd.read_csv('results/aedes_classical/comparison.csv')[['model', 'domain', 'emitted_aedes_recall', 'emitted_false_positive_rate', 'known_coverage']].round(4))
display(Image(filename='reports/assets/aedes_classical_comparison.png'))
display(pd.read_csv('results/aedes_classical/fresh_external/metrics.csv').round(4))
display(pd.read_csv('results/aedes_classical/explanation_gradient_boosting.csv').head(12))
''')]
