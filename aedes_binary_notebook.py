"""Read-only notebook section; no automatic training or firmware compilation."""
import nbformat as nbf

def cells():
    return [nbf.v4.new_markdown_cell('''## 25. Detector binário de Aedes aegypti

CNN dedicada: Aedes aegypti versus outras espécies rotuladas/ruído. Mosquito sem espécie resolvida não vira negativo. Pesos por espécie/origem/gravação; mesmas partições do estudo multifonte. A UFRGS e o áudio privado ficam fora do treino. Há rejeição explícita: INCERTO ou AUDIO_INVALIDO não significam não-Aedes.

O modelo é experimental: melhora a sensibilidade em algumas fontes, mas produz muitos falsos positivos no HumBug. Nenhuma adaptação para Arduino. [Relatório e comando de uso](reports/detector_binario_aedes.md). As tabelas históricas foram examinadas antes; somente a nova seleção HumBug foi obtida após congelar este modelo, e não representa um novo domínio.'''),nbf.v4.new_code_cell('''import json
import pandas as pd
from pathlib import Path
binary_audit = json.loads(Path('results/aedes_binary/training_audit.json').read_text())
display(pd.read_csv('results/aedes_binary/historical_metrics.csv').round(4))
display(pd.read_csv('results/aedes_binary/fresh_external/metrics.csv').round(4))
print('Modelo congelado:', binary_audit['checkpoint_sha256'])
''')]
