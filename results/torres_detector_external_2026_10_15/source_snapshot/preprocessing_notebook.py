"""Read-only cells for paired preprocessing sensitivity analysis."""
import nbformat as nbf

def cells():
 return [nbf.v4.new_markdown_cell('''## 23. Pré-processamento adicional com pesos fixos

O modelo já remove a média, normaliza RMS por janela e usa log-mel/contraste
espectral. Esta análise testa cinco alternativas adicionais, definidas antes de
ver os resultados: highpass 120 Hz; bandpass 150–3500 Hz; bandpass 250–2500 Hz;
subtração espectral leve; e bandpass com subtração espectral. Pesos, temperatura,
limiares e rejeições de qualidade originais permanecem fixos. O áudio não é
repetido, concatenado nem prolongado; os filtros são offline, não causais.

É uma análise exploratória em testes já examinados. Os cinco Aedes novos da
rodada anterior agora são retrospectivos; não são cinco testes inéditos de
pré-processamento. Nenhum treinamento ou compilação de Arduino foi executado.
'''),nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown
prep = Path("results/preprocessing")
display(pd.Series(json.loads((prep/"protocol.json").read_text())).to_frame("Protocolo"))
display(pd.read_csv(prep/"comparison.csv").round(4))
display(pd.DataFrame(json.loads((prep/"private_audio_summary.json").read_text())))
display(pd.read_csv(prep/"ufrgs_negatives.csv"))
display(pd.Series(json.loads((prep/"decision.json").read_text())).to_frame("Decisão"))
display(Markdown("[Relatório de pré-processamento](reports/preprocessamento_adicional.md)"))
'''),nbf.v4.new_markdown_cell('''### Resultado e limite

Nenhum método identificou Aedes no enxame privado. A redução espectral leve
melhorou modestamente HumBug e a cobertura dos cinco Aedes retrospectivos,
mas piorou Dryad e UFRGS. Bandpass e highpass prejudicaram o modelo atual;
alguns aumentaram identificações de espécie em ruído. Não substituir o
frontend apenas por um ganho isolado. Se uma nova rodada testar filtros no
próprio treino, validação e calibração precisam usar o mesmo pipeline, e outro
áudio inédito deve ser buscado após o novo congelamento. Esse experimento
não prova que filtros nunca ajudem um modelo treinado para recebê-los.
''')]
