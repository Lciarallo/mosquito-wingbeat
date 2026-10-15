"""Notebook results for the computer-only residual CNN."""
import nbformat as nbf


def cells():
    return [
        nbf.v4.new_markdown_cell('''## 21. Modelo grande de espectrogramas — avaliação em computador

Por solicitação do usuário, esta rodada não tenta encaixar o modelo no Arduino.
A CNN residual tem duas entradas derivadas do som: espectrograma log-mel e
contraste espectral local. FFT de 2.048 amostras, hop de 256, 128 bandas entre
80 e 4.000 Hz, sem preenchimento no tempo. Há quatro blocos residuais,
uma camada de 512 unidades e saídas separadas para 20 espécies e presença.

As partições são as mesmas da rodada ampliada. Não há janelas da mesma origem
em treino e teste. O ruído misturado no treino vem apenas do treino; sobreposição
de sons da mesma espécie é aumento sintético, sem validar enxames reais.
Validação escolhe a época; calibração define temperatura e limiares. Depois de
congelar pesos/limiares, busca-se outro áudio externo, excluindo a gravação
MosquitoSong+ já examinada na rodada anterior e qualquer origem do treino.

Reprodução: `.venv/bin/python train_large_cnn.py`. A previsão de arquivos está
em `test_large_audio.py`. Nenhum firmware é gerado ou compilado nesta rodada.
'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown, Image
large = Path("results/large_cnn")
display(pd.Series(json.loads((large/"protocol.json").read_text())).to_frame("Protocolo"))
display(pd.read_csv(large/"comparison.csv").round(4))
display(pd.read_csv(large/"per_class.csv").round(4))
'''),
        nbf.v4.new_markdown_cell('''### 21.1 Aprendizado, teste adicional e decisão

O crescimento do modelo é uma intervenção exploratória motivada pelas falhas
anteriores; os testes históricos já foram examinados no projeto. A nova gravação
buscada depois do treinamento é uma verificação adicional, não uma estimativa
da acurácia das 20 espécies. Verificar precisão, cobertura, erros, ruído de fundo
e possíveis regressões conjuntamente. O áudio privado continua fora do treino,
com identificação de Aedes aegypti informada pelo usuário.
'''),
        nbf.v4.new_code_cell('''history = pd.read_json(large/"history.json")
display(history)
display(Image(filename="reports/assets/large_cnn_comparison.png"))
display(pd.Series(json.loads((large/"training_audit.json").read_text())).to_frame("Congelamento"))
display(pd.Series(json.loads((large/"fresh_external/analysis.json").read_text())).to_frame("Novo áudio"))
display(pd.Series(json.loads((large/"reliability_decision.json").read_text())).to_frame("Decisão"))
display(Markdown("[Relatório do modelo grande](reports/modelo_grande_cnn.md)"))
'''),
    ]
