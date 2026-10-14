"""Reproducible notebook section for the separate expanded Arduino model."""
import nbformat as nbf


def cells():
    return [
        nbf.v4.new_markdown_cell('''## 20. Ampliação com dados externos HumBugDB

Após falhas em exemplos externos de Anopheles arabiensis e Aedes aegypti,
treinamos novas redes com os dados existentes e uma amostra adicional do
[HumBugDB](https://zenodo.org/records/4904800). A seleção usa somente metadados,
com semente fixa, gravações posteriores a 2017 e limite por gravação/classe.
Arquivos históricos do CDC foram excluídos para reduzir risco de reutilização.

Trechos de uma mesma gravação permanecem na mesma partição. Os dois exemplos
já examinados viraram **adaptação**, sem valor de teste independente. O teste
Dryad da dobra 2 permaneceu separado, mas já foi analisado no trabalho anterior;
o teste novo HumBug foi reservado antes do treinamento. Ruídos participam do
treinamento da presença. Espécies externas não resolvidas ou fora das 20 classes
participam somente da presença; não são convertidas em classes exatas.

Há equilíbrio por classe, corpus e gravação de origem. Duas arquiteturas MLP
foram comparadas na validação; a calibração dos limiares usa outra partição.
A rejeição é uma regra, não uma 21ª classe treinada. O áudio privado de enxame
não foi usado como exemplo com espécie conhecida.

Reprodução: `.venv/bin/python fetch_humbug.py`, depois
`.venv/bin/python train_expanded.py`.
Os resultados estão em `results/expanded/`; os resultados anteriores continuam
sendo os da versão antiga. Por solicitação do usuário, esta etapa não gera nem
compila firmware: primeiro é necessário demonstrar confiabilidade, incluindo
novo áudio obtido somente após congelar o modelo. O instalador Windows já
publicado contém o modelo antigo.
'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown
expanded = Path("results/expanded")
counts = pd.read_csv(expanded / "data_counts.csv")
display(counts)
data_protocol = json.loads((expanded / "data_protocol.json").read_text())
display(pd.Series({key: value for key, value in data_protocol.items() if key != "groups"}).to_frame("Protocolo"))
'''),
        nbf.v4.new_markdown_cell('''### 20.1 Comparação no mesmo teste

Precisão entre as identificações aceitas deve ser lida junto com cobertura
e fração total corretamente identificada. Escolha forçada é uma medida auxiliar,
que ignora os limiares e a confirmação do firmware. Janelas de uma mesma gravação
são correlacionadas. Anotações externas não resolvidas não permitem afirmar que
toda emissão de uma espécie específica foi um erro: a coluna correspondente
mede emissão nesses trechos, não acurácia. As espécies sem novas gravações
independentes não têm nova validação externa demonstrada.
'''),
        nbf.v4.new_code_cell('''comparison = pd.read_csv(expanded / "comparison.csv")
display(comparison.round(4))
display(pd.Series(json.loads((expanded / "training_audit.json").read_text())).to_frame("Treinamento"))
if (expanded / "native_audit.json").exists():
    display(pd.Series(json.loads((expanded / "native_audit.json").read_text())).to_frame("Equivalência C++"))
if (expanded / "compile_audit.json").exists():
    display(pd.Series(json.loads((expanded / "compile_audit.json").read_text())).to_frame("Compilação"))
display(pd.read_csv(expanded / "per_class.csv").round(4))
display(pd.Series(json.loads((expanded / "reliability_decision.json").read_text())).to_frame("Liberação"))
'''),
        nbf.v4.new_markdown_cell('''### 20.2 Exemplos usados na adaptação

O quadro abaixo mostra os dois arquivos que motivaram a ampliação.
**Esses arquivos entraram no treino**, portanto eventual acerto é apenas
uma verificação de adaptação e não validação externa. A espécie da gravação
é a anotada pelos autores; não houve nova identificação entomológica.
Compilar e testar C++ em computador não valida microfone, distância ou
latência na Nano 33 BLE Sense física.
'''),
        nbf.v4.new_code_cell('''adaptation = pd.read_csv(expanded / "adaptation_examples.csv")
display(adaptation.groupby(["id", "label", "identified_species"]).size().rename("janelas").to_frame())
display(Markdown("[Relatório da ampliação](reports/treinamento_ampliado.md)"))
fresh = json.loads((expanded / "fresh_mosquitosong/analysis.json").read_text())
display(pd.DataFrame(fresh["model_results"]))
display(Markdown("**O áudio novo do MosquitoSong+ não foi reconhecido. A versão permanece em avaliação offline.**"))
'''),
    ]
