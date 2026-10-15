"""Executed, read-only summary cells for the additional-source experiment."""
import nbformat as nbf

def cells():
 return [nbf.v4.new_markdown_cell('''## 22. Mais datasets: ajuste multifonte e teste por origem

Foram baixados os WAV de Figshare, UFRGS, MosquitoSong, o ZIP Development do
BioDCASE e as misturas Synthetic Swarm. Treinamento adiciona Figshare e
MosquitoSong, preserva as partições antigas e aceita 0,3 s reais sem repetir ou
preencher a onda. UFRGS inteira fica fora do treino, validação e calibração.

Cortes/canais irmãos ficam na mesma partição. Figshare tem identidade de origem
insuficiente para justificar teste aleatório de recortes: seu uso é adaptação.
Aedes MosquitoSong de cinco dias é desafio retrospectivo; a identidade individual
entre idades não está comprovada. BioDCASE D1–D4 é benchmark auxiliar, com
remoção de duplicatas binárias, sem alegar independência dos corpora anteriores.
D5 tem origem/modalidade pendente e não entra. Misturas Synthetic Swarm não
entram porque falta rastreabilidade completa dos ancestrais Abuzz.

A CNN anterior foi preservada. Seleção usa validação e calibração separadas.
Após congelar o ajuste, cinco novos Aedes foram selecionados na fonte pública
HumBug, excluindo as origens anteriores, e avaliados sem retoque de pesos ou
limiares. O Arduino permanece aguardando autorização posterior explícita.
'''),nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown, Image
multi = Path("results/multisource")
display(pd.read_csv(multi/"additional_counts.csv"))
display(pd.Series(json.loads((multi/"data_audit.json").read_text())).to_frame("Auditoria de dados"))
display(pd.read_csv(multi/"comparison.csv").round(4))
display(Image(filename="reports/assets/multisource_comparison.png"))
'''),nbf.v4.new_markdown_cell('''### 22.1 Ganhos, rejeições e falhas externas

Mais dados produzem ganhos parciais, não confiabilidade global. Comparar acerto
sem rejeição com cobertura e acerto nas emissões. Na UFRGS, a nova versão ainda
não emite identificação correta de Aedes. No enxame privado, remove as emissões
erradas da CNN anterior, mas ainda não identifica Aedes. Sua identidade vem do
usuário e não há anotação de eventos por instante.

Os cinco novos áudios HumBug têm novas origens, mas a campanha/microfone são
familiares. Ambas as CNNs identificam quatro arquivos; a nova rejeita mais
janelas. Não apresentar isso como melhora independente de generalização.
'''),nbf.v4.new_code_cell('''display(pd.read_csv(multi/"fresh_external/summary.csv"))
display(pd.read_csv(multi/"fresh_external/aggregate.csv"))
display(pd.Series(json.loads((multi/"biodcase_audit.json").read_text())).to_frame("Benchmark auxiliar"))
display(pd.Series(json.loads((multi/"reliability_decision.json").read_text())).to_frame("Decisão"))
display(Markdown("[Relatório multifonte completo](reports/retreinamento_multifonte.md)"))
''')]
