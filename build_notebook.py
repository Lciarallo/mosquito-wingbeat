"""Assemble the full Portuguese research notebook, including downloader code."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
sections = {}
for chunk in (ROOT / "notebook_code.py").read_text().split("# %% ")[1:]:
    name, source = chunk.split("\n", 1)
    sections[name.strip()] = source.strip()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

md("""
# Mosquitos pelo som das asas: literatura, dados reais e classificação

**Notebook de pesquisa em português.** Leitura dos três PDFs enviados, download completo do
dataset **10.5061/dryad.98d7s**, auditoria e experimentos executáveis com sinais acústicos.
Resultados publicados e resultados locais são apresentados separadamente. Os valores locais
aparecem nas saídas das células.

O percurso é: literatura → integridade → inventário → espectros/MFCC/F0 → avaliação por grupos
→ classificadores tradicionais e CNNs → TinyML de quatro classes → aparelhos não vistos
→ ruído → abstenção → INT8/C → conclusões.

**Pergunta operacional:** quanta informação sobre espécie permanece ao reservar gravações
inteiras e ao trocar aparelho ou acrescentar ruído? O notebook estima classificação exploratória;
não estima incidência de doenças.
""")
md("""
## 1. Leitura comparada dos três trabalhos

| Trabalho | Sensor e dados | Método | Resultado publicado e contexto |
|---|---|---|---|
| [Mukundarajan et al., 2017](https://elifesciences.org/articles/27854) | Celulares, 20 espécies; referências de frequência de fêmeas | STFT, distribuições de frequência, identificação com filtro geográfico | Aproximadamente 35% → 65% em média nas espécies testadas ao incluir localização |
| [Altayeb, Zennaro e Rovai, 2022](https://doi.org/10.1145/3524458.3547258) | Subseleção do Dryad e ruídos; quatro classes | Espectrograma, CNN1D, quantização, Edge Impulse | 93,8% de acurácia geral de teste no subconjunto/protocolo próprios |
| [Fanioudakis, Geismar e Potamitis, 2018](https://doi.org/10.23919/EUSIPCO.2018.8553542) | **Sensor óptico**, Wingbeats: 279.566 eventos, seis espécies, ambos os sexos | PSD, espectrogramas e redes profundas | 96% com DenseNet121 no holdout aleatório de 20% dos eventos ópticos |

As porcentagens respondem a problemas diferentes. Wingbeats óptico não é o corpus de celulares
baixado aqui. Implementamos estratégias inspiradas nos artigos e extensões próprias; não
reproduzimos exatamente os pesos, recortes, critérios de limpeza ou percentuais dos autores.
Os PDFs originais consultados ficam apenas na cópia local; os links acima apontam às publicações.
""")
md("""
### 1.1 Celulares: distribuição de frequência e limitações

O artigo de 2017 compara a medição acústica com vídeo de alta velocidade e entre aparelhos.
A identificação usa a distribuição de frequências durante o voo, não somente uma nota média.
Sobreposições entre espécies limitam o método; população, temperatura, sexo e coleta importam.
O resultado de 65% não é uma avaliação cega com 20 classes uniformes.

Nos métodos, a regra descrita **soma densidades por janela** para evitar que um outlier com
probabilidade zero vete uma espécie. Essa regra é diferente da log-verossimilhança convencional.
Implementamos ambas, com suavização explícita. Janelas STFT adjacentes são correlacionadas;
escores não equivalem a probabilidades calibradas de confiança.

O ganho com localização usa uma matriz de presença por país e localização escolhida em simulação.
Não reconstruímos essa matriz nem inventamos GPS, sexo ou temperatura para arquivos sem
metadados suficientes. Ver métodos nas pp. 17–19 e resultados nas pp. 9–14 do PDF.
""")
md("""
### 1.2 TinyML: tarefa, representação e implantação

Altayeb et al. convertem áudio para 16 kHz, selecionam quatro classes e usam janelas de 1 s
com avanço de 250 ms. O espectrograma usa frames de 25 ms, avanço de 12,5 ms e 128 bandas.
A rede tem convoluções de 32/64 filtros, pooling e dropout de 50%. O treinamento usa
LR 0,001, batch 32 e early stopping; a tabela reporta 93,8% no teste.
RAM/flash/latência são avaliadas para quantização e compilação EON.

Baixamos os **ruídos reais publicados pelos autores** no
[repositório oficial](https://github.com/Mjrovai/wingbeat-mosquito-tinyml), fixado em um commit.
Cortes da mesma fonte de ruído ficam juntos. Nossa tarefa de quatro classes usa mais gravações
do acervo auditado. A CNN é compacta, com outra arquitetura/frontend e orçamento de 25 épocas.
A exportação INT8 adicional usa um modelo linear, não a rede EON. Não medimos Arduino,
TFLite Micro, LoRaWAN, bateria ou energia.

No artigo, INT8 + EON tem estimativas de **15,2 KB de RAM / 51,6 KB de flash**, com latências
de **133/27/71 ms** para Nano/Portenta/Wio. Essas medidas pertencem ao pipeline dos autores
e não são comparáveis ao payload isolado do nosso classificador linear.
""")
md("""
### 1.3 Deep learning óptico e estratégias adicionais

Fanioudakis et al. registram passagens de insetos em gaiolas por um sensor de luz. Cada evento
tem 5.000 amostras a 8 kHz. Comparam sinal bruto, PSD de Welch e espectrogramas, com a mesma
divisão aleatória de eventos. Informação abaixo de 100 Hz inclui movimento corporal no
sensor óptico; não deve ser automaticamente atribuída ao microfone.

Também exploram clustering, Grad-CAM e transferência. Aqui priorizamos representações
acústicas, modelos pequenos e generalização. Não treinamos DenseNet no Wingbeats nem
afirmamos reproduzir a validação óptica.

Resultados da Tabela II do artigo, todos no mesmo holdout óptico:

| Modelo | Entrada | Acurácia publicada |
|---|---|---|
| DenseNet121 | Espectrograma | 96,00% |
| CNN de cinco camadas | PSD | 92,10% |
| CNN de cinco camadas | Sinal bruto | 91,20% |
| InceptionV3 | Espectrograma | 95,20% |
| MobileNet | Espectrograma | 95,62% |
| Xception | Espectrograma | 92,18% |
| NASNetMobile | Espectrograma | 94,85% |
| XGBoost | PSD | 81,81% |
| LightGBM | PSD | 82,40% |

[MosquitoSong+ (Supratak et al., 2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121)
motiva uma extensão: adicionar ruído ao treino e testar fontes de ruído reservadas.
Adotamos mistura com SNR explícito e outra rede/acervo/duração. Esta adaptação não reproduz
os resultados publicados de MosquitoSong+.
""")

notes = {
"Setup": """
## 2. Ambiente e configuração

O notebook contém os algoritmos e o downloader; pode ser executado na própria pasta em
Jupyter/VS Code ou Colab com armazenamento suficiente. Instale **requirements.txt**.
O sistema precisa de **libarchive**, **ffmpeg/ffprobe** e compilador C (**cc**) para a exportação.
Dados e caches ocupam vários GB. Arquivos originais ficam em **data/** e saídas em **results/**.
Seeds e hiperparâmetros são fixados; versões são registradas. GPU é opcional.
Alguns kernels ROCm/CUDA não garantem determinismo bit a bit; o manifesto registra esse limite.
""",
"Download": """
## 3. Download completo e integridade

Os metadados Dryad informam **20 RARs / 1.232.964.187 bytes** na versão auditada.
Usamos o [espelho Zenodo com o mesmo DOI](https://zenodo.org/records/4964774), pois as rotas
Dryad retornaram 401/403 neste ambiente. Antes do download, nomes, tamanhos e MD5 de ambos
os repositórios devem coincidir. Depois, cada arquivo é verificado por MD5 e recebe SHA-256 local.
As URLs efetivamente utilizadas ficam no manifesto.

RAR é extraído por libarchive; caminhos dos arquivos compactados são conferidos, downloads são
atômicos e arquivos íntegros são reutilizados. Ruídos TinyML têm commit/SHA-256 fixados.
A licença declarada do dataset Dryad/Zenodo é CC0. O código dos autores tem licença própria
e não foi incorporado ao classificador.
""",
"Inventory functions": """
## 4. Inventário e rótulos

O rótulo de espécie vem da pasta, **não de anotação por segundo**. Arquivos **Background**
e ruídos identificados pelos autores recebem classe **noise**.
Inspecionamos formatos, taxas, durações e hashes.

Para Aedes sierrensis priorizamos recortes manualmente isolados; outras versões e cópias
ficam fora da avaliação. Outros recortes **Snipped** sem ligação segura com a origem são
excluídos. Formatos comprimidos são decodificados quando não há WAV correspondente.
Duplicatas exatas e conflitos de rótulo são registrados.

Aparelho e sessão são inferidos de pastas, com **unknown** quando insuficientes.
Não são IDs certificados de indivíduo/data/sessão. Podem persistir relações entre
gravações que os nomes e hashes não permitem reconstruir.
""",
"Feature functions": """
## 5. Sinais, F0, PSD e MFCC

Reamostramos com anti-alias para 16 kHz e combinamos canais por média. O frontend Mel usa
100–3900 Hz, compatível com a faixa de arquivos originalmente a 8 kHz.
Reamostrar de 8 para 16 kHz não acrescenta informação acima do Nyquist original.

Janelas de **1 s sem sobreposição**, no máximo 60 por arquivo, distribuídas ao longo da gravação.
Arquivos curtos e janelas praticamente nulas são excluídos. O download inclui todo o dataset;
este limite diz respeito à amostra computacional.

Cada janela gera 13 médias/13 desvios de MFCC, PSD em frequências fixas e descritores
de frequência/harmônicos. Para F0, usamos STFT de 200 ms, passo de 20 ms e bins de 5 Hz.
A busca em 200–900 Hz amplia a faixa do foco original em fêmeas. É uma estimativa por pico;
pode selecionar ruído/harmônicos. Não selecionamos sexo a partir de metadados curados.
As CNNs usam 40 bandas Mel, FFT de 512 e avanço de 256 (32/16 ms).

Também verificamos formas de onda idênticas após reamostragem. Arquivos conectados recebem
o mesmo grupo; trechos repetidos/conflitantes não são contados duas vezes. Isso reduz
vazamento verificável, sem provar independência entre indivíduos/colônias.
""",
"Validation functions": """
## 6. Avaliação por gravação/fonte

Distribuímos grupos inteiros em três dobras estratificadas **no nível do grupo**, depois
alocamos as janelas. Gravações e trechos relacionados ficam juntos. Asserções verificam
grupos e formas de onda idênticas entre treino e teste.
Espécies com menos de três grupos seriam excluídas e reportadas.

Padronização e modelos são ajustados no treino. CNNs usam validação interna por grupos
para early stopping; teste externo fica reservado. Pesos/amostragem equilibram classes
e grupos, evitando domínio de arquivos longos.

Reportamos accuracy, balanced accuracy/macro recall e macro-F1 por janela.
Por grupo, agregamos escores e calculamos macro recall. Bootstrap reamostra grupos
dentro de cada espécie. Os intervalos são descritivos e condicionais a este acervo/modelo:
não incorporam toda a dependência por colônia ou incerteza populacional.
Comparar oito modelos nas mesmas dobras é exploratório; não há teste final independente.
""",
"Classical models": """
## 7. Métodos tradicionais

Comparamos classe majoritária, distribuição de F0 com soma de densidades/log-verossimilhança,
PSD + SVM linear, MFCC + regressão logística e ExtraTrees com todos os descritores.
Referências de frequência são construídas apenas no treino.

Escore aditivo: $s_k = \\sum_b h_b p_{kb}$. Variante logarítmica:
$s_k = \\sum_b h_b \\log(p_{kb}+\\epsilon)$.
Softmax de escores/margens facilita agregação, mas não demonstra calibração.
Nomes de espécies/pastas não são features acústicas.
""",
"Acoustic overlap": """
### Sobreposição e erros

Bhattacharyya usa todas as gravações para **visualização descritiva**. Não entra no treino
ou na seleção de parâmetros. A confusão usa previsões fora da dobra.
Sobreposição de F0 não determina, sozinha, o resultado de modelos que usam harmônicos
e dinâmica espectral.
""",
"CNN functions": """
## 8. CNNs com validação interna

A CNN1D recebe bandas Mel como canais e aplica filtros no tempo. A CNN2D recebe
tempo × frequência e acrescenta ruído real em metade dos exemplos de treino,
com SNR de 5–25 dB. O pool de ruídos também fica reservado por grupo/fonte.
O orçamento de 25 épocas e as arquiteturas são fixos.

O frontend SciPy/Torch é comparado numericamente. Resultados inferiores a baselines
permanecem na tabela. Checkpoints são reutilizados somente quando as assinaturas
de dados/divisões/configuração coincidem.
""",
"Four-class TinyML task": """
## 9. Tarefa inspirada no TinyML: quatro classes

Usamos **aegypti**, **albopictus**, **other** e **noise** com ruídos reais.
Cortes da mesma fonte de ruído ficam juntos. A rede recebe somente ruídos de treino.
Partições e distribuição de classes diferem da tarefa de 20 espécies e do subconjunto
de 93,8% do artigo; os percentuais não são diretamente comparáveis.
**Other** reúne espécies conhecidas no acervo, sem validar insetos desconhecidos.
Os ruídos não cobrem toda a diversidade ambiental.
""",
"Device transfer": """
## 10. Teste com aparelho não visto

Reservamos um aparelho inteiro e treinamos com outros. Avaliamos espécies presentes
no teste e com pelo menos dois grupos no treino; o número de classes varia.
Grupos compartilhados são retirados do treino.
O teste ainda mistura população/sexo/ambiente/coleta. Resultados com conjuntos diferentes
de espécies não são um ranking causal de qualidade dos microfones.
""",
"Noise robustness": """
## 11. Teste de estresse com ruído reservado

Selecionamos previamente até oito janelas por espécie/dobra. Acrescentamos ruído a
20, 10 e 0 dB: fontes reais reservadas na dobra de teste e ruído gaussiano.
Usamos os modelos já treinados, sem ajustar no teste.
SNR é relativo ao **áudio observado**, que já pode conter ruído, e não ao mosquito puro.
Mistura controlada não equivale a validação prospectiva em campo.
Curvas representam médias das dobras, sem teste de significância.
""",
"Selective classification and metadata diagnostic": """
## 12. Abstenção e confundimento da aquisição

Limiares predefinidos mostram cobertura/acerto das amostras aceitas. Escores não foram
calibrados para campo. O teste só usa espécies conhecidas e não demonstra detecção
de espécie desconhecida.

Um diagnóstico separado usa somente aparelho e taxa declarada no cabeçalho.
O encoding é ajustado no treino; nomes de espécies/pastas são excluídos.
Desempenho sem áudio revela associações da aquisição com os rótulos.
""",
"INT8 functions": """
## 13. INT8 e exportação C

Quantizamos os pesos da logística MFCC por classe e entradas padronizadas por escala global;
a calibração usa treino. Entradas/pesos são INT8, acumulação INT32, bias/escalas finais float.
Comparamos previsões fora da dobra com o modelo original.

Exportamos **mosquito_mfcc_int8.h**, compilamos C e comparamos 400 previsões contra Python.
O arquivo é o modelo da dobra 0. O payload exclui frontend áudio/MFCC, FFT, firmware,
strings, RAM de trabalho e hardware. A validação é local; não mede Arduino/TFLite/EON.
""",
"Inference function": """
## 14. Classificar outra gravação

**classify_recording("arquivo.wav")** decodifica, reamostra e agrega escores.
A demonstração usa um arquivo inteiro reservado na dobra 0 e seu modelo.
Para novos dados, confirme qualidade, taxonomia e coleta. O rótulo da pasta não garante
mosquito em cada segundo; limiares de abstenção ainda precisam de validação externa.
""",
"Results and provenance": """
## 15. Conclusões verificáveis

O maior macro recall observado abaixo depende deste acervo/protocolo. Selecionar o maior
entre vários modelos não produz uma estimativa imparcial de desempenho de um produto novo.
Consulte também aparelhos reservados, ruído e suporte por espécie.

A auditoria resolve duplicatas/relações verificáveis. Persistem dependências por
indivíduo/colônia e espécies concentradas em poucas sessões. Gravações longas têm rótulos
fracos e não passaram por anotação temporal independente de mosquito.

Para avançar: novas populações/sessões com identificação de espécimes, anotações de atividade,
sexo/temperatura, conjunto externo reservado, rejeição com espécies/ruídos desconhecidos
e medição do pipeline completo no hardware. Ganhos de classificação não podem ser
atribuídos exclusivamente à biologia do batimento alar.

Manifests registram origens, hashes, ambiente, configuração, cobertura, exclusões e checagens.
Tabelas, figuras e modelos são regenerados pelo notebook.
""",
}

for name, source in sections.items():
    if name in notes:
        md(notes[name])
    if name == "Setup":
        source = source.replace(
            'assert (ROOT / "fetch_data.py").exists(), "Abra o notebook na pasta mosquito-wingbeat."\nsys.path.insert(0, str(ROOT))\nfrom fetch_data import fetch_dataset, fetch_noise',
            'sys.path.insert(0, str(ROOT))')
    if name == "Download":
        downloader = (ROOT / "fetch_data.py").read_text().split('\nif __name__ == "__main__":')[0]
        downloader = downloader.replace('ROOT = Path(__file__).resolve().parent', '# ROOT definido na configuração.')
        cells.append(nbf.v4.new_code_cell(downloader))
    cells.append(nbf.v4.new_code_cell(source))

cells.append(nbf.v4.new_code_cell("""
display(Markdown(
    f"**Resultado da etapa inicial:** {len(SPECIES)} espécies, {len(PRIMARY):,} janelas e "
    f"{segments.iloc[PRIMARY].group.nunique()} grupos. O maior macro recall por grupo "
    f"foi **{best.recording_macro_recall:.1%}** com **{best.model}**.\\n\\n"
    f"**Generalização:** no teste com aparelhos não vistos e sete ou mais classes, "
    f"o macro recall por grupo variou de "
    f"**{transfer_summary.loc[transfer_summary.n_classes >= 7, 'recording_macro_recall'].min():.1%}** a "
    f"**{transfer_summary.loc[transfer_summary.n_classes >= 7, 'recording_macro_recall'].max():.1%}**. "
    f"Os conjuntos de espécies variam entre aparelhos. Apenas aparelho/taxa de amostragem "
    f"já alcançaram **{metadata_diagnostic['balanced_accuracy']:.1%}** "
    f"de acurácia balanceada por janela, mostrando associação entre aquisição e rótulo.\\n\\n"
    f"**INT8:** acurácia balanceada por janela de **{quantized_metrics['balanced_accuracy']:.1%}**, "
    f"payload de **{payload_bytes} bytes** e **{len(c_test)}/{len(c_test)}** "
    f"previsões C/Python correspondentes. Frontend e hardware não foram medidos."
))
""".strip()))
md("""
## 16. Melhorias executadas: contraste, dinâmica e novos modelos

Esta extensão reutiliza **as mesmas janelas e três dobras por grupo** da etapa inicial.
Acrescenta contraste espectral local em 75 frequências (média/desvio), variação temporal
de MFCC e rastreamento de pico: 181 características novas, 293 no total. O contraste
remove uma referência espectral local ampla; não garante remoção de ruído ou de
diferenças entre microfones. O RBF usa as 112 características originais; árvores,
boosting e MLP usam as 293. Mudam representação e modelo, sem isolar cada contribuição.

O ensemble combina RBF, ExtraTrees e boosting com pesos iguais fixados. A MLP
293 -> 128 -> 64 -> 20 tem validação interna por grupos. A comparação é uma
**reanálise adaptativa do corpus já utilizado**, não um novo teste externo.
O bootstrap pareado por grupo é descritivo e condicionado às previsões observadas.
Os algoritmos desta extensão estão nos scripts incluídos no repositório.
""")
cells.append(nbf.v4.new_code_cell("""
import improve_device
improve_device.main()
improved_metrics = pd.read_csv(RESULTS / "improvements/species_metrics.csv")
display(improved_metrics[["model", "accuracy", "balanced_accuracy", "recording_macro_recall"]].round(4))
display(pd.read_csv(RESULTS / "improvements/paired_group_differences.csv").round(4))
ax = improved_metrics.plot.barh(x="model", y=["balanced_accuracy", "recording_macro_recall"], figsize=(10, 4))
ax.set_xlabel("Métrica (0–1)"); ax.legend(["Balanceada por janela", "Macro recall por grupo"])
plt.tight_layout(); plt.show()
""".strip()))
md("""
## 17. Presença de mosquito para Arduino: frontend completo e modelos compactos

Detecção binária e reconhecimento de 20 espécies são tarefas distintas. Os rótulos
de presença continuam **fracos, por arquivo**; não anotamos cada batimento/voo.
O novo frontend usa mono a 16 kHz, Hann de 512 amostras sem sobreposição e 31 frames
por decisão (**0,992 s**). Calcula 68 características de energia relativa, dispersão,
flatness e pico de 219–875 Hz. É incremental e não depende da normalização usando
uma janela inteira ou de áudio futuro. A implementação C++ guarda um frame de FFT.

Comparamos logística, árvore de profundidade 6, floresta de 32 árvores de profundidade 5
e uma **rede neural de 16 unidades ReLU** (1.121 parâmetros). Cada dobra externa
reserva grupos para calibração; outra divisão interna seleciona o candidato por ROC AUC
ponderada por fonte. A rede seleciona épocas por perda ponderada nessa validação interna
e treina novamente no conjunto de ajuste pelo mesmo número de épocas.
Padronizadores, calibradores, limiares e épocas não usam os grupos de teste externo.

A calibração binária usa pesos iguais para fontes positivas e negativas, equivalente
a uma mistura 50/50, **sem representar prevalência real**. Comparamos limiar 0,5
e metas de FPR médio por fonte de 5%/10% na calibração. Metas não garantem desempenho
no teste, e controlar alarmes pode perder sinais. Há somente 44 fontes de ruído.
""")
cells.append(nbf.v4.new_code_cell("""
import train_arduino
train_arduino.main()
arduino_metrics = pd.read_csv(RESULTS / "arduino/metrics.csv")
display(arduino_metrics[["model", "mode", "window_recall", "window_false_positive_rate",
                         "source_mean_recall", "source_mean_false_positive_rate"]].round(4))
selected = arduino_metrics.loc[arduino_metrics.model.eq("Escolha por validação interna")]
display(selected[["mode", "source_recall_ci_low", "source_recall_ci_high",
                  "source_false_positive_rate_ci_low", "source_false_positive_rate_ci_high"]].round(4))
ax = selected.plot.bar(x="mode", y=["source_mean_recall", "source_mean_false_positive_rate"], figsize=(9, 4), rot=0)
ax.set_ylabel("Média por fonte (0–1)"); ax.legend(["Recall positivo", "Falsos positivos no ruído"])
plt.tight_layout(); plt.show()
""".strip()))
md("""
### 17.1 Confirmação temporal e ruído persistente

A regra causal exige duas de três janelas completas e reinicia quando muda arquivo
ou há uma lacuna. Como o corpus amostra até 60 janelas por arquivo, muitas não são
contíguas. A comparação abaixo usa **os mesmos endpoints elegíveis** para uma janela
e para confirmação, sem tratar janelas espaçadas como áudio contínuo.
Restaram somente **duas fontes de ruído** nesse subconjunto; não há validação suficiente
de falsos alarmes contínuos. A correlação temporal pode manter ruídos como falsos positivos.
Estas taxas não permitem calcular alarmes/hora ou contar mosquitos individuais.
""")
cells.append(nbf.v4.new_code_cell("""
display(pd.read_csv(RESULTS / "arduino/persistence_metrics.csv").round(4))
import probe_arduino_noise
probe_arduino_noise.main()
arduino_noise = pd.read_csv(RESULTS / "arduino/noise_probe.csv")
display(arduino_noise.groupby(["mode", "noise", "added_snr_db"], dropna=False).positive_window_recall.mean().round(4))
""".strip()))
md("""
## 18. Explicabilidade e validação do processamento embarcado

Permutamos cinco famílias de características em conjunto, cinco repetições por dobra,
e medimos a queda de ROC AUC ponderada por fonte nos dados de teste reservados.
Dentro da família, preservamos a relação entre características; entre famílias,
a permutação pode criar entradas incomuns. É uma análise de **sensibilidade preditiva**,
sem explicação causal, identificação biológica exclusiva ou atribuição local SHAP/Grad-CAM.
A dispersão abaixo combina dobras/repetições correlacionadas, não é intervalo inferencial.
""")
cells.append(nbf.v4.new_code_cell("""
importance = pd.read_csv(RESULTS / "arduino/permutation_importance.csv")
importance_summary = importance.groupby("family").source_weighted_auc_drop.agg(["mean", "std"])
display(importance_summary.round(4))
importance_summary.sort_values("mean").plot.barh(y="mean", legend=False, figsize=(9, 3.5))
plt.xlabel("Queda de ROC AUC após permutar a família"); plt.tight_layout(); plt.show()
import verify_arduino
verify_arduino.main()
display(pd.Series(json.loads((RESULTS / "arduino/native_audit.json").read_text())).to_frame("Auditoria nativa"))
""".strip()))
md("""
### 18.1 Firmware Arduino e limites do aparelho

O sketch [MosquitoPresence.ino](firmware/MosquitoPresence/MosquitoPresence.ino) captura
PDM, extrai características, executa a rede e confirma candidatos em 2/3 janelas.
Está compilado para **Arduino Nano 33 BLE Sense / Sense Rev2**, CLI 1.5.1/core 4.6.0.
O Uno R3/Nano clássico tem memória insuficiente para este firmware; outro modelo
Arduino exige seu adaptador de captura e nova compilação/avaliação.

**Não houve upload ou teste em placa física**. O compilador informa memória estática,
sem medir pico de pilha/heap, consumo ou prazo em tempo real. Os testes C++/Python
validam equivalência numérica e lógica. O modelo exportado é da dobra 0; o resultado
agregado usa cada modelo de sua própria dobra, sem retreino para produção.
O limiar moderado (meta 10% na calibração) é o padrão do sketch, antes da confirmação.
Os guardas de silêncio/clipping da captura ainda não têm taxas de campo medidas.
A primeira confirmação requer 2,976 s de áudio; passagens muito curtas podem ser
perdidas. Não há validação por voo isolado.
""")
cells.append(nbf.v4.new_code_cell("""
compile_path = RESULTS / "arduino/compile_audit.json"
if compile_path.exists():
    display(pd.Series(json.loads(compile_path.read_text())).to_frame("Compilação registrada"))
else:
    print("Para recompilar: arduino-cli compile --fqbn arduino:mbed_nano:nano33ble firmware/MosquitoPresence")
default = selected.loc[selected["mode"].eq("Calibração FPR 10%")].iloc[0]
display(Markdown(
    f"**O aparelho ainda erraria:** no teste com rótulos por arquivo, o modo padrão "
    f"perdeu **{1-default.window_recall:.1%}** dos trechos positivos e marcou "
    f"**{default.window_false_positive_rate:.1%}** dos trechos de ruído como candidatos, "
    f"antes da confirmação. Isso não estima erros por mosquito individual.\\n\\n"
    "Para melhorar de forma verificável: coletar áudio com a placa no ambiente real, "
    "anotar atividade/distância e horas sem mosquito, reservar dias/locais, treinar "
    "com os ruídos locais e medir sensibilidade por evento, falsos alarmes/hora e latência. "
    "Não há evidência de superioridade sobre os artigos com protocolos diferentes."
))
""".strip()))
md("""
## Referências e fontes

1. Mukundarajan H. et al. (2017). *Using mobile phones as acoustic sensors for high-throughput mosquito surveillance*. eLife 6:e27854. [Artigo](https://elifesciences.org/articles/27854), [DOI](https://doi.org/10.7554/eLife.27854).
2. Dados, versão 02/10/2018: [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.98d7s), [espelho Zenodo do mesmo DOI](https://zenodo.org/records/4964774).
3. Altayeb M., Zennaro M., Rovai M. (2022). *Classifying mosquito wingbeat sound using TinyML*. GoodIT, 132–137. [DOI](https://doi.org/10.1145/3524458.3547258), [dados/código](https://github.com/Mjrovai/wingbeat-mosquito-tinyml).
4. Fanioudakis E., Geismar M., Potamitis I. (2018). *Mosquito wingbeat analysis and classification using deep learning*. EUSIPCO, 2410–2414. [DOI](https://doi.org/10.23919/EUSIPCO.2018.8553542). Wingbeats é outro corpus, óptico, e não foi baixado.
5. Supratak A. et al. (2024). *MosquitoSong+*. PLOS ONE 19:e0310121. [Artigo](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0310121).
6. Documentação: [SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html), [avaliação agrupada scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data), [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html), [PyTorch](https://docs.pytorch.org/docs/stable/index.html).
""")
notebook = nbf.v4.new_notebook(cells=cells, metadata=dict(
    kernelspec=dict(display_name="Python 3", language="python", name="python3"),
    language_info=dict(name="python", version="3.14.7"),
    title="Mosquitos: literatura, acústica e classificação"))
nbf.validate(notebook)
nbf.write(notebook, ROOT / "mosquito_wingbeat_estudo.ipynb")
print(f"Notebook criado: {len(cells)} células, {sum(c.cell_type == 'code' for c in cells)} de código.")
