# Apresentação: "Zzzz… é Aedes?"

Apresentação para alunos da Eletrônica sobre o trabalho de um desenvolvedor de ML (29 slides, com referências no penúltimo): o processo, as ferramentas, os dados, os atributos, os modelos (árvores, redes neurais MLP e CNN, e a SVM, que é o modelo final), o ranking de 9 modelos na mesma prova, a validação por grupo, as métricas (matriz de confusão, curva ROC, limiar), o resultado, a interpretabilidade, cada método de ML em poucas linhas de código, o uso de assistentes de IA no código (Claude Code, Codex, Hermes, DeepSeek) e a ponte com o hardware (Arduino).

- `Apresentacao_Mosquito_Interpretabilidade.html`: versão animada e autocontida. Abra no navegador; setas ou clique para navegar, `f` para tela cheia.
- `Apresentacao_Mosquito_Interpretabilidade.pdf`: versão estática.
- `demo/detector_aedes_demo.ipynb`: material extra (não aparece nos slides), caso queira mostrar o código rodando no VS Code. Já vem executado, com os gráficos salvos; rodar de novo leva cerca de 1 minuto.
- `experimento_mistura/`: scripts e resultados (JSON).

## Vídeos

Os quatro vídeos (`videos/`) vêm de Mukundarajan H., Hol F. J. H., Castillo E. A., Newby C., Prakash M. *Using mobile phones as acoustic sensors for high-throughput mosquito surveillance*. **eLife** 6:e27854 (2017), [doi:10.7554/eLife.27854](https://doi.org/10.7554/eLife.27854), licença **CC BY 4.0**. Foram recortados e comprimidos (vídeos 2, 3, 4 e 5 do artigo). Estão embutidos no HTML (WebM e MP4), que continua funcionando offline como arquivo único; nos slides, cada vídeo toca sozinho, sem som, quando o slide abre.

| Arquivo | Slide | Conteúdo |
|---|---|---|
| `asa_camera_lenta` | 3 | *Culex tarsalis* filmado a 10 kHz, mostrado 100× mais devagar |
| `gravando_garrafa`, `gravando_decolagem` | 7 | Gravação com celular: mosquito na garrafa e decolando |
| `espectrograma_ao_vivo` | 9 | Espectrograma de *Anopheles gambiae* sendo montado |

## Demo opcional no VS Code

1. Abra a pasta do repositório no VS Code com as extensões Python e Jupyter.
2. Abra `apresentacao/demo/detector_aedes_demo.ipynb` e escolha o kernel do `.venv-v3`.
3. As saídas já estão no arquivo; para rodar ao vivo, use "Run All". O notebook lê `experimento_mistura/pool.npz` (atributos de todas as fontes), que não é versionado: gere-o com `pool_features.py` na cópia local.

## Scripts

| Arquivo | O que faz |
|---|---|
| `pool_features.py` | Monta o dataset único (atributos de todas as fontes) em `pool.npz`, fora do Git |
| `aedes_torres_funchal.py` | Receita de dados com árvores (modelo final anterior): Aedes de Torres + Funchal; outras fontes só com não-Aedes; normalização e pesos por fonte; dois estágios |
| `redes_neurais.py` | MLP e CNN (PyTorch) na mesma receita do modelo final, comparadas com as árvores (3 repetições) |
| `seq_features.py` | Sequências no tempo (32 instantes × 16 faixas) alinhadas ao `pool.npz`, para LSTM/Bi-LSTM (`seq.npy`, fora do Git) |
| `metodos_restantes.py` | Logística, SVM, DANN, CORAL + árvores, LSTM e Bi-LSTM na mesma receita (3 repetições; LSTM na GPU via ROCm) |
| `svm_final.py` | **Modelo final dos slides (SVM RBF)**: matriz de confusão, ROC, métricas por limiar e importância por faixa (`curvas_svm.json`, `interpretabilidade_svm.json`) |
| `curvas_modelo_final.py` | Matriz de confusão, curva ROC e métricas por limiar do modelo final |
| `interpretabilidade_final.py` | Importância por banda de frequência (permutação) e probe de origem no modelo final |
| `probe_torres.py`, `mistura_torres_funchal.py`, `melhorias.py`, `melhorias2.py`, `so_torres_funchal.py` | Experimentos exploratórios anteriores, mantidos como registro |

Todos são **retrospectivos** (dados já conhecidos do projeto; não é teste cego). Dependem de atributos e caches de áudio locais e de `train_torres_aedes_transfer.py`, que ficam só na cópia local. Limites: a detecção de ~95% no Torres reflete reconhecimento da fonte (o Torres só tem Aedes); Aedes de aparelhos fora do Torres e do Funchal ainda não são reconhecidos (AUROC 0,41 em `so_torres_funchal.json`).
