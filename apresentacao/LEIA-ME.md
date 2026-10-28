# Apresentação: "Zzzz… é Aedes?"

Apresentação didática do projeto (25 slides): o problema, os métodos de ML, os resultados, a interpretabilidade e o experimento de mistura de datasets.

- `Apresentacao_Mosquito_Interpretabilidade.html`: versão animada e autocontida. Abra no navegador; use as setas ou clique para navegar e `f` para tela cheia.
- `Apresentacao_Mosquito_Interpretabilidade.pdf`: versão estática.
- `experimento_mistura/`: scripts e resultados (JSON) dos experimentos mostrados nos slides 12–14 e 20–23.

## Experimento de mistura (resumo)

| Arquivo | O que faz |
|---|---|
| `probe_torres.py` | Probe de origem: Aedes do Torres × Aedes originais |
| `mistura_torres_funchal.py` | Mistura Torres + Funchal: divisão por clipe × por fonte, probe de origem com 3 fontes |
| `pool_features.py` | Monta o dataset único (atributos de todas as fontes) em `pool.npz`, fora do Git |
| `melhorias.py`, `melhorias2.py` | Variantes: todas as fontes, normalização por fonte, limiar, ensemble, dois estágios |
| `so_torres_funchal.py` | Aedes só de Torres + Funchal no treino, testado nos Aedes das outras fontes |
| `aedes_torres_funchal.py` | Receita final (slides 20–23): Aedes de Torres + Funchal; outras fontes só com não-Aedes |

Todos são **retrospectivos**: usam dados já conhecidos do projeto e não constituem teste cego. Os atributos vêm de `results/torres_aedes_transfer_2026_10_15/` e dos caches de áudio locais, que não são versionados (ver o README principal). A extração de atributos também usa `train_torres_aedes_transfer.py`, que fica só na cópia local (`results/torres_aedes_transfer_2026_10_15/source/`). Executar com o ambiente `.venv-v3`, a partir da raiz do repositório, por exemplo `python apresentacao/experimento_mistura/pool_features.py`.

Limites principais: os ~95% de detecção no Torres refletem reconhecimento da fonte (o Torres só tem Aedes e não tem identificação de indivíduo); Aedes de aparelhos fora do Torres e do Funchal ainda não são reconhecidos (AUROC 0,41 em `so_torres_funchal.json`).
