# Apresentação: "Zzzz… é Aedes?"

Apresentação para alunos da Eletrônica sobre o trabalho de um desenvolvedor de ML (23 slides): o processo, as ferramentas, os dados, os atributos, os modelos de árvore, a validação por grupo, as métricas (matriz de confusão, curva ROC, limiar), o resultado, a interpretabilidade, cada método de ML em poucas linhas de código, o uso de assistentes de IA no código (Claude Code, Codex, Hermes, DeepSeek) e a ponte com o hardware (Arduino).

- `Apresentacao_Mosquito_Interpretabilidade.html`: versão animada e autocontida. Abra no navegador; setas ou clique para navegar, `f` para tela cheia.
- `Apresentacao_Mosquito_Interpretabilidade.pdf`: versão estática.
- `demo/detector_aedes_demo.ipynb`: material extra (não aparece nos slides), caso queira mostrar o código rodando no VS Code. Já vem executado, com os gráficos salvos; rodar de novo leva cerca de 1 minuto.
- `experimento_mistura/`: scripts e resultados (JSON).

## Demo opcional no VS Code

1. Abra a pasta do repositório no VS Code com as extensões Python e Jupyter.
2. Abra `apresentacao/demo/detector_aedes_demo.ipynb` e escolha o kernel do `.venv-v3`.
3. As saídas já estão no arquivo; para rodar ao vivo, use "Run All". O notebook lê `experimento_mistura/pool.npz` (atributos de todas as fontes), que não é versionado: gere-o com `pool_features.py` na cópia local.

## Scripts

| Arquivo | O que faz |
|---|---|
| `pool_features.py` | Monta o dataset único (atributos de todas as fontes) em `pool.npz`, fora do Git |
| `aedes_torres_funchal.py` | Modelo final: Aedes de Torres + Funchal; outras fontes só com não-Aedes; normalização e pesos por fonte; dois estágios |
| `curvas_modelo_final.py` | Matriz de confusão, curva ROC e métricas por limiar do modelo final |
| `interpretabilidade_final.py` | Importância por banda de frequência (permutação) e probe de origem no modelo final |
| `probe_torres.py`, `mistura_torres_funchal.py`, `melhorias.py`, `melhorias2.py`, `so_torres_funchal.py` | Experimentos exploratórios anteriores, mantidos como registro |

Todos são **retrospectivos** (dados já conhecidos do projeto; não é teste cego). Dependem de atributos e caches de áudio locais e de `train_torres_aedes_transfer.py`, que ficam só na cópia local. Limites: a detecção de ~95% no Torres reflete reconhecimento da fonte (o Torres só tem Aedes); Aedes de aparelhos fora do Torres e do Funchal ainda não são reconhecidos (AUROC 0,41 em `so_torres_funchal.json`).
