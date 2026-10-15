# Análise de lacunas: Torres et al. (2025)

## Objetivo autorizado
Aplicar as técnicas do artigo ao dataset dos autores e às bases do projeto. Manter os resultados anteriores e a exclusão de MosquitoSong do treino.

## Evidências lidas
- PDF fornecido: SHA-256 `32dc201db2ee94ade9c81f558ee4eba0c0a682dfdfb2c971f5bd0c1d43584dfc`.
- Repositório dos autores: https://github.com/NakanoMariko/Dengue-Infected-Mosquito-Detection, commit `6c96ccd5347cf5bafc9b424579700af7b5ffb7ce`.
- Dataset público: 300 tensores mel de 188 passos por 32 bandas, 150 rótulos por classe; não inclui identificação de indivíduo ou gravação. Os 300 tensores são distintos por hash exato. Dois WAVs públicos são exemplos, não o corpus completo em áudio.
- Técnicas: duas camadas LSTM/Bi-LSTM, 256 unidades, dropout 0,20, Adam, inferência estocástica por 100 passagens e abstenção conforme discordância.

## Lacunas e decisões
1. O artigo trata de infecção por DENV-2, enquanto Dryad tem rótulos de espécie. Executar tarefas separadas; não inventar rótulos de infecção para as gravações do projeto.
2. O texto usa divisão 80/20, mas o código público divide 260/40 e escolhe checkpoint pela perda nesse conjunto. Reexecução dos pesos originais será identificada como retrospectiva. Novos treinos terão treino, validação e teste separados.
3. A seleção de arquitetura no artigo usa teste. Na adaptação, arquitetura/época e normalização dependem somente de treino/validação.
4. A base JSON dos autores não permite comprovar separação por indivíduo. Agrupar duplicatas exatas por hash e explicitar a ausência dessa identidade; não chamar a divisão de validação independente por indivíduo.
5. Os splits históricos Dryad contêm um componente de colisão PCM atravessando as quatro partições. Purgar o componente do treino e da validação quando aparecer no teste e manter o mapa registrado. Não apresentar o resultado antigo da CNN/MLP como comparação nas mesmas condições; ajustar referência linear no novo treino.
6. A técnica de abstenção aumenta potencialmente a acurácia condicional ao reduzir cobertura. Reportar simultaneamente desempenho em todas as entradas, cobertura, abstenções e desempenho entre decisões aceitas. Concordância de 100 passagens não equivale a certeza biológica.
7. No dataset atual, adaptar frontend para áudio de 16 kHz, sem fingir que reamostragem recupera frequências ausentes. Preservar os 32 atributos por passo e a ordem temporal.
8. O artigo descreve 100 passagens de Monte Carlo; o script público executa 10. Fixar 100 nesta rodada e registrar que o gerador de dropout não reproduz exatamente a execução original.

## Execução
Baixar arquivos oficiais com commit e hashes fixados; implementar pipeline e inferência com Monte Carlo dropout; reexecutar os pesos dos autores; treinar LSTM/Bi-LSTM em ambos os datasets; validar contratos de isolamento, votação, abstenção e modo estocástico; salvar modelos, previsões, curvas e relatório. A avaliação de infecção é pesquisa retrospectiva, sem alteração do detector operacional ou do firmware.

## Ajustes de implementação constatados na execução
- O dropout interno do LSTM multicamada no backend MIOpen depende de um cabeçalho ausente nesta máquina. Foram empilhadas duas camadas de LSTM com dropout explícito entre elas; testes verificam equivalência determinística com os pesos e o readout originais, em ambas as direções.
- IDs de grupo nos NPZ foram convertidos de arrays de objetos para strings Unicode, permitindo leitura com `allow_pickle=False`. Probabilidades, alvos e pesos foram preservados; `artifact_format_migration.json` registra os hashes anteriores e posteriores. A pasta `source/` mantém o código efetivamente usado no treino; o executor atual já grava no formato corrigido.
- A coleta ampla de `pytest` importou scripts executáveis fora de `tests/` e alterou incidentalmente dois artefatos históricos. As cópias incidentais foram preservadas em `tmp/`, os dois arquivos foram restaurados aos bytes anteriores e `pytest.ini` passou a limitar a coleta à suíte do projeto.

Os resultados medidos e os comandos estão no [relatório final](aplicacao_torres_2025_2026_10_15.md).
