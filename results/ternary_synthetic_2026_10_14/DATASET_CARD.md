# Dataset de robustez ternário

2.740 áudios mono em WAV FLOAT, 16 kHz. Há 2.580 derivados semissintéticos e 160 negativos inteiramente procedurais. As espécies dos derivados são referências herdadas de 300 grupos doadores do corpus principal, não novas identificações biológicas.

Para consumir em outra máquina, leia `manifest_relative.csv`: a coluna `path` é relativa ao diretório deste cartão. `manifest.csv` conserva os caminhos absolutos da execução original para auditoria. As classes são `0=no_mosquito`, `1=other_mosquito`, `2=aedes_aegypti`.

As condições e sementes estão em `protocol.json`; os hashes dos WAVs estão no manifesto. `parents.csv` e as colunas `parent_`/`partner_` identificam a ancestralidade. Variações de um mesmo doador e os componentes de misturas devem permanecer juntos em qualquer divisão futura; não se deve criar partições aleatórias por WAV.

`predictions.csv` distingue resultados `heldout_` (modelo cuja dobra excluiu o doador e os componentes) de `final_` (modelo final ajustado nos doadores reais). Para os negativos procedurais, consulte `final_`. Predição `-1` significa áudio sem janela válida e continua no denominador.

O README contém resultados, unidades e limites. A reprodução da inferência exige o repositório e os modelos congelados, que não estão incluídos neste pacote de dados.
