# Datasets candidatos para a v4

Levantamento de 22/10/2026. Cada entrada foi conferida no registro do repositório ou no artigo, e no disco local quando havia cópia. **Nada foi baixado nesta etapa.**

## Conclusão principal

**Nenhum corpus disponível serve como teste novo pelas regras da v4.** Para servir, ele precisaria ser acústico, rotulado, ter clusters com *Aedes aegypti* e vir de aparelhos, sessões ou corpora que nunca entraram em desenvolvimento.

- Os candidatos públicos com *Ae. aegypti* acústico (Abuzz/Dryad, HumBugDB, MosquitoSong, UFRGS, BioDCASE D1–D4 e Funchal/Figshare) já estão no disco e foram usados no desenvolvimento de alguma rodada. No caso da UFRGS, o material saiu do mesmo aparelho e da mesma sessão do teste reservado da v3.
- O conjunto de avaliação do BioDCASE 2026 não tem rótulos públicos.
- O dataset de Gana (KNUST) não tem disponibilidade confirmada.

Consequência para a v4: o protocolo pode ser pré-registrado e congelado agora, mas **a afirmação de melhora depende de dados que ainda não existem**. As rotas estão na seção "Teste novo reservado".

## O que já é usado ou está no disco

| Corpus | No disco | Uso até a v3 | Observação verificada |
|---|---|---|---|
| Abuzz / Dryad, Mukundarajan et al. 2017 ([Zenodo 4964774](https://zenodo.org/records/4964774), DOI 10.5061/dryad.98d7s, CC0) | `data/raw/<espécie>/` (20 espécies) | Desenvolvimento v2/v3 (`abuzz`) | O registro tem as mesmas 20 espécies das pastas locais (1,2 GB). Arquivos fora do manifesto são duplicatas ou segmentos removidos na deduplicação. |
| HumBugDB, Kiskin et al. 2021 (Zenodo 10.5281/zenodo.4904800, CC BY 4.0) | `data/external/humbugdb_full/` (9.295 registros) | 5.795 em desenvolvimento | Os 3.500 restantes (57 *Ae. aegypti*) vêm de aparelhos já usados no desenvolvimento: tascam, olympus, itel, Alcatel. Nenhum aparelho novo. |
| MosquitoSong / MosquitoSong+ ([GitHub](https://github.com/akaraspt/mosquitosongp)) | `data/external/additional/mosquitosong/` (1.322) | Externo na v2; LODO na v3 | 16 indivíduos (4 *Ae. aegypti*), dois sexos, 25 °C. |
| BioDCASE 2026 CD-MSC ([Zenodo 20478577](https://zenodo.org/records/20478577), CC BY 4.0) | `data/external/biodcase_dev/` (271.380 clipes; 6,9 GB) | D1–D4 em desenvolvimento; **D3 = teste reservado da v3** | D3 coincide com as gravações do insetário da UFRGS: `S_1_D_3` = 192 = RD5 (fêmeas *Ae. aegypti*) e `S_2_D_3` = 19 = RD7. **D5 (265.652 clipes) é o Wingbeats óptico** (Fanioudakis et al. 2018), excluído por não ser microfone. O conjunto de avaliação (471 MB) não tem rótulos publicados. |
| UFRGS, Paim et al. 2024 ([página](https://www.inf.ufrgs.br/aedes-vigilance/), CC BY 4.0) | 15 WAV em `data/external/additional/ufrgs/` | Externo na v2; teste reservado na v3 | O dataset completo (RD5–RD9, LG K10, Porto Alegre, fev/2021) inclui 345 machos (RD6) nunca usados, mas do mesmo aparelho, insetário e período do teste da v3. |
| Funchal / Figshare, Vasconcelos et al. 2020 ([figshare 11902125](https://figshare.com/articles/dataset/Mosquitoes_Bioacoustic_Features_-_A_Public_Dataset/11902125); [artigo](https://www.nature.com/articles/s41597-020-00725-6)) | `data/external/additional/figshare/` (2.988 WAV) | Treino de adaptação em `retreinamento_multifonte.md` e LOSO de 03/10; excluído na v2/v3 | Gravado com **microfone** (Teensy 3.2, Huawei P20 Lite, iPhone 4), em laboratório no Funchal. O motivo registrado na v2/v3 ("not an acoustic-microphone corpus", "optical") contradiz o artigo. São trechos de 0,3 s (2.944 de 2.988) e não há identidade de indivíduo ou sessão. |
| ESC-50 | local | Ruído de desenvolvimento e teste de ruído da v3 | — |
| Ruído Pete | `data/noise/` | Ruído de desenvolvimento e teste de ruído da v3 | — |
| Torres et al. 2025 (DENV-2) | `data/external/torres_2025/` | Rodadas anteriores à v2 | Só espectrogramas pré-calculados (300 × 188 × 32), 100% *Aedes*. Não entra na v4. |
| Synthetic swarm (arXiv 2512.12365) | `data/external/additional/synthetic/` (315) | Rodadas anteriores à v2 | Sintético; no máximo para aumento de dados. |
| Áudio privado do Drive | — | Excluído | Continua excluído. |

## Candidatos novos, por valor esperado

| # | Dataset | Licença | Tamanho de download | Conteúdo verificado | Papel proposto | Risco de vazamento ou de nova decodificação de fonte | Decisão |
|---|---|---|---|---|---|---|---|
| 1 | **Coleta prospectiva** própria ou de colaboradores (IFSC, UFRGS com aparelhos novos, Fiocruz) | do projeto | — | Ainda não existe | **Teste novo congelado** | Baixo, se aparelhos, locais e sessões forem novos e cada classe for gravada em todos os blocos | **Recomendada.** Protocolo mínimo em `plano_v4.md`. |
| 2 | Dataset de Gana, KNUST 2025 ([resumo KEMS](https://kems.knust.edu.gh/event/48/contributions/273/)): 34.153 gravações de mosquito e 34.111 de não mosquito | não publicada | ? | Gênero, sexo e idade citados no resumo; espécie não confirmada | Teste novo, se houver *Ae. aegypti* rotulado e identidade de sessão | País e aparelhos novos | Pedir aos autores. |
| 3 | Funchal / Figshare (local) | não declarada para o áudio; metadados CC0 | já local | 1.876 trechos de *Ae. aegypti* (12–18 espécimes), 1.040 de *Culex*, 72 de *Culiseta*; 3 aparelhos | Desenvolvimento (cluster = aparelho × taxa de amostragem), **não** teste | Já entrou em treino antes da v2; sem identidade de indivíduo | Usar só em desenvolvimento, com o rótulo de não cego. |
| 4 | UFRGS RD6, 345 machos de *Ae. aegypti* (zip completo) | CC BY 4.0 | provavelmente < 1 GB (não informado) | Mesmo LG K10 e insetário do teste da v3 | Desenvolvimento (sexo masculino) | Mesmo domínio do teste da v3: **não** pode ser teste | Opcional. |
| 5 | Wingbeats óptico (BioDCASE D5 local; [Kaggle](https://www.kaggle.com/datasets/potamitis/wingbeats)) | licença não confirmada | já local | 279.566 registros, 6 espécies, sensor óptico | **Pré-treino** da representação do batimento | Domínio diferente (óptico); não deve entrar em teste nem em calibração | Opcional, para pré-treino. |
| 6 | InsectSet459 ([arXiv 2503.15074](https://arxiv.org/abs/2503.15074)) | CC BY 4.0 segundo o artigo (um README de terceiros cita BY-NC-SA; conferir no Zenodo) | não informado | 26.399 arquivos, 459 espécies de Orthoptera e Cicadidae | Negativos duros (outros insetos) | Fontes de gravação diferentes das dos mosquitos: o modelo pode aprender "fonte" em vez de "inseto" | Útil, mas sem dípteros. |
| 7 | AudioSet: "Mosquito" ([342 vídeos](https://research.google.com/audioset/dataset/mosquito.html)), "Buzz", "Fly, housefly", "Bee, wasp" | rótulos CC BY 4.0; áudio do YouTube com termos próprios | download de vídeos, variável | Rótulo fraco, espécie desconhecida | Positivos fracos de presença; negativos duros de zumbido | Rótulo fraco; vídeos indisponíveis | Só como negativos duros, após checagem manual. |
| 8 | FSD50K ([arXiv 2010.00475](https://arxiv.org/abs/2010.00475)) | CC por clipe (parte NC) | > 20 GB (não confirmado) | 51.197 clipes, 200 classes; presença de "Insect"/"Buzz" não confirmada | Negativos e ruído | Licenças mistas | Baixa prioridade; pedir confirmação antes de baixar. |
| 9 | MUSAN ([OpenSLR 17](https://openslr.org/17/)) | CC BY 4.0, com licença por arquivo | 11 GB | Cerca de 109 h de música, fala e ruído, 16 kHz | Aumento de dados (ruído de fundo) | Baixo | **Acima de 5 GB: pedir confirmação.** Só a pasta `noise/` é necessária. |
| 10 | RIRs e ruídos ([OpenSLR 28](https://openslr.org/28/)) | Apache 2.0 (verificar as fontes RWCP, REVERB e AIR) | 1,3 GB | RIRs reais e simuladas e ruídos isotrópicos, 16 kHz | Aumento de dados (reverberação) | Baixo | Download pequeno; pode ser feito. |
| 11 | DEMAND ([Zenodo 1227121](https://zenodo.org/records/1227121)) | CC BY-SA 3.0 | não confirmado (16 canais por ambiente) | Ruído de ambientes reais (cozinha, rua, escritório...) | Aumento de dados (ruído realista) | Baixo | Baixar só 1 canal, a 16 kHz, por ambiente. |
| 12 | UrbanSound8K | CC BY-NC 3.0 (não verificado nesta sessão) | cerca de 6 GB (não verificado) | Ruído urbano em 10 classes | Aumento de dados | Licença não comercial | Baixa prioridade. |

Descartados: o "Synthetic Swarm" (arXiv 2512.12365), que é sintético e não tem dataset publicado no resumo; e as gravações de convergência harmônica (Cator et al.), cujos dados não foram localizados.

## Teste novo reservado

**Nenhum dataset existente foi reservado como teste novo.** Pelos critérios da v4, nenhum se qualifica:
- Os corpora com *Ae. aegypti* acústico rotulado já entraram em desenvolvimento.
- O BioDCASE de avaliação não tem rótulos.
- O de Gana não está publicado.

Reservar algum deles como "novo" violaria a regra do protocolo.

Rotas para criar o teste novo, todas dependentes de decisão do autor:
1. **Coleta prospectiva**, com o desenho mínimo de `plano_v4.md`: aparelhos, locais e dias novos, *Ae. aegypti* e negativos difíceis (*Ae. albopictus*, *Culex quinquefasciatus*) gravados no mesmo bloco, e sexo e temperatura registrados. As janelas e o manifesto ficam congelados antes de qualquer treino.
2. **Pedido de dados** aos autores do dataset de Gana (KNUST) e aos do MosquitoSong+ (gravações não publicadas), com a condição de que o material chegue sem ser inspecionado pela equipe de modelagem.
3. **Confirmação não cega**, apenas descritiva: o Funchal pode ser usado como verificação adicional, mas **não sustenta afirmação de melhora**.

## Fontes

- [Zenodo 4964774: Using mobile phones as acoustic sensors (Abuzz/Dryad)](https://zenodo.org/records/4964774)
- [BioDCASE 2026 CD-MSC, Zenodo 20478577](https://zenodo.org/records/20478577)
- [UFRGS Aedes Vigilance](https://www.inf.ufrgs.br/aedes-vigilance/) e [Paim et al., arXiv 2306.10091](https://arxiv.org/html/2306.10091v1)
- [Vasconcelos et al. 2020, Scientific Data (PMC7659005)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7659005)
- [HumBugDB, NeurIPS 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/65ded5353c5ee48d0b7d48c591b8f430-Paper-round2.pdf)
- [MosquitoSong+](https://pmc.ncbi.nlm.nih.gov/articles/PMC11524479/)
- [Wingbeats (Kaggle)](https://www.kaggle.com/datasets/potamitis/wingbeats)
- [InsectSet459, arXiv 2503.15074](https://arxiv.org/abs/2503.15074)
- [AudioSet: Mosquito](https://research.google.com/audioset/dataset/mosquito.html)
- [FSD50K, arXiv 2010.00475](https://arxiv.org/abs/2010.00475)
- [MUSAN, OpenSLR 17](https://openslr.org/17/), [RIRs, OpenSLR 28](https://openslr.org/28/), [DEMAND, Zenodo 1227121](https://zenodo.org/records/1227121)
- [KNUST, KEMS 2025](https://kems.knust.edu.gh/event/48/contributions/273/)
- [Synthetic Swarm, arXiv 2512.12365](https://arxiv.org/abs/2512.12365)
