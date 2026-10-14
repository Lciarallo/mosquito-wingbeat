# Trabalhos recentes ainda não citados — 03/10/2026

Busca feita em arXiv, OpenAlex, Hugging Face e GitHub. Foram excluídos os trabalhos que o projeto já cita (DR-BioL, MosquitoSong+, enxames sintéticos, TinyML, EUSIPCO 2018, eLife 2017, baseline BioDCASE 2026, Paim et al.). Só li resumos e metadados, não os artigos completos. O status de verificação de cada item está na tabela.

## Resultado principal

Só um item traz números comparáveis, e ele **não é uma solução resolvida**: no CD-MSC (9 espécies, 5 domínios), a métrica `BA_unseen` vai de ~0,17 (baseline MTRCNN) para ~0,31 (sonda sobre embeddings Perch congelados) e ~0,36 (comitê com portão de concordância). O acaso nessa tarefa é ~0,11. Esses valores são autodeclarados no README de um repositório da comunidade, sem revisão por pares. O próprio artigo do baseline diz que o desempenho "degrada marcadamente" em domínios não vistos. Isso é coerente com o que medimos no LOSO: o problema de generalização entre fontes continua aberto na literatura.

## Candidatos

| Trabalho | O que propõe | Verificação |
| --- | --- | --- |
| [Repositório CD-MSC (aptemvs)](https://github.com/aptemvs/Cross-Domain-Mosquito-Species-Classification) e [caches no Hugging Face](https://huggingface.co/datasets/aptemvs/mosquitoes-biodcase2026-task5) | Embeddings congelados de Perch, BirdMAE e sl-BEATs com sondas leves; características harmônicas e com fundo branqueado; portão de concordância entre modelos; perdas do DR-BioL | README e API conferidos. Números autodeclarados, tarefa multiclasse de espécies, não binária |
| Yepdjio Nkouanga e Singh, ICMLA 2022, [10.1109/icmla55696.2022.00246](https://doi.org/10.1109/icmla55696.2022.00246) | Remover ruído, reduzir dimensionalidade e classificar. Declara ganho de até ~28% sobre o baseline no HumBugDB | Só o resumo. Obra de 2022. Direta sobre mudança de distribuição |
| Augustin et al., PLOS ONE 2026, [10.1371/journal.pone.0343060](https://doi.org/10.1371/journal.pone.0343060) | 475 indivíduos de 15 espécies. Temperatura, umidade e tamanho corporal alteram a frequência do 1º harmônico | Só o resumo. Dá embasamento para a deriva de frequência entre fontes |
| MosquiTUNER, [10.5281/zenodo.18922703](https://doi.org/10.5281/zenodo.18922703) | Reúne os sons de mosquito já publicados e acrescenta espécies europeias: 437.231 sons, 55 espécies | Descrição do Zenodo. Por compilar bases publicadas, deve repetir dados que já usamos; exige dedup por hash |
| Rajan et al., Sci Rep 2025, [10.1038/s41598-025-89608-7](https://doi.org/10.1038/s41598-025-89608-7) | Mudança do tom de voo ao longo da vida adulta em *Ae. albopictus* e *Cx. quinquefasciatus* | Só o resumo. Estudo de laboratório, não é um método |
| Szekeres et al., Ecol. Informatics 2025, [10.1016/j.ecoinf.2025.103055](https://doi.org/10.1016/j.ecoinf.2025.103055) | CF-ResNet-1D, com camadas de Fourier (FNO) | O resumo não relata teste entre domínios. **Não dá para afirmar que seja melhor** |
| Hill et al., AIRC 2026, [10.1109/airc69745.2026.11631322](https://doi.org/10.1109/airc69745.2026.11631322) | Base de espectrogramas independentes do sensor e CNN-Transformer hierárquico | Sem resumo disponível. Não verificado |
| Grau-Haro et al., 2026, [10.1007/978-3-032-31144-3_18](https://doi.org/10.1007/978-3-032-31144-3_18) | Detecção acústica de voo de fêmeas para programas de SIT | Sem resumo disponível. Não verificado |
| [Wingbeat óptico, Tanzânia](https://doi.org/10.5281/zenodo.19111358) | Dados de sensor óptico, só de fêmeas | Descrição curta. Dados novos, mas de sensor óptico |

## Não encontrado

- Os relatórios técnicos dos participantes do BioDCASE 2026 (resultados de 06/07/2026) não ficaram acessíveis: a página da tarefa só entregou o cabeçalho. Eles seriam a fonte mais direta de soluções melhores.
- Nenhum trabalho de 2025 em diante que reporte detecção binária de *Aedes aegypti* com validação entre fontes e métricas de precisão e sensibilidade.
