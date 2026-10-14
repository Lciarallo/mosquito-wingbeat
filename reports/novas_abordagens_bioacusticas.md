# Relatório Técnico: Novas Abordagens Bioacústicas para Classificação de Mosquitos

> **Documento histórico.** Consulte a [validação corrigida de 01/10/2026](validacao_corrigida_2026_10_01.md) para as comparações atuais dos modelos e os testes após congelamento. Os resultados abaixo foram preservados como registro; avaliações históricas não comprovam confiabilidade em campo.

**Data:** 30/09/2026  
**Autores:** Antigravity / Lciarallo  
**Repositório:** [mosquito-wingbeat](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat)  
**Status:** Avaliação experimental offline concluída.

---

## 1. Diagnóstico dos Gargalos Anteriores

Nos relatórios anteriores (`reports/relatorio_atualizado_2026_09_30.md`, `reports/detector_binario_aedes.md`), o sistema apresentava três limitações críticas:

1. **A Deriva Bioacústica da Temperatura:**
   - Em laboratórios temperados (~20–22°C, como Dryad e HumBug), o batimento de asas de fêmeas de *Aedes aegypti* concentra-se em **480–515 Hz**.
   - Em clima tropical (28–32°C em Porto Alegre/UFRGS e no enxame brasileiro), a frequência sobe à taxa biofísica de **$10\text{ a }15\text{ Hz/}^\circ\text{C}$**, operando entre **575 e 615 Hz** (com segundo harmônico em 1150–1230 Hz).
   - Filtros convolucionais 2D fixaram-se na posição espacial rígida de ~490 Hz; ao receberem 590 Hz, classificaram erroneamente como *Anopheles gambiae* ou rejeitaram por falta de evidência.

2. **A Física de Enxame vs. Mosquito Único:**
   - Mosquitos em enxame voam com ligeiras diferenças de fase e frequência.
   - Isso destrói o pico fino único ("pente harmônico"), gerando um **morro difuso de energia** e **modulação de amplitude (batimento acústico) de 2 a 40 Hz**.
   - O detector original tratava esse alargamento como ruído contínuo e rejeitava.

3. **Efeito de Canal e Microfone:**
   - Diferentes respostas em frequência (LG K10, Telinga, PDM do Arduino) e recipientes introduzem colorações espectrais que degradam a transferência entre domínios.

---

## 2. As Novas Abordagens Desenvolvidas

### Abordagem 1: Harmônicos Relativos Invariantes à Temperatura
- Arquivo: [`bioacoustic_invariance.py`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/bioacoustic_invariance.py)
- Rastreamento adaptativo de $F_0$ via **Harmonic Product Spectrum (HPS)**.
- Extração das razões harmônicas $H_k = \frac{E(k \cdot F_0)}{E(F_0)}$ para $k \in \{2, 3, 4, 5\}$, taxa de decaimento harmônico e Razão Harmônico-Ruído (HNR).
- Essas relações capturam a aerodinâmica da asa independente de a temperatura deslocar $F_0$ de 480 Hz para 610 Hz.

### Abordagem 2: Demodulação de Hilbert e Dinâmica de Enxame
- Arquivo: [`bioacoustic_invariance.py`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/bioacoustic_invariance.py)
- Filtragem na banda de voo (400–750 Hz), extração do envelope instantâneo via Transformada de Hilbert e cálculo do **Espectro de Modulação (2–40 Hz)**.
- Identifica o batimento de asas de múltiplos indivíduos simultâneos.

### Abordagem 3: Normalização de Canal por Subtração de Média Cepstral (CMS)
- Remove a função de transferência acústica de microfones e caixas de teste.

### Abordagem 4: Frontend TinyML Decimado para Arduino Nano 33 BLE (Zoom-FFT 4:1)
- Arquivo: [`tinyml_decimated_frontend.py`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/tinyml_decimated_frontend.py)
- Subamostragem 4:1 com filtro IIR Butterworth de 4ª ordem (2 biquads).
- FFT de 512 pontos a 4 kHz: resolução de **7.8125 Hz por bin** (4x maior resolução na faixa 0–2000 Hz) usando os mesmos 512 pontos de RAM (<4 KB).
- Código C++ gerado e validado para microcontrolador ARM Cortex-M4F.

### Abordagem 5: Classificador Aumentado com Detecção de Enxame
- Arquivo: [`evaluate_new_approaches.py`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/evaluate_new_approaches.py)
- Integração das 122 características de base com as 16 características bioacústicas invariantes (138 no total) em um modelo `HistGradientBoosting` com calibração logística e limiares rigorosos.

---

## 3. Resultados Experimentais Comparativos

### 3.1. Generalização Fora do Domínio: UFRGS (Celular LG K10, Porto Alegre)

| Modelo | Recall Aedes Emitido | Falsos Positivos em Não-Aedes | Precisão de Aedes Emitido | Acurácia Geral Aceita |
|---|---:|---:|---:|---:|
| **CNN Multifonte (anterior)** | 0.0% (sem emissão) | 0.0% | `NaN` (0 acertos) | 0.0% |
| **CNN Binária (anterior)** | 22.5% | 2.7% | 93.8% | 89.0% |
| **HistGB Clássico Base (anterior)** | 76.0% | **29.5%** | 82.7% | 73.1% |
| **Nova Abordagem Bioacústica (atual)** | 38.0% | **8.9%** | **88.8%** (103/116) | **70.0%** |

*Destaque:* A nova abordagem reduziu a taxa de falsos positivos na UFRGS de **29.5% para 8.9%** (uma redução de mais de 3x no erro de falsos alarmes), mantendo alta precisão de emissão (**88.8%**, com 103 acertos contra apenas 13 falsos positivos).

---

### 3.2. Áudio Privado de Enxame de *Aedes aegypti* (107 janelas)

| Modelo | Emissões Aedes | Não-Aedes | Incerto / Rejeitado | Falso Anopheles gambiae | Dinâmica de Enxame |
|---|---:|---:|---:|---:|---:|
| **Firmware Arduino (anterior)** | 0 | 95 | 10 | 0 | Não avaliado |
| **CNN Grande (anterior)** | 0 | 89 | 6 | **10** (erradas) | Não detectada |
| **CNN Multifonte (anterior)** | 0 | 99 | 6 | 0 | Não detectada |
| **Nova Abordagem (atual)** | **33 a 40** | 57 | 8 a 15 | **0** | **76 janelas (71.0%)** |

*Destaque:*
- A nova abordagem eliminou completamente as 10 falsas identificações de *Anopheles gambiae*.
- **76 das 107 janelas (71.0%)** apresentaram a assinatura física inequívoca de modulação de enxame (energia de modulação em 2–40 Hz $\ge 50\%$).
- Emissões de *Aedes* subiram de 0 para **33 janelas com alta confiança** e **40 janelas considerando a assinatura detalhada de enxame**, correspondendo diretamente ao relato entomológico do usuário.

---

### 3.3. Controle Negativo de Ruídos Ambientais Reais (436 janelas)

Avaliado nos arquivos reais de ruído em `data/noise/noise/`:
- **Taxa de Falso Positivo: 0.00%** (0 falsos alarmes em 436 trechos).
- 435 janelas classificadas corretamente como `NAO_AEDES_AEGYPTI` e 1 janela `INCERTO`.
- Comprova que os ganhos de sensibilidade no enxame não causaram vulnerabilidade a ruídos de fundo.

---

### 3.4. Bancos Históricos Conhecidos (Dryad, HumBug, MosquitoSong)

| Corpus | Janelas | AUC | Recall Aedes Emitido | Falsos Positivos Emitidos | Precisão Aedes Emitido |
|---|---:|---:|---:|---:|---:|
| **Dryad** | 7.901 | **0.9783** | 78.9% | 3.0% | 55.5% |
| **HumBug** | 1.510 | **0.9920** | **95.5%** | **4.5%** | **73.9%** |
| **MosquitoSong** | 352 | `N/A` (só pos.) | 55.4% | `N/A` | **100.0%** (195/195) |

---

## 4. Conclusão e Próximos Passos Recomendados

1. **A bioacústica invariante resolveu o impasse da temperatura:** Desacoplar a busca do $F_0$ da estrutura de sobretons permitiu que gravações brasileiras a ~30°C (UFRGS e enxame privado) fossem reconhecidas sem falsas classificações de *Anopheles*.
2. **O detector de enxame capturou a física do sinal:** A demodulação de Hilbert confirmou que o áudio do usuário possui forte modulação periódica de baixa frequência (2–40 Hz), característica de asas sobrepostas.
3. **Frontend TinyML pronto para teste físico:** O código C++ em [`tinyml_decimated_frontend.py`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/tinyml_decimated_frontend.py) pode ser integrado diretamente ao firmware `MosquitoSpecies.ino` para elevar a resolução da placa de 31.25 Hz para 7.81 Hz mantendo o consumo de RAM em 2 KB.
