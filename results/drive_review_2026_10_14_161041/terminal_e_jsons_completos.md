# Terminal e JSONs completos — revisão recebida do Drive em 14/10/2026

Saídas integrais das três execuções, teste e métricas; os arquivos públicos individuais estão em per_file/.

## emulated_brazil

```text
Command: /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/.venv/bin/python /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/scripts/predict_audio.py --audio data/raw/drive_test/drive_audio.wav --region brazil --output-csv /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/emulated_brazil_predictions.csv --output-summary /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/emulated_brazil_summary.json --emulate-arduino

==================================================================================
 RELATÓRIO HIERÁRQUICO E SEGMENTAÇÃO TEMPORAL DE ENXAME DE AEDES (ML + SHS) 
==================================================================================
Arquivo:               data/raw/drive_test/drive_audio.wav
Duração Total:         106.24 s
Emulação Mic Arduino:  ATIVADA (ST MP34DT05)
Prior Epidemiológico:  BRAZIL
Janelas de Voo Ativo:  139/211 (65.88%)
Janelas de Enxame:     125/211 (59.24%)
Diagnóstico Tier 1:    ENXAME DE AEDES DETECTADO
Fundamental Mediana:   F0 = 600.0 Hz (IQR = 43.8 Hz)

--- TIER 2: VETOR EPIDEMIOLÓGICO (GÊNERO COM CORREÇÃO DE CARDINALIDADE) ---
Gênero Identificado:   Aedes (Confirmação Causal: True)
  * Aedes         : 121 janelas ( 87.1%)
  * Culiseta      :  14 janelas ( 10.1%)
  * Culex         :   4 janelas (  2.9%)

--- TIER 3: ESPÉCIE ESPECÍFICA (FUSÃO HARMÔNICA + NEURAL) ---
Espécie Confirmada:    Aedes aegypti
  * Aedes aegypti             : 132 janelas ( 95.0%)
  * Culex quinquefasciatus    :   5 janelas (  3.6%)
  * Culex pipiens             :   2 janelas (  1.4%)

--- MAPA DE INTERVALOS TEMPORAIS (QUANDO O ENXAME APARECE vs QUANDO NÃO APARECE) ---
Intervalo (s)    | Duração  | Status / Regime              | F0 Med   | SNR Aedes  | Espécie Dominante
--------------------------------------------------------------------------------------------------
  0.0 -  23.5s   |  23.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -2.3 dB   | ---
 23.0 -  25.0s   |   2.0s   | PASSAGEM_CURTA_ISOLADA       | 640.0Hz  |  -0.3 dB   | Aedes aegypti
 24.5 -  26.0s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -1.0 dB   | ---
 25.5 -  26.5s   |   1.0s   | ENXAME_AEDES_PRESENTE        | 595.0Hz  |  +1.6 dB   | Aedes aegypti
 26.0 -  27.0s   |   1.0s   | PASSAGEM_CURTA_ISOLADA       | 525.0Hz  |  +0.4 dB   | Aedes aegypti
 26.5 -  34.0s   |   7.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -2.6 dB   | ---
 33.5 -  35.5s   |   2.0s   | PASSAGEM_CURTA_ISOLADA       | 575.0Hz  |  -0.3 dB   | Aedes aegypti
 35.0 -  36.0s   |   1.0s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -0.7 dB   | ---
 35.5 -  37.5s   |   2.0s   | ENXAME_AEDES_PRESENTE        | 590.0Hz  |  +2.5 dB   | Aedes aegypti
 37.0 -  38.5s   |   1.5s   | PASSAGEM_CURTA_ISOLADA       | 622.5Hz  |  +0.7 dB   | Aedes aegypti
 38.0 -  39.5s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -0.7 dB   | ---
 39.0 -  40.0s   |   1.0s   | ENXAME_AEDES_PRESENTE        | 625.0Hz  |  +1.1 dB   | Aedes aegypti
 39.5 -  40.5s   |   1.0s   | PASSAGEM_CURTA_ISOLADA       | 600.0Hz  |  -0.1 dB   | Aedes aegypti
 40.0 -  41.5s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -1.5 dB   | ---
 41.0 -  47.5s   |   6.5s   | ENXAME_AEDES_PRESENTE        | 611.2Hz  |  +6.1 dB   | Aedes aegypti
 47.0 -  48.5s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -1.8 dB   | ---
 48.0 -  91.5s   |  43.5s   | ENXAME_AEDES_PRESENTE        | 602.5Hz  | +10.9 dB   | Aedes aegypti
 91.0 -  92.0s   |   1.0s   | PASSAGEM_CURTA_ISOLADA       | 547.5Hz  |  +0.2 dB   | Aedes aegypti
 91.5 -  92.5s   |   1.0s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -0.6 dB   | ---
 92.0 -  95.0s   |   3.0s   | ENXAME_AEDES_PRESENTE        | 590.0Hz  |  +3.6 dB   | Aedes aegypti
 94.5 -  96.0s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -9.8 dB   | ---
 95.5 -  99.5s   |   4.0s   | ENXAME_AEDES_PRESENTE        | 572.5Hz  |  +4.5 dB   | Aedes aegypti
 99.0 - 100.5s   |   1.5s   | PASSAGEM_CURTA_ISOLADA       | 590.0Hz  |  +0.7 dB   | Aedes aegypti
100.0 - 106.0s   |   6.0s   | ENXAME_AEDES_PRESENTE        | 597.5Hz  |  +3.0 dB   | Aedes aegypti
==================================================================================

-> Relatório detalhado salvo em CSV: /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/emulated_brazil_predictions.csv
-> Resumo analítico salvo em JSON: /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/emulated_brazil_summary.json

Exit code: 0

```

```json
{
  "audio_file": "data/raw/drive_test/drive_audio.wav",
  "duration_seconds": 106.24,
  "emulate_arduino_mic": true,
  "region_prior": "brazil",
  "total_windows": 211,
  "detected_windows": 139,
  "aedes_swarm_windows": 125,
  "active_ratio_pct": 65.88,
  "swarm_ratio_pct": 59.24,
  "mosquito_present": true,
  "aedes_swarm_detected": true,
  "median_flight_f0_hz": 600.0,
  "iqr_flight_f0_hz": 43.8,
  "primary_genus": "Aedes",
  "confirmed_genus": "Aedes",
  "primary_species": "Aedes aegypti",
  "confirmed_species": "Aedes aegypti",
  "genus_distribution": {
    "Aedes": 121,
    "Culiseta": 14,
    "Culex": 4
  },
  "species_distribution": {
    "Aedes aegypti": 132,
    "Culex quinquefasciatus": 5,
    "Culex pipiens": 2
  },
  "temporal_intervals": [
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 0.0,
      "end_s": 23.49,
      "duration_s": 23.49,
      "windows_count": 46,
      "median_f0_hz": 468.8,
      "median_aedes_snr_db": -2.32,
      "mean_presence_score": 0.38,
      "dominant_species": null
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 23.0,
      "end_s": 24.99,
      "duration_s": 1.99,
      "windows_count": 3,
      "median_f0_hz": 640.0,
      "median_aedes_snr_db": -0.28,
      "mean_presence_score": 0.762,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 24.5,
      "end_s": 25.99,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 610.0,
      "median_aedes_snr_db": -1.0,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 25.5,
      "end_s": 26.49,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 595.0,
      "median_aedes_snr_db": 1.65,
      "mean_presence_score": 0.7996,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 26.0,
      "end_s": 26.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 525.0,
      "median_aedes_snr_db": 0.37,
      "mean_presence_score": 0.7779,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 26.5,
      "end_s": 33.99,
      "duration_s": 7.49,
      "windows_count": 14,
      "median_f0_hz": 557.5,
      "median_aedes_snr_db": -2.56,
      "mean_presence_score": 0.4082,
      "dominant_species": null
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 33.5,
      "end_s": 35.49,
      "duration_s": 1.99,
      "windows_count": 3,
      "median_f0_hz": 575.0,
      "median_aedes_snr_db": -0.28,
      "mean_presence_score": 0.7667,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 35.0,
      "end_s": 35.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 570.0,
      "median_aedes_snr_db": -0.67,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 35.5,
      "end_s": 37.49,
      "duration_s": 1.99,
      "windows_count": 3,
      "median_f0_hz": 590.0,
      "median_aedes_snr_db": 2.53,
      "mean_presence_score": 0.7925,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 37.0,
      "end_s": 38.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 622.5,
      "median_aedes_snr_db": 0.66,
      "mean_presence_score": 0.7625,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 38.0,
      "end_s": 39.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 550.0,
      "median_aedes_snr_db": -0.74,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 39.0,
      "end_s": 39.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 625.0,
      "median_aedes_snr_db": 1.1,
      "mean_presence_score": 0.7799,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 39.5,
      "end_s": 40.49,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 600.0,
      "median_aedes_snr_db": -0.09,
      "mean_presence_score": 0.7634,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 40.0,
      "end_s": 41.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 587.5,
      "median_aedes_snr_db": -1.49,
      "mean_presence_score": 0.2311,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 41.0,
      "end_s": 47.49,
      "duration_s": 6.49,
      "windows_count": 12,
      "median_f0_hz": 611.2,
      "median_aedes_snr_db": 6.14,
      "mean_presence_score": 0.8061,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 47.0,
      "end_s": 48.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 596.2,
      "median_aedes_snr_db": -1.8,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 48.0,
      "end_s": 91.49,
      "duration_s": 43.49,
      "windows_count": 86,
      "median_f0_hz": 602.5,
      "median_aedes_snr_db": 10.95,
      "mean_presence_score": 0.8004,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 91.0,
      "end_s": 91.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 547.5,
      "median_aedes_snr_db": 0.2,
      "mean_presence_score": 0.6989,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 91.5,
      "end_s": 92.49,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 570.0,
      "median_aedes_snr_db": -0.57,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 92.0,
      "end_s": 94.99,
      "duration_s": 2.99,
      "windows_count": 5,
      "median_f0_hz": 590.0,
      "median_aedes_snr_db": 3.59,
      "mean_presence_score": 0.8063,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 94.5,
      "end_s": 95.99,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 610.0,
      "median_aedes_snr_db": -9.82,
      "mean_presence_score": 0.1935,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 95.5,
      "end_s": 99.49,
      "duration_s": 3.99,
      "windows_count": 7,
      "median_f0_hz": 572.5,
      "median_aedes_snr_db": 4.51,
      "mean_presence_score": 0.8298,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 99.0,
      "end_s": 100.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 590.0,
      "median_aedes_snr_db": 0.65,
      "mean_presence_score": 0.8223,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 100.0,
      "end_s": 105.99,
      "duration_s": 5.99,
      "windows_count": 11,
      "median_f0_hz": 597.5,
      "median_aedes_snr_db": 2.97,
      "mean_presence_score": 0.78,
      "dominant_species": "Aedes aegypti"
    }
  ],
  "swarm_timeline": [
    {
      "start_s": 0.0,
      "end_s": 23.0,
      "duration_s": 23.0,
      "swarm_present": false,
      "median_f0_hz": 468.8,
      "mean_aedes_snr_db": -4.176739130434783,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 23.0,
      "end_s": 24.5,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 640.0,
      "mean_aedes_snr_db": -0.12666666666666668,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 24.5,
      "end_s": 25.5,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 610.0,
      "mean_aedes_snr_db": -0.9950000000000001,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.65,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 26.0,
      "end_s": 26.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 525.0,
      "mean_aedes_snr_db": 0.37,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 26.5,
      "end_s": 33.5,
      "duration_s": 7.0,
      "swarm_present": false,
      "median_f0_hz": 557.5,
      "mean_aedes_snr_db": -3.365714285714286,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 33.5,
      "end_s": 35.0,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 575.0,
      "mean_aedes_snr_db": -0.18666666666666668,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 35.0,
      "end_s": 35.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 570.0,
      "mean_aedes_snr_db": -0.67,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 35.5,
      "end_s": 37.0,
      "duration_s": 1.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 2.793333333333333,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 37.0,
      "end_s": 38.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 622.5,
      "mean_aedes_snr_db": 0.66,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 38.0,
      "end_s": 39.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 550.0,
      "mean_aedes_snr_db": -0.745,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 625.0,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.5,
      "end_s": 40.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 600.0,
      "mean_aedes_snr_db": -0.09,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 40.0,
      "end_s": 41.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 587.5,
      "mean_aedes_snr_db": -1.49,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 41.0,
      "end_s": 47.0,
      "duration_s": 6.0,
      "swarm_present": true,
      "median_f0_hz": 611.2,
      "mean_aedes_snr_db": 6.1025,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 47.0,
      "end_s": 48.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 596.2,
      "mean_aedes_snr_db": -1.805,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 48.0,
      "end_s": 91.0,
      "duration_s": 43.0,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.27674418604651,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 91.0,
      "end_s": 91.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 547.5,
      "mean_aedes_snr_db": 0.2,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 91.5,
      "end_s": 92.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 570.0,
      "mean_aedes_snr_db": -0.57,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 92.0,
      "end_s": 94.5,
      "duration_s": 2.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 3.4659999999999997,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 94.5,
      "end_s": 95.5,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 610.0,
      "mean_aedes_snr_db": -9.82,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 95.5,
      "end_s": 99.0,
      "duration_s": 3.5,
      "swarm_present": true,
      "median_f0_hz": 572.5,
      "mean_aedes_snr_db": 4.611428571428571,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 99.0,
      "end_s": 100.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 0.65,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.14,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "swarm_active_intervals": [
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.65,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 35.5,
      "end_s": 37.0,
      "duration_s": 1.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 2.793333333333333,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 625.0,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 41.0,
      "end_s": 47.0,
      "duration_s": 6.0,
      "swarm_present": true,
      "median_f0_hz": 611.2,
      "mean_aedes_snr_db": 6.1025,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 48.0,
      "end_s": 91.0,
      "duration_s": 43.0,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.27674418604651,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 92.0,
      "end_s": 94.5,
      "duration_s": 2.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 3.4659999999999997,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 95.5,
      "end_s": 99.0,
      "duration_s": 3.5,
      "swarm_present": true,
      "median_f0_hz": 572.5,
      "mean_aedes_snr_db": 4.611428571428571,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.14,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "causal_swarm_timeline": [
    {
      "start_s": 0.0,
      "end_s": 23.0,
      "duration_s": 23.0,
      "swarm_present": false,
      "median_f0_hz": 468.8,
      "mean_aedes_snr_db": -4.176739130434783,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 23.0,
      "end_s": 24.5,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 640.0,
      "mean_aedes_snr_db": -0.12666666666666668,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 24.5,
      "end_s": 25.5,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 610.0,
      "mean_aedes_snr_db": -0.9950000000000001,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.65,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 26.0,
      "end_s": 26.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 525.0,
      "mean_aedes_snr_db": 0.37,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 26.5,
      "end_s": 33.5,
      "duration_s": 7.0,
      "swarm_present": false,
      "median_f0_hz": 557.5,
      "mean_aedes_snr_db": -3.365714285714286,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 33.5,
      "end_s": 35.0,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 575.0,
      "mean_aedes_snr_db": -0.18666666666666668,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 35.0,
      "end_s": 35.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 570.0,
      "mean_aedes_snr_db": -0.67,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 35.5,
      "end_s": 37.5,
      "duration_s": 2.0,
      "swarm_present": true,
      "median_f0_hz": 601.2,
      "mean_aedes_snr_db": 2.34,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 37.5,
      "end_s": 38.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 0.34,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 38.0,
      "end_s": 39.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 550.0,
      "mean_aedes_snr_db": -0.745,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 625.0,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.5,
      "end_s": 40.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 600.0,
      "mean_aedes_snr_db": -0.09,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 40.0,
      "end_s": 41.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 587.5,
      "mean_aedes_snr_db": -1.49,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 41.0,
      "end_s": 47.5,
      "duration_s": 6.5,
      "swarm_present": true,
      "median_f0_hz": 611.2,
      "mean_aedes_snr_db": 5.546153846153847,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 47.5,
      "end_s": 48.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 572.5,
      "mean_aedes_snr_db": -2.48,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 48.0,
      "end_s": 91.5,
      "duration_s": 43.5,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.160919540229884,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 91.5,
      "end_s": 92.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 570.0,
      "mean_aedes_snr_db": -0.57,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 92.0,
      "end_s": 95.0,
      "duration_s": 3.0,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 1.2449999999999999,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 95.0,
      "end_s": 95.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 610.0,
      "mean_aedes_snr_db": -9.78,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 95.5,
      "end_s": 99.5,
      "duration_s": 4.0,
      "swarm_present": true,
      "median_f0_hz": 573.8,
      "mean_aedes_snr_db": 4.09,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 99.5,
      "end_s": 100.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 605.0,
      "mean_aedes_snr_db": 0.86,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.14,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "causal_swarm_active_intervals": [
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.65,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 35.5,
      "end_s": 37.5,
      "duration_s": 2.0,
      "swarm_present": true,
      "median_f0_hz": 601.2,
      "mean_aedes_snr_db": 2.34,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 625.0,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 41.0,
      "end_s": 47.5,
      "duration_s": 6.5,
      "swarm_present": true,
      "median_f0_hz": 611.2,
      "mean_aedes_snr_db": 5.546153846153847,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 48.0,
      "end_s": 91.5,
      "duration_s": 43.5,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.160919540229884,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 92.0,
      "end_s": 95.0,
      "duration_s": 3.0,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 1.2449999999999999,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 95.5,
      "end_s": 99.5,
      "duration_s": 4.0,
      "swarm_present": true,
      "median_f0_hz": 573.8,
      "mean_aedes_snr_db": 4.09,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.14,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "timeline_resolution_s": 0.5,
  "analysis_window_s": 0.992,
  "unanalysed_tail_s": 0.248,
  "timeline_interpretation": "Estimated classifier decisions; swarm_timeline uses offline 1-step smoothing, whereas causal_swarm_timeline uses strictly causal past-only persistence."
}
```

## raw_brazil

```text
Command: /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/.venv/bin/python /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/scripts/predict_audio.py --audio data/raw/drive_test/drive_audio.wav --region brazil --output-csv /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/raw_brazil_predictions.csv --output-summary /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/raw_brazil_summary.json

==================================================================================
 RELATÓRIO HIERÁRQUICO E SEGMENTAÇÃO TEMPORAL DE ENXAME DE AEDES (ML + SHS) 
==================================================================================
Arquivo:               data/raw/drive_test/drive_audio.wav
Duração Total:         106.24 s
Emulação Mic Arduino:  DESATIVADA (Áudio Bruto)
Prior Epidemiológico:  BRAZIL
Janelas de Voo Ativo:  139/211 (65.88%)
Janelas de Enxame:     125/211 (59.24%)
Diagnóstico Tier 1:    ENXAME DE AEDES DETECTADO
Fundamental Mediana:   F0 = 600.0 Hz (IQR = 40.0 Hz)

--- TIER 2: VETOR EPIDEMIOLÓGICO (GÊNERO COM CORREÇÃO DE CARDINALIDADE) ---
Gênero Identificado:   Aedes (Confirmação Causal: True)
  * Aedes         : 122 janelas ( 87.8%)
  * Culiseta      :  13 janelas (  9.4%)
  * Culex         :   4 janelas (  2.9%)

--- TIER 3: ESPÉCIE ESPECÍFICA (FUSÃO HARMÔNICA + NEURAL) ---
Espécie Confirmada:    Aedes aegypti
  * Aedes aegypti             : 132 janelas ( 95.0%)
  * Culex quinquefasciatus    :   5 janelas (  3.6%)
  * Culex pipiens             :   2 janelas (  1.4%)

--- MAPA DE INTERVALOS TEMPORAIS (QUANDO O ENXAME APARECE vs QUANDO NÃO APARECE) ---
Intervalo (s)    | Duração  | Status / Regime              | F0 Med   | SNR Aedes  | Espécie Dominante
--------------------------------------------------------------------------------------------------
  0.0 -  23.5s   |  23.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -2.3 dB   | ---
 23.0 -  25.0s   |   2.0s   | PASSAGEM_CURTA_ISOLADA       | 640.0Hz  |  -0.3 dB   | Aedes aegypti
 24.5 -  26.0s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -1.0 dB   | ---
 25.5 -  26.5s   |   1.0s   | ENXAME_AEDES_PRESENTE        | 595.0Hz  |  +1.6 dB   | Aedes aegypti
 26.0 -  27.0s   |   1.0s   | PASSAGEM_CURTA_ISOLADA       | 525.0Hz  |  +0.4 dB   | Aedes aegypti
 26.5 -  34.0s   |   7.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -2.6 dB   | ---
 33.5 -  35.5s   |   2.0s   | PASSAGEM_CURTA_ISOLADA       | 575.0Hz  |  -0.3 dB   | Aedes aegypti
 35.0 -  36.0s   |   1.0s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -0.7 dB   | ---
 35.5 -  37.5s   |   2.0s   | ENXAME_AEDES_PRESENTE        | 590.0Hz  |  +2.5 dB   | Aedes aegypti
 37.0 -  38.5s   |   1.5s   | PASSAGEM_CURTA_ISOLADA       | 613.8Hz  |  +0.7 dB   | Aedes aegypti
 38.0 -  39.5s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -0.8 dB   | ---
 39.0 -  40.0s   |   1.0s   | ENXAME_AEDES_PRESENTE        | 622.5Hz  |  +1.1 dB   | Aedes aegypti
 39.5 -  40.5s   |   1.0s   | PASSAGEM_CURTA_ISOLADA       | 600.0Hz  |  -0.1 dB   | Aedes aegypti
 40.0 -  41.5s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -1.5 dB   | ---
 41.0 -  47.5s   |   6.5s   | ENXAME_AEDES_PRESENTE        | 613.8Hz  |  +6.1 dB   | Aedes aegypti
 47.0 -  48.5s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -1.8 dB   | ---
 48.0 -  91.5s   |  43.5s   | ENXAME_AEDES_PRESENTE        | 602.5Hz  | +10.9 dB   | Aedes aegypti
 91.0 -  92.0s   |   1.0s   | PASSAGEM_CURTA_ISOLADA       | 547.5Hz  |  +0.2 dB   | Aedes aegypti
 91.5 -  92.5s   |   1.0s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -0.6 dB   | ---
 92.0 -  95.0s   |   3.0s   | ENXAME_AEDES_PRESENTE        | 590.0Hz  |  +3.6 dB   | Aedes aegypti
 94.5 -  96.0s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -9.8 dB   | ---
 95.5 -  99.5s   |   4.0s   | ENXAME_AEDES_PRESENTE        | 572.5Hz  |  +4.5 dB   | Aedes aegypti
 99.0 - 100.5s   |   1.5s   | PASSAGEM_CURTA_ISOLADA       | 596.2Hz  |  +0.7 dB   | Aedes aegypti
100.0 - 106.0s   |   6.0s   | ENXAME_AEDES_PRESENTE        | 597.5Hz  |  +3.0 dB   | Aedes aegypti
==================================================================================

-> Relatório detalhado salvo em CSV: /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/raw_brazil_predictions.csv
-> Resumo analítico salvo em JSON: /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/raw_brazil_summary.json

Exit code: 0

```

```json
{
  "audio_file": "data/raw/drive_test/drive_audio.wav",
  "duration_seconds": 106.24,
  "emulate_arduino_mic": false,
  "region_prior": "brazil",
  "total_windows": 211,
  "detected_windows": 139,
  "aedes_swarm_windows": 125,
  "active_ratio_pct": 65.88,
  "swarm_ratio_pct": 59.24,
  "mosquito_present": true,
  "aedes_swarm_detected": true,
  "median_flight_f0_hz": 600.0,
  "iqr_flight_f0_hz": 40.0,
  "primary_genus": "Aedes",
  "confirmed_genus": "Aedes",
  "primary_species": "Aedes aegypti",
  "confirmed_species": "Aedes aegypti",
  "genus_distribution": {
    "Aedes": 122,
    "Culiseta": 13,
    "Culex": 4
  },
  "species_distribution": {
    "Aedes aegypti": 132,
    "Culex quinquefasciatus": 5,
    "Culex pipiens": 2
  },
  "temporal_intervals": [
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 0.0,
      "end_s": 23.49,
      "duration_s": 23.49,
      "windows_count": 46,
      "median_f0_hz": 458.8,
      "median_aedes_snr_db": -2.33,
      "mean_presence_score": 0.3853,
      "dominant_species": null
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 23.0,
      "end_s": 24.99,
      "duration_s": 1.99,
      "windows_count": 3,
      "median_f0_hz": 640.0,
      "median_aedes_snr_db": -0.28,
      "mean_presence_score": 0.774,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 24.5,
      "end_s": 25.99,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 610.0,
      "median_aedes_snr_db": -1.0,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 25.5,
      "end_s": 26.49,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 595.0,
      "median_aedes_snr_db": 1.64,
      "mean_presence_score": 0.8049,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 26.0,
      "end_s": 26.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 525.0,
      "median_aedes_snr_db": 0.36,
      "mean_presence_score": 0.7823,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 26.5,
      "end_s": 33.99,
      "duration_s": 7.49,
      "windows_count": 14,
      "median_f0_hz": 557.5,
      "median_aedes_snr_db": -2.56,
      "mean_presence_score": 0.408,
      "dominant_species": null
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 33.5,
      "end_s": 35.49,
      "duration_s": 1.99,
      "windows_count": 3,
      "median_f0_hz": 575.0,
      "median_aedes_snr_db": -0.29,
      "mean_presence_score": 0.7765,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 35.0,
      "end_s": 35.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 570.0,
      "median_aedes_snr_db": -0.68,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 35.5,
      "end_s": 37.49,
      "duration_s": 1.99,
      "windows_count": 3,
      "median_f0_hz": 590.0,
      "median_aedes_snr_db": 2.52,
      "mean_presence_score": 0.7957,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 37.0,
      "end_s": 38.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 613.8,
      "median_aedes_snr_db": 0.66,
      "mean_presence_score": 0.7655,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 38.0,
      "end_s": 39.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 550.0,
      "median_aedes_snr_db": -0.76,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 39.0,
      "end_s": 39.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 622.5,
      "median_aedes_snr_db": 1.1,
      "mean_presence_score": 0.7812,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 39.5,
      "end_s": 40.49,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 600.0,
      "median_aedes_snr_db": -0.1,
      "mean_presence_score": 0.7647,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 40.0,
      "end_s": 41.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 593.8,
      "median_aedes_snr_db": -1.5,
      "mean_presence_score": 0.2326,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 41.0,
      "end_s": 47.49,
      "duration_s": 6.49,
      "windows_count": 12,
      "median_f0_hz": 613.8,
      "median_aedes_snr_db": 6.14,
      "mean_presence_score": 0.8105,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 47.0,
      "end_s": 48.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 596.2,
      "median_aedes_snr_db": -1.81,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 48.0,
      "end_s": 91.49,
      "duration_s": 43.49,
      "windows_count": 86,
      "median_f0_hz": 602.5,
      "median_aedes_snr_db": 10.94,
      "mean_presence_score": 0.8002,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 91.0,
      "end_s": 91.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 547.5,
      "median_aedes_snr_db": 0.19,
      "mean_presence_score": 0.6971,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 91.5,
      "end_s": 92.49,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 565.0,
      "median_aedes_snr_db": -0.57,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 92.0,
      "end_s": 94.99,
      "duration_s": 2.99,
      "windows_count": 5,
      "median_f0_hz": 590.0,
      "median_aedes_snr_db": 3.6,
      "mean_presence_score": 0.8078,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 94.5,
      "end_s": 95.99,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 607.5,
      "median_aedes_snr_db": -9.83,
      "mean_presence_score": 0.19,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 95.5,
      "end_s": 99.49,
      "duration_s": 3.99,
      "windows_count": 7,
      "median_f0_hz": 572.5,
      "median_aedes_snr_db": 4.49,
      "mean_presence_score": 0.8297,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 99.0,
      "end_s": 100.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 596.2,
      "median_aedes_snr_db": 0.65,
      "mean_presence_score": 0.8272,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 100.0,
      "end_s": 105.99,
      "duration_s": 5.99,
      "windows_count": 11,
      "median_f0_hz": 597.5,
      "median_aedes_snr_db": 2.96,
      "mean_presence_score": 0.7833,
      "dominant_species": "Aedes aegypti"
    }
  ],
  "swarm_timeline": [
    {
      "start_s": 0.0,
      "end_s": 23.0,
      "duration_s": 23.0,
      "swarm_present": false,
      "median_f0_hz": 458.8,
      "mean_aedes_snr_db": -4.310434782608696,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 23.0,
      "end_s": 24.5,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 640.0,
      "mean_aedes_snr_db": -0.13333333333333333,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 24.5,
      "end_s": 25.5,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 610.0,
      "mean_aedes_snr_db": -1.0,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.64,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 26.0,
      "end_s": 26.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 525.0,
      "mean_aedes_snr_db": 0.36,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 26.5,
      "end_s": 33.5,
      "duration_s": 7.0,
      "swarm_present": false,
      "median_f0_hz": 557.5,
      "mean_aedes_snr_db": -3.3700000000000006,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 33.5,
      "end_s": 35.0,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 575.0,
      "mean_aedes_snr_db": -0.19666666666666666,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 35.0,
      "end_s": 35.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 570.0,
      "mean_aedes_snr_db": -0.68,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 35.5,
      "end_s": 37.0,
      "duration_s": 1.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 2.783333333333333,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 37.0,
      "end_s": 38.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 613.8,
      "mean_aedes_snr_db": 0.655,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 38.0,
      "end_s": 39.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 550.0,
      "mean_aedes_snr_db": -0.755,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 622.5,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.5,
      "end_s": 40.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 600.0,
      "mean_aedes_snr_db": -0.1,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 40.0,
      "end_s": 41.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 593.8,
      "mean_aedes_snr_db": -1.5,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 41.0,
      "end_s": 47.0,
      "duration_s": 6.0,
      "swarm_present": true,
      "median_f0_hz": 613.8,
      "mean_aedes_snr_db": 6.095,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 47.0,
      "end_s": 48.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 596.2,
      "mean_aedes_snr_db": -1.81,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 48.0,
      "end_s": 91.0,
      "duration_s": 43.0,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.271162790697673,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 91.0,
      "end_s": 91.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 547.5,
      "mean_aedes_snr_db": 0.19,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 91.5,
      "end_s": 92.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 565.0,
      "mean_aedes_snr_db": -0.57,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 92.0,
      "end_s": 94.5,
      "duration_s": 2.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 3.464,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 94.5,
      "end_s": 95.5,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 607.5,
      "mean_aedes_snr_db": -9.829999999999998,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 95.5,
      "end_s": 99.0,
      "duration_s": 3.5,
      "swarm_present": true,
      "median_f0_hz": 572.5,
      "mean_aedes_snr_db": 4.6014285714285705,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 99.0,
      "end_s": 100.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 596.2,
      "mean_aedes_snr_db": 0.645,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.1363636363636362,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "swarm_active_intervals": [
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.64,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 35.5,
      "end_s": 37.0,
      "duration_s": 1.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 2.783333333333333,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 622.5,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 41.0,
      "end_s": 47.0,
      "duration_s": 6.0,
      "swarm_present": true,
      "median_f0_hz": 613.8,
      "mean_aedes_snr_db": 6.095,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 48.0,
      "end_s": 91.0,
      "duration_s": 43.0,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.271162790697673,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 92.0,
      "end_s": 94.5,
      "duration_s": 2.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 3.464,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 95.5,
      "end_s": 99.0,
      "duration_s": 3.5,
      "swarm_present": true,
      "median_f0_hz": 572.5,
      "mean_aedes_snr_db": 4.6014285714285705,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.1363636363636362,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "causal_swarm_timeline": [
    {
      "start_s": 0.0,
      "end_s": 23.0,
      "duration_s": 23.0,
      "swarm_present": false,
      "median_f0_hz": 458.8,
      "mean_aedes_snr_db": -4.310434782608696,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 23.0,
      "end_s": 24.5,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 640.0,
      "mean_aedes_snr_db": -0.13333333333333333,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 24.5,
      "end_s": 25.5,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 610.0,
      "mean_aedes_snr_db": -1.0,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.64,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 26.0,
      "end_s": 26.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 525.0,
      "mean_aedes_snr_db": 0.36,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 26.5,
      "end_s": 33.5,
      "duration_s": 7.0,
      "swarm_present": false,
      "median_f0_hz": 557.5,
      "mean_aedes_snr_db": -3.3700000000000006,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 33.5,
      "end_s": 35.0,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 575.0,
      "mean_aedes_snr_db": -0.19666666666666666,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 35.0,
      "end_s": 35.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 570.0,
      "mean_aedes_snr_db": -0.68,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 35.5,
      "end_s": 37.5,
      "duration_s": 2.0,
      "swarm_present": true,
      "median_f0_hz": 601.2,
      "mean_aedes_snr_db": 2.33,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 37.5,
      "end_s": 38.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 0.34,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 38.0,
      "end_s": 39.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 550.0,
      "mean_aedes_snr_db": -0.755,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 622.5,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.5,
      "end_s": 40.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 600.0,
      "mean_aedes_snr_db": -0.1,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 40.0,
      "end_s": 41.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 593.8,
      "mean_aedes_snr_db": -1.5,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 41.0,
      "end_s": 47.5,
      "duration_s": 6.5,
      "swarm_present": true,
      "median_f0_hz": 613.8,
      "mean_aedes_snr_db": 5.538461538461538,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 47.5,
      "end_s": 48.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 572.5,
      "mean_aedes_snr_db": -2.48,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 48.0,
      "end_s": 91.5,
      "duration_s": 43.5,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.155287356321839,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 91.5,
      "end_s": 92.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 565.0,
      "mean_aedes_snr_db": -0.57,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 92.0,
      "end_s": 95.0,
      "duration_s": 3.0,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 1.241666666666667,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 95.0,
      "end_s": 95.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 605.0,
      "mean_aedes_snr_db": -9.79,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 95.5,
      "end_s": 99.5,
      "duration_s": 4.0,
      "swarm_present": true,
      "median_f0_hz": 580.0,
      "mean_aedes_snr_db": 4.08,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 99.5,
      "end_s": 100.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 605.0,
      "mean_aedes_snr_db": 0.86,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.1363636363636362,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "causal_swarm_active_intervals": [
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.64,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 35.5,
      "end_s": 37.5,
      "duration_s": 2.0,
      "swarm_present": true,
      "median_f0_hz": 601.2,
      "mean_aedes_snr_db": 2.33,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 622.5,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 41.0,
      "end_s": 47.5,
      "duration_s": 6.5,
      "swarm_present": true,
      "median_f0_hz": 613.8,
      "mean_aedes_snr_db": 5.538461538461538,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 48.0,
      "end_s": 91.5,
      "duration_s": 43.5,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.155287356321839,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 92.0,
      "end_s": 95.0,
      "duration_s": 3.0,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 1.241666666666667,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 95.5,
      "end_s": 99.5,
      "duration_s": 4.0,
      "swarm_present": true,
      "median_f0_hz": 580.0,
      "mean_aedes_snr_db": 4.08,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.1363636363636362,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "timeline_resolution_s": 0.5,
  "analysis_window_s": 0.992,
  "unanalysed_tail_s": 0.248,
  "timeline_interpretation": "Estimated classifier decisions; swarm_timeline uses offline 1-step smoothing, whereas causal_swarm_timeline uses strictly causal past-only persistence."
}
```

## raw_global

```text
Command: /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/.venv/bin/python /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/scripts/predict_audio.py --audio data/raw/drive_test/drive_audio.wav --region global --output-csv /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/raw_global_predictions.csv --output-summary /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/raw_global_summary.json

==================================================================================
 RELATÓRIO HIERÁRQUICO E SEGMENTAÇÃO TEMPORAL DE ENXAME DE AEDES (ML + SHS) 
==================================================================================
Arquivo:               data/raw/drive_test/drive_audio.wav
Duração Total:         106.24 s
Emulação Mic Arduino:  DESATIVADA (Áudio Bruto)
Prior Epidemiológico:  GLOBAL
Janelas de Voo Ativo:  139/211 (65.88%)
Janelas de Enxame:     125/211 (59.24%)
Diagnóstico Tier 1:    ENXAME DE AEDES DETECTADO
Fundamental Mediana:   F0 = 600.0 Hz (IQR = 40.0 Hz)

--- TIER 2: VETOR EPIDEMIOLÓGICO (GÊNERO COM CORREÇÃO DE CARDINALIDADE) ---
Gênero Identificado:   Aedes (Confirmação Causal: True)
  * Aedes         : 123 janelas ( 88.5%)
  * Culex         :  14 janelas ( 10.1%)
  * Anopheles     :   2 janelas (  1.4%)

--- TIER 3: ESPÉCIE ESPECÍFICA (FUSÃO HARMÔNICA + NEURAL) ---
Espécie Confirmada:    Aedes aegypti
  * Aedes aegypti             : 112 janelas ( 80.6%)
  * Anopheles gambiae         :  14 janelas ( 10.1%)
  * Culex tarsalis            :   5 janelas (  3.6%)
  * Anopheles stephensi       :   4 janelas (  2.9%)
  * Culex pipiens             :   2 janelas (  1.4%)

--- MAPA DE INTERVALOS TEMPORAIS (QUANDO O ENXAME APARECE vs QUANDO NÃO APARECE) ---
Intervalo (s)    | Duração  | Status / Regime              | F0 Med   | SNR Aedes  | Espécie Dominante
--------------------------------------------------------------------------------------------------
  0.0 -  23.5s   |  23.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -2.3 dB   | ---
 23.0 -  25.0s   |   2.0s   | PASSAGEM_CURTA_ISOLADA       | 640.0Hz  |  -0.3 dB   | Aedes aegypti
 24.5 -  26.0s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -1.0 dB   | ---
 25.5 -  26.5s   |   1.0s   | ENXAME_AEDES_PRESENTE        | 595.0Hz  |  +1.6 dB   | Aedes aegypti
 26.0 -  27.0s   |   1.0s   | PASSAGEM_CURTA_ISOLADA       | 525.0Hz  |  +0.4 dB   | Aedes aegypti
 26.5 -  34.0s   |   7.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -2.6 dB   | ---
 33.5 -  35.5s   |   2.0s   | PASSAGEM_CURTA_ISOLADA       | 575.0Hz  |  -0.3 dB   | Aedes aegypti
 35.0 -  36.0s   |   1.0s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -0.7 dB   | ---
 35.5 -  37.5s   |   2.0s   | ENXAME_AEDES_PRESENTE        | 590.0Hz  |  +2.5 dB   | Aedes aegypti
 37.0 -  38.5s   |   1.5s   | PASSAGEM_CURTA_ISOLADA       | 613.8Hz  |  +0.7 dB   | Aedes aegypti
 38.0 -  39.5s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -0.8 dB   | ---
 39.0 -  40.0s   |   1.0s   | ENXAME_AEDES_PRESENTE        | 622.5Hz  |  +1.1 dB   | Aedes aegypti
 39.5 -  40.5s   |   1.0s   | PASSAGEM_CURTA_ISOLADA       | 600.0Hz  |  -0.1 dB   | Aedes aegypti
 40.0 -  41.5s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -1.5 dB   | ---
 41.0 -  47.5s   |   6.5s   | ENXAME_AEDES_PRESENTE        | 613.8Hz  |  +6.1 dB   | Aedes aegypti
 47.0 -  48.5s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -1.8 dB   | ---
 48.0 -  91.5s   |  43.5s   | ENXAME_AEDES_PRESENTE        | 602.5Hz  | +10.9 dB   | Aedes aegypti
 91.0 -  92.0s   |   1.0s   | PASSAGEM_CURTA_ISOLADA       | 547.5Hz  |  +0.2 dB   | Anopheles gambiae
 91.5 -  92.5s   |   1.0s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -0.6 dB   | ---
 92.0 -  95.0s   |   3.0s   | ENXAME_AEDES_PRESENTE        | 590.0Hz  |  +3.6 dB   | Aedes aegypti
 94.5 -  96.0s   |   1.5s   | AUSENTE_RUIDO_FUNDO          |   ---    |  -9.8 dB   | ---
 95.5 -  99.5s   |   4.0s   | ENXAME_AEDES_PRESENTE        | 572.5Hz  |  +4.5 dB   | Aedes aegypti
 99.0 - 100.5s   |   1.5s   | PASSAGEM_CURTA_ISOLADA       | 596.2Hz  |  +0.7 dB   | Aedes aegypti
100.0 - 106.0s   |   6.0s   | ENXAME_AEDES_PRESENTE        | 597.5Hz  |  +3.0 dB   | Aedes aegypti
==================================================================================

-> Relatório detalhado salvo em CSV: /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/raw_global_predictions.csv
-> Resumo analítico salvo em JSON: /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_review_2026_10_14_161041/raw_global_summary.json

Exit code: 0

```

```json
{
  "audio_file": "data/raw/drive_test/drive_audio.wav",
  "duration_seconds": 106.24,
  "emulate_arduino_mic": false,
  "region_prior": "global",
  "total_windows": 211,
  "detected_windows": 139,
  "aedes_swarm_windows": 125,
  "active_ratio_pct": 65.88,
  "swarm_ratio_pct": 59.24,
  "mosquito_present": true,
  "aedes_swarm_detected": true,
  "median_flight_f0_hz": 600.0,
  "iqr_flight_f0_hz": 40.0,
  "primary_genus": "Aedes",
  "confirmed_genus": "Aedes",
  "primary_species": "Aedes aegypti",
  "confirmed_species": "Aedes aegypti",
  "genus_distribution": {
    "Aedes": 123,
    "Culex": 14,
    "Anopheles": 2
  },
  "species_distribution": {
    "Aedes aegypti": 112,
    "Anopheles gambiae": 14,
    "Culex tarsalis": 5,
    "Anopheles stephensi": 4,
    "Culex pipiens": 2,
    "Aedes sierrensis": 1,
    "Aedes albopictus": 1
  },
  "temporal_intervals": [
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 0.0,
      "end_s": 23.49,
      "duration_s": 23.49,
      "windows_count": 46,
      "median_f0_hz": 458.8,
      "median_aedes_snr_db": -2.33,
      "mean_presence_score": 0.3853,
      "dominant_species": null
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 23.0,
      "end_s": 24.99,
      "duration_s": 1.99,
      "windows_count": 3,
      "median_f0_hz": 640.0,
      "median_aedes_snr_db": -0.28,
      "mean_presence_score": 0.774,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 24.5,
      "end_s": 25.99,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 610.0,
      "median_aedes_snr_db": -1.0,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 25.5,
      "end_s": 26.49,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 595.0,
      "median_aedes_snr_db": 1.64,
      "mean_presence_score": 0.8049,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 26.0,
      "end_s": 26.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 525.0,
      "median_aedes_snr_db": 0.36,
      "mean_presence_score": 0.7823,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 26.5,
      "end_s": 33.99,
      "duration_s": 7.49,
      "windows_count": 14,
      "median_f0_hz": 557.5,
      "median_aedes_snr_db": -2.56,
      "mean_presence_score": 0.408,
      "dominant_species": null
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 33.5,
      "end_s": 35.49,
      "duration_s": 1.99,
      "windows_count": 3,
      "median_f0_hz": 575.0,
      "median_aedes_snr_db": -0.29,
      "mean_presence_score": 0.7765,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 35.0,
      "end_s": 35.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 570.0,
      "median_aedes_snr_db": -0.68,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 35.5,
      "end_s": 37.49,
      "duration_s": 1.99,
      "windows_count": 3,
      "median_f0_hz": 590.0,
      "median_aedes_snr_db": 2.52,
      "mean_presence_score": 0.7957,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 37.0,
      "end_s": 38.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 613.8,
      "median_aedes_snr_db": 0.66,
      "mean_presence_score": 0.7655,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 38.0,
      "end_s": 39.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 550.0,
      "median_aedes_snr_db": -0.76,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 39.0,
      "end_s": 39.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 622.5,
      "median_aedes_snr_db": 1.1,
      "mean_presence_score": 0.7812,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 39.5,
      "end_s": 40.49,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 600.0,
      "median_aedes_snr_db": -0.1,
      "mean_presence_score": 0.7647,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 40.0,
      "end_s": 41.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 593.8,
      "median_aedes_snr_db": -1.5,
      "mean_presence_score": 0.2326,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 41.0,
      "end_s": 47.49,
      "duration_s": 6.49,
      "windows_count": 12,
      "median_f0_hz": 613.8,
      "median_aedes_snr_db": 6.14,
      "mean_presence_score": 0.8105,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 47.0,
      "end_s": 48.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 596.2,
      "median_aedes_snr_db": -1.81,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 48.0,
      "end_s": 91.49,
      "duration_s": 43.49,
      "windows_count": 86,
      "median_f0_hz": 602.5,
      "median_aedes_snr_db": 10.94,
      "mean_presence_score": 0.8002,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 91.0,
      "end_s": 91.99,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 547.5,
      "median_aedes_snr_db": 0.19,
      "mean_presence_score": 0.6971,
      "dominant_species": "Anopheles gambiae"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 91.5,
      "end_s": 92.49,
      "duration_s": 0.99,
      "windows_count": 1,
      "median_f0_hz": 565.0,
      "median_aedes_snr_db": -0.57,
      "mean_presence_score": 0.45,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 92.0,
      "end_s": 94.99,
      "duration_s": 2.99,
      "windows_count": 5,
      "median_f0_hz": 590.0,
      "median_aedes_snr_db": 3.6,
      "mean_presence_score": 0.8078,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "AUSENTE_RUIDO_FUNDO",
      "aedes_present": false,
      "is_sustained_swarm": false,
      "start_s": 94.5,
      "end_s": 95.99,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 607.5,
      "median_aedes_snr_db": -9.83,
      "mean_presence_score": 0.19,
      "dominant_species": null
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 95.5,
      "end_s": 99.49,
      "duration_s": 3.99,
      "windows_count": 7,
      "median_f0_hz": 572.5,
      "median_aedes_snr_db": 4.49,
      "mean_presence_score": 0.8297,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "PASSAGEM_CURTA_ISOLADA",
      "aedes_present": true,
      "is_sustained_swarm": false,
      "start_s": 99.0,
      "end_s": 100.49,
      "duration_s": 1.49,
      "windows_count": 2,
      "median_f0_hz": 596.2,
      "median_aedes_snr_db": 0.65,
      "mean_presence_score": 0.8272,
      "dominant_species": "Aedes aegypti"
    },
    {
      "segment_type": "ENXAME_AEDES_PRESENTE",
      "aedes_present": true,
      "is_sustained_swarm": true,
      "start_s": 100.0,
      "end_s": 105.99,
      "duration_s": 5.99,
      "windows_count": 11,
      "median_f0_hz": 597.5,
      "median_aedes_snr_db": 2.96,
      "mean_presence_score": 0.7833,
      "dominant_species": "Aedes aegypti"
    }
  ],
  "swarm_timeline": [
    {
      "start_s": 0.0,
      "end_s": 23.0,
      "duration_s": 23.0,
      "swarm_present": false,
      "median_f0_hz": 458.8,
      "mean_aedes_snr_db": -4.310434782608696,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 23.0,
      "end_s": 24.5,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 640.0,
      "mean_aedes_snr_db": -0.13333333333333333,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 24.5,
      "end_s": 25.5,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 610.0,
      "mean_aedes_snr_db": -1.0,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.64,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 26.0,
      "end_s": 26.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 525.0,
      "mean_aedes_snr_db": 0.36,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 26.5,
      "end_s": 33.5,
      "duration_s": 7.0,
      "swarm_present": false,
      "median_f0_hz": 557.5,
      "mean_aedes_snr_db": -3.3700000000000006,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 33.5,
      "end_s": 35.0,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 575.0,
      "mean_aedes_snr_db": -0.19666666666666666,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 35.0,
      "end_s": 35.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 570.0,
      "mean_aedes_snr_db": -0.68,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 35.5,
      "end_s": 37.0,
      "duration_s": 1.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 2.783333333333333,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 37.0,
      "end_s": 38.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 613.8,
      "mean_aedes_snr_db": 0.655,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 38.0,
      "end_s": 39.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 550.0,
      "mean_aedes_snr_db": -0.755,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 622.5,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.5,
      "end_s": 40.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 600.0,
      "mean_aedes_snr_db": -0.1,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 40.0,
      "end_s": 41.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 593.8,
      "mean_aedes_snr_db": -1.5,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 41.0,
      "end_s": 47.0,
      "duration_s": 6.0,
      "swarm_present": true,
      "median_f0_hz": 613.8,
      "mean_aedes_snr_db": 6.095,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 47.0,
      "end_s": 48.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 596.2,
      "mean_aedes_snr_db": -1.81,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 48.0,
      "end_s": 91.0,
      "duration_s": 43.0,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.271162790697673,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 91.0,
      "end_s": 91.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 547.5,
      "mean_aedes_snr_db": 0.19,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 91.5,
      "end_s": 92.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 565.0,
      "mean_aedes_snr_db": -0.57,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 92.0,
      "end_s": 94.5,
      "duration_s": 2.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 3.464,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 94.5,
      "end_s": 95.5,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 607.5,
      "mean_aedes_snr_db": -9.829999999999998,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 95.5,
      "end_s": 99.0,
      "duration_s": 3.5,
      "swarm_present": true,
      "median_f0_hz": 572.5,
      "mean_aedes_snr_db": 4.6014285714285705,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 99.0,
      "end_s": 100.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 596.2,
      "mean_aedes_snr_db": 0.645,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.1363636363636362,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "swarm_active_intervals": [
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.64,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 35.5,
      "end_s": 37.0,
      "duration_s": 1.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 2.783333333333333,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 622.5,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 41.0,
      "end_s": 47.0,
      "duration_s": 6.0,
      "swarm_present": true,
      "median_f0_hz": 613.8,
      "mean_aedes_snr_db": 6.095,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 48.0,
      "end_s": 91.0,
      "duration_s": 43.0,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.271162790697673,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 92.0,
      "end_s": 94.5,
      "duration_s": 2.5,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 3.464,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 95.5,
      "end_s": 99.0,
      "duration_s": 3.5,
      "swarm_present": true,
      "median_f0_hz": 572.5,
      "mean_aedes_snr_db": 4.6014285714285705,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.1363636363636362,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "causal_swarm_timeline": [
    {
      "start_s": 0.0,
      "end_s": 23.0,
      "duration_s": 23.0,
      "swarm_present": false,
      "median_f0_hz": 458.8,
      "mean_aedes_snr_db": -4.310434782608696,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 23.0,
      "end_s": 24.5,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 640.0,
      "mean_aedes_snr_db": -0.13333333333333333,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 24.5,
      "end_s": 25.5,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 610.0,
      "mean_aedes_snr_db": -1.0,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.64,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 26.0,
      "end_s": 26.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 525.0,
      "mean_aedes_snr_db": 0.36,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 26.5,
      "end_s": 33.5,
      "duration_s": 7.0,
      "swarm_present": false,
      "median_f0_hz": 557.5,
      "mean_aedes_snr_db": -3.3700000000000006,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 33.5,
      "end_s": 35.0,
      "duration_s": 1.5,
      "swarm_present": false,
      "median_f0_hz": 575.0,
      "mean_aedes_snr_db": -0.19666666666666666,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 35.0,
      "end_s": 35.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 570.0,
      "mean_aedes_snr_db": -0.68,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 35.5,
      "end_s": 37.5,
      "duration_s": 2.0,
      "swarm_present": true,
      "median_f0_hz": 601.2,
      "mean_aedes_snr_db": 2.33,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 37.5,
      "end_s": 38.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 0.34,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 38.0,
      "end_s": 39.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 550.0,
      "mean_aedes_snr_db": -0.755,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 622.5,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.5,
      "end_s": 40.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 600.0,
      "mean_aedes_snr_db": -0.1,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 40.0,
      "end_s": 41.0,
      "duration_s": 1.0,
      "swarm_present": false,
      "median_f0_hz": 593.8,
      "mean_aedes_snr_db": -1.5,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 41.0,
      "end_s": 47.5,
      "duration_s": 6.5,
      "swarm_present": true,
      "median_f0_hz": 613.8,
      "mean_aedes_snr_db": 5.538461538461538,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 47.5,
      "end_s": 48.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 572.5,
      "mean_aedes_snr_db": -2.48,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 48.0,
      "end_s": 91.5,
      "duration_s": 43.5,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.155287356321839,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 91.5,
      "end_s": 92.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 565.0,
      "mean_aedes_snr_db": -0.57,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 92.0,
      "end_s": 95.0,
      "duration_s": 3.0,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 1.241666666666667,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 95.0,
      "end_s": 95.5,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 605.0,
      "mean_aedes_snr_db": -9.79,
      "segment_type": "AUSENTE_RUIDO_FUNDO"
    },
    {
      "start_s": 95.5,
      "end_s": 99.5,
      "duration_s": 4.0,
      "swarm_present": true,
      "median_f0_hz": 580.0,
      "mean_aedes_snr_db": 4.08,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 99.5,
      "end_s": 100.0,
      "duration_s": 0.5,
      "swarm_present": false,
      "median_f0_hz": 605.0,
      "mean_aedes_snr_db": 0.86,
      "segment_type": "PASSAGEM_CURTA_ISOLADA"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.1363636363636362,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "causal_swarm_active_intervals": [
    {
      "start_s": 25.5,
      "end_s": 26.0,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 595.0,
      "mean_aedes_snr_db": 1.64,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 35.5,
      "end_s": 37.5,
      "duration_s": 2.0,
      "swarm_present": true,
      "median_f0_hz": 601.2,
      "mean_aedes_snr_db": 2.33,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 39.0,
      "end_s": 39.5,
      "duration_s": 0.5,
      "swarm_present": true,
      "median_f0_hz": 622.5,
      "mean_aedes_snr_db": 1.1,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 41.0,
      "end_s": 47.5,
      "duration_s": 6.5,
      "swarm_present": true,
      "median_f0_hz": 613.8,
      "mean_aedes_snr_db": 5.538461538461538,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 48.0,
      "end_s": 91.5,
      "duration_s": 43.5,
      "swarm_present": true,
      "median_f0_hz": 602.5,
      "mean_aedes_snr_db": 10.155287356321839,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 92.0,
      "end_s": 95.0,
      "duration_s": 3.0,
      "swarm_present": true,
      "median_f0_hz": 590.0,
      "mean_aedes_snr_db": 1.241666666666667,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 95.5,
      "end_s": 99.5,
      "duration_s": 4.0,
      "swarm_present": true,
      "median_f0_hz": 580.0,
      "mean_aedes_snr_db": 4.08,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    },
    {
      "start_s": 100.0,
      "end_s": 105.992,
      "duration_s": 5.992,
      "swarm_present": true,
      "median_f0_hz": 597.5,
      "mean_aedes_snr_db": 3.1363636363636362,
      "segment_type": "ENXAME_AEDES_PRESENTE"
    }
  ],
  "timeline_resolution_s": 0.5,
  "analysis_window_s": 0.992,
  "unanalysed_tail_s": 0.248,
  "timeline_interpretation": "Estimated classifier decisions; swarm_timeline uses offline 1-step smoothing, whereas causal_swarm_timeline uses strictly causal past-only persistence."
}
```

## Suíte completa

```text
........................................................................ [ 33%]
........................................................................ [ 66%]
........F............................................................... [100%]
=================================== FAILURES ===================================
_____________ test_cepstrum_is_scale_normalized_and_retains_shape ______________

    def test_cepstrum_is_scale_normalized_and_retains_shape():
        rng = np.random.default_rng(8)
        p = np.exp(rng.normal(size=(801, 91)))
        floor, _ = local_noise_floor(p, 10)
        a = centred_cepstrum(p, floor)
        b = centred_cepstrum(p * 100, floor * 100)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-11)
>       assert a[0][0] == 0 and np.isclose(np.sqrt(np.mean(a[0][1:] ** 2)), 2.5, atol=1e-5)
E       AssertionError: assert (np.float64(0.0) == 0 and np.False_)
E        +  where np.False_ = <function isclose at 0x7f6584140cb0>(np.float64(0.2895001905324099), 2.5, atol=1e-05)
E        +    where <function isclose at 0x7f6584140cb0> = np.isclose
E        +    and   np.float64(0.2895001905324099) = <ufunc 'sqrt'>(np.float64(0.08381036031830164))
E        +      where <ufunc 'sqrt'> = np.sqrt
E        +      and   np.float64(0.08381036031830164) = <function mean at 0x7f658412e2b0>((array([-0.04763368, -0.0128026 ,  0.56676818,  0.2522713 , -0.42106866,\n       -0.04393659,  0.20228878,  0.44435051, -0.27779018, -0.16784621,\n        0.30038108,  0.072516  ]) ** 2))
E        +        where <function mean at 0x7f658412e2b0> = np.mean

tests/test_nested_species_v5.py:69: AssertionError
=============================== warnings summary ===============================
tests/test_nested_species.py: 38 warnings
tests/test_nested_species_v2.py: 76 warnings
tests/test_nested_species_v3.py: 9092 warnings
tests/test_nested_species_v4.py: 36064 warnings
tests/test_reproduction.py: 4002 warnings
  /home/lciarallo/.local/lib/python3.14/site-packages/joblib/numpy_pickle.py:207: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    array.shape = self.shape

tests/test_nested_species_v2.py::test_completed_v2_artifacts_and_scalers_respect_fit_only
  /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/tests/test_nested_species_v2.py:130: DtypeWarning: Columns (0: fold) have mixed types. Specify dtype option on import or set low_memory=False.
    parts=pd.read_csv(OUT/'partitions.csv')

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_nested_species_v5.py::test_cepstrum_is_scale_normalized_and_retains_shape
1 failed, 215 passed, 49273 warnings in 37.39s

```

## Reexecução por arquivo

```text
CLI emulated_brazil 139 {'Aedes aegypti': 132, 'Culex quinquefasciatus': 5, 'Culex pipiens': 2}
CLI raw_brazil 139 {'Aedes aegypti': 132, 'Culex quinquefasciatus': 5, 'Culex pipiens': 2}
CLI raw_global 139 {'Aedes aegypti': 112, 'Anopheles gambiae': 14, 'Culex tarsalis': 5, 'Anopheles stephensi': 4, 'Culex pipiens': 2, 'Aedes sierrensis': 1, 'Aedes albopictus': 1}
REPLAY previous S_1_D_3_75.wav Aedes aegypti correct=True
REPLAY previous S_1_D_3_76.wav Aedes aegypti correct=True
REPLAY previous S_1_D_3_77.wav Aedes sierrensis correct=False
REPLAY previous S_1_D_3_78.wav Aedes aegypti correct=True
REPLAY previous S_1_D_3_79.wav Aedes aegypti correct=True
REPLAY previous S_1_D_3_80.wav Aedes aegypti correct=True
REPLAY previous S_1_D_3_81.wav Aedes albopictus correct=False
REPLAY previous S_1_D_3_82.wav Aedes aegypti correct=True
REPLAY previous S_1_D_3_85.wav Aedes sierrensis correct=False
REPLAY previous S_1_D_3_86.wav Anopheles farauti correct=False
REPLAY previous S_1_D_3_87.wav Aedes aegypti correct=True
REPLAY previous S_1_D_3_88.wav Aedes sierrensis correct=False
REPLAY previous S_2_D_3_14.wav Aedes aegypti correct=False
REPLAY previous S_2_D_3_15.wav Aedes aegypti correct=False
REPLAY previous S_2_D_3_16.wav Aedes sierrensis correct=False
REPLAY previous S_2_D_3_17.wav Aedes aegypti correct=False
REPLAY previous S_2_D_3_18.wav Anopheles farauti correct=False
REPLAY previous S_2_D_3_19.wav Aedes sierrensis correct=False
REPLAY previous S_7_D_3_14.wav Anopheles minimus correct=False
REPLAY previous S_7_D_3_15.wav Aedes albopictus correct=False
REPLAY previous S_7_D_3_16.wav Aedes aegypti correct=False
REPLAY previous S_7_D_3_17.wav Aedes albopictus correct=False
REPLAY previous S_7_D_3_18.wav Anopheles minimus correct=False
REPLAY previous S_7_D_3_19.wav Aedes albopictus correct=False
REPLAY previous S_7_D_3_20.wav Aedes sierrensis correct=False
REPLAY previous S_7_D_3_21.wav Aedes sierrensis correct=False
REPLAY previous S_7_D_3_22.wav Aedes sierrensis correct=False
REPLAY previous S_8_D_3_14.wav Anopheles minimus correct=True
REPLAY previous S_8_D_3_15.wav Anopheles minimus correct=True
REPLAY previous S_8_D_3_16.wav Anopheles minimus correct=True
REPLAY previous S_8_D_3_17.wav Anopheles minimus correct=True
REPLAY previous S_8_D_3_18.wav Anopheles minimus correct=True
REPLAY previous S_8_D_3_19.wav Anopheles minimus correct=True
REPLAY previous S_8_D_3_20.wav Anopheles minimus correct=True
REPLAY previous S_8_D_3_21.wav Aedes aegypti correct=False
REPLAY previous S_8_D_3_22.wav NO_CANDIDATE correct=False
REPLAY previous S_9_D_3_15.wav Anopheles minimus correct=False
REPLAY previous S_9_D_3_16.wav Anopheles stephensi correct=True
REPLAY previous S_9_D_3_17.wav Anopheles stephensi correct=True
REPLAY previous S_9_D_3_18.wav Aedes aegypti correct=False
REPLAY previous S_9_D_3_19.wav Aedes albopictus correct=False
REPLAY previous S_9_D_3_20.wav Anopheles stephensi correct=True
REPLAY previous S_9_D_3_21.wav NO_CANDIDATE correct=False
REPLAY previous S_9_D_3_22.wav Anopheles dirus correct=False
REPLAY previous drive_audio.wav Aedes aegypti correct=True
REPLAY current S_1_D_3_75.wav Aedes albopictus correct=False
REPLAY current S_1_D_3_76.wav Aedes aegypti correct=True
REPLAY current S_1_D_3_77.wav Aedes aegypti correct=True
REPLAY current S_1_D_3_78.wav Aedes aegypti correct=True
REPLAY current S_1_D_3_79.wav Aedes albopictus correct=False
REPLAY current S_1_D_3_80.wav Aedes aegypti correct=True
REPLAY current S_1_D_3_81.wav Aedes albopictus correct=False
REPLAY current S_1_D_3_82.wav Aedes aegypti correct=True
REPLAY current S_1_D_3_85.wav Aedes sierrensis correct=False
REPLAY current S_1_D_3_86.wav Aedes aegypti correct=True
REPLAY current S_1_D_3_87.wav Aedes aegypti correct=True
REPLAY current S_1_D_3_88.wav Aedes aegypti correct=True
REPLAY current S_2_D_3_14.wav Aedes albopictus correct=True
REPLAY current S_2_D_3_15.wav Aedes albopictus correct=True
REPLAY current S_2_D_3_16.wav Aedes albopictus correct=True
REPLAY current S_2_D_3_17.wav Aedes albopictus correct=True
REPLAY current S_2_D_3_18.wav Aedes albopictus correct=True
REPLAY current S_2_D_3_19.wav Aedes sierrensis correct=False
REPLAY current S_7_D_3_14.wav Culex pipiens correct=True
REPLAY current S_7_D_3_15.wav Culex pipiens correct=True
REPLAY current S_7_D_3_16.wav Culex pipiens correct=True
REPLAY current S_7_D_3_17.wav Culex pipiens correct=True
REPLAY current S_7_D_3_18.wav Culex pipiens correct=True
REPLAY current S_7_D_3_19.wav Culex pipiens correct=True
REPLAY current S_7_D_3_20.wav Aedes sierrensis correct=False
REPLAY current S_7_D_3_21.wav Culex pipiens correct=True
REPLAY current S_7_D_3_22.wav Aedes sierrensis correct=False
REPLAY current S_8_D_3_14.wav Anopheles minimus correct=True
REPLAY current S_8_D_3_15.wav Anopheles minimus correct=True
REPLAY current S_8_D_3_16.wav Anopheles minimus correct=True
REPLAY current S_8_D_3_17.wav Anopheles minimus correct=True
REPLAY current S_8_D_3_18.wav Anopheles minimus correct=True
REPLAY current S_8_D_3_19.wav Anopheles minimus correct=True
REPLAY current S_8_D_3_20.wav Aedes aegypti correct=False
REPLAY current S_8_D_3_21.wav Aedes aegypti correct=False
REPLAY current S_8_D_3_22.wav NO_CANDIDATE correct=False
REPLAY current S_9_D_3_15.wav Anopheles stephensi correct=True
REPLAY current S_9_D_3_16.wav Anopheles stephensi correct=True
REPLAY current S_9_D_3_17.wav Anopheles stephensi correct=True
REPLAY current S_9_D_3_18.wav Anopheles stephensi correct=True
REPLAY current S_9_D_3_19.wav Aedes albopictus correct=False
REPLAY current S_9_D_3_20.wav Anopheles stephensi correct=True
REPLAY current S_9_D_3_21.wav NO_CANDIDATE correct=False
REPLAY current S_9_D_3_22.wav Aedes aegypti correct=False
REPLAY current drive_audio.wav Aedes aegypti correct=True
METRIC {"n": 28, "correct": 12, "accuracy": 0.42857142857142855, "balanced_accuracy": 0.4111111111111111, "macro_f1_support_only": 0.39244264507422405, "macro_f1_strict_union": 0.28031617505301715, "macro_f1_support_minus_strict": 0.1121264700212069, "predicted_taxa_outside_support": ["Aedes sierrensis", "Anopheles farauti"], "coverage": 0.9285714285714286, "emitted": 26, "accepted_accuracy": 0.46153846153846156, "correct_emissions_over_all": 0.42857142857142855, "accepted_output_is_candidate_mode": false, "scope": "received_28", "task": "species", "version": "previous"}
METRIC {"n": 28, "correct": 26, "accuracy": 0.9285714285714286, "balanced_accuracy": 0.9555555555555555, "macro_f1_support_only": 0.9568181818181818, "macro_f1_strict_union": 0.7973484848484849, "macro_f1_support_minus_strict": 0.15946969696969693, "predicted_taxa_outside_support": ["Aedes sierrensis"], "coverage": 0.8928571428571429, "emitted": 25, "accepted_accuracy": 0.92, "correct_emissions_over_all": 0.8214285714285714, "accepted_output_is_candidate_mode": true, "scope": "received_28", "task": "species", "version": "current"}
METRIC {"n": 28, "correct": 20, "accuracy": 0.7142857142857143, "balanced_accuracy": 0.5687830687830688, "macro_f1_support_only": 0.5294117647058824, "macro_f1_strict_union": 0.5294117647058824, "macro_f1_support_minus_strict": 0.0, "predicted_taxa_outside_support": [], "coverage": 0.9642857142857143, "emitted": 27, "accepted_accuracy": 0.7037037037037037, "correct_emissions_over_all": 0.6785714285714286, "accepted_output_is_candidate_mode": true, "scope": "received_28", "task": "genus", "version": "previous"}
METRIC {"n": 28, "correct": 26, "accuracy": 0.9285714285714286, "balanced_accuracy": 0.9259259259259259, "macro_f1_support_only": 0.9361111111111112, "macro_f1_strict_union": 0.9361111111111112, "macro_f1_support_minus_strict": 0.0, "predicted_taxa_outside_support": [], "coverage": 0.9642857142857143, "emitted": 27, "accepted_accuracy": 0.9259259259259259, "correct_emissions_over_all": 0.8928571428571429, "accepted_output_is_candidate_mode": true, "scope": "received_28", "task": "genus", "version": "current"}
METRIC {"n": 17, "correct": 6, "accuracy": 0.35294117647058826, "balanced_accuracy": 0.3, "macro_f1_support_only": 0.3633333333333333, "macro_f1_strict_union": 0.22708333333333333, "macro_f1_support_minus_strict": 0.13624999999999995, "predicted_taxa_outside_support": ["Aedes sierrensis", "Anopheles dirus", "NO_CANDIDATE"], "coverage": 0.8235294117647058, "emitted": 14, "accepted_accuracy": 0.42857142857142855, "correct_emissions_over_all": 0.35294117647058826, "accepted_output_is_candidate_mode": true, "scope": "paired_17", "task": "species", "version": "previous"}
METRIC {"n": 17, "correct": 6, "accuracy": 0.35294117647058826, "balanced_accuracy": 0.3, "macro_f1_support_only": 0.3822222222222222, "macro_f1_strict_union": 0.273015873015873, "macro_f1_support_minus_strict": 0.1092063492063492, "predicted_taxa_outside_support": ["Aedes sierrensis", "NO_CANDIDATE"], "coverage": 0.7058823529411765, "emitted": 12, "accepted_accuracy": 0.5, "correct_emissions_over_all": 0.35294117647058826, "accepted_output_is_candidate_mode": true, "scope": "paired_17", "task": "species", "version": "current"}
METRIC {"n": 17, "correct": 7, "accuracy": 0.4117647058823529, "balanced_accuracy": 0.4166666666666667, "macro_f1_support_only": 0.31851851851851853, "macro_f1_strict_union": 0.2388888888888889, "macro_f1_support_minus_strict": 0.07962962962962963, "predicted_taxa_outside_support": ["NO_CANDIDATE"], "coverage": 0.8235294117647058, "emitted": 14, "accepted_accuracy": 0.5, "correct_emissions_over_all": 0.4117647058823529, "accepted_output_is_candidate_mode": true, "scope": "paired_17", "task": "genus", "version": "previous"}
METRIC {"n": 17, "correct": 10, "accuracy": 0.5882352941176471, "balanced_accuracy": 0.625, "macro_f1_support_only": 0.6262626262626262, "macro_f1_strict_union": 0.4696969696969696, "macro_f1_support_minus_strict": 0.15656565656565657, "predicted_taxa_outside_support": ["NO_CANDIDATE"], "coverage": 0.7647058823529411, "emitted": 13, "accepted_accuracy": 0.6923076923076923, "correct_emissions_over_all": 0.5294117647058824, "accepted_output_is_candidate_mode": true, "scope": "paired_17", "task": "genus", "version": "current"}
METRIC {"n": 27, "correct": 11, "accuracy": 0.4074074074074074, "balanced_accuracy": 0.4, "macro_f1_support_only": 0.3812971342383107, "macro_f1_strict_union": 0.2723550958845076, "macro_f1_support_minus_strict": 0.10894203835380306, "predicted_taxa_outside_support": ["Aedes sierrensis", "Anopheles farauti"], "coverage": 0.9259259259259259, "emitted": 25, "accepted_accuracy": 0.44, "correct_emissions_over_all": 0.4074074074074074, "accepted_output_is_candidate_mode": false, "scope": "public_27", "task": "species", "version": "previous"}
METRIC {"n": 27, "correct": 25, "accuracy": 0.9259259259259259, "balanced_accuracy": 0.95, "macro_f1_support_only": 0.9532467532467532, "macro_f1_strict_union": 0.7943722943722943, "macro_f1_support_minus_strict": 0.15887445887445895, "predicted_taxa_outside_support": ["Aedes sierrensis"], "coverage": 0.8888888888888888, "emitted": 24, "accepted_accuracy": 0.9166666666666666, "correct_emissions_over_all": 0.8148148148148148, "accepted_output_is_candidate_mode": true, "scope": "public_27", "task": "species", "version": "current"}
METRIC {"n": 27, "correct": 19, "accuracy": 0.7037037037037037, "balanced_accuracy": 0.566951566951567, "macro_f1_support_only": 0.5245098039215687, "macro_f1_strict_union": 0.5245098039215687, "macro_f1_support_minus_strict": 0.0, "predicted_taxa_outside_support": [], "coverage": 0.9629629629629629, "emitted": 26, "accepted_accuracy": 0.6923076923076923, "correct_emissions_over_all": 0.6666666666666666, "accepted_output_is_candidate_mode": true, "scope": "public_27", "task": "genus", "version": "previous"}
METRIC {"n": 27, "correct": 25, "accuracy": 0.9259259259259259, "balanced_accuracy": 0.9259259259259259, "macro_f1_support_only": 0.9345238095238096, "macro_f1_strict_union": 0.9345238095238096, "macro_f1_support_minus_strict": 0.0, "predicted_taxa_outside_support": [], "coverage": 0.9629629629629629, "emitted": 26, "accepted_accuracy": 0.9230769230769231, "correct_emissions_over_all": 0.8888888888888888, "accepted_output_is_candidate_mode": true, "scope": "public_27", "task": "genus", "version": "current"}

```

## Métricas por gravação

```json
[
  {
    "n": 28,
    "correct": 12,
    "accuracy": 0.42857142857142855,
    "balanced_accuracy": 0.4111111111111111,
    "macro_f1_support_only": 0.39244264507422405,
    "macro_f1_strict_union": 0.28031617505301715,
    "macro_f1_support_minus_strict": 0.1121264700212069,
    "predicted_taxa_outside_support": [
      "Aedes sierrensis",
      "Anopheles farauti"
    ],
    "coverage": 0.9285714285714286,
    "emitted": 26,
    "accepted_accuracy": 0.46153846153846156,
    "correct_emissions_over_all": 0.42857142857142855,
    "accepted_output_is_candidate_mode": false,
    "scope": "received_28",
    "task": "species",
    "version": "previous"
  },
  {
    "n": 28,
    "correct": 26,
    "accuracy": 0.9285714285714286,
    "balanced_accuracy": 0.9555555555555555,
    "macro_f1_support_only": 0.9568181818181818,
    "macro_f1_strict_union": 0.7973484848484849,
    "macro_f1_support_minus_strict": 0.15946969696969693,
    "predicted_taxa_outside_support": [
      "Aedes sierrensis"
    ],
    "coverage": 0.8928571428571429,
    "emitted": 25,
    "accepted_accuracy": 0.92,
    "correct_emissions_over_all": 0.8214285714285714,
    "accepted_output_is_candidate_mode": true,
    "scope": "received_28",
    "task": "species",
    "version": "current"
  },
  {
    "n": 28,
    "correct": 20,
    "accuracy": 0.7142857142857143,
    "balanced_accuracy": 0.5687830687830688,
    "macro_f1_support_only": 0.5294117647058824,
    "macro_f1_strict_union": 0.5294117647058824,
    "macro_f1_support_minus_strict": 0.0,
    "predicted_taxa_outside_support": [],
    "coverage": 0.9642857142857143,
    "emitted": 27,
    "accepted_accuracy": 0.7037037037037037,
    "correct_emissions_over_all": 0.6785714285714286,
    "accepted_output_is_candidate_mode": true,
    "scope": "received_28",
    "task": "genus",
    "version": "previous"
  },
  {
    "n": 28,
    "correct": 26,
    "accuracy": 0.9285714285714286,
    "balanced_accuracy": 0.9259259259259259,
    "macro_f1_support_only": 0.9361111111111112,
    "macro_f1_strict_union": 0.9361111111111112,
    "macro_f1_support_minus_strict": 0.0,
    "predicted_taxa_outside_support": [],
    "coverage": 0.9642857142857143,
    "emitted": 27,
    "accepted_accuracy": 0.9259259259259259,
    "correct_emissions_over_all": 0.8928571428571429,
    "accepted_output_is_candidate_mode": true,
    "scope": "received_28",
    "task": "genus",
    "version": "current"
  },
  {
    "n": 17,
    "correct": 6,
    "accuracy": 0.35294117647058826,
    "balanced_accuracy": 0.3,
    "macro_f1_support_only": 0.3633333333333333,
    "macro_f1_strict_union": 0.22708333333333333,
    "macro_f1_support_minus_strict": 0.13624999999999995,
    "predicted_taxa_outside_support": [
      "Aedes sierrensis",
      "Anopheles dirus",
      "NO_CANDIDATE"
    ],
    "coverage": 0.8235294117647058,
    "emitted": 14,
    "accepted_accuracy": 0.42857142857142855,
    "correct_emissions_over_all": 0.35294117647058826,
    "accepted_output_is_candidate_mode": true,
    "scope": "paired_17",
    "task": "species",
    "version": "previous"
  },
  {
    "n": 17,
    "correct": 6,
    "accuracy": 0.35294117647058826,
    "balanced_accuracy": 0.3,
    "macro_f1_support_only": 0.3822222222222222,
    "macro_f1_strict_union": 0.273015873015873,
    "macro_f1_support_minus_strict": 0.1092063492063492,
    "predicted_taxa_outside_support": [
      "Aedes sierrensis",
      "NO_CANDIDATE"
    ],
    "coverage": 0.7058823529411765,
    "emitted": 12,
    "accepted_accuracy": 0.5,
    "correct_emissions_over_all": 0.35294117647058826,
    "accepted_output_is_candidate_mode": true,
    "scope": "paired_17",
    "task": "species",
    "version": "current"
  },
  {
    "n": 17,
    "correct": 7,
    "accuracy": 0.4117647058823529,
    "balanced_accuracy": 0.4166666666666667,
    "macro_f1_support_only": 0.31851851851851853,
    "macro_f1_strict_union": 0.2388888888888889,
    "macro_f1_support_minus_strict": 0.07962962962962963,
    "predicted_taxa_outside_support": [
      "NO_CANDIDATE"
    ],
    "coverage": 0.8235294117647058,
    "emitted": 14,
    "accepted_accuracy": 0.5,
    "correct_emissions_over_all": 0.4117647058823529,
    "accepted_output_is_candidate_mode": true,
    "scope": "paired_17",
    "task": "genus",
    "version": "previous"
  },
  {
    "n": 17,
    "correct": 10,
    "accuracy": 0.5882352941176471,
    "balanced_accuracy": 0.625,
    "macro_f1_support_only": 0.6262626262626262,
    "macro_f1_strict_union": 0.4696969696969696,
    "macro_f1_support_minus_strict": 0.15656565656565657,
    "predicted_taxa_outside_support": [
      "NO_CANDIDATE"
    ],
    "coverage": 0.7647058823529411,
    "emitted": 13,
    "accepted_accuracy": 0.6923076923076923,
    "correct_emissions_over_all": 0.5294117647058824,
    "accepted_output_is_candidate_mode": true,
    "scope": "paired_17",
    "task": "genus",
    "version": "current"
  },
  {
    "n": 27,
    "correct": 11,
    "accuracy": 0.4074074074074074,
    "balanced_accuracy": 0.4,
    "macro_f1_support_only": 0.3812971342383107,
    "macro_f1_strict_union": 0.2723550958845076,
    "macro_f1_support_minus_strict": 0.10894203835380306,
    "predicted_taxa_outside_support": [
      "Aedes sierrensis",
      "Anopheles farauti"
    ],
    "coverage": 0.9259259259259259,
    "emitted": 25,
    "accepted_accuracy": 0.44,
    "correct_emissions_over_all": 0.4074074074074074,
    "accepted_output_is_candidate_mode": false,
    "scope": "public_27",
    "task": "species",
    "version": "previous"
  },
  {
    "n": 27,
    "correct": 25,
    "accuracy": 0.9259259259259259,
    "balanced_accuracy": 0.95,
    "macro_f1_support_only": 0.9532467532467532,
    "macro_f1_strict_union": 0.7943722943722943,
    "macro_f1_support_minus_strict": 0.15887445887445895,
    "predicted_taxa_outside_support": [
      "Aedes sierrensis"
    ],
    "coverage": 0.8888888888888888,
    "emitted": 24,
    "accepted_accuracy": 0.9166666666666666,
    "correct_emissions_over_all": 0.8148148148148148,
    "accepted_output_is_candidate_mode": true,
    "scope": "public_27",
    "task": "species",
    "version": "current"
  },
  {
    "n": 27,
    "correct": 19,
    "accuracy": 0.7037037037037037,
    "balanced_accuracy": 0.566951566951567,
    "macro_f1_support_only": 0.5245098039215687,
    "macro_f1_strict_union": 0.5245098039215687,
    "macro_f1_support_minus_strict": 0.0,
    "predicted_taxa_outside_support": [],
    "coverage": 0.9629629629629629,
    "emitted": 26,
    "accepted_accuracy": 0.6923076923076923,
    "correct_emissions_over_all": 0.6666666666666666,
    "accepted_output_is_candidate_mode": true,
    "scope": "public_27",
    "task": "genus",
    "version": "previous"
  },
  {
    "n": 27,
    "correct": 25,
    "accuracy": 0.9259259259259259,
    "balanced_accuracy": 0.9259259259259259,
    "macro_f1_support_only": 0.9345238095238096,
    "macro_f1_strict_union": 0.9345238095238096,
    "macro_f1_support_minus_strict": 0.0,
    "predicted_taxa_outside_support": [],
    "coverage": 0.9629629629629629,
    "emitted": 26,
    "accepted_accuracy": 0.9230769230769231,
    "correct_emissions_over_all": 0.8888888888888888,
    "accepted_output_is_candidate_mode": true,
    "scope": "public_27",
    "task": "genus",
    "version": "current"
  }
]

```

## Verificação do histórico

```json
{
  "checked_utc": "2026-10-14T16:26:21.545565+00:00",
  "verification": {
    "v1_freeze": {
      "ok": true,
      "files": null
    },
    "v2_freeze": {
      "ok": true,
      "files": null
    },
    "v3_freeze": {
      "ok": true,
      "files": null
    },
    "v4_freeze": {
      "ok": true,
      "files": null
    },
    "v2_history": {
      "ok": true,
      "files": 940
    },
    "v3_history": {
      "ok": true,
      "files": 1210
    },
    "v4_history": {
      "ok": true,
      "files": 1239
    }
  },
  "baseline_files": 1633,
  "unchanged_baseline_files": 1631,
  "authorized_active_source_updates": [
    "scripts/predict_audio.py",
    "scripts/robust_audio_features.py"
  ],
  "historical_completion_manifests_unchanged": true,
  "snapshots_rewritten": false
}

```

## Contraexemplo indutivo

```json
{
  "tested_utc": "2026-10-14T16:23:17.066267+00:00",
  "altered_other_rows_only": true,
  "altered_indices": [
    31,
    46,
    32,
    58
  ],
  "low_second_row_value": 0.0,
  "high_second_row_value": 4.0,
  "controlled_feature_intervention_not_physical_new_audio": true,
  "found_counterexample": true,
  "probes": [
    {
      "window_index": 51,
      "start_s": 25.5,
      "same_first_feature_row": true,
      "same_pcm_for_first_window": true,
      "predictions_bit_identical": false,
      "maximum_probability_difference": 0.5788549780845642,
      "low_context_candidate": "Aedes aegypti",
      "high_context_candidate": "Aedes albopictus",
      "first_row_sha256": "f1105a6f71da7d92d8958aa0294e11c722682908970100654f8f0ae297922d51",
      "low_probabilities": [
        0.6125667095184326,
        0.06684348732233047,
        0.004614575300365686,
        0.05469413474202156,
        0.026987744495272636,
        0.018142569810152054,
        0.003861534409224987,
        0.021753860637545586,
        0.029415931552648544,
        0.01116840448230505,
        0.03985796123743057,
        0.004428056534379721,
        0.03678864985704422,
        0.0036706048995256424,
        0.0015682269586250186,
        0.03614998981356621,
        0.00964423455297947,
        0.013584590516984463,
        0.004101286176592112,
        0.0001574680645717308
      ],
      "high_probabilities": [
        0.09235470741987228,
        0.6456984877586365,
        0.0037704661954194307,
        0.0446893535554409,
        0.02205108292400837,
        0.01482388749718666,
        0.0031551735009998083,
        0.017774593085050583,
        0.024035099893808365,
        0.009125453419983387,
        0.03256705030798912,
        0.0036180661991238594,
        0.03005918301641941,
        0.0029991690535098314,
        0.0012813631910830736,
        0.029537349939346313,
        0.007880088873207569,
        0.011099664494395256,
        0.00335106928832829,
        0.00012866363977082074
      ]
    }
  ]
}

```
