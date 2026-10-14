# Saída integral — versão do Drive com ajustes solicitados
PATH="$PWD/.venv/bin:$PATH"; PYTHONPATH=.
O interpretador python3 é o do .venv. Resultados novos em diretório separado para preservar snapshots.

## emulated_brazil
```bash
python3 scripts/predict_audio.py --audio data/raw/drive_test/drive_audio.wav --region brazil --emulate-arduino --output-csv results/drive_audio_drive_2026_10_14/drive_audio_predictions.csv --output-summary results/drive_audio_drive_2026_10_14/drive_audio_summary.json
```
### stdout/stderr completo
```text

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
  * Aedes         : 138 janelas ( 99.3%)
  * Culex         :   1 janelas (  0.7%)

--- TIER 3: ESPÉCIE ESPECÍFICA (FUSÃO HARMÔNICA + NEURAL) ---
Espécie Confirmada:    Aedes aegypti
  * Aedes aegypti             : 139 janelas (100.0%)

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

-> Relatório detalhado salvo em CSV: results/drive_audio_drive_2026_10_14/drive_audio_predictions.csv
-> Resumo analítico salvo em JSON: results/drive_audio_drive_2026_10_14/drive_audio_summary.json

```
### JSON completo
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
    "Aedes": 138,
    "Culex": 1
  },
  "species_distribution": {
    "Aedes aegypti": 139
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
  "timeline_resolution_s": 0.5,
  "analysis_window_s": 0.992,
  "unanalysed_tail_s": 0.248,
  "timeline_interpretation": "Estimated offline classifier decisions, not entomological ground truth; interval smoothing uses the next window."
}
```

## raw_brazil
```bash
python3 scripts/predict_audio.py --audio data/raw/drive_test/drive_audio.wav --region brazil --output-csv results/drive_audio_drive_2026_10_14/drive_audio_raw_predictions.csv --output-summary results/drive_audio_drive_2026_10_14/drive_audio_raw_summary.json
```
### stdout/stderr completo
```text

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
  * Aedes         : 138 janelas ( 99.3%)
  * Culex         :   1 janelas (  0.7%)

--- TIER 3: ESPÉCIE ESPECÍFICA (FUSÃO HARMÔNICA + NEURAL) ---
Espécie Confirmada:    Aedes aegypti
  * Aedes aegypti             : 139 janelas (100.0%)

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

-> Relatório detalhado salvo em CSV: results/drive_audio_drive_2026_10_14/drive_audio_raw_predictions.csv
-> Resumo analítico salvo em JSON: results/drive_audio_drive_2026_10_14/drive_audio_raw_summary.json

```
### JSON completo
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
    "Aedes": 138,
    "Culex": 1
  },
  "species_distribution": {
    "Aedes aegypti": 139
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
  "timeline_resolution_s": 0.5,
  "analysis_window_s": 0.992,
  "unanalysed_tail_s": 0.248,
  "timeline_interpretation": "Estimated offline classifier decisions, not entomological ground truth; interval smoothing uses the next window."
}
```

## raw_global
```bash
python3 scripts/predict_audio.py --audio data/raw/drive_test/drive_audio.wav --region global --output-csv results/drive_audio_drive_2026_10_14/drive_audio_global_predictions.csv --output-summary results/drive_audio_drive_2026_10_14/drive_audio_global_summary.json
```
### stdout/stderr completo
```text

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
  * Aedes         : 137 janelas ( 98.6%)
  * Culex         :   2 janelas (  1.4%)

--- TIER 3: ESPÉCIE ESPECÍFICA (FUSÃO HARMÔNICA + NEURAL) ---
Espécie Confirmada:    Aedes aegypti
  * Aedes aegypti             : 122 janelas ( 87.8%)
  * Anopheles gambiae         :   7 janelas (  5.0%)
  * Aedes sierrensis          :   7 janelas (  5.0%)
  * Aedes albopictus          :   2 janelas (  1.4%)
  * Culex tarsalis            :   1 janelas (  0.7%)

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
 35.5 -  37.5s   |   2.0s   | ENXAME_AEDES_PRESENTE        | 590.0Hz  |  +2.5 dB   | Aedes sierrensis
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

-> Relatório detalhado salvo em CSV: results/drive_audio_drive_2026_10_14/drive_audio_global_predictions.csv
-> Resumo analítico salvo em JSON: results/drive_audio_drive_2026_10_14/drive_audio_global_summary.json

```
### JSON completo
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
    "Aedes": 137,
    "Culex": 2
  },
  "species_distribution": {
    "Aedes aegypti": 122,
    "Anopheles gambiae": 7,
    "Aedes sierrensis": 7,
    "Aedes albopictus": 2,
    "Culex tarsalis": 1
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
      "dominant_species": "Aedes sierrensis"
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
  "timeline_resolution_s": 0.5,
  "analysis_window_s": 0.992,
  "unanalysed_tail_s": 0.248,
  "timeline_interpretation": "Estimated offline classifier decisions, not entomological ground truth; interval smoothing uses the next window."
}
```

## Listas swarm_timeline e swarm_active_intervals completas
```text
drive_audio_summary.json
swarm_timeline:
[
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
]
swarm_active_intervals:
[
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
]
drive_audio_raw_summary.json
swarm_timeline:
[
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
]
swarm_active_intervals:
[
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
]
drive_audio_global_summary.json
swarm_timeline:
[
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
]
swarm_active_intervals:
[
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
]

```

## pytest completo
```text
============================= test session starts ==============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat
plugins: platformdirs-4.12.2, anyio-4.13.0
collected 202 items

tests/test_aedes_binary.py ...                                           [  1%]
tests/test_aedes_classical.py ...                                        [  2%]
tests/test_audio_preprocessing.py ....                                   [  4%]
tests/test_corrected_validation.py .......                               [  8%]
tests/test_cross_domain_strategies.py ............                       [ 14%]
tests/test_expanded_protocol.py ...                                      [ 15%]
tests/test_faithful_reproduction.py ........                             [ 19%]
tests/test_install_arduino.py .......................                    [ 31%]
tests/test_large_audio_model.py ...                                      [ 32%]
tests/test_multisource_protocol.py ....                                  [ 34%]
tests/test_nested_species.py .............                               [ 41%]
tests/test_nested_species_v2.py ....................                     [ 50%]
tests/test_nested_species_v3.py .....................                    [ 61%]
tests/test_nested_species_v4.py .........................                [ 73%]
tests/test_reliability.py ...........                                    [ 79%]
tests/test_reproduction.py ...........                                   [ 84%]
tests/test_research_audit.py ............                                [ 90%]
tests/test_sex_balanced_reliability.py ......                            [ 93%]
tests/test_windows_installer.py .............                            [100%]

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
===================== 202 passed, 49273 warnings in 37.52s =====================

```
