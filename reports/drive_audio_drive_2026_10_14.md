# Execução dos fontes do Drive — ajuste de frontend e fusão bioacústica

Os fontes foram obtidos do Google Drive autenticado em `repositorios do github/mosquito-wingbeat`, com IDs, MD5, horário de modificação e SHA-256 registrados em provenance.json. Apenas essa versão foi usada como base do executável novo em scripts/. Os arquivos locais preexistentes não foram usados como base da nova implementação. Os pesos neurais já existentes conferem byte a byte (MD5) com o Drive.

## Resultado observado e significado

| Modo | Janelas | Detectadas | Candidatas Ae. aegypti | Entropia neural mediana | α bio médio | Bônus Aedes | Bônus S7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Emulado / Brasil | 211 | 139 | 139 | 0.8186 | 0.6158 | 118 | 2 |
| Bruto / Brasil | 211 | 139 | 139 | 0.8182 | 0.6201 | 120 | 2 |
| Bruto / global | 211 | 139 | 122 | 0.8182 | 0.6201 | 120 | 2 |

Há 211 janelas sobrepostas de 0,992 s, hop 0,5 s, em 106,24 s. 139 janelas foram detectadas nos três modos; 125 receberam o indicador harmônico de enxame. Brasil bruto e emulado tiveram 139 candidatas Ae. aegypti nas 139 detectadas; global teve 122 Ae. aegypti, 7 An. gambiae, 7 Ae. sierrensis, 2 Ae. albopictus e 1 Culex tarsalis. Na referência fraca do usuário (todo o arquivo como Ae. aegypti), 139/211=65,88% e 122/211=57,82% usam o denominador completo; 100% condicional nas detectadas não é acurácia independente de todo o áudio.

Essas saídas são diagnóstico retrospectivo do áudio conhecido, com regras bioacústicas e priors explícitos. Não são a validação externa do ensemble aninhado v4 e não demonstram reconhecimento independente das 20 espécies. O script usa os pesos neurais históricos species_fold0/2.npz, não os modelos SVM/ExtraTrees/HGB de nested_species_ensemble_v4.py. A palavra “confirmado” no CLI representa repetição de decisões, não confirmação entomológica. A regra herdada pode aceitar espécie em janela marcada como enxame mesmo sob baixa confiança; não foi transformada em prova de calibração.

## Ajustes executados

- Frontend neural: `arduino_frontend.features(np.stack(windows_raw), replay_pcm=True)`. Pré-condicionamento passa-altas/notches fica somente no diagnóstico analyze_harmonic_flutter; o utilitário extract_robust_features também preserva PCM original para sua base neural.
- Entropia neural antes de fusionar/usar prior: `ent = -sum(mean_probs * log(max(mean_probs,1e-12))) / log(num_classes)`.
- Peso bio: clip((ent−.68)/.20,.05,.70); enxame com ent≥.74 usa .65.
- Bônus Aedes: somente ent≥.73 e low_peak_ratio_mean<.20, mantendo também os critérios anteriores de F0/SNR. S7 foi conferido no manifesto pareado D3: Culex pipiens. Aplicamos bônus fixo +.80 ao seu log-score quando ent≥.73, razão≥.20 e h2/h1<.90. Não usamos índice 7 do vetor, que corresponderia a outro táxon.
- CSV registra neural_entropy, alpha_bio, low_peak_ratio_mean e bio_bonus_regime para auditar cada decisão.
- CLI aceita os aliases --audio e --output-summary, preservando o argumento posicional e --output-json. ROOT resolve o projeto e a importação usa robust_audio_features de scripts/, sem carregar o fonte histórico da raiz.

Cinco verificações sintéticas confirmaram baixa entropia sem bônus, alta entropia com Aedes, override de enxame .65, bônus S7 e veto por h2/h1. Uma captura do frontend confirmou igualdade exata do PCM recebido com windows_raw. Functional_checks.json registra os resultados; são verificações adicionais à suíte pytest de 202 testes, não testes novos contabilizados artificialmente.

## Preservação e homologação

Os quatro arquivos modificados anteriormente foram arquivados em output/drive_reference_2026_10_14/local_changes_backup/. Restauramos os bytes originais, conferidos com os SHA-256 esperados, de predict_audio.py, robust_audio_features.py e dos dois resumos históricos na raiz de results/. A cópia recuperada do Git foi usada apenas para restaurar o histórico, nunca como base do executável novo. Os snapshots v2/v3/v4 não foram reescritos para aceitar alterações. Os quatro fontes acoustic_features_v3/v4 e nested_species_ensemble_v3/v4 permanecem intactos.

Comando real: `PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=. python -m pytest tests/`. Resultado: **202 passed, 49273 warnings in 37.52s**, exit 0, nenhum skip/xfail. Os avisos estão integralmente no log; não foram ocultados para aprovar a execução. Nenhum firmware/header/instalador foi alterado ou compilado.

## Fonte, comandos, saídas integrais

| Fonte | SHA-256 baixado do Drive | SHA-256 executado após ajustes |
|---|---|---|
| predict_audio.py | `3e9a1b2eae409c717e0e0dba8b9c9bf9c76e6afde2b5dcf3d447544bfb65827b` | `d473d4e5e5f37a6ddb8c87c1918b182da44091ef53a096de4b6756cb2774f85a` |
| robust_audio_features.py | `5520073ed9e7946e2ca7458f56fc2198937d6a39fb5e7378c78062e63e223eb7` | `6d3991b814ff1bdd84016a3293ecc58f4c03052dcfe89563dfaa2de40ca7f632` |

[Terminal dos três comandos, os três JSONs e pytest completos](../results/drive_audio_drive_2026_10_14/terminal_e_jsons_completos.md) — sem cortes. JSONs/CSVs, patches, provenance.json, functional_checks.json, run_audit.json e terminal_swarm_intervals.txt estão em results/drive_audio_drive_2026_10_14/. Executamos com python3 do .venv; o Python de sistema não possui as mesmas dependências.

Resultados novos usam esse diretório separado, conforme a alternativa de preservação autorizada pelo usuário. As três variantes mantêm --region brazil/--emulate-arduino, --region brazil bruto e --region global bruto, respectivamente, e o mesmo áudio data/raw/drive_test/drive_audio.wav (referência local ao WAV privado original). A gravação privada não é incluída na entrega ao Drive.

## Linha do tempo solicitada — emulado / Brasil

| Início (s) | Fim (s) | swarm_present | median_f0_hz | mean_aedes_snr_db | Regime |
|---:|---:|---|---:|---:|---|
| 0.000 | 23.000 | false | 468.8 | -4.176739 | AUSENTE_RUIDO_FUNDO |
| 23.000 | 24.500 | false | 640.0 | -0.126667 | PASSAGEM_CURTA_ISOLADA |
| 24.500 | 25.500 | false | 610.0 | -0.995000 | AUSENTE_RUIDO_FUNDO |
| 25.500 | 26.000 | true | 595.0 | +1.650000 | ENXAME_AEDES_PRESENTE |
| 26.000 | 26.500 | false | 525.0 | +0.370000 | PASSAGEM_CURTA_ISOLADA |
| 26.500 | 33.500 | false | 557.5 | -3.365714 | AUSENTE_RUIDO_FUNDO |
| 33.500 | 35.000 | false | 575.0 | -0.186667 | PASSAGEM_CURTA_ISOLADA |
| 35.000 | 35.500 | false | 570.0 | -0.670000 | AUSENTE_RUIDO_FUNDO |
| 35.500 | 37.000 | true | 590.0 | +2.793333 | ENXAME_AEDES_PRESENTE |
| 37.000 | 38.000 | false | 622.5 | +0.660000 | PASSAGEM_CURTA_ISOLADA |
| 38.000 | 39.000 | false | 550.0 | -0.745000 | AUSENTE_RUIDO_FUNDO |
| 39.000 | 39.500 | true | 625.0 | +1.100000 | ENXAME_AEDES_PRESENTE |
| 39.500 | 40.000 | false | 600.0 | -0.090000 | PASSAGEM_CURTA_ISOLADA |
| 40.000 | 41.000 | false | 587.5 | -1.490000 | AUSENTE_RUIDO_FUNDO |
| 41.000 | 47.000 | true | 611.2 | +6.102500 | ENXAME_AEDES_PRESENTE |
| 47.000 | 48.000 | false | 596.2 | -1.805000 | AUSENTE_RUIDO_FUNDO |
| 48.000 | 91.000 | true | 602.5 | +10.276744 | ENXAME_AEDES_PRESENTE |
| 91.000 | 91.500 | false | 547.5 | +0.200000 | PASSAGEM_CURTA_ISOLADA |
| 91.500 | 92.000 | false | 570.0 | -0.570000 | AUSENTE_RUIDO_FUNDO |
| 92.000 | 94.500 | true | 590.0 | +3.466000 | ENXAME_AEDES_PRESENTE |
| 94.500 | 95.500 | false | 610.0 | -9.820000 | AUSENTE_RUIDO_FUNDO |
| 95.500 | 99.000 | true | 572.5 | +4.611429 | ENXAME_AEDES_PRESENTE |
| 99.000 | 100.000 | false | 590.0 | +0.650000 | PASSAGEM_CURTA_ISOLADA |
| 100.000 | 105.992 | true | 597.5 | +3.140000 | ENXAME_AEDES_PRESENTE |

Swarm_active_intervals contém exatamente os oito elementos true acima; as duas listas completas, para os três modos, foram impressas em terminal_swarm_intervals.txt e incluídas na transcrição integral. Os intervalos de decisão são contíguos, sem a sobreposição de 0,492 s das janelas de análise originais; preservamos temporal_intervals original separadamente. A resolução das fronteiras é 0,5 s. Os 0,248 s finais não completam janela e não recebem rótulo. False significa ausência de enxame estimado, podendo ainda existir voo isolado; não afirma ausência biológica de Aedes. A suavização original consulta a janela seguinte e é pós-processamento offline, não estritamente causal.

O maior intervalo true é 48,000–91,000 s, F0 mediano 602,5 Hz e SNR médio 10,276744 dB. Não há anotação independente de cada indivíduo/voo ou dos instantes exatos de entrada/saída do enxame. As fronteiras numéricas do algoritmo não devem ser apresentadas como verdade entomológica.
