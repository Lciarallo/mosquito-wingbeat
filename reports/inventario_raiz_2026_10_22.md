# Inventário da raiz do diretório de trabalho (22/10/2026)

Gerado automaticamente em 22/10/2026; **nada foi movido nem apagado**. A movimentação depende de confirmação do autor. 224 itens; 25 versionados.

- *Usado pelo pipeline atual*: módulo importado, direta ou transitivamente, por algum `.py` versionado (raiz, `reanalysis/`, `tests/`), ou item versionado.
- *Citado em manifesto de hash*: o nome aparece em JSON com `sha256` — `tracked` (versionado) ou `results` (rodadas locais). Mover o item quebra a verificação dessas rodadas.
- *Fase/versão*: inferida do nome (`_vN`, data, backup).

| Item | Tipo | Fase/versão | Versionado | Usado pelo pipeline atual | Citado em manifesto de hash | Importado por (local) |
|---|---|---|---|---|---|---|
| `.claude` | dir | — | não | não | — | — |
| `.gitattributes` | gitattributes | — | não | não | results | — |
| `.github` | dir | — | não | não | results | — |
| `.gitignore` | gitignore | — | sim | sim | results | — |
| `.gitmodules` | gitmodules | — | não | não | — | — |
| `.pytest_cache` | dir | — | não | não | — | — |
| `.venv` | dir | — | não | não | results | — |
| `.venv-v3` | dir | — | não | não | — | — |
| `DRIVE_REVIEW_2026_10_14.md` | md | 2026-10-14 | não | não | — | — |
| `GITHUB_PUBLICATION_2026_10_14.md` | md | 2026-10-14 | não | não | — | — |
| `MODEL_VALIDATION.md` | md | — | não | não | results | — |
| `MODEL_VALIDATION_v4.md` | md | v4 | não | não | results | — |
| `README.md` | md | — | sim | sim | results | — |
| `README_TORRES_2025.md` | md | — | não | não | — | — |
| `README_v4.md` | md | v4 | não | não | results | — |
| `__pycache__` | dir | — | não | não | — | — |
| `acoustic_features_v2.py` | py | v2 | não | não | results | nested_species_ensemble_v2.py, test_nested_species_v2.py |
| `acoustic_features_v3.py` | py | v3 | não | não | results | nested_species_ensemble_v3.py, test_nested_species_v3.py |
| `acoustic_features_v4.py` | py | v4 | não | não | results | analyze_notebook_transfer.py, nested_species_ensemble_v4.py, reproduce_notebook_transfer.py … |
| `acoustic_features_v5.py` | py | v5 | não | não | results | nested_species_ensemble_v5.py, test_nested_species_v5.py |
| `acquire_notebook_transfer_check.py` | py | — | não | não | — | — |
| `aedes_binary.py` | py | — | não | não | results | analyze_frequency_shift.py, benchmark_all_methods.py, compare_reduced_databases.py … |
| `aedes_binary_notebook.py` | py | — | não | não | results | build_notebook.py |
| `aedes_classical.py` | py | — | sim | sim (importado) | results | benchmark_all_methods.py, compare_reduced_databases.py, corrected_validation.py … |
| `aedes_classical_notebook.py` | py | — | não | não | results | build_notebook.py |
| `analyze_frequency_shift.py` | py | — | não | não | results | — |
| `analyze_notebook_transfer.py` | py | — | não | não | — | — |
| `apply_torres_2025.py` | py | — | não | não | results | evaluate_species_preprocessing.py, predict_torres_2025.py |
| `arduino_frontend.py` | py | — | não | não | results | cross_validate_drive_audio.py, experiment_mixed_datasets.py, predict_audio.py … |
| `arduino_models.py` | py | — | não | não | results | train_arduino.py, train_expanded.py |
| `arduino_species_models.py` | py | — | não | não | results | train_arduino_species.py, train_expanded.py, verify_arduino_species.py |
| `audio_io.py` | py | — | não | não | — | predict_audio.py, predict_audio_v1_backup.py |
| `audio_preprocessing.py` | py | — | não | não | results | evaluate_preprocessing.py, test_audio_preprocessing.py |
| `audit_author_mosquitosong.py` | py | — | não | não | results | — |
| `audit_git_species.py` | py | — | não | não | results | — |
| `audit_research.py` | py | — | não | não | results | — |
| `benchmark_all_methods.py` | py | — | não | não | results | test_corrected_validation.py |
| `bioacoustic_invariance.py` | py | — | sim | sim (importado) | results | benchmark_all_methods.py, compare_reduced_databases.py, corrected_validation.py … |
| `build_faithful_reproduction_artifacts.py` | py | — | não | não | results | — |
| `build_notebook.py` | py | — | não | não | results | — |
| `build_reliability_artifacts.py` | py | — | não | não | results | — |
| `build_report.py` | py | — | não | não | results | — |
| `comparacao_bases_reduzidas_2026_10_02.html` | html | 2026-10-02 | não | não | — | — |
| `comparacao_bases_reduzidas_2026_10_02.ipynb` | ipynb | 2026-10-02 | não | não | — | — |
| `comparacao_frequencias_2026_10_03.html` | html | 2026-10-03 | não | não | — | — |
| `comparacao_frequencias_2026_10_03.ipynb` | ipynb | 2026-10-03 | não | não | — | — |
| `compare_detection_levels.py` | py | — | não | não | results | — |
| `compare_detection_levels_v2.py` | py | v2 | não | não | results | — |
| `compare_reduced_databases.py` | py | — | não | não | results | summarize_reduced_databases.py, test_reduced_databases_unseen.py |
| `compare_species_v3.py` | py | v3 | não | não | results | compare_species_v4.py, nested_species_ensemble_v3.py |
| `compare_species_v4.py` | py | v4 | não | não | results | compare_species_v5.py, nested_species_ensemble_v4.py |
| `compare_species_v5.py` | py | v5 | não | não | results | nested_species_ensemble_v5.py |
| `confiabilidade_binaria_aedes_2026_10_01.html` | html | 2026-10-01 | não | não | — | — |
| `confiabilidade_binaria_aedes_2026_10_01.ipynb` | ipynb | 2026-10-01 | não | não | — | — |
| `confiabilidade_binaria_aedes_adaptacao_2026_10_01.html` | html | 2026-10-01 | não | não | — | — |
| `confiabilidade_binaria_aedes_adaptacao_2026_10_01.ipynb` | ipynb | 2026-10-01 | não | não | — | — |
| `corrected_validation.py` | py | — | não | não | results | benchmark_all_methods.py, compare_reduced_databases.py, evaluate_biodcase_d5.py … |
| `cross_domain_strategy_models.py` | py | — | não | não | results | evaluate_cross_domain_strategies.py, test_cross_domain_strategies.py, train_cross_domain_strategies.py |
| `cross_validate_drive_audio.py` | py | — | não | não | results | — |
| `data` | dir | — | sim | sim | — | — |
| `delivery_manifest.json` | json | — | não | não | — | — |
| `dist` | dir | — | não | não | — | — |
| `estrategias_contrastivas_enxames_2026_10_03.html` | html | 2026-10-03 | não | não | — | — |
| `estrategias_contrastivas_enxames_2026_10_03.ipynb` | ipynb | 2026-10-03 | não | não | — | — |
| `evaluate_aedes_binary.py` | py | — | não | não | results | — |
| `evaluate_aedes_classical.py` | py | — | não | não | results | — |
| `evaluate_author_mosquitosong_external.py` | py | — | não | não | results | — |
| `evaluate_biodcase_d5.py` | py | — | não | não | results | — |
| `evaluate_corrected_fresh.py` | py | — | não | não | results | — |
| `evaluate_cross_domain_strategies.py` | py | — | não | não | results | test_cross_domain_strategies.py |
| `evaluate_hard_negative_reproduction.py` | py | — | não | não | results | evaluate_reliability.py, evaluate_sex_balanced_reliability.py, test_reproduction.py |
| `evaluate_multisource.py` | py | — | não | não | results | — |
| `evaluate_nested_genus.py` | py | — | não | não | results | — |
| `evaluate_nested_species.py` | py | — | não | não | results | compare_detection_levels.py, evaluate_nested_genus.py, fetch_species_v2_fresh.py … |
| `evaluate_new_approaches.py` | py | — | não | não | results | — |
| `evaluate_new_database.py` | py | — | não | não | results | corrected_validation.py |
| `evaluate_preprocessing.py` | py | — | não | não | results | — |
| `evaluate_reliability.py` | py | — | não | não | results | summarize_reliability.py, test_reliability.py |
| `evaluate_reproduction.py` | py | — | não | não | results | evaluate_hard_negative_reproduction.py, fetch_reproduction_fresh.py, robust_reproduction_ensemble.py … |
| `evaluate_sex_balanced_reliability.py` | py | — | não | não | results | evaluate_cross_domain_strategies.py, summarize_sex_balanced_reliability.py, test_sex_balanced_reliability.py |
| `evaluate_species_preprocessing.py` | py | — | não | não | results | — |
| `evaluate_ternary_event_pipeline.py` | py | — | sim | sim (importado) | results | test_ternary_detector.py, test_ternary_verified.py |
| `evaluate_ternary_source_transfer.py` | py | — | não | não | — | test_drive_exclusion.py |
| `evaluate_ternary_synthetic.py` | py | — | não | não | results | — |
| `evaluate_torres_aedes_datasets.py` | py | — | não | não | — | — |
| `execute_notebook.py` | py | — | não | não | results | — |
| `expanded_notebook.py` | py | — | não | não | results | build_notebook.py |
| `experiment_mixed_datasets.py` | py | — | não | não | results | — |
| `explain_aedes_classical.py` | py | — | não | não | results | — |
| `external` | dir | — | não | não | results | — |
| `fetch_additional_audio.py` | py | — | não | não | results | — |
| `fetch_biodcase.py` | py | — | não | não | results | — |
| `fetch_corrected_fresh.py` | py | — | não | não | results | — |
| `fetch_cross_domain_strategy_test.py` | py | — | não | não | results | — |
| `fetch_data.py` | py | — | não | não | results | notebook_code.py |
| `fetch_faithful_dr_biol.py` | py | — | não | não | results | — |
| `fetch_fresh_aedes_binary.py` | py | — | não | não | results | — |
| `fetch_fresh_aedes_classical.py` | py | — | não | não | results | — |
| `fetch_fresh_multisource_test.py` | py | — | não | não | results | — |
| `fetch_hard_negative_final_test.py` | py | — | não | não | results | — |
| `fetch_humbug.py` | py | — | não | não | results | acquire_notebook_transfer_check.py, evaluate_nested_species.py, fetch_species_v2_fresh.py … |
| `fetch_reliability_data.py` | py | — | não | não | results | evaluate_reliability.py, fetch_reliability_fresh.py, fetch_sex_balanced_reliability_fresh.py … |
| `fetch_reliability_fresh.py` | py | — | não | não | results | — |
| `fetch_reproduction_fresh.py` | py | — | não | não | results | — |
| `fetch_sex_balanced_reliability_fresh.py` | py | — | não | não | results | — |
| `fetch_species_v2_fresh.py` | py | v2 | não | não | results | — |
| `fetch_species_v2_fresh_checked.py` | py | v2 | não | não | results | — |
| `fetch_species_v3_fresh.py` | py | v3 | não | não | results | nested_species_ensemble_v3.py |
| `fetch_species_v4_fresh.py` | py | v4 | não | não | results | nested_species_ensemble_v4.py |
| `fetch_species_v5_fresh.py` | py | v5 | não | não | results | nested_species_ensemble_v5.py |
| `fetch_ternary_fresh.py` | py | — | não | não | results | — |
| `fetch_torres_2025.py` | py | — | não | não | results | — |
| `firmware` | dir | — | não | não | results | — |
| `improve_device.py` | py | — | não | não | results | build_notebook.py, compare_detection_levels_v2.py, compare_species_v3.py … |
| `large_audio_model.py` | py | — | não | não | results | aedes_binary.py, evaluate_multisource.py, evaluate_preprocessing.py … |
| `large_cnn_notebook.py` | py | — | não | não | results | build_notebook.py |
| `leakage_auditor.py` | py | — | não | não | results | audit_git_species.py, train_hierarchical_models.py |
| `literature` | dir | — | não | não | — | — |
| `loso_generalization.py` | py | — | não | não | results | audit_research.py, test_research_audit.py |
| `models_committee.py` | py | — | não | não | results | benchmark_all_methods.py, test_corrected_validation.py |
| `models_deep.py` | py | — | não | não | results | corrected_validation.py, evaluate_biodcase_d5.py, evaluate_new_database.py … |
| `models_trees.py` | py | — | não | não | results | models_committee.py |
| `mosquito_wingbeat_estudo.html` | html | — | não | não | — | — |
| `mosquito_wingbeat_estudo.ipynb` | ipynb | — | não | não | — | — |
| `multisource_notebook.py` | py | — | não | não | results | build_notebook.py |
| `nested_species_ensemble.py` | py | — | não | não | results | compare_detection_levels_v2.py, compare_species_v3.py, compare_species_v4.py … |
| `nested_species_ensemble_v2.py` | py | v2 | não | não | results | acquire_notebook_transfer_check.py, compare_detection_levels_v2.py, compare_species_v4.py … |
| `nested_species_ensemble_v3.py` | py | v3 | não | não | results | compare_species_v3.py, compare_species_v4.py, fetch_species_v3_fresh.py … |
| `nested_species_ensemble_v4.py` | py | v4 | não | não | results | compare_species_v4.py, compare_species_v5.py, fetch_species_v4_fresh.py … |
| `nested_species_ensemble_v5.py` | py | v5 | não | não | results | compare_species_v5.py, fetch_species_v5_fresh.py, test_nested_species_v5.py |
| `notebook_code.py` | py | — | não | não | results | — |
| `notebook_transfer_features.py` | py | — | não | não | results | analyze_notebook_transfer.py, reproduce_notebook_transfer.py |
| `notebooks` | dir | — | sim | sim | results,tracked | — |
| `optimize_ensemble_oof.py` | py | — | não | não | results | — |
| `output` | dir | — | não | não | results,tracked | — |
| `pattern_classifiers.py` | py | — | sim | sim (importado) | results,tracked | predict_pattern.py, test_pattern_classifiers.py, train_pattern_classifiers.py … |
| `pattern_reporting.py` | py | — | sim | sim (versionado) | — | — |
| `predict_audio.py` | py | — | não | não | results | — |
| `predict_audio_v1_backup.py` | py | v1 backup | não | não | — | — |
| `predict_notebook_transfer.py` | py | — | não | não | — | — |
| `predict_pattern.py` | py | — | sim | sim (importado) | — | test_pattern_classifiers.py |
| `predict_ternary.py` | py | — | sim | sim (importado) | results | fetch_ternary_fresh.py, test_ternary_synthetic.py, test_ternary_verified.py |
| `predict_ternary_lstm.py` | py | — | sim | sim (importado) | — | test_notebook_workflow.py |
| `predict_torres_2025.py` | py | — | não | não | results | — |
| `predict_torres_aedes_transfer.py` | py | — | não | não | — | verify_torres_aedes_transfer.py |
| `prepare_biodcase_benchmark.py` | py | — | não | não | results | — |
| `prepare_delivery.py` | py | — | não | não | results | — |
| `prepare_multisource.py` | py | — | não | não | results | — |
| `prepare_reliability_data.py` | py | — | não | não | results | evaluate_cross_domain_strategies.py, evaluate_reliability.py, evaluate_sex_balanced_reliability.py … |
| `preprocessing_notebook.py` | py | — | não | não | results | build_notebook.py |
| `probe_arduino_noise.py` | py | — | não | não | results | build_notebook.py |
| `pytest.ini` | ini | — | sim | sim | — | — |
| `reanalysis` | dir | — | sim | sim | results,tracked | — |
| `reliability_metrics.py` | py | — | não | não | results | compare_reduced_databases.py, evaluate_cross_domain_strategies.py, evaluate_reliability.py … |
| `reliability_models.py` | py | — | não | não | results | evaluate_reliability.py, evaluate_sex_balanced_reliability.py, test_reliability.py … |
| `report_species_preprocessing.py` | py | — | não | não | results | — |
| `report_ternary_lstm.py` | py | — | não | não | — | — |
| `reports` | dir | — | sim | sim | results | — |
| `reproducao_estudos_originais_2026_10_03.html` | html | 2026-10-03 | não | não | results | — |
| `reproducao_estudos_originais_2026_10_03.ipynb` | ipynb | 2026-10-03 | não | não | results | — |
| `reproduce_notebook_transfer.py` | py | — | não | não | results | acquire_notebook_transfer_check.py, analyze_notebook_transfer.py, predict_notebook_transfer.py |
| `reproduction_models.py` | py | — | não | não | results | cross_domain_strategy_models.py, evaluate_reproduction.py, reliability_models.py … |
| `requirements-report.txt` | txt | — | não | não | — | — |
| `requirements.txt` | txt | — | sim | sim | — | — |
| `research_metrics.py` | py | — | não | não | results | audit_research.py, test_research_audit.py |
| `research_metrics_v2.py` | py | v2 | não | não | results | analyze_notebook_transfer.py, compare_detection_levels.py, compare_detection_levels_v2.py … |
| `results` | dir | — | não | não | results,tracked | — |
| `robust_audio_features.py` | py | — | não | não | results | experiment_mixed_datasets.py, predict_audio.py, predict_audio_v1_backup.py |
| `robust_audio_features_v1_backup.py` | py | v1 backup | não | não | — | — |
| `robust_reproduction_ensemble.py` | py | — | não | não | results | test_reproduction.py, train_reliability.py, train_sex_balanced_reliability.py |
| `run_author_dr_biol.py` | py | — | não | não | results | — |
| `run_author_mosquitosong.py` | py | — | não | não | results | — |
| `scripts` | dir | — | não | não | results | — |
| `serve.py` | py | — | não | não | results | — |
| `species_preprocessing.py` | py | — | não | não | results | evaluate_species_preprocessing.py, predict_torres_2025.py, report_species_preprocessing.py … |
| `summarize_aedes_classical.py` | py | — | não | não | results | — |
| `summarize_corrected_validation.py` | py | — | não | não | results | — |
| `summarize_cross_domain_strategies.py` | py | — | não | não | results | — |
| `summarize_frequency_shift.py` | py | — | não | não | results | — |
| `summarize_multisource.py` | py | — | não | não | results | — |
| `summarize_nested_species.py` | py | — | não | não | results | — |
| `summarize_reduced_databases.py` | py | — | não | não | results | — |
| `summarize_reliability.py` | py | — | não | não | results | summarize_sex_balanced_reliability.py |
| `summarize_sex_balanced_reliability.py` | py | — | não | não | results | — |
| `summarize_species_v2.py` | py | v2 | não | não | results | — |
| `summarize_torres_2025.py` | py | — | não | não | — | — |
| `swarm_adapted_ensemble.py` | py | — | não | não | results | corrected_validation.py, test_corrected_validation.py |
| `ternary_protocol.py` | py | — | sim | sim (importado) | results | evaluate_ternary_event_pipeline.py, evaluate_ternary_source_transfer.py, evaluate_ternary_synthetic.py … |
| `ternary_reporting.py` | py | — | sim | sim (importado) | results | evaluate_ternary_event_pipeline.py, evaluate_ternary_source_transfer.py, evaluate_ternary_synthetic.py … |
| `test_expanded_audio.py` | py | — | não | não | results | — |
| `test_large_audio.py` | py | — | não | não | results | — |
| `test_reduced_databases_unseen.py` | py | — | não | não | results | — |
| `tests` | dir | — | sim | sim | results | — |
| `tinyml_decimated_frontend.py` | py | — | não | não | results | — |
| `tmp` | dir | — | não | não | — | — |
| `torres_lstm.py` | py | — | sim | sim (importado) | results | apply_torres_2025.py, evaluate_species_preprocessing.py, fetch_torres_2025.py … |
| `trabalho_ifsc` | dir | — | não | não | — | — |
| `train_aedes_binary.py` | py | — | não | não | results | benchmark_all_methods.py, evaluate_aedes_binary.py, evaluate_aedes_classical.py … |
| `train_aedes_classical.py` | py | — | não | não | results | — |
| `train_arduino.py` | py | — | não | não | results | build_notebook.py, probe_arduino_noise.py, train_arduino_species.py … |
| `train_arduino_species.py` | py | — | não | não | results | build_notebook.py, verify_arduino_species.py |
| `train_cross_domain_strategies.py` | py | — | não | não | results | evaluate_cross_domain_strategies.py, fetch_cross_domain_strategy_test.py, summarize_cross_domain_strategies.py … |
| `train_expanded.py` | py | — | não | não | results | test_expanded_audio.py, test_expanded_protocol.py, test_large_audio.py … |
| `train_hard_negative_reproduction.py` | py | — | não | não | results | — |
| `train_hierarchical_models.py` | py | — | não | não | results | audit_git_species.py, predict_audio.py, predict_audio_v1_backup.py |
| `train_large_cnn.py` | py | — | não | não | results | prepare_multisource.py |
| `train_multisource.py` | py | — | não | não | results | evaluate_multisource.py, evaluate_preprocessing.py |
| `train_pattern_classifiers.py` | py | — | sim | sim (importado) | results,tracked | predict_pattern.py |
| `train_reliability.py` | py | — | não | não | results | test_reliability.py |
| `train_reproduction.py` | py | — | não | não | results | analyze_frequency_shift.py, compare_reduced_databases.py, evaluate_cross_domain_strategies.py … |
| `train_sex_balanced_reliability.py` | py | — | não | não | results | evaluate_sex_balanced_reliability.py, fetch_sex_balanced_reliability_fresh.py, summarize_sex_balanced_reliability.py … |
| `train_ternary_lstm.py` | py | — | sim | sim (importado) | results | pattern_classifiers.py, predict_pattern.py, predict_ternary_lstm.py … |
| `train_ternary_models.py` | py | — | não | não | — | — |
| `train_ternary_nested_models.py` | py | — | sim | sim (importado) | results | evaluate_ternary_event_pipeline.py, evaluate_ternary_source_transfer.py, evaluate_ternary_synthetic.py … |
| `train_torres_aedes_transfer.py` | py | — | não | não | — | evaluate_torres_aedes_datasets.py, predict_torres_aedes_transfer.py, verify_torres_aedes_datasets.py |
| `validacao_corrigida_2026_10_01.html` | html | 2026-10-01 | não | não | results | — |
| `validacao_corrigida_2026_10_01.ipynb` | ipynb | 2026-10-01 | não | não | results | — |
| `verify_arduino.py` | py | — | não | não | results | build_notebook.py |
| `verify_arduino_species.py` | py | — | não | não | results | build_notebook.py |
| `verify_pattern_classifiers.py` | py | — | sim | sim (importado) | — | test_pattern_classifiers.py, train_pattern_classifiers.py |
| `verify_ternary_lstm.py` | py | — | sim | sim (importado) | — | test_ternary_lstm.py |
| `verify_torres_2025.py` | py | — | não | não | — | — |
| `verify_torres_aedes_datasets.py` | py | — | não | não | — | — |
| `verify_torres_aedes_transfer.py` | py | — | não | não | — | — |

## Proposta de layout (não executada)

Ficam na raiz: os itens versionados, e também `data/`, `results/`, `output/`, `external/`, `literature/`, `dist/`, `tmp/`, ambientes e caches. Os protocolos e manifestos usam esses caminhos como entrada.

| Destino | Itens |
|---|---:|
| `experiments/` | 145 |
| `legacy/` | 4 |
| `legacy/docs/` | 6 |
| `legacy/misc/` | 3 |
| `legacy/notebooks/` | 16 |
| `legacy/scripts/` | 13 |

| Risco | Itens |
|---|---:|
| Alto | 157 |
| Baixo | 28 |
| Médio | 2 |

**Recomendação.** Mover agora só os itens de risco baixo. Os de risco médio só podem ir junto com seus importadores. Os de risco **alto** (`experiments/`) devem ficar onde estão: movê-los invalida a verificação por hash das rodadas antigas, porque os manifestos registram o caminho relativo à raiz. Outra opção é deixá-los na raiz e usar este inventário como índice.

Riscos gerais: os testes locais não versionados importam módulos da raiz via `pythonpath = .`; notebooks datados podem ler arquivos por caminho relativo; e `.gitignore` usa allowlist (`/*`), então pastas novas como `legacy/` continuam ignoradas sem ajuste.

| Item | Destino proposto | Importado por | Manifestos com SHA-256 que o citam | Risco |
|---|---|---|---|---|
| `.gitattributes` | `legacy/misc/` | — | 1 (`results/arduino_species/windows_build_audit.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `.github` | `legacy/` | — | 1 (`results/arduino_species/windows_build_audit.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `DRIVE_REVIEW_2026_10_14.md` | `legacy/docs/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `GITHUB_PUBLICATION_2026_10_14.md` | `legacy/docs/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `MODEL_VALIDATION.md` | `legacy/docs/` | — | 4 (`results/species_nested_2026_10_13/completion_manifest.json` e mais 3) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `MODEL_VALIDATION_v4.md` | `legacy/docs/` | — | 1 (`results/species_nested_v4_2026_10_14/completion_manifest.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `README_TORRES_2025.md` | `legacy/docs/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `README_v4.md` | `legacy/docs/` | — | 1 (`results/species_nested_v4_2026_10_14/completion_manifest.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `acoustic_features_v2.py` | `experiments/` | `nested_species_ensemble_v2.py`, `tests/test_nested_species_v2.py` | 14 (`results/species_nested_2026_10_13/completion_manifest.json` e mais 13) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `acoustic_features_v3.py` | `experiments/` | `nested_species_ensemble_v3.py`, `tests/test_nested_species_v3.py` | 9 (`results/species_nested_v3_2026_10_14/completion_manifest.json` e mais 8) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `acoustic_features_v4.py` | `experiments/` | `analyze_notebook_transfer.py`, `nested_species_ensemble_v4.py`, `reproduce_notebook_transfer.py` e mais 1 | 6 (`results/notebook_transfer_2026_10_14_2030/freeze.json` e mais 5) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `acoustic_features_v5.py` | `experiments/` | `nested_species_ensemble_v5.py`, `tests/test_nested_species_v5.py` | 1 (`results/drive_review_2026_10_14_161041/source_provenance.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `acquire_notebook_transfer_check.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `aedes_binary.py` | `experiments/` | `analyze_frequency_shift.py`, `benchmark_all_methods.py`, `compare_reduced_databases.py` e mais 30 | 5 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 4) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `aedes_binary_notebook.py` | `experiments/` | `build_notebook.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `aedes_classical_notebook.py` | `experiments/` | `build_notebook.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `analyze_frequency_shift.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `analyze_notebook_transfer.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `apply_torres_2025.py` | `experiments/` | `evaluate_species_preprocessing.py`, `predict_torres_2025.py` | 3 (`results/species_preprocessing_2026_10_15/protocol.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `arduino_frontend.py` | `experiments/` | `cross_validate_drive_audio.py`, `experiment_mixed_datasets.py`, `predict_audio.py` e mais 3 | 3 (`results/drive_review_2026_10_14_161041/review_protocol.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `arduino_models.py` | `experiments/` | `train_arduino.py`, `train_expanded.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `arduino_species_models.py` | `experiments/` | `train_arduino_species.py`, `train_expanded.py`, `verify_arduino_species.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `audio_io.py` | `legacy/scripts/` | `predict_audio.py`, `predict_audio_v1_backup.py` | 0 | Médio: quebra os imports listados, a menos que os importadores sejam movidos juntos e o `sys.path` ajustado. |
| `audio_preprocessing.py` | `experiments/` | `evaluate_preprocessing.py`, `tests/test_audio_preprocessing.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `audit_author_mosquitosong.py` | `experiments/` | — | 3 (`results/faithful_reproduction_2026_10_03/artifact_verification.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `audit_git_species.py` | `experiments/` | — | 3 (`results/git_review_2026_10_08/manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `audit_research.py` | `experiments/` | — | 3 (`results/research_audit_2026_10_07/manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `benchmark_all_methods.py` | `experiments/` | `tests/test_corrected_validation.py` | 3 (`results/corrected_2026_10_01/delivery_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `build_faithful_reproduction_artifacts.py` | `experiments/` | — | 3 (`results/faithful_reproduction_2026_10_03/artifact_verification.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `build_notebook.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `build_reliability_artifacts.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `build_report.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `comparacao_bases_reduzidas_2026_10_02.html` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `comparacao_bases_reduzidas_2026_10_02.ipynb` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `comparacao_frequencias_2026_10_03.html` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `comparacao_frequencias_2026_10_03.ipynb` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `compare_detection_levels.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `compare_detection_levels_v2.py` | `experiments/` | — | 6 (`results/species_nested_2026_10_13/completion_manifest.json` e mais 5) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `compare_reduced_databases.py` | `experiments/` | `summarize_reduced_databases.py`, `test_reduced_databases_unseen.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `compare_species_v3.py` | `experiments/` | `compare_species_v4.py`, `nested_species_ensemble_v3.py` | 9 (`results/species_nested_v3_2026_10_14/completion_manifest.json` e mais 8) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `compare_species_v4.py` | `experiments/` | `compare_species_v5.py`, `nested_species_ensemble_v4.py` | 4 (`results/species_nested_v4_2026_10_14/completion_manifest.json` e mais 3) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `compare_species_v5.py` | `experiments/` | `nested_species_ensemble_v5.py` | 1 (`results/drive_review_2026_10_14_161041/source_provenance.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `confiabilidade_binaria_aedes_2026_10_01.html` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `confiabilidade_binaria_aedes_2026_10_01.ipynb` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `confiabilidade_binaria_aedes_adaptacao_2026_10_01.html` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `confiabilidade_binaria_aedes_adaptacao_2026_10_01.ipynb` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `corrected_validation.py` | `experiments/` | `benchmark_all_methods.py`, `compare_reduced_databases.py`, `evaluate_biodcase_d5.py` e mais 21 | 3 (`results/corrected_2026_10_01/delivery_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `cross_domain_strategy_models.py` | `experiments/` | `evaluate_cross_domain_strategies.py`, `tests/test_cross_domain_strategies.py`, `train_cross_domain_strategies.py` | 3 (`results/cross_domain_strategies_2026_10_03/freeze.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `cross_validate_drive_audio.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `delivery_manifest.json` | `legacy/misc/` | — | 4 (`results/species_nested_2026_10_13/historical_snapshot.json` e mais 3) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `estrategias_contrastivas_enxames_2026_10_03.html` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `estrategias_contrastivas_enxames_2026_10_03.ipynb` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `evaluate_aedes_binary.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_aedes_classical.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_author_mosquitosong_external.py` | `experiments/` | — | 3 (`results/faithful_reproduction_2026_10_03/artifact_verification.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_biodcase_d5.py` | `experiments/` | — | 3 (`results/corrected_2026_10_01/delivery_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_corrected_fresh.py` | `experiments/` | — | 3 (`results/corrected_2026_10_01/delivery_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_cross_domain_strategies.py` | `experiments/` | `tests/test_cross_domain_strategies.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_hard_negative_reproduction.py` | `experiments/` | `evaluate_reliability.py`, `evaluate_sex_balanced_reliability.py`, `tests/test_reproduction.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_multisource.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_nested_genus.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_nested_species.py` | `experiments/` | `compare_detection_levels.py`, `evaluate_nested_genus.py`, `fetch_species_v2_fresh.py` e mais 8 | 14 (`results/species_nested_2026_10_09/execution_verification.json` e mais 13) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_new_approaches.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_new_database.py` | `experiments/` | `corrected_validation.py` | 3 (`results/corrected_2026_10_01/delivery_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_preprocessing.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_reliability.py` | `experiments/` | `summarize_reliability.py`, `tests/test_reliability.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_reproduction.py` | `experiments/` | `evaluate_hard_negative_reproduction.py`, `fetch_reproduction_fresh.py`, `robust_reproduction_ensemble.py` e mais 2 | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_sex_balanced_reliability.py` | `experiments/` | `evaluate_cross_domain_strategies.py`, `summarize_sex_balanced_reliability.py`, `tests/test_sex_balanced_reliability.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_species_preprocessing.py` | `experiments/` | — | 1 (`results/species_preprocessing_2026_10_15/protocol.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_ternary_source_transfer.py` | `legacy/scripts/` | `tests/test_drive_exclusion.py` | 0 | Médio: quebra os imports listados, a menos que os importadores sejam movidos juntos e o `sys.path` ajustado. |
| `evaluate_ternary_synthetic.py` | `experiments/` | — | 1 (`results/ternary_synthetic_2026_10_14/delivery_manifest.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `evaluate_torres_aedes_datasets.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `execute_notebook.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `expanded_notebook.py` | `experiments/` | `build_notebook.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `experiment_mixed_datasets.py` | `experiments/` | — | 3 (`results/git_review_2026_10_08/manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `explain_aedes_classical.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_additional_audio.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_biodcase.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_corrected_fresh.py` | `experiments/` | — | 3 (`results/corrected_2026_10_01/delivery_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_cross_domain_strategy_test.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_data.py` | `experiments/` | `notebook_code.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_faithful_dr_biol.py` | `experiments/` | — | 3 (`results/faithful_reproduction_2026_10_03/artifact_verification.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_fresh_aedes_binary.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_fresh_aedes_classical.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_fresh_multisource_test.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_hard_negative_final_test.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_humbug.py` | `experiments/` | `acquire_notebook_transfer_check.py`, `evaluate_nested_species.py`, `fetch_species_v2_fresh.py` e mais 5 | 16 (`results/species_nested_2026_10_13/freeze.json` e mais 15) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_reliability_data.py` | `experiments/` | `evaluate_reliability.py`, `fetch_reliability_fresh.py`, `fetch_sex_balanced_reliability_fresh.py` e mais 5 | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_reliability_fresh.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_reproduction_fresh.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_sex_balanced_reliability_fresh.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_species_v2_fresh.py` | `experiments/` | — | 6 (`results/species_nested_2026_10_13/completion_manifest.json` e mais 5) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_species_v2_fresh_checked.py` | `experiments/` | — | 3 (`results/species_nested_2026_10_13/completion_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_species_v3_fresh.py` | `experiments/` | `nested_species_ensemble_v3.py` | 9 (`results/species_nested_v3_2026_10_14/completion_manifest.json` e mais 8) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_species_v4_fresh.py` | `experiments/` | `nested_species_ensemble_v4.py` | 4 (`results/species_nested_v4_2026_10_14/completion_manifest.json` e mais 3) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_species_v5_fresh.py` | `experiments/` | `nested_species_ensemble_v5.py` | 1 (`results/drive_review_2026_10_14_161041/source_provenance.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_ternary_fresh.py` | `experiments/` | — | 3 (`results/ternary_verified_2026_10_14_corrections/freeze.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `fetch_torres_2025.py` | `experiments/` | — | 2 (`results/torres_2025_2026_10_15/infection/protocol.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `firmware` | `legacy/` | — | 9 (`results/arduino/compile_audit.json` e mais 8) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `improve_device.py` | `experiments/` | `build_notebook.py`, `compare_detection_levels_v2.py`, `compare_species_v3.py` e mais 30 | 15 (`results/git_review_2026_10_08/manifest.json` e mais 14) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `large_audio_model.py` | `experiments/` | `aedes_binary.py`, `evaluate_multisource.py`, `evaluate_preprocessing.py` e mais 6 | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `large_cnn_notebook.py` | `experiments/` | `build_notebook.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `leakage_auditor.py` | `experiments/` | `audit_git_species.py`, `train_hierarchical_models.py` | 3 (`results/git_review_2026_10_08/manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `loso_generalization.py` | `experiments/` | `audit_research.py`, `tests/test_research_audit.py` | 3 (`results/research_audit_2026_10_07/manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `models_committee.py` | `experiments/` | `benchmark_all_methods.py`, `tests/test_corrected_validation.py` | 3 (`results/corrected_2026_10_01/delivery_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `models_deep.py` | `experiments/` | `corrected_validation.py`, `evaluate_biodcase_d5.py`, `evaluate_new_database.py` e mais 1 | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `models_trees.py` | `experiments/` | `models_committee.py` | 3 (`results/corrected_2026_10_01/delivery_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `mosquito_wingbeat_estudo.html` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `mosquito_wingbeat_estudo.ipynb` | `legacy/notebooks/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `multisource_notebook.py` | `experiments/` | `build_notebook.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `nested_species_ensemble.py` | `experiments/` | `compare_detection_levels_v2.py`, `compare_species_v3.py`, `compare_species_v4.py` e mais 15 | 15 (`results/species_nested_2026_10_09/execution_verification.json` e mais 14) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `nested_species_ensemble_v2.py` | `experiments/` | `acquire_notebook_transfer_check.py`, `compare_detection_levels_v2.py`, `compare_species_v4.py` e mais 8 | 14 (`results/species_nested_2026_10_13/completion_manifest.json` e mais 13) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `nested_species_ensemble_v3.py` | `experiments/` | `compare_species_v3.py`, `compare_species_v4.py`, `fetch_species_v3_fresh.py` e mais 3 | 9 (`results/species_nested_v3_2026_10_14/completion_manifest.json` e mais 8) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `nested_species_ensemble_v4.py` | `experiments/` | `compare_species_v4.py`, `compare_species_v5.py`, `fetch_species_v4_fresh.py` e mais 2 | 4 (`results/species_nested_v4_2026_10_14/completion_manifest.json` e mais 3) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `nested_species_ensemble_v5.py` | `experiments/` | `compare_species_v5.py`, `fetch_species_v5_fresh.py`, `tests/test_nested_species_v5.py` | 1 (`results/drive_review_2026_10_14_161041/source_provenance.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `notebook_code.py` | `experiments/` | — | 3 (`results/species_nested_2026_10_09/protocol.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `notebook_transfer_features.py` | `experiments/` | `analyze_notebook_transfer.py`, `reproduce_notebook_transfer.py` | 2 (`results/notebook_transfer_2026_10_14_2030/freeze.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `optimize_ensemble_oof.py` | `experiments/` | — | 4 (`results/git_review_2026_10_08/manifest.json` e mais 3) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `predict_audio.py` | `experiments/` | — | 6 (`results/drive_audio_drive_2026_10_14/provenance.json` e mais 5) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `predict_audio_v1_backup.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `predict_notebook_transfer.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `predict_torres_2025.py` | `experiments/` | — | 1 (`results/species_preprocessing_2026_10_15/validation.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `predict_torres_aedes_transfer.py` | `experiments/` | `verify_torres_aedes_transfer.py` | 1 (`results/torres_aedes_transfer_2026_10_15/manifest.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `prepare_biodcase_benchmark.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `prepare_delivery.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `prepare_multisource.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `prepare_reliability_data.py` | `experiments/` | `evaluate_cross_domain_strategies.py`, `evaluate_reliability.py`, `evaluate_sex_balanced_reliability.py` e mais 3 | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `preprocessing_notebook.py` | `experiments/` | `build_notebook.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `probe_arduino_noise.py` | `experiments/` | `build_notebook.py`, `mosquito_wingbeat_estudo.ipynb` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `reliability_metrics.py` | `experiments/` | `compare_reduced_databases.py`, `evaluate_cross_domain_strategies.py`, `evaluate_reliability.py` e mais 7 | 4 (`results/reliability_2026_10_01/freeze.json` e mais 3) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `reliability_models.py` | `experiments/` | `evaluate_reliability.py`, `evaluate_sex_balanced_reliability.py`, `tests/test_reliability.py` e mais 3 | 4 (`results/reliability_2026_10_01/freeze.json` e mais 3) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `report_species_preprocessing.py` | `experiments/` | — | 1 (`results/species_preprocessing_2026_10_15/validation.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `report_ternary_lstm.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `reproducao_estudos_originais_2026_10_03.html` | `legacy/notebooks/` | — | 1 (`results/faithful_reproduction_2026_10_03/artifact_verification.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `reproducao_estudos_originais_2026_10_03.ipynb` | `legacy/notebooks/` | — | 1 (`results/faithful_reproduction_2026_10_03/artifact_verification.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `reproduce_notebook_transfer.py` | `experiments/` | `acquire_notebook_transfer_check.py`, `analyze_notebook_transfer.py`, `predict_notebook_transfer.py` | 2 (`results/notebook_transfer_2026_10_14_2030/freeze.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `reproduction_models.py` | `experiments/` | `cross_domain_strategy_models.py`, `evaluate_reproduction.py`, `reliability_models.py` e mais 4 | 3 (`results/reproduction_2026_10_01/freeze.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `requirements-report.txt` | `legacy/misc/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `research_metrics.py` | `experiments/` | `audit_research.py`, `tests/test_research_audit.py` | 3 (`results/research_audit_2026_10_07/manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `research_metrics_v2.py` | `experiments/` | `analyze_notebook_transfer.py`, `compare_detection_levels.py`, `compare_detection_levels_v2.py` e mais 12 | 16 (`results/notebook_transfer_2026_10_14_2030/freeze.json` e mais 15) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `robust_audio_features.py` | `experiments/` | `experiment_mixed_datasets.py`, `predict_audio.py`, `predict_audio_v1_backup.py` | 6 (`results/drive_audio_drive_2026_10_14/provenance.json` e mais 5) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `robust_audio_features_v1_backup.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `robust_reproduction_ensemble.py` | `experiments/` | `tests/test_reproduction.py`, `train_reliability.py`, `train_sex_balanced_reliability.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `run_author_dr_biol.py` | `experiments/` | — | 3 (`results/faithful_reproduction_2026_10_03/artifact_verification.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `run_author_mosquitosong.py` | `experiments/` | — | 3 (`results/faithful_reproduction_2026_10_03/artifact_verification.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `scripts` | `legacy/` | — | 4 (`results/arduino_species/windows_build_audit.json` e mais 3) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `serve.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `species_preprocessing.py` | `experiments/` | `evaluate_species_preprocessing.py`, `predict_torres_2025.py`, `report_species_preprocessing.py` e mais 1 | 1 (`results/species_preprocessing_2026_10_15/protocol.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_aedes_classical.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_corrected_validation.py` | `experiments/` | — | 3 (`results/corrected_2026_10_01/delivery_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_cross_domain_strategies.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_frequency_shift.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_multisource.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_nested_species.py` | `experiments/` | — | 3 (`results/species_nested_2026_10_09/execution_verification.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_reduced_databases.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_reliability.py` | `experiments/` | `summarize_sex_balanced_reliability.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_sex_balanced_reliability.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_species_v2.py` | `experiments/` | — | 3 (`results/species_nested_2026_10_13/completion_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `summarize_torres_2025.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `swarm_adapted_ensemble.py` | `experiments/` | `corrected_validation.py`, `tests/test_corrected_validation.py` | 3 (`results/corrected_2026_10_01/delivery_manifest.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `test_expanded_audio.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `test_large_audio.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `test_reduced_databases_unseen.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `tinyml_decimated_frontend.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `trabalho_ifsc` | `legacy/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `train_aedes_binary.py` | `experiments/` | `benchmark_all_methods.py`, `evaluate_aedes_binary.py`, `evaluate_aedes_classical.py` e mais 3 | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_aedes_classical.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_arduino.py` | `experiments/` | `build_notebook.py`, `mosquito_wingbeat_estudo.ipynb`, `probe_arduino_noise.py` e mais 3 | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_arduino_species.py` | `experiments/` | `build_notebook.py`, `mosquito_wingbeat_estudo.ipynb`, `verify_arduino_species.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_cross_domain_strategies.py` | `experiments/` | `evaluate_cross_domain_strategies.py`, `fetch_cross_domain_strategy_test.py`, `summarize_cross_domain_strategies.py` e mais 1 | 3 (`results/cross_domain_strategies_2026_10_03/freeze.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_expanded.py` | `experiments/` | `test_expanded_audio.py`, `test_large_audio.py`, `tests/test_expanded_protocol.py` e mais 2 | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_hard_negative_reproduction.py` | `experiments/` | — | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_hierarchical_models.py` | `experiments/` | `audit_git_species.py`, `predict_audio.py`, `predict_audio_v1_backup.py` | 4 (`results/drive_review_2026_10_14_161041/review_protocol.json` e mais 3) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_large_cnn.py` | `experiments/` | `prepare_multisource.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_multisource.py` | `experiments/` | `evaluate_multisource.py`, `evaluate_preprocessing.py` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_reliability.py` | `experiments/` | `tests/test_reliability.py` | 3 (`results/reliability_2026_10_01/freeze.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_reproduction.py` | `experiments/` | `analyze_frequency_shift.py`, `compare_reduced_databases.py`, `evaluate_cross_domain_strategies.py` e mais 23 | 3 (`results/reproduction_2026_10_01/freeze.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_sex_balanced_reliability.py` | `experiments/` | `evaluate_sex_balanced_reliability.py`, `fetch_sex_balanced_reliability_fresh.py`, `summarize_sex_balanced_reliability.py` e mais 1 | 3 (`results/sex_balanced_reliability_2026_10_01/freeze.json` e mais 2) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `train_ternary_models.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `train_torres_aedes_transfer.py` | `experiments/` | `evaluate_torres_aedes_datasets.py`, `predict_torres_aedes_transfer.py`, `verify_torres_aedes_datasets.py` | 2 (`results/torres_aedes_transfer_2026_10_15/manifest.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `validacao_corrigida_2026_10_01.html` | `legacy/notebooks/` | — | 1 (`results/corrected_2026_10_01/delivery_manifest.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `validacao_corrigida_2026_10_01.ipynb` | `legacy/notebooks/` | — | 1 (`results/corrected_2026_10_01/delivery_manifest.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `verify_arduino.py` | `experiments/` | `build_notebook.py`, `mosquito_wingbeat_estudo.ipynb` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `verify_arduino_species.py` | `experiments/` | `build_notebook.py`, `mosquito_wingbeat_estudo.ipynb` | 2 (`results/species_nested_v3_2026_10_14/historical_snapshot.json` e mais 1) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
| `verify_torres_2025.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `verify_torres_aedes_datasets.py` | `legacy/scripts/` | — | 0 | Baixo: nenhum import nem manifesto o cita. |
| `verify_torres_aedes_transfer.py` | `experiments/` | — | 1 (`results/torres_aedes_transfer_2026_10_15/manifest.json`) | **Alto**: o caminho entra em manifestos de rodadas com SHA-256; os verificadores dessas rodadas deixariam de achar o arquivo. |
