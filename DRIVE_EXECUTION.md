# Versão executável atual do Google Drive

Use **scripts/predict_audio.py** e **scripts/robust_audio_features.py** para os diagnósticos novos. Foram baixados do Drive autenticado, ajustados conforme o pedido e verificados. Os arquivos homônimos da raiz são referências históricas congeladas, restauradas aos SHA-256 esperados; as versões recebidas estão preservadas em output/drive_reference_2026_10_14/ e local_changes_backup/. Nenhum historical_snapshot.json foi reescrito.

Ative o .venv e execute:

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/predict_audio.py --audio data/raw/drive_test/drive_audio.wav --region brazil --emulate-arduino --output-csv results/drive_audio_drive_2026_10_14/drive_audio_predictions.csv --output-summary results/drive_audio_drive_2026_10_14/drive_audio_summary.json
PYTHONPATH=. python3 scripts/predict_audio.py --audio data/raw/drive_test/drive_audio.wav --region brazil --output-csv results/drive_audio_drive_2026_10_14/drive_audio_raw_predictions.csv --output-summary results/drive_audio_drive_2026_10_14/drive_audio_raw_summary.json
PYTHONPATH=. python3 scripts/predict_audio.py --audio data/raw/drive_test/drive_audio.wav --region global --output-csv results/drive_audio_drive_2026_10_14/drive_audio_global_predictions.csv --output-summary results/drive_audio_drive_2026_10_14/drive_audio_global_summary.json
PYTHONPATH=. python -m pytest tests/
```

Áudio privado disponível apenas no ambiente local autorizado, sem inclusão nesta entrega. A cópia data/raw/drive_test/drive_audio.wav é uma referência local ao original.

[Relatório dos ajustes e intervalos](reports/drive_audio_drive_2026_10_14.md) · [Terminal e todos os JSONs, completos](results/drive_audio_drive_2026_10_14/terminal_e_jsons_completos.md) · [Comparação v1/v2/v3/v4](reports/ensemble_especies_v4_2026_10_14.md).

Suíte: **202 passed**, sem alteração de snapshots/testes anteriores. Intervalos são decisões estimadas em janelas sobrepostas, não anotação entomológica independente. O diagnóstico heurístico neural é separado do ensemble aninhado v4; a confiabilidade externa das 20 espécies continua não estabelecida. Tudo foi executado offline, sem compilação/alteração de firmware.
