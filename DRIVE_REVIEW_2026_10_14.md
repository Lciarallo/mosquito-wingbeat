# Revisão da versão recebida do Drive — 14/10/2026, 16:10 UTC

Referência atual baixada: `scripts/predict_audio.py`, `scripts/robust_audio_features.py` e os quatro scripts da v5, sem alteração dos arquivos recebidos. As versões da raiz de `predict_audio.py` e `robust_audio_features.py` continuam históricas e congeladas.

O JSON novo informa 28/28; a reexecução encontrou **26/28 espécies e gêneros**. Nos mesmos 17 arquivos da avaliação histórica, o script neural/heurístico atual acertou **6/17 espécies e 10/17 gêneros**. A v5 aninhada tem código recebido, mas não havia pesos, OOF ou `freeze.json` v5 no Drive. Não confundir os dois pipelines.

Suíte completa recebida: **215 aprovados e uma falha nova v5**, com os 202 testes históricos aprovados. Não foi modificada para produzir um resultado artificialmente positivo. Os hashes de v1–v4 passaram antes e depois. A dependência da classificação entre janelas foi demonstrada e impede declarar essa versão estritamente indutiva.

[Relatório técnico](reports/revisao_resultados_drive_2026_10_14_161041.md) · [Resultados reproduzidos](results/drive_review_2026_10_14_161041/metrics.json) · [Terminal e JSONs completos](results/drive_review_2026_10_14_161041/terminal_e_jsons_completos.md) · [Pacote completo com previsões individuais e manifesto SHA-256](results/drive_review_2026_10_14_161041/drive_review_complete.zip).

Para repetir a auditoria em uma nova pasta de saída, adapte apenas os caminhos de `WORK` e `OUT` no script de auditoria, mantendo os scripts de inferência e pesos recebidos. O protocolo registra os hashes antes de qualquer previsão e não ajusta parâmetros. Não sobrescreva os resultados desta revisão ao repetir.

As três execuções CLI com os parâmetros já solicitados estão documentadas no terminal completo e usam `results/drive_review_2026_10_14_161041/`. O WAV privado não integra a entrega. A inferência continua offline e experimental, sem compilação de Arduino.
