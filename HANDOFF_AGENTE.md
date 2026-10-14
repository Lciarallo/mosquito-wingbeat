# Relatório Técnico de Handoff — Projeto Mosquito-Wingbeat

**Data:** 14 de Outubro de 2026  
**Repositório Local:** `/home/lciarallo/Documentos/Git_projects/mosquito-wingbeat`  
**Destino Google Drive:** `my_drive:repositorios do github/mosquito-wingbeat`  
**Ambiente Python:** Python 3.14.8 (`.venv` com suporte a `soundfile`, `scikit-learn`, `librosa`, `torch`)

---

## 1. Visão Geral da Operação

Esta rodada de intervenção cobriu:
1. **Download dos arquivos mais recentes do Google Drive** para a máquina local.
2. **Execução integral da bateria de testes** (`pytest`) com a regra estrita de **zero alteração de código**.
3. **Upload incremental da pasta `results/`** para o Google Drive para manter os modelos e matrizes de confusão sincronizados sem travar a rede em 45 GB de áudios brutos.
4. **Documentação estruturada** dos diagnósticos para continuidade pelo próximo agente.

---

## 2. Arquivos Atualizados Baixados do Google Drive

Os seguintes arquivos foram modificados no Drive e baixados para o repositório local:

| Arquivo | Tamanho | Descrição da Atualização |
| :--- | :--- | :--- |
| [`predict_audio.py`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/predict_audio.py) | 32.3 KB | **Nova implementação**: Diagnóstico hierárquico (Tier 1 a 3) com **Segmentação Temporal de Enxames de Aedes** (pente harmônico SHS multi-pitch, correção Bonferroni, marginalização de potência com cardinalidade corrigida `gamma=1.0`, `tau=1.5` e histerese causal). |
| [`robust_audio_features.py`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/robust_audio_features.py) | 12.3 KB | Funções expandidas de análise harmônica, estimativa de SNR por largura de banda e detecção de regimes acústicos. |
| [`predict_audio_v1_backup.py`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/predict_audio_v1_backup.py) | 24.8 KB | Backup da versão v1 de inferência. |
| [`robust_audio_features_v1_backup.py`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/robust_audio_features_v1_backup.py) | 4.5 KB | Backup da versão v1 dos extratores acústicos. |
| [`results/drive_audio_raw_summary.json`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_audio_raw_summary.json) | 9.4 KB | Sumário atualizado da predição em áudio bruto para `drive_audio.wav`. |
| [`results/drive_audio_summary.json`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/drive_audio_summary.json) | 9.3 KB | Sumário atualizado da predição em áudio emulado (ST MP34DT05). |
| [`results/drive_audio_*_v1_backup.json`](file:///home/lciarallo/Documentos/Git_projects/mosquito-wingbeat/results/) | ~1.2 KB | Backups dos relatórios de inferência anteriores. |

---

## 3. Resultado da Bateria de Testes (`pytest`)

### Comando de Execução:
```bash
cd /home/lciarallo/Documentos/Git_projects/mosquito-wingbeat
source .venv/bin/activate
PYTHONPATH=. python -m pytest tests/
```

### Placar Geral:
- **Total de testes:** 202
- **Aprovados (PASSED):** **199** (98.5%)
- **Falhas (FAILED):** **3** (1.5%)
- **Tempo de execução:** ~30 segundos

### Módulos 100% Verificados e Aprovados:
- `tests/test_aedes_binary.py` (3/3)
- `tests/test_aedes_classical.py` (3/3)
- `tests/test_audio_preprocessing.py` (4/4)
- `tests/test_corrected_validation.py` (7/7)
- `tests/test_cross_domain_strategies.py` (12/12)
- `tests/test_expanded_protocol.py` (3/3)
- `tests/test_faithful_reproduction.py` (8/8)
- `tests/test_install_arduino.py` (23/23)
- `tests/test_large_audio_model.py` (3/3)
- `tests/test_multisource_protocol.py` (4/4)
- `tests/test_nested_species.py` (13/13)
- `tests/test_reliability.py` (11/11)
- `tests/test_reproduction.py` (11/11)
- `tests/test_research_audit.py` (12/12)
- `tests/test_sex_balanced_reliability.py` (6/6)
- `tests/test_windows_installer.py` (13/13)

---

## 4. Diagnóstico Detalhado das 3 Falhas

Todas as 3 falhas ocorreram estritamente em verificações de **congelamento histórico de hash SHA-256** (`verify_history`):

1. **`test_nested_species_v2.py::test_all_historical_results_sha256_are_intact`**
   - **Erro:** `ValueError: Historical file changed or disappeared: results/drive_audio_raw_summary.json`
   - **Causa Raiz:** O arquivo de resumo de inferência foi re-gerado pela nova versão do detector no Drive.
2. **`test_nested_species_v3.py::test_all_preexisting_artifacts_are_untouched`**
   - **Erro:** `ValueError: Historical artifact changed: predict_audio.py`
   - **Causa Raiz:** O `predict_audio.py` recebeu o novo algoritmo de segmentação temporal de enxames no Drive, alterando seu hash SHA-256.
3. **`test_nested_species_v4.py::test_strict_f1_and_all_historical_files_are_intact`**
   - **Erro:** `ValueError: Historical artifact changed: predict_audio.py`
   - **Causa Raiz:** Mesma causa do teste acima.

> [!NOTE]
> **Nenhum teste de regressão funcional, cálculo matemático, pré-processamento de sinal ou modelo ML falhou.** A falha é unicamente um guarda-chuva de conformidade que alerta que um script existente foi modificado após o congelamento do snapshot da versão anterior.

---

## 5. Teste de Validação Prática do Novo Algoritmo

Foi executado o teste de fumaça e validação ponta a ponta com o áudio real:
```bash
source .venv/bin/activate
python3 predict_audio.py drive_audio.wav --compare
```

**Resultado Observado:**
- **Duração do áudio:** 106.24 segundos.
- **Janelas analisadas:** 211 janelas de 1 segundo (hop de 0.5s).
- **Enxame detectado:** 125 janelas de enxame ativo (59.24%).
- **Identificação taxonômica:** 100% *Aedes aegypti* (em áudio bruto) e 99.3% (em microfone ST MP34DT05 emulado).
- **Segmentação temporal contígua:** Mapeamento exato de silêncio vs aproximação vs enxame estabilizado:
  - `0.0 - 23.5s`: `AUSENTE_RUIDO_FUNDO` (SNR -2.3 dB)
  - `48.0 - 91.5s`: `ENXAME_AEDES_PRESENTE` (43.5s ininterruptos de enxame de alta densidade, SNR +10.9 dB, F0 mediano de 602.5 Hz).

---

## 6. Estado do Envio ao Google Drive (`results/`)

Para evitar tráfego de dados desnecessário (os ~35 GB de bases de áudio brutas já sincronizadas), apenas a pasta de modelos e saídas experimentais foi direcionada:
- **Destino:** `my_drive:repositorios do github/mosquito-wingbeat/results`
- **Volume:** ~4.9 GB distribuídos entre matrizes, relatórios, checkpoints e modelos treinados.
- **Progresso:** Mais de 70% dos modelos sincronizados, com o restante sendo finalizado em segundo plano.

---

## 7. Instruções para o Próximo Agente

Ao assumir o repositório, oriente o próximo agente com as seguintes diretrizes:

1. **Sobre o ambiente Python:**
   - Sempre utilize o interpretador do `.venv`:
     ```bash
     source .venv/bin/activate
     PYTHONPATH=. python3 <script.py>
     ```
   - O Python de sistema (`/usr/bin/python3`) não possui as bibliotecas bioacústicas instaladas globalmente.
2. **Sobre os 3 testes com falha de hash:**
   - Se a intenção for **homologar a nova versão de `predict_audio.py`**, o agente deve atualizar os hashes correspondentes nos arquivos `results/species_nested_v*/historical_snapshot.json` calculando o novo `sha256` de `predict_audio.py` e `results/drive_audio_raw_summary.json`.
   - Se a intenção for manter o benchmark congelado estritamente como estava no v1, o agente pode restaurar `predict_audio_v1_backup.py` para `predict_audio.py`.
3. **Novo recurso disponível:**
   - O CLI `predict_audio.py` agora aceita `--compare`, `--region brazil`, `--emulate-arduino` e gera relatórios em CSV e JSON com segmentação temporal por intervalos contíguos.
