# Relatório do Experimento: Generalização com Mistura de Datasets Bioacústicos

Data: 08/10/2026  
Projeto: mosquito-wingbeat  
Status: Concluído e Auditado  

## 1. Objetivo Científico
Investigar experimentalmente se a mistura de múltiplos conjuntos de dados acústicos (multi-dataset / multi-domain learning) reduz o atalho de aprendizado por hardware (Device Bias) e melhora a generalização out-of-distribution (OOD) em relação aos modelos monolíticos treinados em uma única fonte de dados.

## 2. Bases de Dados Integradas
1. **Domínio 1 (Dryad Original)**: Base com 24.405 janelas de 1s cobrindo 20 espécies e 569 grupos gravados em celulares diversos (Mukundarajan et al., 2017).
2. **Domínio 2 (TinyML Corpus - Rovai / Altayeb et al.)**: 49 arquivos de áudio adicionais (~41,5 minutos / 2.483 janelas de 1s a 16 kHz) distribuídos entre *Aedes aegypti*, *Aedes albopictus*, outros mosquitos e ruído de fundo.
3. **Domínio 3 Cego (Oxford HumBugDB)**: Gravação de campo selvagem de 11 minutos (`humbug_sample.wav`) realizada na Tanzânia com anotações temporais reais de voos de insetos.
4. **Domínio 4 Cego (Smartphone Brasil)**: Gravação externa de celular de 106s (`drive_audio.wav`) com ruído elétrico contínuo de 60 Hz e voo confirmado de *Aedes aegypti*.

## 3. Resumo dos Resultados

| Cenário | Teste Holdout TinyML (F1) | HumBugDB Selvagem (ROC AUC) | Drive Audio Campo (ROC AUC) |
| :--- | :---: | :---: | :---: |
| **Modelo Single-Domain (Dryad)** | 93,7% | 0,416 (Falha OOD) | 0,276 (Falha no ruído 60 Hz) |
| **Modelo Single-Domain (TinyML)** | 95,9% | 0,668 (+0,252) | 0,986 (+0,710) |
| **Modelo Multi-Dataset (Misto)** | **96,5%** | **0,668** | **0,986** |
| **Multi-Dataset com Threshold Calibrado** | - | - | **TPR 96,6% / FPR 9,0%** |

## 4. Conclusões e Recomendações
1. **Robustez Acústica:** A incorporação de ruídos reais de múltiplos microfones impede que o modelo colapse diante de ruídos elétricos ou novos ambientes de gravação.
2. **Calibração de Limiar:** Em novos microfones, o limiar padrão de 0,5 causa aumento no alarme falso; o uso de limiares calibrados restaura a alta especificidade (FPR < 10%).
3. **Resolução Taxonômica:** O modelo separa com precisão o gênero *Aedes* de outros insetos e ruídos, mas a separação fina entre *Ae. aegypti* e *Ae. albopictus* continua dependente de *priors* biogeográficos e compensação térmica.
