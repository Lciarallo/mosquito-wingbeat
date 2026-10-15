"""Recalculate tables and scientific figures from the frozen ternary LSTM run."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from train_ternary_lstm import frontend

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'results/ternary_lstm_2026_10_15'
OUT = ROOT / 'output/ifsc_ternario_lstm_2026_10_15'
ASSETS = OUT / 'assets'


def main():
    ASSETS.mkdir(exist_ok=True)
    metrics = json.loads((RUN / 'metrics.json').read_text())
    rows = [dict(metrics['reference'], label='Cascata clássica', decision='cascata')]
    names = {'lstm': 'LSTM', 'bilstm': 'Bi-LSTM'}
    for r in metrics['neural_models']:
        rows.append(dict(r, label=names[r['model']] + (' · direta' if r['decision'] == 'argmax' else ' · calibrada')))
    selected = ['label', 'accuracy', 'balanced_accuracy', 'aedes_recall', 'aedes_precision',
                'noise_to_aedes_fpr', 'other_to_aedes_fpr', 'noise_to_mosquito_fpr']
    pd.DataFrame(rows)[selected].to_csv(RUN / 'comparison.csv', index=False)
    font_manager.fontManager.addfont(str(ROOT / 'trabalho_ifsc/assets/Oswald.ttf'))
    plt.rcParams.update({'font.family': ['Oswald', 'DejaVu Sans'], 'font.size': 23,
        'axes.titlesize': 25, 'axes.labelsize': 23, 'xtick.labelsize': 21, 'ytick.labelsize': 23,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#53615A',
        'text.color': '#17241B', 'axes.labelcolor': '#17241B', 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'axes.axisbelow': True, 'grid.color': '#E2E9E4'})
    green, gold, teal = '#278333', '#E5AA00', '#087C83'
    def save(f, name):
        f.savefig(ASSETS / name, dpi=180)
        plt.close(f)
    def pct(v):
        return f'{100*v:.2f}%'.replace('.', ',')

    frame = pd.read_csv(RUN / 'segments.csv.gz', low_memory=False)
    row = frame.loc[frame.true_label.eq(2)].iloc[0]
    wave = np.array(np.load(ROOT / 'data/cache/reliability_2026_10_01/long_8k.npy', mmap_mode='r')[int(row.wave_index)])
    # First retained Aedes window: the illustration is selected independently of prediction success.
    (ASSETS / 'audio-example.json').write_text(json.dumps(dict(path=row.path, wave_index=int(row.wave_index),
        start_s=float(row.start_s), duration_s=.992, selection='first retained positive row; not selected by model score'), indent=2))
    t = np.arange(len(wave))/8000
    f, axs = plt.subplots(1, 2, figsize=(12, 5.3))
    axs[0].plot(t, wave, lw=.8, color=green)
    axs[0].set(xlabel='Tempo (s)', ylabel='Amplitude', xlim=(0, .992), title='Janela real · 0,992 s')
    spec = np.abs(np.fft.rfft((wave-wave.mean())*np.hanning(len(wave))))
    axs[1].plot(np.fft.rfftfreq(len(wave), 1/8000), 20*np.log10(np.maximum(spec/spec.max(), 1e-4)), color=teal, lw=1.3)
    axs[1].set(xlabel='Frequência (Hz)', ylabel='Nível relativo (dB)', xlim=(80, 2500), ylim=(-80, 2), title='Espectro do trecho', xticks=[500,1000,1500,2000])
    for a in axs: a.grid()
    f.subplots_adjust(left=.08, right=.98, bottom=.20, top=.86, wspace=.3)
    save(f, 'onda_fft.png')

    with torch.no_grad():
        v = torch.tensor(wave[None], dtype=torch.float32)
        v = v-v.mean(-1, keepdim=True)
        v = v/(v.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
        s = torch.stft(v, 1024, 128, 1024, torch.hann_window(1024), center=False, return_complex=True).abs().square()[0].numpy()
        mel = frontend()(torch.tensor(wave[None], dtype=torch.float32))[0].numpy()
    db = np.clip(10*np.log10(np.maximum(s, 1e-10)/max(s.max(), 1e-10)), -80, 0)
    f, a = plt.subplots(figsize=(12, 5.3))
    im = a.imshow(db, origin='lower', aspect='auto', extent=[.064, .928, 0, 4000], cmap='magma', vmin=-80, vmax=0)
    a.set(xlabel='Centro da janela (s)', ylabel='Frequência (Hz)', ylim=(80, 2500))
    f.colorbar(im, ax=a, label='Potência relativa (dB)', pad=.025)
    f.subplots_adjust(left=.10, right=.95, bottom=.19, top=.97)
    save(f, 'stft.png')
    f, a = plt.subplots(figsize=(12, 5.3))
    im = a.imshow(mel.T, origin='lower', aspect='auto', extent=[.064, .928, 0, 64], cmap='magma', vmin=-80, vmax=0)
    a.set(xlabel='Tempo (s) · 55 passos', ylabel='Filtro mel · 80–4.000 Hz')
    f.colorbar(im, ax=a, label='Potência relativa (dB)', pad=.025)
    f.subplots_adjust(left=.10, right=.95, bottom=.19, top=.97)
    save(f, 'mel.png')

    groups = frame.groupby('group').agg(label=('true_label', 'max'), fold=('outer_fold', 'first'))
    counts = groups.label.value_counts().reindex([0, 1, 2]).to_numpy()
    f, axs = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={'width_ratios': [1, 1.15]})
    axs[0].barh(['Ruído', 'Outro mosquito', 'Aedes aegypti'], counts, color=[teal, green, gold], height=.55)
    for y, n in enumerate(counts): axs[0].text(n+20, y, str(n), va='center')
    axs[0].set(xlim=(0, 1180), xlabel='Grupos avaliados'); axs[0].invert_yaxis()
    folds = pd.crosstab(groups.fold, groups.label).reindex(columns=[0, 1, 2], fill_value=0)
    left = np.zeros(5)
    for c, color, name in zip([0, 1, 2], [teal, green, gold], ['Ruído', 'Outro', 'Aedes']):
        axs[1].barh(np.arange(5), folds[c], left=left, color=color, label=name, height=.6)
        left += folds[c].to_numpy()
    axs[1].set(yticks=np.arange(5), yticklabels=[f'Dobra {i+1}' for i in range(5)], xlabel='Grupos na dobra de teste', xlim=(0, 360))
    axs[1].invert_yaxis()
    handles, labels = axs[1].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper center', bbox_to_anchor=(.73, 1), ncol=3, fontsize=18,
             frameon=False, columnspacing=.9, handlelength=1.3)
    f.subplots_adjust(left=.20, right=.98, bottom=.19, top=.86, wspace=.55)
    save(f, 'populacao.png')

    f, axs = plt.subplots(1, 2, figsize=(12, 5.3))
    for a, name in zip(axs, ['lstm', 'bilstm']):
        h = pd.DataFrame(json.loads((RUN / f'final/{name}/inner_history.json').read_text()))
        best = json.loads((RUN / f'final/{name}/selection.json').read_text())['best_epoch']
        a.plot(h.epoch, h.train_loss, color=green, lw=2.7, label='Treino · janelas')
        a.plot(h.epoch, h.validation_group_loss, color=gold, lw=2.7, label='Validação · grupos')
        a.axvline(best, color=teal, ls='--', lw=1.8)
        a.set(xlabel='Época', ylabel='Entropia cruzada', title=f'{names[name]} · época {best}', ylim=(0, 1.2))
        a.set_xticks([1, best, int(h.epoch.max())]); a.grid()
    handles, labels = axs[0].get_legend_handles_labels()
    f.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .0), ncol=2, frameon=False, fontsize=22)
    f.subplots_adjust(left=.09, right=.98, bottom=.30, top=.85, wspace=.3)
    save(f, 'aprendizado.png')

    f, a = plt.subplots(figsize=(12, 5.3))
    y = np.arange(4); neural = rows[1:]
    a.barh(y, [r['accuracy']*100 for r in neural], color=[green, teal, green, teal], height=.58)
    for yy, r in zip(y, neural): a.text(r['accuracy']*100+1, yy, pct(r['accuracy']), va='center', fontsize=25,
        zorder=5, bbox={'facecolor':'white','edgecolor':'none','pad':.4})
    a.axvline(rows[0]['accuracy']*100, color=gold, lw=2.5, ls='--', label='Cascata clássica · 90,03%')
    a.set(yticks=y, yticklabels=[r['label'] for r in neural], xlim=(0, 103), xlabel='Acurácia geral por grupo (%)')
    a.invert_yaxis(); a.grid(axis='x')
    a.legend(loc='upper center', bbox_to_anchor=(.5, 1.20), frameon=False, fontsize=25)
    f.subplots_adjust(left=.28, right=.97, bottom=.20, top=.80)
    save(f, 'acuracia.png')

    f, a = plt.subplots(figsize=(12, 5.3))
    cm = np.array(metrics['reference']['confusion_matrix_raw'])
    norm = cm/cm.sum(1, keepdims=True)
    a.imshow(norm, cmap='Greens', vmin=0, vmax=1, aspect='auto')
    for i in range(3):
        for j in range(3):
            a.text(j, i, f'{cm[i,j]}\n{100*norm[i,j]:.1f}%'.replace('.', ','), ha='center', va='center',
                   fontsize=29, color='white' if norm[i,j]>.5 else '#17241B')
    labels = ['Ruído', 'Outro mosquito', 'Aedes aegypti']
    a.set(xticks=range(3), xticklabels=labels, yticks=range(3), yticklabels=labels, xlabel='Classe prevista', ylabel='Classe real')
    a.tick_params(length=0); f.subplots_adjust(left=.25, right=.95, bottom=.23, top=.97)
    save(f, 'confusao.png')

    table = '| Modelo / decisão | Acurácia geral | Sensibilidade Aedes | Ruído → Aedes | Outro → Aedes |\n|---|---:|---:|---:|---:|\n'
    for r in rows:
        table += '| ' + r['label'] + ' | ' + ' | '.join(pct(r[k]) for k in ['accuracy','aedes_recall','noise_to_aedes_fpr','other_to_aedes_fpr'])+' |\n'
    report = '''# Comparação LSTM/Bi-LSTM para o detector ternário — 15/10/2026

Autor: Luiz Eduardo do Nascimento Ciarallo.

## Resultado

A cascata LR + HistGB mantém 90,03% de acurácia geral. Nas configurações avaliadas, LSTM e Bi-LSTM não mantiveram esse valor. A decisão direta das redes aumentou a sensibilidade de Aedes, acompanhada de mais falsos positivos. Os limiares conservadores reduziram falsos positivos e também a sensibilidade. Ambos os modos foram preservados, sem selecionar retrospectivamente apenas o melhor número.

''' + table + '''
Os denominadores são os mesmos: 1.494 grupos, sendo 426 de ruído, 1.008 de outros mosquitos e 60 de Aedes. Os 90,03% equivalem a 1.345/1.494 acertos. Zero ruído → Aedes (0/426) não é zero ruído → qualquer mosquito: na cascata, há 102/426 ruídos classificados como outro mosquito.

## Protocolo executado

Foram treinadas redes ternárias novas, sem importar pesos ou rótulos de infecção: duas camadas LSTM de 128 unidades, unidirecionais ou bidirecionais, dropout 0,20, camada linear de três saídas e softmax na inferência. A arquitetura é uma adaptação do artigo de Torres et al. (2025), que utiliza 256 unidades por camada em outro problema.

As 42.029 janelas de 0,992 s são exatamente as retidas no estudo clássico auditado. Os cinco testes externos usam as mesmas linhas e as mesmas partições; 1.441 blocos de PCM não atravessam treino e teste. Grupos com rótulos de janela mistos recebem o maior rótulo, conforme o contrato anterior. MosquitoSong permanece fora de todo desenvolvimento.

O frontend usa áudio de 8 kHz, FFT 1.024, Hann de 128 ms, avanço de 16 ms, 64 filtros mel de 80 a 4.000 Hz e 55 passos temporais. A separação entre bins é 7,8125 Hz. Não há recuperação de informação acima do Nyquist do áudio de origem. DC/RMS por janela, potência relativa em dB limitada a [-80,0], média e desvio por banda calculados somente no treino de cada ajuste.

O treino usa Adam, taxa 0,0006, lotes de 256, entropia cruzada, clipping da norma do gradiente em 3 e amostragem composta com gamma 0,50, teto de 60 janelas e massa equilibrada entre as três classes. Limite de 30 épocas e paciência de 5.

Dentro de cada treino externo, uma partição interna agrupada (primeira divisão de StratifiedGroupKFold com 4 partes) escolhe a época pela perda de grupos equilibrada por classe. Os limiares são ajustados exclusivamente nessa validação interna, com tetos empíricos de 0,75% ruído → Aedes e 1,75% outro → Aedes. Depois o modelo e o normalizador são reajustados em todo o treino externo pelo número de épocas escolhido. Os testes externos são aplicados após congelar os arquivos.

A seleção interna das redes é um holdout agrupado, enquanto a referência clássica usa calibração interna OOF mais ampla. A população e as cinco dobras externas são idênticas, mas os processos internos de seleção não são idênticos. Os limites internos de FPR não garantem limites populacionais.

A agregação por grupo pondera cada janela por (1-p_ruído)^1,5. Foram salvas as probabilidades OOF, a decisão argmax e a decisão calibrada. As métricas principais são determinísticas. A inferência final também implementa 100 passagens de Monte Carlo dropout; a discordância de decisões é um indicador de estabilidade da rede, sem garantia de correção. O exemplo de áudio salvo é apenas uma verificação de execução, não um teste independente.

O modelo neural final foi escolhido pela validação interna: LSTM, época 4; Bi-LSTM selecionou época 2. Os dois candidatos e todas as dobras foram preservados. A LSTM final tem 231.811 parâmetros e a Bi-LSTM 594.691.

## Verificações

O verificador reproduziu todas as 42.029 probabilidades de cada arquitetura nas cinco dobras, com diferença máxima zero; confirmou 24 normalizadores calculados apenas no treino, hashes dos pesos/fontes, partições e a seleção sem uso do teste externo. A suíte completa passou: 313 testes. A inferência de um WAV real e as 100 passagens de dropout também foram executadas.

Comandos de reprodução:

```bash
.venv/bin/python train_ternary_lstm.py
.venv/bin/python verify_ternary_lstm.py results/ternary_lstm_2026_10_15
.venv/bin/python report_ternary_lstm.py
.venv/bin/python -m pytest -q
```

O treinador recusa sobrescrever uma execução existente. Para refazer, escolha um diretório novo com --output. Evidências em results/ternary_lstm_2026_10_15: protocol.json, metrics.json, comparison.csv, verification.json, freeze.json, pesos, históricos, normalizadores e previsões de cada dobra.

## Discussão final

O resultado positivo da cascata é mantido e atribuído corretamente. O aumento de sensibilidade das redes diretas revela um compromisso com os falsos positivos; esta rodada não demonstrou vantagem de acurácia geral das redes. A avaliação é retrospectiva em um corpus conhecido, sem ablação causal de cada técnica. Novos indivíduos, sessões, microfones e ambientes continuam sendo o próximo teste de generalização. O objetivo apresentado é detecção ternária, sem inferência de dengue e sem afirmar execução embarcada.

Referência: Torres et al. (2025), https://latamt.ieeer9.org/index.php/transactions/article/view/9221 ; implementação pública https://github.com/NakanoMariko/Dengue-Infected-Mosquito-Detection .
'''
    (ROOT / 'reports/lstm_ternaria_2026_10_15.md').write_text(report, encoding='utf-8')
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ASSETS.iterdir() if p.is_file()}
    (OUT / 'figure-provenance.json').write_text(json.dumps(dict(run=str(RUN), first_audio_row=int(row.row_id),
        frontend=frontend().config, assets_sha256=manifest), ensure_ascii=False, indent=2))
    print('Scientific figures and measured report ready:', len(manifest), 'assets')


if __name__ == '__main__':
    main()
