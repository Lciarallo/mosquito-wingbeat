"""Produz PDF e HTML com os fundos autênticos do modelo IFSC e dados calculados.

Uso: .venv/bin/python trabalho_ifsc/preparar_material.py --assets /tmp/mosquito-ifsc-20261011
O PPTX oficial é preservado; este script não altera arquivos PowerPoint.
"""
from __future__ import annotations
import argparse
import base64
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.pagesizes import A4
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'results/ifsc_cnn_2026_10_11'
IMPROVED = ROOT / 'results/ifsc_melhorias_2026_10_11'
OUT = ROOT / 'output/ifsc_2026_10_11'
GREEN = '#287f35'
YELLOW = '#e8b000'
IFSC = 'https://www.ifsc.edu.br/en/documentos-uteis'


def pct(v):
    return f'{100*v:.1f}%'.replace('.', ',')


def plots(assets):
    directory = assets / 'graficos'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})
    history = pd.read_json(RESULT / 'history.json')
    fig, ax = plt.subplots(figsize=(9.3, 4.1), constrained_layout=True)
    ax.plot(history.epoch, history.train_loss, color=GREEN, marker='o', label='Treino')
    ax.plot(history.epoch, history.validation_loss, color=YELLOW, marker='s', label='Validação')
    epoch = json.loads((RESULT / 'freeze.json').read_text())['best_epoch']
    ax.axvline(epoch, color='#777', ls='--', label=f'Época escolhida: {epoch}')
    ax.set(xlabel='Época', ylabel='Perda (entropia cruzada)')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.18), frameon=False, fontsize=12)
    ax.grid(axis='y', alpha=.16)
    fig.savefig(directory / 'aprendizado.png', dpi=170)
    plt.close(fig)
    metrics = json.loads((RESULT / 'metrics.json').read_text())
    fig, ax = plt.subplots(figsize=(9.3, 3.8), constrained_layout=True)
    values = [m['recording_accuracy'] * 100 for m in metrics]
    ax.barh(['CNN: quatro convoluções', 'Modelo linear'], values, color=[GREEN, YELLOW], height=.5)
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel='Gravações corretas (%)')
    for i, m in enumerate(metrics):
        ax.text(values[i]+2, i, f"{pct(m['recording_accuracy'])}\n{m['correct_recordings']}/190", va='center', fontsize=15)
    ax.grid(axis='x', alpha=.16)
    fig.savefig(directory / 'comparacao.png', dpi=170)
    plt.close(fig)
    improved = json.loads((IMPROVED / 'metrics.json').read_text())
    fig, ax = plt.subplots(figsize=(9.3, 3.8), constrained_layout=True)
    values = [metrics[0]['recording_accuracy']*100] + [r['recording_accuracy']*100 for r in improved] + [metrics[1]['recording_accuracy']*100]
    labels = ['CNN inicial', 'CNN: frequência preservada', 'MLP profunda', 'Referência linear']
    ax.barh(labels, values, color=['#999999', GREEN, '#176f70', YELLOW], height=.58)
    ax.invert_yaxis(); ax.set(xlim=(0,100), xlabel='Gravações corretas (%)')
    for i,v in enumerate(values):
        ax.text(v+2,i,f'{v:.1f}%'.replace('.',','),va='center',fontsize=15)
    ax.grid(axis='x',alpha=.16)
    fig.savefig(directory / 'melhorias.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.3, 3.8), constrained_layout=True)
    x = np.arange(2)
    a, b = [96.49, 71.70], [41.18, 23.53]
    ax.bar(x-.18, a, .34, color=GREEN, label='Corpus conhecido: 569 gravações')
    ax.bar(x+.18, b, .34, color=YELLOW, label='Externo: 17 arquivos')
    for dx, vals in [(-.18, a), (.18, b)]:
        for i, val in enumerate(vals):
            ax.text(i+dx, val+2, f'{val:.1f}%'.replace('.', ','), ha='center', fontsize=15)
    ax.set(xticks=x, xticklabels=['Gênero', 'Espécie'], ylim=(0, 116), ylabel='Acerto por gravação (%)')
    ax.legend(loc='upper center', fontsize=11, frameon=False)
    ax.grid(axis='y', alpha=.12)
    fig.savefig(directory / 'genero_especie.png', dpi=170)
    plt.close(fig)
    frame = pd.read_csv(ROOT / 'results/expanded/segments.csv', low_memory=False)
    # Exemplo definido por classe/partição e ordem original, sem busca por aparência.
    row = frame.loc[frame.domain.eq('dryad') & frame.label.eq('Aedes aegypti') & frame.split.eq('validation')].iloc[0]
    wave = np.load(ROOT / 'data/cache/large_cnn/waves.npy', mmap_mode='r')[row.name]
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 4.7), constrained_layout=True,
                             gridspec_kw={'height_ratios': [1, 2]})
    axes[0].plot(np.arange(len(wave))/16000, wave, color=GREEN, lw=.6)
    axes[0].set(ylabel='Amplitude\nnormalizada', xlim=(0, .992))
    f, t, z = stft(wave, fs=16000, nperseg=2048, noverlap=1792, boundary=None, padded=False)
    db = 10*np.log10(np.abs(z)**2 + 1e-12)
    im = axes[1].pcolormesh(t, f, db-db.max(), cmap='magma', vmin=-70, vmax=0, shading='auto')
    axes[1].set(ylim=(80, 4000), xlim=(0, .992), xlabel='Tempo (s)', ylabel='Frequência (Hz)')
    fig.colorbar(im, ax=axes[1], label='dB relativos', pad=.01)
    fig.savefig(directory / 'som_espectrograma.png', dpi=170)
    plt.close(fig)
    (assets / 'exemplo_audio.txt').write_text(f"Exemplo real de validação: {row.path}; início {row.start_s}s; rótulo do corpus: {row.label}.\n")
    return directory


def contents():
    m = json.loads((RESULT / 'metrics.json').read_text())
    cnn, linear = m
    transfer = json.loads((RESULT / 'transfer_metrics.json').read_text())
    freeze = json.loads((RESULT / 'freeze.json').read_text())
    improved = json.loads((IMPROVED / 'metrics.json').read_text())
    selection = json.loads((IMPROVED / 'selection.json').read_text())
    slides = []
    def add(title, body, note, source, seconds=55, image=None, caption=None, rows=None):
        slides.append(dict(title=title, body=body, note=note, source=source, seconds=seconds,
                           image=image, caption=caption, rows=rows))
    add('Mosquitos\npelo som', ['Redes neurais para classificação acústica',
          'Luiz Eduardo do Nascimento Ciarallo', 'Computação Científica para a Indústria | IFSC'],
        'Vou apresentar um projeto que tenta identificar mosquitos pelo som do voo. O trabalho usa séries temporais de áudio e uma rede neural profunda. Minha pergunta é se padrões aprendidos em gravações conhecidas continuam úteis quando o som vem de outra origem. Vou mostrar a implementação, os resultados e as dificuldades, incluindo resultados que não foram melhores que os de um método mais simples. O projeto tem avaliação computacional; a identificação confiável em campo ainda não foi demonstrada.',
        'Modelo institucional IFSC; enunciado Trabalho_Descrição.pdf.', 30)
    add('Por que analisar esse som?', [
        'O voo produz padrões acústicos que podem ser analisados.',
        'Objetivo: apoiar o monitoramento e a identificação de mosquitos.',
        'Pergunta: a rede reconhece a espécie em outra gravação?'],
        'A aplicação é monitoramento acústico de mosquitos. A literatura investiga o som das asas como uma alternativa para apoiar a identificação automática. O interesse está em analisar uma série temporal com custo de aquisição relativamente baixo. Isso não significa que ouvir um som com o celular já permita identificar corretamente a espécie. Minha pergunta de pesquisa é mais específica: depois de aprender com uma base, o modelo consegue reconhecer sons de outra origem? Essa diferença entre aprender o conjunto e generalizar é o ponto central do trabalho.',
        'Mukundarajan et al., 2017, DOI 10.7554/eLife.27854; Supratak et al., 2024.', 55)
    add('Três perguntas diferentes', [
        'Presença: há mosquito ou apenas ruído?',
        'Gênero: Aedes, Anopheles, Culex ou Culiseta?',
        'Espécie: por exemplo, Aedes aegypti.'],
        'Essas tarefas precisam ficar separadas. Presença significa distinguir mosquito de ausência ou ruído. Gênero identifica uma categoria taxonômica mais ampla. Espécie exige uma distinção mais fina. A CNN nova deste trabalho resolve escolha de espécie entre vinte classes, supondo que a entrada seja de mosquito. Ela sempre escolhe uma classe e não tem um detector de ausência validado. Se eu passar ruído, também posso receber o nome de uma espécie. Portanto, um classificador de espécies sozinho não é um detector de mosquito. No projeto existem experiências distintas para as três perguntas.',
        'Repo: reports/tres_niveis_2026_10_09.md; novo experimento: protocol.json.', 60)
    add('O som vira uma imagem', [],
        'Este exemplo foi calculado a partir de uma janela real rotulada como Aedes aegypti no Dryad. Acima está o sinal ao longo do tempo; abaixo, o espectrograma, que mostra a distribuição de energia entre frequências. Faixas horizontais podem indicar componentes relativamente estáveis do voo. A rede recebe uma representação log-mel e um canal de contraste espectral, não o nome do arquivo. Essas representações tornam padrões locais de frequência e tempo acessíveis às convoluções. A janela do experimento tem 0,992 segundo a 16 kHz. O gráfico é uma transformação do áudio real, não uma imagem ilustrativa.',
        'Dryad, DOI 10.5061/dryad.98d7s; áudio real do cache; large_audio_model.py.',
        65, 'som_espectrograma.png', 'Exemplo real: Aedes aegypti, Dryad | janela de 0,992 s'),
    add('Abordagens e trabalhos correlatos', [
        'FFT e descritores: resumir frequências e padrões do sinal.',
        'CNN em espectrograma: aprender padrões de tempo e frequência.',
        'MosquitoSong+: CNN 1D em áudio, com ruído no treinamento.'],
        'As abordagens mais simples usam a frequência fundamental ou descritores acústicos, com classificadores como SVM e árvores. Uma CNN pode receber o espectrograma e aprender características, em vez de depender apenas de uma lista manual de descritores. O trabalho de Fanioudakis e colaboradores é uma referência de aprendizado profundo nesse domínio. MosquitoSong+ usa convoluções 1D no áudio e aumento de dados com ruído e variação de volume. Essas pesquisas orientam o projeto, mas seus percentuais não podem ser comparados diretamente com os meus: espécies, dados, partições e unidades de avaliação diferem. Eu não refiz integralmente o treinamento desses artigos nesta rodada.',
        'Fanioudakis et al., 2018, DOI 10.23919/EUSIPCO.2018.8553542; Supratak et al., 2024.', 65)
    add('Os dados não são todos iguais', [
        'Dryad: áudio de celulares; 20 espécies no estudo.',
        'HumBug: outra coleção e diferentes condições de captura.',
        'Áudio externo: serve para investigar transferência.'],
        'Dryad é uma base acústica de celulares, não uma base óptica. Para a CNN nova usei as vinte espécies e as partições existentes dessa coleção. O projeto também contém HumBug e gravações públicas de outras origens. Essas fontes têm distribuições diferentes de espécies, equipamentos e condições. Muitas janelas podem vir da mesma gravação, por isso milhares de janelas não representam milhares de mosquitos independentes. Na avaliação histórica da CNN maior, o subconjunto de teste HumBug tinha quatro classes, enquanto Dryad tinha vinte. Isso é uma diferença importante para interpretar os números.',
        'Repo: reports/modelo_grande_cnn.md; results/ifsc_cnn_2026_10_11/partitions.csv.', 55)
    add('O programa que implementei', [
        '1. Converter cada janela em log-mel e contraste espectral.',
        '2. Treinar a CNN usando rótulos das espécies.',
        '3. Selecionar a melhor época pela validação.',
        '4. Congelar os pesos e avaliar o teste.'],
        'Para atender ao trabalho, implementei e executei uma CNN didática nova em PyTorch. O fluxo começa com janelas reais de áudio e usa um frontend fixo, com a mesma normalização individual no treinamento e na inferência. A validação escolhe a época; o teste não participa dessa escolha. Depois os pesos são salvos e congelados antes de calcular as métricas do teste. O programa também tem um modo de inferência para um arquivo WAV. O código, os pesos, as curvas e as previsões por gravação acompanham o material. Essa implementação fica separada dos modelos históricos do repositório.',
        'Código novo: trabalho_ifsc/experimento_cnn.py; execução de 11/10/2026.', 60)
    add('A arquitetura da primeira CNN', [
        'Entrada: 2 canais, 64 bandas e 55 quadros de tempo.',
        '4 convoluções: 16, 32, 64 e 96 canais.',
        'Camada de 128 unidades; saída para 20 espécies.',
        '130.900 parâmetros treináveis.'],
        'Uma convolução aprende filtros que respondem a padrões locais. As primeiras camadas podem captar estruturas simples, e as seguintes combinam essas respostas. Cada bloco tem convolução, normalização, ReLU e redução de resolução. Depois uma camada densa de 128 unidades combina a representação, e a saída produz escores para vinte espécies. São quatro convoluções aprendidas e duas camadas lineares, por isso a proposta é uma rede profunda. O dropout atua no treinamento como regularização. O softmax organiza os escores entre classes, mas não oferece garantia de acerto fora da distribuição usada no estudo.',
        'CNNDidatica em experimento_cnn.py; protocol.json: 130900 parâmetros.', 65)
    add('Separar dados antes de treinar', [],
        'Treino ajusta os pesos; validação seleciona a época; teste avalia o modelo congelado. Mantive a partição de calibração existente reservada, sem usá-la, porque este experimento não ajusta limiares nem rejeição. O código confere a separação por gravação, caminho e hashes de janelas declarados nos metadados. Também verifica a correspondência das linhas entre os arquivos de metadados e o cache. Isso reduz vazamento direto de dados, mas não prova independência entre indivíduos ou sessões, porque essa informação é incompleta. A validação possui dezenove das vinte espécies. O teste é histórico e já havia orientado pesquisas anteriores: esta avaliação é retrospectiva.',
        'protocol.json, audit.json e partitions.csv; partições históricas preservadas.',
        70, rows=[['Partição', 'Gravações', 'Janelas'], ['Treino', '233', '9.873'],
                 ['Validação', '63', '2.735'], ['Calibração reservada', '83', '3.424'], ['Teste interno', '190', '7.845']],
        caption='A mesma gravação permanece em uma única partição.'),
    add('Primeira CNN: sinais de sobreajuste', [],
        f'O treinamento executou doze épocas e selecionou a época {freeze["best_epoch"]}, com menor perda de validação. A curva de treinamento cai continuamente, mas a validação deixa de acompanhar e oscila. Isso é compatível com sobreajuste: o modelo se adapta ao treino sem melhorar da mesma forma em gravações separadas. Não escolhi a última época só porque ela aprende mais o treino. A função de seleção da validação equilibra espécies e gravações, enquanto a média da perda de treino vem de amostragem balanceada; as curvas são úteis para acompanhar a evolução, mas não têm exatamente o mesmo esquema de agregação. O resultado não justifica escolher uma época pelo teste.',
        'Nova execução: history.json e freeze.json; perdas de treino e validação.',
        65, 'aprendizado.png', f'Época {freeze["best_epoch"]} selecionada pela validação; parada antecipada.'),
    add('Resultado da primeira rodada', [],
        f'No mesmo teste interno, a CNN acertou {cnn["correct_recordings"]} de 190 gravações, ou {pct(cnn["recording_accuracy"])}. O modelo linear acertou {linear["correct_recordings"]}, ou {pct(linear["recording_accuracy"])}. A previsão de uma gravação é a média das probabilidades das suas janelas. O modelo linear usa média e desvio temporal dos mesmos canais espectrais, padronização aprendida só no treino e regressão logística. Essa é uma referência simples e explícita. Ele foi melhor neste protocolo. O acerto balanceado da CNN foi {pct(cnn["recording_balanced_accuracy"])} e seu F1 macro foi {pct(cnn["recording_macro_f1"])}. A conclusão é que a implementação neural funciona como programa, mas não demonstrou vantagem sobre a referência linear.',
        'Nova execução: metrics.csv, cnn_recordings.csv e linear_recordings.csv.',
        65, 'comparacao.png', 'Mesmas 190 gravações; escolha forçada entre 20 espécies.'),
    add('Alterações melhoraram as redes', [],
        'Testei duas alterações com as mesmas partições. A CNN revista preserva dezesseis posições de frequência, em vez de quatro, e tem 224.036 parâmetros. Isso mantém mais informação sobre onde estão os padrões espectrais. A MLP profunda usa média e desvio temporal dos mesmos canais, com três camadas ocultas de 256, 128 e 64 unidades e 109.140 parâmetros. A padronização foi ajustada apenas no treino. Também aumentei dropout e penalização dos pesos, reduzi a taxa de aprendizado e usei seleção por gravação. Os dois modelos acertaram 132 de 190 gravações, ou 69,5%, contra 105 da primeira CNN: mais 27 acertos e 14,2 pontos percentuais. A MLP foi escolhida pela menor perda de validação, antes de olhar o teste. Ela teve F1 macro de 77,1% e acerto balanceado de 77,4%. A referência linear ainda teve acerto global ligeiramente maior. Esse ganho interno não demonstra melhoria garantida em novas coletas.',
        'Nova rodada: results/ifsc_melhorias_2026_10_11/; seleção pela validação.',
        65, 'melhorias.png', 'Mesmas 190 gravações; MLP escolhida pela validação, antes do teste.'),
    add('Transferência ainda é difícil', [
        'Redes revistas: 3/17 arquivos corretos (17,6%).',
        'Primeira CNN: 0/17 arquivos corretos.',
        'Cinco espécies; outra origem de gravação.',
        'Avaliação retrospectiva; sem controles de ausência/ruído.'],
        'Depois de congelar os pesos, avaliei as duas redes revistas nos mesmos dezessete arquivos públicos usados anteriormente. Cada uma acertou três arquivos, contra zero da CNN inicial. A melhora existe nesses arquivos, mas dezessete é um conjunto pequeno e conhecido da pesquisa. Esses arquivos não entraram no treinamento das redes, porém já tinham sido estudados, então não vou chamar isso de teste cego ou nova coleta independente. A ancestralidade dos indivíduos e sessões também é incompleta; dezessete arquivos não significam dezessete mosquitos. Cada rede acertou dois arquivos de Anopheles minimus e um de Anopheles stephensi; não acertou Aedes nem Culex nesse conjunto. Como só há mosquito, esse conjunto não mede especificidade nem falsos alarmes de presença.',
        'Rodada revista: external_metrics.csv; BioDCASE D3, zenodo.org/records/19095788.', 60)
    add('O projeto também tem uma CNN maior', [
        'CNN residual: 3,33 milhões de parâmetros.',
        'Dryad: 62,7% de acerto por janela, 20 classes.',
        'HumBug: 89,0% de acerto por janela, 4 classes.',
        'Resultados históricos; protocolos diferentes da CNN nova.'],
        'O repositório já contém uma CNN residual maior, treinada anteriormente com Dryad e HumBug. Ela tem aproximadamente 3,33 milhões de parâmetros e duas saídas: espécie e presença. Os números aqui são por janela: 62,7% em Dryad e 89% no subconjunto HumBug. Não compare esses percentuais diretamente com os 55,3% da CNN nova, que apresentei por gravação. Os dados de treinamento e o número de classes do teste também diferem. Na CNN maior, o acerto nas identificações emitidas é superior, mas a cobertura cai: por exemplo, ela emite identificação em cerca de 31% das janelas conhecidas Dryad. Rejeitar entradas pode aumentar o acerto das emissões sem resolver todas as gravações.',
        'Histórico: reports/modelo_grande_cnn.md; results/large_cnn/metrics.csv.', 65)
    add('Uma rede maior ainda falhou', [
        'Aedes externo: 0/4 janelas com a espécie correta no topo.',
        'Enxame informado como Aedes: 10 emissões de Anopheles.',
        'Microfone, ambiente e sons sobrepostos mudam a entrada.'],
        'Na rodada histórica da CNN maior, uma gravação curta de Aedes aegypti de outro indivíduo e outra origem foi selecionada depois do congelamento. Nenhuma das quatro janelas colocou Aedes como primeira classe. Isso é uma falha daquele caso, não uma estimativa da acurácia de todas as gravações externas. No áudio privado que informei como enxame de Aedes, houve dez emissões de Anopheles gambiae. Pela referência informada, essas emissões são incorretas, mas não houve confirmação entomológica independente nem anotação de cada voo. O áudio privado não acompanha o pacote. Esses exemplos mostram por que bons números internos e mais parâmetros não bastam para demonstrar generalização.',
        'Histórico: reports/modelo_grande_cnn.md; referência do enxame informada pelo usuário.', 60)
    add('Gênero é mais fácil que espécie?', [],
        'Este gráfico vem de outro experimento: o ensemble clássico corrigido de 09/10, e não da CNN nova. No corpus conhecido, ele acertou 549 de 569 gravações no gênero e 408 na espécie. Fora desse corpus, foram sete de dezessete no gênero e quatro na espécie. Os números sugerem que o gênero foi mais fácil naquele cenário, mas a queda externa permaneceu. Gênero foi calculado somando probabilidades das espécies do mesmo gênero, sem treinamento dedicado. Não existe nesta comparação um sistema único com presença, gênero e espécie validados em sequência. O detector histórico de presença é separado e apresentou muitos falsos positivos.',
        'Ensemble clássico, 09/10: reports/tres_niveis_2026_10_09.md.',
        65, 'genero_especie.png', 'Experimento histórico distinto: ensemble clássico, escolha forçada.'),
    add('Desafios que estou enfrentando', [
        'Generalização: trocar o microfone muda o padrão observado.',
        'Ambiguidade: espécies podem ter frequências parecidas.',
        'Ruído e enxames: vários sons se sobrepõem.',
        'Detector histórico: 29,2% de falsos positivos em janelas de ruído.'],
        'Estou enfrentando dificuldades de dados, de sinal e de validação. A mudança de domínio faz o equipamento e o ambiente produzirem características que o modelo pode confundir com a espécie. Há sobreposição acústica entre espécies, somada às variações biológicas e de gravação. Ruído e múltiplos mosquitos tornam o sinal mais complexo. No detector histórico separado de presença, 29,17% das 528 janelas de ruído, de 44 grupos, acionaram o detector no modo padrão. Sua sensibilidade foi 72,63% em janelas positivas com rótulos fracos. Esses percentuais não são falsos alarmes por hora nem resultados da CNN nova. Também preciso verificar a divisão dos arquivos e os dados usados na calibração. A correção metodológica não resolveu a transferência.',
        'Repo: reports/tres_niveis_2026_10_09.md; auditoria_pesquisa_2026_10_07.md.', 65)
    add('Próximos passos do projeto', [
        'Coletar gravações com espécie confirmada e diferentes aparelhos.',
        'Anotar também períodos sem mosquito e ruídos reais.',
        'Separar indivíduos, sessões e dispositivos na avaliação.',
        'Validar o sistema completo antes de levar à placa.'],
        'O próximo avanço precisa ser nos dados e na avaliação, além da arquitetura. Quero uma coleta com identidade da espécie confirmada, diversidade de microfones, indivíduos, condições e ambiente. Os períodos sem mosquito precisam ser anotados para avaliar falsos alarmes e especificidade. A separação deve considerar indivíduos, sessões e aparelhos, não apenas o nome do arquivo. Depois faz sentido comparar um sistema completo: presença, identificação e rejeição, com métricas por evento ou por gravação e falsos alarmes por hora quando houver anotação adequada. Levar para Arduino é um objetivo de implantação que depende dessa validação e de testes de tempo e memória na placa.',
        'Proposta de continuidade baseada nas limitações registradas no repositório.', 55)
    add('Conclusão do trabalho', [
        'Implementei e comparei redes neurais profundas em áudio real.',
        'Alterações: de 55,3% para 69,5% de acerto por gravação.',
        'O maior desafio continua sendo reconhecer novas condições.',
        'Código, pesos, métricas e demonstração acompanham o material.'],
        'O trabalho atende à proposta de analisar séries temporais de um domínio específico com redes neurais profundas implementadas. A contribuição inclui código executável, pesos treinados, partições e previsões salvas. A primeira CNN acertou 55,3% das gravações; preservar a frequência na CNN e usar uma MLP espectral aumentou para 69,5%. A MLP foi selecionada pela validação. Os resultados revisados ficaram próximos da referência linear, mas ainda não a superaram no acerto global. Os três acertos externos em dezessete arquivos mostram que a dificuldade de generalização permanece. A aplicação continua como pesquisa: confiabilidade em campo precisa ser demonstrada com uma coleta independente.',
        'Síntese da execução nova e dos relatórios históricos, sem validação em campo.', 55)
    add('Referências e materiais', [
        'Mukundarajan et al. (2017): sensores acústicos em celulares.',
        'Fanioudakis et al. (2018): classificação com deep learning.',
        'Supratak et al. (2024): MosquitoSong+.',
        'Repositório mosquito-wingbeat; modelo oficial de slides IFSC.'],
        'As referências completas e os caminhos das métricas ficam no roteiro e no pacote de entrega. O modelo visual foi baixado da página oficial de documentos úteis do IFSC. O estudo distingue resultados dos artigos, experimentos históricos do projeto e o experimento novo executado para este trabalho. Perguntas podem abordar arquitetura, particionamento dos dados, resultados e os próximos passos.',
        'DOIs e links completos no roteiro e no LEIA-ME do pacote.', 20)
    return slides


def wrap(text, font, size, width):
    lines, current = [], ''
    for word in text.split():
        nxt = (current + ' ' + word).strip()
        if pdfmetrics.stringWidth(nxt, font, size) > width and current:
            lines.append(current)
            current = word
        else:
            current = nxt
    return lines + [current]


def draw_text(c, text, x, top, font='OswaldLight', size=21.5, width=555, leading=None):
    lines = wrap(text, font, size, width)
    c.setFont(font, size)
    c.setFillColor('#171717')
    leading = leading or size * 1.32
    for j, line in enumerate(lines):
        c.drawString(x, 405-top-j*leading-size, line)
    return top + len(lines)*leading


def slides_pdf(slides, assets, graph_dir):
    path = OUT / 'Apresentacao_Mosquitos_IFSC_Luiz_Ciarallo.pdf'
    c = canvas.Canvas(str(path), pagesize=(720, 405))
    c.setTitle('Mosquitos pelo som - Luiz Eduardo do Nascimento Ciarallo')
    c.setAuthor('Luiz Eduardo do Nascimento Ciarallo')
    qa = []
    for number, slide in enumerate(slides, 1):
        bg = assets / 'template' / ('image1.jpg' if number == 1 else 'image2.jpg')
        c.drawImage(str(bg), 0, 0, 720, 405)
        if number == 1:
            for i, line in enumerate(slide['title'].split('\n')):
                draw_text(c, line, 20, 45 + i*80, 'OswaldBold', 68, 440, 80)
            for i, text in enumerate(slide['body']):
                draw_text(c, text, 22, [226, 270, 304][i], 'Oswald' if i < 2 else 'OswaldLight', [21.5, 21.5, 14][i], 447)
        else:
            title_width = pdfmetrics.stringWidth(slide['title'], 'Oswald', 31)
            assert title_width <= 555, (number, slide['title'], title_width)
            draw_text(c, slide['title'], 31, 23, 'Oswald', 31)
            if slide['image']:
                c.drawImage(str(graph_dir / slide['image']), 31, 88, 548, 228, preserveAspectRatio=True, anchor='c')
                draw_text(c, slide['caption'], 31, 334, 'OswaldLight', 15, 553)
            elif slide['rows']:
                for r, row in enumerate(slide['rows']):
                    for x, val in zip([34, 339, 465], row):
                        draw_text(c, val, x, 90+r*42, 'Oswald' if r == 0 else 'OswaldLight', 21.5, [290, 115, 110][[34,339,465].index(x)])
                draw_text(c, slide['caption'], 31, 332, 'OswaldLight', 16)
            else:
                y = 100
                for text in slide['body']:
                    y = draw_text(c, text, 32, y, 'OswaldLight', 21.5, 540) + 18
                assert y <= 371, (number, 'overflow', y)
            # Fonte discreta para permitir rastreamento sem carregar a fala.
            draw_text(c, slide['source'], 31, 377, 'OswaldLight', 7.5, 532, 9)
            c.setFont('OswaldLight', 9)
            c.drawRightString(585, 14, f'{number:02d}/{len(slides)}')
        qa.append(dict(slide=number, title=slide['title'], title_fits=True, source=slide['source']))
        c.showPage()
    c.save()
    (assets / 'qa_layout.txt').write_text(json.dumps(qa, ensure_ascii=False, indent=2))
    return path


def encode(path):
    suffix = Path(path).suffix
    mime = {'.jpg':'image/jpeg','.png':'image/png','.ttf':'font/ttf'}[suffix]
    return f'data:{mime};base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def slides_html(slides, assets, graph_dir):
    fonts = '\n'.join(f"@font-face{{font-family:{name};src:url('{encode(assets / (name+'.ttf'))}')}}" for name in ['Oswald','OswaldLight','OswaldBold'])
    sections = []
    for i, slide in enumerate(slides):
        bg = encode(assets / 'template' / ('image1.jpg' if i == 0 else 'image2.jpg'))
        title = html.escape(slide['title']).replace('\n','<br>')
        if i == 0:
            body = ''.join(f'<div class="cover-line l{j}">{html.escape(t)}</div>' for j,t in enumerate(slide['body']))
        elif slide['image']:
            body = f'<img class="plot" alt="{html.escape(slide["caption"])}" src="{encode(graph_dir / slide["image"])}"><div class="caption">{html.escape(slide["caption"])}</div>'
        elif slide['rows']:
            body = '<table>' + ''.join('<tr>'+''.join(f'<td>{html.escape(t)}</td>' for t in row)+'</tr>' for row in slide['rows']) + '</table><div class="caption">'+html.escape(slide['caption'])+'</div>'
        else:
            body = '<div class="body">'+''.join(f'<p>{html.escape(t)}</p>' for t in slide['body'])+'</div>'
        footer = '' if i == 0 else f'<div class="source">{html.escape(slide["source"])}</div><div class="page">{i+1:02d}/{len(slides)}</div>'
        sections.append(f'<section class="slide {"cover" if i == 0 else ""}" style="background-image:url({bg})"><h1>{title}</h1>{body}{footer}<aside>{html.escape(slide["note"])}</aside></section>')
    css = fonts + '''
*{box-sizing:border-box}body{margin:0;background:#202020;color:#171717;font-family:OswaldLight}
#stage{width:720px;height:405px;position:absolute;transform-origin:top left}.slide{display:none;width:720px;height:405px;background-size:100% 100%;position:absolute;inset:0}.slide.active{display:block}
h1{position:absolute;left:31px;top:23px;width:555px;margin:0;font:31px/1.32 Oswald;white-space:nowrap}.body{position:absolute;left:32px;top:100px;width:540px;font:21.5px/1.32 OswaldLight}.body p{margin:0 0 18px}.source{position:absolute;left:31px;top:377px;width:532px;font-size:7.5px;line-height:9px}.page{position:absolute;right:135px;bottom:11px;font-size:9px}.plot{position:absolute;left:31px;top:89px;width:548px;height:228px;object-fit:contain}.caption{position:absolute;left:31px;top:334px;width:553px;font-size:15px;line-height:1.32}
table{position:absolute;left:31px;top:85px;width:548px;border-collapse:collapse;font:21.5px/42px OswaldLight}td{padding:0}td:first-child{width:305px}tr:first-child{font-family:Oswald}
.cover h1{left:20px;top:45px;width:440px;font:68px/80px OswaldBold;white-space:normal}.cover-line{position:absolute;left:22px;width:447px}.l0{top:226px;font:21.5px/1.32 Oswald}.l1{top:270px;font:21.5px/1.32 Oswald}.l2{top:304px;font:14px/1.32 OswaldLight}aside{display:none}body.notes aside{display:block;position:absolute;left:22px;top:418px;width:675px;color:#fff;background:#222;font:15px/1.5 sans-serif;padding:12px}nav{position:fixed;bottom:5px;left:12px;color:#eee;font:12px sans-serif}button{font:inherit;cursor:pointer;background:transparent;color:inherit;border:1px solid #666;padding:3px 9px}#help{margin-left:12px}@media print{body{background:white}#stage{position:static!important;transform:none!important;width:720px;height:auto}.slide{display:block;position:relative;page-break-after:always}nav,aside{display:none!important}@page{size:720px 405px;margin:0}}
'''
    js = '''let i=0;const slides=[...document.querySelectorAll('.slide')];function resize(){const k=Math.min(innerWidth/720,(innerHeight-36)/(document.body.classList.contains('notes')?570:405));const s=document.getElementById('stage');s.style.transform=`scale(${k})`;s.style.left=`${(innerWidth-720*k)/2}px`;s.style.top='8px'}function show(n){i=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,j)=>s.classList.toggle('active',j===i));document.getElementById('count').textContent=`${i+1}/${slides.length}`;resize()}document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(i+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(i-1)}if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1);if(e.key.toLowerCase()==='n'){document.body.classList.toggle('notes');resize()}if(e.key.toLowerCase()==='f')document.documentElement.requestFullscreen?.()});addEventListener('resize',resize);show(0);'''
    path = OUT / 'Apresentacao_Mosquitos_IFSC.html'
    path.write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mosquitos pelo som - Luiz Ciarallo</title><style>'+css+'</style><main id="stage">'+''.join(sections)+'</main><nav><button onclick="show(i-1)">Anterior</button> <span id="count"></span> <button onclick="show(i+1)">Próximo</button><span id="help">Setas: navegar · F: tela cheia · N: roteiro</span></nav><script>'+js+'</script></html>', encoding='utf-8')
    return path


def notes_pdf(slides):
    path = OUT / 'Roteiro_de_Apresentacao_Luiz_Ciarallo.pdf'
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Fala', fontName='Helvetica', fontSize=10.5, leading=15, spaceAfter=10))
    styles.add(ParagraphStyle(name='Fonte', fontName='Helvetica', fontSize=8.5, leading=12, textColor='#555555', spaceAfter=12))
    story = [Paragraph('Roteiro de apresentação', styles['Title']),
             Paragraph('Luiz Eduardo do Nascimento Ciarallo', styles['Heading2']),
             Paragraph('Mosquitos pelo som - Computação Científica para a Indústria / IFSC', styles['Fala']),
             Paragraph(f'Tempo planejado: {sum(s["seconds"] for s in slides)//60} min {sum(s["seconds"] for s in slides)%60} s. Use o roteiro para ensaiar; os slides têm apenas os pontos principais. Os tempos são sugestões, com espaço para pausas.', styles['Fala']),
             Paragraph('As avaliações são retrospectivas. A primeira CNN foi revista com uma CNN que preserva a frequência e uma MLP profunda. O acerto interno passou de 55,3% a 69,5%, ainda abaixo de 71,6% da referência linear. O projeto ainda não tem confiabilidade em campo comprovada.', styles['Fala'])]
    for number, slide in enumerate(slides, 1):
        story.append(KeepTogether([
            Paragraph(f'{number:02d}. {html.escape(slide["title"].replace(chr(10), " "))} ({slide["seconds"]} s)', styles['Heading2']),
            Paragraph(html.escape(slide['note']), styles['Fala']),
            Paragraph('Fonte: '+html.escape(slide['source']), styles['Fonte'])]))
    story += [PageBreak(), Paragraph('Referências e evidências', styles['Heading1'])]
    refs = [
        'IFSC. Modelo de Apresentação de Slides. '+IFSC,
        'Mukundarajan, H. et al. Using mobile phones as acoustic sensors for high-throughput mosquito surveillance. eLife, 2017. https://doi.org/10.7554/eLife.27854',
        'Fanioudakis, E.; Geismar, M.; Potamitis, I. Mosquito wingbeat analysis and classification using deep learning. EUSIPCO, 2018. https://doi.org/10.23919/EUSIPCO.2018.8553542',
        'Supratak, A. et al. MosquitoSong+: A noise-robust deep learning model for mosquito classification from wingbeat sounds. PLOS ONE, 2024. https://doi.org/10.1371/journal.pone.0310121',
        'Dryad. Coleção de dados acústicos. https://doi.org/10.5061/dryad.98d7s',
        'BioDCASE. Origem dos 17 arquivos públicos usados na avaliação retrospectiva de transferência. https://zenodo.org/records/19095788',
        'Repositório mosquito-wingbeat, checkout local: commit 1ab88b8 com alterações locais preservadas. Evidências novas em results/ifsc_cnn_2026_10_11/ e results/ifsc_melhorias_2026_10_11/.',
        'Relatórios históricos: reports/modelo_grande_cnn.md; reports/tres_niveis_2026_10_09.md; reports/ensemble_especies_calibracao_separada_2026_10_09.md.',
        'Enunciado: Trabalho_Descrição.pdf, fornecido pelo usuário. Professores: Gregory Chagas Da Costa Gomes e Maurício Edgar Stivanello. Exige contexto, correlatos, proposta implementada, resultados e discussão; apresentação de 15 a 20 minutos.'
    ]
    for ref in refs:
        story.append(Paragraph(html.escape(ref), styles['Fala']))
    story += [Paragraph('Respostas rápidas para a discussão', styles['Heading1'])]
    qa = [('Por que é uma rede profunda?', 'As CNNs têm quatro convoluções aprendidas e duas camadas lineares. A MLP revista tem três camadas ocultas e uma camada de saída; seus pesos são treinados.'),
          ('As alterações melhoraram?', 'Sim, neste teste interno: de 105 para 132 acertos em 190 gravações, um ganho de 14,2 pontos percentuais. Nos arquivos externos, de zero para três acertos em dezessete. Isso não demonstra confiabilidade em campo.'),
          ('Como foi escolhido o modelo?', 'Pela menor perda de validação por gravação, equilibrada por espécie. A MLP foi escolhida antes da avaliação do teste. O teste e os arquivos externos já eram conhecidos da pesquisa, portanto a análise é retrospectiva.'),
          ('Por que o método linear ainda foi melhor?', 'A referência linear teve 136 acertos e as redes revistas, 132. A diferença global é pequena, e não houve comprovação estatística de superioridade. Não há motivo para presumir que uma rede profunda vencerá em todos os cenários.'),
          ('Qual unidade define os percentuais?', 'Na CNN nova e na referência linear, o resultado principal é por gravação, agregando probabilidades das janelas. A CNN maior usa números históricos por janela. Os denominadores ficam nos slides.'),
          ('Essa rede detecta ausência de mosquito?', 'Não. A CNN nova é um classificador de 20 espécies por escolha forçada. Ela sempre retorna uma classe. O experimento não valida presença nem falsos alarmes.'),
          ('Isso já pode ir para o Arduino?', 'Ainda depende de validação externa independente e testes na placa. Esta rodada só realizou avaliação computacional; não alterou firmware ou instaladores.'),
          ('Os 17 arquivos são um teste cego?', 'Não. Estavam fora do treino desta CNN, mas já haviam sido estudados no projeto. É avaliação retrospectiva e não garante indivíduos independentes.'),
          ('Como repetir e demonstrar?', 'Siga o LEIA-ME do pacote. A inferência usa pesos já treinados; um novo treinamento exige os dados e caches do repositório e uma nova pasta de resultados.')]
    for q, a in qa:
        story.append(KeepTogether([Paragraph(html.escape(q), styles['Heading3']), Paragraph(html.escape(a), styles['Fala'])]))
    def footer(c, doc):
        c.setFont('Helvetica', 8)
        c.setFillColor('#555555')
        c.drawString(42, 25, 'Luiz Eduardo do Nascimento Ciarallo | Trabalho IFSC | 11/10/2026')
        c.drawRightString(A4[0]-42, 25, str(doc.page))
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=38, bottomMargin=42,
                      title='Roteiro de apresentação - Mosquitos pelo som',
                      author='Luiz Eduardo do Nascimento Ciarallo').build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def notebook():
    import nbformat
    n = nbformat.v4.new_notebook()
    n.metadata['kernelspec'] = dict(display_name='Python (mosquito-wingbeat)', language='python', name='python3')
    n.cells = [nbformat.v4.new_markdown_cell('# CNN para classificação acústica de mosquitos\n\nLuiz Eduardo do Nascimento Ciarallo — IFSC.\n\nCNN realmente treinada em 11/10/2026. Avaliação **retrospectiva**, sem validação de presença ou de campo. Este notebook lê e verifica os resultados congelados; não inicia um treinamento ao abrir.'),
               nbformat.v4.new_code_cell("from pathlib import Path\nimport json, hashlib\nimport pandas as pd\nROOT = Path.cwd()\nif not (ROOT / 'results/ifsc_cnn_2026_10_11').exists():\n    ROOT = ROOT.parent\nOUT = ROOT / 'results/ifsc_cnn_2026_10_11'\nassert OUT.exists(), 'Abra na raiz do repo ou do pacote de entrega.'\nprotocol = json.loads((OUT/'protocol.json').read_text())\nprotocol"),
               nbformat.v4.new_markdown_cell('## Arquitetura e partições\nEntrada espectral com dois canais; quatro convoluções; camada de 128 unidades; saída de vinte espécies. Treino ajusta os pesos; validação escolhe a época. A calibração histórica fica reservada e não é usada.'),
               nbformat.v4.new_code_cell("frame = pd.read_csv(OUT/'partitions.csv')\nassert frame.groupby('group').split.nunique().max() == 1\nassert frame.groupby('path').split.nunique().max() == 1\nassert frame.groupby('content_sha256').split.nunique().max() == 1\nframe.groupby('split').agg(gravações=('group','nunique'), janelas=('group','size'), espécies=('label','nunique'))"),
               nbformat.v4.new_code_cell("history = pd.read_json(OUT/'history.json')\nhistory.plot(x='epoch', y=['train_loss','validation_loss'], title='Treino e validação: sinais de sobreajuste');"),
               nbformat.v4.new_code_cell("metrics = pd.read_csv(OUT/'metrics.csv')\nmetrics[['model','recordings','correct_recordings','recording_accuracy','recording_balanced_accuracy','recording_macro_f1']]"),
               nbformat.v4.new_markdown_cell('## Transferência retrospectiva\nOs dezessete arquivos externos já foram examinados no projeto. Não entraram no treinamento desta CNN. Não são uma nova coleta cega; a independência de indivíduos é desconhecida. Não há ausência/ruído para medir falsos alarmes.'),
               nbformat.v4.new_code_cell("json.loads((OUT/'transfer_metrics.json').read_text())"),
               nbformat.v4.new_code_cell("freeze = json.loads((OUT/'freeze.json').read_text())\nassert hashlib.sha256((OUT/'cnn.pt').read_bytes()).hexdigest() == freeze['cnn_sha256']\npred = pd.read_csv(OUT/'cnn_recordings.csv')\nassert len(pred) == int(metrics.loc[metrics.model.eq('cnn'),'recordings'].iloc[0])\nassert int(pred.truth.eq(pred.prediction).sum()) == int(metrics.loc[metrics.model.eq('cnn'),'correct_recordings'].iloc[0])\nprint('Pesos e contagens conferidos. Não equivale a validação de campo.')"),
               nbformat.v4.new_markdown_cell('## Redes revistas após a primeira rodada\nCNN com posições de frequência preservadas e MLP profunda. As mesmas partições foram mantidas. A escolha entre as redes e as épocas usa a validação por gravação; o teste e o desafio externo são retrospectivos.'),
               nbformat.v4.new_code_cell("REV = ROOT/'results/ifsc_melhorias_2026_10_11'\nselection = json.loads((REV/'selection.json').read_text())\nselection"),
               nbformat.v4.new_code_cell("revised = pd.read_csv(REV/'metrics.csv')\ncomparison = pd.concat([metrics,revised],ignore_index=True)\ncomparison[['model','correct_recordings','recording_accuracy','recording_balanced_accuracy','recording_macro_f1']]"),
               nbformat.v4.new_code_cell("pd.read_csv(REV/'external_metrics.csv')[['model','correct_recordings','recordings','recording_accuracy']]"),
               nbformat.v4.new_code_cell("rev_freeze = json.loads((REV/'freeze.json').read_text())\nfor name,expected in rev_freeze['model_sha256'].items():\n    assert hashlib.sha256((REV/f'{name}.pt').read_bytes()).hexdigest() == expected\n    pred = pd.read_csv(REV/f'{name}_recordings.csv')\n    row = revised.loc[revised.model.eq(name)].iloc[0]\n    assert len(pred) == row.recordings\n    assert pred.truth.eq(pred.prediction).sum() == row.correct_recordings\nassert hashlib.sha256((REV/'mlp_scaler.joblib').read_bytes()).hexdigest() == rev_freeze['scaler_sha256']\nprint('Redes revistas, scaler e métricas conferidos.')"),
               nbformat.v4.new_markdown_cell('Inferência do modelo escolhido pela validação:\n```bash\n.venv/bin/python trabalho_ifsc/inferir_melhorias.py arquivo.wav\n```\nA rede sempre escolhe uma espécie; a entrada não foi validada como presença de mosquito.'),
               nbformat.v4.new_markdown_cell('## Executar o programa\n\nInferência com pesos prontos, na raiz:\n```bash\n.venv/bin/python trabalho_ifsc/experimento_cnn.py --audio arquivo.wav\n```\n\nNova rodada de treino com dados/caches já preparados no repo:\n```bash\n.venv/bin/python trabalho_ifsc/experimento_cnn.py --output results/ifsc_nova_rodada\n```\nO programa recusa sobrescrever resultados existentes. Não use a classe escolhida como prova de presença: o modelo sempre escolhe uma espécie.')]
    nbformat.write(n, ROOT / 'trabalho_ifsc/CNN_Mosquitos_IFSC.ipynb')


def main(assets):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['Oswald', 'OswaldLight', 'OswaldBold']:
        pdfmetrics.registerFont(TTFont(name, str(assets / (name + '.ttf'))))
    graph_dir = plots(assets)
    slides = contents()
    (OUT / 'conteudo_editavel.json').write_text(json.dumps(slides, ensure_ascii=False, indent=2))
    paths = [slides_pdf(slides, assets, graph_dir), slides_html(slides, assets, graph_dir), notes_pdf(slides)]
    notebook()
    for p in paths:
        print(p)
    print('slides', len(slides), 'seconds', sum(s['seconds'] for s in slides))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    main(p.parse_args().assets)
