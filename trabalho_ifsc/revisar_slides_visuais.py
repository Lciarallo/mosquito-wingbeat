#!/usr/bin/env python3
"""Prepare native Google Slides requests; does not authenticate or write to Drive.

Uses the complete snapshot of the existing IFSC deck. Existing titles, charts,
table, notes, master and layouts remain native. Run once against the original
snapshot; inspect the generated requests before applying with the connector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/ifsc_2026_10_13_visual'
GREEN = {'red': 0.157, 'green': 0.49, 'blue': 0.196}
RED = {'red': 0.75, 'green': 0.08, 'blue': 0.12}
BLACK = {'red': 0.05, 'green': 0.05, 'blue': 0.05}
GRAY = {'red': 0.35, 'green': 0.35, 'blue': 0.35}
CDC = 'https://www.cdc.gov/mosquitoes/about/about-mosquitoes-in-the-united-states.html'
ARDUINO = 'https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2/'
PAPERS = [
    'https://doi.org/10.7554/eLife.27854',
    'https://doi.org/10.23919/EUSIPCO.2018.8553542',
    'https://doi.org/10.1371/journal.pone.0310121',
]
REPO = 'https://github.com/Lciarallo/mosquito-wingbeat'

# Each retained native slide is its own inspected exemplar. Reused chart and
# table objects are kept; text is replaced with explicitly reconstructed runs.
PLAN = [
    ('p1', 'Mosquitos pelo som', 25,
     'Apresento a investigação sobre classificação acústica de mosquitos. O trabalho implementa redes neurais profundas e compara o resultado com um modelo linear. A pergunta central é se um padrão aprendido em uma coleção continua útil em outra gravação. Os resultados internos melhoraram, mas a transferência ainda é a principal limitação.', [REPO]),
    ('ifsc_slide_02', 'O voo deixa uma assinatura sonora', 65,
     'O movimento das asas produz padrões acústicos que podem ajudar no monitoramento. A foto é uma referência visual de Aedes aegypti, e não um mosquito capturado pelo meu experimento. O classificador recebe áudio; a fotografia não é uma entrada da rede. O objetivo é investigar reconhecimento acústico e generalização entre gravações. Frequências semelhantes, microfones, distância e ambiente impedem tratar o som como uma identidade infalível. Não estamos diagnosticando doenças nem estimando risco epidemiológico.', [PAPERS[0], PAPERS[2], CDC]),
    ('ifsc_slide_03', 'Detectar e identificar são tarefas distintas', 75,
     'Presença pergunta se há mosquito ou apenas ruído. Gênero agrupa espécies; espécie é a identificação mais específica. As três fotos mostram exemplos de Aedes aegypti, Anopheles quadrimaculatus e Culex tarsalis; não representam todas as classes do estudo. O projeto também considera Culiseta. Uma rede treinada somente com espécies de mosquito sempre escolhe uma classe, mesmo quando recebe ruído. Por isso acurácia de espécie não mede detecção nem falsos alarmes. O desempenho precisa ser avaliado separadamente em cada tarefa.', [CDC, REPO + '/blob/main/reports/tres_niveis_2026_10_09.md']),
    ('ifsc_slide_04', 'O áudio vira um espectrograma', 70,
     'O gráfico superior é o sinal no tempo; o inferior mostra como a energia se distribui nas frequências. O exemplo é um trecho real rotulado Aedes aegypti no Dryad, com 0,992 segundo. Cores claras indicam energia relativa maior. O exemplo foi selecionado por classe, partição e ordem original, sem procurar uma imagem bonita. Este gráfico explicativo usa STFT; a entrada da CNN usa log-mel e contraste espectral, com dois canais. A fotografia do slide anterior apenas apresenta o inseto. Não confundimos a identificação informada pelo corpus com confirmação independente.', ['https://doi.org/10.5061/dryad.98d7s', 'trabalho_ifsc/preparar_material.py', 'results/ifsc_cnn_2026_10_11/protocol.json']),
    ('ifsc_slide_05', 'Três estudos orientaram o experimento', 85,
     'Mukundarajan e colegas investigaram aquisição acústica com celulares. Fanioudakis e colegas trabalharam com sinais de batimento de asas obtidos por sensores ópticos e redes profundas, incluindo representações de espectrograma. Essa aquisição não é a mesma coisa que o áudio de celular usado aqui. MosquitoSong+ usa CNN 1D em áudio e amplia o treino com ruído e variação de volume. Os estudos sugerem caminhos, mas suas porcentagens não são diretamente comparáveis: bases, classes, sensores e protocolos diferem. Meu experimento é uma implementação própria e não uma reprodução integral dos treinamentos dos artigos.', PAPERS),
    ('ifsc_slide_09', 'Separar as gravações antes de treinar', 80,
     'As partições têm 233 gravações de treino, 63 de validação, 83 de calibração reservada e 190 de teste. A calibração não foi usada: não ajustei rejeição nem limiar de presença nesta rodada. Janelas da mesma gravação ficam na mesma partição. A validação escolhe o modelo e a época; o teste só mede o modelo congelado. Existem 20 espécies, mas a validação contém 19. A separação disponível por arquivo, grupo e hash não prova independência de indivíduos ou sessões. O teste interno é histórico e já tinha sido examinado no projeto; sua interpretação é retrospectiva.', ['results/ifsc_cnn_2026_10_11/protocol.json', 'results/ifsc_cnn_2026_10_11/freeze.json']),
    ('ifsc_slide_08', 'A primeira CNN aprende no espectrograma', 90,
     'A primeira CNN tem 130.900 parâmetros. Cada janela entra com dois canais, 64 bandas e 55 quadros: log-mel e contraste. Quatro convoluções usam 16, 32, 64 e 96 canais, com normalização, ativação e redução de resolução; depois há uma camada de 128 unidades e a saída de 20 espécies. O treinamento é supervisionado com rótulos do corpus. A CNN revista preserva mais posições de frequência e tem 224.036 parâmetros. A MLP compara média e desvio temporal dos mesmos canais, com camadas ocultas de 256, 128 e 64 unidades. São modelos computacionais; estas redes novas não foram incorporadas ao firmware.', ['trabalho_ifsc/experimento_cnn.py', 'trabalho_ifsc/comparar_melhorias.py', 'results/ifsc_cnn_2026_10_11/protocol.json', 'results/ifsc_melhorias_2026_10_11/protocol.json']),
    ('ifsc_slide_10', 'Aprender o treino não basta', 70,
     'A curva verde mostra a perda de treino da primeira CNN; a amarela mostra a validação. Menor perda é melhor. A rede continua melhorando no treino, mas isso não se repete de forma sustentada na validação: há sinais de sobreajuste. A época 6 foi escolhida pela validação entre 12 épocas executadas, e não pelo teste. Isso motivou mudanças na arquitetura e na regularização. Esta curva descreve a primeira CNN; não é o histórico da CNN revista nem da MLP.', ['results/ifsc_cnn_2026_10_11/history.json', 'results/ifsc_cnn_2026_10_11/freeze.json', 'trabalho_ifsc/assets/graficos/aprendizado.png']),
    ('ifsc_slide_12', 'As redes chegaram a 69,5% de acerto', 105,
     'Nas mesmas 190 gravações, a CNN inicial acertou 105, e tanto a CNN revista quanto a MLP acertaram 132. Foram 27 acertos adicionais, ou 14,21 pontos percentuais. A referência linear acertou 136, portanto manteve a maior acurácia global. A CNN revista preserva frequência; a MLP usa descritores temporais e três camadas ocultas. Foram reforçadas regularização e dropout e alterados aprendizado e seleção. A MLP foi selecionada pela validação antes da inferência no teste. Ela obteve F1 macro de 77,06%, mas isso não demonstra superioridade estatística. Este gráfico avalia escolha forçada de espécie por gravação, não presença em ruído.', ['results/ifsc_cnn_2026_10_11/metrics.json', 'results/ifsc_melhorias_2026_10_11/metrics.json', 'results/ifsc_melhorias_2026_10_11/selection.json']),
    ('ifsc_slide_13', 'Em outra origem, só 3 de 17 acertos', 95,
     'A MLP selecionada e a CNN revista acertaram três dos 17 arquivos externos, ou 17,65%. Esses três acertos correspondem a dois arquivos de Anopheles minimus e um de Anopheles stephensi. Não acertaram Aedes ou Culex nessa amostra. A primeira CNN acertou zero dos 17. As porcentagens comparam dois conjuntos distintos, não o mesmo teste antes e depois. A amostra externa é pequena, contém cinco espécies e já tinha sido examinada no projeto; portanto a análise é retrospectiva e não uma validação independente. Não há controles de ausência ou ruído neste conjunto. O resultado indica dificuldade de transferência, sem provar confiabilidade em campo.', ['results/ifsc_melhorias_2026_10_11/external_metrics.json', 'results/ifsc_cnn_2026_10_11/transfer_metrics.json', 'trabalho_ifsc/LEIA-ME.md']),
    ('ifsc_slide_16', 'Até o gênero perde acerto fora da base', 70,
     'Este gráfico é de outro experimento, histórico, com ensemble clássico; não é da CNN ou da MLP anteriores. No corpus conhecido, gênero acertou 549 de 569 e espécie 408 de 569. Nos 17 arquivos externos, gênero acertou 7 e espécie 4. Agrupar em gênero facilita a escolha nessa amostra, mas também sofre perda importante fora da base. Os dois modelos sempre escolhem uma classe de mosquito: isso não testa se há mosquito. Não somar esses números aos resultados das redes novas nem atribuir todos os gráficos a um único protocolo.', ['reports/tres_niveis_2026_10_09.md', 'results/three_levels_2026_10_09/', 'trabalho_ifsc/assets/graficos/genero_especie.png']),
    ('ifsc_slide_18', 'Arduino: aplicação em investigação', 75,
     'A placa alvo do projeto é o Arduino Nano 33 BLE Sense ou Sense Rev2; a imagem é uma fotografia oficial do Rev2, não uma montagem física realizada por mim. O microfone digital PDM integrado permite capturar áudio. O projeto tem protótipos históricos de firmware; a compilação por software não equivale a teste físico. As novas CNNs e a MLP desta apresentação foram avaliadas no computador e não foram incorporadas ao Arduino. Ainda faltam validar microfone, distância, memória e tempo de processamento na placa. O detector histórico marcou 29,17% das janelas de ruído como candidatas, reforçando a necessidade de medir falsos alarmes. Nenhuma compilação ou instalação foi feita nesta revisão de slides.', [ARDUINO, 'https://store.arduino.cc/products/nano-33-ble-sense-rev2', 'firmware/README.md', 'trabalho_ifsc/LEIA-ME.md']),
    ('ifsc_slide_06', 'O próximo avanço depende de novos dados', 80,
     'Primeiro é preciso coletar gravações com espécie confirmada e variar dispositivos, distância e ambiente. Depois devem ser anotados períodos sem mosquito e ruídos reais, para medir falsos alarmes. A avaliação deve reservar indivíduos, sessões, aparelhos e locais inteiros antes de decidir a arquitetura. O teste final precisa permanecer independente dos ajustes. Somente após esse protocolo faz sentido avaliar presença, identificação e rejeição como um sistema completo, medir alarmes por hora quando houver anotação e testar o processamento na placa. Os desafios não se resolvem apenas aumentando o número de camadas.', ['MODEL_VALIDATION.md', 'reports/tres_niveis_2026_10_09.md', 'firmware/README.md']),
    ('ifsc_slide_19', 'A rede melhorou; a generalização falta', 50,
     'O trabalho implementou redes neurais profundas em áudio real e comparou alternativas nas mesmas partições. As redes passaram de 55,26% a 69,47% de acerto interno por gravação; a referência linear manteve 71,58%. Em outra origem, as redes revistas ficaram em três de 17 arquivos. O principal resultado é aprender a separar melhora interna de generalização. O próximo passo é uma coleta independente com controles de ruído, seguida de avaliação integrada e teste na placa. Código, pesos, métricas e notas permitem discutir os resultados sem apresentar o sistema como pronto para uso em campo.', ['results/ifsc_cnn_2026_10_11/metrics.json', 'results/ifsc_melhorias_2026_10_11/metrics.json', 'results/ifsc_melhorias_2026_10_11/external_metrics.json']),
    ('ifsc_slide_20', 'Fontes e material do projeto', 25,
     'Os artigos completos, o conjunto Dryad, a documentação da placa e os créditos das fotografias estão nos links abaixo. As fotografias dos mosquitos vêm do CDC; a placa vem da loja oficial Arduino. Os gráficos usam resultados e áudio já presentes no repositório. O modelo oficial IFSC, seu mestre e seus 11 layouts foram mantidos. Este slide oferece referências; a conclusão científica está no slide anterior.', PAPERS + ['https://doi.org/10.5061/dryad.98d7s', CDC, ARDUINO, REPO, 'https://ifsc.edu.br/documents/d/guest/slides?download=true']),
]


class Requests:
    def __init__(self, resource):
        self.resource = resource
        self.slides = {s['objectId']: s for s in resource['slides']}
        self.elements = {e['objectId']: e for s in resource['slides'] for e in s.get('pageElements', [])}
        self.items = []
        self.assets = []
        self.visible = {sid: [] for sid, *_ in PLAN}
        self.metadata = json.loads((OUTPUT / 'assets/fontes_imagens.json').read_text())

    def add(self, key, value):
        self.items.append({key: value})

    @staticmethod
    def props(sid, x, y, w, h):
        return {'pageObjectId': sid, 'size': {'width': {'magnitude': w, 'unit': 'PT'}, 'height': {'magnitude': h, 'unit': 'PT'}}, 'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': x, 'translateY': y, 'unit': 'PT'}}

    def style(self, eid, text, size=21.5, weight=300, color=BLACK, start=0, end=None, italic=False):
        self.add('updateTextStyle', {'objectId': eid, 'textRange': {'type': 'FIXED_RANGE', 'startIndex': start, 'endIndex': len(text) if end is None else end}, 'style': {'fontFamily': 'Oswald', 'weightedFontFamily': {'fontFamily': 'Oswald', 'weight': weight}, 'fontSize': {'magnitude': size, 'unit': 'PT'}, 'bold': False, 'italic': italic, 'foregroundColor': {'opaqueColor': {'rgbColor': color}}, 'backgroundColor': {}}, 'fields': 'fontFamily,weightedFontFamily,fontSize,bold,italic,foregroundColor,backgroundColor'})

    def replace(self, sid, eid, text, size=21.5, weight=300, color=BLACK):
        self.add('deleteText', {'objectId': eid, 'textRange': {'type': 'ALL'}})
        self.add('insertText', {'objectId': eid, 'insertionIndex': 0, 'text': text})
        self.style(eid, text, size, weight, color)
        self.visible[sid].append(text)

    def transform(self, eid, x, y, w, h):
        e = self.elements[eid]
        def pt(v):
            return v['magnitude'] / 12700 if v['unit'] == 'EMU' else v['magnitude']
        self.add('updatePageElementTransform', {'objectId': eid, 'applyMode': 'ABSOLUTE', 'transform': {'scaleX': w / pt(e['size']['width']), 'scaleY': h / pt(e['size']['height']), 'translateX': x, 'translateY': y, 'unit': 'PT'}})

    def box(self, sid, name, text, x, y, w, h, size=21.5, weight=300, color=BLACK, align='START', italic=False):
        eid = 'visual_' + name
        self.add('createShape', {'objectId': eid, 'shapeType': 'TEXT_BOX', 'elementProperties': self.props(sid, x, y, w, h)})
        self.add('insertText', {'objectId': eid, 'insertionIndex': 0, 'text': text})
        self.style(eid, text, size, weight, color, italic=italic)
        self.add('updateParagraphStyle', {'objectId': eid, 'textRange': {'type': 'ALL'}, 'style': {'alignment': align, 'lineSpacing': 100, 'spaceAbove': {'magnitude': 0, 'unit': 'PT'}, 'spaceBelow': {'magnitude': 0, 'unit': 'PT'}}, 'fields': 'alignment,lineSpacing,spaceAbove,spaceBelow'})
        self.visible[sid].append(text)
        return eid

    def photograph(self, sid, name, x, y, width):
        m = next(v for v in self.metadata if v['name'] == name)
        height = width * m['height'] / m['width']
        eid = 'visual_photo_' + name.replace('.', '_')
        self.assets.append({'createImage': {'objectId': eid, 'url': m['url'], 'elementProperties': self.props(sid, x, y, width, height)}})
        self.assets.append({'updatePageElementAltText': {'objectId': eid, 'title': m['credit'], 'description': m['credit'] + '. Fonte: ' + m['source_page']}})

    def node(self, sid, name, text, x, width):
        eid = 'visual_node_' + name
        self.add('createShape', {'objectId': eid, 'shapeType': 'RECTANGLE', 'elementProperties': self.props(sid, x, 192, width, 93)})
        self.add('updateShapeProperties', {'objectId': eid, 'shapeProperties': {'shapeBackgroundFill': {'solidFill': {'color': {'rgbColor': {'red': 0.96, 'green': 0.98, 'blue': 0.96}}, 'alpha': 1}}, 'outline': {'outlineFill': {'solidFill': {'color': {'rgbColor': GREEN}}}, 'weight': {'magnitude': 1, 'unit': 'PT'}}, 'contentAlignment': 'MIDDLE'}, 'fields': 'shapeBackgroundFill,outline,contentAlignment'})
        self.add('insertText', {'objectId': eid, 'text': text})
        self.style(eid, text, 20, 300)
        self.add('updateParagraphStyle', {'objectId': eid, 'textRange': {'type': 'ALL'}, 'style': {'alignment': 'CENTER'}, 'fields': 'alignment'})
        self.visible[sid].append(text)


def build(resource):
    q = Requests(resource)
    for idx, (sid, title, seconds, note, sources) in enumerate(PLAN, 1):
        slide = q.slides[sid]
        eid = slide['pageElements'][0]['objectId']
        if sid != 'p1':
            q.replace(sid, eid, title, 31, 400)
            q.transform(eid, 24.5433, 20.2042, 562.9, 53)
            q.box(sid, f'page_{idx:02}', f'{idx:02}', 574, 374, 28, 18, 11, 300, GRAY, 'END')
        else:
            q.visible[sid].append('Mosquitos pelo som\nLuiz Eduardo do Nascimento Ciarallo')
            q.box(sid, 'cover_subtitle', 'Classificação acústica com redes neurais', 15.4, 213, 380, 30, 21.5)
        notes_id = slide['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
        notes = f'Tempo sugerido: {seconds} segundos.\n\n{note}\n\n[Sources]\n' + '\n'.join(sources) + '\n[/Sources]\n'
        q.add('deleteText', {'objectId': notes_id, 'textRange': {'type': 'ALL'}})
        q.add('insertText', {'objectId': notes_id, 'text': notes})

    sid = 'ifsc_slide_02'
    q.box(sid, 'why_body', 'As asas produzem\npadrões de frequência.\n\nObjetivo: reconhecer\nmosquitos em áudio.', 26, 98, 238, 163)
    q.photograph(sid, 'aedes_repouso.png', 280, 97, 298)
    q.box(sid, 'aedes_caption', 'Aedes aegypti · fotografia CDC', 280, 270, 298, 25, 14, 300, GRAY, italic=True)
    q.box(sid, 'why_question', 'O padrão se mantém em outra gravação?', 26, 308, 559, 46, 26, 400, GREEN)

    sid = 'ifsc_slide_03'
    for j, (label, detail) in enumerate([('Presença', 'Mosquito ou ruído?'), ('Gênero', 'Aedes, Anopheles, Culex…'), ('Espécie', 'Por exemplo, Aedes aegypti')]):
        q.box(sid, f'task_{j}', label, 26, 86 + j*39, 145, 34, 21.5, 400, GREEN)
        q.box(sid, f'task_detail_{j}', detail, 177, 86 + j*39, 411, 34)
    for j, (name, caption) in enumerate([('aedes_aegypti.jpg', 'Aedes aegypti'), ('anopheles_quadrimaculatus.jpg', 'Anopheles quadrimaculatus'), ('culex_tarsalis.jpg', 'Culex tarsalis')]):
        x = 31 + j*187
        q.photograph(sid, name, x, 213, 173)
        q.box(sid, f'genus_caption_{j}', caption, x-3, 314, 181, 28, 14, 300, BLACK, 'CENTER', italic=True)
    q.box(sid, 'mosquito_source', 'Fotografias: CDC · exemplos de espécies de três gêneros', 26, 352, 559, 25, 14, 300, GRAY)

    sid = 'ifsc_slide_04'
    q.replace(sid, 'ifsc_caption_04', 'Aedes aegypti · áudio real do Dryad · janela de 0,992 s', 14)

    sid = 'ifsc_slide_05'
    for j, (year, author, detail) in enumerate([('2017', 'Mukundarajan et al.', 'Celulares capturam o som do voo.'), ('2018', 'Fanioudakis et al.', 'Sensores ópticos e redes profundas.'), ('2024', 'MosquitoSong+ · Supratak et al.', 'CNN 1D com ruído e variação de volume.')]):
        y = 92 + j*78
        q.box(sid, f'paper_year_{j}', year, 26, y, 84, 50, 31, 400, GREEN)
        q.box(sid, f'paper_author_{j}', author, 122, y, 466, 35, 23, 400)
        q.box(sid, f'paper_detail_{j}', detail, 122, y+33, 466, 35)
    q.box(sid, 'paper_limit', 'Bases, sensores e protocolos diferentes; resultados não são equivalentes.', 26, 352, 562, 27, 14, 300, GRAY)

    sid = 'ifsc_slide_09'
    q.replace(sid, 'ifsc_caption_09', 'Calibração reservada; separar arquivos não prova independência.', 14)

    sid = 'ifsc_slide_08'
    q.box(sid, 'cnn_parameters', '130.900 parâmetros', 26, 93, 370, 46, 31, 400, GREEN)
    q.box(sid, 'cnn_dimensions', 'Entrada: 2 canais · 64 bandas · 55 quadros de tempo', 26, 139, 563, 37)
    # Draw arrows before nodes so they remain behind the entity shapes.
    for j, x in enumerate([149, 301, 442]):
        eid = f'visual_arrow_{j}'
        q.add('createShape', {'objectId': eid, 'shapeType': 'RIGHT_ARROW', 'elementProperties': q.props(sid, x, 232, 18, 13)})
        q.add('updateShapeProperties', {'objectId': eid, 'shapeProperties': {'shapeBackgroundFill': {'solidFill': {'color': {'rgbColor': GREEN}, 'alpha': 1}}, 'outline': {'propertyState': 'NOT_RENDERED'}}, 'fields': 'shapeBackgroundFill,outline'})
    q.node(sid, 'input', 'Log-mel\n+ contraste', 31, 110)
    q.node(sid, 'convolutions', '4 convoluções\n16/32/64/96', 175, 118)
    q.node(sid, 'dense', 'Densa\n128 unidades', 327, 107)
    q.node(sid, 'output', 'Saída\n20 espécies', 468, 110)
    q.box(sid, 'cnn_comparisons', 'Também comparei uma CNN revista e uma MLP profunda.', 26, 311, 563, 36)
    q.box(sid, 'cnn_limit', 'A saída sempre escolhe uma espécie; presença é outra tarefa.', 26, 352, 563, 27, 14, 300, GRAY)

    sid = 'ifsc_slide_10'
    q.replace(sid, 'ifsc_caption_10', 'Primeira CNN · época 6 selecionada pela validação', 14)

    sid = 'ifsc_slide_12'
    q.replace(sid, 'ifsc_caption_12', '+27 acertos nas mesmas 190 gravações; referência linear: 136/190.', 14)

    sid = 'ifsc_slide_13'
    for j, (label, number, count, color) in enumerate([('Teste interno', '69,5%', '132/190 gravações', GREEN), ('Outra origem', '17,6%', '3/17 arquivos', RED)]):
        x = 31 + j*280
        q.box(sid, f'external_label_{j}', label, x, 93, 254, 39, 21.5, 400)
        q.box(sid, f'external_number_{j}', number, x-3, 135, 260, 107, 68, 400, color)
        q.box(sid, f'external_count_{j}', count, x, 249, 254, 37)
    q.box(sid, 'external_message', 'Trocar a origem derrubou o desempenho.', 26, 303, 563, 42, 26, 400)
    q.box(sid, 'external_limit', 'MLP selecionada · amostra retrospectiva de 17 arquivos; sem coleta cega.', 26, 352, 562, 27, 14, 300, GRAY)

    sid = 'ifsc_slide_16'
    q.replace(sid, 'ifsc_caption_16', 'Ensemble clássico histórico · escolha forçada, sem detectar presença.', 14)

    sid = 'ifsc_slide_18'
    q.box(sid, 'arduino_name', 'Nano 33 BLE\nSense / Sense Rev2', 26, 95, 259, 77, 25, 400, GREEN)
    q.box(sid, 'arduino_body', 'Microfone PDM integrado.\n\nObjetivo: analisar o áudio\nna própria placa.', 26, 184, 260, 126)
    q.photograph(sid, 'arduino_nano_sense_rev2.jpg', 287, 95, 302)
    q.box(sid, 'arduino_caption', 'Nano 33 BLE Sense Rev2 · foto oficial Arduino', 285, 328, 304, 24, 12.5, 300, GRAY)
    q.box(sid, 'arduino_status', 'Ainda falta validar o sistema na placa física.', 26, 359, 561, 27, 18, 400, RED)

    sid = 'ifsc_slide_06'
    for j, (number, title, detail) in enumerate([('1', 'Coletar', 'Espécie confirmada, aparelhos e ambientes variados.'), ('2', 'Controlar', 'Anotar ruído e períodos sem mosquito.'), ('3', 'Validar', 'Reservar indivíduos e sessões; depois testar a placa.')]):
        y = 96 + j*79
        q.box(sid, f'next_number_{j}', number, 26, y-2, 51, 62, 40, 400, GREEN)
        q.box(sid, f'next_title_{j}', title, 86, y, 503, 33, 23, 400)
        q.box(sid, f'next_detail_{j}', detail, 86, y+33, 503, 35)
    q.box(sid, 'next_limit', 'Microfone, ruído e sons sobrepostos continuam sendo desafios.', 26, 352, 563, 27, 14, 300, GRAY)

    sid = 'ifsc_slide_19'
    q.box(sid, 'conclusion_gain', '55,3% → 69,5%', 26, 93, 554, 86, 52, 400, GREEN)
    q.box(sid, 'conclusion_internal', 'Melhora interna nas mesmas 190 gravações.', 26, 184, 563, 44)
    q.box(sid, 'conclusion_external', 'Em outra origem: 3/17 acertos.', 26, 246, 563, 44, 26, 400)
    q.box(sid, 'conclusion_next', 'Próximo passo: coleta e avaliação independentes.', 26, 309, 563, 40)
    q.box(sid, 'conclusion_linear', 'Referência linear: 71,6% de acerto interno.', 26, 352, 563, 27, 14, 300, GRAY)

    sid = 'ifsc_slide_20'
    for j, text in enumerate(['Mukundarajan et al. (2017) · celulares e áudio', 'Fanioudakis et al. (2018) · batimentos de asas e CNNs', 'Supratak et al. (2024) · MosquitoSong+', 'Dados: Dryad · código e resultados: mosquito-wingbeat', 'Fotografias: CDC e Arduino · modelo: IFSC']):
        q.box(sid, f'reference_{j}', text, 26, 92+j*49, 563, 40)
    q.box(sid, 'references_links', 'Links completos e créditos nas notas de cada slide.', 26, 352, 563, 27, 14, 300, GRAY)
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUTPUT / '.work/source-resource.json')
    args = parser.parse_args()
    resource = json.loads(args.source.read_text())
    if resource['presentationId'] != '1n48d31ukSs-jxyxYAOrLTg8wtTj0B-YNSTBzyE425b4':
        raise ValueError('Unexpected source presentation')
    q = build(resource)
    work = OUTPUT / '.work'
    work.mkdir(parents=True, exist_ok=True)
    for name, data in [('content-requests.json', q.items), ('image-requests.json', q.assets)]:
        (work / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    ids = [v[0] for v in PLAN]
    deletions = [{'deleteObject': {'objectId': s['objectId']}} for s in resource['slides'] if s['objectId'] not in ids]
    (work / 'delete-requests.json').write_text(json.dumps(deletions, ensure_ascii=False, indent=2))
    # Google requires each multi-slide move to follow the current presentation
    # order. Prepend one slide at a time in reverse desired order instead.
    (work / 'order-requests.json').write_text(json.dumps([{'updateSlidesPosition': {'slideObjectIds': [sid], 'insertionIndex': 0}} for sid in reversed(ids)], indent=2))
    manifest = {'presentation_id': resource['presentationId'], 'date': '2026-10-13', 'seconds': sum(v[2] for v in PLAN), 'slides': [{'slide_id': sid, 'title': title, 'seconds': sec, 'visible_text': q.visible[sid], 'speaker_notes': note, 'sources': sources, 'template_exemplar': sid, 'media_mapping': [{'object_id': e['objectId'], 'action': 'keep'} for e in q.slides[sid].get('pageElements', []) if 'image' in e or 'table' in e]} for sid, title, sec, note, sources in PLAN]}
    (OUTPUT / 'conteudo_editavel.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (work / 'plano-revisao.txt').write_text('Público: turma IFSC; explicar experimento e limites de generalização.\nManter o mestre, 11 layouts, fundos e fonte Oswald.\nReduzir 20 para 15 slides; fontes e detalhes nas notas.\n' + '\n'.join(f'{i:02} {sid}: {title}; exemplar nativo = {sid}' for i, (sid, title, *_ ) in enumerate(PLAN, 1)))
    print(json.dumps({'content_requests': len(q.items), 'image_requests': len(q.assets), 'slides': len(ids), 'seconds': manifest['seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
