#!/usr/bin/env python3
"""Prepare native updates and transcribe Sinka et al. 2021, Table 1.

No authentication or Drive writes. The source snapshot must contain three native
duplicates of the existing IFSC table slide, identified as ifsc_frequency_1/2/3.
One published observation per species is displayed; every original observation
is preserved in CSV/JSON. Do not aggregate different experiments into a range.
"""
import csv
import hashlib
import json
from pathlib import Path

from revisar_slides_visuais import ROOT, OUTPUT, Requests, GREEN, BLACK

WORK = OUTPUT / '.work/frequencias'
DOI = 'https://doi.org/10.1111/2041-210X.13663'
PDF_URL = 'https://ora.ox.ac.uk/objects/uuid%3A2ba708eb-dad2-4474-be29-4891431009c5/files/rmg74qm63c'
# Values and notation transcribed from p. 1853, PDF page 6; missing ages remain
# missing. The meaning of each ± is not generalized beyond its source study.
C = 'Clements (1999); '
DATA = [
    ['Aedes aegypti', '2–5', '23', '557–600', '414–453', C+'Tischener and Schief (1955)'],
    ['Aedes aegypti', '1–10', '22', '750 ± 47', '508 ± 66', C+'Moore et al. (1986)'],
    ['Aedes aegypti', '4', '25', '715', '458', C+'Brogdon (1994)'],
    ['Aedes aegypti', '—', '32.6 ± 0.5', '975.08 ± 8.09', '—', 'Cator et al. (2011)'],
    ['Aedes aegypti', '4', '25', '—', '460', 'Brogdon (1994)'],
    ['Aedes aegypti', '7–28', '23', '711 ± 78', '511 ± 46', 'Arthur et al. (2014)'],
    ['Aedes albopictus', '4', '25', '724', '544', C+'Brogdon (1994)'],
    ['Aedes albopictus', '4', '25', '', '536', 'Brogdon (1994)'],
    ['Aedes diantaeus', '—', '24', '538 ± 20', '330 ± 13', C+'Tamarina et al. (1980)'],
    ['Aedes diantaeus', '—', '28', '—', '380 ± 22', C+'Tamarina et al. (1980)'],
    ['Aedes diantaeus', '—', '32', '—', '412 ± 22', C+'Tamarina et al. (1980)'],
    ['Aedes communis', '—', '24', '555 ± 14', '350 ± 9', C+'Tamarina et al. (1980)'],
    ['Aedes punctor', '—', '24', '503 ± 17', '308 ± 10', C+'Tamarina et al. (1980)'],
    ['Aedes triseriatus', '—', '22', '592 ± 47', '388 ± 33', C+'Moore et al. (1986)'],
    ['Aedes vexans', '—', '21', '—', '300–350', C+'Belton and Costello (1979)'],
    ['Culex pipiens', '5', '21', '—', '329–370', C+'Belton and Costello (1979)'],
    ['Culiseta inornata', '—', '21', '—', '180–250', C+'Belton and Costello (1979)'],
    ['Anopheles earlei', '', '21', '—', '190–250', C+'Belton and Costello (1979)'],
    ['Anopheles subpictus', '', '28', '520–580', '330–385', C+'Tischner (1953)'],
    ['Anopheles arabiensis', '2 to >2', '24–26', '580–820', '360–520', C+'Wekesa et al. (1998)'],
    ['Anopheles arabiensis', '1–2', '25', '703 ± 8.72', '435 ± 4.88', 'Brogdon (1998)'],
    ['Anopheles gambiae', '2->2', '24–26', '660–900', '420–600', C+'Wekesa et al. (1998)'],
    ['Anopheles gambiae', '1–2', '25', '789 ± 4.43', '533 ± 6.99', 'Brogdon (1998)'],
    ['Anopheles melas', '1–2', '25', '571 ± 8', '372 ± 4.62', 'Brogdon (1998)'],
    ['Anopheles merus', '1–2', '25', '586 ± 7.06', '380 ± 7.15', 'Brogdon (1998)'],
]
SELECTED = [[2, 6, 8, 11, 12], [13, 14, 15, 16, 17], [18, 20, 22, 23, 24]]
TITLES = ['Frequências publicadas: Aedes', 'Frequências de outros mosquitos', 'Frequências publicadas: Anopheles']
WIDTHS = [195, 97, 97, 63, 88]


def main():
    r = json.loads((WORK/'duplicated-resource.json').read_text())
    q = Requests(r)
    before = WORK/'manifest-before-frequencies.json'
    manifest = json.loads((before if before.exists() else OUTPUT/'conteudo_editavel.json').read_text())
    old_ids = [s['slide_id'] for s in manifest['slides']]
    assert len(old_ids) == 15, 'Use the 15-slide manifest captured before this addition'
    additions = []
    for i, indices in enumerate(SELECTED, 1):
        sid = f'ifsc_frequency_{i}'
        q.visible[sid] = []
        title = TITLES[i-1]
        q.replace(sid, f'frequency_title_{i}', title, 31, 400)
        q.replace(sid, f'frequency_caption_{i}', 'Sinka et al. (2021), Tabela 1 · uma observação publicada por espécie.\nSexo, idade e temperatura alteram os valores; — = não informado.', 12.5)
        q.transform(f'frequency_caption_{i}', 25, 350, 563, 42)
        q.replace(sid, f'frequency_page_{i}', f'{i+3:02}', 11)
        table = f'frequency_table_{i}'
        old_table = q.elements[table]['table']
        old_rows, old_cols = old_table['rows'], old_table['columns']
        if old_cols < 5:
            q.add('insertTableColumns', {'tableObjectId': table, 'cellLocation': {'rowIndex': 0, 'columnIndex': old_cols-1}, 'insertRight': True, 'number': 5-old_cols})
        if old_rows < len(indices)+1:
            q.add('insertTableRows', {'tableObjectId': table, 'cellLocation': {'rowIndex': old_rows-1, 'columnIndex': 0}, 'insertBelow': True, 'number': len(indices)+1-old_rows})
        for row_index in reversed(range(len(indices)+1, old_rows)):
            q.add('deleteTableRow', {'tableObjectId': table, 'cellLocation': {'rowIndex': row_index, 'columnIndex': 0}})
        for col, width in enumerate(WIDTHS):
            q.add('updateTableColumnProperties', {'objectId': table, 'columnIndices': [col], 'tableColumnProperties': {'columnWidth': {'magnitude': width, 'unit': 'PT'}}, 'fields': 'columnWidth'})
        q.add('updatePageElementTransform', {'objectId': table, 'applyMode': 'ABSOLUTE', 'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 31, 'translateY': 88, 'unit': 'PT'}})
        rows = [['Espécie', 'Machos (Hz)', 'Fêmeas (Hz)', 'Temp. °C', 'Idade (d)']]
        rows += [[DATA[k][0], DATA[k][3], DATA[k][4], DATA[k][2], DATA[k][1] or '—'] for k in indices]
        q.add('updateTableRowProperties', {'objectId': table, 'rowIndices': list(range(len(rows))), 'tableRowProperties': {'minRowHeight': {'magnitude': 27, 'unit': 'PT'}}, 'fields': 'minRowHeight'})
        for row, cells in enumerate(rows):
            for col, text in enumerate(cells):
                text = text.replace('.', ',') if row and col else text
                loc = {'rowIndex': row, 'columnIndex': col}
                if row < old_rows and col < old_cols and any(t.get('textRun', {}).get('content', '').strip() for t in old_table['tableRows'][row]['tableCells'][col].get('text', {}).get('textElements', [])):
                    q.add('deleteText', {'objectId': table, 'cellLocation': loc, 'textRange': {'type': 'ALL'}})
                q.add('insertText', {'objectId': table, 'cellLocation': loc, 'text': text})
                q.add('updateTextStyle', {'objectId': table, 'cellLocation': loc, 'textRange': {'type': 'FIXED_RANGE', 'startIndex': 0, 'endIndex': len(text)}, 'style': {'fontFamily': 'Oswald', 'weightedFontFamily': {'fontFamily': 'Oswald', 'weight': 400 if row == 0 else 300}, 'fontSize': {'magnitude': 18.5, 'unit': 'PT'}, 'bold': False, 'italic': bool(row and col == 0), 'foregroundColor': {'opaqueColor': {'rgbColor': BLACK}}}, 'fields': 'fontFamily,weightedFontFamily,fontSize,bold,italic,foregroundColor'})
                q.add('updateParagraphStyle', {'objectId': table, 'cellLocation': loc, 'textRange': {'type': 'ALL'}, 'style': {'alignment': 'START' if col == 0 else 'CENTER', 'spaceAbove': {'magnitude': 0, 'unit': 'PT'}, 'spaceBelow': {'magnitude': 0, 'unit': 'PT'}, 'lineSpacing': 100}, 'fields': 'alignment,spaceAbove,spaceBelow,lineSpacing'})
                q.add('updateTableCellProperties', {'objectId': table, 'tableRange': {'location': loc, 'rowSpan': 1, 'columnSpan': 1}, 'tableCellProperties': {'contentAlignment': 'MIDDLE', 'tableCellBackgroundFill': {'solidFill': {'color': {'rgbColor': {'red': 0.94, 'green': 0.97, 'blue': 0.94} if row == 0 else {'red': 1, 'green': 1, 'blue': 1}}, 'alpha': 1}}}, 'fields': 'contentAlignment,tableCellBackgroundFill'})
        note = ('A tabela foi adaptada de Sinka et al. (2021), Tabela 1, página 1853, sexta página do PDF. '
                'São as espécies listadas nessa tabela, não todas as vinte classes do experimento. '
                'Selecionei uma observação publicada por espécie, mantendo sexo, temperatura e idade; não combinei estudos em uma faixa artificial. '
                'Os traços representam dados não informados; o símbolo ± preserva a notação de cada estudo, sem assumir uma definição estatística comum. '
                'Essas frequências de batimento das asas não são frequências de amostragem do microfone. '
                'Frequências se sobrepõem entre espécies e variam com sexo e condições; não funcionam como um identificador único. '
                'Todas as 25 linhas originais, inclusive medições repetidas de uma espécie, estão no CSV que acompanha o material.\n\n'
                'Registros selecionados, com a fonte original compilada por Sinka:\n' + '\n'.join(' | '.join(DATA[k]) for k in indices))
        sources = [DOI, PDF_URL, 'literature/sinka_2021_humbug.pdf', 'output/ifsc_2026_10_13_visual/Tabela_Frequencias_Sinka_2021_Completa.csv']
        nid = q.slides[sid]['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
        q.add('deleteText', {'objectId': nid, 'textRange': {'type': 'ALL'}})
        q.add('insertText', {'objectId': nid, 'text': 'Tempo sugerido: 30 segundos.\n\n'+note+'\n\n[Sources]\n'+'\n'.join(sources)+'\n[/Sources]\n'})
        additions.append({'slide_id': sid, 'title': title, 'seconds': 30, 'visible_text': q.visible[sid], 'table_rows': rows, 'speaker_notes': note, 'sources': sources, 'template_exemplar': 'ifsc_slide_09', 'media_mapping': [{'object_id': table, 'action': 'replace cell content and expand native table'}]})
    # Add the newly used article to the reference slide and retain existing links.
    refs = ['Mukundarajan et al. (2017) · celulares e áudio', 'Fanioudakis et al. (2018) · batimentos de asas e CNNs', 'Supratak et al. (2024) · MosquitoSong+', 'Sinka et al. (2021) · HumBug e frequências publicadas', 'Dados: Dryad · código e resultados: mosquito-wingbeat', 'Fotografias: CDC e Arduino · modelo: IFSC']
    sid = 'ifsc_slide_20'
    for j, text in enumerate(refs):
        if f'visual_reference_{j}' in q.elements:
            q.replace(sid, f'visual_reference_{j}', text)
            q.transform(f'visual_reference_{j}', 26, 88+j*43, 563, 35)
        else:
            q.box(sid, 'reference_5', text, 26, 88+j*43, 563, 35)
    ref_slide = next(s for s in manifest['slides'] if s['slide_id'] == sid)
    ref_slide['visible_text'] = [ref_slide['title']] + refs + ['Links completos e créditos nas notas de cada slide.']
    ref_slide['sources'].append(DOI)
    nid = q.slides[sid]['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
    q.add('deleteText', {'objectId': nid, 'textRange': {'type': 'ALL'}})
    q.add('insertText', {'objectId': nid, 'text': 'Tempo sugerido: 25 segundos.\n\n'+ref_slide['speaker_notes']+' A tabela de frequências foi adicionada de Sinka et al. (2021), Tabela 1.\n\n[Sources]\n'+'\n'.join(ref_slide['sources'])+'\n[/Sources]\n'})
    pos = old_ids.index('ifsc_slide_04')
    manifest['slides'][pos:pos] = additions
    manifest['seconds'] = sum(s['seconds'] for s in manifest['slides'])
    manifest['frequency_table_added'] = {'doi': DOI, 'table': 1, 'printed_page': 1853, 'pdf_page': 6, 'species': 15, 'original_records': 25, 'displayed_records': 15, 'selection': 'one original observation per species; no aggregation'}
    ids = [s['slide_id'] for s in manifest['slides']]
    for n, sid in enumerate(ids, 1):
        if sid == 'p1' or sid.startswith('ifsc_frequency'):continue
        old_n = old_ids.index(sid)+1
        q.replace(sid, f'visual_page_{old_n:02}', f'{n:02}', 11)
    (WORK/'populate-requests.json').write_text(json.dumps(q.items, ensure_ascii=False, indent=2))
    (WORK/'order-requests.json').write_text(json.dumps([{'updateSlidesPosition': {'slideObjectIds': [sid], 'insertionIndex': 0}} for sid in reversed(ids)], indent=2))
    (WORK/'manifest-updated.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    headers = ['species', 'age_days', 'temperature_c', 'male_hz', 'female_hz', 'original_reference']
    with (OUTPUT/'Tabela_Frequencias_Sinka_2021_Completa.csv').open('w', newline='') as f:
        w = csv.writer(f);w.writerow(headers);w.writerows(DATA)
    provenance = {'article': 'Sinka et al. (2021), HumBug – An Acoustic Mosquito Monitoring Tool for use on budget smartphones', 'doi': DOI, 'table': 1, 'printed_page': 1853, 'pdf_page': 6, 'pdf_url': PDF_URL, 'pdf_sha256': hashlib.sha256((ROOT/'literature/sinka_2021_humbug.pdf').read_bytes()).hexdigest(), 'records': [dict(zip(headers, row)) for row in DATA], 'selected_zero_based_record_indices': SELECTED, 'normalization': 'repeat species names for originally blank cells; original number notation retained in CSV; decimal comma used on slides; blank age shown as dash on slides'}
    (OUTPUT/'Tabela_Frequencias_Sinka_2021_Fontes.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    assert len(set(row[0] for row in DATA)) == 15 and len(DATA) == 25
    print(json.dumps({'requests': len(q.items), 'slides': len(ids), 'original_records': len(DATA), 'seconds': manifest['seconds']}))


if __name__ == '__main__':
    main()
