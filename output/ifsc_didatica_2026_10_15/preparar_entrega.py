from pathlib import Path
import hashlib,json,zipfile,shutil,datetime
W=Path(__file__).resolve().parent;ROOT=W.parents[1];U=Path('/tmp/mosquito_didatica_upload_2026_10_15');U.mkdir(exist_ok=True)
receipt=json.loads((W/'verification.json').read_text());shutil.copyfile(W/receipt['pdf'],U/'IFSC_Mosquitos_Introducao_Luiz_Eduardo_do_Nascimento_Ciarallo.pdf')
for name in ['Guia_de_estudo_e_fala.txt','Roteiro_apresentacao_IFSC.txt','Verificacao_dos_requisitos.txt']:shutil.copyfile(W/name,U/name)
prior=(ROOT/'tmp/drive-entrega-2026-10-15/readme-antes-didatica.txt').read_text();assert 'MOSQUITO-WINGBEAT' in prior
prior+='''\nREVISÃO DIDÁTICA E VISUAL — 15/10/2026\nVERSÃO ATUAL: PDF avulso e slides editáveis atualizados,20slides e roteiro de18min. A foto do campus foi substituída pela marca oficial IFSC. Sequência: problema,dados,processamento,modelos,treinamento,resultados e coleta. Conceitos e transições explicados nos slides e nas notas. Fotos reais do protótipo preservadas.\nGuia_de_estudo_e_fala.txt: conceito,função,motivação,método,fala sugerida e perguntas para revisão.\nRoteiro_apresentacao_IFSC.txt: fala e transições com tempos sugeridos.\nVerificacao_dos_requisitos.txt: auditoria do enunciado,implementação,slides e itens pendentes.\nAtualizacao_didatica_IFSC_2026-10-15.zip: arquivos desta revisão,logo oficial,gráficos,notas e evidências. Este pacote é posterior ao ZIP principal e à atualização de fotos do protótipo. O PDF avulso é a versão mais recente. Não houve novo treino nem alteração das métricas. O envio ao professor continua pendente.\n'''
for a,b in [('atualizados,20slides','atualizados, 20 slides'),('de18min','de 18 min'),('problema,dados,processamento,modelos,treinamento,resultados','problema, dados, processamento, modelos, treinamento, resultados'),('conceito,função,motivação,método,fala','conceito, função, motivação, método, fala'),('enunciado,implementação,slides','enunciado, implementação, slides'),('revisão,logo oficial,gráficos,notas','revisão, logo oficial, gráficos, notas')]:prior=prior.replace(a,b)
(U/'LEIA-ME.txt').write_text(prior)
entries=[]
for p in sorted(W.rglob('*')):
 if p.is_file() and 'renders' not in p.relative_to(W).parts:entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(W/'manifesto_entrega.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries),indent=2,ensure_ascii=False))
package=U/'Atualizacao_didatica_IFSC_2026-10-15.zip'
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for e in entries:z.write(ROOT/e['path'],e['path'])
 z.write(W/'manifesto_entrega.json',str((W/'manifesto_entrega.json').relative_to(ROOT)))
with zipfile.ZipFile(package) as z:
 assert z.testzip() is None
 for e in entries:
  with z.open(e['path']) as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==e['sha256']
print('Checked',len(entries),'files in supplement; ZIP bytes',package.stat().st_size)
