from pathlib import Path
import json,hashlib,shutil,zipfile
W=Path(__file__).resolve().parent;old=W.parent/'ifsc_didatica_2026_10_15';p=json.loads((W/'raw-output.json').read_text())['structuredContent'];b=json.loads((W/'raw-before.json').read_text())['structuredContent'];plan=json.loads((W/'plan.json').read_text())
assert p['masters']==b['masters'] and p['layouts']==b['layouts'];assert len(p['slides'])==19
assert [s['objectId']for s in p['slides']]==[r['slideId']for r in plan]
updates={
'ifsc26_s06':('A STFT calcula as frequências em pequenos trechos sucessivos. O resultado é este espectrograma: o eixo horizontal indica o tempo, o vertical indica a frequência e as cores mostram a intensidade do som.','O próximo passo é transformar essa informação em uma entrada compacta para a rede.'),
'ifsc26_s07':('Agrupamos as frequências em 64 bandas mel, uma escala inspirada na percepção auditiva. Cada coluna mostra um instante. Os 55 instantes formam a sequência que a rede vai analisar.','Agora podemos entender como cada modelo usa o áudio.'),
'ifsc26_s11':('A regressão logística e as árvores em boosting analisam cada trecho. Combinamos as probabilidades, dando mais peso aos trechos em que há atividade. A primeira etapa decide se há mosquito; a segunda decide se é Aedes.','As redes neurais usam outra abordagem: acompanham a sequência do som.'),
'ifsc26_s09':('A LSTM acompanha o som em sequência, usando o que veio antes. A Bi-LSTM analisa o trecho nos dois sentidos. As duas redes têm duas camadas e produzem probabilidades para as três classes.','Vamos ver como o erro mudou durante o treinamento.'),
'ifsc26_s20':('A cascata teve a maior acurácia geral nesta comparação. As duas redes foram implementadas e avaliadas no mesmo corpus. Esses resultados se aplicam aos dados estudados; a próxima etapa é coletar gravações próprias em sessões novas e avaliar o detector sem novos ajustes. O MosquitoSong, avaliado separadamente, mostrou que a transferência entre fontes ainda precisa melhorar.','Abrir para perguntas.')}
for r in plan:
 if r['slideId']in updates:r['speech'],r['transition']=updates[r['slideId']]
 if r['slideId']=='ifsc26_s02':r['transition']='Antes dos detalhes, vamos conhecer os três modelos.'
 if r['slideId']=='ifsc26_s09':r['how']='Sequência de 55 instantes × 64 bandas mel; duas camadas recorrentes; 128 unidades por direção.'
 if r['slideId']=='ifsc26_s14':r['transition']='Com o treinamento concluído, podemos comparar os acertos.'
 if r['slideId']=='ifsc26_s15':r['transition']='A acurácia geral precisa ser lida junto com a sensibilidade e os falsos positivos.'
 if r['slideId']=='ifsc26_s10':r['transition']='A matriz de confusão detalha os erros da cascata.'
(W/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
head='APRESENTAÇÃO IFSC — REVISÃO ORAL — 15/10/2026\nLuiz Eduardo do Nascimento Ciarallo\n19 slides; roteiro estimado: 16min55s. Cronometrar no ensaio.\n\n'
script=head+'\n\n'.join(f"{r['number']:02d}. {r['title']} — {r['seconds']} s\n{r['speech']}\nTransição: {r['transition']}"for r in plan)
(W/'Roteiro_apresentacao_IFSC.txt').write_text(script)
glossary='''CONCEITOS PARA EXPLICAR EM VOZ ALTA
FFT: cálculo que separa o som em frequências e mostra a intensidade de cada uma.
STFT: aplica a análise de frequência em pequenos trechos sucessivos. O espectrograma mostra como as frequências mudam com o tempo.
Mel: escala que agrupa frequências em bandas. A rede recebe a intensidade dessas bandas em uma sequência de instantes.
Frequência fundamental: ritmo básico da vibração das asas. Harmônicos: frequências múltiplas desse ritmo. As frequências variam com sexo, temperatura e condições de gravação; as médias publicadas não são limiares fixos de espécie.
Cascata: decisões em etapas. Neste projeto usa modelos clássicos. Cascatas também podem usar redes em outros projetos.
LR: regressão logística; estima probabilidades a partir dos atributos do áudio.
HistGB: conjunto de árvores em boosting; árvores sucessivas ajudam a corrigir erros.
Agregação: combina as probabilidades dos trechos. Aqui é média ponderada por atividade, não voto majoritário.
LSTM: rede neural recorrente com memória, que usa informações anteriores da sequência.
Bi-LSTM: rede que analisa a mesma sequência nos dois sentidos, reunindo contexto anterior e posterior.
Dobra: parte dos dados reservada para avaliação em validação cruzada. São cinco dobras por grupo; cada grupo aparece no teste uma vez. Trechos de um grupo ficam juntos.
Época: um ciclo de treinamento. A época utilizada é escolhida pela menor perda de validação, não pelo método do cotovelo.
Acurácia: fração de todos os grupos classificados corretamente.
Sensibilidade Aedes: fração dos grupos Aedes detectados.
Precisão Aedes: fração dos alertas de Aedes que são Aedes de fato.
FPR: fração de negativos que geraram alerta de Aedes; calculada separadamente para ruído e outro mosquito.

RESULTADOS MOSTRADOS
Clássica: acurácia 90,03%; sensibilidade 78,33%; FPR ruído 0,00%; FPR outro 2,28%.
LSTM: acurácia 86,55%; sensibilidade 33,33%; FPR ruído 0,47%; FPR outro 0,60%.
Bi-LSTM: acurácia 85,27%; sensibilidade 28,33%; FPR ruído 0,00%; FPR outro 0,40%.
Critério: maior acurácia geral observada por arquitetura, entre as regras já avaliadas. As duas redes correspondem às linhas originalmente chamadas calibradas no CSV. É uma comparação retrospectiva; não constitui nova seleção independente pela validação. As cinco linhas originais estão preservadas em comparison.csv. Maior acurácia não significa maior sensibilidade.

DETALHES TÉCNICOS PARA PERGUNTAS
Áudio a 8 kHz; trechos de 7.936 amostras (0,992 s); FFT de 1.024 pontos com janela Hann de 128 ms e avanço de 16 ms; 64 bandas mel de 80 a 4.000 Hz; 55 instantes por trecho.
Duas camadas recorrentes com 128 unidades por direção, dropout 0,20. Adam com taxa 0,0006; lotes de 256; clipping de gradiente 3; máximo de 30 épocas; parada após cinco épocas sem melhora. Adam ajusta os pesos; dropout reduz dependências excessivas; clipping limita gradientes altos. Técnicas implementadas, sem alegação de ganho causal isolado ou ótimo global.
Contexto final: LSTM escolheu época 4 entre 9 observadas; Bi-LSTM época 2 entre 7. Cada contexto escolhe sua época na validação. Depois a rede é reiniciada e treinada no treino externo pelo número escolhido de épocas.
42.029 janelas, 1.494 grupos: 613 Dryad, 592 HumBug, 289 ESC-50. Classes: 426 ruído, 1.008 outro mosquito, 60 Aedes. Mesmas cinco dobras externas na comparação. Rede: uma divisão interna por grupo; cascata: calibração interna OOF. Nenhum arquivo/grupo/bloco PCM cruza treino e teste; sessões/dispositivos podem se repetir. Independência biológica não foi demonstrada.
MosquitoSong: somente avaliação externa retrospectiva, fora do desenvolvimento; 0/4 grupos Aedes detectados. Torres: estudo auxiliar separado sobre infecção em fêmeas Aedes; não foi misturado para produzir 90,03%. Áudio privado do Drive é enxame e está excluído do treino, teste e comparações correntes.
MC dropout com 100 passagens existe na inferência auxiliar; os resultados OOF principais são determinísticos.
Arduino: montagem física mostrada nas fotos; aquisição padronizada em WAV e dataset próprio são etapas futuras. Não foi demonstrada execução das redes na placa.

'''
guide=head+glossary+'GUIA POR SLIDE\n\n'+'\n\n'.join(f"{r['number']:02d}. {r['title']}\nO que é: {r['concept']}\nPara que serve: {r['function']}\nPor que usamos: {r['why']}\nComo foi obtido: {r['how']}\nFala: {r['speech']}\nTransição: {r['transition']}"for r in plan)
(W/'Guia_de_estudo_e_fala.txt').write_text(guide)
check='''VERIFICAÇÃO DOS REQUISITOS — REVISÃO ORAL — 15/10/2026
Autor: Luiz Eduardo do Nascimento Ciarallo
Fonte: Trabalho_Descrição-1.pdf fornecido pelo autor, páginas 2–4; análise completa preservada na entrega anterior.

ATENDIDO: estudar técnicas e implementar rede profunda numa aplicação específica. LSTM e Bi-LSTM reais, duas camadas, código e resultados em results/ternary_lstm_2026_10_15/. Cascata serve como comparação clássica.
ATENDIDO: contextualização (slides 2–3), trabalhos correlatos (5), proposta implementada (6–13), resultados (13–16), discussão final (19).
ATENDIDO: autoria individual, nome completo na capa. Modelo oficial IFSC preservado, com master e 11 layouts intactos. Logotipo oficial, imagens com fontes, espectrogramas e fotos do protótipo preservados.
COMPATÍVEL: exposição de 15–20 minutos. Roteiro dos 19 slides soma 16min55s; confirmar em ensaio.
NÃO VERIFICÁVEL: execução do cronograma em três aulas.
PENDENTE: envio ao professor; material entregue ao autor. Nenhum e-mail enviado.

A pedido do autor, Adam e regularização não têm slides próprios; fundamentos e configurações continuam no guia e nas notas de treinamento. Três quadrados introdutórios apresentam cascata, LSTM e Bi-LSTM. Termos FFT, STFT, mel, etapas e agregação esclarecidos. Resultados exibem maior acurácia observada de cada arquitetura com métricas correspondentes e critério retrospectivo documentado. Não houve retreino nem alteração dos resultados. Não foram demonstrados superioridade estatística, generalização em campo ou execução neural na placa.

VALIDAÇÃO: 19 páginas renderizadas e inspecionadas individualmente; sem recortes, títulos partidos ou sobreposições não intencionais. Verificador estrutural: zero ocorrências. Tabela confrontada com comparison.csv. Marca, autoria, três conceitos e duas fotos conferidos. A disciplina não exige superar a cascata, obter 90% com redes, detectar infecção ou portar a rede ao Arduino.
'''
(W/'Verificacao_dos_requisitos.txt').write_text(check)
name='IFSC_Mosquitos_Revisao_Oral_Luiz_Eduardo_do_Nascimento_Ciarallo.pdf';shutil.copy2(W/'renders/review/presentation.pdf',W/name)
verification={'slides':19,'seconds':1015,'master_unchanged':True,'layouts_unchanged':True,'visual_qa':'Todas as 19 páginas inspecionadas individualmente; sem defeitos concretos observados.','pdf_sha256':hashlib.sha256((W/name).read_bytes()).hexdigest(),'selection':json.loads((W/'selection_summary.json').read_text())}
(W/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
# Updated speaker notes are prepared as a single final text repair; retain sources.
req=[]
for r,s in zip(plan,p['slides']):
 oid=s['slideProperties']['notesPage']['notesProperties']['speakerNotesObjectId']
 notes=r['notes_full']+'\n\nFala revisada: '+r['speech']+'\nTransição revisada: '+r['transition']
 req.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'text':notes,'insertionIndex':0}}])
(W/'notes-final.json').write_text(json.dumps(req,ensure_ascii=False))
print(json.dumps(verification,ensure_ascii=False))
