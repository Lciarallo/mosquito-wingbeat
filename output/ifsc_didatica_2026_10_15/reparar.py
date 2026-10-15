from pathlib import Path
import json
W=Path(__file__).resolve().parent
values={'ifsc26_s05_e00':'Frequência fundamental e harmônicos','ifsc26_s20_e00':'Resultados, limites e próxima coleta','ifsc26_s12_didatica_subtitulo':'Entropia cruzada mede o erro; Adam atualiza os pesos.','tern_adam_node0':'Perda\nmede o erro','tern_adam_node1':'Gradientes\nindicam ajustes','tern_adam_node2':'Adam\nadapta passos','tern_adam_node3':'Atualização\ndos pesos','ifsc26_s14_didatica_subtitulo':'Época: ciclo de treino · checkpoint: estado salvo dos pesos.','ifsc26_s10_e01':'Sensibilidade: fração de Aedes detectados · FPR: taxa de falso positivo.\nDenominadores: 60 Aedes; 426 ruídos; 1.008 outros mosquitos.','ifsc26_s09_e01':'Softmax converte os escores em três probabilidades que somam 1.\nDuas camadas · 128 unidades por direção · dropout 0,20.'}
r=[]
for oid,value in values.items():r.extend([{'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},{'insertText':{'objectId':oid,'insertionIndex':0,'text':value}}])
r.extend([{'createShape':{'objectId':'ifsc26_logo_area_respiro','shapeType':'RECTANGLE','elementProperties':{'pageObjectId':'ifsc26_s01','size':{'width':{'magnitude':184,'unit':'PT'},'height':{'magnitude':251,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':401,'translateY':45,'unit':'PT'}}}},{'updateShapeProperties':{'objectId':'ifsc26_logo_area_respiro','shapeProperties':{'shapeBackgroundFill':{'solidFill':{'color':{'rgbColor':{'red':1,'green':1,'blue':1}},'alpha':1}},'outline':{'propertyState':'NOT_RENDERED'}},'fields':'shapeBackgroundFill,outline'}},{'updatePageElementsZOrder':{'pageElementObjectIds':['ifsc26_logo_area_respiro'],'operation':'SEND_TO_BACK'}}])
(W/'repair.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
# Keep authoring source and learning material aligned with the approved shorter titles.
p=W/'conteudo.py';t=p.read_text().replace('O voo contém uma fundamental e seus harmônicos','Frequência fundamental e harmônicos').replace('O estudo compara desempenho e orienta a coleta','Resultados, limites e próxima coleta');p.write_text(t)
print(json.dumps(r,ensure_ascii=False,separators=(',',':')))
