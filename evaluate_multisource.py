"""Evaluate both frozen CNNs on identical holdouts and an auxiliary benchmark."""
from pathlib import Path
import json
import numpy as np,pandas as pd,torch
from scipy.special import softmax,expit
from large_audio_model import load_model
from train_multisource import bucket_predict,weighted_metrics
from improve_device import atomic_json,file_hash
ROOT=Path(__file__).resolve().parent;O=ROOT/'results/multisource'
def run():
 torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
 all_rows=[];all_classes=[];allpred=[]
 frame=pd.read_csv(O/'segments.csv',low_memory=False);test=np.flatnonzero(frame.split.isin(['test','challenge']))
 bench=pd.read_csv(O/'biodcase_segments.csv',low_memory=False)
 for name,path in [('large_cnn',ROOT/'results/large_cnn/models/large_cnn.pt'),('multisource_cnn',O/'models/multisource_cnn.pt')]:
  before=file_hash(path);model,s=load_model(path,device)
  for kind,f,ix,cache in [('heldout',frame,test,'multisource'),('auxiliary',bench,np.arange(len(bench)),'biodcase_benchmark')]:
   waves={b:np.load(ROOT/f'data/cache/{cache}/{b}.npy',mmap_mode='r') for b in ['long','short']}
   l,g=bucket_predict(model,f,ix,waves,device);p=softmax(l/s['temperature'],axis=1);g=expit(s['presence_slope']*g+s['presence_bias'])
   rows,classes,pred=weighted_metrics(f.iloc[ix].reset_index(drop=True),p,g,s['classes'],s['presence_threshold'],s['species_threshold'],name)
   for row in rows:row['evaluation_kind']=kind
   all_rows.extend(rows);all_classes.extend(classes);allpred.append(pred)
   pred.to_csv(O/f'{name}_{kind}_predictions.csv.gz',index=False,compression=dict(method='gzip',mtime=0))
   # Negative UFRGS references only establish not-Aedes; neither exact species
   # nor presence is known from their folder names.
   negatives=pred[pred.label.eq('not_aedes')]
   if len(negatives):
    atomic_json(O/f'{name}_ufrgs_negative.json',dict(windows=len(negatives),files=negatives.path.nunique(),reference='not_Aedes_aegypti_only',forced_Aedes_windows=int(negatives.forced_species.eq('Aedes aegypti').sum()),emitted_Aedes_windows=int(negatives.identified_species.eq('Aedes aegypti').sum()),other_species_outputs_not_evaluated_as_correct_or_wrong=True))
  assert file_hash(path)==before
  del model
 pd.DataFrame(all_rows).to_csv(O/'comparison.csv',index=False);pd.DataFrame(all_classes).to_csv(O/'per_class_comparison.csv',index=False)
 print(pd.DataFrame(all_rows)[['model','domain','known_windows','forced_window_accuracy','known_coverage','accepted_known_accuracy']].to_string(index=False),flush=True)
 atomic_json(O/'paired_evaluation_audit.json',dict(models_unchanged=True,same_windows_and_duration_buckets=True,ufrgs_no_training_validation_calibration=True,biodcase_auxiliary_only_ancestry_not_fully_resolved=True))
if __name__=='__main__':run()
