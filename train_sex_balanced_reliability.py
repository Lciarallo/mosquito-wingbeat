"""New adaptation: known female negatives, selection at actual frozen cut.

The prior validation female recordings now become TRAINING feedback. They no
longer count as independent validation. No private, UFRGS or CDMSC adaptation.
"""
import json,time
import numpy as np,pandas as pd,torch,joblib
from scipy.special import logit,expit
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from train_reproduction import ROOT,now,sha
from fetch_reliability_data import OUT as BASE
from prepare_reliability_data import CACHE
from reliability_models import JointDetector
from reliability_metrics import binary_metrics
from corrected_validation import assert_disjoint
from aedes_binary import targets,weights
from robust_reproduction_ensemble import source_class_recalls

OUT=ROOT/'results/sex_balanced_reliability_2026_10_01'


def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False))


def load_encoder(device):
    path=BASE/'models/pcen_large.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
    model=JointDetector('pcen_large',np.zeros(138),np.ones(138)).to(device);model.load_state_dict(saved['state_dict']);model.eval()
    return model


def embeddings(model,f,w,device):
    out=np.empty((len(f),768),dtype='float32')
    with torch.no_grad():
        for b in ['long','short']:
            ix=np.flatnonzero(f.bucket.eq(b));wi=f.iloc[ix].wave_index.astype(int).to_numpy()
            for start in range(0,len(ix),64):
                k=ix[start:start+64];a=torch.tensor(np.array(w[b][wi[start:start+64]]),device=device)
                out[k]=model.project(model.encoder(model.frontend(a))).cpu().numpy()
    return out


def cut(frame,p,partition):
    if partition!='calibration':raise ValueError('Test cannot choose threshold.')
    y=targets(frame);cuts=np.unique(np.r_[0,np.quantile(p,np.linspace(0,1,301)),1.000001]);best=None
    for c in cuts:
        m=binary_metrics(y,p,np.ones(len(y),bool),c);cells=source_class_recalls(frame,p,c)
        prec=m['aedes_precision'] or 0;score=min(prec,m['aedes_recall'],min(cells.values()))+.02*m['macro_f1']
        if best is None or score>best[0]:best=(score,float(c),m,cells)
    return best[1],dict(precision=best[2]['aedes_precision'],recall=best[2]['aedes_recall'],macro_f1=best[2]['macro_f1'],minimum_source_class_recall=min(best[3].values()))


def main():
    OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    if (OUT/'freeze.json').exists():raise FileExistsError('Do not retune frozen adaptation.')
    f=pd.read_csv(BASE/'segments.csv',low_memory=False);female=f.domain.eq('mosquitosong')&f.split.eq('validation')
    adapted_groups=f.loc[female,'group'].unique().tolist();assert len(adapted_groups)==3 and not (targets(f.loc[female])==1).any()
    f.loc[female,'split']='train';assert_disjoint(f);f.to_csv(OUT/'segments.csv',index=False)
    (OUT/'additional_selection.json').write_bytes((BASE/'additional_selection.json').read_bytes())
    (OUT/'device_files_before_sha256.json').write_bytes((BASE/'device_files_before_sha256.json').read_bytes())
    (OUT/'original_weights_sha256.json').write_bytes((BASE/'original_weights_sha256.json').read_bytes())
    write('protocol.json',dict(created_utc=now(),feedback='Three female negative MosquitoSong groups previously used for validation now training, NEVER independent validation.',adapted_groups=adapted_groups,
        configurations=['ExtraTrees500 depth16','ExtraTrees500 unlimited','HistGB300','Frozen9M-PCEN embeddings plus acoustic features and two logistic heads C=.1'],
        no_private_training=True,no_ufrgs_or_cdmsc_training=True,calibration='Platt calibration and cut maximize minimum precision/recall/source-class recall + .02 macroF1, calibration only.',
        model_selection='Same score on validation AT EACH CANDIDATE FROZEN CALIBRATION CUT, not raw probability .5. No final test used for model selection.',
        validation_limitation='No independent MosquitoSong female negative group remains; validation sources Dryad, HumBug and ESC50.',
        reliability_gate=dict(precision_min=.6,recall_min=.6,macro_f1_min=.6,specificity_min=.6,bootstrap_lower95_min=.6,every_negative_taxon_specificity_min=.6),
        original_and_prior_checkpoints_preserved=True,arduino_compiled=False))
    fresh_protocol=json.loads((BASE/'fresh_protocol.json').read_text());fresh_protocol['created_utc']=now();write('fresh_protocol.json',fresh_protocol)
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    w={b:np.load(CACHE/(b+'_8k.npy'),mmap_mode='r') for b in ['long','short']};x=np.load(CACHE/'features.npy',mmap_mode='r')
    frames={s:f[f.split.eq(s)&f.quality_ok&(targets(f)>=0)] for s in ['train','validation','calibration']};tr,v,c=[frames[s] for s in frames]
    needed=f[f.split.isin(frames)&f.quality_ok&(targets(f)>=0)];embfile=ROOT/'data/cache/sex_balanced_reliability_embeddings.npy'
    if not embfile.exists():
        print('Extracting frozen PCEN representations',len(needed),flush=True);model=load_encoder(device)
        e=embeddings(model,needed,w,device);full=np.full((len(f),768),np.nan,dtype='float32');full[needed.index]=e;np.save(embfile,full);del model
        torch.cuda.empty_cache()
    e=np.load(embfile,mmap_mode='r');deep=np.concatenate([x,e],1);ti=tr.index.to_numpy();tw=weights(tr);records=[]
    for name in ['trees_regularized_female','trees_deep_female','hist_boost_female','pcen_embedding_logistic']:
        if name.startswith('trees'):
            make=lambda:ExtraTreesClassifier(n_estimators=500,max_depth=16 if name=='trees_regularized_female' else None,min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=20261015)
        elif name.startswith('hist'):make=lambda:HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=31,l2_regularization=3,learning_rate=.05,early_stopping=False,random_state=20261015)
        else:make=lambda:LogisticRegression(C=.1,max_iter=2000)
        matrix=deep if name=='pcen_embedding_logistic' else x
        scaler=StandardScaler().fit(matrix[ti],sample_weight=tw) if name=='pcen_embedding_logistic' else None
        xx=scaler.transform(matrix[needed.index]) if scaler else matrix[needed.index]
        by_index=pd.Series(np.arange(len(needed)),index=needed.index)
        def select(index):return xx[by_index.loc[index].to_numpy()]
        yi=targets(tr);pi=tr.label.ne('noise').to_numpy().astype(int);pw=tw.copy()
        for label in [0,1]:pw[pi==label]/=pw[pi==label].sum()
        print('Training',name,len(tr),flush=True);pm=make().fit(select(ti),pi,sample_weight=pw/pw.mean());mask=pi==1
        sw=weights(tr.loc[mask]);sm=make().fit(select(ti)[mask],yi[mask],sample_weight=sw)
        def predict(index):
            a=select(index);return np.clip(pm.predict_proba(a)[:,1]*sm.predict_proba(a)[:,1],1e-6,1-1e-6)
        pc=predict(c.index);cal=LogisticRegression(C=10,max_iter=2000).fit(logit(pc)[:,None],targets(c),sample_weight=weights(c))
        slope,bias=float(cal.coef_[0,0]),float(cal.intercept_[0]);assert slope>0
        threshold,cm=cut(c,expit(slope*logit(pc)+bias),'calibration');pv=expit(slope*logit(predict(v.index))+bias)
        vm=binary_metrics(targets(v),pv,np.ones(len(v),bool),threshold);cells=source_class_recalls(v,pv,threshold)
        score=min(vm['aedes_precision'] or 0,vm['aedes_recall'],min(cells.values()))+.02*vm['macro_f1']
        path=OUT/f'models/{name}.joblib';joblib.dump(dict(name=name,presence_model=pm,species_model=sm,scaler=scaler,embedding=name=='pcen_embedding_logistic',slope=slope,bias=bias,cut=threshold),path)
        r=dict(name=name,selection_score=score,validation_precision=vm['aedes_precision'],validation_recall=vm['aedes_recall'],validation_macro_f1=vm['macro_f1'],validation_minimum_recall=min(cells.values()),validation_source_class_recalls=cells,cut=threshold,calibration=cm,sha256=sha(path))
        records.append(r);print('Fitted',json.dumps(r),flush=True)
    write('validation_search.json',records);chosen=max(records,key=lambda r:r['selection_score']);write('selection.json',dict(selected_model=chosen['name'],selected_before_new_test=True,**{k:chosen[k] for k in ['selection_score','validation_precision','validation_recall','validation_minimum_recall']}))
    files=list((OUT/'models').iterdir())+[BASE/'models/pcen_large.pt']
    write('freeze.json',dict(frozen_utc=now(),models_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},source_sha256={p:sha(ROOT/p) for p in ['train_sex_balanced_reliability.py','reliability_models.py','reliability_metrics.py','results/sex_balanced_reliability_2026_10_01/segments.csv']},protocol_sha256=sha(OUT/'protocol.json'),selection_sha256=sha(OUT/'selection.json'),arduino_compiled=False))
    print('ADAPTATION FROZEN',chosen['name'],flush=True)


if __name__=='__main__':main()
