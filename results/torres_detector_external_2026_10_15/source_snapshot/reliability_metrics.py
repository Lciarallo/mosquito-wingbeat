"""Confusion/precision/recall/F1 metrics including explicit invalid predictions."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support,matthews_corrcoef,roc_auc_score,average_precision_score


def binary_metrics(y,p,quality,threshold):
    y=np.asarray(y);p=np.asarray(p);q=np.asarray(quality,bool);known=y>=0
    y=y[known];p=p[known];q=q[known];pred=np.where(q,(p>=threshold).astype(int),2)
    matrix=np.array([[int(((y==i)&(pred==j)).sum()) for j in [0,1,2]] for i in [0,1]])
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    two=np.unique(y).size==2
    out=dict(known_windows=len(y),unknown_windows=int((~known).sum()),invalid=int((~q).sum()),
        confusion_rows=['not_aedes','aedes'],confusion_columns=['not_aedes','aedes','invalid'],confusion=matrix.tolist(),
        true_negative=int(matrix[0,0]),false_positive=int(matrix[0,1]),false_negative=int(matrix[1,0]),true_positive=int(matrix[1,1]),
        positive_invalid=int(matrix[1,2]),negative_invalid=int(matrix[0,2]),
        aedes_precision=float(precision[1]) if (pred==1).any() else None,
        precision_informative_for_binary_detection=bool(two and (pred==1).any()),
        aedes_recall=float(recall[1]) if (y==1).any() else None,
        aedes_f1=float(f1[1]) if (y==1).any() else None,
        non_aedes_precision=float(precision[0]) if (pred==0).any() else None,
        specificity=float(recall[0]) if (y==0).any() else None,
        non_aedes_f1=float(f1[0]) if (y==0).any() else None,
        macro_f1=float(f1.mean()) if two else None,
        balanced_accuracy=float(recall.mean()) if two else None,
        emitted_false_positive_rate=float(matrix[0,1]/(y==0).sum()) if (y==0).any() else None,
        coverage=float(q.mean()) if len(q) else None,
        accuracy=float((pred==y).mean()) if len(y) else None,
        roc_auc=float(roc_auc_score(y,p)) if two else None,
        average_precision=float(average_precision_score(y,p)) if two else None,
        mcc_on_valid_audio=float(matthews_corrcoef(y[q],pred[q])) if np.unique(y[q]).size==2 else None,
        threshold=float(threshold))
    return out


def precision_at_prevalence(tpr,fpr,prevalence):
    """Scenario calculation; not an observed field precision estimate."""
    denominator=tpr*prevalence+fpr*(1-prevalence)
    return float(tpr*prevalence/denominator) if denominator else None
