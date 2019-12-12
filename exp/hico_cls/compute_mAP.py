from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics


def compute_mAP(y_true,y_score):
    num_classes = y_score.shape[1]
    APs = []
    mAP = 0
    for i in tqdm(range(num_classes)):
        ap = metrics.average_precision_score(
            y_true = y_true[:,i],
            y_score = y_score[:,i])
        
        if np.isnan(ap)==True:
            ap = 0
        
        APs.append(ap)
        mAP += ap
    
    mAP = mAP / num_classes

    return mAP, APs


def compute_mAP_given_neg_labels(y_true,y_false,y_score):
    num_classes = y_score.shape[1]
    APs = []
    mAP = 0
    ids = (y_true+y_false)==1
    for i in tqdm(range(num_classes)):
        ap = metrics.average_precision_score(
            y_true = y_true[ids[:,i],i],
            y_score = y_score[ids[:,i],i])
        
        if np.isnan(ap)==True:
            ap = 0
        
        APs.append(ap)
        mAP += ap
    
    mAP = mAP / num_classes

    return mAP, APs