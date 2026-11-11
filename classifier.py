"""Train a classical baseline and a hybrid quantum fidelity-feature classifier."""
import numpy as np
import pandas as pd
from quantum import features

FEATURE_COLUMNS = ["balance", "age", "pdays", "previous", "campaign"]

def encode(train, other):
    numeric = train.select_dtypes(include="number").columns
    categoricals = train.columns.difference(numeric)
    mean = train[numeric].mean()
    scale = train[numeric].std().replace(0, 1)
    train_numeric = ((train[numeric].fillna(mean)-mean)/scale).clip(-5,5)
    other_numeric = ((other[numeric].fillna(mean)-mean)/scale).clip(-5,5)
    train_cat = pd.get_dummies(train[categoricals].fillna("missing"), dtype=float)
    other_cat = pd.get_dummies(other[categoricals].fillna("missing"), dtype=float)
    other_cat = other_cat.reindex(columns=train_cat.columns, fill_value=0)
    return train_numeric, other_numeric, np.c_[train_numeric,train_cat], np.c_[other_numeric,other_cat]

def probability(x, coef):
    return 1/(1+np.exp(-np.clip(x @ coef,-35,35)))

def logistic(train, labels, test, penalty):
    train = np.c_[np.ones(len(train)),train]
    test = np.c_[np.ones(len(test)),test]
    coef = np.zeros(train.shape[1])
    reg = np.eye(train.shape[1])*penalty
    reg[0,0] = 0
    for _ in range(30):
        p = probability(train,coef)
        hessian = train.T @ ((p*(1-p))[:,None]*train) + reg
        update = np.linalg.solve(hessian,train.T @ (p-labels)+reg @ coef)
        coef -= update
        if np.max(np.abs(update)) < 1e-8:
            break
    return probability(test,coef)

def predict(train, labels, other, seed):
    a,b,x,z = encode(train,other)
    f = features(a[FEATURE_COLUMNS].to_numpy())
    g = features(b[FEATURE_COLUMNS].to_numpy())
    landmarks = np.random.default_rng(seed).choice(len(train),min(len(train),48),replace=False)
    anchor = f[landmarks].conj().T
    baseline = logistic(x,labels,z,3)
    hybrid = logistic(np.c_[x,np.abs(f @ anchor)**2],labels,
                      np.c_[z,np.abs(g @ anchor)**2],3)
    return baseline,hybrid

def metrics(labels,p):
    order = np.argsort(-p,kind="stable")
    positives = labels.sum()
    ranks = labels[order]
    positive = p[labels == 1, None]
    negative = p[None,labels == 0]
    return dict(roc_auc=float(np.mean((positive > negative)+0.5*(positive == negative))),
                average_precision=float(np.sum(np.cumsum(ranks)/(np.arange(len(ranks))+1)*ranks)/positives),
                brier=float(np.mean((p-labels)**2)),
                log_loss=float(-np.mean(labels*np.log(p)+(1-labels)*np.log1p(-p))),
                precision_at_10=float(ranks[:10].mean()),recall_at_10=float(ranks[:10].sum()/positives))

def validate(train, labels, seed):
    rng = np.random.default_rng(seed)
    folds = [np.array_split(rng.permutation(np.flatnonzero(labels==target)),5) for target in (0,1)]
    predictions = np.empty((2,len(train)))
    for fold in range(5):
        test = np.r_[folds[0][fold],folds[1][fold]]
        fit = np.setdiff1d(np.arange(len(train)),test)
        predictions[:,test] = predict(train.iloc[fit],labels[fit],train.iloc[test],seed+fold)
    return {name:metrics(labels,p) for name,p in zip(("baseline","hybrid"),predictions)}

