# src/model.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

def train_eval(X, y, dates, n_splits=5, calibration='isotonic'):
    # Sortierung nach Datum – **DataFrame/Series behalten**, keine .values
    order = np.argsort(dates.values, kind='mergesort')  # stabil
    Xo, yo = X.iloc[order], y.iloc[order]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, lls, brs = [], [], []

    for tr_idx, te_idx in tscv.split(Xo):
        Xtr, Xte = Xo.iloc[tr_idx], Xo.iloc[te_idx]
        ytr, yte = yo.iloc[tr_idx], yo.iloc[te_idx]

        # multi_class-Param weglassen (Default ist künftig 'multinomial')
        base = LogisticRegression(max_iter=2000, class_weight='balanced')
        # Tipp: Bei kleinen Datensätzen ist 'sigmoid' oft stabiler als 'isotonic'
        clf = CalibratedClassifierCV(base, method=calibration, cv=3)
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xte)
        accs.append((proba.argmax(1) == yte.to_numpy()).mean())
        lls.append(log_loss(yte, proba))
        brs.append(np.mean([
            brier_score_loss((yte == k).astype(int), proba[:, k]) for k in [0, 1, 2]
        ]))

    base_final = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf_final = CalibratedClassifierCV(base_final, method=calibration, cv=5)
    clf_final.fit(Xo, yo)

    metrics = {
        "cv_splits": n_splits,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "logloss_mean": float(np.mean(lls)),
        "logloss_std": float(np.std(lls)),
        "brier_mean": float(np.mean(brs)),
        "brier_std": float(np.std(brs)),
    }
    return clf_final, metrics
