from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

def train_eval(X, y, dates):
    split_date = dates.quantile(0.75)  # letzte ~25% als Test
    tr, te = dates < split_date, dates >= split_date
    clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
    clf.fit(X[tr], y[tr])
    proba = clf.predict_proba(X[te])
    return clf, {
        "accuracy": float((proba.argmax(1)==y[te]).mean()),
        "logloss": float(log_loss(y[te], proba))
    }
