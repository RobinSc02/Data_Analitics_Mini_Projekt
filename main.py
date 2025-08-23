from src.features import load_matches, add_basic_features, build_Xy
from src.model import train_eval

FILES = ["data/D2_24-25.csv", "data/D2_25-26.csv"]  # nach Bedarf anpassen

def main():
    df = load_matches(FILES)
    df = add_basic_features(df)
    X, y = build_Xy(df)
    clf, metrics = train_eval(X, y, df['Date'])
    print("Metrics:", metrics)

    # Beispiel-Vorhersage: Quote anpassen!
    import numpy as np
    sample = {
        'b365_H': 1/2.0, 'b365_D': 1/3.5, 'b365_A': 1/3.6,
        'form5_pts_diff': 0.4, 'h2h_home_pts_last3': 1.0, 'home_flag': 1
    }
    import pandas as pd
    P = clf.predict_proba(pd.DataFrame([sample]))[0]
    print("P(H/D/A):", np.round(P,3))

if __name__ == "__main__":
    main()
