# main.py
import pandas as pd
from src.features import load_matches, add_basic_features, add_elo_and_rest_features, build_Xy, make_features_for_fixture
from src.model import train_eval
from pathlib import Path
import re

def collect_season_files(data_dir="data", prefix="D2_", min_year=2018):
    """
    Sucht CSVs wie D2_18-19.csv im data/-Ordner und nimmt nur Seasons mit Startjahr >= min_year.
    '18-19' -> 2018, '96-97' -> 1996 (Century-Heuristik: >=90 -> 1900er, sonst 2000er).
    """
    pat = re.compile(rf'^{re.escape(prefix)}(\d{{2}})-(\d{{2}})\.csv$', re.I)
    files = []
    for p in Path(data_dir).glob("*.csv"):
        m = pat.match(p.name)
        if not m:
            continue
        start2 = int(m.group(1))
        start_year = 1900 + start2 if start2 >= 90 else 2000 + start2
        if start_year >= min_year:
            files.append((start_year, str(p)))
    files.sort(key=lambda x: x[0])  # chronologisch
    return [f for _, f in files]

FILES = collect_season_files("data", prefix="D2_", min_year=2018)
print("Lade Dateien:", FILES)

def main():
    
    df = load_matches(FILES)
    df = add_basic_features(df)
    df = add_elo_and_rest_features(df)

    X, y = build_Xy(df)
    clf, metrics = train_eval(X, y, df['Date'], calibration='sigmoid')
    print("CV-Metrics:", metrics)

    cols = X.columns  

    fixtures = [
        ("Holstein Kiel", "Hannover", "2025-08-30", 2.65, 3.75, 2.30),
        ("Dusseldorf", "Karlsruhe", "2025-08-30", 2.27, 3.80, 2.87),
    ]

    for home, away, when, oh, od, oa in fixtures:
        x_inf = make_features_for_fixture(df, home, away, when, oh, od, oa)
        x_inf = x_inf.reindex(columns=cols, fill_value=0.0)
        proba = clf.predict_proba(x_inf)[0]       
        labels = ['H', 'D', 'A']
        pred = labels[proba.argmax()]
        print(f"{when}: {home} vs {away} -> P(H/D/A) = {proba.round(3)} -> Tipp: {pred}")

if __name__ == "__main__":
    main()
