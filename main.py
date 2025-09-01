# main.py
import re
from pathlib import Path
import math
import pandas as pd

from src.features import (
    load_matches,
    add_basic_features,
    add_elo_and_rest_features,
    build_Xy,
    make_features_for_fixture,
    compute_home_win_rate,        # optional weiterhin nutzbar
    _init_features_logger,        # Logger für Kennzahlen
    home_win_stats,               # liefert beide Kennzahlen + Zähler
    performance_home_away_stats,  # Leistungs-Vergleich xG/xPTS/xP
)
from src.model import train_eval

# Am Skript verankern – Working Directory ist damit egal
BASE_DIR = Path(__file__).parent.resolve()


def collect_season_files(data_dir: str = "data", prefix: str = "D2_", min_year: int = 2018):
    """
    Sucht rekursiv nach CSVs wie D2_18-19.csv unterhalb von data_dir und nimmt nur
    Seasons mit Startjahr >= min_year.
    '18-19' -> 2018, '96-97' -> 1996 (Century-Heuristik: >=90 -> 1900er, sonst 2000er).
    """
    pat = re.compile(rf'^{re.escape(prefix)}(\d{{2}})-(\d{{2}})\.csv$', re.I)
    files = []
    for p in (BASE_DIR / data_dir).rglob("*.csv"):
        m = pat.match(p.name)
        if not m:
            continue
        start2 = int(m.group(1))
        start_year = 1900 + start2 if start2 >= 90 else 2000 + start2
        if start_year >= min_year:
            files.append((start_year, str(p.resolve())))
    files.sort(key=lambda x: x[0])  # chronologisch
    return [f for _, f in files]


def collect_files_for_run(
    data_dir: str = "data",
    prefix: str = "D2_",
    min_year: int = 2018,
    extra_dir: str = "data/performance",
):
    """
    Kombiniert:
      - alle Season-CSV nach Schema D2_YY-YY.csv (wie bisher) aus data_dir
      - ALLE *.csv aus extra_dir (z. B. data/performance/**)
    Entfernt Duplikate und sortiert.
    """
    season_files = collect_season_files(data_dir=data_dir, prefix=prefix, min_year=min_year)

    perf_root = BASE_DIR / extra_dir
    perf_files = []
    if perf_root.exists():
        perf_files = [str(p.resolve()) for p in perf_root.rglob("*.csv")]

    # Kombinieren + Duplikate entfernen (Reihenfolge beibehalten)
    combined = list(dict.fromkeys(season_files + perf_files).keys())

    print(f"Gefundene Season-CSV: {len(season_files)}")
    print(f"Gefundene Performance-CSV in '{extra_dir}': {len(perf_files)}")
    print(f"Summe CSV für Run: {len(combined)}")

    return combined


def fmt_pct(x: float) -> str:
    """Formatiert Prozent sicher, auch wenn NaN."""
    return "n/a" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2%}"


def main():
    # Kombiniert Season-Dateien + alles unter data/performance/**
    files = collect_files_for_run(
        data_dir="data",
        prefix="D2_",
        min_year=2018,
        extra_dir="data/performance",
    )

    print("Lade Dateien:", files)
    print("Anzahl Dateien:", len(files))
    if not files:
        raise SystemExit(
            "Keine passenden CSVs gefunden. Prüfe Ordnerstruktur "
            "(z. B. data/... und data/performance/...) oder passe collect_files_for_run(...) an."
        )

    # ----------------- Daten laden (Season + Performance werden im Loader vereinheitlicht) -----------------
    df = load_matches(files)

    # ----------------- Weitere Features -----------------
    df = add_basic_features(df)
    df = add_elo_and_rest_features(df)

    # ----------------- Heim-Kennzahlen + Logging -----------------
    logger = _init_features_logger()  # schreibt nach <projekt>/logs/metrics.log
    try:
        stats = home_win_stats(df, logger=logger)
    except ValueError as e:
        print("Warnung:", e)
        rate_only = compute_home_win_rate(df, logger=logger)
        stats = {
            "total_games": len(df),
            "home_wins": None,
            "away_wins": None,
            "draws": None,
            "home_win_rate": rate_only,
            "share_of_wins_that_are_home": float("nan"),
        }

    share_home = stats["share_of_wins_that_are_home"]
    share_away = (1.0 - share_home) if (share_home is not None and not math.isnan(share_home)) else float("nan")
    exists_hfa = (share_home is not None and not math.isnan(share_home) and share_home > 0.5)

    print(
        "Heim-Stats:\n"
        f"- Spiele gesamt: {stats['total_games']}\n"
        f"- Heimsiege: {stats['home_wins']}\n"
        f"- Auswärtssiege: {stats['away_wins']}\n"
        f"- Remis: {stats['draws']}\n"
        f"- Heimsieg-Wahrscheinlichkeit (Heimsiege/alle): {fmt_pct(stats['home_win_rate'])}\n"
        f"- Anteil der Siege, die Heim waren (Heimsiege/(Heim+Auswärts)): {fmt_pct(share_home)}\n"
        f"- Anteil der Siege, die Auswärts waren: {fmt_pct(share_away)}\n"
        f"- Existiert ein Heimvorteil? {'Ja' if exists_hfa else 'Nicht nach dieser Kennzahl'}"
    )

    # ----------------- Leistungs-Vergleich (xG/xPTS/xP) -----------------
    perf = performance_home_away_stats(df, logger=logger)

    def _fmt_decision(name, d):
        if not d or d["p_best"] is None or (isinstance(d["p_best"], float) and math.isnan(d["p_best"])):
            return f"{name}: keine Tests / unklar"
        flag = "Ja" if d["evidence"] else "Eher nein"
        return f"{name}: Δ={d['delta']:.3f}, best p={d['p_best']:.3g} -> Heimvorteil in Leistung? {flag}"

    print("\nLeistungs-Comparison (Home vs Away):")
    if "xg" in perf:
        print(f"- xG:   mean_home={perf['xg']['mean_home']:.3f}, mean_away={perf['xg']['mean_away']:.3f}, "
              f"Δ={perf['xg']['mean_diff_home_minus_away']:.3f}")
        print("  ", _fmt_decision("xG", perf['xg']['decision']))
    if "xpts" in perf:
        print(f"- xPTS: mean_home={perf['xpts']['mean_home']:.3f}, mean_away={perf['xpts']['mean_away']:.3f}, "
              f"Δ={perf['xpts']['mean_diff_home_minus_away']:.3f}")
        print("  ", _fmt_decision("xPTS", perf['xpts']['decision']))
    if "xp" in perf:
        print(f"- xP:   mean_home={perf['xp']['mean_home']:.3f}, mean_away={perf['xp']['mean_away']:.3f}, "
              f"Δ={perf['xp']['mean_diff_home_minus_away']:.3f}")
        print("  ", _fmt_decision("xP", perf['xp']['decision']))

    # ----------------- Modell -----------------
    X, y = build_Xy(df)
    clf, metrics = train_eval(X, y, df["Date"], calibration="sigmoid")
    print("CV-Metrics:", metrics)

    cols = X.columns

    # ----------------- Beispiel-Inferenz -----------------
    fixtures = [
        ("Holstein Kiel", "Hannover", "2025-08-30", 2.65, 3.75, 2.30),
        ("Dusseldorf", "Karlsruhe", "2025-08-30", 2.27, 3.80, 2.87),
    ]

    for home, away, when, oh, od, oa in fixtures:
        x_inf = make_features_for_fixture(df, home, away, when, oh, od, oa)
        x_inf = x_inf.reindex(columns=cols, fill_value=0.0)
        proba = clf.predict_proba(x_inf)[0]
        labels = ["H", "D", "A"]
        pred = labels[proba.argmax()]
        print(f"{when}: {home} vs {away} -> P(H/D/A) = {proba.round(3)} -> Tipp: {pred}")


if __name__ == "__main__":
    main()
