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
    compute_home_win_rate,
    _init_features_logger,
    home_win_stats,
    performance_home_away_stats,
    ensure_xpts_from_xg,
)
from src.model import train_eval

BASE_DIR = Path(__file__).parent.resolve()


def collect_season_files(data_dir: str = "data", prefix: str = "D2_", min_year: int = 2018):
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
    files.sort(key=lambda x: x[0])
    return [f for _, f in files]


def collect_files_for_run(
    data_dir: str = "data",
    prefix: str = "D2_",
    min_year: int = 2018,
    extra_dir: str = "data/performance",
):
    season_files = collect_season_files(data_dir=data_dir, prefix=prefix, min_year=min_year)

    perf_root = BASE_DIR / extra_dir
    perf_files = []
    if perf_root.exists():
        perf_files = [str(p.resolve()) for p in perf_root.rglob("*.csv")]

    combined = list(dict.fromkeys(season_files + perf_files).keys())

    print(f"Gefundene Season-CSV: {len(season_files)}")
    print(f"Gefundene Performance-CSV in '{extra_dir}': {len(perf_files)}")
    print(f"Summe CSV für Run: {len(combined)}")

    return combined


def fmt_pct(x: float) -> str:
    return "n/a" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2%}"


def _parse_code_and_season_from_filename(fn: str):
    if not isinstance(fn, str):
        return None, None
    m = re.search(r'([A-Z0-9]+)[^0-9]*?(\d{2,4})-(\d{2,4})\.csv$', fn, flags=re.I)
    if not m:
        return None, None
    code = m.group(1).upper()
    s_raw, e_raw = m.group(2), m.group(3)

    def norm_year(ystr: str) -> int:
        if len(ystr) == 4:
            return int(ystr)
        yy = int(ystr)
        return 1900 + yy if yy >= 90 else 2000 + yy

    start = norm_year(s_raw)
    end   = norm_year(e_raw)
    return code, f"{start}-{end}"


def make_season_from_date(d: pd.Timestamp, start_month: int = 8) -> str | None:
    if pd.isna(d):
        return None
    y = int(d.year)
    return f"{y}-{y+1}" if d.month >= start_month else f"{y-1}-{y}"


def _attach_season_and_league(df: pd.DataFrame) -> pd.DataFrame:
    """Hilfsfunktion: Season (aus Datum/Dateiname) + Liga-Code sauber anfügen."""
    out = df.copy()
    out["Season_date"] = out["Date"].apply(make_season_from_date) if "Date" in out.columns else None

    if "__source_file" in out.columns:
        meta = out["__source_file"].apply(_parse_code_and_season_from_filename)
        out[["__league_code", "Season_file"]] = pd.DataFrame(meta.tolist(), index=out.index)
    else:
        out["__league_code"] = None
        out["Season_file"] = None

    out["Season"] = out["Season_date"].where(out["Season_date"].notna(), out["Season_file"])
    return out


def _print_metric_hfa_summary(df: pd.DataFrame, home_col: str, away_col: str, label: str):
    """
    Generische Zusammenfassung für Home-vs-Away-Metriken (xG, xPTS, ...):
      - Overall Δ (Home−Away) + n
      - Saisonweise Δ + Coverage (n_metric / n_all)
      - Liga-gewichteter Δ über Seasons (Gewicht = n_metric je Season)
    """
    df = _attach_season_and_league(df)
    if home_col not in df.columns or away_col not in df.columns:
        print(f"\n{label}-HFA: Spalten '{home_col}'/'{away_col}' nicht gefunden.")
        return

    # Delta & Gültigkeitsmaske
    delta_col = f"__delta_{label.lower()}"
    df[delta_col] = pd.to_numeric(df[home_col], errors="coerce") - pd.to_numeric(df[away_col], errors="coerce")
    valid = df[delta_col].notna()

    # Overall
    overall_mean = df.loc[valid, delta_col].mean()
    overall_n = int(valid.sum())

    print(f"\nHeimvorteil ({label}-Basis):")
    print(f"- Overall Δ{label} (Home − Away): {overall_mean:.3f} (n={overall_n})")

    # Saisonweise
    by_season_all = (
        df.dropna(subset=["Season"])
          .groupby("Season")
          .size()
          .rename("n_all")
    )
    by_season_metric = (
        df[valid].dropna(subset=["Season"])
          .groupby("Season")[delta_col]
          .agg(n_metric="count", mean="mean")
    )
    seasons = sorted(set(by_season_all.index) | set(by_season_metric.index))
    if seasons:
        print(f"- Saisonweise Δ{label} (mit Coverage):")
        for s in seasons:
            n_all = int(by_season_all.get(s, 0))
            row = by_season_metric.loc[s] if s in by_season_metric.index else None
            if row is None or int(row["n_metric"]) == 0:
                print(f"  · {s}: Δ=— (n_{label}=0 / n_all={n_all}) – keine {label} verfügbar")
            else:
                print(f"  · {s}: Δ={row['mean']:.3f} (n_{label}={int(row['n_metric'])} / n_all={n_all})")
    else:
        print(f"- Saisonweise Δ{label}: (keine Season-Zuordnung verfügbar)")

    # Liga-gewichteter Δ über Seasons
    league_season_stats = (
        df[valid]
          .dropna(subset=["__league_code","Season"])
          .groupby(["__league_code","Season"])[delta_col]
          .agg(n_metric="count", mean="mean")
          .reset_index()
    )
    if not league_season_stats.empty:
        print(f"- Nach Liga-Code (gewichteter Δ{label} über Saisonwerte):")
        for code, block in league_season_stats.groupby("__league_code"):
            weights = block["n_metric"].astype(float)
            means   = block["mean"].astype(float)
            n_total = int(weights.sum())
            if n_total == 0:
                continue
            weighted_mean = float((weights * means).sum() / n_total)
            k_seasons = int(block.shape[0])
            print(f"  · {code}: Δ={weighted_mean:.3f} (n={n_total}, Seasons={k_seasons})")


def print_xg_hfa_summary(df: pd.DataFrame):
    _print_metric_hfa_summary(df, home_col="xg_home",  away_col="xg_away",  label="xG")


def print_xpts_hfa_summary(df: pd.DataFrame):
    # sicherstellen, dass xPTS existiert (falls nur xG vorliegt)
    if {"xpts_home","xpts_away"}.issubset(df.columns) is False:
        df = ensure_xpts_from_xg(df.copy())
    _print_metric_hfa_summary(df, home_col="xpts_home", away_col="xpts_away", label="xPTS")


def main():
    files = collect_files_for_run(
        data_dir="data",
        prefix="D2_",
        min_year=2018,
        extra_dir="data/performance",
    )

    print("Lade Dateien:", files)
    print("Anzahl Dateien:", len(files))
    if not files:
        raise SystemExit("Keine passenden CSVs gefunden.")

    df = load_matches(files)

    df = add_basic_features(df)
    df = add_elo_and_rest_features(df)

    logger = _init_features_logger()
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
        print("  ", _fmt_decision("xP", perf['xp']['decision'] if 'xp' in perf else None))

    # Neu: xG saisonweise (inkl. Coverage + liga-gewichtet)
    print_xg_hfa_summary(df)

    # xPTS saisonweise (inkl. Coverage + liga-gewichtet)
    print_xpts_hfa_summary(df)

    X, y = build_Xy(df)
    clf, metrics = train_eval(X, y, df["Date"], calibration="sigmoid")
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
        labels = ["H", "D", "A"]
        pred = labels[proba.argmax()]
        print(f"{when}: {home} vs {away} -> P(H/D/A) = {proba.round(3)} -> Tipp: {pred}")


if __name__ == "__main__":
    main()
