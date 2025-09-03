# src/features.py
from __future__ import annotations
import logging
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd

# =============================================================================
# Logging
# =============================================================================

def _init_features_logger(log_dir: str | Path = None) -> logging.Logger:
    logger = logging.getLogger("features_metrics")
    if logger.handlers:
        return logger
    if log_dir is None:
        log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "metrics.log"

    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Logger initialisiert. Logfile: %s", log_path)
    return logger


# =============================================================================
# Loader-Helfer
# =============================================================================

def _to_num(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(
        s.astype(str)
         .str.replace("\u2009", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(",", ".", regex=False)
         .str.replace(r"[^0-9\.\-]", "", regex=True),
        errors="coerce"
    )

def _parse_dates_any(s: pd.Series) -> pd.Series:
    """Robuste Datumserkennung: ISO ‘YYYY-mm-dd’ bevorzugt, sonst dayfirst=True."""
    vals = s.astype(str).str.strip()
    iso_mask = vals.str.match(r"^\d{4}-\d{2}-\d{2}$")
    if iso_mask.mean() >= 0.8:
        return pd.to_datetime(vals, errors="coerce", format="%Y-%m-%d")
    # fallback: häufig EU-Format
    return pd.to_datetime(vals, errors="coerce", dayfirst=True)

def _normalize_result_from_scores(hg: pd.Series, ag: pd.Series) -> pd.Series:
    return np.where(hg > ag, "H", np.where(hg < ag, "A", "D"))

def _cols_lower_map(df: pd.DataFrame): return {c.lower(): c for c in df.columns}

# Kandidatenlisten
HOME_GOAL_COLS = ["fthg","homegoals","home_goals","hg","home_score","home_ftg","hgoals","score_x","goals_home"]
AWAY_GOAL_COLS = ["ftag","awaygoals","away_goals","ag","away_score","away_ftg","agoals","score_y","goals_away"]
RESULT_COLS    = ["ftr","result","ft_result","full_time_result","outcome"]

_XG_HOME_COLS  = ["xg_home","home_xg","hxg","xg_h","xgh","xg_x","homexg","xg (h)","xg (home)"]
_XG_AWAY_COLS  = ["xg_away","away_xg","axg","xg_a","xga","xg_y","awayxg","xg (a)","xg (away)"]
_XPTS_HOME_COLS= ["xpts_home","home_xpts","xpoints_home","xp_home","xph","xpts (h)","xpts_x","exppts_home","exp_points_home"]
_XPTS_AWAY_COLS= ["xpts_away","away_xpts","xpoints_away","xp_away","xpa","xpts (a)","xpts_y","exppts_away","exp_points_away"]
_XP_HOME_COLS  = ["xp_home","home_xp","xpoints_home","exppts_home","exp_points_home","xpoints_h","xp (h)","xp_x"]
_XP_AWAY_COLS  = ["xp_away","away_xp","xpoints_away","exppts_away","exp_points_away","xpoints_a","xp (a)","xp_y"]

def _regex_find_pairs(df: pd.DataFrame, key="xg"):
    names = list(df.columns)
    low   = [n.lower() for n in names]
    home_idx = [i for i, n in enumerate(low) if re.search(rf"\b{key}\b", n) and re.search(r"(home|^h[^a-z]?|_h\b|\(h\)|-home|\bhome\b)", n)]
    away_idx = [i for i, n in enumerate(low) if re.search(rf"\b{key}\b", n) and re.search(r"(away|^a[^a-z]?|_a\b|\(a\)|-away|\baway\b)", n)]
    if home_idx and away_idx:
        return names[home_idx[0]], names[away_idx[0]]
    x_idx = [i for i, n in enumerate(low) if re.search(rf"\b{key}\b", n) and re.search(r"(_x\b|\(x\))", n)]
    y_idx = [i for i, n in enumerate(low) if re.search(rf"\b{key}\b", n) and re.search(r"(_y\b|\(y\))", n)]
    if x_idx and y_idx:
        return names[x_idx[0]], names[y_idx[0]]
    return None, None

def _pick_col(cols_map, candidates):
    for c in candidates:
        c = c.lower()
        if c in cols_map:
            return cols_map[c]
    return None

def _find_metric_pair(df: pd.DataFrame, home_cands: list[str], away_cands: list[str], key: str):
    cm = _cols_lower_map(df)
    ch = _pick_col(cm, home_cands)
    ca = _pick_col(cm, away_cands)
    if ch and ca:
        return ch, ca, "candidates"
    rh, ra = _regex_find_pairs(df, key=key)
    if rh and ra:
        return rh, ra, "regex"
    key_cols = [orig for orig, lo in zip(df.columns, [c.lower() for c in df.columns]) if key in lo]
    if len(key_cols) == 2:
        return key_cols[0], key_cols[1], "heuristic(2match)"
    return None, None, None


# =============================================================================
# Long-Format (eine Zeile pro Team) -> Match-Format
# =============================================================================

def _is_home_flag(venue: str | float) -> float:
    if isinstance(venue, str):
        v = venue.strip().lower()
        if v in ("h","home","heim"):
            return 1.0
        if v in ("a","away","auswärts","away "):
            return 0.0
    return np.nan

def _pair_key_from_long(row: pd.Series) -> str:
    t, o = str(row["Team"]), str(row["Opponent"])
    d = row["Date"]
    return (t if t < o else o) + "___" + (o if t < o else t) + "___" + str(d)

def _try_build_from_long_per_team(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    cm = _cols_lower_map(df)

    team = next((cm[k] for k in ["team","mannschaft","squad"] if k in cm), None)
    opp  = next((cm[k] for k in ["opponent","gegner","opp"] if k in cm), None)
    date = next((cm[k] for k in ["date","match_date","datum"] if k in cm), None)
    venue= next((cm[k] for k in ["venue","homeaway","ha","ort"] if k in cm), None)

    xg_for  = next((cm[k] for k in ["xg_for","xg (for)","expectedgoalsfor","xgf","team_xg","xg"] if k in cm), None)
    xg_against = next((cm[k] for k in ["xg_against","xg (against)","expectedgoalsagainst","xga","opp_xg","xg_allowed"] if k in cm), None)

    gf = next((cm[k] for k in ["goalsfor","gf","goals for","team_goals","goals"] if k in cm), None)
    ga = next((cm[k] for k in ["goalsagainst","ga","goals against","opp_goals","against"] if k in cm), None)
    score = next((cm[k] for k in ["score","result","endstand"] if k in cm), None)

    if not (team and opp and date):
        return pd.DataFrame()

    tmp = pd.DataFrame()
    tmp["Team"] = df[team].astype(str)
    tmp["Opponent"] = df[opp].astype(str)
    tmp["Date"] = _parse_dates_any(df[date])
    tmp["home_flag_row"] = df[venue].apply(_is_home_flag) if venue else np.nan

    tmp["xg_for"] = _to_num(df[xg_for]) if xg_for else np.nan
    tmp["xg_against"] = _to_num(df[xg_against]) if xg_against else np.nan

    tmp["gf"] = _to_num(df[gf]) if gf else np.nan
    tmp["ga"] = _to_num(df[ga]) if ga else np.nan

    if score and tmp["gf"].isna().all() and tmp["ga"].isna().all():
        sc = df[score].astype(str)
        m = sc.str.extract(r"(?P<h>\d+)\D+(?P>a>\d+)")
        tmp["gf"] = _to_num(m["h"])
        tmp["ga"] = _to_num(m["a"])

    tmp["pair_key"] = tmp.apply(_pair_key_from_long, axis=1)

    rows = []
    for k, block in tmp.groupby("pair_key"):
        if len(block) < 2:
            continue
        b = block.sort_values(["home_flag_row"], ascending=False)
        h = b.iloc[0]
        a = b.iloc[1]
        out = {
            "Date": h["Date"],
            "HomeTeam": h["Team"],
            "AwayTeam": a["Team"],
            "FTHG": h["gf"] if not pd.isna(h["gf"]) else a["ga"],
            "FTAG": a["gf"] if not pd.isna(a["gf"]) else h["ga"],
        }
        if pd.notna(out["FTHG"]) and pd.notna(out["FTAG"]):
            out["FTR"] = _normalize_result_from_scores(pd.Series([out["FTHG"]]), pd.Series([out["FTAG"]]))[0]
        else:
            out["FTR"] = np.nan

        out["xg_home"] = h["xg_for"] if not pd.isna(h["xg_for"]) else a["xg_against"]
        out["xg_away"] = a["xg_for"] if not pd.isna(a["xg_for"]) else h["xg_against"]

        rows.append(out)

    if not rows:
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    logger.info("Legacy Long-Format erkannt und gepivotet: %d Matches", len(res))
    return res


# =============================================================================
# Loader: Matches
# =============================================================================

def _to_match_schema(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Vereinheitlicht auf: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR (+ optional xg_home/xg_away, xpts_*)
    Unterstützt:
      1) Klassisches Season-Format (HomeTeam/AwayTeam/FTHG/FTAG/FTR[/Date])
      2) Performance-Format wide (Home/Away + Score_x/Score_y + xG_x/xG_y)
      3) Legacy-Format long (Team/Opponent/Venue + xG_for/xG_against + GoalsFor/GoalsAgainst)
      4) Multi-Section CSVs (Match + Players): wir extrahieren den Match-Block
    """
    cols = {c.lower(): c for c in df.columns}

    # (1) Klassisches Season-Format
    if all(k in cols for k in ["hometeam", "awayteam", "fthg", "ftag"]) and "ftr" in cols:
        out = df.rename(columns={
            cols["hometeam"]: "HomeTeam",
            cols["awayteam"]: "AwayTeam",
            cols["fthg"]: "FTHG",
            cols["ftag"]: "FTAG",
            cols["ftr"]: "FTR",
        }).copy()
        date_col = cols.get("date") or cols.get("match_date") or cols.get("datum")
        out["Date"] = _parse_dates_any(out[date_col]) if date_col else pd.NaT
        # evtl. xG already
        xg_h = cols.get("xg_home") or cols.get("xg (home)") or cols.get("xg_h")
        xg_a = cols.get("xg_away") or cols.get("xg (away)") or cols.get("xg_a")
        if xg_h: out["xg_home"] = _to_num(df[xg_h])
        if xg_a: out["xg_away"] = _to_num(df[xg_a])
        return out

    # (2) Performance-Format wide
    if "home" in cols and "away" in cols:
        out = pd.DataFrame()
        out["HomeTeam"] = df[cols["home"]].astype(str)
        out["AwayTeam"] = df[cols["away"]].astype(str)

        fthg = cols.get("score_x") or cols.get("home_score") or cols.get("fthg") or cols.get("home_goals")
        ftag = cols.get("score_y") or cols.get("away_score") or cols.get("ftag") or cols.get("away_goals")
        if fthg and ftag:
            out["FTHG"] = _to_num(df[fthg])
            out["FTAG"] = _to_num(df[ftag])
            out["FTR"] = _normalize_result_from_scores(out["FTHG"], out["FTAG"])
        else:
            logger.warning("Performance-CSV ohne Score-Spalten; Datei wird übersprungen.")
            return pd.DataFrame()

        xg_h = cols.get("xg_x") or cols.get("xg_home") or cols.get("xg (home)") or cols.get("xg_h")
        xg_a = cols.get("xg_y") or cols.get("xg_away") or cols.get("xg (away)") or cols.get("xg_a")
        if xg_h is not None:
            out["xg_home"] = _to_num(df[xg_h])
        if xg_a is not None:
            out["xg_away"] = _to_num(df[xg_a])

        xpts_h = cols.get("xpts_home") or cols.get("xp_home") or cols.get("exp_points_home")
        xpts_a = cols.get("xpts_away") or cols.get("xp_away") or cols.get("exp_points_away")
        if xpts_h is not None:
            out["xpts_home"] = _to_num(df[xpts_h])
        if xpts_a is not None:
            out["xpts_away"] = _to_num(df[xpts_a])

        date_col = cols.get("date") or cols.get("match_date") or cols.get("datum")
        out["Date"] = _parse_dates_any(df[date_col]) if date_col else pd.NaT
        return out

    # (3) Legacy Long-Format
    legacy = _try_build_from_long_per_team(df, logger)
    if not legacy.empty:
        return legacy

    # Kein Treffer
    logger.info("CSV-Format nicht als Match erkannt. Erste Spalten: %s", list(df.columns)[:12])
    return pd.DataFrame()


def load_matches(files: list[str]) -> pd.DataFrame:
    logger = _init_features_logger()
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            raw = pd.read_csv(f, low_memory=False)
        except Exception as e:
            logger.warning("CSV konnte nicht gelesen werden: %s (%s)", f, e)
            continue

        mapped = _to_match_schema(raw, logger)

        lower_cols = [c.lower() for c in raw.columns]
        is_player_table = ("player_name" in lower_cols) or ("player" in lower_cols)
        if mapped.empty and is_player_table:
            logger.info("Überspringe Player-CSV ohne Match-Block: %s", f)
            continue

        if not mapped.empty:
            mapped["__source_file"] = Path(f).name
            frames.append(mapped)
        else:
            logger.info("Datei ohne verwertbare Match-Struktur verworfen: %s", f)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    if "Date" not in df.columns:
        df["Date"] = pd.NaT
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    required = ["HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]
    df = (df.sort_values("Date")
            .dropna(subset=[c for c in required if c in df.columns])
            .copy())

    df = df.assign(
        HomePoints = df["FTR"].map({"H": 3, "D": 1, "A": 0}),
        AwayPoints = df["FTR"].map({"H": 0, "D": 1, "A": 3}),
        GD         = pd.to_numeric(df["FTHG"], errors="coerce") - pd.to_numeric(df["FTAG"], errors="coerce"),
    )

    df = ensure_xpts_from_xg(df)  # ergänzt xPTS nur bei vorhandenen xG
    return df


# =============================================================================
# Basis-Features
# =============================================================================

def implied_probs_row(row, h, d, a):
    ph = 1 / row[h]
    pdraw = 1 / row[d]
    pa = 1 / row[a]
    s = ph + pdraw + pa
    return pd.Series({"b365_H": ph/s, "b365_D": pdraw/s, "b365_A": pa/s})

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    if set(["B365H","B365D","B365A"]).issubset(df.columns):
        mask = df[["B365H","B365D","B365A"]].notna().all(axis=1)
        df.loc[mask, ["b365_H","b365_D","b365_A"]] = df.loc[mask].apply(
            implied_probs_row, axis=1, args=("B365H","B365D","B365A")
        )

    df["home_form5_pts"] = (
        df.groupby("HomeTeam")["HomePoints"]
          .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    df["away_form5_pts"] = (
        df.groupby("AwayTeam")["AwayPoints"]
          .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    df["form5_pts_diff"] = df["home_form5_pts"] - df["away_form5_pts"]

    pid = np.where(
        df["HomeTeam"] < df["AwayTeam"],
        df["HomeTeam"] + "___" + df["AwayTeam"],
        df["AwayTeam"] + "___" + df["HomeTeam"],
    )
    df["pair_id"] = pid
    df = df.reset_index(drop=True)

    h2h = []
    for i, r in df.iterrows():
        g = df[(df["pair_id"] == r["pair_id"]) & (df.index < i)].tail(3)
        pts = []
        for _, rr in g.iterrows():
            pts.append({ "H":3, "D":1, "A":0 }[rr["FTR"]] if rr["HomeTeam"] == r["HomeTeam"]
                       else { "H":0, "D":1, "A":3 }[rr["FTR"]])
        h2h.append(np.mean(pts) if pts else np.nan)
    df["h2h_home_pts_last3"] = h2h

    df["home_flag"] = 1
    return df

def build_Xy(df: pd.DataFrame):
    feats = [
        "b365_H","b365_D","b365_A",
        "form5_pts_diff","h2h_home_pts_last3",
        "elo_diff","rest_diff",
        "home_flag",
    ]
    X = df[feats].fillna(0.0)
    y = df["FTR"].map({"H":0, "D":1, "A":2})
    return X, y


# =============================================================================
# Elo + Resttage
# =============================================================================

def add_elo_and_rest_features(df: pd.DataFrame, K=20.0, HFA=60.0) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    last_date: dict[str, pd.Timestamp] = {}

    home_pre, away_pre = [], []
    rest_h, rest_a = [], []

    for _, r in df.iterrows():
        h, a, d = r["HomeTeam"], r["AwayTeam"], r["Date"]
        eh = ratings.get(h, 1500.0)
        ea = ratings.get(a, 1500.0)

        home_pre.append(eh)
        away_pre.append(ea)

        rest_h.append((d - last_date[h]).days if h in last_date else np.nan)
        rest_a.append((d - last_date[a]).days if a in last_date else np.nan)

        exp_home = 1.0 / (1.0 + 10.0 ** (-((eh + HFA) - ea) / 400.0))
        score_home = 1.0 if r["FTR"] == "H" else 0.5 if r["FTR"] == "D" else 0.0

        ratings[h] = eh + K * (score_home - exp_home)
        ratings[a] = ea + K * ((1.0 - score_home) - (1.0 - exp_home))

        last_date[h] = d
        last_date[a] = d

    df["elo_home_pre"] = home_pre
    df["elo_away_pre"] = away_pre
    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]

    df["rest_home_days"] = rest_h
    df["rest_away_days"] = rest_a
    df["rest_diff"] = df["rest_home_days"].fillna(0) - df["rest_away_days"].fillna(0)

    return df


def make_features_for_fixture(df_hist: pd.DataFrame, home: str, away: str, when, oddsH, oddsD, oddsA,
                              K=20.0, HFA=60.0) -> pd.DataFrame:
    when = pd.to_datetime(when)
    hist = df_hist[df_hist["Date"] < when].sort_values("Date")

    ratings: dict[str, float] = {}
    last_date: dict[str, pd.Timestamp] = {}
    for _, r in hist.iterrows():
        h, a, d = r["HomeTeam"], r["AwayTeam"], r["Date"]
        eh = ratings.get(h, 1500.0)
        ea = ratings.get(a, 1500.0)
        exp_home = 1.0 / (1.0 + 10.0 ** (-((eh + HFA) - ea) / 400.0))
        s_home = 1.0 if r["FTR"] == "H" else 0.5 if r["FTR"] == "D" else 0.0
        ratings[h] = eh + K * (s_home - exp_home)
        ratings[a] = ea + K * ((1.0 - s_home) - (1.0 - exp_home))
        last_date[h] = d
        last_date[a] = d

    h_elo = ratings.get(home, 1500.0)
    a_elo = ratings.get(away, 1500.0)
    elo_diff = h_elo - a_elo

    rest_home = (when - last_date[home]).days if home in last_date else np.nan
    rest_away = (when - last_date[away]).days if away in last_date else np.nan
    rest_diff = (0 if pd.isna(rest_home) else rest_home) - (0 if pd.isna(rest_away) else rest_away)

    ph, pdraw, pa = 1/oddsH, 1/oddsD, 1/oddsA
    s = ph + pdraw + pa
    bH, bD, bA = ph/s, pdraw/s, pa/s

    home_hist = hist[hist["HomeTeam"] == home]["HomePoints"].shift(1).rolling(5, min_periods=1).mean()
    away_hist = hist[hist["AwayTeam"] == away]["AwayPoints"].shift(1).rolling(5, min_periods=1).mean()
    home_form5 = home_hist.iloc[-1] if len(home_hist) else np.nan
    away_form5 = away_hist.iloc[-1] if len(away_hist) else np.nan
    form5_diff = (home_form5 - away_form5) if pd.notna(home_form5) and pd.notna(away_form5) else 0.0

    pid = home + "___" + away if home < away else away + "___" + home
    g = hist[hist["pair_id"] == pid].tail(3)
    pts = []
    for _, rr in g.iterrows():
        pts.append({ "H":3, "D":1, "A":0 }[rr["FTR"]] if rr["HomeTeam"] == home
                   else { "H":0, "D":1, "A":3 }[rr["FTR"]])
    h2h3 = np.mean(pts) if pts else 0.0

    return pd.DataFrame([{
        "b365_H": bH, "b365_D": bD, "b365_A": bA,
        "form5_pts_diff": form5_diff,
        "h2h_home_pts_last3": h2h3,
        "elo_diff": elo_diff,
        "rest_diff": rest_diff,
        "home_flag": 1
    }])


# =============================================================================
# Kennzahlen: Heimsieg-Quoten
# =============================================================================

def _lower_map(columns): return {c.lower(): c for c in columns}

def _derive_home_win_series(df: pd.DataFrame, logger: logging.Logger) -> pd.Series:
    cm = _lower_map(df.columns)
    h = next((cm[c] for c in HOME_GOAL_COLS if c in cm), None)
    a = next((cm[c] for c in AWAY_GOAL_COLS if c in cm), None)
    r = next((cm[c] for c in RESULT_COLS    if c in cm), None)

    if h is not None and a is not None:
        s = pd.to_numeric(df[h], errors="coerce") > pd.to_numeric(df[a], errors="coerce")
        logger.info("Heimsiege aus Toren berechnet (%s vs %s).", h, a)
        return s
    if r is not None:
        sr = df[r].astype(str).str.upper().str.strip()
        s = sr.isin(["H","HOME","1"])
        logger.info("Heimsiege aus Ergebnis-Spalte berechnet (%s).", r)
        return s

    msg = "Konnte Heimsiege nicht ableiten – keine passenden Spalten gefunden."
    logger.error(msg)
    raise ValueError(msg)

def compute_home_win_rate(df: pd.DataFrame, logger: logging.Logger | None = None) -> float:
    logger = logger or _init_features_logger()
    n_total = int(len(df))
    s = _derive_home_win_series(df, logger)
    n_wins = int(s.sum())
    rate = n_wins / n_total if n_total else float("nan")
    logger.info("compute_home_win_rate: total=%d, home_wins=%d, rate=%.6f", n_total, n_wins, rate)
    return rate

def share_of_wins_that_are_home(df: pd.DataFrame, logger: logging.Logger | None = None) -> float:
    logger = logger or _init_features_logger()
    cm = _lower_map(df.columns)
    h = next((cm[c] for c in HOME_GOAL_COLS if c in cm), None)
    a = next((cm[c] for c in AWAY_GOAL_COLS if c in cm), None)
    r = next((cm[c] for c in RESULT_COLS    if c in cm), None)

    if h is not None and a is not None:
        hg = pd.to_numeric(df[h], errors="coerce")
        ag = pd.to_numeric(df[a], errors="coerce")
        home_wins = int((hg > ag).sum())
        away_wins = int((ag > hg).sum())
    elif r is not None:
        sr = df[r].astype(str).str.upper().str.strip()
        home_wins = int(sr.isin(["H","HOME","1"]).sum())
        away_wins = int(sr.isin(["A","AWAY","2"]).sum())
    else:
        msg = "Konnte Anteil-Heimsiege nicht berechnen – keine passenden Spalten gefunden."
        logger.error(msg)
        raise ValueError(msg)

    basis = home_wins + away_wins
    share = home_wins / basis if basis else float("nan")
    logger.info("share_of_wins_that_are_home: home=%d, away=%d, basis=%d, share=%.6f",
                home_wins, away_wins, basis, share)
    return share

def home_win_stats(df: pd.DataFrame, logger: logging.Logger | None = None) -> dict:
    logger = logger or _init_features_logger()
    s = _derive_home_win_series(df, logger)

    total_games = int(len(df))
    home_wins = int(s.sum())

    cm = _lower_map(df.columns)
    h = next((cm[c] for c in HOME_GOAL_COLS if c in cm), None)
    a = next((cm[c] for c in AWAY_GOAL_COLS if c in cm), None)
    r = next((cm[c] for c in RESULT_COLS    if c in cm), None)

    if h is not None and a is not None:
        away_wins = int((pd.to_numeric(df[a], errors="coerce") > pd.to_numeric(df[h], errors="coerce")).sum())
        draws = int((pd.to_numeric(df[a], errors="coerce") == pd.to_numeric(df[h], errors="coerce")).sum())
    elif r is not None:
        sr = df[r].astype(str).str.upper().str.strip()
        away_wins = int(sr.isin(["A","AWAY","2"]).sum())
        draws = int(sr.isin(["D","DRAW","X"]).sum())
    else:
        away_wins = None
        draws = None

    home_rate = home_wins / total_games if total_games else float("nan")
    if away_wins is not None:
        basis = home_wins + away_wins
        share_home_wins = home_wins / basis if basis else float("nan")
    else:
        share_home_wins = float("nan")

    logger.info(
        "home_win_stats: total=%d, home=%d, away=%s, draws=%s, home_rate=%.6f, share_home=%.6f",
        total_games, home_wins, str(away_wins), str(draws), home_rate, share_home_wins
    )

    return {
        "total_games": total_games,
        "home_wins": home_wins,
        "away_wins": away_wins,
        "draws": draws,
        "home_win_rate": home_rate,
        "share_of_wins_that_are_home": share_home_wins,
    }


# =============================================================================
# Performance-Stats (xG/xPTS/xP)
# =============================================================================

def _ttest_paired(x, y):
    try:
        from scipy.stats import ttest_rel
        stat, p = ttest_rel(x, y, nan_policy="omit")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")

def _wilcoxon_signed(x, y):
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(x - y, zero_method="wilcox", correction=False, alternative="two-sided", nan_policy="omit")  # type: ignore
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")

def performance_home_away_stats(df: pd.DataFrame, logger: logging.Logger | None = None) -> dict:
    logger = logger or _init_features_logger()
    out: dict[str, dict] = {}

    def _calc_pair(metric_key, home_cands, away_cands):
        ch, ca, how = _find_metric_pair(df, home_cands, away_cands, key=metric_key)
        if not ch or not ca:
            logger.warning("Keine %s-Home/Away-Spalten erkannt.", metric_key.upper())
            return None
        H = pd.to_numeric(df[ch], errors="coerce")
        A = pd.to_numeric(df[ca], errors="coerce")
        diff = H - A
        t_stat, t_p = _ttest_paired(H, A)
        w_stat, w_p = _wilcoxon_signed(H, A)
        res = {
            "n": int(diff.notna().sum()),
            "mean_home": float(H.mean(skipna=True)),
            "mean_away": float(A.mean(skipna=True)),
            "mean_diff_home_minus_away": float(diff.mean(skipna=True)),
            "t_paired_stat": t_stat, "t_paired_p": t_p,
            "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
            "detected_home_col": ch, "detected_away_col": ca, "detected_via": how,
        }
        logger.info("%s via %s (%s vs %s) n=%d Δ=%.4f t-p=%.4g w-p=%.4g",
                    metric_key.upper(), how, ch, ca, res["n"], res["mean_diff_home_minus_away"], t_p, w_p)
        return res

    xg_res = _calc_pair("xg", _XG_HOME_COLS, _XG_AWAY_COLS)
    if xg_res: out["xg"] = xg_res

    xpts_res = _calc_pair("xpts", _XPTS_HOME_COLS, _XPTS_AWAY_COLS)
    if xpts_res:
        out["xpts"] = xpts_res
    else:
        xp_res = _calc_pair("xp", _XP_HOME_COLS, _XP_AWAY_COLS)
        if xp_res: out["xp"] = xp_res

    def _decision(metric_key):
        if metric_key not in out: return None
        d = out[metric_key]
        delta = d["mean_diff_home_minus_away"]
        p_vals = [p for p in [d["t_paired_p"], d["wilcoxon_p"]] if not math.isnan(p)]
        p_best = min(p_vals) if p_vals else float("nan")
        return dict(delta=delta, p_best=p_best, evidence=(delta > 0 and p_best < 0.05))

    for k in list(out.keys()):
        out[k]["decision"] = _decision(k)

    return out


# =============================================================================
# xPTS aus xG – robust
# =============================================================================

def ensure_xpts_from_xg(
    df: pd.DataFrame,
    logger: logging.Logger | None = None,
    win_points: float = 3.0,
    draw_points: float = 1.0,
    tie_eps: float = 1e-6,
    create_delta_col: bool = True,
    force_recompute: bool = False,
) -> pd.DataFrame:
    logger = logger or _init_features_logger()

    ch, ca, how = _find_metric_pair(df, _XG_HOME_COLS, _XG_AWAY_COLS, key="xg")
    if not ch or not ca:
        logger.warning("ensure_xpts_from_xg: Keine xG-Home/Away-Spalten erkannt – keine xPTS-Ableitung möglich.")
        return df

    xg_h = pd.to_numeric(df[ch], errors="coerce")
    xg_a = pd.to_numeric(df[ca], errors="coerce")
    mask_valid = xg_h.notna() & xg_a.notna()
    if not mask_valid.any():
        logger.info("ensure_xpts_from_xg: Keine Zeilen mit validen xG gefunden – keine xPTS ergänzt.")
        if create_delta_col and {"xpts_home", "xpts_away"}.issubset(df.columns):
            df["xpts_delta_home_minus_away"] = pd.to_numeric(df["xpts_home"], errors="coerce") - pd.to_numeric(df["xpts_away"], errors="coerce")
        return df

    if "xpts_home" not in df.columns or force_recompute:
        df["xpts_home"] = np.nan
    if "xpts_away" not in df.columns or force_recompute:
        df["xpts_away"] = np.nan

    mask_fill_h = mask_valid & (force_recompute | df["xpts_home"].isna())
    mask_fill_a = mask_valid & (force_recompute | df["xpts_away"].isna())
    mask_fill   = mask_fill_h | mask_fill_a
    if not mask_fill.any():
        if create_delta_col:
            df["xpts_delta_home_minus_away"] = pd.to_numeric(df["xpts_home"], errors="coerce") - pd.to_numeric(df["xpts_away"], errors="coerce")
        return df

    diff = xg_h - xg_a
    h_win = diff >  tie_eps
    a_win = diff < -tie_eps
    draw  = ~(h_win | a_win)

    home_new = pd.Series(np.nan, index=df.index, dtype="float64")
    away_new = pd.Series(np.nan, index=df.index, dtype="float64")

    home_new[mask_fill & h_win] = win_points
    home_new[mask_fill & a_win] = 0.0
    home_new[mask_fill & draw]  = draw_points

    away_new[mask_fill & a_win] = win_points
    away_new[mask_fill & h_win] = 0.0
    away_new[mask_fill & draw]  = draw_points

    if mask_fill_h.any():
        df.loc[mask_fill_h, "xpts_home"] = home_new[mask_fill_h]
    if mask_fill_a.any():
        df.loc[mask_fill_a, "xpts_away"] = away_new[mask_fill_a]

    if create_delta_col:
        xh = pd.to_numeric(df["xpts_home"], errors="coerce")
        xa = pd.to_numeric(df["xpts_away"], errors="coerce")
        have_both = xh.notna() & xa.notna()
        delta = pd.Series(np.nan, index=df.index, dtype="float64")
        delta[have_both] = (xh - xa)[have_both]
        df["xpts_delta_home_minus_away"] = delta

    n_done = int(mask_fill_h.sum() + mask_fill_a.sum())
    logger.info("ensure_xpts_from_xg: xPTS ergänzt/gesetzt für %d Werte. Delta-Spalte=%s",
                n_done, "ja" if create_delta_col else "nein")
    return df
