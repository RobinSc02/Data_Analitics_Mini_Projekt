# src/features.py
import pandas as pd, numpy as np

def load_matches(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').dropna(subset=['HomeTeam','AwayTeam','FTR','FTHG','FTAG'])
    df['HomePoints'] = df['FTR'].map({'H':3,'D':1,'A':0})
    df['AwayPoints'] = df['FTR'].map({'H':0,'D':1,'A':3})
    df['GD'] = df['FTHG'] - df['FTAG']
    return df

def implied_probs_row(row, h, d, a):
    ph = 1 / row[h]
    pdraw = 1 / row[d]   # <-- nicht 'pd' nennen!
    pa = 1 / row[a]
    s = ph + pdraw + pa
    return pd.Series({'b365_H': ph/s, 'b365_D': pdraw/s, 'b365_A': pa/s})


def add_basic_features(df):
    # Implied probs (einfach normalisiert)
    mask = df[['B365H','B365D','B365A']].notna().all(axis=1)
    df.loc[mask, ['b365_H','b365_D','b365_A']] = df.loc[mask].apply(
        implied_probs_row, axis=1, args=('B365H','B365D','B365A')
    )

    # Form (rolling 5, ohne Leakage via shift)
    df['home_form5_pts'] = (
    df.groupby('HomeTeam')['HomePoints']
      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    df['away_form5_pts'] = (
    df.groupby('AwayTeam')['AwayPoints']
      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    df['form5_pts_diff'] = df['home_form5_pts'] - df['away_form5_pts']

    # Pair-ID für H2H
    pid = np.where(df['HomeTeam'] < df['AwayTeam'],
                   df['HomeTeam']+'___'+df['AwayTeam'],
                   df['AwayTeam']+'___'+df['HomeTeam'])
    df['pair_id'] = pid
    df = df.reset_index(drop=True)

    # H2H-Punkte aus Sicht des aktuellen Heimteams (letzte 3)
    h2h = []
    for i, r in df.iterrows():
        g = df[(df['pair_id']==r['pair_id']) & (df.index < i)].tail(3)
        pts = []
        for _, rr in g.iterrows():
            pts.append({'H':3,'D':1,'A':0}[rr['FTR']] if rr['HomeTeam']==r['HomeTeam']
                       else {'H':0,'D':1,'A':3}[rr['FTR']])
        h2h.append(np.mean(pts) if pts else np.nan)
    df['h2h_home_pts_last3'] = h2h

    df['home_flag'] = 1
    return df

def build_Xy(df):
    feats = [
        'b365_H','b365_D','b365_A',
        'form5_pts_diff','h2h_home_pts_last3',
        'elo_diff','rest_diff',
        'home_flag'
    ]
    X = df[feats].fillna(0.0)
    y = df['FTR'].map({'H':0,'D':1,'A':2})
    return X, y


# -------- Inferenz-Featurebau für ein kommendes Spiel (kein Leakage) --------
def make_features_for_fixture(df_hist, home, away, when, oddsH, oddsD, oddsA):
    """
    df_hist: vollständige History mit Features (aus add_basic_features) bis VOR 'when'
    home, away: Teamnamen, exakt wie in df
    when: pd.Timestamp des geplanten Spiels
    odds*: Decimal-Quoten (z.B. 2.10, 3.50, 3.60)
    """
    when = pd.to_datetime(when)
    hist = df_hist[df_hist['Date'] < when].copy()

    # implied probs
    ph, pd_, pa = 1/oddsH, 1/oddsD, 1/oddsA
    s = ph + pd_ + pa
    bH, bD, bA = ph/s, pd_/s, pa/s

    # Form-Features (nur Spiele des jeweiligen Teams VOR 'when')
    # Wir bilden die "HomePoints/AwayPoints"-Serien analog zur Trainingslogik.
    home_hist = hist[hist['HomeTeam']==home]['HomePoints'].shift(1).rolling(5, min_periods=1).mean()
    away_hist = hist[hist['AwayTeam']==away]['AwayPoints'].shift(1).rolling(5, min_periods=1).mean()
    home_form5 = home_hist.iloc[-1] if len(home_hist) else np.nan
    away_form5 = away_hist.iloc[-1] if len(away_hist) else np.nan
    form5_diff = (home_form5 - away_form5) if pd.notna(home_form5) and pd.notna(away_form5) else np.nan

    # H2H (letzte 3)
    pid = home+'___'+away if home < away else away+'___'+home
    g = hist[hist['pair_id']==pid].tail(3)
    pts = []
    for _, rr in g.iterrows():
        pts.append({'H':3,'D':1,'A':0}[rr['FTR']] if rr['HomeTeam']==home
                   else {'H':0,'D':1,'A':3}[rr['FTR']])
    h2h3 = np.mean(pts) if pts else np.nan

    row = {
        'b365_H': bH, 'b365_D': bD, 'b365_A': bA,
        'form5_pts_diff': form5_diff if pd.notna(form5_diff) else 0.0,
        'h2h_home_pts_last3': h2h3 if pd.notna(h2h3) else 0.0,
        'home_flag': 1
    }
    return pd.DataFrame([row])

# --- Elo + Resttage ----------------------------------------------------------
def add_elo_and_rest_features(df, K=20.0, HFA=60.0):
    """
    Fügt pro Spiel die PRE-Match Elo-Werte (ohne Leakage) + Resttage hinzu.
    K: Update-Faktor, HFA: Home-Field-Advantage in Elo-Punkten (nur im Erwartungswert).
    """
    import numpy as np
    df = df.sort_values('Date').reset_index(drop=True)

    ratings = {}      # team -> current Elo
    last_date = {}    # team -> last match date

    home_pre, away_pre = [], []
    rest_h, rest_a = [], []

    for _, r in df.iterrows():
        h, a, d = r['HomeTeam'], r['AwayTeam'], r['Date']
        eh = ratings.get(h, 1500.0)
        ea = ratings.get(a, 1500.0)

        home_pre.append(eh)
        away_pre.append(ea)

        rest_h.append((d - last_date[h]).days if h in last_date else np.nan)
        rest_a.append((d - last_date[a]).days if a in last_date else np.nan)

        # Erwartungswert inkl. Home-Advantage nur in der Erwartung
        exp_home = 1.0 / (1.0 + 10.0 ** (-((eh + HFA) - ea) / 400.0))
        res = r['FTR']
        score_home = 1.0 if res == 'H' else 0.5 if res == 'D' else 0.0

        ratings[h] = eh + K * (score_home - exp_home)
        ratings[a] = ea + K * ((1.0 - score_home) - (1.0 - exp_home))

        last_date[h] = d
        last_date[a] = d

    df['elo_home_pre'] = home_pre
    df['elo_away_pre'] = away_pre
    df['elo_diff'] = df['elo_home_pre'] - df['elo_away_pre']

    df['rest_home_days'] = rest_h
    df['rest_away_days'] = rest_a
    df['rest_diff'] = df['rest_home_days'].fillna(0) - df['rest_away_days'].fillna(0)

    return df

def make_features_for_fixture(df_hist, home, away, when, oddsH, oddsD, oddsA, K=20.0, HFA=60.0):
    """
    Baut Features für ein kommendes Spiel NUR aus Historie < when.
    Rechnet Elo/Resttage on-the-fly (keine Leakage).
    """
    import numpy as np
    import pandas as pd

    when = pd.to_datetime(when)
    hist = df_hist[df_hist['Date'] < when].sort_values('Date')

    # ---- Elo/Resttage bis 'when' hochrechnen
    ratings, last_date = {}, {}
    for _, r in hist.iterrows():
        h, a, d = r['HomeTeam'], r['AwayTeam'], r['Date']
        eh = ratings.get(h, 1500.0)
        ea = ratings.get(a, 1500.0)
        exp_home = 1.0 / (1.0 + 10.0 ** (-((eh + HFA) - ea) / 400.0))
        s_home = 1.0 if r['FTR'] == 'H' else 0.5 if r['FTR'] == 'D' else 0.0
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

    # ---- Implied probabilities aus Quoten
    ph, pdraw, pa = 1/oddsH, 1/oddsD, 1/oddsA
    s = ph + pdraw + pa
    bH, bD, bA = ph/s, pdraw/s, pa/s

    # ---- Form/H2H wie bisher
    # Form (rolling 5, VOR 'when')
    home_hist = hist[hist['HomeTeam'] == home]['HomePoints'].shift(1).rolling(5, min_periods=1).mean()
    away_hist = hist[hist['AwayTeam'] == away]['AwayPoints'].shift(1).rolling(5, min_periods=1).mean()
    home_form5 = home_hist.iloc[-1] if len(home_hist) else np.nan
    away_form5 = away_hist.iloc[-1] if len(away_hist) else np.nan
    form5_diff = (home_form5 - away_form5) if pd.notna(home_form5) and pd.notna(away_form5) else 0.0

    pid = home+'___'+away if home < away else away+'___'+home
    g = hist[hist['pair_id'] == pid].tail(3)
    pts = []
    for _, rr in g.iterrows():
        pts.append({'H':3,'D':1,'A':0}[rr['FTR']] if rr['HomeTeam']==home else {'H':0,'D':1,'A':3}[rr['FTR']])
    h2h3 = np.mean(pts) if pts else 0.0

    return pd.DataFrame([{
        'b365_H': bH, 'b365_D': bD, 'b365_A': bA,
        'form5_pts_diff': form5_diff,
        'h2h_home_pts_last3': h2h3,
        'elo_diff': elo_diff,
        'rest_diff': rest_diff,
        'home_flag': 1
    }])

