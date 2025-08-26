import pandas as pd, numpy as np

def load_matches(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').dropna(subset=['HomeTeam','AwayTeam','FTR','FTHG','FTAG'])
    df['HomePoints'] = df['FTR'].map({'H':3,'D':1,'A':0})
    df['AwayPoints'] = df['FTR'].map({'H':0,'D':1,'A':3})
    df['GD'] = df['FTHG'] - df['FTAG']
    return df

def implied_probs(row, h, d, a):
    ph, pd, pa = 1/row[h], 1/row[d], 1/row[a]
    s = ph+pd+pa
    return pd.Series({'b365_H':ph/s,'b365_D':pd/s,'b365_A':pa/s})

def add_basic_features(df):
    mask = df[['B365H','B365D','B365A']].notna().all(axis=1)
    df.loc[mask, ['b365_H','b365_D','b365_A']] = df.loc[mask].apply(
        implied_probs, axis=1, args=('B365H','B365D','B365A')
    )
    # Form (rolling 5, ohne Leakage)
    df['home_form5_pts'] = df.groupby('HomeTeam')['HomePoints'].apply(lambda s: s.shift(1).rolling(5,min_periods=1).mean())
    df['away_form5_pts'] = df.groupby('AwayTeam')['AwayPoints'].apply(lambda s: s.shift(1).rolling(5,min_periods=1).mean())
    df['form5_pts_diff'] = df['home_form5_pts'] - df['away_form5_pts']
    # H2H (letzte 3, Punkte aus Sicht des aktuellen Heimteams)
    pid = np.where(df['HomeTeam'] < df['AwayTeam'],
                   df['HomeTeam']+'___'+df['AwayTeam'],
                   df['AwayTeam']+'___'+df['HomeTeam'])
    df['pair_id'] = pid
    df = df.reset_index(drop=True)
    h2h = []
    for i, r in df.iterrows():
        g = df[(df['pair_id']==r['pair_id']) & (df.index < i)].tail(3)
        pts = []
        for _, rr in g.iterrows():
            pts.append({'H':3,'D':1,'A':0}[rr['FTR']] if rr['HomeTeam']==r['HomeTeam'] else {'H':0,'D':1,'A':3}[rr['FTR']])
        h2h.append(np.mean(pts) if pts else np.nan)
    df['h2h_home_pts_last3'] = h2h
    df['home_flag'] = 1
    return df

def build_Xy(df):
    feats = ['b365_H','b365_D','b365_A','form5_pts_diff','h2h_home_pts_last3','home_flag']
    X = df[feats].fillna(0)
    y = df['FTR'].map({'H':0,'D':1,'A':2})
    return X, y
