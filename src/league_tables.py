import pandas as pd
import numpy as np

# === Helper: Ergebnisse in Heim- und Auswärts-Sicht auf Team-Zeilen bringen ===
def long_results(df):
    # Heim
    home = pd.DataFrame({
        "Season": df["Season"],
        "Team": df["HomeTeam"],
        "GF": df["FTHG"], "GA": df["FTAG"],
        "W": (df["FTR"]=="H").astype(int),
        "D": (df["FTR"]=="D").astype(int),
        "L": (df["FTR"]=="A").astype(int),
        "Pts": df["FTR"].map({"H":3,"D":1,"A":0}).astype(int),
        "MP": 1
    })
    # Auswärts
    away = pd.DataFrame({
        "Season": df["Season"],
        "Team": df["AwayTeam"],
        "GF": df["FTAG"], "GA": df["FTHG"],
        "W": (df["FTR"]=="A").astype(int),
        "D": (df["FTR"]=="D").astype(int),
        "L": (df["FTR"]=="H").astype(int),
        "Pts": df["FTR"].map({"A":3,"D":1,"H":0}).astype(int),
        "MP": 1
    })
    return pd.concat([home, away], ignore_index=True)

# === Saison-Tabelle (eine Saison) ===
def season_table(df, season: str, sort_keys=("Pts","GD","GF")):
    long = long_results(df[df["Season"]==season])
    tab = (long.groupby("Team", as_index=False)
                .sum(numeric_only=True))
    tab["GD"] = tab["GF"] - tab["GA"]
    tab["PPG"] = (tab["Pts"] / tab["MP"]).round(3)
    tab = tab.sort_values(list(sort_keys), ascending=[False, False, False]).reset_index(drop=True)
    tab.index = tab.index + 1  # Platzierung
    return tab[["MP","W","D","L","GF","GA","GD","Pts","PPG"]].rename_axis("Pos").reset_index().assign(Team=tab["Team"])

# === Tabellen für alle Saisons (stacked) ===
def tables_all_seasons(df):
    long = long_results(df)
    g = (long.groupby(["Season","Team"], as_index=False)
               .sum(numeric_only=True))
    g["GD"] = g["GF"] - g["GA"]
    g["PPG"] = (g["Pts"] / g["MP"]).round(3)
    # innerhalb jeder Saison sortieren
    g = g.sort_values(["Season","Pts","GD","GF"], ascending=[True,False,False,False])
    # Platz je Saison
    g["Pos"] = g.groupby("Season").cumcount() + 1
    cols = ["Season","Pos","Team","MP","W","D","L","GF","GA","GD","Pts","PPG"]
    return g[cols]

# === All‑Time‑Tabelle (über alle Saisons) ===
def all_time_table(df):
    long = long_results(df)
    tab = (long.groupby("Team", as_index=False)
                .sum(numeric_only=True))
    tab["GD"] = tab["GF"] - tab["GA"]
    tab["PPG"] = (tab["Pts"] / tab["MP"]).round(3)
    tab = tab.sort_values(["Pts","GD","GF"], ascending=[False,False,False]).reset_index(drop=True)
    tab.index = tab.index + 1
    return tab.rename_axis("Pos").reset_index()
