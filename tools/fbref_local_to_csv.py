# tools/fbref_local_to_csv.py
import re
import sys
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup, Comment

def _read_candidates_from_html(html: str) -> list[pd.DataFrame]:
    """Liest alle Tabellen-Kandidaten: direkt im DOM + in HTML-Kommentaren."""
    soup = BeautifulSoup(html, "lxml")
    dfs: list[pd.DataFrame] = []

    # 1) direkte Tabellen
    for t in soup.find_all("table"):
        try:
            dfs.append(pd.read_html(str(t))[0])
        except Exception:
            pass

    # 2) Tabellen in Kommentar-Blöcken
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        try:
            sub = BeautifulSoup(c, "lxml")
            for t in sub.find_all("table"):
                try:
                    dfs.append(pd.read_html(str(t))[0])
                except Exception:
                    pass
        except Exception:
            pass
    return dfs

def _looks_like_sched(df: pd.DataFrame) -> bool:
    cols = [str(c).strip() for c in df.columns]
    need = {"Date","Home","Score","Away"}
    return need.issubset(set(cols))

def _pick_best_sched(dfs: list[pd.DataFrame]) -> pd.DataFrame | None:
    candidates = [d for d in dfs if _looks_like_sched(d)]
    if not candidates:
        return None
    # nützlichste nehmen (meiste Zeilen)
    candidates.sort(key=lambda d: len(d), reverse=True)
    return candidates[0]

def _split_score(s: str):
    if not isinstance(s, str): return pd.NA, pd.NA
    s = s.strip()
    m = re.split(r"\s*[–-]\s*", s)  # en dash / hyphen robust
    if len(m) != 2: return pd.NA, pd.NA
    try: return int(m[0]), int(m[1])
    except: return pd.NA, pd.NA

def parse_local_html(path_html: Path) -> pd.DataFrame:
    html = path_html.read_text(encoding="utf-8", errors="ignore")
    dfs = _read_candidates_from_html(html)
    df = _pick_best_sched(dfs)
    if df is None:
        raise RuntimeError("Keine geeignete Tabelle gefunden (erwarte Header: Date, Home, Score, Away).")

    # Pflichtspalten geprüft; jetzt xG-Spalten heuristisch suchen
    cols = [str(c) for c in df.columns]
    xg_pos = [i for i,c in enumerate(cols) if str(c).strip().lower() == "xg"]
    if len(xg_pos) < 2:
        xg_pos = [i for i,c in enumerate(cols) if str(c).strip().lower().startswith("xg")]
    xg_home = df.columns[xg_pos[0]] if len(xg_pos) >= 1 else None
    xg_away = df.columns[xg_pos[1]] if len(xg_pos) >= 2 else None

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    out["HomeTeam"] = df["Home"].astype(str)
    out["AwayTeam"] = df["Away"].astype(str)

    ft = df["Score"].apply(_split_score)
    out["FTHG"] = [t[0] for t in ft]
    out["FTAG"] = [t[1] for t in ft]
    out["FTR"] = out.apply(
        lambda r: "H" if pd.notna(r["FTHG"]) and pd.notna(r["FTAG"]) and r["FTHG"] > r["FTAG"]
        else ("A" if pd.notna(r["FTHG"]) and pd.notna(r["FTAG"]) and r["FTHG"] < r["FTAG"]
              else ("D" if pd.notna(r["FTHG"]) and pd.notna(r["FTAG"]) and r["FTHG"] == r["FTAG"]
                    else pd.NA)), axis=1
    )

    if xg_home and xg_away:
        out["xg_home"] = pd.to_numeric(df[xg_home], errors="coerce")
        out["xg_away"] = pd.to_numeric(df[xg_away], errors="coerce")
    else:
        out["xg_home"] = pd.NA
        out["xg_away"] = pd.NA

    out = out.dropna(subset=["Date","HomeTeam","AwayTeam"]).sort_values("Date").reset_index(drop=True)
    return out

def infer_league_season_from_filename(p: Path) -> tuple[str,str]:
    name = p.stem
    # Season wie 2020-2021 aus Dateiname greifen
    m_season = re.search(r"(20\d{2}-20\d{2})", name)
    season = m_season.group(1) if m_season else "unknown"
    # Liga heuristisch
    code = "D2" if re.search(r"\b2\b.*Bundesliga|2\.\s*Bundesliga", name, flags=re.I) else "D1"
    return code, season

if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 3):
        print("Usage:")
        print("  python tools/fbref_local_to_csv.py <input_html> [output_csv]")
        sys.exit(2)

    inp = Path(sys.argv[1])
    if not inp.exists():
        print(f"✗ Datei nicht gefunden: {inp}")
        sys.exit(1)

    df = parse_local_html(inp)

    if len(sys.argv) == 3:
        out = Path(sys.argv[2])
        code, season = infer_league_season_from_filename(inp)
    else:
        code, season = infer_league_season_from_filename(inp)
        out = Path("data/performance") / f"{code}_{season}.csv"

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n = len(df)
    nxg = int((df["xg_home"].notna() & df["xg_away"].notna()).sum())
    print(f"✓ gespeichert: {out} (Spiele: {n}, mit xG: {nxg})")
