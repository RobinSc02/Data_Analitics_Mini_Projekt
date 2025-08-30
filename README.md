# Setup & Quickstart

> **Ziel:** Repo lokal mit **Anaconda + VS Code** lauffähig machen.

---

## Voraussetzungen

* **Git**, **Anaconda Navigator**, **VS Code**
* VS-Code-Extensions: *Python* & *Jupyter*
* Zugriff auf GitHub-Repo

---

## Quickstart 

1. `git clone https://github.com... && cd Data_Analitics_Mini_Projekt`
2. `conda env create -f environment.yml` *(oder Env manuell einrichten, siehe unten)*
3. `conda activate da-mini && python -m ipykernel install --user --name da-mini --display-name "Python (da-mini)"`
4. VS Code öffnen → **Python-Interpreter/Kernal: *Python (da-mini)*** wählen
5. **Starten:** Notebook in `notebooks/` ausführen **oder** `python main.py` im Repo-Root

---

## Repo-Struktur (Kurzüberblick)

```
data/           # Datenbasis 
notebooks/      # Analysen/EDA
results/        # Ausgaben/Plots/Modelle
src/            # Code/Pipeline
main.py         # Einstiegspunkt für Pipeline
```

---

## Umgebung einrichten (falls **ohne** `environment.yml`)

```bash
# Env anlegen
conda create -n da-mini python=3.11 -y
conda activate da-mini

# Pakete
conda install -y pandas numpy scikit-learn matplotlib seaborn scipy statsmodels jupyterlab
pip install -U pip ipykernel python-dotenv requests tqdm

# Kernel registrieren
python -m ipykernel install --user --name da-mini --display-name "Python (da-mini)"
```

**Optional: `environment.yml` Inhalt**

```yaml
name: da-mini
channels: [conda-forge, defaults]
dependencies:
  - python=3.11
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - scipy
  - statsmodels
  - jupyterlab
  - pip
  - pip:
      - ipykernel
      - python-dotenv
      - requests
      - tqdm
```

Installation:

```bash
conda env create -f environment.yml
conda activate da-mini
python -m ipykernel install --user --name da-mini --display-name "Python (da-mini)"
```

---

## VS-Code Einstellungen (empfohlen)

* **Interpreter/Kernal:** `Python (da-mini)`
* **Working Directory:** VS-Code Setting *Jupyter: Notebook File Root* → `workspaceFolder`
  *(oder in der ersten Notebook-Zelle:)*

  ```python
  import os, pathlib
  os.chdir(pathlib.Path().resolve())  # vom Repo-Root starten
  ```

---

## (Optional) API-Keys

`.env` im Repo-Root anlegen:

```
FOOTBALL_DATA_API_KEY=DEIN_KEY
```

Im Code laden:

```python
from dotenv import load_dotenv; load_dotenv()
import os; api_key = os.getenv("FOOTBALL_DATA_API_KEY")
```

---

## Git-Workflow (kurz)

```bash
git checkout main
git pull origin main
git checkout -b feature/mein-thema
# Änderungen …
git add .
git commit -m "Kurze, präzise Nachricht"
git push -u origin feature/mein-thema
# PR auf GitHub erstellen
```

---

## Häufige Fehler & schnelle Fixes

| Problem                                    | Ursache                                      | Lösung                                                                         |
| ------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------------ |
| VS Code nutzt `base` statt `da-mini`       | Falscher Interpreter/Kernal                  | *Python: Select Interpreter* → **Python (da-mini)** wählen                     |
| `ModuleNotFoundError: xyz`                 | Paket fehlt in aktiver Env                   | `conda install xyz` **oder** `pip install xyz` (nach `conda activate da-mini`) |
| Kernel startet nicht / „dies unexpectedly“ | Konflikt Paketversionen                      | `pip install -U ipykernel jupyter_client tornado`                              |
| `FileNotFoundError` (Daten)                | Falsches Arbeitsverzeichnis/fehlender Ordner | Vom **Repo-Root** starten; Ordner (`data/`, `results/`) anlegen                |
| `UnicodeDecodeError` bei CSV               | Encoding abweichend                          | `pd.read_csv(path, encoding="utf-8", engine="python")` *(ggf. `latin-1`)*      |
| Inkonstante Längen vor `fit()`             | `X`/`y` nach Filtern nicht fluchtend         | Gemeinsames `dropna` oder `X, y = X.align(y, join="inner", axis=0)`            |
| „Updates were rejected“ beim Push          | Main ist neuer                               | `git pull --rebase origin main` → erneut `git push`                            |
| Große CSV/ZIP blockieren Push              | LFS fehlt                                    | `git lfs install && git lfs track "*.csv" "*.zip"` → commit `.gitattributes`   |


cd C:\Users\admin\IdeaProjects\Data_Analitics_Mini_Projekt
python main.py


---

## Start

* **Notebooks:** in `notebooks/` öffnen → Kernal `Python (da-mini)` → ausführen
* **Pipeline:** im Repo-Root `python main.py`

