# Dota 2 Trajectory Analyzer

Analyse non supervisée de trajectoires de joueurs dans DotA 2 : compression MDL → clustering TRACLUS + AP → fouille de motifs PrefixSpan.

> Projet L3 Informatique — Université de Caen Normandie (2024-2025)  
> Elias Okat · Uzay Turkmenel · Paul Aubert

---

## Installation

**Prérequis** : Python 3.10+

```bash
# Cloner et entrer dans le projet
git clone <https://redmine-etu.unicaen.fr/git/sinfl6c1-analyse-de-trajectoires-dans-dota-2-aubert-turkmenel-okat>
cd sinfl6c1-analyse-de-trajectoires-dans-dota-2-aubert-turkmenel-okat

# Installer les dépendances (crée un venv automatiquement)
# Windows :
.\install.bat
# Linux / macOS :
./install.sh
```

Ou manuellement :
```bash
python -m venv .venv
# Windows : .venv\Scripts\activate  |  Linux : source .venv/bin/activate
pip install -r requirements.txt
```

### PC de la fac (Ubuntu sans pip)

Les dépendances sont embarquées dans `vendor/`. Utiliser le lanceur `fac.sh` :

```bash
chmod +x fac.sh
./fac.sh run.py compress --w_error 12
./fac.sh gui.py
./fac.sh benchmark/run_all.py --exp 0
```

> Pour regénérer `vendor/` : `python bundle.py` (sur une machine avec pip).

### Données

Placer les fichiers CSV dans `data-dota/` (convention : `coord_<match_id>.csv`).

---

## Utilisation

> Activer le venv avant exécution : `.venv\Scripts\activate` (Windows) ou `source .venv/bin/activate` (Linux).

### CLI

```bash
# Compression MDL
python run.py compress --w_error 12

# Visualisation original vs compressé
python run.py visualize --w_error 12 --match_id 3841893562

# Overlay sur la carte
python run.py overlay --w_error 12 --match_id 3841893562 --interactive

# Clustering
python run.py cluster --w_error 12 --max_files 10

# Fouille de motifs PrefixSpan
python run.py mine --cluster_dir output/clusters --min_support 15
```

### GUI

```bash
python gui.py
```

### Benchmarks

```bash
python benchmark/run_all.py            # Toutes les expériences
python benchmark/run_all.py --exp 0    # Sensibilité w_error uniquement
python benchmark/run_all.py --plots    # Générer les figures
```

### Tests

```bash
pytest tests/ -v
```

---

## Équipe

| Membre | Contributions |
|--------|--------------|
| **Elias Okat** | Parsing CSV, compression MDL, K-Médoïdes, benchmarks |
| **Uzay Turkmenel** | Calculs géométriques TRACLUS, clustering |
| **Paul Aubert** | Recodage RLE, PrefixSpan, graphe de Markov |
