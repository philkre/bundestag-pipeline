# Bundestag Voting Analysis Pipeline

Analysis of German parliamentary voting behaviour from 2005 to 2029. Measures pairwise voting similarity between MPs using Cohen's κ, then visualises ideological positioning, polarisation trends, and coalition cohesion across all Bundestag periods.

## Pipeline overview

```
scrape.py          → votes.jsonl, polls.jsonl        (raw API data)
ingest.py          → nodes.csv, edges.csv, graph.gexf (per-period graphs)
compute_allpairs_kappa.py → edges_allpairs.csv        (all MP pairs)
```

Analysis and visualisation scripts consume those outputs independently.

## Scripts

| Script | What it produces |
|---|---|
| `scrape.py` | Downloads roll-call votes and polls from the abgeordnetenwatch API v2 |
| `ingest.py` / `main.py` | Builds weighted MP-pair network (edges = Cohen's κ); outputs CSVs and GEXF |
| `compute_allpairs_kappa.py` | Full all-pairs κ matrix; writes `edges_allpairs.csv` |
| `mp_mds_strip.py` | 1D MDS strip chart — one dot per MP, positioned by voting similarity, coloured by party |
| `mds_drift.py` | Party-level ideological drift across periods on a normalised left–right axis |
| `polarisation.py` | Time series of government vs. opposition voting alignment |
| `polarisation_by_party.py` | Decomposes polarisation by individual opposition party |
| `coalition_cohesion.py` | Coalition internal cohesion and pair-wise κ time series |
| `kappa_heatmap.py` | Party × party mean κ heatmap across all six periods |
| `network.py` / `plot.py` | Network graph rendering (force-atlas2 or party layout) |

## Data

One output directory per parliamentary period under `output/`:

```
output/
  bundestag_2005_2009/
    nodes.csv
    edges.csv
    edges_allpairs.csv
  bundestag_2009_2013/
  ...
  img/                   ← all rendered PNGs
```

Periods covered: 2005–2009, 2009–2013, 2013–2017, 2017–2021, 2021–2025, 2025–2029.

## Usage

```bash
# Scrape raw data for a period
python scrape.py bundestag_2021_2025

# Build graph
python ingest.py bundestag_2021_2025

# Compute all-pairs kappa (slow; skip if edges_allpairs.csv exists)
python compute_allpairs_kappa.py bundestag_2021_2025

# Render MP strip chart (dark + light)
python mp_mds_strip.py bundestag_2021_2025
python mp_mds_strip.py bundestag_2021_2025 light
```

## Configuration

| File | Purpose |
|---|---|
| `renderer/party_colours.json` | Hex colour per party |
| `renderer/coalitions.json` / `coalitions.json` | Coalition composition per period |

## Requirements

```
pip install -r requirements.txt
```

Python 3.10+. Node.js required for `renderer/render.js`.

## Methodology

Voting similarity between two MPs is measured as Cohen's κ on their shared votes, controlling for chance agreement. A κ of 1 means identical voting record; 0 means no better than chance; negative values indicate systematic disagreement.

1D MDS (classical, eigendecomposition of the double-centred distance matrix) projects each MP onto a single ideological axis. Within each period the axis is oriented so coalition parties have positive (right) mean — giving a consistent **opposition ← → coalition** interpretation across all charts.
