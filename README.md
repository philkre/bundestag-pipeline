# Bundestag Voting Analysis Pipeline

Analysis of German parliamentary voting behaviour from 2005 to 2029 across six Bundestag periods. Measures pairwise voting similarity between MPs using Cohen's κ, then applies network analysis, dimensionality reduction, and statistical physics methods to visualise ideological positioning, polarisation, and coalition cohesion.

## Project structure

```
bundestag-pipeline/
├── pipeline/        data acquisition and graph construction
├── analysis/        visualisation and statistical analysis scripts
├── config/          party colours and coalition definitions
├── renderer/        JS network renderer (Node.js)
└── output/
    ├── bundestag_YYYY_YYYY/   per-period nodes.csv, edges.csv, raw.json
    └── img/                   rendered PNGs
```

## Data flow

```
pipeline/scrape.py                →  output/{period}/raw.json
                                      polls.jsonl, votes.jsonl
pipeline/ingest.py                →  output/{period}/nodes.csv
                                      output/{period}/edges.csv
pipeline/compute_allpairs_kappa.py →  output/{period}/edges_allpairs.csv

analysis/*.py                     →  output/img/*.png
```

Analysis scripts read from `output/` independently — they can be re-run without repeating the scrape or ingest steps.

## Pipeline scripts

| Script | What it does |
|---|---|
| `pipeline/scrape.py` | Downloads roll-call votes and polls from the abgeordnetenwatch API v2 |
| `pipeline/ingest.py` | Builds a weighted MP-pair network (edge weight = Cohen's κ); writes CSVs and GEXF |
| `pipeline/compute_allpairs_kappa.py` | Computes the full all-pairs κ matrix; writes `edges_allpairs.csv` |
| `pipeline/network.py` | Graph metrics (density, components, cross-party edge counts) |
| `pipeline/plot.py` | Network graph rendering (force-layout or party layout) |
| `pipeline/cli.py` | CLI argument parsing for `main.py` |
| `main.py` | Entry point: scrape → ingest → plot in one command |

## Analysis scripts

| Script | Output |
|---|---|
| `analysis/mp_mds_strip.py` | 1D MDS strip — one dot per MP, positioned by voting similarity |
| `analysis/mds_drift.py` | Party-level ideological drift across all six periods |
| `analysis/mp_coalition_alignment.py` | Per-MP coalition alignment over time |
| `analysis/mp_cross_aisle.py` | Party × party mean κ heatmaps per period |
| `analysis/mp_cross_aisle_simple.py` | Coalition vs. opposition κ over time (line chart) |
| `analysis/polarisation.py` | Government vs. opposition voting alignment time series |
| `analysis/polarisation_by_party.py` | Polarisation decomposed by opposition party |
| `analysis/coalition_cohesion.py` | Within-coalition cohesion and pair-wise κ time series |
| `analysis/kappa_heatmap.py` | Party × party mean κ heatmap grid |
| `analysis/mp_influence.py` | Per-MP mutual information I(σᵢ; γ) — how predictive each MP's vote is of the chamber majority; chancellor and faction leaders highlighted |
| `analysis/mp_ising.py` | Mean-field Ising model fitted to voting data; effective temperature per period |
| `analysis/mp_ising_exact.py` | Exact inverse Ising model at party level; energy landscape and critical temperature |

All analysis scripts accept an optional `light` argument for a light-mode version:
```bash
python analysis/mp_influence.py light
```

## Usage

```bash
# 1. Scrape raw data for a period
python pipeline/scrape.py --outdir output/bundestag_2021_2025

# 2. Build the voting similarity graph
python main.py --votes output/bundestag_2021_2025/votes.jsonl \
               --polls output/bundestag_2021_2025/polls.jsonl \
               --out-dir output/bundestag_2021_2025

# 3. Compute all-pairs kappa (slow; skip if edges_allpairs.csv already exists)
python pipeline/compute_allpairs_kappa.py bundestag_2021_2025

# 4. Run any analysis script
python analysis/mp_influence.py
python analysis/mp_cross_aisle.py light
```

Periods covered: **2005–09 · 2009–13 · 2013–17 · 2017–21 · 2021–25 · 2025–29**

## Configuration

| File | Purpose |
|---|---|
| `config/party_colours.json` | Hex colour per party (single source of truth) |
| `config/coalitions.json` | Coalition composition per period |

## Requirements

```bash
pip install -r requirements.txt   # Python 3.10+
```

Node.js is required only for `renderer/render.js` (interactive network SVG).

## Methodology

**Voting similarity** between two MPs is Cohen's κ on their shared votes, controlling for chance agreement: κ = 1 means identical voting record; κ = 0 means no better than chance; κ < 0 means systematic disagreement.

**Ideological positioning** uses 1D classical MDS (eigendecomposition of the double-centred κ-distance matrix). Within each period the axis is oriented so coalition parties sit on the positive side, giving a consistent *opposition ← → coalition* interpretation across all charts.

**MP influence** is measured as the mutual information I(σᵢ; γ) between an MP's individual vote σᵢ and the chamber majority direction γ. Higher values mean the MP's vote is a stronger predictor of the overall outcome.

**Ising model** treats each party as a spin (±1). The exact inverse Ising procedure finds coupling strengths J_ij and biases h_i such that the maximum-entropy model reproduces the observed vote correlations. The ratio T_eff / T_c indicates how close the system is to a phase transition — values below 1 indicate an ordered (polarised) regime.
