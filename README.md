# Stress, Sleep & Music — SSM Seminar Analysis

Statistical analysis pipeline for a study examining the relationships between **war-related trauma exposure**, **psychological distress** (PTSD, anxiety, depression, GHQ), **sleep quality** (PSQI, ISI), and **music engagement habits** in an Israeli civilian sample (N = 110) during and after the October 7th war.

---

## Research Questions

- Do continuous stress and sleep scores correlate with objective trauma exposure?
- Which music habits (ordinal/Likert) correlate with stress and sleep outcomes?
- After regressing each outcome on exposure level or distress, who performs better or worse than predicted?
- Which specific music behaviours explain the resilience/vulnerability gap across different baseline models?

---

## Dataset

`data_final.xlsx` — sheet `"Completed"`, N = 110 participants.

`data_final.xlsx` is included in the repository and should be kept in the same folder as `SSM_analysis.py`.

---

## Analysis Pipeline

### Part A — Bivariate Correlation Matrices

| Variables | Method |
|---|---|
| Continuous × Continuous (exposure, PCL-5, DASS, GHQ, PSQI, ISI) | **Pearson r** (primary) + **Spearman ρ** (robustness check) |
| Ordinal × Continuous (all 16 music variables × outcomes) | **Spearman ρ only** — Pearson is never computed for ordinal scales |
| Stress × Sleep heatmap (CGI, DASS, GHQ, PCL-5 × PSQI, ISI) | **Pearson r** |

### Part B — OLS Regression Models (via `statsmodels`)

All continuous variables are Z-scored before regression. Ordinal music variables are never standardised.

| Family | Model |
|---|---|
| A) Exposure-Based | `exposure_index → pcl_score` |
| A) Exposure-Based | `exposure_index → isi_total` |
| B) Distress-Based | `ghq_score → isi_total` |
| B) Distress-Based | `pcl_score → psqi_total` |

### Part C — Music Associations with Residuals

For each of the 4 models, residuals (y − ŷ) are computed and labelled:

- **Residual < 0** → *Better than expected (Resilient)*
- **Residual ≥ 0** → *Worse than expected (Vulnerable)*

Spearman correlations are then computed between the residuals and all 16 music variables to identify which habits explain who copes better or worse than predicted.

---

## Outputs

All outputs are saved to a `results/` folder that is created automatically on first run.

| File | Contents |
|---|---|
| `residual_analysis_report.txt` | Full console log of every section |
| `correlation_matrices.txt` | All Part A correlation tables (text format) |
| `heatmap_pearson_stress_sleep.png` | Pearson heatmap — stress vars × sleep outcomes |
| `heatmap_spearman_continuous.png` | Spearman heatmap — continuous × continuous |
| `heatmap_spearman_ordinal.png` | Spearman heatmap — ordinal music × continuous |
| `scatter_resid_<X>_<Y>.png` | Regression scatter coloured by resilience group |
| `bar_resid_<X>_<Y>.png` | Music factors vs residuals bar chart (per model) |

---

## Requirements

Python 3.9+ is recommended.

Install all dependencies with:

```bash
pip install pandas openpyxl numpy scipy matplotlib statsmodels
```

---

## How to Run

1. Clone the repository (`data_final.xlsx` is included).
2. Install dependencies (see above).
3. Run the script:

```bash
python SSM_analysis.py
```

The `results/` folder will be created automatically and all outputs will be saved there. The full analysis log is also printed to the console in real time.

---

## Variable Notes

### Measurement scale rules

- **Continuous** (Pearson allowed): `exposure_index`, `pcl_score`, `dass_*`, `ghq_score`, `psqi_total`, `isi_total`
- **Ordinal / Likert** (Spearman only): all 16 music variables — `listen_week`, `listen_time`, `listen_war`, `listen_war_time`, `music_help_sleep`, `music_help_stress`, and the 10 music-engagement items

### Recoding applied at load time

| Variable | Transformation |
|---|---|
| `evac` | 1 → 0 (No), 2 → 1 (Yes) |
| Music engagement (10 items) | `6 - x` after replacing 6 (prefer not to answer) with NaN — so 5 = much more, 1 = much less |
| `listen_time`, `listen_war_time` | 9 → NaN (prefer not to answer) |
| `listen_week`, `listen_war` | 8 → NaN (prefer not to answer) |

### Missing value handling

**Listwise deletion** is used throughout — no imputation is applied. Each analysis drops only the rows missing on the specific variables involved in that calculation. Only `psqi_total` has missing values (n = 6), so models involving `psqi_total` run on n = 104; all others use the full n = 110.

