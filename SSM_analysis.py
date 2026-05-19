"""
SSM Seminar — Stress, Sleep & Music Analysis
See README.md for full documentation, pipeline description, and variable notes.

Dependencies:
  pip install pandas openpyxl numpy scipy matplotlib statsmodels
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data_final.xlsx")
OUT_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(OUT_DIR, "residual_analysis_report.txt")
CORR_FILE   = os.path.join(OUT_DIR, "correlation_matrices.txt")


# ── Tee: mirrors every print() to both the terminal and the report file ───────
class _Tee:
    def __init__(self, filepath: str):
        self._terminal = sys.stdout
        self._log = open(filepath, "w", encoding="utf-8")

    def write(self, msg: str):
        self._terminal.write(msg)
        self._log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()
        sys.stdout = self._terminal


_tee = _Tee(REPORT_FILE)
sys.stdout = _tee

BAR = "═" * 70


def section(title: str) -> None:
    print(f"\n{BAR}\n  {title}\n{BAR}")


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_fig(name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# VARIABLE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# Continuous sum-score variables — Pearson r is valid for these
CONTINUOUS_VARS = [
    "exposure_index",
    "pcl_score",
    "dass_anx_score", "dass_dep_score", "dass_str_score", "dass_total_score",
    "ghq_score",
    "psqi_total", "isi_total",
]

# Variables eligible for Z-scoring before regression — strictly continuous
# sum scores only.  listen_time and listen_war_time are ordinal categories
# (1–8) and are deliberately excluded from standardisation.
Z_SCORE_VARS = [
    "exposure_index", "pcl_score",
    "dass_anx_score", "dass_dep_score", "dass_str_score", "dass_total_score",
    "ghq_score", "psqi_total", "isi_total",
]

# Stress and sleep subsets used in regression
STRESS_OUTCOME_VARS = ["pcl_score", "dass_total_score", "ghq_score"]
SLEEP_OUTCOME_VARS  = ["psqi_total", "isi_total"]
OUTCOME_VARS        = STRESS_OUTCOME_VARS + SLEEP_OUTCOME_VARS

# Ordinal / Likert music variables — Spearman ρ ONLY; Pearson is NEVER computed
# for any pair that includes one of these variables.
ORDINAL_MUSIC_VARS = [
    "listen_time", "listen_war_time",
    "listen_week", "listen_war",
    "music_play", "sing", "dance", "music_help_sleep",
]

# Full set of 16 music variables used in the residual analysis
ALL_MUSIC_VARS = [
    "listen_week", "listen_time",
    "listen_war",  "listen_war_time",
    "music_help_sleep", "music_help_stress",
    "music_gen", "music_con", "music_h", "music_s", "music_n",
    "live_shows", "rec_shows",
    "music_play", "dance", "sing",
]

# Likert items whose raw direction needs reversing (6=prefer not to answer)
MUSIC_ENGAGEMENT_VARS = [
    "music_gen", "music_con", "music_h", "music_s", "music_n",
    "live_shows", "rec_shows", "music_play", "dance", "sing",
]

# 35-item composite exposure index
EXPOSURE_INDEX_VARS = [
    "ptsd_1", "ptsd_2", "ptsd_3", "ptsd_4", "ptsd_5", "ptsd_10", "evac",
    "job_loss_a", "job_trauma_a", "wedding_planning_a", "house_deal_a",
    "reno_a", "stolen_a", "li_a", "econi_a",
    "fam_desease_a", "partner_desease_a", "friend_desease_a",
    "fam_death_a", "pdeath_a", "frdeath_a", "pet_death_a",
    "devorce_a", "fam_fight_a", "breakup_a", "fight_a",
    "friendshipi_a", "edu_a", "treat_a", "unplanned_preg_a",
    "abor_a", "phys_des_a", "harr_a", "sharr_a", "stressful_event_a",
]

CONTINUOUS_LABELS = {
    "exposure_index":    "Exposure Index",
    "pcl_score":         "PCL-5 (PTSD)",
    "dass_anx_score":    "DASS Anxiety",
    "dass_dep_score":    "DASS Depression",
    "dass_str_score":    "DASS Stress",
    "dass_total_score":  "DASS Total",
    "ghq_score":         "GHQ",
    "psqi_total":        "PSQI (Sleep Quality)",
    "isi_total":         "ISI (Insomnia)",
}

MUSIC_LABELS = {
    "listen_week":       "Weekly Listening (days)",
    "listen_time":       "Daily Listen Time",
    "listen_war":        "Listened During War",
    "listen_war_time":   "War Listening Time",
    "music_help_sleep":  "Music Helps Sleep",
    "music_help_stress": "Music Helps Stress",
    "music_gen":         "Music – General",
    "music_con":         "Music – Concentrated",
    "music_h":           "Music – Hedonic",
    "music_s":           "Music – Social",
    "music_n":           "Music – Background",
    "live_shows":        "Attends Live Shows",
    "rec_shows":         "Watches Recorded Shows",
    "music_play":        "Plays an Instrument",
    "dance":             "Dancing",
    "sing":              "Singing",
}

# Colour coding for residual groups in plots (binary: sign of residual)
GROUP_COLORS = {
    "Better than expected (Resilient)":  "steelblue",
    "Worse than expected (Vulnerable)":  "crimson",
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & RECODING
# ══════════════════════════════════════════════════════════════════════════════

section("1. DATA LOADING & RECODING")

df_raw = pd.read_excel(DATA_FILE, sheet_name="Completed", engine="openpyxl")
print(f"  Raw rows (incl. summary rows): {len(df_raw)}")

# The last 2 rows are aggregate summary rows added by the survey export — remove them.
df = df_raw.iloc[:-2].dropna(how="all").copy()
print(f"  Participant rows kept: {len(df)}  (expected 110)")

# Force all columns to numeric; free-text or error strings become NaN.
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# evac was coded 1=No / 2=Yes — recode to standard binary 0/1
if "evac" in df.columns:
    df["evac"] = df["evac"].map({1: 0, 2: 1})
    print("  [recode] evac: 1→0 (No), 2→1 (Yes)")

# Music engagement items: 6="prefer not to answer" → NaN,
# then reverse-score (6 - x) so that 5=Much more, 1=Much less engagement.
for col in MUSIC_ENGAGEMENT_VARS:
    if col in df.columns:
        df[col] = df[col].replace(6, np.nan)
        df[col] = 6 - df[col]
print(f"  [recode] Music engagement vars reversed (5=Much more, 1=Much less)")

# Survey "prefer not to answer" sentinels → NaN (not true zeros)
for col in ["listen_time", "listen_war_time"]:
    if col in df.columns:
        df[col] = df[col].replace(9, np.nan)
for col in ["listen_week", "listen_war"]:
    if col in df.columns:
        df[col] = df[col].replace(8, np.nan)
print("  [recode] Sentinel values (8/9 = prefer not to answer) → NaN")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COMPOSITE CONSTRUCTION & MEAN IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

section("2. COMPOSITE CONSTRUCTION & MISSING VALUE REPORT")

# Build exposure_index as an unweighted sum of all available items.
# min_count=1 ensures a participant needs at least one non-NaN item to get a score.
ei_avail = [c for c in EXPOSURE_INDEX_VARS if c in df.columns]
df["exposure_index"] = df[ei_avail].sum(axis=1, min_count=1)
ei = df["exposure_index"].dropna()
print(f"\n  exposure_index: {len(ei_avail)} items summed")
print(f"    mean={ei.mean():.2f}  SD={ei.std():.2f}  "
      f"min={ei.min():.0f}  max={ei.max():.0f}  n={len(ei)}")

# No mean imputation is applied.  All analyses use listwise deletion:
#   • Correlation functions call dropna() per variable pair.
#   • Each regression model drops rows missing on its X or Y variable.
#   • Residual–music Spearman correlations drop per pair as well.
# Only psqi_total has missing values (n=6), so the two models involving
# psqi_total run on n=104; all others run on the full n=110.

ALL_ANALYSIS_VARS = [v for v in CONTINUOUS_VARS + ALL_MUSIC_VARS if v in df.columns]
print(f"\n  Missing values (listwise deletion applied per analysis):")
miss = df[ALL_ANALYSIS_VARS].isnull().sum()
miss_nonzero = miss[miss > 0]
if miss_nonzero.empty:
    print("    None.")
else:
    print(miss_nonzero.to_string())
print(f"\n  Total participants loaded: {len(df)}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PART A: BIVARIATE CORRELATION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
#
# Rule 1 — Continuous × Continuous:
#   Both Pearson r and Spearman ρ are computed and printed side-by-side.
#   Pearson is the primary coefficient; Spearman serves as a robustness check
#   for cases where residuals deviate from normality.
#
# Rule 2 — Ordinal × Continuous:
#   Spearman ρ ONLY.  Pearson is NEVER computed for any pair that involves
#   an ordinal/Likert variable, as it assumes interval-scale data.
# ══════════════════════════════════════════════════════════════════════════════

section("3. PART A — BIVARIATE CORRELATION MATRICES")

# Open the output file now; all matrices will be appended to it.
corr_out = open(CORR_FILE, "w", encoding="utf-8")

def write_corr(text: str) -> None:
    """Print to console and write to correlation_matrices.txt simultaneously."""
    print(text)
    corr_out.write(text + "\n")


# ── Helper: compute pairwise r/ρ and p-value ─────────────────────────────────

def pearson_pair(df: pd.DataFrame, x: str, y: str):
    sub = df[[x, y]].dropna()
    if len(sub) < 5:
        return np.nan, np.nan, len(sub)
    r, p = stats.pearsonr(sub[x], sub[y])
    return round(r, 3), round(p, 4), len(sub)


def spearman_pair(df: pd.DataFrame, x: str, y: str):
    sub = df[[x, y]].dropna()
    if len(sub) < 5:
        return np.nan, np.nan, len(sub)
    r, p = stats.spearmanr(sub[x], sub[y])
    return round(r, 3), round(p, 4), len(sub)


# ── Helper: build and format a square/rectangular correlation matrix ──────────

def build_corr_matrix(df: pd.DataFrame, row_vars: list, col_vars: list,
                      method: str, label_map: dict) -> pd.DataFrame:
    """
    Returns a DataFrame where each cell is formatted as 'r***' or 'r  '.
    method must be 'pearson' or 'spearman'.
    """
    fn = pearson_pair if method == "pearson" else spearman_pair
    rows = {}
    for rv in row_vars:
        row_label = label_map.get(rv, rv)
        row_data  = {}
        for cv in col_vars:
            col_label = label_map.get(cv, cv)
            if rv == cv:
                row_data[col_label] = "  —  "
                continue
            r, p, _ = fn(df, rv, cv)
            if np.isnan(r):
                row_data[col_label] = "  n/a"
            else:
                row_data[col_label] = f"{r:+.3f}{stars(p):<3}"
        rows[row_label] = row_data
    return pd.DataFrame(rows).T


def print_matrix(mat: pd.DataFrame, title: str, note: str = "") -> None:
    width = max(len(title), 72)
    write_corr(f"\n{'─' * width}")
    write_corr(f"  {title}")
    if note:
        write_corr(f"  {note}")
    write_corr(f"{'─' * width}")
    write_corr(mat.to_string())
    write_corr("")


# ══════════════════════════════════════════════════════════════════════════════
# MATRIX A1 — Continuous × Continuous  (Pearson r  +  Spearman ρ)
# ══════════════════════════════════════════════════════════════════════════════

cont_avail = [v for v in CONTINUOUS_VARS if v in df.columns]
all_labels = {**CONTINUOUS_LABELS, **MUSIC_LABELS}

write_corr("=" * 72)
write_corr("  MATRIX A1 — CONTINUOUS × CONTINUOUS")
write_corr("  Variables: " + ", ".join(cont_avail))
write_corr("=" * 72)

# ── A1a: Pearson r ────────────────────────────────────────────────────────────
mat_pearson = build_corr_matrix(df, cont_avail, cont_avail,
                                method="pearson", label_map=CONTINUOUS_LABELS)
print_matrix(
    mat_pearson,
    title="A1a — Pearson r  (primary coefficient for continuous variables)",
    note="Significance: * p<.05  ** p<.01  *** p<.001",
)

# ── A1b: Spearman ρ (robustness check) ───────────────────────────────────────
mat_spearman = build_corr_matrix(df, cont_avail, cont_avail,
                                 method="spearman", label_map=CONTINUOUS_LABELS)
print_matrix(
    mat_spearman,
    title="A1b — Spearman ρ  (robustness check — use when residuals violate normality)",
    note="Significance: * p<.05  ** p<.01  *** p<.001",
)

# ── Side-by-side comparison table (Pearson r vs Spearman ρ per pair) ─────────
write_corr(f"\n{'─' * 72}")
write_corr("  A1c — Pearson r vs Spearman ρ comparison (upper triangle only)")
write_corr(f"  {'Pair':<40} {'Pearson r':>10}  {'Spearman ρ':>10}  {'Agreement'}")
write_corr(f"  {'─'*40} {'─'*10}  {'─'*10}  {'─'*9}")

for i, x in enumerate(cont_avail):
    for y in cont_avail[i+1:]:
        pr, pp, _  = pearson_pair(df, x, y)
        sr, sp, n_ = spearman_pair(df, x, y)
        agree = "✓" if (not np.isnan(pr) and not np.isnan(sr)
                        and abs(pr - sr) < 0.10) else "Δ≥0.10"
        x_lbl = CONTINUOUS_LABELS.get(x, x)
        y_lbl = CONTINUOUS_LABELS.get(y, y)
        pair_lbl = f"{x_lbl} × {y_lbl}"
        write_corr(f"  {pair_lbl:<40} "
                   f"{str(pr)+stars(pp):>10}  "
                   f"{str(sr)+stars(sp):>10}  {agree}")

write_corr("")

# ══════════════════════════════════════════════════════════════════════════════
# MATRIX A2 — Ordinal × Continuous  (Spearman ρ ONLY)
# ══════════════════════════════════════════════════════════════════════════════
#
# Pearson r is intentionally and permanently excluded from this block.
# Ordinal Likert scales do not have equal intervals between response options,
# so Pearson r would carry a false assumption of interval-level measurement.
# ══════════════════════════════════════════════════════════════════════════════

ord_avail  = [v for v in ORDINAL_MUSIC_VARS if v in df.columns]

write_corr("=" * 72)
write_corr("  MATRIX A2 — ORDINAL × CONTINUOUS  (Spearman ρ ONLY)")
write_corr("  Pearson r is NOT computed for these pairs — ordinal scale.")
write_corr("  Ordinal vars: " + ", ".join(ord_avail))
write_corr("  Continuous vars: " + ", ".join(cont_avail))
write_corr("=" * 72)

mat_ord = build_corr_matrix(df, ord_avail, cont_avail,
                            method="spearman", label_map=all_labels)
print_matrix(
    mat_ord,
    title="A2 — Spearman ρ: Ordinal Music Variables × Continuous Outcomes",
    note="Significance: * p<.05  ** p<.01  *** p<.001  |  Pearson NOT computed",
)

# ── Detailed table sorted by |ρ| for the main outcomes ───────────────────────
write_corr(f"\n{'─' * 72}")
write_corr("  A2 Detailed — Spearman ρ ranked by absolute magnitude")
write_corr(f"  {'Ordinal Variable':<30} {'Continuous Variable':<24} "
           f"{'ρ':>7}  {'p':>8}  {'n':>4}  sig")
write_corr(f"  {'─'*30} {'─'*24} {'─'*7}  {'─'*8}  {'─'*4}  {'─'*3}")

detail_rows = []
for ov in ord_avail:
    for cv in cont_avail:
        r, p, n_ = spearman_pair(df, ov, cv)
        if not np.isnan(r):
            detail_rows.append({
                "ov": MUSIC_LABELS.get(ov, ov),
                "cv": CONTINUOUS_LABELS.get(cv, cv),
                "r": r, "p": p, "n": n_,
            })

detail_rows.sort(key=lambda x: abs(x["r"]), reverse=True)
for row in detail_rows:
    write_corr(f"  {row['ov']:<30} {row['cv']:<24} "
               f"{row['r']:>+7.3f}  {row['p']:>8.4f}  {row['n']:>4}  {stars(row['p'])}")

write_corr("")
corr_out.close()
print(f"\n  [saved] {CORR_FILE}")

# ── Heatmap helper ────────────────────────────────────────────────────────────

def _corr_heatmap(r_mat: np.ndarray, p_mat: np.ndarray,
                  row_labels: list, col_labels: list,
                  title: str, filename: str) -> None:
    """
    Draw a colour-coded correlation heatmap with significance annotations.
    Cells where p ≥ .05 are shown at reduced opacity to de-emphasise them.
    """
    n_rows, n_cols = r_mat.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.1), max(4, n_rows * 0.7)))

    im = ax.imshow(r_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="r / ρ")

    for i in range(n_rows):
        for j in range(n_cols):
            r_val = r_mat[i, j]
            p_val = p_mat[i, j]
            if np.isnan(r_val):
                continue
            sig  = stars(p_val)
            text = f"{r_val:.2f}{sig}" if not np.isnan(p_val) else f"{r_val:.2f}"
            alpha = 1.0 if (np.isnan(p_val) or p_val < 0.05) else 0.45
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7.5, color="black", alpha=alpha)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title + "\n* p<.05  ** p<.01  *** p<.001  (faded = n.s.)",
                 fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    save_fig(filename)


# ── Heatmap A1: Pearson r — Stress × Sleep ────────────────────────────────────
# Rows: stress / distress variables (PCL-5, DASS subscales + total, GHQ, CGI)
# Cols: sleep outcome variables (PSQI, ISI)
HEATMAP_STRESS_VARS = [
    "pcl_score",
    "dass_anx_score", "dass_dep_score", "dass_str_score", "dass_total_score",
    "ghq_score",
    "current_feeling",   # CGI
]
HEATMAP_SLEEP_VARS  = ["psqi_total", "isi_total"]

HEATMAP_LABELS = {
    **CONTINUOUS_LABELS,
    "current_feeling": "CGI",
}

# Force current_feeling to numeric if present
if "current_feeling" in df.columns:
    df["current_feeling"] = pd.to_numeric(df["current_feeling"], errors="coerce")

stress_hm = [v for v in HEATMAP_STRESS_VARS if v in df.columns]
sleep_hm  = [v for v in HEATMAP_SLEEP_VARS  if v in df.columns]

n_s, n_sl = len(stress_hm), len(sleep_hm)
r_arr_p   = np.full((n_s, n_sl), np.nan)
p_arr_p   = np.full((n_s, n_sl), np.nan)
for i, x in enumerate(stress_hm):
    for j, y in enumerate(sleep_hm):
        r, p, _ = pearson_pair(df, x, y)
        r_arr_p[i, j], p_arr_p[i, j] = r, p

s_row_labels = [HEATMAP_LABELS.get(v, v) for v in stress_hm]
s_col_labels = [HEATMAP_LABELS.get(v, v) for v in sleep_hm]
_corr_heatmap(r_arr_p, p_arr_p, s_row_labels, s_col_labels,
              "Pearson r — Stress Variables × Sleep Outcomes",
              "heatmap_pearson_stress_sleep.png")

# ── Heatmap A1b: Spearman ρ — Continuous × Continuous ────────────────────────
n_c     = len(cont_avail)
c_labels = [CONTINUOUS_LABELS.get(v, v) for v in cont_avail]
r_arr_s  = np.full((n_c, n_c), np.nan)
p_arr_s  = np.full((n_c, n_c), np.nan)
for i, x in enumerate(cont_avail):
    for j, y in enumerate(cont_avail):
        if i == j:
            r_arr_s[i, j] = 1.0
            p_arr_s[i, j] = 0.0
        else:
            r, p, _ = spearman_pair(df, x, y)
            r_arr_s[i, j], p_arr_s[i, j] = r, p

_corr_heatmap(r_arr_s, p_arr_s, c_labels, c_labels,
              "Spearman ρ — Continuous × Continuous (robustness check)",
              "heatmap_spearman_continuous.png")

# ── Heatmap A2: Spearman ρ — Ordinal Music × Continuous ──────────────────────
n_o     = len(ord_avail)
r_arr_o = np.full((n_o, n_c), np.nan)
p_arr_o = np.full((n_o, n_c), np.nan)
for i, ov in enumerate(ord_avail):
    for j, cv in enumerate(cont_avail):
        r, p, _ = spearman_pair(df, ov, cv)
        r_arr_o[i, j], p_arr_o[i, j] = r, p

o_labels = [MUSIC_LABELS.get(v, v) for v in ord_avail]
_corr_heatmap(r_arr_o, p_arr_o, o_labels, c_labels,
              "Spearman ρ — Ordinal Music × Continuous  (Pearson NOT computed)",
              "heatmap_spearman_ordinal.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PART B: OLS REGRESSIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# Four models in two families:
#   A) Exposure-Based  — objective baseline (how much trauma predicts outcome)
#   B) Distress-Based  — clinical baseline (how much distress predicts sleep)
#
# All continuous predictors and outcomes are Z-scored before regression so
# that β coefficients are directly comparable across models.
# Ordinal music variables (listen_time etc.) are NOT in Z_SCORE_VARS and
# are never standardised here.
# ══════════════════════════════════════════════════════════════════════════════

section("4. PART B — OLS REGRESSIONS")

# Z-score only the strictly continuous sum scores (Z_SCORE_VARS).
# listen_time and listen_war_time are ordinal (1–8) and excluded.
df_z = df.copy()
for col in Z_SCORE_VARS:
    if col in df_z.columns:
        mu, sd = df_z[col].mean(), df_z[col].std()
        df_z[col] = (df_z[col] - mu) / sd

print("\n  Z-score applied to strictly continuous variables only:")
for col in Z_SCORE_VARS:
    if col in df_z.columns:
        print(f"    {CONTINUOUS_LABELS.get(col, col):<28} → mean≈0  SD≈1")
print("  listen_time / listen_war_time: ordinal — NOT z-scored.\n")

REGRESSION_MODELS = [
    # A) Exposure-Based (objective baseline)
    {"x": "exposure_index", "y": "pcl_score",  "group": "A) Exposure-Based"},
    {"x": "exposure_index", "y": "isi_total",  "group": "A) Exposure-Based"},
    # B) Distress-Based (clinical baseline)
    {"x": "ghq_score",      "y": "isi_total",  "group": "B) Distress-Based"},
    {"x": "pcl_score",      "y": "psqi_total", "group": "B) Distress-Based"},
]

# residuals_store[(X, Y)] is populated here and consumed in Sections 5 & 6.
residuals_store = {}

for model in REGRESSION_MODELS:
    X, Y, grp = model["x"], model["y"], model["group"]
    x_lbl = CONTINUOUS_LABELS.get(X, X)
    y_lbl = CONTINUOUS_LABELS.get(Y, Y)

    print(f"\n  {'─'*68}")
    print(f"  {grp}:  {x_lbl}  →  {y_lbl}")
    print(f"  {'─'*68}")

    if X not in df_z.columns or Y not in df_z.columns:
        print(f"  Variable not found — skipping.")
        continue

    sub = df_z[[X, Y]].dropna().copy()
    if len(sub) < 10:
        print(f"  Insufficient data (n={len(sub)}) — skipping.")
        continue

    # statsmodels OLS — gives full regression table (β, SE, t, p, R², F)
    X_sm   = sm.add_constant(sub[X].values)
    result = sm.OLS(sub[Y].values, X_sm).fit()
    sub["residual"] = result.resid

    print(result.summary(
        xname=["const", x_lbl],
        yname=y_lbl,
        title=f"OLS: {x_lbl} → {y_lbl}",
    ))

    residuals_store[(X, Y)] = {
        "sub":    sub,
        "result": result,
        "X": X, "Y": Y,
        "x_lbl": x_lbl, "y_lbl": y_lbl,
        "group":  grp,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PART C: MUSIC ASSOCIATIONS WITH RESIDUALS
# ══════════════════════════════════════════════════════════════════════════════
#
# Residual labeling (binary, based on sign):
#   residual < 0  →  "Better than expected (Resilient)"
#                    participant's outcome is lower than the model predicts
#                    given their exposure/distress level — a protective signal.
#   residual ≥ 0  →  "Worse than expected (Vulnerable)"
#                    participant's outcome is higher than predicted — a risk signal.
#
# Spearman ρ is used for all music correlations (ordinal × continuous residual).
# ══════════════════════════════════════════════════════════════════════════════

section("5. PART C — MUSIC ASSOCIATIONS WITH RESIDUALS")

music_avail = [v for v in ALL_MUSIC_VARS if v in df.columns]

for key, data in list(residuals_store.items()):
    X, Y     = data["X"], data["Y"]
    sub      = data["sub"].copy()
    x_lbl    = data["x_lbl"]
    y_lbl    = data["y_lbl"]
    grp      = data["group"]
    result   = data["result"]

    print(f"\n{'─' * 70}")
    print(f"  {grp}:  {x_lbl}  →  {y_lbl}")
    print(f"{'─' * 70}")

    resid = sub["residual"]

    # ── Binary classification — sign of residual only ─────────────────────────
    sub["group"] = resid.apply(
        lambda r: "Better than expected (Resilient)"
                  if r < 0 else "Worse than expected (Vulnerable)"
    )

    n_resilient  = (sub["group"] == "Better than expected (Resilient)").sum()
    n_vulnerable = (sub["group"] == "Worse than expected (Vulnerable)").sum()
    print(f"\n  Residual classification (sign of residual):")
    print(f"    Better than expected (Resilient)   N={n_resilient}")
    print(f"    Worse than expected  (Vulnerable)  N={n_vulnerable}")

    # ── Spearman ρ: residuals × all 16 music variables ───────────────────────
    # Spearman operates on ranks, so original (non-z-scored) music values are
    # used regardless of the z-scored regression that produced the residuals.
    print(f"\n  Spearman ρ: residuals × music variables  (n={len(sub)})")
    print(f"  {'Music Variable':<30} {'ρ':>7}  {'p':>8}  {'n':>4}  sig")
    print(f"  {'─'*30} {'─'*7}  {'─'*8}  {'─'*4}  {'─'*3}")

    resid_music_rows = []
    for mv in music_avail:
        combined = pd.concat([resid.rename("resid"),
                               df[mv].rename(mv)], axis=1).dropna()
        if len(combined) < 5:
            continue
        rho, p = stats.spearmanr(combined["resid"], combined[mv])
        rho, p = round(rho, 3), round(p, 4)
        resid_music_rows.append({
            "var": mv, "label": MUSIC_LABELS.get(mv, mv),
            "rho": rho, "p": p, "n": len(combined),
        })
        print(f"  {MUSIC_LABELS.get(mv, mv):<30} {rho:>+7.3f}  {p:>8.4f}  "
              f"{len(combined):>4}  {stars(p)}")

    resid_music_rows.sort(key=lambda r: abs(r["rho"]), reverse=True)

    # Group means on music variables
    sub_music  = sub.join(df[music_avail].reindex(sub.index), how="left")
    group_means = (
        sub_music.groupby("group")[music_avail]
        .mean()
        .rename(columns=MUSIC_LABELS)
        .round(3)
    )
    print(f"\n  Group means on music variables:")
    print(group_means.to_string())

    # Update store entry with classification and music rows for Section 6
    residuals_store[key]["sub"]              = sub
    residuals_store[key]["resid_music_rows"] = resid_music_rows

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

section("6. VISUALISATIONS")

# Passive/listening vars get teal; active engagement vars get blue.
LISTENING_VARS = {"listen_week", "listen_time", "listen_war", "listen_war_time",
                  "music_help_sleep", "music_help_stress"}


def bar_color(var: str, p: float) -> tuple:
    alpha = 1.0 if p < 0.05 else 0.30
    if var in LISTENING_VARS:
        return (0.09, 0.63, 0.58, alpha)   # teal — passive listening
    return (0.13, 0.47, 0.71, alpha)       # blue — active engagement


for key, data in residuals_store.items():
    if not isinstance(data, dict) or "resid_music_rows" not in data:
        continue

    sub        = data["sub"]
    X, Y       = data["X"], data["Y"]
    x_lbl      = data["x_lbl"]
    y_lbl      = data["y_lbl"]
    grp        = data["group"]
    music_rows = data["resid_music_rows"]
    result     = data["result"]

    # Regression line parameters from statsmodels result
    intercept_val = result.params[0]
    slope_val     = result.params[1]

    # ── Plot 1: Scatter coloured by residual group (2 groups) ─────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    x_line = np.linspace(sub[X].min(), sub[X].max(), 200)
    ax.plot(x_line, intercept_val + slope_val * x_line,
            color="black", linewidth=1.8, zorder=1, label="Regression line")

    for group_label, color in GROUP_COLORS.items():
        mask  = sub["group"] == group_label
        n_grp = mask.sum()
        ax.scatter(sub.loc[mask, X], sub.loc[mask, Y],
                   color=color, alpha=0.72, edgecolors="none", s=48,
                   zorder=2, label=f"{group_label} (n={n_grp})")

    ax.set_xlabel(f"{x_lbl} (z-scored)", fontsize=10)
    ax.set_ylabel(f"{y_lbl} (z-scored)", fontsize=10)
    ax.set_title(
        f"{x_lbl}  →  {y_lbl}\nColoured by Residual Group",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    save_fig(f"scatter_resid_{X}_{Y}.png")

    # ── Plot 2: Horizontal bar chart — music factors vs residuals ─────────────
    if not music_rows:
        continue

    df_bars = pd.DataFrame(music_rows).sort_values("rho", ascending=True)
    colors  = [bar_color(r["var"], r["p"]) for _, r in df_bars.iterrows()]

    fig_h = max(5, len(df_bars) * 0.42)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    bars = ax.barh(df_bars["label"], df_bars["rho"],
                   color=colors, edgecolor="white", height=0.68)

    for bar, (_, row) in zip(bars, df_bars.iterrows()):
        s = stars(row["p"])
        if s:
            x_pos = row["rho"] + (0.012 if row["rho"] >= 0 else -0.012)
            ha    = "left" if row["rho"] >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    s, va="center", ha=ha, fontsize=9,
                    fontweight="bold", color="black")

    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Spearman ρ with residuals", fontsize=10)
    ax.set_title(
        f"{x_lbl}  →  {y_lbl}\n"
        f"Negative ρ = linked to better-than-predicted outcome (Resilience)",
        fontsize=10, fontweight="bold",
    )

    legend_handles = [
        mpatches.Patch(color=(0.13, 0.47, 0.71, 1.0),  label="Active engagement (p < .05)"),
        mpatches.Patch(color=(0.13, 0.47, 0.71, 0.30), label="Active engagement (n.s.)"),
        mpatches.Patch(color=(0.09, 0.63, 0.58, 1.0),  label="Passive listening (p < .05)"),
        mpatches.Patch(color=(0.09, 0.63, 0.58, 0.30), label="Passive listening (n.s.)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    save_fig(f"bar_resid_{X}_{Y}.png")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

section("ANALYSIS COMPLETE")
print(f"  Results folder       : {OUT_DIR}")
print(f"  Full report          : {REPORT_FILE}")
print(f"  Correlation tables   : {CORR_FILE}")
print(f"  Plots per model      : scatter_resid_<X>_<Y>.png, bar_resid_<X>_<Y>.png")
print()

_tee.close()
