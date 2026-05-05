#!/usr/bin/env python3
"""
STAT 482 — FinalReport EDA and Fixed-Effect Modeling
=================================================
Fetches 2025 MLB Statcast data via pybaseball, cleans it per the report
criteria, and generates the EDA plus fixed-effect model figures used in
FinalReport.

Figures produced (saved to ../build/figures/):
  1. fig_whiff_by_pitch_type.png
  2. fig_physical_distributions.png
  3. fig_sequence_heatmap.png
  4. fig_batter_whiff_dist.png
  5. fig_correlation_matrix.png
  6. fig_baseline_coefficients.png
  7. fig_sequencing_terms.png

Usage:
    python3 code/eda_analysis_regression.py
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pybaseball import statcast

# ── paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
BUILD_DIR    = os.path.join(PROJECT_ROOT, "build")
FIG_DIR      = os.path.join(BUILD_DIR, "figures")
RESULT_DIR   = os.path.join(BUILD_DIR, "results")
CACHE_FILE   = os.path.join(DATA_DIR, "statcast_2025.csv")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BUILD_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = sns.color_palette("colorblind")

# ── pitch-type labels (mapping Statcast codes → readable names) ───
PITCH_LABELS = {
    "FF": "4-Seam FB",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle-Curve",
    "CH": "Changeup",
    "FS": "Splitter",
}
KEEP_PITCH_TYPES = set(PITCH_LABELS.keys())

# Whiff descriptions
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}

# Non-competitive descriptions to remove
NON_COMPETITIVE = {
    "pitchout", "intent_ball", "intentional_ball",
    "pitchout_hit_into_play",
}


# ════════════════════════════════════════════════════════════════════
#  1. DATA FETCH
# ════════════════════════════════════════════════════════════════════
def fetch_data() -> pd.DataFrame:
    """Fetch 2025 Statcast data; cache to CSV for subsequent runs."""
    if os.path.exists(CACHE_FILE):
        print(f"[INFO] Loading cached data from {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, low_memory=False)
    else:
        print("[INFO] Fetching 2025 Statcast data from Baseball Savant...")
        print("       (This may take several minutes.)")
        # Fetch in month-long chunks to avoid timeouts
        chunks = []
        date_ranges = [
            ("2025-03-27", "2025-04-30"),
            ("2025-05-01", "2025-05-31"),
            ("2025-06-01", "2025-06-30"),
            ("2025-07-01", "2025-07-31"),
            ("2025-08-01", "2025-08-31"),
            ("2025-09-01", "2025-09-28"),
        ]
        for start, end in date_ranges:
            print(f"       Fetching {start} to {end} ...")
            chunk = None
            last_err = None
            for attempt in range(1, 4):
                try:
                    chunk = statcast(start_dt=start, end_dt=end, parallel=False)
                    break
                except Exception as err:
                    last_err = err
                    print(f"       [WARN] Attempt {attempt}/3 failed: {err}")
                    time.sleep(5 * attempt)

            if chunk is None:
                raise RuntimeError(
                    f"Failed to fetch Statcast data for {start} to {end} after 3 attempts"
                ) from last_err

            chunks.append(chunk)
            print(f"       → {len(chunk):,} pitches")
        df = pd.concat(chunks, ignore_index=True)
        print(f"[INFO] Saving {len(df):,} raw pitches to {CACHE_FILE}")
        df.to_csv(CACHE_FILE, index=False)
    return df


# ════════════════════════════════════════════════════════════════════
#  2. DATA CLEANING
# ════════════════════════════════════════════════════════════════════
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the project cleaning filters used in FinalReport."""
    n_raw = len(raw)
    print(f"\n[CLEAN] Starting with {n_raw:,} raw pitches")

    df = raw.copy()

    # 1. Remove non-competitive events
    df = df[~df["description"].isin(NON_COMPETITIVE)]
    # Also filter by pitch type if there's a dedicated column
    if "pitch_type" in df.columns:
        df = df[~df["pitch_type"].isin(["PO", "IN"])]  # pitchout / intentional
    print(f"  After removing non-competitive: {len(df):,}")

    # 2. Drop missing physical measurements
    phys_cols = ["release_speed", "pfx_x", "pfx_z", "release_extension"]
    df = df.dropna(subset=phys_cols)
    print(f"  After dropping missing physics:  {len(df):,}")

    # 3. Remove unrealistic release speeds
    df = df[(df["release_speed"] >= 65) & (df["release_speed"] <= 105)]
    print(f"  After speed filter [65–105]:     {len(df):,}")

    # 4. Keep only the 9 pitch types
    df = df[df["pitch_type"].isin(KEEP_PITCH_TYPES)]
    print(f"  After keeping 9 pitch types:     {len(df):,}")

    # Create readable labels
    df["pitch_label"] = df["pitch_type"].map(PITCH_LABELS)

    # 5. Convert pfx_z and pfx_x from feet to inches
    df["ivb_inches"] = df["pfx_z"] * 12
    df["hb_inches"]  = df["pfx_x"] * 12

    # 6. Binary whiff column
    df["whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(int)

    # 6. Lag variables within each plate appearance
    #    Group by game + at-bat number
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    grp = df.groupby(["game_pk", "at_bat_number"])
    df["prev_pitch_type"]  = grp["pitch_type"].shift(1).fillna("None")
    df["prev_pitch_label"] = df["prev_pitch_type"].map(
        lambda x: PITCH_LABELS.get(x, "None")
    )
    df["prev_zone"]        = grp["zone"].shift(1).fillna(-1).astype(int)

    # Simplified previous outcome categories
    def classify_outcome(desc):
        if pd.isna(desc):
            return "None"
        if "ball" in desc and "foul" not in desc:
            return "Ball"
        if "called_strike" in desc:
            return "Called Strike"
        if "swinging_strike" in desc:
            return "Swinging Strike"
        if "foul" in desc:
            return "Foul"
        if "hit_into_play" in desc:
            return "In Play"
        return "Other"

    df["prev_outcome"] = grp["description"].shift(1).apply(classify_outcome)

    n_clean = len(df)
    print(f"\n[CLEAN] Final cleaned dataset: {n_clean:,} pitches")

    return df, n_raw


# ════════════════════════════════════════════════════════════════════
#  3. SUMMARY STATISTICS
# ════════════════════════════════════════════════════════════════════
def print_summary(df: pd.DataFrame, n_raw: int):
    """Print LaTeX-ready summary statistics."""
    overall_whiff = df["whiff"].mean() * 100

    # Batters with >= 200 pitches seen
    batter_whiffs = (
        df.groupby("batter")
        .agg(n_pitches=("whiff", "size"), whiff_rate=("whiff", "mean"))
    )
    batter_200 = batter_whiffs[batter_whiffs["n_pitches"] >= 200]
    batter_200_rate = batter_200["whiff_rate"] * 100

    n_pitchers = df["pitcher"].nunique()

    print("\n" + "=" * 60)
    print("  LATEX MACRO VALUES  (copy to build/FinalReport.tex)")
    print("=" * 60)
    print(f"  \\totalRaw     = {n_raw:,}")
    print(f"  \\totalPitches = {len(df):,}")
    print(f"  \\overallWhiff = {overall_whiff:.1f}")
    print(f"  \\numBatters   = {len(batter_200)}")
    print(f"  \\numPitchers  = {n_pitchers}")
    print(f"  \\batterLow    = {batter_200_rate.min():.1f}")
    print(f"  \\batterHigh   = {batter_200_rate.max():.1f}")
    print("=" * 60)

    # Summary statistics table (pfx in inches)
    phys = {
        "Release Speed (mph)":          df["release_speed"],
        "Induced Vertical Break (in.)": df["ivb_inches"],
        "Horizontal Break (in.)":       df["hb_inches"],
        "Release Extension (ft)":       df["release_extension"],
    }
    print("\n  SUMMARY STATISTICS TABLE  (lines 170-184)")
    print("-" * 60)
    for name, col in phys.items():
        print(f"  {name:35s}  {col.mean():7.1f}  {col.std():5.1f}  "
              f"{col.min():7.1f}  {col.max():6.1f}")
    print("-" * 60)

    # Whiff by pitch type table
    whiff_tbl = (
        df.groupby("pitch_label")
        .agg(total=("whiff", "size"), whiffs=("whiff", "sum"))
    )
    whiff_tbl["rate"] = whiff_tbl["whiffs"] / whiff_tbl["total"] * 100
    whiff_tbl = whiff_tbl.sort_values("rate", ascending=False)
    print("\n  WHIFF BY PITCH TYPE TABLE  (lines 201-220)")
    print("-" * 60)
    for label, row in whiff_tbl.iterrows():
        print(f"  {label:15s}  {int(row['total']):>8,}  {int(row['whiffs']):>7,}  "
              f"{row['rate']:5.1f}%")
    print("-" * 60)

    return batter_200_rate, overall_whiff


# ════════════════════════════════════════════════════════════════════
#  4. FIGURE GENERATION
# ════════════════════════════════════════════════════════════════════

def fig_whiff_by_pitch_type(df: pd.DataFrame):
    """Figure 1 — bar chart of whiff rate by pitch type."""
    rates = (
        df.groupby("pitch_label")["whiff"]
        .mean()
        .sort_values(ascending=False) * 100
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(rates.index, rates.values, color=PALETTE[:len(rates)],
                  edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Whiff Rate (%)")
    ax.set_xlabel("Pitch Type")
    ax.set_title("Whiff Rate by Pitch Type — 2025 MLB Regular Season")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    # Add value labels on bars
    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_whiff_by_pitch_type.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved {path}")


def fig_physical_distributions(df: pd.DataFrame):
    """Figure 2 — 2×2 density plots of physical characteristics."""
    cols  = ["release_speed", "ivb_inches", "hb_inches", "release_extension"]
    names = ["Release Speed (mph)", "Induced Vertical Break (in.)",
             "Horizontal Break (in.)", "Release Extension (ft)"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col, name in zip(axes.flat, cols, names):
        for label, color in [("No Whiff", PALETTE[0]), ("Whiff", PALETTE[1])]:
            subset = df[df["whiff"] == (1 if label == "Whiff" else 0)]
            ax.hist(subset[col], bins=80, density=True, alpha=0.5,
                    label=label, color=color)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
    fig.suptitle("Physical Pitch Characteristics — Whiff vs. No Whiff",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_physical_distributions.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved {path}")


def fig_sequence_heatmap(df: pd.DataFrame):
    """Figure 3 — heatmap of whiff rate by (previous → current) pitch type."""
    # Exclude first pitches (prev = "None")
    seq = df[df["prev_pitch_label"] != "None"].copy()

    # Compute whiff rate per cell
    ct = seq.groupby(["prev_pitch_label", "pitch_label"]).agg(
        n=("whiff", "size"), whiffs=("whiff", "sum")
    ).reset_index()
    ct["rate"] = ct["whiffs"] / ct["n"] * 100

    # Filter cells with fewer than 100 observations
    ct = ct[ct["n"] >= 100]

    # Pivot
    pivot = ct.pivot(index="prev_pitch_label", columns="pitch_label", values="rate")

    # Order by overall whiff rate
    order = (
        df.groupby("pitch_label")["whiff"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    # Keep only labels that exist in our data
    order = [o for o in order if o in pivot.columns]
    row_order = [o for o in order if o in pivot.index]
    pivot = pivot.reindex(index=row_order, columns=order)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Whiff Rate (%)"})
    ax.set_xlabel("Current Pitch Type")
    ax.set_ylabel("Previous Pitch Type")
    ax.set_title("Whiff Rate by Pitch Sequence — 2025 MLB")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_sequence_heatmap.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved {path}")


def fig_batter_whiff_dist(batter_rates: pd.Series, overall_whiff: float):
    """Figure 4 — histogram of per-batter whiff rates."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(batter_rates, bins=40, color=PALETTE[0], edgecolor="white",
            linewidth=0.8, alpha=0.85)
    ax.axvline(overall_whiff, color="red", linestyle="--", linewidth=1.5,
               label=f"League avg = {overall_whiff:.1f}%")
    ax.set_xlabel("Batter Whiff Rate (%)")
    ax.set_ylabel("Number of Batters")
    ax.set_title("Distribution of Batter-Level Whiff Rates (≥ 200 pitches seen)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_batter_whiff_dist.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved {path}")


def fig_correlation_matrix(df: pd.DataFrame):
    """Figure 5 — correlation matrix of continuous predictors + whiff."""
    cols = ["release_speed", "ivb_inches", "hb_inches", "release_extension", "whiff"]
    labels = ["Speed", "IVB", "HB", "Extension", "Whiff"]
    corr = df[cols].corr()
    corr.index   = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                square=True)
    ax.set_title("Correlation Matrix — Pitch Characteristics & Whiff Outcome")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_correlation_matrix.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[FIG] Saved {path}")

# ════════════════════════════════════════════════════════════════════
#  BASELINE REGRESSION  
# ════════════════════════════════════════════════════════════════════

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def run_logistic_regression(df: pd.DataFrame):
    """Baseline logistic regression model for whiff prediction."""

    print("\n[MODEL] Running baseline logistic regression...")

    # --- Features (simple baseline, numeric only) ---
    features = ["release_speed", "ivb_inches", "hb_inches", "release_extension"]
    X = df[features]
    y = df["whiff"]

    # Drop any remaining missing values (safety)
    X = X.dropna()
    y = y.loc[X.index]

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Fit model ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # --- Predictions ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Evaluation ---
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n[MODEL RESULTS]")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC-AUC:  {auc:.4f}")

    # --- Coefficients ---
    print("\n[MODEL COEFFICIENTS]")
    for name, coef in zip(features, model.coef_[0]):
        print(f"  {name:20s}: {coef: .4f}")

    return model
  
# ════════════════════════════════════════════════════════════════════
#  5. SEQUENCING MODEL  (Baseline + Sequencing Model, Eq. 2 & 3)
# ════════════════════════════════════════════════════════════════════
#
#    Baseline (Eq. 2):
#      logit(p_i) = β X_i
#      where X_i = [Speed, IVB, HB, Ext]  (physical characteristics)
#
#    Sequencing (Eq. 3):
#      logit(p_i) = β X_i + γ Z_{i-1} + α (T_i * T_{i-1})
#      where Z_{i-1} = [prev_pitch_type, prev_zone_cat, prev_outcome]
#            T_i * T_{i-1} = dummy-coded current × previous pitch-type
#                            interaction
#
#  Estimation: Newton-Raphson / IRLS via scipy.optimize.minimize
#  (L-BFGS-B), which replicates the MLE that statsmodels or R's glm()
#  would produce.  Standard errors come from the observed Fisher
#  information (inverse Hessian at the MLE), identical to what glm()
#  reports as Wald SEs.
#
#  Model comparison: Likelihood Ratio Test (χ² with Δdf degrees of
#  freedom) and BIC
# ════════════════════════════════════════════════════════════════════

from scipy import optimize, stats
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── helpers ──────────────────────────────────────────────────────────

def _sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _neg_loglik(beta, X, y):
    """Negative log-likelihood for logistic regression."""
    p = _sigmoid(X @ beta)
    # Clip to avoid log(0)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def _neg_loglik_grad(beta, X, y):
    """Gradient of negative log-likelihood."""
    p = _sigmoid(X @ beta)
    return X.T @ (p - y)


def _fit_logistic(X, y, verbose=False):
    """
    Fit logistic regression via L-BFGS-B (equivalent to glm(..., family=binomial)
    in R).  Returns (beta_hat, se, log_lik, converged).

    Standard errors are derived from the observed Fisher information matrix
    (inverse Hessian at the MLE), matching the Wald SEs reported by
    statsmodels.Logit and R's glm().
    """
    n, k = X.shape
    beta0 = np.zeros(k)

    result = optimize.minimize(
        _neg_loglik,
        beta0,
        args=(X, y),
        jac=_neg_loglik_grad,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )

    beta_hat = result.x
    converged = result.success

    if not converged and verbose:
        print(f"  [WARN] Optimizer did not fully converge: {result.message}")

    # Observed Fisher information  I = X^T W X  where W = diag(p(1-p))
    p_hat = _sigmoid(X @ beta_hat)
    w = p_hat * (1.0 - p_hat)
    # Clamp weights to avoid near-zero division
    w = np.clip(w, 1e-10, None)
    XtWX = (X * w[:, None]).T @ X

    try:
        cov = np.linalg.inv(XtWX)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        # If information matrix is singular, fall back to pseudo-inverse
        cov = np.linalg.pinv(XtWX)
        se = np.sqrt(np.abs(np.diag(cov)))
        if verbose:
            print("  [WARN] Information matrix singular; using pseudo-inverse for SEs.")

    log_lik = -result.fun
    return beta_hat, se, log_lik, converged


def _build_design_matrix(df, feature_cols, add_intercept=True):
    """
    Stack feature columns into a design matrix.
    All columns must already be numeric (dummies expanded upstream).
    Returns X as a float64 ndarray.
    """
    X = df[feature_cols].to_numpy(dtype=np.float64)
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    return X


def _dummy_encode(df, col, drop_first=True):
    """
    One-hot encode a categorical column and return new dummy column names.
    Uses pd.get_dummies with drop_first to avoid perfect collinearity.
    """
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first, dtype=float)
    return dummies


def _likelihood_ratio_test(log_lik_full, log_lik_reduced, df_diff):
    """
    Likelihood Ratio Test: H0 = reduced model fits as well as full model.
    Returns (LR_statistic, p_value).
    """
    lr_stat = 2.0 * (log_lik_full - log_lik_reduced)
    p_value = stats.chi2.sf(lr_stat, df=df_diff)
    return lr_stat, p_value


def _bic(log_lik, n_params, n_obs):
    """BIC = -2·ℓ + k·ln(n)."""
    return -2.0 * log_lik + n_params * np.log(n_obs)


# ── zone binning ─────────────────────────────────────────────────────
# Statcast zones 1–9 are in the strike zone, 11–14 are balls.
# We bin them into four coarser location categories to limit the
# number of dummy parameters introduced by the 14-level zone variable,
# while still capturing the key ball-vs-strike distinction.
def _bin_zone(zone):
    """
    Map Statcast zone codes to four location categories:
      'Heart'  — zones 5 (middle-middle)
      'Strike' — zones 1–4, 6–9 (other strike-zone cells)
      'Chase'  — zones 11–14 (balls just off the plate)
      'None'   — first pitch of at-bat (no previous pitch)
    """
    if zone == -1 or pd.isna(zone):
        return "None"
    z = int(zone)
    if z == 5:
        return "Heart"
    if 1 <= z <= 9:
        return "Strike"
    if 11 <= z <= 14:
        return "Chase"
    return "None"


def export_report2_model_input(df: pd.DataFrame):
    """Save the cleaned modeling dataset for the combined mixed model."""
    export_df = df.copy()
    export_df["prev_zone_cat"] = export_df["prev_zone"].apply(_bin_zone)
    export_df["pitch_seq"] = export_df["prev_pitch_label"] + "__" + export_df["pitch_label"]

    cols = [
        "whiff",
        "release_speed",
        "ivb_inches",
        "hb_inches",
        "release_extension",
        "pitch_label",
        "prev_pitch_label",
        "prev_zone_cat",
        "prev_outcome",
        "pitch_seq",
        "batter",
    ]

    out_path = os.path.join(DATA_DIR, "report2_model_input.csv")
    export_df[cols].to_csv(out_path, index=False)
    print(f"[DATA] Saved combined-model input to {out_path}")


# ════════════════════════════════════════════════════════════════════
#  SEQUENCING FUNCTION
# ════════════════════════════════════════════════════════════════════

def run_sequencing_model(df: pd.DataFrame, seed: int = 42):
    """
    Fit and compare the Baseline (Eq. 2) and Sequencing (Eq. 3) logistic
    regression models proposed in the STAT 482 project.

    Parameters
    ----------
    df   : cleaned DataFrame produced by clean_data()
    seed : random seed for numpy operations

    Prints
    ------
    Coefficient tables for both models (Wald z and p-values), LRT
    result, BIC comparison, and LaTeX macro block for FinalReport.tex.

    Returns
    -------
    dict with fitted coefficients, SEs, log-likelihoods, LRT/BIC
    results, column name lists, and the full sequencing coefficient
    DataFrame (seq_coef_df) for downstream use.
    """

    np.random.seed(seed)

    print("\n" + "=" * 70)
    print("  SEQUENCING MODEL ANALYSIS")
    print("=" * 70)

    # ── 5.1  Prepare modelling dataset ───────────────────────────────
    # Drop first pitches of each plate appearance (prev_pitch_label == "None")
    # for the sequencing model, but retain them for the baseline model.
    # We build two separate design matrices accordingly.

    phys_cols = ["release_speed", "ivb_inches", "hb_inches", "release_extension"]

    # Bin the previous zone into four location categories
    df = df.copy()
    df["prev_zone_cat"] = df["prev_zone"].apply(_bin_zone)

    # Construct current × previous pitch-type interaction label
    df["pitch_seq"] = df["prev_pitch_label"] + "__" + df["pitch_label"]

    # ── 5.1a  Baseline model dataset (all pitches with valid physics) ─
    base_df = df[phys_cols + ["whiff"]].dropna()
    print(f"\n[MODEL] Baseline dataset:    {len(base_df):,} pitches")

    # ── 5.1b  Sequencing model dataset (exclude first-pitch-of-PA rows
    #          where no lagged information exists) ──────────────────────
    seq_df = df[
        (df["prev_pitch_label"] != "None") &
        df[phys_cols].notna().all(axis=1)
    ].copy()
    print(f"[MODEL] Sequencing dataset:  {len(seq_df):,} pitches "
          f"(first pitches of PA excluded)")

    # ── 5.2  Dummy-encode categorical sequencing variables ────────────
    # Variables entering Z_{i-1}: prev_pitch_label, prev_zone_cat,
    #                               prev_outcome
    # Variable entering T_i * T_{i-1}: pitch_seq (interaction)

    dum_prev_type    = _dummy_encode(seq_df, "prev_pitch_label", drop_first=True)
    dum_prev_zone    = _dummy_encode(seq_df, "prev_zone_cat",    drop_first=True)
    dum_prev_outcome = _dummy_encode(seq_df, "prev_outcome",     drop_first=True)

    # For the interaction term T_i * T_{i-1} we retain sequences with
    # at least MIN_SEQ_OBS observations to avoid perfect separation on
    # extremely rare pitch-type pairs.
    MIN_SEQ_OBS = 100

    seq_df["pitch_seq"] = seq_df["pitch_seq"].astype(str)  # drop categorical if present

    seq_counts_filtered = seq_df["pitch_seq"].value_counts()
    common_seqs = seq_counts_filtered[seq_counts_filtered >= MIN_SEQ_OBS].index
    seq_df["pitch_seq_filtered"] = np.where(
        seq_df["pitch_seq"].isin(common_seqs),
        seq_df["pitch_seq"],
        "Other__Other",
    )

    dum_seq = _dummy_encode(seq_df, "pitch_seq_filtered", drop_first=True)


    n_rare = (seq_df["pitch_seq"].isin(common_seqs) == False).sum()
    print(f"[MODEL] Pitch-type sequences with >= {MIN_SEQ_OBS} obs retained: "
          f"{len(common_seqs)} of {seq_counts_filtered.shape[0]} unique sequences "
          f"({n_rare:,} pitches collapsed to 'Other__Other')")

    # Assemble full sequencing design matrix
    seq_feat_df = pd.concat(
        [seq_df[phys_cols].reset_index(drop=True),
         dum_prev_type.reset_index(drop=True),
         dum_prev_zone.reset_index(drop=True),
         dum_prev_outcome.reset_index(drop=True),
         dum_seq.reset_index(drop=True)],
        axis=1,
    )
    seq_feat_cols = seq_feat_df.columns.tolist()

    y_seq  = seq_df["whiff"].to_numpy(dtype=np.float64)
    X_seq  = _build_design_matrix(seq_feat_df, seq_feat_cols)

    # Baseline uses only the subset of seq_df (same rows) so LRT is valid
    # (models must be fit on identical observations).
    y_base = y_seq.copy()
    X_base = _build_design_matrix(seq_df, phys_cols)   # intercept added inside

    print(f"[MODEL] Baseline design matrix:   {X_base.shape[0]:,} × {X_base.shape[1]}")
    print(f"[MODEL] Sequencing design matrix: {X_seq.shape[0]:,} × {X_seq.shape[1]}")

    # ── 5.3  Standardise physical predictors for numerical stability ──
    # We z-score the four continuous columns in-place on both matrices
    # (columns 1–4 in both; column 0 is the intercept).
    phys_idx = slice(1, 5)   # columns 1,2,3,4 are the four physical vars
    mu_phys  = X_base[:, phys_idx].mean(axis=0)
    sd_phys  = X_base[:, phys_idx].std(axis=0)
    sd_phys[sd_phys == 0] = 1.0   # guard against constant column

    X_base[:, phys_idx] = (X_base[:, phys_idx] - mu_phys) / sd_phys
    X_seq[:, phys_idx]  = (X_seq[:, phys_idx]  - mu_phys) / sd_phys

    # ── 5.4  Fit models ───────────────────────────────────────────────
    print("\n[MODEL] Fitting Baseline model (Eq. 2) ...")
    beta_base, se_base, ll_base, conv_base = _fit_logistic(X_base, y_base, verbose=True)
    print(f"  Log-likelihood: {ll_base:,.2f}  |  Converged: {conv_base}")

    print("[MODEL] Fitting Sequencing model (Eq. 3) ...")
    beta_seq, se_seq, ll_seq, conv_seq = _fit_logistic(X_seq, y_seq, verbose=True)
    print(f"  Log-likelihood: {ll_seq:,.2f}  |  Converged: {conv_seq}")

    # ── 5.5  Likelihood Ratio Test ────────────────────────────────────
    # H0: The sequencing terms (γ Z_{i-1} + α T_i*T_{i-1}) jointly = 0
    df_diff  = X_seq.shape[1] - X_base.shape[1]
    lr_stat, lr_pval = _likelihood_ratio_test(ll_seq, ll_base, df_diff)

    print(f"\n[LRT]  LR statistic  = {lr_stat:,.2f}  (df = {df_diff})")
    print(f"[LRT]  p-value       = {lr_pval:.2e}")
    if lr_pval < 0.001:
        print("[LRT]  → Sequencing terms are jointly highly significant (p < 0.001).")
    elif lr_pval < 0.05:
        print("[LRT]  → Sequencing terms are jointly significant at α = 0.05.")
    else:
        print("[LRT]  → Sequencing terms are NOT jointly significant at α = 0.05.")

    # ── 5.6  BIC comparison ───────────────────────────────────────────
    n_obs     = len(y_base)
    bic_base  = _bic(ll_base, X_base.shape[1], n_obs)
    bic_seq   = _bic(ll_seq,  X_seq.shape[1],  n_obs)
    delta_bic = bic_seq - bic_base   # negative = sequencing model preferred

    print(f"\n[BIC]  Baseline model:    BIC = {bic_base:,.2f}  (k = {X_base.shape[1]})")
    print(f"[BIC]  Sequencing model:  BIC = {bic_seq:,.2f}  (k = {X_seq.shape[1]})")
    print(f"[BIC]  ΔBIC (Seq − Base) = {delta_bic:,.2f}")
    if delta_bic < -10:
        print("[BIC]  → Strong evidence favouring the Sequencing model.")
    elif delta_bic < 0:
        print("[BIC]  → Sequencing model preferred; evidence is modest.")
    else:
        print("[BIC]  → Baseline model preferred on BIC (penalty outweighs gain).")

    # ── 5.7  Coefficient tables ───────────────────────────────────────
    # 5.7a  Baseline model — physical characteristics
    base_col_names = ["Intercept"] + phys_cols
    z_base = beta_base / se_base
    p_base = 2.0 * stats.norm.sf(np.abs(z_base))

    print("\n" + "=" * 70)
    print("  TABLE: BASELINE MODEL — Physical Characteristics")
    print("  (standardised predictors; coefficients are log-odds per SD)")
    print("-" * 70)
    print(f"  {'Predictor':<32}  {'Coef':>8}  {'SE':>6}  {'z':>7}  {'p':>8}")
    print("-" * 70)
    for nm, b, s, z, p in zip(base_col_names, beta_base, se_base, z_base, p_base):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {nm:<32}  {b:8.4f}  {s:6.4f}  {z:7.3f}  {p:8.4f} {sig}")
    print("-" * 70)

    # 5.7b  Sequencing model — sequencing terms only, sorted by |z|
    seq_col_names = ["Intercept"] + seq_feat_cols
    z_seq = beta_seq / se_seq
    p_seq = 2.0 * stats.norm.sf(np.abs(z_seq))

    phys_set    = {"Intercept"} | set(phys_cols)
    seq_indices = [i for i, nm in enumerate(seq_col_names) if nm not in phys_set]

    seq_coef_df = pd.DataFrame({
        "predictor": [seq_col_names[i] for i in seq_indices],
        "coef":       beta_seq[seq_indices],
        "se":         se_seq[seq_indices],
        "z":          z_seq[seq_indices],
        "p":          p_seq[seq_indices],
    }).sort_values("z", key=np.abs, ascending=False)

    print("\n" + "=" * 70)
    print("  TABLE: SEQUENCING MODEL — Top Sequencing Terms (|z| ranked)")
    print("  (physical terms same as baseline, not repeated here)")
    print("-" * 70)
    print(f"  {'Predictor':<45}  {'Coef':>8}  {'SE':>6}  {'z':>7}  {'p':>8}")
    print("-" * 70)
    for _, row in seq_coef_df.head(25).iterrows():
        nm = row["predictor"].replace("prev_pitch_label_", "PrevType:") \
                              .replace("prev_zone_cat_",    "PrevZone:") \
                              .replace("prev_outcome_",     "PrevOut:") \
                              .replace("pitch_seq_filtered_", "Seq:")
        sig = ("***" if row["p"] < 0.001 else
               ("**" if row["p"] < 0.01 else ("*" if row["p"] < 0.05 else "")))
        print(f"  {nm:<45}  {row['coef']:8.4f}  {row['se']:6.4f}  "
              f"{row['z']:7.3f}  {row['p']:8.4f} {sig}")
    if len(seq_coef_df) > 25:
        print(f"  ... ({len(seq_coef_df) - 25} additional terms not shown)")
    print("-" * 70)

    # ── 5.8  LaTeX macro values ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  LATEX MACRO VALUES  (copy to FinalReport.tex)")
    print("=" * 70)
    print(f"  \\seqLLBase     = {ll_base:,.2f}")
    print(f"  \\seqLLSeq      = {ll_seq:,.2f}")
    print(f"  \\seqLRStat     = {lr_stat:,.2f}")
    print(f"  \\seqLRDf       = {df_diff}")
    print(f"  \\seqLRPval     = {lr_pval:.2e}")
    print(f"  \\seqBICBase    = {bic_base:,.2f}")
    print(f"  \\seqBICSeq     = {bic_seq:,.2f}")
    print(f"  \\seqDeltaBIC   = {delta_bic:,.2f}")
    print("=" * 70)

    return {
        "beta_base":      beta_base,
        "se_base":        se_base,
        "ll_base":        ll_base,
        "beta_seq":       beta_seq,
        "se_seq":         se_seq,
        "ll_seq":         ll_seq,
        "lr_stat":        lr_stat,
        "lr_pval":        lr_pval,
        "df_diff":        df_diff,
        "bic_base":       bic_base,
        "bic_seq":        bic_seq,
        "delta_bic":      delta_bic,
        "col_names_base": ["Intercept"] + phys_cols,
        "col_names_seq":  seq_col_names,
        "seq_coef_df":    seq_coef_df,
    }


def save_model_fit_summary(seq_results):
    """Persist baseline and sequencing fit metrics for downstream reporting."""
    fit_df = pd.DataFrame(
        [
            {
                "model": "Baseline",
                "logLik": seq_results["ll_base"],
                "BIC": seq_results["bic_base"],
                "n_params": len(seq_results["beta_base"]),
            },
            {
                "model": "Sequencing",
                "logLik": seq_results["ll_seq"],
                "BIC": seq_results["bic_seq"],
                "n_params": len(seq_results["beta_seq"]),
            },
        ]
    )
    out_path = os.path.join(RESULT_DIR, "model_fit_summary.csv")
    fit_df.to_csv(out_path, index=False)
    print(f"[RESULT] Saved {out_path}")


def fig_baseline_coefficients(seq_results):
    """Figure 6 — baseline physical-effect estimates with 95% CIs."""
    coef_df = pd.DataFrame(
        {
            "predictor": seq_results["col_names_base"][1:],
            "coef": seq_results["beta_base"][1:],
            "se": seq_results["se_base"][1:],
        }
    )
    coef_df["low"] = coef_df["coef"] - 1.96 * coef_df["se"]
    coef_df["high"] = coef_df["coef"] + 1.96 * coef_df["se"]

    label_map = {
        "release_speed": "Release Speed",
        "ivb_inches": "IVB",
        "hb_inches": "Horizontal Break",
        "release_extension": "Extension",
    }
    coef_df["label"] = coef_df["predictor"].map(label_map).fillna(coef_df["predictor"])
    coef_df = coef_df.sort_values("coef")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.errorbar(
        coef_df["coef"],
        coef_df["label"],
        xerr=1.96 * coef_df["se"],
        fmt="o",
        color=PALETTE[0],
        ecolor=PALETTE[2],
        elinewidth=2,
        capsize=4,
    )
    ax.set_xlabel("Log-Odds Change per 1 SD Increase")
    ax.set_ylabel("")
    ax.set_title("Baseline Logistic Regression Coefficients")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "fig_baseline_coefficients.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved {fig_path}")

    out_path = os.path.join(RESULT_DIR, "baseline_coefficients.csv")
    coef_df.to_csv(out_path, index=False)
    print(f"[RESULT] Saved {out_path}")


def fig_sequencing_terms(seq_results, top_n: int = 12):
    """Figure 7 — strongest current/previous pitch-type sequencing terms."""
    seq_df = seq_results["seq_coef_df"].copy()
    seq_df = seq_df[seq_df["predictor"].str.startswith("pitch_seq_filtered_")].copy()
    seq_df = seq_df[np.isfinite(seq_df["se"]) & np.isfinite(seq_df["z"])].copy()
    seq_df = seq_df[seq_df["p"] < 0.05].copy()
    seq_df = seq_df.head(top_n)

    if seq_df.empty:
        print("[WARN] No sequencing interaction terms available for plotting.")
        return

    seq_df["low"] = seq_df["coef"] - 1.96 * seq_df["se"]
    seq_df["high"] = seq_df["coef"] + 1.96 * seq_df["se"]

    def clean_label(value: str) -> str:
        label = value.replace("pitch_seq_filtered_", "")
        parts = label.split("__")
        if len(parts) == 2:
            return f"{parts[0]} -> {parts[1]}"
        return label

    seq_df["label"] = seq_df["predictor"].map(clean_label)
    seq_df = seq_df.sort_values("coef")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.errorbar(
        seq_df["coef"],
        seq_df["label"],
        xerr=1.96 * seq_df["se"],
        fmt="o",
        color=PALETTE[1],
        ecolor=PALETTE[3],
        elinewidth=2,
        capsize=4,
    )
    ax.set_xlabel("Log-Odds Change Relative to Reference Sequence")
    ax.set_ylabel("")
    ax.set_title("Strongest Sequencing Interaction Terms")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "fig_sequencing_terms.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] Saved {fig_path}")

    out_path = os.path.join(RESULT_DIR, "sequencing_top_terms.csv")
    seq_df.to_csv(out_path, index=False)
    print(f"[RESULT] Saved {out_path}")

# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    raw = fetch_data()
    df, n_raw = clean_data(raw)
    batter_rates, overall_whiff = print_summary(df, n_raw)
    export_report2_model_input(df)

    print("\n[INFO] Generating figures...")
    fig_whiff_by_pitch_type(df)
    fig_physical_distributions(df)
    fig_sequence_heatmap(df)
    fig_batter_whiff_dist(batter_rates, overall_whiff)
    fig_correlation_matrix(df)
    print("\n[DONE] All figures saved to", FIG_DIR)

    run_logistic_regression(df)
    seq_results = run_sequencing_model(df)
    save_model_fit_summary(seq_results)
    fig_baseline_coefficients(seq_results)
    fig_sequencing_terms(seq_results)


if __name__ == "__main__":
    main()
