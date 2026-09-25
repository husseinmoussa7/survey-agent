#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STEP 3: Build per-model, per-penalty averages over seeds.

For each penalty in PENALTIES, produces a CSV where:
  - each row is a model (OLS, OLS_pen, Lasso)
  - metrics are averaged over all seeds
  - we include metrics for BOTH targets:
      * 'd'    (Cohen's d)
      * 'norm' (normalized bias)

ACTUAL directory structure from step 2 (based on find output):

outputs/step2/
  waves_1_to_4_combined_results_same_seeds_d/
    no_penalty/linear_with_intercept/<ANALYSIS>_linear_with_intercept_ALL_SEEDS_SUMMARY.csv
    no_penalty/lasso_on_bias/<ANALYSIS>_lasso_on_bias_ALL_SEEDS_SUMMARY.csv
    pen_<pen>_fixed_beta/linear_with_penalty_fixed_pen_<pen>/<ANALYSIS>_linear_with_penalty_fixed_pen_<pen>_ALL_SEEDS_SUMMARY.csv

  waves_1_to_4_combined_results_same_seeds_norm/
    (same structure)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ANALYSIS_NAME = "waves_1_to_4_combined"

PENALTIES = [0.001, 0.1, 1, 10, 20]

# metrics we care about from ALL_SEEDS_SUMMARY
METRIC_COLS = [
    "correct_direction",
    "V_MSE_d_Base",
    "V_MSE_d_Deb",
    "T_MSE_d_Base",
    "T_MSE_d_Deb",
    "R2_Valid",
    "R2_Train",
    "improvement_pct",
]


def _project_root() -> Path:
    """
    Assumes this file lives in:
      <PROJECT_ROOT>/code/simulate_response/step3_...
    So PROJECT_ROOT is parents[2].
    """
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STEP2_ROOT = OUTPUT_DIR / "step2"
STEP3_ROOT = OUTPUT_DIR / "step3"

BASE_D = STEP2_ROOT / f"{ANALYSIS_NAME}_results_same_seeds_d"
BASE_N = STEP2_ROOT / f"{ANALYSIS_NAME}_results_same_seeds_norm"


def _ensure_improvement_pct(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    If improvement_pct is missing, compute it per-row as:
      ((V_MSE_d_Base - V_MSE_d_Deb) / V_MSE_d_Base) * 100
    Requires V_MSE_d_Base and V_MSE_d_Deb.
    """
    if "improvement_pct" in df.columns:
        return df

    needed = {"V_MSE_d_Base", "V_MSE_d_Deb"}
    if not needed.issubset(df.columns):
        # Can't compute; leave missing and let caller error if required.
        return df

    base = df["V_MSE_d_Base"].astype(float)
    deb = df["V_MSE_d_Deb"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["improvement_pct"] = ((base - deb) / base) * 100.0

    print(f"[INFO] Computed improvement_pct for: {path.name}")
    return df


def load_mean_metrics(path: Path) -> dict:
    """
    Load an ALL_SEEDS_SUMMARY.csv and return metric means over seeds.
    Adds correct_direction_rate if valid_N exists.

    Robustness:
      - If improvement_pct is missing, compute it if possible.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot find summary file: {path}")

    df = pd.read_csv(path)
    df = _ensure_improvement_pct(df, path)

    missing = [c for c in METRIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in: {path}")

    metric_means = df[METRIC_COLS].mean(numeric_only=True).to_dict()

    if "correct_direction" in df.columns and "valid_N" in df.columns:
        metric_means["correct_direction_rate"] = (df["correct_direction"] / df["valid_N"]).mean()
    else:
        metric_means["correct_direction_rate"] = np.nan

    return metric_means


def load_optional_hparams(path: Path) -> tuple[float, float]:
    """
    Return (Pen_Coef_mean, Alpha_mean) if columns exist; else (nan, nan).
    Note: fixed-beta summaries typically DO NOT include these columns.
    """
    df = pd.read_csv(path)
    pen_mean = float(df["Pen_Coef"].mean()) if "Pen_Coef" in df.columns else np.nan
    alpha_mean = float(df["Alpha"].mean()) if "Alpha" in df.columns else np.nan
    return pen_mean, alpha_mean


def build_summary_for_penalty(penalty: float) -> pd.DataFrame:
    """
    For a given penalty value, build a small DataFrame with rows:
      - OLS
      - OLS_pen (fixed-beta directional penalty for this penalty)
      - Lasso
    Each row has metrics for both targets: *_d and *_norm.
    """
    rows = []

    targets = [
        ("d", BASE_D, "d"),
        ("norm", BASE_N, "norm"),
    ]

    # --- 1) OLS (no penalty) ---
    # UPDATED to match Step 2 outputs:
    # no_penalty/linear_with_intercept/<ANALYSIS>_linear_with_intercept_ALL_SEEDS_SUMMARY.csv
    ols_row = None
    for _, base_root, suffix in targets:
        ols_path = (
            base_root
            / "no_penalty"
            / "linear_with_intercept"
            / f"{ANALYSIS_NAME}_linear_with_intercept_ALL_SEEDS_SUMMARY.csv"
        )
        if not ols_path.exists():
            print(f"[WARN] Missing OLS file for target={suffix}: {ols_path}")
            continue

        metrics = load_mean_metrics(ols_path)
        pen_mean, alpha_mean = load_optional_hparams(ols_path)

        if ols_row is None:
            ols_row = {
                "model_label": "OLS",
                "model_type": "linear_with_intercept",
                "penalty_value": penalty,
                "Pen_Coef_mean": pen_mean,
                "Alpha_mean": alpha_mean,
            }

        for col, val in metrics.items():
            ols_row[f"{col}_{suffix}"] = val

    if ols_row is not None:
        rows.append(ols_row)
    else:
        print(f"[WARN] No OLS row built for penalty={penalty} (check Step 2 outputs).")

    # --- 2) OLS_pen (fixed-beta directional penalty for this penalty) ---
    ols_pen_row = None
    model_type = f"linear_with_penalty_fixed_pen_{penalty:g}"
    for _, base_root, suffix in targets:
        pen_path = (
            base_root
            / f"pen_{penalty:g}_fixed_beta"
            / model_type
            / f"{ANALYSIS_NAME}_{model_type}_ALL_SEEDS_SUMMARY.csv"
        )
        if not pen_path.exists():
            print(f"[WARN] Missing fixed-beta file for target={suffix}, pen={penalty}: {pen_path}")
            continue

        metrics = load_mean_metrics(pen_path)

        if ols_pen_row is None:
            # Fixed-beta summaries usually don't include Pen_Coef/Alpha, so set them explicitly.
            ols_pen_row = {
                "model_label": "OLS_pen",
                "model_type": model_type,
                "penalty_value": penalty,
                "Pen_Coef_mean": float(penalty),
                "Alpha_mean": np.nan,
            }

        for col, val in metrics.items():
            ols_pen_row[f"{col}_{suffix}"] = val

    if ols_pen_row is not None:
        rows.append(ols_pen_row)
    else:
        print(f"[WARN] No OLS_pen row built for penalty={penalty} (check fixed-beta folders).")

    # --- 3) Lasso (no directional penalty) ---
    # UPDATED to match Step 2 outputs:
    # no_penalty/lasso_on_bias/<ANALYSIS>_lasso_on_bias_ALL_SEEDS_SUMMARY.csv
    lasso_row = None
    for _, base_root, suffix in targets:
        lasso_path = (
            base_root
            / "no_penalty"
            / "lasso_on_bias"
            / f"{ANALYSIS_NAME}_lasso_on_bias_ALL_SEEDS_SUMMARY.csv"
        )
        if not lasso_path.exists():
            print(f"[WARN] Missing Lasso file for target={suffix}: {lasso_path}")
            continue

        metrics = load_mean_metrics(lasso_path)
        pen_mean, alpha_mean = load_optional_hparams(lasso_path)

        if lasso_row is None:
            lasso_row = {
                "model_label": "Lasso",
                "model_type": "lasso_on_bias",
                "penalty_value": penalty,
                "Pen_Coef_mean": pen_mean,
                "Alpha_mean": alpha_mean,
            }

        for col, val in metrics.items():
            lasso_row[f"{col}_{suffix}"] = val

    if lasso_row is not None:
        rows.append(lasso_row)
    else:
        print(f"[WARN] No Lasso row built for penalty={penalty} (check Step 2 outputs).")

    # Ensure consistent columns even if something is missing
    base_cols = ["model_label", "model_type", "penalty_value", "Pen_Coef_mean", "Alpha_mean"]
    metric_cols = []
    for suffix in ["d", "norm"]:
        metric_cols += [f"{c}_{suffix}" for c in METRIC_COLS]
        metric_cols += [f"correct_direction_rate_{suffix}"]  # derived
    all_cols = base_cols + metric_cols

    df_out = pd.DataFrame(rows)
    for c in all_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan

    return df_out[all_cols]


def main() -> None:
    STEP3_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[INFO] Reading Step 2 from: {STEP2_ROOT}")
    print(f"[INFO] Writing Step 3 to: {STEP3_ROOT}")

    all_penalty_tables = []

    for pen in PENALTIES:
        df_pen = build_summary_for_penalty(pen)
        out_path = STEP3_ROOT / f"{ANALYSIS_NAME}_AVG_METRICS_pen_{pen:g}.csv"
        df_pen.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")
        all_penalty_tables.append(df_pen.assign(penalty_value=pen))

    combined = pd.concat(all_penalty_tables, ignore_index=True)
    combined_path = STEP3_ROOT / f"{ANALYSIS_NAME}_AVG_METRICS_ALL_PENALTIES.csv"
    combined.to_csv(combined_path, index=False)
    print(f"✅ Saved: {combined_path}")


if __name__ == "__main__":
    main()
