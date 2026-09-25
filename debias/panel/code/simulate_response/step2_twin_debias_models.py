#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STEP 2: Debiasing + seed-hunting on a pre-built master dataframe.

What this script does
- Loads outputs/combined_waves_1_to_4_master.pkl (created by step1_twin_build_master.py).
- Runs seed-hunting for baseline models (OLS, Lasso) where FactorAnalysis is fit per seed split.
- Trains a directional-penalty linear model once on all data per penalty level, selects a best epoch,
  then evaluates that fixed beta across the same seed splits using a shared FactorAnalysis fit on all data.
- Saves per-seed summaries and prediction-level outputs to outputs/step2/.

Assumptions about master_df columns
- Embedding: array-like vector per row
- y_standardized_bias: Cohen's d bias (or overwritten to Normalized_Bias when target="norm")
- Normalized_Bias: normalized bias target
- Average_LLM_Response, Average_Human_Response, s_pooled: for raw-space debiasing metrics
- Variable_Name: identifier for each item
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Optional: PyTorch models (only if available)
try:
    import torch
except ImportError:
    torch = None


# ==========================
# Configuration
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STEP2_OUT_DIR = OUTPUT_DIR / "step2"
MASTER_PKL = OUTPUT_DIR / "combined_waves_1_to_4_master.pkl"

FIXED_N_COMPONENTS = 50
VALIDATION_SIZE = 0.2

IMPROVEMENT_THRESHOLD_PCT = 10.0
DIRECTION_THRESHOLD_FRAC = 0.55

# For baseline seed-hunting when seed_list is not provided
DEFAULT_N_SEEDS_TO_TRY = 200


# ==========================
# Small shared helpers
# ==========================

def predict_d(X_fa: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Predict y in standardized space (d-space) using beta.
    beta can be (k,) or (k+1,) where beta[0] is intercept.
    """
    if beta.shape[0] == X_fa.shape[1] + 1:
        return X_fa @ beta[1:] + beta[0]
    return X_fa @ beta


def raw_mse_and_debiased(
    llm: np.ndarray,
    human: np.ndarray,
    s_pooled: np.ndarray,
    pred_d: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute:
      - baseline raw MSE: mean((llm - human)^2)
      - debiased raw MSE: mean(( (llm - pred_d*s_pooled) - human )^2)
      - debiased predictions in raw space: llm - pred_d*s_pooled
    """
    mse_base = float(np.mean((llm - human) ** 2))
    debiased = llm - pred_d * s_pooled
    mse_deb = float(np.mean((debiased - human) ** 2))
    return mse_base, mse_deb, debiased


def mse_d(y_true_d: np.ndarray, y_pred_d: np.ndarray) -> Tuple[float, float]:
    """
    Returns (mse_baseline_in_d, mse_debiased_in_d).
    Baseline in d-space here is y_true vs 0 (i.e., mean(y_true^2)).
    """
    mse_base = float(np.mean(y_true_d ** 2))
    mse_deb = float(np.mean((y_true_d - y_pred_d) ** 2))
    return mse_base, mse_deb


def improvement_pct(mse_base: float, mse_deb: float) -> float:
    """Percent improvement relative to baseline MSE."""
    if mse_base <= 0:
        return float("nan")
    return float((mse_base - mse_deb) / mse_base * 100.0)


def count_direction_improved(
    llm: np.ndarray,
    human: np.ndarray,
    debiased: np.ndarray,
) -> int:
    """
    Counts how many items got closer to human after debiasing:
      |debiased - human| < |llm - human|
    """
    llm_bias = llm - human
    deb_bias = debiased - human
    return int(np.sum(np.abs(deb_bias) < np.abs(llm_bias)))


def save_all_seeds_summary(
    all_results: List[Dict],
    out_csv_path: Path,
    header_label: str,
) -> None:
    """Print distribution summary and save per-seed CSV."""
    if not all_results:
        print("WARNING: No seeds successfully evaluated; no distribution summary.")
        return

    all_df = pd.DataFrame(all_results)
    dir_corr = all_df["correct_direction"].to_numpy()
    valid_n = int(all_df["valid_N"].iloc[0])

    print(f"\n--- {header_label} ---")
    print(f"Mean Dir.Corr:   {dir_corr.mean():.2f}/{valid_n}")
    print(f"Median Dir.Corr: {np.median(dir_corr):.0f}/{valid_n}")
    print(f"Min/Max Dir.Corr:{dir_corr.min():.0f}/{valid_n}  to  {dir_corr.max():.0f}/{valid_n}")

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_csv_path, index=False)
    print(f"Saved per-seed summary (ALL seeds) -> {out_csv_path}")


# ==========================
# Model training helpers
# ==========================

def train_linear_model(
    X_train_fa: np.ndarray,
    y_train_d: np.ndarray,
    with_intercept: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Vanilla OLS in standardized bias space (d-space).
    Returns (beta_hat, train_mse_in_d_space).
      - If with_intercept=True: beta_hat length is (k+1), with intercept first.
      - Else: beta_hat length is (k,).
    """
    model = LinearRegression(fit_intercept=with_intercept)
    model.fit(X_train_fa, y_train_d)

    if with_intercept:
        beta_hat = np.hstack([model.intercept_, model.coef_])
        y_hat = X_train_fa @ model.coef_ + model.intercept_
    else:
        beta_hat = model.coef_
        y_hat = X_train_fa @ model.coef_

    mse_train = float(np.mean((y_train_d - y_hat) ** 2))
    return beta_hat, mse_train


def train_ols_on_bias(X_train_fa: np.ndarray, y_train_d: np.ndarray) -> Tuple[np.ndarray, float]:
    return train_linear_model(X_train_fa, y_train_d, with_intercept=True)


def train_lasso_on_bias(
    X_train_fa: np.ndarray,
    y_train_d: np.ndarray,
    alpha: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """
    sklearn Lasso in d-space (with intercept).
    Returns (beta_hat (k+1,), train_mse_in_d_space).
    """
    lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000, random_state=0)
    lasso.fit(X_train_fa, y_train_d)

    beta_hat = np.hstack([lasso.intercept_, lasso.coef_])
    y_hat = X_train_fa @ lasso.coef_ + lasso.intercept_
    mse_train = float(np.mean((y_train_d - y_hat) ** 2))
    return beta_hat, mse_train


def train_linear_with_penalty(
    X_train_fa: np.ndarray,
    y_train_d: np.ndarray,
    penalty_coef: float = 20.0,
    lr: float = 1e-2,
    epochs: int = 200,
    debug: bool = False,
    debug_prefix: str = "",
    return_history: bool = False,
) -> Tuple[np.ndarray, float, Optional[List[Dict]]]:
    """
    Linear model in d-space with intercept + sign-based directional penalty (PyTorch).
    Optimizes: MSE(d) + penalty_coef * mean( 1[y * y_hat < 0] )

    Returns:
      - beta_hat (k+1,)
      - train_mse_in_d_space (computed using final beta)
      - history (optional): per-epoch dicts with beta and losses
    """
    if torch is None:
        raise ImportError("PyTorch is required for 'train_linear_with_penalty', but torch is not installed.")

    X_np = np.asarray(X_train_fa, dtype=np.float32)
    y_np = np.asarray(y_train_d, dtype=np.float32)

    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)

    n_features = X_t.shape[1]
    beta = torch.zeros(n_features + 1, requires_grad=True)
    X_t_with_intercept = torch.cat([torch.ones(X_t.shape[0], 1), X_t], dim=1)

    opt = torch.optim.Adam([beta], lr=lr)

    history: Optional[List[Dict]] = [] if return_history else None

    for epoch in range(epochs):
        d_hat = X_t_with_intercept @ beta
        mse_loss = torch.mean((y_t - d_hat) ** 2)

        worse_mask = (y_t * d_hat < 0).float()
        dir_penalty = torch.mean(worse_mask)

        loss = mse_loss + penalty_coef * dir_penalty

        opt.zero_grad()
        loss.backward()
        opt.step()

        if debug:
            prefix = f"{debug_prefix} " if debug_prefix else ""
            print(
                f"[DEBUG] {prefix}epoch={epoch+1:03d}/{epochs} | "
                f"penalty={penalty_coef:.3g} | "
                f"mse_loss={mse_loss.item():.6f} | "
                f"dir_penalty={dir_penalty.item():.6f} | "
                f"total_loss={loss.item():.6f}"
            )

        if return_history and history is not None:
            history.append({
                "epoch": epoch + 1,
                "mse_loss": float(mse_loss.item()),
                "dir_penalty": float(dir_penalty.item()),
                "total_loss": float(loss.item()),
                "beta": beta.detach().cpu().numpy().copy(),
            })

    beta_hat = beta.detach().cpu().numpy()
    y_hat_final = X_np @ beta_hat[1:] + beta_hat[0]
    mse_train = float(np.mean((y_np - y_hat_final) ** 2))

    return beta_hat, mse_train, history


# ==========================
# Data prep
# ==========================

def prepare_target_df(master_df: pd.DataFrame, target: str = "d") -> pd.DataFrame:
    """
    Returns a copy of master_df where y_standardized_bias holds the target we want to predict:
      - target="d": use existing y_standardized_bias (Cohen's d)
      - target="norm": overwrite y_standardized_bias with Normalized_Bias
    """
    df = master_df.copy()
    if target == "d":
        return df
    if target == "norm":
        df["y_standardized_bias"] = df["Normalized_Bias"]
        return df
    raise ValueError(f"Unknown target type: {target}")


# ==========================
# Prediction-level metrics output
# ==========================

def compute_prediction_metrics_df(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    beta_hat: np.ndarray,
    fa_model: FactorAnalysis,
    seed_id: int,
    model_type: str,
) -> pd.DataFrame:
    """
    Builds a row-level dataframe containing y_true and y_hat in d-space for train & valid.
    Adds constant columns for MSE/RSS/TSS computed within each subset.
    """
    def _subset_df(df: pd.DataFrame, set_name: str) -> pd.DataFrame:
        X_orig = np.vstack(df["Embedding"].values)
        y_true = df["y_standardized_bias"].to_numpy()
        X_fa = fa_model.transform(X_orig)

        y_hat = predict_d(X_fa, beta_hat)

        errors = y_true - y_hat
        mse_debiased = float(np.mean(errors ** 2))
        rss = float(np.sum(errors ** 2))
        tss = float(np.sum((y_true - np.mean(y_true)) ** 2))

        mse_baseline = float(np.mean(y_true ** 2))

        return pd.DataFrame({
            "Variable_Name": df["Variable_Name"].values,
            "y_true_d": y_true,
            "y_hat_d": y_hat,
            "Average_LLM_Response": df["Average_LLM_Response"].values,
            "Average_Human_Response": df["Average_Human_Response"].values,
            "set": set_name,
            "MSE_d_debiased": mse_debiased,
            "MSE_d_baseline": mse_baseline,
            "RSS_d": rss,
            "TSS_d": tss,
            "seed": seed_id,
            "model_type": model_type,
        })

    out = pd.concat([_subset_df(train_df, "train"), _subset_df(valid_df, "valid")], ignore_index=True)
    return out


# ==========================
# Seed-hunting (baseline models; FA fit per split)
# ==========================

def run_seed_hunting_analysis(
    wave_name: str,
    model_type: str,
    df_processed: pd.DataFrame,
    best_params: Optional[Dict],
    base_results_dir: Optional[Path] = None,
    fixed_seed: Optional[int] = None,
    seed_list: Optional[List[int]] = None,
    save_best_seed_artifacts: bool = True,
) -> None:
    """
    Seed-hunting analysis for a given model type.

    Model types supported:
      - "linear_with_intercept"
      - "lasso_on_bias"
      - "linear_with_penalty"  (trains per split; generally NOT used in the fixed-beta design)

    Notes:
      - FactorAnalysis is fit per seed split in this function.
      - Improvement thresholding is computed in d-space (not raw-space).
    """
    if df_processed is None or len(df_processed) < 50:
        print(f"Halting: insufficient data for analysis '{wave_name}'.")
        return

    best_params = best_params or {}

    # Seeds
    if fixed_seed is not None:
        seeds = [fixed_seed]
    elif seed_list is not None:
        seeds = seed_list
    else:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, 100000, size=DEFAULT_N_SEEDS_TO_TRY).tolist()

    n_seeds = len(seeds)

    # thresholds
    valid_n_expected = int(len(df_processed) * VALIDATION_SIZE)
    direction_threshold = int(np.ceil(valid_n_expected * DIRECTION_THRESHOLD_FRAC))

    best_penalty = float(best_params.get("penalty_coef", 0.1))
    best_alpha = float(best_params.get("alpha", 0.01))

    # output folders
    if base_results_dir is None:
        base_results_dir = STEP2_OUT_DIR / f"{wave_name}_results"

    base_dir = Path(base_results_dir) / model_type
    diag_dir = base_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print(f"SEED-HUNTING ANALYSIS: {wave_name}")
    print(f"Model type: {model_type}")
    print(f"Data: N={len(df_processed)} | test_size={VALIDATION_SIZE:.2f} | k={FIXED_N_COMPONENTS}")
    print("Tuned params (if applicable):")
    print(f"  penalty_coef: {best_penalty if model_type == 'linear_with_penalty' else 'N/A'}")
    print(f"  alpha:        {best_alpha if model_type == 'lasso_on_bias' else 'N/A'}")
    print(f"Criteria over {n_seeds} seeds:")
    print(f"  1) Valid d-space MSE improvement > {IMPROVEMENT_THRESHOLD_PCT:.1f}%")
    print(f"  2) Direction improvement >= {direction_threshold}/{valid_n_expected} ({DIRECTION_THRESHOLD_FRAC*100:.1f}%)")
    print("#" * 70)

    all_results: List[Dict] = []
    good_results: List[Dict] = []
    all_pred_dfs: List[pd.DataFrame] = []

    for i, seed in enumerate(seeds):
        if (i + 1) % 10 == 0 or i == 0 or n_seeds < 20:
            print(f"  ...seed {i+1}/{n_seeds} (seed={seed})")

        try:
            train_df, valid_df = train_test_split(
                df_processed,
                test_size=VALIDATION_SIZE,
                random_state=seed,
                shuffle=True,
            )

            # FA fit per split
            X_train_orig = np.vstack(train_df["Embedding"].values)
            X_valid_orig = np.vstack(valid_df["Embedding"].values)

            fa_model = FactorAnalysis(n_components=FIXED_N_COMPONENTS, random_state=0)
            X_train_fa = fa_model.fit_transform(X_train_orig)
            X_valid_fa = fa_model.transform(X_valid_orig)

            y_train_d = train_df["y_standardized_bias"].to_numpy()
            y_valid_d = valid_df["y_standardized_bias"].to_numpy()

            # Train model
            if model_type == "linear_with_intercept":
                beta_hat, _ = train_ols_on_bias(X_train_fa, y_train_d)

            elif model_type == "lasso_on_bias":
                beta_hat, _ = train_lasso_on_bias(X_train_fa, y_train_d, alpha=best_alpha)

            elif model_type == "linear_with_penalty":
                beta_hat, _, hist = train_linear_with_penalty(
                    X_train_fa,
                    y_train_d,
                    penalty_coef=best_penalty,
                    return_history=True,
                )
                assert hist is not None
                best_epoch_rec = min(hist, key=lambda d: d["total_loss"])
                beta_hat = best_epoch_rec["beta"]

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # Predictions in d-space
            pred_train_d = predict_d(X_train_fa, beta_hat)
            pred_valid_d = predict_d(X_valid_fa, beta_hat)

            # R^2 in d-space
            r2_train = float(r2_score(y_train_d, pred_train_d))
            r2_valid = float(r2_score(y_valid_d, pred_valid_d))

            # d-space MSE + improvement
            valid_mse_d_base, valid_mse_d_deb = mse_d(y_valid_d, pred_valid_d)
            train_mse_d_base, train_mse_d_deb = mse_d(y_train_d, pred_train_d)
            imp_pct = improvement_pct(valid_mse_d_base, valid_mse_d_deb)

            # raw-space metrics
            v_llm = valid_df["Average_LLM_Response"].to_numpy()
            v_human = valid_df["Average_Human_Response"].to_numpy()
            v_sp = valid_df["s_pooled"].to_numpy()

            t_llm = train_df["Average_LLM_Response"].to_numpy()
            t_human = train_df["Average_Human_Response"].to_numpy()
            t_sp = train_df["s_pooled"].to_numpy()

            v_mse_raw_base, v_mse_raw_deb, v_debiased = raw_mse_and_debiased(v_llm, v_human, v_sp, pred_valid_d)
            t_mse_raw_base, t_mse_raw_deb, _ = raw_mse_and_debiased(t_llm, t_human, t_sp, pred_train_d)

            # direction
            correct_direction = count_direction_improved(v_llm, v_human, v_debiased)

            # Log per seed (ALL seeds)
            all_results.append({
                "seed": seed,
                "correct_direction": correct_direction,
                "valid_N": len(valid_df),
                "improvement_pct_d": imp_pct,

                # RAW MSEs
                "V_MSE_raw_Base": v_mse_raw_base,
                "V_MSE_raw_Deb":  v_mse_raw_deb,
                "T_MSE_raw_Base": t_mse_raw_base,
                "T_MSE_raw_Deb":  t_mse_raw_deb,

                # d-space MSEs
                "V_MSE_d_Base":   valid_mse_d_base,
                "V_MSE_d_Deb":    valid_mse_d_deb,
                "T_MSE_d_Base":   train_mse_d_base,
                "T_MSE_d_Deb":    train_mse_d_deb,

                # R^2
                "R2_Valid": r2_valid,
                "R2_Train": r2_train,

                # hyperparams (if relevant)
                "Pen_Coef": best_penalty if model_type == "linear_with_penalty" else np.nan,
                "Alpha": best_alpha if model_type == "lasso_on_bias" else np.nan,
            })

            # Save prediction-level df for this seed (all seeds, not just good ones)
            pred_df = compute_prediction_metrics_df(
                train_df=train_df,
                valid_df=valid_df,
                beta_hat=beta_hat,
                fa_model=fa_model,
                seed_id=seed,
                model_type=model_type,
            )
            all_pred_dfs.append(pred_df)

            # Check "good seed" criteria
            if (imp_pct > IMPROVEMENT_THRESHOLD_PCT) and (correct_direction >= direction_threshold):
                good_results.append({
                    "seed": seed,
                    "improvement_pct_d": imp_pct,
                    "correct_direction": correct_direction,
                    "r2_valid": r2_valid,
                    "r2_train": r2_train,
                    "best_penalty_coef": best_penalty if model_type == "linear_with_penalty" else np.nan,
                    "best_alpha": best_alpha if model_type == "lasso_on_bias" else np.nan,
                })
                print(f"    ✓ good seed {seed}: imp={imp_pct:.1f}%, dir={correct_direction}/{valid_n_expected}, R2_valid={r2_valid:.3f}")

        except Exception as e:
            print(f"    WARNING: seed {seed} failed with error: {e}")
            continue

    # Save ALL seeds summary CSV + print distribution summary
    all_summary_csv = base_dir / f"{wave_name}_{model_type}_ALL_SEEDS_SUMMARY.csv"
    save_all_seeds_summary(
        all_results=all_results,
        out_csv_path=all_summary_csv,
        header_label=f"Directional improvement distribution over ALL seeds | model={model_type}",
    )

    # Save combined prediction-level outputs across all seeds
    if all_pred_dfs:
        combined_pred = pd.concat(all_pred_dfs, ignore_index=True)
        pred_out = base_dir / f"{wave_name}_{model_type}_all_seeds_predictions.csv"
        combined_pred.to_csv(pred_out, index=False)
        print(f"Saved prediction-level outputs (ALL seeds) -> {pred_out}")
    else:
        print("WARNING: No prediction-level outputs were created (no successful seeds?).")

    # Report best seeds
    print("\n" + "=" * 70)
    print(f"MODEL SUMMARY: {model_type.upper()}")
    print(f"ANALYSIS: {wave_name}")
    print(f"Total seeds tested: {n_seeds}")
    print(f"Good seeds found:   {len(good_results)} ({(len(good_results)/n_seeds)*100:.1f}%)")
    print(f"Outputs saved under: {base_dir}")
    print("=" * 70)

    if not good_results:
        print("No seeds found meeting all criteria.")
        return

    good_results.sort(key=lambda x: x["improvement_pct_d"], reverse=True)
    best = good_results[0]
    print(f"Best seed: {best['seed']} with {best['improvement_pct_d']:.1f}% d-space improvement.")

    if not save_best_seed_artifacts:
        return

    # Save raw train/valid and factors for best seed (optional)
    try:
        best_seed = int(best["seed"])
        best_train_df, best_valid_df = train_test_split(
            df_processed,
            test_size=VALIDATION_SIZE,
            random_state=best_seed,
            shuffle=True,
        )

        train_csv = base_dir / f"{wave_name}_best_seed_{best_seed}_{model_type}_RAW_TRAIN_DATA.csv"
        valid_csv = base_dir / f"{wave_name}_best_seed_{best_seed}_{model_type}_RAW_VALID_DATA.csv"
        best_train_df.to_csv(train_csv, index=False)
        best_valid_df.to_csv(valid_csv, index=False)
        print(f"Saved raw training data -> {train_csv}")
        print(f"Saved raw validation data -> {valid_csv}")

        fa_best = FactorAnalysis(n_components=FIXED_N_COMPONENTS, random_state=0)
        X_train_best = np.vstack(best_train_df["Embedding"].values)
        X_valid_best = np.vstack(best_valid_df["Embedding"].values)

        X_train_best_fa = fa_best.fit_transform(X_train_best)
        X_valid_best_fa = fa_best.transform(X_valid_best)

        def build_fa_df(orig_df: pd.DataFrame, X_fa: np.ndarray) -> pd.DataFrame:
            fa_df = pd.DataFrame({f"x{i+1}": X_fa[:, i] for i in range(X_fa.shape[1])})
            fa_df["Variable_Name"] = orig_df["Variable_Name"].values
            fa_df["y_standardized_bias"] = orig_df["y_standardized_bias"].values
            return fa_df

        train_fa_df = build_fa_df(best_train_df.reset_index(drop=True), X_train_best_fa)
        valid_fa_df = build_fa_df(best_valid_df.reset_index(drop=True), X_valid_best_fa)

        train_fa_csv = base_dir / f"{wave_name}_best_seed_{best_seed}_{model_type}_FACTORS_TRAIN.csv"
        valid_fa_csv = base_dir / f"{wave_name}_best_seed_{best_seed}_{model_type}_FACTORS_VALID.csv"
        train_fa_df.to_csv(train_fa_csv, index=False)
        valid_fa_df.to_csv(valid_fa_csv, index=False)
        print(f"Saved factor training data -> {train_fa_csv}")
        print(f"Saved factor validation data -> {valid_fa_csv}")

    except Exception as e:
        print(f"WARNING: Could not save best-seed artifacts. Error: {e}")


# ==========================
# Seed-hunting (fixed beta; shared FA)
# ==========================

def run_seed_hunting_fixed_beta(
    wave_name: str,
    df_processed: pd.DataFrame,
    beta_hat_fixed: np.ndarray,
    fa_model: FactorAnalysis,
    base_results_dir: Path,
    seed_list: List[int],
    model_label: str = "linear_with_penalty_fixed",
) -> None:
    """
    Seed-hunting evaluation for a *fixed* beta_hat in d-space.

    - Uses ONE FactorAnalysis model (fa_model) shared across all seeds.
    - Does NOT train inside; just evaluates the fixed beta_hat.
    - Computes raw-space and d-space metrics, R^2, and direction counts per seed.
    """
    if df_processed is None or len(df_processed) < 50:
        print(f"Halting: insufficient data for analysis '{wave_name}'.")
        return

    seeds = seed_list
    n_seeds = len(seeds)

    valid_n_expected = int(len(df_processed) * VALIDATION_SIZE)
    direction_threshold = int(np.ceil(valid_n_expected * DIRECTION_THRESHOLD_FRAC))

    base_dir = Path(base_results_dir) / model_label
    diag_dir = base_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print(f"FIXED-BETA SEED-HUNTING: {wave_name}")
    print(f"Model label: {model_label}")
    print(f"Data: N={len(df_processed)} | test_size={VALIDATION_SIZE:.2f} | k={FIXED_N_COMPONENTS}")
    print(f"Criteria over {n_seeds} seeds:")
    print(f"  1) Valid d-space MSE improvement > {IMPROVEMENT_THRESHOLD_PCT:.1f}%")
    print(f"  2) Direction improvement >= {direction_threshold}/{valid_n_expected} ({DIRECTION_THRESHOLD_FRAC*100:.1f}%)")
    print("#" * 70)

    all_results: List[Dict] = []
    all_pred_dfs: List[pd.DataFrame] = []
    good_results: List[Dict] = []

    for i, seed in enumerate(seeds):
        if (i + 1) % 10 == 0 or i == 0 or n_seeds < 20:
            print(f"  ...seed {i+1}/{n_seeds} (seed={seed})")

        try:
            train_df, valid_df = train_test_split(
                df_processed,
                test_size=VALIDATION_SIZE,
                random_state=seed,
                shuffle=True,
            )

            X_train_orig = np.vstack(train_df["Embedding"].values)
            X_valid_orig = np.vstack(valid_df["Embedding"].values)

            # shared FA transform
            X_train_fa = fa_model.transform(X_train_orig)
            X_valid_fa = fa_model.transform(X_valid_orig)

            y_train_d = train_df["y_standardized_bias"].to_numpy()
            y_valid_d = valid_df["y_standardized_bias"].to_numpy()

            pred_train_d = predict_d(X_train_fa, beta_hat_fixed)
            pred_valid_d = predict_d(X_valid_fa, beta_hat_fixed)

            r2_train = float(r2_score(y_train_d, pred_train_d))
            r2_valid = float(r2_score(y_valid_d, pred_valid_d))

            valid_mse_d_base, valid_mse_d_deb = mse_d(y_valid_d, pred_valid_d)
            train_mse_d_base, train_mse_d_deb = mse_d(y_train_d, pred_train_d)
            imp_pct = improvement_pct(valid_mse_d_base, valid_mse_d_deb)

            v_llm = valid_df["Average_LLM_Response"].to_numpy()
            v_human = valid_df["Average_Human_Response"].to_numpy()
            v_sp = valid_df["s_pooled"].to_numpy()

            t_llm = train_df["Average_LLM_Response"].to_numpy()
            t_human = train_df["Average_Human_Response"].to_numpy()
            t_sp = train_df["s_pooled"].to_numpy()

            v_mse_raw_base, v_mse_raw_deb, v_debiased = raw_mse_and_debiased(v_llm, v_human, v_sp, pred_valid_d)
            t_mse_raw_base, t_mse_raw_deb, _ = raw_mse_and_debiased(t_llm, t_human, t_sp, pred_train_d)

            correct_direction = count_direction_improved(v_llm, v_human, v_debiased)

            all_results.append({
                "seed": seed,
                "correct_direction": correct_direction,
                "valid_N": len(valid_df),
                "improvement_pct_d": imp_pct,

                "V_MSE_raw_Base": v_mse_raw_base,
                "V_MSE_raw_Deb":  v_mse_raw_deb,
                "T_MSE_raw_Base": t_mse_raw_base,
                "T_MSE_raw_Deb":  t_mse_raw_deb,

                "V_MSE_d_Base":   valid_mse_d_base,
                "V_MSE_d_Deb":    valid_mse_d_deb,
                "T_MSE_d_Base":   train_mse_d_base,
                "T_MSE_d_Deb":    train_mse_d_deb,

                "R2_Valid": r2_valid,
                "R2_Train": r2_train,
            })

            # prediction-level df
            pred_df = compute_prediction_metrics_df(
                train_df=train_df,
                valid_df=valid_df,
                beta_hat=beta_hat_fixed,
                fa_model=fa_model,
                seed_id=seed,
                model_type=model_label,
            )
            all_pred_dfs.append(pred_df)

            if (imp_pct > IMPROVEMENT_THRESHOLD_PCT) and (correct_direction >= direction_threshold):
                good_results.append({
                    "seed": seed,
                    "improvement_pct_d": imp_pct,
                    "correct_direction": correct_direction,
                    "r2_valid": r2_valid,
                    "r2_train": r2_train,
                })
                print(f"    ✓ good seed {seed}: imp={imp_pct:.1f}%, dir={correct_direction}/{valid_n_expected}, R2_valid={r2_valid:.3f}")

        except Exception as e:
            print(f"    WARNING: seed {seed} failed with error: {e}")
            continue

    # Save ALL seeds summary + distribution
    all_summary_csv = base_dir / f"{wave_name}_{model_label}_ALL_SEEDS_SUMMARY.csv"
    save_all_seeds_summary(
        all_results=all_results,
        out_csv_path=all_summary_csv,
        header_label=f"Directional improvement distribution over ALL seeds (FIXED BETA) | model={model_label}",
    )

    # Save prediction-level outputs (ALL seeds)
    if all_pred_dfs:
        combined_pred = pd.concat(all_pred_dfs, ignore_index=True)
        pred_out = base_dir / f"{wave_name}_{model_label}_all_seeds_predictions.csv"
        combined_pred.to_csv(pred_out, index=False)
        print(f"Saved prediction-level outputs (ALL seeds) -> {pred_out}")
    else:
        print("WARNING: No prediction-level outputs were created (no successful seeds?).")

    print("\n" + "=" * 70)
    print(f"FIXED-BETA MODEL SUMMARY: {model_label.upper()}")
    print(f"ANALYSIS: {wave_name}")
    print(f"Total seeds tested: {n_seeds}")
    print(f"Good seeds found:   {len(good_results)} ({(len(good_results)/n_seeds)*100:.1f}%)")
    print(f"Outputs saved under: {base_dir}")
    print("=" * 70)


# ==========================
# Main
# ==========================

if __name__ == "__main__":
    STEP2_OUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    if not MASTER_PKL.exists():
        raise FileNotFoundError(f"{MASTER_PKL} not found. Run step1_twin_build_master.py first to create it.")

    master_df = pd.read_pickle(MASTER_PKL)
    analysis_name = "waves_1_to_4_combined"

    print("\n" + "=" * 70)
    print(f"Loaded master DF: {MASTER_PKL}")
    print(f"Total variables: N={len(master_df)}")
    print(f"Analysis name: {analysis_name}")
    print("=" * 70)

    # Reproducible seed list shared across everything
    GLOBAL_N_SEEDS = 200
    rng = np.random.default_rng(0)
    global_seeds = rng.integers(0, 100000, size=GLOBAL_N_SEEDS).tolist()

    # Optional: make torch training reproducible too (if torch exists)
    if torch is not None:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    PENALTIES = [0.001, 0.1, 1, 10, 20]

    MODELS_NO_PENALTY = [
        "linear_with_intercept",  # OLS
        "lasso_on_bias",          # Lasso
    ]

    # Run twice: normalized bias and Cohen's d
    for target in ["norm", "d"]:
        print(f"\n\n========== RUNNING TARGET = {target} ==========")
        df_target = prepare_target_df(master_df, target=target)

        base_root = STEP2_OUT_DIR / f"{analysis_name}_results_same_seeds_{target}"
        base_root.mkdir(parents=True, exist_ok=True)

        # 1) Baseline models (FA fit per seed split)
        base_no_pen = base_root / "no_penalty"
        base_no_pen.mkdir(parents=True, exist_ok=True)

        for model in MODELS_NO_PENALTY:
            print(f"\n---- Running baseline model '{model}' for target={target} ----")
            run_seed_hunting_analysis(
                wave_name=analysis_name,
                model_type=model,
                df_processed=df_target,
                best_params={},
                seed_list=global_seeds,
                base_results_dir=base_no_pen,
            )

        # 2) Fixed-beta directional penalty models
        best_epochs: Dict[float, int] = {}

        X_all_orig = np.vstack(df_target["Embedding"].values)
        y_all_d = df_target["y_standardized_bias"].to_numpy()

        # Shared FA fit on ALL data for this target
        fa_fixed = FactorAnalysis(n_components=FIXED_N_COMPONENTS, random_state=0)
        X_all_fa = fa_fixed.fit_transform(X_all_orig)

        for pen in PENALTIES:
            print(f"\n---- Building FIXED-BETA penalty model for target={target}, pen={pen:g} ----")

            base_pen_fixed = base_root / f"pen_{pen:g}_fixed_beta"
            base_pen_fixed.mkdir(parents=True, exist_ok=True)

            beta_hat_full, _, hist_full = train_linear_with_penalty(
                X_train_fa=X_all_fa,
                y_train_d=y_all_d,
                penalty_coef=pen,
                return_history=True,
            )
            assert hist_full is not None and len(hist_full) >= 2

            # choose best epoch by total_loss, skipping first epoch
            hist_skip_first = hist_full[1:]
            best_epoch_rec = min(hist_skip_first, key=lambda d: d["total_loss"])
            best_epoch = int(best_epoch_rec["epoch"])
            beta_hat_fixed = best_epoch_rec["beta"]
            best_epochs[pen] = best_epoch

            print(
                f"Selected epoch {best_epoch} for pen={pen:g} "
                f"(total_loss={best_epoch_rec['total_loss']:.6f})"
            )

            # sanity check MSE(d) on all data with chosen beta
            y_hat_all = predict_d(X_all_fa, beta_hat_fixed)
            mse_train_fixed = float(np.mean((y_all_d - y_hat_all) ** 2))
            print(f"Train MSE_d (best-epoch beta) = {mse_train_fixed:.6f}")

            # save beta
            beta_path = base_pen_fixed / f"{analysis_name}_target_{target}_pen_{pen:g}_fixed_beta.npy"
            np.save(beta_path, beta_hat_fixed)
            print(f"Saved beta -> {beta_path}")

            # evaluate fixed beta across seeds using shared FA
            model_label = f"linear_with_penalty_fixed_pen_{pen:g}"
            run_seed_hunting_fixed_beta(
                wave_name=analysis_name,
                df_processed=df_target,
                beta_hat_fixed=beta_hat_fixed,
                fa_model=fa_fixed,
                base_results_dir=base_pen_fixed,
                seed_list=global_seeds,
                model_label=model_label,
            )

        # Save summary of best epochs
        print(f"\n[SUMMARY] Best epochs per penalty (target={target})")
        for pen, ep in best_epochs.items():
            print(f"  pen={pen:g} -> best_epoch={ep}")

        epochs_csv = base_root / f"{analysis_name}_best_epochs_target_{target}.csv"
        pd.DataFrame([{"penalty": p, "best_epoch": e} for p, e in best_epochs.items()]).to_csv(epochs_csv, index=False)
        print(f"Saved best-epochs summary -> {epochs_csv}")

    end = time.time()
    print("\n" + "=" * 70)
    print("ALL MODELS COMPLETE.")
    print(f"Total time: {end - start:.1f} seconds.")
    print("=" * 70)
