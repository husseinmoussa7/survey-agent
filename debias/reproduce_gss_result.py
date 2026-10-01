#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproduce the reported GSS debiasing result.

Published figures this script checks:
    baseline (uncorrected) held-out MSE  0.639
    debiased held-out MSE                0.444   (30.5% reduction)
    correct bias direction               9 of 11 items
    training MSE                         0.903

Why this script exists
----------------------
factor-based-debias.py selects its holdout with
`train_test_split(df, train_size=100, random_state=8566)`. Over the 112-row anchor set that call
returns a DIFFERENT 12-item holdout whose uncorrected baseline MSE is 0.173, so it cannot reproduce
the published numbers, and no seed does: a search over 100,000 seed/split-size combinations found
none that selects the reported 11 items. The reported holdout is recorded only in the manuscript's
appendix table, so this script pins it explicitly by item identity instead of by seed.

It also computes every published figure. factor-based-debias.py prints only the debiased MSE, so the
baseline, the percentage improvement, and the directional accuracy were not outputs of any shipped
code.

Usage
-----
    python debias/reproduce_gss_result.py

The baseline MSE and the holdout check need only numpy, pandas and scikit-learn. Fitting the
correction additionally requires torch; without it the script reports what it can and stops.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
PICKLE = os.path.join(HERE, "survey_with_embeddings.pkl")

# Published hyperparameters.
N_COMPONENTS = 50
PENALTY_COEF = 20.0
LR = 1e-2
EPOCHS = 200

# Published results, for the comparison printed at the end.
PUBLISHED = {"baseline_mse": 0.639, "debiased_mse": 0.444,
             "improvement_pct": 30.5, "direction": "9/11", "train_mse": 0.903}

# Two rows in the shipped anchor set repeat a question with IDENTICAL text (`aged`,
# `libhomo`), i.e. a second LLM draw of the same stimulus. Those are collapsed, which is
# what makes the result reproducible. Five further pairs share a GSS variable name but ask
# it with DIFFERENT wording (`abnomore`, `conbus`, `fehire`, `helpblk`, `natsci`); those are
# distinct stimuli and are deliberately kept, since collapsing them would discard a
# prompt-wording comparison and degrades the correction (see README.md).
COLLAPSE_IDENTICAL_TEXT_DUPLICATES = True

# The reported holdout, read off appendix table tab:full_results. Each item is identified by
# (Variable_Name, Average_LLM_Response) because five GSS variables are asked twice with different
# wording, sharing a human average but not an LLM response, so the name alone is not unique.
HOLDOUT = [
    ("letinasn", 3.000), ("natsci",   2.000), ("discaffm", 3.333), ("natsci",  1.333),
    ("rotapple", 1.167), ("libhomo",  2.000), ("helppoor", 1.000), ("abnomore", 1.167),
    ("protest3", 1.000), ("discaff",  3.000), ("rdiscaff", 2.667),
]


def load_anchor_set() -> pd.DataFrame:
    df = pd.read_pickle(PICKLE).reset_index(drop=True)
    if COLLAPSE_IDENTICAL_TEXT_DUPLICATES:
        before = len(df)
        df = df[~df.duplicated(subset=["Variable_Name", "Question"], keep="first")]
        df = df.reset_index(drop=True)
        print(f"collapsed {before - len(df)} repeat draw(s) of an identical question text")
    return df


def resolve_holdout(df: pd.DataFrame) -> list[int]:
    """Map each reported holdout item to exactly one row, without reusing a row."""
    idx: list[int] = []
    for name, llm in HOLDOUT:
        match = df[(df["Variable_Name"] == name)
                   & (df["Average_LLM_Response"].round(3) == round(llm, 3))
                   & (~df.index.isin(idx))]
        if match.empty:
            raise RuntimeError(f"holdout item not found in anchor set: {name} (LLM={llm})")
        idx.append(int(match.index[0]))
    return idx


def fit_beta_factor_penalty(train_df: pd.DataFrame):
    """Factor model plus directional-penalty regression. Identical objective to the paper."""
    if torch is None:
        raise ImportError(
            "PyTorch is required to fit the correction. Install it with `pip install torch` "
            "(see README.md for platform notes: no wheels exist for Intel macOS past 2.2.2, "
            "and none for Python 3.13)."
        )
    X_orig = np.vstack(train_df["Embedding"].values)
    llm = train_df["Average_LLM_Response"].to_numpy()
    human = train_df["Average_Human_Response"].to_numpy()
    y = llm - human

    fa = FactorAnalysis(n_components=N_COMPONENTS, random_state=0)
    X = fa.fit_transform(X_orig)

    X_t, y_t = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    llm_t, human_t = torch.from_numpy(llm).float(), torch.from_numpy(human).float()

    beta = torch.zeros(N_COMPONENTS, requires_grad=True)
    opt = torch.optim.Adam([beta], lr=LR)
    for _ in range(EPOCHS):
        pred = X_t @ beta
        debiased = llm_t - pred
        mse_loss = torch.mean((debiased - human_t) ** 2)
        mask = (y_t * pred < 0).float()
        penalty = torch.mean(mask * torch.abs(y_t - pred))
        loss = mse_loss + PENALTY_COEF * penalty
        opt.zero_grad()
        loss.backward()
        opt.step()

    beta_hat = beta.detach().numpy()
    train_mse = float(np.mean((llm - X.dot(beta_hat) - human) ** 2))
    return beta_hat, train_mse, fa


def main() -> None:
    df = load_anchor_set()
    holdout_idx = resolve_holdout(df)
    valid_df = df.loc[holdout_idx]
    train_df = df.drop(index=holdout_idx)

    print("=" * 72)
    print("GSS debiasing result: reproduction")
    print("=" * 72)
    print(f"anchor set rows            : {len(df)}")
    print(f"unique GSS variables       : {df['Variable_Name'].nunique()}"
          f"  (5 are asked twice with different wording, kept as distinct stimuli)")
    print(f"holdout items (pinned)     : {len(valid_df)}")
    print(f"training items             : {len(train_df)}")
    if len(train_df) != 100:
        print(f"  NOTE: the paper reports 100 training items against {len(train_df)} here. The "
              f"published\n        run therefore used one row more than this set; which one is not "
              f"recorded.\n        See README.md.")

    llm = valid_df["Average_LLM_Response"].to_numpy()
    human = valid_df["Average_Human_Response"].to_numpy()
    baseline_mse = float(np.mean((llm - human) ** 2))

    print()
    print("-" * 72)
    print(f"{'quantity':34s} {'reproduced':>12s} {'published':>11s}")
    print("-" * 72)
    print(f"{'baseline (uncorrected) MSE':34s} {baseline_mse:12.3f} {PUBLISHED['baseline_mse']:11.3f}")

    if torch is None:
        print("-" * 72)
        print("\nThe baseline above is the error of the RAW LLM averages against the human")
        print("averages on the holdout. It involves no model, which is why it reproduces")
        print("without torch. Fitting the correction needs torch; install it and rerun to")
        print("check the debiased MSE, the improvement, and the directional accuracy.")
        return

    beta_hat, train_mse, fa = fit_beta_factor_penalty(train_df)
    X_valid = fa.transform(np.vstack(valid_df["Embedding"].values))
    debiased = llm - X_valid.dot(beta_hat)
    debiased_mse = float(np.mean((debiased - human) ** 2))
    improvement = 100.0 * (baseline_mse - debiased_mse) / baseline_mse
    correct_dir = int(np.sum(np.abs(debiased - human) < np.abs(llm - human)))

    print(f"{'debiased MSE':34s} {debiased_mse:12.3f} {PUBLISHED['debiased_mse']:11.3f}")
    print(f"{'improvement (%)':34s} {improvement:12.1f} {PUBLISHED['improvement_pct']:11.1f}")
    print(f"{'moved toward human mean':34s} {str(correct_dir) + '/' + str(len(valid_df)):>12s} "
          f"{PUBLISHED['direction']:>11s}")
    print(f"{'training MSE':34s} {train_mse:12.3f} {PUBLISHED['train_mse']:11.3f}")
    print("-" * 72)

    print("\nPer-item detail (compare against appendix table tab:full_results):")
    detail = pd.DataFrame({
        "variable": valid_df["Variable_Name"].values,
        "human": human.round(3),
        "llm": llm.round(3),
        "debiased": debiased.round(3),
    })
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
