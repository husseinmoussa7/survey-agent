#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STEP 1: Build master dataframe for debiasing pipeline.

- Generates embeddings per wave (if missing).
- Preprocesses each wave: merges human + LLM + codes + embeddings.
- Computes s_pooled and y_standardized_bias (Cohen's d).
- Combines all waves into one master_df, saves to disk.

Run this only when raw input changes.
"""

import os
import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import openai

tqdm.pandas()

# === CONFIGURATION SECTION ===
# Project root = the panel/ directory two levels up from this file, matching
# the convention already used by step2_twin_debias_models.py and step3_make_averages.py.
BASE_DIR = str(Path(__file__).resolve().parents[2])

DATA_DIR = os.path.join(BASE_DIR, "data_raw", "waves")
LLM_DIR = os.path.join(BASE_DIR, "data_llm")
QUESTIONS_DIR = os.path.join(BASE_DIR, "data_questions")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Where to save the combined dataframe
MASTER_PKL = os.path.join(OUTPUT_DIR, "combined_waves_1_to_4_master.pkl")
MASTER_CSV = os.path.join(OUTPUT_DIR, "combined_waves_1_to_4_master.csv")

# ==========================
# Embedding helper
# ==========================

def get_embedding(text, model="text-embedding-3-small"):
    """
    Fetches an embedding for a given text string using OpenAI.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or export it.")

    try:
        clean_text = str(text).replace("\n", " ").strip()
        if not clean_text:
            print("  Warning: Empty question string found, returning None.")
            return None

        response = openai.embeddings.create(
            input=[clean_text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"  Error getting embedding for text: {str(text)[:50]}... \n  Error: {e}")
        return None


def generate_embeddings_if_needed():
    """
    Loop through waves and generate .pkl files with embeddings once.
    Output per wave: final_data_with_embeddings_wave_{wave}.pkl
    """
    print("\n" + "#" * 70)
    print("🚀 STARTING EMBEDDING GENERATION (if needed)")
    print("#" * 70)

    waves_to_embed = [1, 2, 3, 4]

    for wave_number in waves_to_embed:
        print(f"\n--- Processing Wave {wave_number} ---")

        # v7 logic for the kept-variables file
        if wave_number == 1:
            input_csv = os.path.join(QUESTIONS_DIR, "final_kept_variables_wave_1.csv")
        else:
            input_csv = os.path.join(
            QUESTIONS_DIR,
            f"final_kept_variables_wave_{wave_number}_REENGINEERED_v7_FINAL.csv"
        )

        output_pkl = os.path.join(
    OUTPUT_DIR,
    f"final_data_with_embeddings_wave_{wave_number}.pkl"
)

        if os.path.exists(output_pkl):
            print(f"✅ Wave {wave_number}: '{output_pkl}' already exists. Skipping.")
            continue

        try:
            print(f"  Loading '{input_csv}' for Wave {wave_number}...")
            df_wave = pd.read_csv(input_csv)

            if "Question" not in df_wave.columns:
                print(f"  ❌ ERROR: Wave {wave_number} - No 'Question' column in '{input_csv}'.")
                continue
            if "Variable_Name" not in df_wave.columns:
                print(f"  ❌ ERROR: Wave {wave_number} - No 'Variable_Name' column in '{input_csv}'.")
                continue

            print(f"  Generating embeddings for {len(df_wave)} questions in Wave {wave_number}...")
            df_wave["Embedding"] = df_wave["Question"].progress_apply(get_embedding)

            initial_count = len(df_wave)
            df_wave = df_wave.dropna(subset=["Embedding"])
            final_count = len(df_wave)

            if initial_count > final_count:
                print(f"  Dropped {initial_count - final_count} rows due to embedding failures.")

            df_wave_minimal = df_wave[["Variable_Name", "Question", "Embedding"]]
            df_wave_minimal.to_pickle(output_pkl)
            print(f"✅ Wave {wave_number}: Saved {final_count} embeddings to '{output_pkl}'.")

        except FileNotFoundError:
            print(f"  ❌ ERROR: Input file '{input_csv}' not found.")
            raise
        except Exception as e:
            print(f"  ❌ ERROR: Wave {wave_number} - Unexpected error: {e}")
            raise

    print("\n" + "#" * 70)
    print("🎉 EMBEDDING GENERATION COMPLETE")
    print("#" * 70)


# ==========================
# Wave pre-processing
# ==========================

def preprocess_wave_data(wave_number):
    """
    Loads all raw data for a wave, calculates stats, merges,
    FILTERS OUT problematic variables, and computes Cohen's d.
    Returns a single, clean DataFrame for analysis.
    """
    print(f"\n--- Pre-processing Wave {wave_number} ---")

    # List of variables to remove
    variables_to_remove = [
        "Q214", "Q211", "Q213", "Q210", "Q212", "Q209", "Q215",
        "Q225", "Q226", "Q227", "Q228", "Ut", "Q192", "Q193"
    ]
    if wave_number == 2:
        pass

    # 1. Human data
    try:
        human_file = os.path.join(DATA_DIR, f"wave_{wave_number}_numbers_anonymized.csv")
        print(f"  Loading raw human data: {human_file}")
        df_raw_human = pd.read_csv(human_file)

        if "TWIN_ID" not in df_raw_human.columns:
            print(f"  ❌ ERROR: 'TWIN_ID' not found in {human_file}.")
            return None

        n_Human = df_raw_human["TWIN_ID"].nunique()
        id_vars = ["TWIN_ID"]
        variable_cols = [c for c in df_raw_human.columns if c not in id_vars]

        print(f"  Coercing {len(variable_cols)} human variables to numeric...")
        df_raw_human[variable_cols] = df_raw_human[variable_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        print("  ✅ Coercion complete.")

        df_human_long = df_raw_human.melt(
            id_vars=id_vars,
            value_vars=variable_cols,
            var_name="Variable_Name",
            value_name="Response",
        )

        human_stats = df_human_long.groupby("Variable_Name")["Response"].agg(
            Average_Human_Response="mean",
            StdDev_Human_Response="std",
        ).reset_index()
        human_stats["n_Human"] = n_Human
        print(f"  ✅ Calculated human stats (n_Human={n_Human}) for {len(human_stats)} vars.")
    except FileNotFoundError:
        print(f"  ❌ ERROR: Raw human data file not found: {human_file}")
        return None
    except Exception as e:
        print(f"  ❌ ERROR processing human data: {e}")
        return None

    # 2. LLM data
    try:
        llm_file = ""
        llm_response_col = ""

        if wave_number == 1:
            llm_file_original_path = os.path.join(LLM_DIR, "llm_pooled_responses_wave_1_original_method.csv")
            llm_file_combined_path = os.path.join(LLM_DIR, "llm_pooled_responses_wave_1_combined_method.csv")

            if os.path.exists(llm_file_original_path):
                llm_file = llm_file_original_path
                llm_response_col = "LLM_Response_Number"
            elif os.path.exists(llm_file_combined_path):
                llm_file = llm_file_combined_path
                llm_response_col = "LLM_Response"
        else:
            llm_file_v7_path = os.path.join(LLM_DIR, f"llm_pooled_responses_wave_{wave_number}_v7_FINAL.csv")
            if os.path.exists(llm_file_v7_path):
                llm_file = llm_file_v7_path
                llm_response_col = "LLM_Response_Number"

        if not llm_file:
            print(f"  ❌ ERROR: No LLM response file found for Wave {wave_number} in LLM_DIR.")
            return None

        print(f"  Loading raw LLM data: {llm_file}")
        df_raw_llm = pd.read_csv(llm_file)

        if llm_response_col not in df_raw_llm.columns:
            print(f"  ⚠️ Warning: Column '{llm_response_col}' not found. Searching for fallbacks...")
            if "LLM_Response_Number" in df_raw_llm.columns:
                llm_response_col = "LLM_Response_Number"
            elif "LLM_Response" in df_raw_llm.columns:
                llm_response_col = "LLM_Response"
            elif "Response" in df_raw_llm.columns:
                llm_response_col = "Response"
            else:
                print(f"  ❌ ERROR: LLM file {llm_file} missing response column.")
                return None
            print(f"  Found fallback response column: '{llm_response_col}'")

        required_llm_cols = ["ParticipantID", "Variable_Name", llm_response_col]
        if not all(col in df_raw_llm.columns for col in required_llm_cols):
            print(f"  ❌ ERROR: LLM file {llm_file} missing required columns.")
            print(f"         Expected: {required_llm_cols}")
            print(f"         Found: {df_raw_llm.columns.to_list()}")
            return None

        df_raw_llm[llm_response_col] = pd.to_numeric(df_raw_llm[llm_response_col], errors="coerce")

        llm_stats_n = df_raw_llm.groupby("Variable_Name")["ParticipantID"].nunique().reset_index(name="n_LLM")
        llm_stats_vals = df_raw_llm.groupby("Variable_Name")[llm_response_col].agg(
            Average_LLM_Response="mean",
            StdDev_LLM_Response="std",
        ).reset_index()
        llm_stats = llm_stats_n.merge(llm_stats_vals, on="Variable_Name")
        print(f"  ✅ Calculated LLM stats for {len(llm_stats)} vars.")
    except FileNotFoundError:
        print(f"  ❌ ERROR: Raw LLM data file not found: {llm_file}")
        return None
    except Exception as e:
        print(f"  ❌ ERROR processing LLM data: {e}")
        return None

    # 3. Question type data
    try:
        if wave_number == 1:
            type_file = os.path.join(QUESTIONS_DIR, "final_kept_variables_wave_1.csv")
        else:
            type_file = os.path.join(
    QUESTIONS_DIR,
    f"final_kept_variables_wave_{wave_number}_REENGINEERED_v7_FINAL.csv"
)

        print(f"  Loading variable type data: {type_file}")
        df_type = pd.read_csv(type_file)

        def classify_question_type(codes_str):
            if pd.isna(codes_str):
                return "Continuous"
            try:
                codes_dict = ast.literal_eval(codes_str)
                return "Binary" if isinstance(codes_dict, dict) and len(codes_dict) == 2 else "Continuous"
            except Exception:
                return "Continuous"

        df_type["Question_Type"] = df_type["Codes"].apply(classify_question_type)
        type_stats = df_type[["Variable_Name", "Question_Type"]]
        print(f"  ✅ Classified {len(type_stats)} variables.")
    except FileNotFoundError:
        print(f"  ❌ ERROR: Variable type file not found: {type_file}")
        return None

    # 4. Embeddings
    try:
        pkl_file = os.path.join(
    OUTPUT_DIR,
    f"final_data_with_embeddings_wave_{wave_number}.pkl"
)
        print(f"  Loading embedding data: {pkl_file}")
        df_base = pd.read_pickle(pkl_file)
        df_base = df_base[["Variable_Name", "Question", "Embedding"]]
        print(f"  ✅ Loaded {len(df_base)} embeddings.")
    except FileNotFoundError:
        print(f"  ❌ ERROR: Embedding .pkl file not found: {pkl_file}")
        print("      ➡️ Did you run embedding generation first?")
        return None

    # 5. Merge all    # 5. Merge all
    print("  Merging all data sources...")
    df_merged = df_base.merge(human_stats, on="Variable_Name", how="inner")
    df_merged = df_merged.merge(llm_stats, on="Variable_Name", how="inner")
    df_merged = df_merged.merge(type_stats, on="Variable_Name", how="inner")
    print(f"  Merge complete. N={len(df_merged)} variables (before filtering).")

    # --- NEW: compute empirical min / max per question across HUMAN + LLM responses ---
    # We already have df_human_long and df_raw_llm in this function.
    # Build a long "all responses" dataframe: one row per (Variable_Name, Response).
    all_responses_long = pd.concat(
        [
            df_human_long[["Variable_Name", "Response"]],
            df_raw_llm[["Variable_Name", llm_response_col]].rename(
                columns={llm_response_col: "Response"}
            ),
        ],
        ignore_index=True,
    )

    range_stats = all_responses_long.groupby("Variable_Name")["Response"].agg(
        scale_min="min",
        scale_max="max",
    ).reset_index()

    # merge scale_min / scale_max into merged dataframe
    df_merged = df_merged.merge(range_stats, on="Variable_Name", how="left")

    # Filter problematic variables
    initial_count = len(df_merged)
    vars_present = [v for v in variables_to_remove if v in df_merged["Variable_Name"].values]
    df_merged = df_merged[~df_merged["Variable_Name"].isin(vars_present)].copy()
    filtered_count = len(df_merged)
    removed_count = initial_count - filtered_count
    print(f"  Filtering: Removed {removed_count} problematic variables.")
    print(f"  N={filtered_count} variables remaining for analysis.")

    # --- NEW: raw bias + normalized averages on [0,1] scale ---
    print("  Calculating raw and normalized bias...")

    df_merged["Raw_Bias"] = (
        df_merged["Average_LLM_Response"] - df_merged["Average_Human_Response"]
    )

    # avoid division by zero when max == min
    scale_range = df_merged["scale_max"] - df_merged["scale_min"]
    scale_range_replaced = scale_range.replace(0, np.nan)

    df_merged["Normalized_Human_Response"] = (
        (df_merged["Average_Human_Response"] - df_merged["scale_min"]) / scale_range_replaced
    )
    df_merged["Normalized_LLM_Response"] = (
        (df_merged["Average_LLM_Response"] - df_merged["scale_min"]) / scale_range_replaced
    )

    df_merged["Normalized_Bias"] = (
        df_merged["Normalized_LLM_Response"] - df_merged["Normalized_Human_Response"]
    )

    # 6. Compute Cohen's d (same as before)
    print("  Calculating Standardized Mean Difference (Cohen's d)...")
    n_H = df_merged["n_Human"]
    n_L = df_merged["n_LLM"]
    s_H = df_merged["StdDev_Human_Response"].replace(0, 1e-6)
    s_L = df_merged["StdDev_LLM_Response"].replace(0, 1e-6)
    p_H = df_merged["Average_Human_Response"]
    p_L = df_merged["Average_LLM_Response"]

    s_p_cont = np.sqrt(((n_H - 1) * s_H**2 + (n_L - 1) * s_L**2) / (n_H + n_L - 2))
    p_pooled = (n_H * p_H + n_L * p_L) / (n_H + n_L)
    p_pooled_clamped = np.clip(p_pooled, 0, 1)
    s_p_bin = np.sqrt(p_pooled_clamped * (1 - p_pooled_clamped))

    df_merged["s_pooled"] = np.where(df_merged["Question_Type"] == "Binary", s_p_bin, s_p_cont)

    zero_mask = (df_merged["s_pooled"] == 0) | (df_merged["s_pooled"].isna())
    zero_vars = df_merged[zero_mask]["Variable_Name"].tolist()
    if zero_vars:
        print(f"  ⚠️ {len(zero_vars)} questions have zero variance (s_pooled=0) and will be omitted.")
    else:
        print("  ✅ All variables have non-zero variance.")

    df_merged["s_pooled"] = df_merged["s_pooled"].replace(0, np.nan)

    df_merged["y_standardized_bias"] = (
        (df_merged["Average_LLM_Response"] - df_merged["Average_Human_Response"])
        / df_merged["s_pooled"]
    )

    final_cols = [
        "Variable_Name",
        "Question",
        "Embedding",
        "Question_Type",
        "Average_Human_Response",
        "StdDev_Human_Response",
        "n_Human",
        "Average_LLM_Response",
        "StdDev_LLM_Response",
        "n_LLM",
        # NEW summary columns for PI:
        "Raw_Bias",
        "Normalized_Human_Response",
        "Normalized_LLM_Response",
        "Normalized_Bias",
        # existing:
        "s_pooled",
        "y_standardized_bias",
        "scale_min",
        "scale_max",
    ]

    pre_drop_count = len(df_merged)
    df_clean = df_merged[final_cols].dropna().copy()
    post_drop_count = len(df_clean)

    print(
        f"  Pre-processing complete. Final N={post_drop_count} "
        f"(dropped {pre_drop_count - post_drop_count} rows with NaNs)."
    )
    print("--- End Pre-processing ---")

    return df_clean



# ==========================
# Main
# ==========================

if __name__ == "__main__":
    start = time.time()

    # 1. Ensure embeddings exist
    generate_embeddings_if_needed()

    # 2. Preprocess each wave
    waves_to_process = [1, 2, 3, 4]
    all_wave_dataframes = []

    print("=" * 70)
    print("🚀 STARTING COMBINED WAVE PRE-PROCESSING")
    print("=" * 70)

    for wave in waves_to_process:
        df_wave = preprocess_wave_data(wave)
        if df_wave is not None and not df_wave.empty:
            df_wave["wave"] = wave
            all_wave_dataframes.append(df_wave)
            print(f"  ✅ Wave {wave} processed (N={len(df_wave)})")
        else:
            print(f"  ⚠️ Skipping Wave {wave} (no data).")

    if not all_wave_dataframes:
        print("❌ CRITICAL ERROR: No wave dataframes loaded. Halting.")
    else:
        master_df = pd.concat(all_wave_dataframes, ignore_index=True)
        print("\n" + "=" * 70)
        print(f"🎉 COMBINED {len(all_wave_dataframes)} WAVES")
        print(f"   Total variables in master_df: N={len(master_df)}")
        print("=" * 70)

        # For each Variable_Name, list all waves it appears in
        wave_membership = (
            master_df
            .groupby("Variable_Name")["wave"]
            .apply(lambda w: sorted(w.unique()))
            .reset_index(name="Waves_Appeared")
        )

        # Keep variables that appear in Wave 4 AND at least one other wave
        repeated_vars = wave_membership[
            wave_membership["Waves_Appeared"].apply(lambda ws: (4 in ws) and (len(ws) > 1))
        ].copy()
        repeated_vars["Waves_Appeared"] = repeated_vars["Waves_Appeared"].apply(tuple)
        print(f"  Found {len(repeated_vars)} Variable_Name(s) that are repeated and include Wave 4.")
        # Pretty print the repeated variable names and the waves they appear in
        print("\n=== Repeated questions including Wave 4 ===")
        for _, row in repeated_vars.sort_values("Variable_Name").iterrows():
            print(f"  {row['Variable_Name']}: waves {row['Waves_Appeared']}")
                # Pretty print the repeated variable names and the waves they appear in
        print("\n=== Repeated questions including Wave 4 ===")
        for _, row in repeated_vars.sort_values("Variable_Name").iterrows():
            print(f"  {row['Variable_Name']}: waves {row['Waves_Appeared']}")

        # === NEW: create a detailed sheet with all versions of these questions ===
        # Filter master_df to only those Variable_Names that are repeated and include Wave 4
        repeated_var_names = set(repeated_vars["Variable_Name"])

        repeated_versions = (
            master_df[master_df["Variable_Name"].isin(repeated_var_names)]
            .merge(repeated_vars, on="Variable_Name", how="left")   # add Waves_Appeared
            .sort_values(["Variable_Name", "wave"])
        )

        # Keep the key columns you care about
        repeated_versions = repeated_versions[[
            "Variable_Name",
            "wave",             # which wave this row is from
            "Question",         # full question text for that wave
            "Waves_Appeared",   # all waves where this Variable_Name appears
        ]]

        # Optional: rename for PI readability
        repeated_versions = repeated_versions.rename(columns={
            "Variable_Name": "Question_Name",
            "wave": "Wave",
            "Question": "Full_Question",
        })

        repeated_versions_csv = os.path.join(OUTPUT_DIR, "wave4_repeated_questions_all_versions.csv")
        repeated_versions.to_csv(repeated_versions_csv, index=False)
        print(f"  ✅ Saved all versions of repeated questions to: {repeated_versions_csv}")

        # Get the Wave 4 rows for those variables, with question text
        wave4_repeated_questions = (
            master_df[master_df["wave"] == 4]
            .merge(repeated_vars, on="Variable_Name", how="inner")
            .sort_values("Variable_Name")
            [["Variable_Name", "Question", "Waves_Appeared"]]
            .drop_duplicates()
        )

        # Save for PI
        wave4_repeated_csv = os.path.join(OUTPUT_DIR, "wave4_repeated_questions_by_Variable_Name.csv")
        wave4_repeated_questions.to_csv(wave4_repeated_csv, index=False)
        print(f"  ✅ Saved Wave 4 repeated questions to: {wave4_repeated_csv}")
# === NEW: global z-score normalization across all questions ===
        master_df["Human_norm"] = (
            (master_df["Average_Human_Response"] - master_df["Average_Human_Response"].mean())
            / master_df["Average_Human_Response"].std()
        )

        master_df["LLM_norm"] = (
            (master_df["Average_LLM_Response"] - master_df["Average_LLM_Response"].mean())
            / master_df["Average_LLM_Response"].std()
        )

        # normalized bias in z-score space
        master_df["y_normalized_bias"] = master_df["LLM_norm"] - master_df["Human_norm"]

        master_df.to_pickle(MASTER_PKL)
        master_df.to_csv(MASTER_CSV, index=False)
                # Give Cohen's d a clearer name for the PI
        master_df["Cohens_d"] = master_df["y_standardized_bias"]

        pi_cols = [
            # IDs / text
            "Variable_Name",                 # question name
            "Question",                      # full question text

            # Human summary stats
            "Average_Human_Response",
            "StdDev_Human_Response",
            "n_Human",

            # LLM summary stats
            "Average_LLM_Response",
            "StdDev_LLM_Response",
            "n_LLM",

            # Scale info (useful to interpret means)
            "scale_min",
            "scale_max",

            # Bias metrics
            "Raw_Bias",
            "Cohens_d",                      # standardized mean diff = Cohen's d
            "Normalized_Bias",              # bias on [0,1] scale
        ]

        pi_summary = master_df[pi_cols].copy()

        # Optional: make column names prettier for PI
        pi_summary = pi_summary.rename(columns={
            "Variable_Name": "Question_Name",
            "Question": "Full_Question",
            "Average_Human_Response": "Human_Mean",
            "StdDev_Human_Response": "Human_SD",
            "Average_LLM_Response": "LLM_Mean",
            "StdDev_LLM_Response": "LLM_SD",
        })

        pi_summary_csv = os.path.join(OUTPUT_DIR, "combined_waves_1_to_4_PI_question_summary.csv")
        pi_summary.to_csv(pi_summary_csv, index=False)
        print(f"  ✅ Saved PI question-level summary to: {pi_summary_csv}")

        # === NEW: PI-style sheet for Wave 4 repeated questions ===

        # Get the set of repeated Variable_Names
        repeated_var_names = set(repeated_vars["Variable_Name"])

        # Filter master_df to Wave 4 + repeated variables
        wave4_repeated_full = (
            master_df[
                (master_df["wave"] == 4)
                & (master_df["Variable_Name"].isin(repeated_var_names))
            ]
            .merge(repeated_vars, on="Variable_Name", how="left")   # adds Waves_Appeared
        )

        # Columns to keep, similar to pi_summary, plus wave + Waves_Appeared
        pi_cols_repeated = [
            "Variable_Name",                 # question name
            "Question",                      # full question text
            "Average_Human_Response",
            "StdDev_Human_Response",
            "n_Human",
            "Average_LLM_Response",
            "StdDev_LLM_Response",
            "n_LLM",
            "scale_min",
            "scale_max",
            "Raw_Bias",
            "y_standardized_bias",          # Cohen's d
            "Normalized_Bias",
            "wave",
            "Waves_Appeared",
        ]

        wave4_repeated_pi = wave4_repeated_full[pi_cols_repeated].copy()

        # Make names pretty for PI
        wave4_repeated_pi = wave4_repeated_pi.rename(columns={
            "Variable_Name": "Question_Name",
            "Question": "Full_Question",
            "Average_Human_Response": "Human_Mean",
            "StdDev_Human_Response": "Human_SD",
            "Average_LLM_Response": "LLM_Mean",
            "StdDev_LLM_Response": "LLM_SD",
            "y_standardized_bias": "Cohens_d",
            "wave": "Wave",
        })

        wave4_repeated_pi_csv = os.path.join(OUTPUT_DIR, "wave4_repeated_questions_PI_summary.csv")
        wave4_repeated_pi.to_csv(wave4_repeated_pi_csv, index=False)
        print(f"  ✅ Saved PI summary for repeated Wave 4 questions to: {wave4_repeated_pi_csv}")
        # Should be 0 if there are no true duplicates
        dup_count = master_df.duplicated(subset=["Variable_Name", "wave"]).sum()
        print("Exact duplicate rows per (Variable_Name, wave):", dup_count)

        print(f"  ✅ Saved master dataframe to:")
        print(f"     - {MASTER_PKL}")
        print(f"     - {MASTER_CSV}")

    end = time.time()
    print(f"\n⏱ Done in {end - start:.1f} seconds.")
