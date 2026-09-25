#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from llm_openai import openai_llm
from simulate_response import run_all_survey_responses_str


@dataclass
class Config:
    project_root: Path
    questions_dir: Path
    output_dir: Path
    participant_csv: Path

    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 50
    api_delay_seconds: float = 0.5
    waves_to_process: List[int] = None

    survey_prompt_template: str = (
        "You are a $age-year-old $gender identifying as $race. "
        "Please answer the following survey question."
    )

    max_participants: Optional[int] = None   # useful for dry-runs
    max_questions: Optional[int] = None      # useful for dry-runs


def extract_number(response_text: str) -> Optional[str]:
    if not isinstance(response_text, str):
        return None
    m = re.search(r"^\s*(-?\d+(\.\d+)?)", response_text)
    return m.group(1) if m else None


def build_survey_string(variable_name: str, question_text: str, codes_str: str) -> str:
    """
    Mimics your combined logic:
      - if "_TEXT" in question_text => numeric entry
      - else parse codes and include choices + instruction
    """
    options_string = ""

    if "_TEXT" in str(question_text):
        response_format_instruction = " Please respond with only a single number (e.g., 42)."
        options_string = ""
    else:
        response_format_instruction = " Please respond with only the single number (e.g., 1, 2, 3) that represents your answer."
        try:
            codes_dict = ast.literal_eval(str(codes_str))
            parsed_codes = {str(k).split(".")[0]: str(v) for k, v in codes_dict.items()}

            # Special-case left as you had it (harmless if not present)
            if str(variable_name).startswith("Agentic Communal"):
                a1 = parsed_codes.get("1", "1").split("\n")[0].strip()
                a5 = parsed_codes.get("5", "5").split("\n")[0].strip()
                a9 = parsed_codes.get("9", "9").split("\n")[0].strip()
                options_string = (
                    f" This is a 9-point scale. Choices are 1..9, where "
                    f"1='{a1}', 5='{a5}', 9='{a9}'."
                )
            else:
                cleaned_codes = []
                try:
                    sorted_keys = sorted(parsed_codes.keys(), key=lambda k: int(k))
                except ValueError:
                    sorted_keys = sorted(parsed_codes.keys())

                for k in sorted_keys:
                    v = parsed_codes[k]
                    if str(v) == str(k):
                        cleaned_codes.append(f"{k}")
                    else:
                        v_clean = v.split("\n")[0]
                        if len(v_clean) > 50:
                            v_clean = v_clean[:50] + "..."
                        cleaned_codes.append(f"{k}-{v_clean}")

                options_string = " Choices: " + ", ".join(cleaned_codes)

        except (SyntaxError, ValueError, TypeError):
            # If codes are malformed, still ask, but without choices
            options_string = ""

    return f"{question_text}{options_string}{response_format_instruction}"


def main() -> None:
    # If you run this from code/simulate_response/, project_root is parents[2]
    project_root = Path(__file__).resolve().parents[2]

    cfg = Config(
        project_root=project_root,
        questions_dir=project_root / "data_questions",
        output_dir=project_root / "data_llm",
        participant_csv=Path(__file__).resolve().parent / "participant_pool.csv",
        waves_to_process=[1,2,3, 4],
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Wrap openai_llm so simulate_response only calls llm(prompt)
    def llm_fn(prompt: str) -> str:
        return openai_llm(
            prompt,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    for wave_number in cfg.waves_to_process:
        print("\n" + "=" * 70)
        print(f"🚀 STARTING POOLED SIMULATION FOR WAVE {wave_number}")
        print("=" * 70)

        input_file = cfg.questions_dir / f"final_kept_variables_wave_{wave_number}_REENGINEERED_v7_FINAL.csv"
        if not input_file.exists():
            print(f"❌ Missing input file: {input_file} (skipping wave)")
            continue

        df_questions = pd.read_csv(input_file)

        if "Question" not in df_questions.columns or "Variable_Name" not in df_questions.columns:
            print(f"❌ Bad schema in {input_file}. Need columns: Variable_Name, Question, Codes")
            continue

        df_questions = df_questions.dropna(subset=["Question"]).copy()
        if "Codes" not in df_questions.columns:
            df_questions["Codes"] = ""

        if cfg.max_questions is not None:
            df_questions = df_questions.head(cfg.max_questions)

        print(f"✅ Loaded '{input_file}' ({len(df_questions)} questions)")
        print(f"Simulating participants from: {cfg.participant_csv}")

        all_responses_for_wave = []

        for _, row in tqdm(df_questions.iterrows(), total=len(df_questions), desc=f"Wave {wave_number} Questions"):
            variable_name = row["Variable_Name"]
            question_text = row["Question"]
            codes_str = row.get("Codes", "")

            survey_str = build_survey_string(variable_name, question_text, codes_str)

            try:
                df_single = run_all_survey_responses_str(
                    llm=llm_fn,
                    participant_csv_path=cfg.participant_csv,
                    survey_prompt_template=cfg.survey_prompt_template,
                    survey_str=survey_str,
                    max_participants=cfg.max_participants,
                )
                df_single["Variable_Name"] = variable_name
                all_responses_for_wave.append(df_single)
            except Exception as e:
                print(f"\n   ❌ Error running simulation for question {variable_name}: {e}")
                all_responses_for_wave.append(
                    pd.DataFrame(columns=["ParticipantID", "Age", "Gender", "Race", "Response", "Variable_Name"])
                )

            time.sleep(cfg.api_delay_seconds)

        if not all_responses_for_wave:
            print(f"\n[⚠️ No responses generated for Wave {wave_number}]")
            continue

        df_wave = pd.concat(all_responses_for_wave, ignore_index=True)
        df_wave["LLM_Response_Number"] = df_wave["Response"].apply(extract_number)
        df_wave = df_wave.rename(columns={"Response": "LLM_Raw_Response"})

        cols_order = [
            "ParticipantID", "Age", "Gender", "Race",
            "Variable_Name", "LLM_Raw_Response", "LLM_Response_Number"
        ]
        for c in cols_order:
            if c not in df_wave.columns:
                df_wave[c] = pd.NA
        df_wave = df_wave[cols_order]

        output_file = cfg.output_dir / f"llm_pooled_responses_wave_{wave_number}_v7_FINAL.csv"
        df_wave.to_csv(output_file, index=False)
        print(f"\n[✅ DONE] Wave {wave_number} complete → {output_file}")

    print("\n" + "=" * 70)
    print("🎉 All pooled simulations finished!")
    print("=" * 70)


if __name__ == "__main__":
    main()
