#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Sequence

import pandas as pd
from tqdm import tqdm


REQUIRED_PARTICIPANT_COLS = ["ParticipantID", "Age", "Gender", "Race"]


def _validate_participants(df: pd.DataFrame, csv_path: str | Path) -> None:
    missing = [c for c in REQUIRED_PARTICIPANT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Participant CSV is missing columns {missing}. "
            f"Found: {list(df.columns)}. File: {csv_path}"
        )


def run_single_survey_response_json(
    llm: Callable[[str], str],
    survey_prompt_template: str,
    survey_context: str,
    participant_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    survey_context: a JSON string with keys: theme, purpose, questions[{question_text,input_config{options}}]
    """
    background_prompt = (
        survey_prompt_template.replace("$age", str(participant_info["Age"]))
        .replace("$gender", str(participant_info["Gender"]))
        .replace("$race", str(participant_info["Race"]))
    )

    ctx = json.loads(survey_context)
    questions = ctx["questions"]

    prompt_body = "\n".join(
        [
            f"Q{i+1}: {q['question_text']}\nOptions: {', '.join(q['input_config']['options'])}"
            for i, q in enumerate(questions)
        ]
    )

    full_prompt = (
        f"{background_prompt}\n\n"
        f"Survey Theme: {ctx.get('theme','')}\n"
        f"Purpose: {ctx.get('purpose','')}\n\n"
        f"Please answer the following questions in JSON format:\n\n{prompt_body}"
    )

    response = llm(full_prompt)
    return {
        "ParticipantID": participant_info["ParticipantID"],
        "Age": participant_info["Age"],
        "Gender": participant_info["Gender"],
        "Race": participant_info["Race"],
        "Response": response,
    }


def run_all_survey_responses_json(
    llm: Callable[[str], str],
    participant_csv_path: str | Path,
    survey_prompt_template: str,
    survey_context: str,
    max_participants: Optional[int] = None,
) -> pd.DataFrame:
    df_participants = pd.read_csv(participant_csv_path)
    _validate_participants(df_participants, participant_csv_path)

    if max_participants is not None:
        df_participants = df_participants.head(max_participants)

    responses = []
    for _, row in tqdm(df_participants.iterrows(), total=len(df_participants)):
        participant_info = {c: row[c] for c in REQUIRED_PARTICIPANT_COLS}
        responses.append(
            run_single_survey_response_json(llm, survey_prompt_template, survey_context, participant_info)
        )

    return pd.DataFrame(responses)


def run_single_survey_response_str(
    llm: Callable[[str], str],
    survey_prompt_template: str,
    survey_str: str,
    participant_info: Dict[str, Any],
) -> Dict[str, Any]:
    background_prompt = (
        survey_prompt_template.replace("$age", str(participant_info["Age"]))
        .replace("$gender", str(participant_info["Gender"]))
        .replace("$race", str(participant_info["Race"]))
    )

    full_prompt = (
        f"{background_prompt}\n\n"
        f"Please answer the following question:\n\n{survey_str}"
    )

    response = llm(full_prompt)
    return {
        "ParticipantID": participant_info["ParticipantID"],
        "Age": participant_info["Age"],
        "Gender": participant_info["Gender"],
        "Race": participant_info["Race"],
        "Response": response,
    }


def run_all_survey_responses_str(
    llm: Callable[[str], str],
    participant_csv_path: str | Path,
    survey_prompt_template: str,
    survey_str: str,
    max_participants: Optional[int] = None,
) -> pd.DataFrame:
    df_participants = pd.read_csv(participant_csv_path)
    _validate_participants(df_participants, participant_csv_path)

    if max_participants is not None:
        df_participants = df_participants.head(max_participants)

    responses = []
    for _, row in tqdm(df_participants.iterrows(), total=len(df_participants)):
        participant_info = {c: row[c] for c in REQUIRED_PARTICIPANT_COLS}
        responses.append(
            run_single_survey_response_str(llm, survey_prompt_template, survey_str, participant_info)
        )
    return pd.DataFrame(responses)
