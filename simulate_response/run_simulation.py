#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   run_simulation.py
@Time    :   2025/05/15 16:03:28
@Author  :   Shijian Liu
@Version :   1.0
@Contact :   lshijian405@gmail.com
@Desc    :   This is the basic demo to run the survey simulation.
'''
from simulate_response import run_all_survey_responses_json
from llm_openai import openai_llm

# response template
with open("survey_response_template.txt", "r") as f:
    survey_template = f.read()

# test survey
with open("test_survey.json", "r") as f:
    survey_context = f.read()

responses_df = run_all_survey_responses_json(
    llm=openai_llm,
    participant_csv_path="participant_pool.csv",
    survey_prompt_template=survey_template,
    survey_context=survey_context
)

responses_df.to_csv("simulated_survey_responses.csv", index=False)
print("Saved responses to simulated_survey_responses.csv")