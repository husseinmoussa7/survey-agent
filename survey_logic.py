# survey_logic.py
# This file contains the refactored core logic from your original survey.py.
# Version 5: Fixed TypeError by accessing the .raw attribute from CrewOutput.
import os
import yaml
import json
import warnings
import asyncio
import tempfile
import shutil
import copy
import logging
from datetime import datetime
from typing import Literal, Dict, List, Any, Union, Optional
from pydantic import BaseModel, ValidationError, Field, field_validator
from crewai import Agent, Task, Crew, Process
from crewai.tasks.task_output import OutputFormat

import sys
# Add the parent directory of 'simulate_response' to the Python path
# This ensures that 'from llm_openai import openai_llm' in simulate_response.py works correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'simulate_response')))

from simulate_response import run_all_survey_responses_json
from llm_openai import openai_llm
from debias.debias import run_debias_pipeline
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import certifi
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import openai
import requests
import zipfile
import io
import time
import pandas as pd
import boto3
from dotenv import load_dotenv
import re
import subprocess
from dotenv import load_dotenv
load_dotenv()
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ========== Pydantic Models (Unchanged) ==========
class Question(BaseModel):
    question_id: str
    question_text: str
    input_type: Literal["multiple_choice", "single_choice", "slider", "text_input", "scale", "checkbox"]
    input_config: Dict[str, Any]

class Survey(BaseModel):
    theme: str
    purpose: str
    questions: List[Question]

# ========== Helper Function for Parsing AI Output ==========
def _parse_and_clean_json(raw_output: str) -> dict:
    """Cleans markdown fences and parses a JSON string from AI output."""
    if not isinstance(raw_output, str):
        if isinstance(raw_output, dict):
            return raw_output
        raise TypeError(f"Expected string for parsing, but got {type(raw_output)}")

    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output.split("```json", 1)[1]
    elif cleaned_output.startswith("```"):
        cleaned_output = cleaned_output.split("```", 1)[1]

    if "```" in cleaned_output:
        cleaned_output = cleaned_output.rsplit("```", 1)[0]

    cleaned_output = cleaned_output.strip()

    try:
        return json.loads(cleaned_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse cleaned JSON output: {e}\\nCleaned output attempt:\\n{cleaned_output}")


# ========== Interactive Survey Enhancement Flow ==========
class SurveyEnhancementFlow:
    """Interactive flow for enhancing surveys with user feedback"""

    def __init__(self):
        """Initialize the enhancement flow"""
        self.convert_agent, self.editor_agent, self.enhancement_agent = self._load_agents()
        self.survey_dict = None
        self.enhanced_dict = None

    def _load_agents(self):
        """Load the necessary agents for the survey enhancement flow"""
        convert_agent = Agent(
            role='Raw Text to JSON Survey Converter',
            goal='Accurately convert raw text containing a survey topic and questions into a structured JSON format. Focus ONLY on conversion, not on improving the content.',
            backstory='You are a precision-focused data entry bot. Your sole purpose is to parse text and structure it into a clean, basic JSON format without altering the content of the questions or topic.',
            verbose=True,
            allow_delegation=False
        )

        editor_agent = Agent(
            role='Academic Survey Editor',
            goal='Critique and enhance a given JSON survey for clarity, neutrality, and academic rigor. You ONLY modify the provided JSON; you DO NOT create new surveys.',
            backstory='You are an experienced research methodologist. You receive pre-formatted JSON surveys and your job is to refine them, providing clear justifications for every change.',
            verbose=True,
            allow_delegation=False
        )
        
        enhancement_agent = Agent(
            name="Survey Enhancement Agent",
            role="Survey Enhancement Agent",
            goal="To improve survey design by incorporating user feedback in an iterative process",
            backstory="I am an AI assistant specialized in survey design and enhancement. I work iteratively with users to refine surveys until they perfectly match the user's needs and standards.",
            verbose=True,
            allow_delegation=False
        )
        return convert_agent, editor_agent, enhancement_agent

    def run(self, survey_text):
        """
        Run the survey processing in a strict, two-step flow to ensure user input is respected.
        Step 1: Convert raw text to a basic JSON structure.
        Step 2: Take the JSON from Step 1 and enhance it.
        """
        print("--- Step 1: Converting Raw Text to Structured JSON ---")
        
        convert_task_description = f"""
        Parse the following raw survey text and convert it into a simple JSON object.
        The JSON object must have a "theme" key and a "questions" key.
        Each question in the "questions" array should have a "question_id", "question_text", "input_type", and "input_config".
        - For question type, make a best guess (e.g., 'text_input', 'multiple_choice', 'slider', 'scale').
        - Do NOT invent new questions or change the wording. Your only job is to structure the provided text.

        Raw Text to Convert:
        ---
        {survey_text}
        ---

        Required JSON Output format:
        ```json
        {{
            "theme": "The topic from the raw text",
            "purpose": "A brief purpose based on the topic",
            "questions": [
                {{
                    "question_id": "q1",
                    "question_text": "The first question from the text",
                    "input_type": "text_input",
                    "input_config": {{}}
                }},
                {{
                    "question_id": "q2",
                    "question_text": "The second question from the text",
                    "input_type": "multiple_choice",
                    "input_config": {{"options": ["Option A", "Option B"]}}
                }}
            ]
        }}
        ```
        """
        convert_task = Task(
            description=convert_task_description,
            agent=self.convert_agent,
            expected_output="A single, clean JSON object representing the structured version of the raw text."
        )

        conversion_crew = Crew(agents=[self.convert_agent], tasks=[convert_task], process=Process.sequential)
        conversion_result = conversion_crew.kickoff()
        
        try:
            # FIX: Access the .raw attribute of the CrewOutput object
            raw_text_output = conversion_result.raw
            initial_survey_json = _parse_and_clean_json(raw_text_output)
            
            if 'questions' not in initial_survey_json or not initial_survey_json['questions']:
                raise ValueError("Initial conversion failed to produce a valid questions array.")
            print("--- Step 1 SUCCESS: Successfully converted text to JSON. ---")
            print(json.dumps(initial_survey_json, indent=2))

        except (ValueError, TypeError) as e:
            raise ValueError(f"Step 1 (Conversion) failed: {e}. Raw output from converter agent: {getattr(conversion_result, 'raw', conversion_result)}")

        print("\n--- Step 2: Enhancing the Structured JSON Survey ---")

        improve_task_description = f"""
        You are an expert survey methodologist. Your task is to critique and improve the following JSON survey.
        DO NOT change the fundamental topic. Your goal is to improve question wording, add appropriate response options, and ensure academic rigor.

        You MUST produce a final JSON object with two main keys: "original_with_comments" and "revised_survey".

        JSON Survey to Improve:
        ---
        {json.dumps(initial_survey_json, indent=2)}
        ---

        Your final output MUST be a single, valid JSON object following this exact schema. Do not include any extra text or markdown fences.
        ```json
        {{
            "original_with_comments": {{
                "survey": {{ ... original survey ... }},
                "question_comments": [{{ "question_id": "q1", "comment": "Your critique." }}],
                "overall_comment": "Your overall critique."
            }},
            "revised_survey": {{ ... your improved survey ... }}
        }}
        ```
        """
        improve_task = Task(
            description=improve_task_description,
            agent=self.editor_agent,
            expected_output="A single JSON object with 'original_with_comments' and 'revised_survey' keys."
        )

        enhancement_crew = Crew(agents=[self.editor_agent], tasks=[improve_task], process=Process.sequential)
        enhancement_result = enhancement_crew.kickoff()

        try:
            # FIX: Access the .raw attribute of the CrewOutput object here as well
            raw_enhancement_output = enhancement_result.raw
            final_survey_json = _parse_and_clean_json(raw_enhancement_output)
            
            if 'revised_survey' not in final_survey_json or 'questions' not in final_survey_json['revised_survey']:
                raise ValueError("Enhancement step failed to produce a valid 'revised_survey' object.")
            print("--- Step 2 SUCCESS: Successfully enhanced the survey. ---")

            self.survey_dict = final_survey_json
            self.enhanced_dict = self.survey_dict
            return self.survey_dict

        except (ValueError, TypeError) as e:
            raise ValueError(f"Step 2 (Enhancement) failed: {e}. Raw output from editor agent: {getattr(enhancement_result, 'raw', enhancement_result)}")

    def run_single_enhancement_cycle(self, user_feedback):
        """Runs one cycle of AI enhancement based on user feedback."""
        if not self.enhanced_dict:
            raise ValueError("No survey has been processed yet.")

        survey_json_to_enhance = json.dumps(self.enhanced_dict.get('revised_survey', self.enhanced_dict), indent=2)

        enhanced_task_description = f"""
        You are a survey design expert. Your task is to revise the JSON survey provided below based on the user's feedback.
        **CRITICAL INSTRUCTIONS:**
        1.  You MUST modify the EXACT JSON survey provided in the "SURVEY TO MODIFY" section.
        2.  DO NOT generate a new survey from scratch.
        3.  Apply the user's feedback to improve the questions, options, or structure.
        4.  Your response MUST be a single JSON object with two keys: 'revised_survey' and 'explanations'.
        --- SURVEY TO MODIFY ---
        ```json
        {survey_json_to_enhance}
        ```
        --- USER FEEDBACK ---
        {user_feedback}
        --- REQUIRED OUTPUT FORMAT ---
        ```json
        {{
            "revised_survey": {{ ... }},
            "explanations": {{ ... }}
        }}
        ```
        """

        enhancement_task = Task(
            name="enhance_survey_iteratively",
            description=enhanced_task_description,
            agent=self.enhancement_agent,
            expected_output="A single JSON object containing 'revised_survey' and 'explanations'.",
            output_format=OutputFormat.JSON
        )

        enhancement_crew = Crew(
            agents=[self.enhancement_agent],
            tasks=[enhancement_task],
            process=Process.sequential,
            verbose=True
        )

        print("\n=== Running AI Enhancement ===")
        enhancement_result = enhancement_crew.kickoff()

        try:
            # FIX: Access the .raw attribute here too
            enhanced_result_raw = enhancement_result.raw
            enhanced_result = _parse_and_clean_json(enhanced_result_raw)
            if 'revised_survey' not in enhanced_result or 'questions' not in enhanced_result['revised_survey']:
                 raise ValueError("The AI's response was missing the required 'revised_survey' structure.")
            self.enhanced_dict = enhanced_result
            return self.enhanced_dict
        except (ValueError, TypeError) as e:
            raise ValueError(f"AI enhancement failed: {e}\\nRaw AI output:\\n{getattr(enhancement_result, 'raw', enhancement_result)}")

# ========== Survey to Qualtrics Conversion ==========
def survey_dict_to_qualtrics_payload(survey_dict: dict) -> dict:
    """
    Converts a custom survey_dict to a Qualtrics v3 API survey-definitions payload
    with a simplified structure that works with the step-by-step creation process.
    """
    survey_meta = survey_dict.get("revised_survey", survey_dict.get("survey", {}))

    questions_to_add = {}

    for i, q in enumerate(survey_meta.get("questions", []), 1):
        raw_id = q["question_id"]
        # Ensure we have a clean QID format
        num = re.sub(r'\D+', '', raw_id) or str(i)
        qid = f"QID{num}"

        qobj = {
            "QuestionText": q["question_text"],
            "DataExportTag": qid,
            "QuestionType": "TE",  # Default to text entry
            "Selector": "SL",      # Default selector
            "Configuration": {
                "QuestionDescriptionOption": "UseText"
            },
            "Validation": {
                "Settings": {
                    "ForceResponse": "OFF",
                    "Type": "None"
                }
            }
        }

        # Handle different question types
        it = q["input_type"]
        cfg = q.get("input_config", {})

        if it in ("multiple_choice", "single_choice"):
            choices = {}
            options = cfg.get("options", [])
            for idx, opt in enumerate(options, 1):
                choices[str(idx)] = {"Display": str(opt)}
            
            qobj.update({
                "QuestionType": "MC",
                "Selector": "SAVR" if it == "multiple_choice" else "SAHR",
                "SubSelector": "TX",
                "Choices": choices
            })
            
        elif it == "slider":
            qobj.update({
                "QuestionType": "Slider",
                "Selector": "HSLIDER",
                "SubSelector": "HBAR",
                "Configuration": {
                    "QuestionDescriptionOption": "UseText",
                    "CSSliderMin": cfg.get("min", 0),
                    "CSSliderMax": cfg.get("max", 100),
                    "GridLines": cfg.get("step", 1),
                    "NumDecimals": "0",
                    "ShowValue": True
                }
            })
            
        elif it == "scale":
            # Handle scale questions as single-choice MC
            scale_min = cfg.get("min", 1)
            scale_max = cfg.get("max", 7)  # Default to 7 instead of 5
            scale_labels = cfg.get("labels", {})
            
            # Check if the question text contains scale info and extract it
            question_text = q["question_text"]
            scale_match = re.search(r'(\d+)\s*=.*?(\d+)\s*=', question_text)
            if scale_match:
                scale_min = int(scale_match.group(1))
                scale_max = int(scale_match.group(2))
                logger.info(f"Detected scale range from question text: {scale_min}-{scale_max}")
            
            choices = {}
            for i in range(scale_min, scale_max + 1):
                choice_key = str(i)
                choice_display = scale_labels.get(str(i), str(i))
                choices[choice_key] = {"Display": choice_display}
            
            logger.info(f"Creating scale question with {len(choices)} choices: {list(choices.keys())}")
            
            qobj.update({
                "QuestionType": "MC",
                "Selector": "SAHR",  # Single Answer Horizontal
                "SubSelector": "TX",
                "Choices": choices
            })
            
        elif it == "checkbox":
            choices = {}
            options = cfg.get("options", [])
            for idx, opt in enumerate(options, 1):
                choices[str(idx)] = {"Display": str(opt)}
                
            qobj.update({
                "QuestionType": "MC",
                "Selector": "MAVR",  # Multiple Answer Vertical
                "SubSelector": "TX",
                "Choices": choices
            })
            
        elif it == "text_input":
            qobj.update({
                "QuestionType": "TE",
                "Selector": "SL",  # Single Line
                "Configuration": {
                    "QuestionDescriptionOption": "UseText"
                }
            })
            
        else:
            # Default to text input for unsupported types
            logger.warning(f"Unsupported input_type '{it}' for question {qid}. Using text input.")

        questions_to_add[qid] = qobj

    # Return a simplified payload structure
    payload = {
        "SurveyName": survey_meta.get("theme", "New Survey"),
        "Language": "EN", 
        "ProjectCategory": "CORE",
        "Questions": questions_to_add
    }
    
    return payload

# ========== Qualtrics API Client ==========
class QualtricsClient:
    """Handles all Qualtrics API interactions"""
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv('QUALTRICS_API_TOKEN')
        self.data_center = os.getenv('QUALTRICS_DATA_CENTER')
        if not self.api_token or not self.data_center:
            raise ValueError("Missing Qualtrics API credentials in environment variables.")
        self.base_url = f"https://{self.data_center}.qualtrics.com/API/v3/"
        self.headers = {"X-API-Token": self.api_token, "Content-Type": "application/json"}
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.verify = certifi.where()

    def get_survey_questions(self, survey_id: str) -> dict:
        url = f"{self.base_url}survey-definitions/{survey_id}"
        response = self.session.get(url, headers=self.headers, verify=self.verify, timeout=10)
        response.raise_for_status()
        questions = response.json().get("result", {}).get("Questions", {})
        return {q_data.get("DataExportTag", qid): re.sub('<[^<]+?>', '', q_data.get("QuestionText", "")).strip() for qid, q_data in questions.items()}

    def get_survey_responses(self, survey_id: str, file_format="csv") -> pd.DataFrame:
        export_url = f"{self.base_url}surveys/{survey_id}/export-responses"
        resp = self.session.post(export_url, headers=self.headers, json={"format": file_format, "useLabels": True}, verify=self.verify, timeout=10)
        resp.raise_for_status()
        progress_id = resp.json()["result"]["progressId"]
        
        status = ""
        while status not in ["complete", "failed"]:
            check_url = f"{export_url}/{progress_id}"
            check = self.session.get(check_url, headers=self.headers, verify=self.verify, timeout=10)
            check.raise_for_status()
            result = check.json()["result"]
            status = result["status"]
            print(f"Export status: {status} ({result.get('percentComplete', 0)}%)")
            time.sleep(2)
        
        if status == "failed":
            raise Exception(f"Export failed: {result.get('errorMessage')}")

        file_id = result["fileId"]
        download_url = f"{export_url}/{file_id}/file"
        dl = self.session.get(download_url, headers=self.headers, verify=self.verify, timeout=60)
        dl.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
            fname = next((n for n in zf.namelist() if n.endswith(f".{file_format}")), None)
            if not fname: raise FileNotFoundError(f"No '.{file_format}' file found.")
            with zf.open(fname) as f:
                return pd.read_csv(f, skiprows=[1])

    def create_survey(self, survey_name, survey_template):
        url = f"{self.base_url}survey-definitions"
        
        # Log the payload for debugging
        logger.info(f"Creating survey with payload structure:")
        logger.info(f"Survey Name: {survey_name}")
        logger.info(f"Payload keys: {list(survey_template.keys())}")
        if "Questions" in survey_template:
            logger.info(f"Number of questions: {len(survey_template['Questions'])}")
            # Log first question structure for debugging
            if survey_template["Questions"]:
                first_q_id = list(survey_template["Questions"].keys())[0]
                first_q = survey_template["Questions"][first_q_id]
                logger.info(f"First question structure: {first_q}")
        
        # Create a simplified payload that matches Qualtrics API expectations
        simplified_payload = {
            "SurveyName": survey_name,
            "Language": "EN",
            "ProjectCategory": "CORE"
        }
        
        try:
            response = self.session.post(url, headers=self.headers, json=simplified_payload, verify=self.verify, timeout=30)
            
            # Enhanced error logging
            if not response.ok:
                logger.error(f"Qualtrics API Error {response.status_code}: {response.text}")
                try:
                    error_details = response.json()
                    logger.error(f"Error details: {error_details}")
                except:
                    pass
            
            response.raise_for_status()
            survey_id = response.json()["result"]["SurveyID"]
            logger.info(f"Survey created successfully with ID: {survey_id}")
            
            # Now add questions to the created survey
            if "Questions" in survey_template and survey_template["Questions"]:
                self.add_questions_to_survey(survey_id, survey_template["Questions"])
            
            return survey_id
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error creating survey: {e}")
            logger.error(f"Response content: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating survey: {e}")
            raise

    def add_questions_to_survey(self, survey_id: str, questions_dict: dict):
        """Add questions to an existing survey one by one"""
        url = f"{self.base_url}survey-definitions/{survey_id}/questions"
        
        for question_id, question_data in questions_dict.items():
            try:
                logger.info(f"Adding question {question_id} to survey {survey_id}")
                response = self.session.post(url, headers=self.headers, json=question_data, verify=self.verify, timeout=10)
                
                if not response.ok:
                    logger.error(f"Failed to add question {question_id}: {response.status_code} - {response.text}")
                
                response.raise_for_status()
                logger.info(f"Successfully added question {question_id}")
                
            except Exception as e:
                logger.error(f"Error adding question {question_id}: {e}")
                # Continue with other questions even if one fails
                continue

    def add_questions(self, survey_id: str, questions: List[dict]):
        """Legacy method - kept for compatibility"""
        url = f"{self.base_url}survey-definitions/{survey_id}/questions"
        for q in questions:
            self.session.post(url, headers=self.headers, json=q)

    def activate_survey(self, survey_id):
        url = f"{self.base_url}surveys/{survey_id}"
        response = self.session.put(url, headers=self.headers, json={"isActive": True})
        response.raise_for_status()

    def create_distribution_link(self, survey_id):
        return f"https://{self.data_center}.qualtrics.com/jfe/form/{survey_id}"
     
# ========== MTurk API Client ==========
class MTurkClient:
    """Handles all MTurk API interactions"""
    def __init__(self, aws_access_key_id: str = None, aws_secret_access_key: str = None, use_sandbox: bool = True):
        load_dotenv()
        self.aws_access_key_id = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("Missing AWS credentials.")
        
        self.use_sandbox = use_sandbox
        endpoint = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com' if use_sandbox else 'https://mturk-requester.us-east-1.amazonaws.com'
        
        self.client = boto3.client(
            'mturk',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name='us-east-1',
            endpoint_url=endpoint
        )

    def create_hit_with_survey_link(self, survey_link, hit_config):
        question_html = f"""<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
            <HTMLContent><![CDATA[
                <!DOCTYPE html>
                <html><body>
                <p>Please complete the survey at the following link:</p>
                <p><a href="{survey_link}" target="_blank">{survey_link}</a></p>
                <p>After completing, enter the completion code below:</p>
                <p><input type="text" name="completion_code" size="40"/></p>
                </body></html>
            ]]></HTMLContent><FrameHeight>400</FrameHeight></HTMLQuestion>"""
        response = self.client.create_hit(Question=question_html, **hit_config)
        return response['HIT']['HITId']

    def get_hit_assignments(self, hit_id):
        paginator = self.client.get_paginator('list_assignments_for_hit')
        pages = paginator.paginate(HITId=hit_id)
        return [assignment for page in pages for assignment in page['Assignments']]

# ========== Qualtrics and MTurk Integration ==========
class QualtricsAndMTurkAutomation:
    def __init__(self, mturk_client: Optional[MTurkClient] = None):
        self.qualtrics = QualtricsClient()
        self.mturk = mturk_client or MTurkClient()

    def run(self, survey_payload: dict, hit_config: dict) -> dict:
        survey_id, survey_link = self.deploy_to_qualtrics_only(survey_payload)
        hit_id = self.mturk.create_hit_with_survey_link(survey_link, hit_config)
        return {"survey_id": survey_id, "survey_link": survey_link, "hit_id": hit_id}

    def deploy_to_qualtrics_only(self, survey_payload: dict):
        try:
            survey_name = survey_payload.get("SurveyName", "New Survey")
            logger.info(f"Creating survey: {survey_name}")
            
            survey_id = self.qualtrics.create_survey(survey_name, survey_payload)
            logger.info(f"Survey created with ID: {survey_id}")

            completion_qid = self._add_completion_question(survey_id, survey_payload)
            logger.info(f"Added completion question with ID: {completion_qid}")

            logger.info("Activating survey...")
            self.qualtrics.activate_survey(survey_id)
            logger.info("Survey activated successfully")
            
            survey_link = self.qualtrics.create_distribution_link(survey_id)
            logger.info(f"Survey deployed successfully: {survey_link}")
            
            return survey_id, survey_link
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise

    def _add_completion_question(self, survey_id: str, survey_payload: dict) -> str:
        """Add a completion code question to the survey"""
        try:
            existing_questions = len(survey_payload.get("Questions", {}))
            comp_qid = f"QID{existing_questions + 1}"
            
            completion_question = {
                "QuestionText": "Thank you for completing the survey! Your completion code is: ${e://Field/ResponseID}",
                "DataExportTag": f"completion_code_{comp_qid}",
                "QuestionType": "DB",  # Descriptive text block
                "Selector": "TB",      # Text block
                "Configuration": {
                    "QuestionDescriptionOption": "UseText"
                }
            }
            
            url = f"{self.qualtrics.base_url}survey-definitions/{survey_id}/questions"
            response = self.qualtrics.session.post(
                url, 
                headers=self.qualtrics.headers, 
                json=completion_question, 
                verify=self.qualtrics.verify, 
                timeout=10
            )
            
            if not response.ok:
                logger.error(f"Failed to add completion question: {response.status_code} - {response.text}")
                response.raise_for_status()
            
            return comp_qid
            
        except Exception as e:
            logger.error(f"Error adding completion question: {e}")
            return "completion_failed"
    
    def collect_and_process_results(self, survey_id: str, hit_id: Optional[str] = None):
        question_map = self.qualtrics.get_survey_questions(survey_id)
        qualtrics_df = self.qualtrics.get_survey_responses(survey_id)
        
        if not hit_id:
            return {"responses": self._format_df_with_interleaved_questions(qualtrics_df, question_map)}

        assignments = self.mturk.get_hit_assignments(hit_id)
        if not assignments:
            return {"responses": self._format_df_with_interleaved_questions(qualtrics_df, question_map)}
        
        mturk_data = []
        for a in assignments:
            try:
                root = ET.fromstring(a['Answer'])
                code_element = root.find(".//{*}Answer[{*}QuestionIdentifier='completion_code']/{*}FreeText")
                mturk_data.append({'WorkerId': a['WorkerId'], 'completion_code': code_element.text if code_element is not None else None})
            except ET.ParseError:
                logger.warning(f"Could not parse XML for assignment {a['AssignmentId']}")

        mturk_df = pd.DataFrame(mturk_data)
        merged_df = pd.merge(qualtrics_df, mturk_df, left_on='ResponseID', right_on='completion_code', how='inner')
        
        return {"responses": self._format_df_with_interleaved_questions(merged_df, question_map)}

    def _format_df_with_interleaved_questions(self, responses_df: pd.DataFrame, question_map: dict) -> pd.DataFrame:
        if not question_map or responses_df.empty: 
            return responses_df
        try:
            question_cols = sorted([col for col in responses_df.columns if col in question_map], key=lambda x: int(re.findall(r'\d+', x)[0]))
        except (IndexError, TypeError):
            question_cols = sorted([col for col in responses_df.columns if col in question_map])
        
        interleaved_data = {}
        for i, q_col in enumerate(question_cols, 1):
            interleaved_data[f'Question {i}'] = question_map.get(q_col, "Unknown")
            interleaved_data[f'Answer {i}'] = responses_df[q_col]
        return pd.DataFrame(interleaved_data)
        
# ========== Simulated Data Collection ==========
def collect_simulated_data(template_path: str, survey_context_path: str, participant_csv_path: str) -> pd.DataFrame:
    """
    Runs the data simulation and returns a DataFrame of the raw simulated responses.
    """
    if not all([run_all_survey_responses_json, openai_llm]):
        raise ImportError("Simulation dependencies are not installed.")
    
    with open(template_path, "r") as f:
        survey_template = f.read()
    with open(survey_context_path, "r") as f:
        survey_context_string = f.read()
    
    survey_json = json.loads(survey_context_string)
    sim_context = survey_json.get('revised_survey', survey_json)
    
    responses_df = run_all_survey_responses_json(
        llm=openai_llm, 
        participant_csv_path=participant_csv_path,
        survey_prompt_template=survey_template, 
        survey_context=json.dumps(sim_context)
    )
    
    # The 'Response' column contains JSON strings of answers, parse them
    parsed_responses = [json.loads(resp) if isinstance(resp, str) else resp for resp in responses_df['Response']]
    
    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(parsed_responses)
    
    return results_df

def debias_simulated_data(simulated_data_df: pd.DataFrame, survey_context: dict) -> pd.DataFrame:
    """
    Applies the debiasing pipeline to a DataFrame of simulated responses.
    This function now restructures the data to the format expected by the pipeline.
    """
    if not run_debias_pipeline:
        raise ImportError("Debias pipeline dependency is not installed.")

    logger.info("Running debias pipeline on simulated data...")

    # Restructure the data to match the format expected by the debias pipeline.
    # Expected format: [{"Question": "text", "Answers": [...]}, ...]
    questions = survey_context.get('revised_survey', survey_context).get('questions', [])
    pipeline_input_data = []
    
    # Assuming the answer columns in the DataFrame are named Q1, Q2, Q3...
    for i, q_data in enumerate(questions):
        # The column name for answers corresponds to the question index.
        answer_col_name = f"Q{i + 1}"
        if answer_col_name in simulated_data_df.columns:
            pipeline_input_data.append({
                "Question": q_data["question_text"],
                "Answers": simulated_data_df[answer_col_name].tolist()
            })
    
    # The debias pipeline requires file paths for both input and output.
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.json', encoding='utf-8') as tmp_input_file, \
         tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.json', encoding='utf-8') as tmp_output_file:
        
        input_filepath = tmp_input_file.name
        output_filepath = tmp_output_file.name

        # 1. Save the newly structured data to the temporary input file.
        json.dump(pipeline_input_data, tmp_input_file, indent=4)
        tmp_input_file.flush() # Ensure data is written to disk before pipeline reads it

        # 2. Run the pipeline with the input and output file paths.
        run_debias_pipeline(input_json=input_filepath, output_json=output_filepath)
        
        # 3. Read the debiased data back from the temporary output file.
        debiased_df = pd.read_json(output_filepath, orient='records')
        
        logger.info("Debias pipeline complete.")
        
    return debiased_df


# ========== Research Paper Generation ==========
def generate_research_paper(csv_path: str, hypothesis: Optional[str] = None):
    with open(csv_path, 'r', encoding='utf-8') as f:
        data_summary = f.read(2000)

    analyst_goal = f"Analyze data to find insights related to: '{hypothesis}'" if hypothesis else "Conduct exploratory data analysis."
    data_analyst = Agent(role='Expert Data Analyst', goal=analyst_goal, backstory="You are a meticulous data analyst...", verbose=True)
    academic_writer = Agent(role='Lead Academic Writer', goal="Write a research paper from the analysis.", backstory="You are a professional academic writer...", verbose=True)

    analysis_task = Task(description=f"Analyze the data: {data_summary}", agent=data_analyst, expected_output="A markdown report of the analysis.")
    writing_task = Task(description="Write a full research paper based on the analysis.", agent=academic_writer, context=[analysis_task], expected_output="A complete research paper in markdown.")
    
    paper_crew = Crew(agents=[data_analyst, academic_writer], tasks=[analysis_task, writing_task], process=Process.sequential, verbose=True)
    
    result = paper_crew.kickoff()
    return result.raw