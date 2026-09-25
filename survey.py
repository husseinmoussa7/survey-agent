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
from knowledge_sources import (
    initial_crew_knowledge,
    enhancement_crew_knowledge,
    paper_crew_knowledge,
    attach_knowledge_to_crew,
)
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


# ========== Enhanced Pydantic Models with Validation ==========
class ChoiceOption(BaseModel):
    text: str
    value: str

class ChoiceConfig(BaseModel):
    options: List[str] = Field(default_factory=list)

    @field_validator('options')
    @classmethod
    def validate_options(cls, v):
        if not v:
            raise ValueError("Options cannot be empty for a choice-based question")
        return v

class SliderConfig(BaseModel):
    min: float = 0
    max: float = 100
    step: float = 1

    @field_validator('max')
    @classmethod
    def validate_max(cls, v, info):
        values = info.data
        if 'min' in values and v <= values['min']:
            raise ValueError("Max value must be greater than min value")
        return v

    @field_validator('step')
    @classmethod
    def validate_step(cls, v):
        if v <= 0:
            raise ValueError("Step value must be positive")
        return v

class TextInputConfig(BaseModel):
    placeholder: Optional[str] = None
    multiline: bool = False

class Question(BaseModel):
    question_id: str
    question_text: str
    input_type: Literal["multiple_choice", "single_choice", "slider", "text_input"]
    input_config: Dict[str, Any]

    @field_validator('question_text')
    @classmethod
    def validate_question_text(cls, v):
        if len(v.strip()) < 5:
            raise ValueError("Question text is too short")
        return v

    @field_validator('input_config')
    @classmethod
    def validate_input_config(cls, v, info):
        values = info.data
        if 'input_type' in values:
            input_type = values['input_type']
            if input_type in ['multiple_choice', 'single_choice']:
                if 'options' not in v or not v['options']:
                    raise ValueError(f"{input_type} must have options defined")
            elif input_type == 'slider':
                required_keys = ['min', 'max']
                for key in required_keys:
                    if key not in v:
                        raise ValueError(f"Slider configuration missing required key: {key}")
        return v

class Survey(BaseModel):
    theme: str
    purpose: str
    questions: List[Question]

    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        if not v:
            raise ValueError("Survey must have at least one question")
        # Ensure question_ids are unique
        ids = [q.question_id for q in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique")
        return v

class QuestionComment(BaseModel):
    question_id: str
    comment: str

class AnnotatedSurvey(BaseModel):
    survey: Survey
    question_comments: List[QuestionComment]
    overall_comment: Optional[str] = None

class SurveyImprovementResult(BaseModel):
    original_with_comments: AnnotatedSurvey
    revised_survey: Survey

class SurveyConversionOutput(BaseModel):
    """Model to validate the output of the survey conversion agent"""
    title: str
    fields: List[Dict[str, Any]]

    @field_validator('fields')
    @classmethod
    def validate_fields(cls, v):
        if not v:
            raise ValueError("Survey must have at least one field")

        for i, field in enumerate(v):
            if 'title' not in field:
                raise ValueError(f"Field {i+1} is missing 'title'")
            if 'type' not in field:
                raise ValueError(f"Field {i+1} is missing 'type'")

            if field['type'] in ['multiple_choice', 'single_choice']:
                if 'options' not in field or not field['options']:
                    raise ValueError(f"Field {i+1} ({field.get('title', 'Untitled')}) is missing options for {field['type']}")

        return v

def validate_conversion_output(raw_output: str) -> Dict:
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output.split("```json", 1)[1]
    elif cleaned_output.startswith("```"):
        cleaned_output = cleaned_output.split("```", 1)[1]

    if "```" in cleaned_output:
        cleaned_output = cleaned_output.rsplit("```", 1)[0]

    cleaned_output = cleaned_output.strip()

    try:
        parsed_dict = json.loads(cleaned_output)
        logger.debug(f"Parsed JSON structure: {json.dumps(parsed_dict, indent=2)}")
        validated = SurveyConversionOutput(**parsed_dict)
        return validated.dict()
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON output: {e}\\nRaw output:\\n{raw_output}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except ValidationError as e:
        error_msg = f"Output validation failed: {e}\\nRaw output:\\n{raw_output}"
        logger.error(error_msg)
        raise ValidationError(e.errors(), SurveyConversionOutput)

def convert_to_question_format(conversion_output: Dict) -> List[Question]:
    questions = []
    for i, field in enumerate(conversion_output["fields"]):
        question_id = f"q{i+1}"
        question_text = field["title"]

        if field["type"] == "multiple_choice":
            input_type = "multiple_choice"
            input_config = {"options": field.get("options", [])}
        elif field["type"] == "text_input":
            input_type = "text_input"
            input_config = {"multiline": False}
        elif field["type"] == "slider":
            input_type = "slider"
            input_config = {"min": 0, "max": 100, "step": 1}
        else:
            input_type = "single_choice"
            input_config = {"options": field.get("options", [])}

        try:
            question = Question(
                question_id=question_id,
                question_text=question_text,
                input_type=input_type,
                input_config=input_config
            )
            questions.append(question)
        except ValidationError as e:
            logger.warning(f"Question validation failed for question {i+1}: {e}")
            fallback_question = Question(
                question_id=question_id,
                question_text=question_text if len(question_text) >= 5 else f"Question {i+1}",
                input_type="multiple_choice" if "options" in field else "text_input",
                input_config={"options": field.get("options", ["Yes", "No"])} if "options" in field else {"multiline": False}
            )
            questions.append(fallback_question)

    return questions

# ========== Interactive Survey Enhancement Flow ==========
class SurveyEnhancementFlow:
    """Interactive flow for enhancing surveys with user feedback"""

    def __init__(self):
        """Initialize the enhancement flow"""
        self.convert_agent, self.editor_agent = self._load_agents()
        self.convert_task, self.research_task, self.improve_task, self.enhancement_task = self._load_tasks(
            self.convert_agent, self.editor_agent
        )
        self.survey_dict = None
        self.enhanced_dict = None
        self.temp_file = None

    def _load_agents(self):
        """Load the necessary agents for the survey enhancement flow"""
        conv_cfg = self._load_yaml("config/agents/survey_convert_agent.yaml").get("survey_convert_agent", {})
        convert_agent = Agent(
            name="Survey Content Conversion Agent",
            role=conv_cfg.get("role", "Survey conversion specialist"),
            goal=conv_cfg.get("goal", "Convert text survey into structured JSON"),
            backstory=conv_cfg.get("backstory", "Expert in survey methodology and format conversion"),
            verbose=conv_cfg.get("verbose", True),
            allow_delegation=conv_cfg.get("allow_delegation", False)
        )

        edit_cfg = self._load_yaml("config/agents/survey_editor.yaml").get("survey_editor", {})
        editor_agent = Agent(
            name="Academic Survey Designer",
            role=edit_cfg.get("role", "Academic survey editor"),
            goal=edit_cfg.get("goal", "Enhance survey quality and academic rigor"),
            backstory=edit_cfg.get("backstory", "Experienced researcher with expertise in survey methodology"),
            verbose=edit_cfg.get("verbose", True),
            allow_delegation=edit_cfg.get("allow_delegation", False)
        )

        return convert_agent, editor_agent

    def _load_yaml(self, path: str) -> dict:
        """Load YAML configuration file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            if 'survey_convert_agent' in path:
                return {"survey_convert_agent": {
                    "role": "Survey conversion specialist",
                    "goal": "Convert text survey into structured JSON",
                    "backstory": "Expert in survey methodology and format conversion",
                    "verbose": True,
                    "allow_delegation": False
                }}
            elif 'survey_editor' in path:
                return {"survey_editor": {
                    "role": "Academic survey editor",
                    "goal": "Enhance survey quality and academic rigor",
                    "backstory": "Experienced researcher with expertise in survey methodology",
                    "verbose": True,
                    "allow_delegation": False
                }}
            else:
                return {}

    def _load_tasks(self, convert_agent, editor_agent):
        """Load the tasks for the survey enhancement flow"""
        conv_t = self._load_yaml("config/tasks/convert_survey_to_json.yaml").get("convert_survey_to_json", {})
        convert_task = Task(
            name="convert_survey_to_json",
            description=conv_t.get("description", "Convert the following survey (provided as raw text) into a structured JSON schema suitable for creating a survey in Qualtrics or similar platforms."),
            agent=convert_agent,
            tool=conv_t.get("tool"),
            expected_output=conv_t.get("expected_output", "A JSON object representing the survey"),
            output_format=OutputFormat.JSON
        )

        res_t = self._load_yaml("config/tasks/apply_survey_enhancements.yaml").get("research_task", {})
        description = res_t.get("description", "Conduct a thorough research about the survey topic Make sure you find any interesting and relevant information given the current year is {current_year}.")
        description = description.replace("{topic}", "the survey topic").replace("{current_year}", str(datetime.now().year))

        expected_output = res_t.get("expected_output", "A bullet-point list of relevant information about {topic}.")
        expected_output = expected_output.replace("{topic}", "the survey topic")

        research_task = Task(
            name="research_task",
            description=description,
            agent=editor_agent,
            expected_output=expected_output,
            output_format=OutputFormat.JSON
        )

        imp_t = self._load_yaml("config/tasks/apply_survey_enhancements.yaml").get("improve_survey", {})
        default_description = """You will receive:
  - original_survey: the survey as a JSON object conforming to the Survey model
  - comments: an array of feedback comments for each question
  - suggestions: an array of proposed improvements for each question and for the survey as a whole

Your tasks:
  1. Produce `original_with_comments`: annotate the original_survey with comments only (do not include suggestions in this part).
  2. Produce revised_survey: apply the suggestions to generate an updated survey JSON, ensuring that each question is rephrased to align with professional academic standards.
  3. Respond with exactly this JSON schema (no extra keys, no markdown fences):

  {{
    "original_with_comments": {{
      "survey": {{
        "theme": "<survey theme>",
        "purpose": "<survey purpose>",
        "questions": [
          {{
            "question_id": "<question_id>",
            "question_text": "<question text>",
            "input_type": "<multiple_choice|single_choice|text_input>",
            "input_config": {{ … }}
          }}
        ]
      }},
      "question_comments": [
        {{ "question_id": "<question_id>", "comment": "<comment>" }}
      ],
      "overall_comment": "<overall comment>"
    }},
    "revised_survey": {{
      "theme": "<survey theme>",
      "purpose": "<survey purpose>",
      "questions": [
        {{
          "question_id": "<question_id>",
          "question_text": "<question text>",
          "input_type": "<multiple_choice|single_choice|text_input>",
          "input_config": {{ … }}
        }}
      ]
    }}
  }}"""

        description = imp_t.get("description", default_description)
        description = description.replace("{", "{{").replace("}", "}}")

        expected_output = imp_t.get("expected_output", "A JSON object with original_with_comments and revised_survey")
        expected_output = expected_output.replace("{", "{{").replace("}", "}}")

        improve_task = Task(
            name="improve_survey",
            description=description,
            agent=editor_agent,
            expected_output=expected_output,
            output_format=OutputFormat.JSON
        )

        enh_t = self._load_yaml("config/tasks/enhance_survey_iteratively.yaml").get("enhance_survey_iteratively", {})
        enhancement_task = Task(
            name="enhance_survey_iteratively",
            description=enh_t.get("description", "Review and enhance the provided survey JSON."),
            agent=editor_agent,
            expected_output=enh_t.get("expected_output", "A JSON object containing 'revised_survey' and 'explanations'."),
            output_format=OutputFormat.JSON
        )

        return convert_task, research_task, improve_task, enhancement_task

    def run(self, survey_text):
        """
        Run the initial survey conversion and enhancement flow
        
        Args:
            survey_text (str): Raw survey text input
            
        Returns:
            dict: Enhanced survey dictionary
        """
        print("Starting survey conversion and enhancement flow...")

        first_line = survey_text.splitlines()[0]
        topic = first_line.replace('Topic:', '').strip()
        current_year = datetime.now().year

        task_inputs = {
            'survey_text': survey_text,
            'topic': topic,
            'current_year': current_year
        }

        # Instantiate Crew with knowledge if the installed CrewAI supports it
        _crew_kwargs = dict(
            agents=[self.convert_agent, self.editor_agent],
            tasks=[self.convert_task, self.research_task, self.improve_task],
            process=Process.sequential,
            verbose=True,
        )
        try:
            initial_crew = Crew(knowledge=initial_crew_knowledge, **_crew_kwargs)
        except TypeError:
            # Older CrewAI without 'knowledge' kwarg
            initial_crew = Crew(**_crew_kwargs)
            attach_knowledge_to_crew(initial_crew, initial_crew_knowledge)

        print("\n=== Running Initial Survey Processing ===")
        crew_result = initial_crew.kickoff(inputs=task_inputs)

        raw = crew_result.raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        if raw.startswith("json"):
            raw = raw[4:].strip()

        try:
            self.survey_dict = json.loads(raw)
            self._validate_survey_structure(self.survey_dict)
            self.enhanced_dict = self.survey_dict
            self._print_survey_summary(self.survey_dict)
            return self.survey_dict

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {e}\\nRaw output:\\n{raw}")
        except ValueError as e:
            raise ValueError(f"Survey validation failed: {e}\\nRaw output:\\n{raw}")

    def _validate_survey_structure(self, survey_dict):
        """
        Validates that the survey has a proper structure
        
        Args:
            survey_dict: The survey dictionary to validate
            
        Raises:
            ValueError: If the survey structure is invalid
        """
        if 'revised_survey' not in survey_dict:
            raise ValueError("Missing 'revised_survey' key in survey dictionary")

        revised = survey_dict['revised_survey']
        if 'questions' not in revised or not isinstance(revised['questions'], list):
            raise ValueError("Missing or invalid 'questions' array in revised survey")
        if len(revised['questions']) == 0:
            raise ValueError("Survey must have at least one question")
        return True

    def _process_enhancement_result(self, result_raw, original_survey_dict):
        """
        Process and standardize the enhancement result from various possible formats
        
        Args:
            result_raw (str): Raw JSON string from enhancement agent
            original_survey_dict (dict): The original survey dictionary before enhancement
            
        Returns:
            dict: Standardized survey dictionary
        """
        try:
            enhanced_result = json.loads(result_raw)

            if self._is_default_survey(enhanced_result, original_survey_dict):
                logger.warning("Enhancement agent may have returned a default survey instead of enhancing the provided one")
                print("\nWarning: The enhancement appears to have used a default survey template rather than your inputted content.")
                print("Attempting to apply enhancements to your original survey content...\n")
                return self._merge_enhancements_with_original(enhanced_result, original_survey_dict)

            if 'survey' in enhanced_result:
                survey_data = enhanced_result['survey']
                if 'questions' in survey_data and isinstance(survey_data['questions'], list):
                    questions = []
                    for i, q in enumerate(survey_data['questions']):
                        question_id = q.get('question_id', q.get('id', f'q{i+1}'))
                        question_text = q.get('question_text', q.get('question', q.get('text', '')))
                        input_type = q.get('input_type', q.get('type', ''))
                        if input_type in ['open_ended', 'open_text', 'text']:
                            input_type = 'text_input'
                        elif input_type in ['multiple_choice', 'single_choice']:
                            pass
                        elif input_type == 'scale':
                            input_type = 'slider'
                        else:
                            input_type = 'multiple_choice'

                        input_config = {}
                        options = q.get('options', [])
                        if options and isinstance(options, list):
                            if isinstance(options[0], dict) and ('label' in options[0] or 'text' in options[0]):
                                input_config['options'] = [opt.get('label', opt.get('text', '')) for opt in options]
                            else:
                                input_config['options'] = options

                        if input_type == 'slider' and ('scale_min' in q or 'scale_max' in q):
                            input_config['min'] = q.get('scale_min', 1)
                            input_config['max'] = q.get('scale_max', 10)
                            input_config['step'] = q.get('scale_step', 1)

                        question = {
                            'question_id': str(question_id),
                            'question_text': question_text,
                            'input_type': input_type,
                            'input_config': input_config
                        }
                        questions.append(question)

                    return {
                        'revised_survey': {
                            'theme': survey_data.get('title', survey_data.get('theme', 'Enhanced Survey')),
                            'purpose': survey_data.get('purpose', survey_data.get('description', '')),
                            'questions': questions
                        },
                        'explanations': enhanced_result.get('explanations', {})
                    }
                else:
                    return {'revised_survey': survey_data}

            elif 'revised_survey' in enhanced_result:
                return enhanced_result

            else:
                if 'questions' in enhanced_result and isinstance(enhanced_result['questions'], list):
                    return {'revised_survey': enhanced_result}
                else:
                    return {'revised_survey': enhanced_result}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse enhancement result: {e}")
            raise ValueError(f"Invalid JSON in enhancement result: {e}")
        except Exception as e:
            logger.error(f"Error processing enhancement result: {e}")
            raise ValueError(f"Error processing enhancement result: {e}")

    def _is_default_survey(self, enhanced_result, original_survey_dict):
        """
        Check if the enhanced result looks like a default survey rather than an enhancement of the original
        
        Args:
            enhanced_result (dict): The enhanced survey dictionary
            original_survey_dict (dict): The original survey dictionary
            
        Returns:
            bool: True if it appears to be a default survey
        """
        original_theme = None
        original_questions = []
        enhanced_theme = None
        enhanced_questions = []

        if 'revised_survey' in original_survey_dict:
            original_theme = original_survey_dict['revised_survey'].get('theme', '')
            original_questions = original_survey_dict['revised_survey'].get('questions', [])
        elif 'survey' in original_survey_dict:
            original_theme = original_survey_dict['survey'].get('theme', original_survey_dict['survey'].get('title', ''))
            original_questions = original_survey_dict['survey'].get('questions', [])

        if 'survey' in enhanced_result:
            enhanced_theme = enhanced_result['survey'].get('theme', enhanced_result['survey'].get('title', ''))
            enhanced_questions = enhanced_result['survey'].get('questions', [])
        elif 'revised_survey' in enhanced_result:
            enhanced_theme = enhanced_result['revised_survey'].get('theme', '')
            enhanced_questions = enhanced_result['revised_survey'].get('questions', [])

        if original_theme and enhanced_theme and original_theme.lower() != enhanced_theme.lower():
            if not (original_theme.lower() in enhanced_theme.lower() or enhanced_theme.lower() in original_theme.lower()):
                return True

        if len(original_questions) > 0 and len(enhanced_questions) > 0:
            if len(enhanced_questions) < len(original_questions) * 0.5 or len(enhanced_questions) > len(original_questions) * 1.5:
                return True

        if len(original_questions) > 0 and len(enhanced_questions) == 0:
            return True

        return False

    def _merge_enhancements_with_original(self, enhanced_result, original_survey_dict):
        """
        Attempt to merge any valid enhancements with the original survey
        
        Args:
            enhanced_result (dict): The potentially default-based enhancement result
            original_survey_dict (dict): The original survey dictionary
            
        Returns:
            dict: The merged survey dictionary
        """
        merged_result = copy.deepcopy(original_survey_dict)
        explanations = {}
        if 'explanations' in enhanced_result:
            explanations = enhanced_result['explanations']
        merged_result['explanations'] = explanations
        return merged_result

    def interactive_enhancement(self):
        """
        Interactive enhancement loop that allows users to review, manually modify,
        and request AI enhancements to the survey.
        """
        if not self.survey_dict:
            raise ValueError("No survey has been processed yet. Run the 'run' method first.")

        self.enhanced_dict = self.survey_dict
        iteration = 1

        while True:
            print("\n" + "="*50)
            print(f"ENHANCEMENT ITERATION #{iteration}")
            print("="*50)

            print("\nWhat would you like to do?")
            print("1. Review current survey")
            print("2. Request AI enhancement with feedback")
            print("3. Manually edit survey JSON")
            print("4. Save survey to file")
            print("5. Deploy survey to Qualtrics")
            print("6. Return to Main Menu") 

            choice = input("\nEnter your choice (1-6): ")

            if choice == "1":
                self._print_survey_summary(self.enhanced_dict)

            elif choice == "2":
                print("\nPlease provide your feedback/suggestions for the survey (enter a blank line to finish):")
                feedback_lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    feedback_lines.append(line)

                user_feedback = "\n".join(feedback_lines)

                if not user_feedback:
                    print("No feedback provided. Returning to menu.")
                    continue
                
                # --- START: MODIFIED CODE ---

                # Convert the current survey state to a clean JSON string for the prompt
                survey_json_to_enhance = json.dumps(self.enhanced_dict.get('revised_survey', self.enhanced_dict), indent=2)

                # Create a single, powerful prompt that leaves no room for ambiguity.
                # This prompt explicitly tells the agent to modify the provided JSON and defines the exact output structure.
                enhanced_task_description = f"""
                You are a survey design expert. Your task is to revise the JSON survey provided below based on the user's feedback.

                **CRITICAL INSTRUCTIONS:**
                1.  You MUST modify the EXACT JSON survey provided in the "SURVEY TO MODIFY" section.
                2.  DO NOT generate a new survey from scratch. Do not use a default template.
                3.  Apply the user's feedback to improve the questions, options, or structure.
                4.  Your response MUST be a single JSON object with two keys: 'revised_survey' and 'explanations'.
                5.  The 'revised_survey' MUST conform to the structure of the original survey.

                --- SURVEY TO MODIFY ---
                ```json
                {survey_json_to_enhance}
                ```

                --- USER FEEDBACK ---
                {user_feedback}

                --- REQUIRED OUTPUT FORMAT ---
                Provide your response as a single, valid JSON object like this:
                ```json
                {{
                    "revised_survey": {{
                        "theme": "The original or an improved theme",
                        "purpose": "The original or an improved purpose",
                        "questions": [
                            {{
                                "question_id": "q1",
                                "question_text": "The revised question text...",
                                "input_type": "...",
                                "input_config": {{...}}
                            }}
                        ]
                    }},
                    "explanations": {{
                        "q1": "Explanation for why you changed question 1.",
                        "overall": "General comments on the enhancements made."
                    }}
                }}
                ```
                """

                # Reload template and format with dynamic content
                enh_t = self._load_yaml("config/tasks/enhance_survey_iteratively.yaml").get("enhance_survey_iteratively", {})
                description_template = enh_t.get("description", "")
                enhanced_task_description = description_template.replace("{survey_json_to_enhance}", survey_json_to_enhance).replace("{user_feedback}", user_feedback)

                enhancement_task = Task(
                    name="enhance_survey_iteratively",
                    description=enhanced_task_description,
                    agent=self.editor_agent,
                    expected_output="A single JSON object containing the 'revised_survey' and 'explanations' keys.",
                    output_format=OutputFormat.JSON
                )

                _enh_kwargs = dict(
                    agents=[self.editor_agent],
                    tasks=[enhancement_task],
                    process=Process.sequential,
                    verbose=True,
                )
                try:
                    enhancement_crew = Crew(knowledge=enhancement_crew_knowledge, **_enh_kwargs)
                except TypeError:
                    enhancement_crew = Crew(**_enh_kwargs)
                    attach_knowledge_to_crew(enhancement_crew, enhancement_crew_knowledge)
                
                print("\n=== Running AI Enhancement ===")
                # We no longer need a complex input dictionary, as everything is in the task description.
                enhancement_result = enhancement_crew.kickoff()

                # --- END: MODIFIED CODE ---

                result_raw = enhancement_result.raw.strip()
                if result_raw.startswith("```"):
                    # Handle ```json ... ``` or ``` ... ```
                    if result_raw.startswith("```json"):
                        result_raw = result_raw.split("```json", 1)[1]
                    else:
                        result_raw = result_raw.split("```", 1)[1]
                    result_raw = result_raw.rsplit("```", 1)[0].strip()
                
                try:
                    # With the new prompt, the output should be more reliable.
                    # We can simplify the processing.
                    enhanced_result = json.loads(result_raw)
                    
                    # Validate the output has the required structure
                    if 'revised_survey' not in enhanced_result or 'questions' not in enhanced_result['revised_survey']:
                         raise ValueError("The AI's response was missing the required 'revised_survey' structure.")

                    self.enhanced_dict = enhanced_result

                    print("\n=== Enhancement Complete ===")
                    self._print_survey_summary(self.enhanced_dict)

                except json.JSONDecodeError as e:
                    print(f"\nError: Failed to parse the AI's JSON response. The original survey will be retained.")
                    print(f"Parsing Error: {e}")
                    print(f"Raw output from AI:\n---\n{result_raw}\n---")
                except ValueError as e:
                    print(f"\nError: {e}. The original survey will be retained.")
                except Exception as e:
                    print(f"\nAn unexpected error occurred: {e}. The original survey will be retained.")

            elif choice == "3":
                self._manual_edit()

            elif choice == "4":
                filename = input("Enter filename to save (default: enhanced_survey.json): ") or "enhanced_survey.json"
                with open(filename, 'w') as f:
                    json.dump(self.enhanced_dict, f, indent=2)
                print(f"Survey saved to {filename}")

            elif choice == "5":
                self._deploy_to_qualtrics() 

            elif choice == "6":
                print("Returning to Main Menu...") 
                return 

            else:
                print("Invalid choice. Please enter a number from 1 to 6.")

            iteration += 1

    def _manual_edit(self):
        """Allow manual editing of the survey JSON with improved editor detection"""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as tf:
            json.dump(self.enhanced_dict, tf, indent=2)
            temp_filename = tf.name

        editor = None
        if 'EDITOR' in os.environ:
            editor = os.environ['EDITOR']
        elif 'VISUAL' in os.environ:
            editor = os.environ['VISUAL']
        else:
            if os.name == 'nt':
                if shutil.which('notepad++'):
                    editor = 'notepad++'
                elif shutil.which('code'):
                    editor = 'code'
                else:
                    editor = 'notepad'
            else:
                if shutil.which('nano'):
                    editor = 'nano'
                elif shutil.which('vim'):
                    editor = 'vim'
                elif shutil.which('vi'):
                    editor = 'vi'
                elif shutil.which('code'):
                    editor = 'code'
                elif shutil.which('gedit'):
                    editor = 'gedit'
                else:
                    editor = 'vi'

        print(f"\nOpening temporary file {temp_filename} with {editor}...")
        print("Edit the JSON file, save, and close the editor to continue.")

        try:
            if editor == 'code':
                subprocess.run([editor, '--wait', temp_filename])
            else:
                os.system(f'{editor} "{temp_filename}"')

            with open(temp_filename, 'r') as f:
                modified_json = f.read()

            try:
                modified_dict = json.loads(modified_json)

                if 'revised_survey' in modified_dict:
                    revised = modified_dict['revised_survey']
                    if not isinstance(revised, dict):
                        raise ValueError("revised_survey must be a dictionary")
                    if 'theme' not in revised:
                        raise ValueError("Survey must have a 'theme' field")
                    if 'questions' not in revised or not isinstance(revised['questions'], list):
                        raise ValueError("Survey must have a 'questions' array")

                self.enhanced_dict = modified_dict
                print("Successfully updated survey from your edits.")
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON. Your edits were not applied: {e}")
                print("The original survey will be retained.")
                retry = input("Would you like to try editing again? (y/n): ").lower()
                if retry == 'y':
                    self._manual_edit()
            except ValueError as e:
                print(f"Error: {e}")
                print("The original survey will be retained.")
                retry = input("Would you like to try editing again? (y/n): ").lower()
                if retry == 'y':
                    self._manual_edit()

        except Exception as e:
            print(f"Error during manual editing: {e}")

        finally:
            try:
                os.unlink(temp_filename)
            except:
                pass

    def _deploy_to_qualtrics(self):
        """
        Deploy the survey to Qualtrics and optionally MTurk.
        This method now returns nothing, allowing the calling loop to continue.
        """
        print("\n=== Deploying Survey to Qualtrics ===")
        print(json.dumps(self.enhanced_dict, indent=2, ensure_ascii=False))

        if 'QualtricsAndMTurkAutomation' not in globals() and not hasattr(self, 'qualtrics_client'):
            print("Qualtrics deployment functionality not available.")
            print("Please ensure the QualtricsAndMTurkAutomation class is defined or implement it.")
            save_option = input("Would you like to save the survey to a file instead? (y/n): ").lower()
            if save_option == 'y':
                filename = input("Enter filename to save (default: enhanced_survey.json): ") or "enhanced_survey.json"
                with open(filename, 'w') as f:
                    json.dump(self.enhanced_dict, f, indent=2)
                print(f"Survey saved to {filename}")
            return

        try:
            qualtrics_payload = survey_dict_to_qualtrics_payload(self.enhanced_dict)
        except NameError:
            print("Function survey_dict_to_qualtrics_payload not found.")
            print("Please ensure the function is defined before attempting deployment.")
            return

        print("▶▶▶ [Debug] qualtrics_payload =")
        print(json.dumps(qualtrics_payload, indent=2, ensure_ascii=False))

        use_mturk = input("Would you like to also create an MTurk HIT? (y/n): ").lower() == 'y'

        hit_config = None
        if use_mturk:
            survey_meta = self.enhanced_dict.get("revised_survey", {})
            hit_config = {
                'Title': f'Complete a survey on {survey_meta.get("theme", "research topic")}',
                'Description': survey_meta.get("purpose", "Complete a short research survey"),
                'Keywords': 'survey, research, feedback',
                'Reward': input("Enter reward amount (default: 0.75): ") or '0.75',
                'MaxAssignments': int(input("Enter max participants (default: 100): ") or '100'),
                'LifetimeInSeconds': 86400,
                'AssignmentDurationInSeconds': 1800,
                'AutoApprovalDelayInSeconds': 86400,
                'QualificationRequirements': []
            }

        try:
            if 'QualtricsAndMTurkAutomation' in globals():
                automation = QualtricsAndMTurkAutomation()

                if use_mturk:
                    results = automation.run(qualtrics_payload, hit_config)
                    print("\nDeployment Results:")
                    print(f"Survey ID: {results['survey_id']}")
                    print(f"Survey Link: {results['survey_link']}")
                    print(f"HIT ID: {results['hit_id']}")
                    # No return here
                else:

                    survey_id = automation.qualtrics.create_survey(
                        qualtrics_payload.get("SurveyName", "New Survey"),
                        qualtrics_payload
                    )

                    questions = []
                    for qid, qobj in qualtrics_payload.get("Questions", {}).items():

                        num = re.sub(r"\D+", "", qid)
                        real_qid = f"QID{num}"

                        q_data = {
                            "question_id":   real_qid,
                            "QuestionID":    real_qid,
                            "QuestionText":  qobj["QuestionText"],
                            "QuestionType":  qobj["QuestionType"],
                            "DataExportTag": real_qid,
                            "Configuration": {"QuestionDescriptionOption": "UseText"},
                            "Validation":    {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
                            "Selector":      qobj["Selector"]
                        }
                        if "SubSelector" in qobj:
                            q_data["SubSelector"] = qobj["SubSelector"]
                        if "Choices" in qobj:
                            q_data["Choices"] = qobj["Choices"]
                        questions.append(q_data)

                    comp_qid = f"QID{len(questions)+1}"
                    questions.append({
                        "question_id":   comp_qid,
                        "QuestionID":    comp_qid,
                        "QuestionText":  "Thank you for completing the survey!\\nYour completion code is: ${e://Field/ResponseID}",
                        "QuestionType":  "DB",   # Descriptive Text
                        "Selector":      "TB",   # Text/Graphic Block
                        "Configuration": {"QuestionDescriptionOption": "UseText"},
                        "Validation":    {"Settings": {"ForceResponse": "OFF", "Type": "None"}}
                    })

                    automation.qualtrics.add_questions(survey_id, questions)
                    automation.qualtrics.activate_survey(survey_id)
                    survey_link = automation.qualtrics.create_distribution_link(survey_id)

                    print("\nDeployment Results:")
                    print(f"Survey ID: {survey_id}")
                    print(f"Survey Link: {survey_link}")
                    # No return here
            else:
                print("QualtricsAndMTurkAutomation class not found.")
                print("Please implement it or ensure it's defined in the global scope.")

        except Exception as e:
            print(f"Error during deployment: {e}")
            # No return here

    def _print_survey_summary(self, survey_dict):
        """Print a summary of the survey in a readable format"""
        print("\n=== SURVEY SUMMARY ===")

        try:
            if 'revised_survey' in survey_dict:
                revised = survey_dict.get('revised_survey', {})
                print(f"Theme:   {revised.get('theme', 'N/A')}")
                print(f"Purpose: {revised.get('purpose', 'N/A')}\\n")

                for q in revised.get('questions', []):
                    if isinstance(q, dict):
                        qid = q.get('question_id', q.get('id', 'unknown'))
                        qtext = q.get('question_text', q.get('question', q.get('text', 'N/A')))
                        print(f"Q{qid}: {qtext}")

                        input_config = q.get('input_config', {})
                        if isinstance(input_config, dict):
                            opts = input_config.get('options', [])
                            if opts:
                                print("  Options:")
                                for o in opts:
                                    print(f"    - {o}")
                            elif 'min' in input_config and 'max' in input_config:
                                print(f"  Slider: {input_config.get('min', 0)} to {input_config.get('max', 100)}, " +
                                      f"step: {input_config.get('step', 1)}")
                            elif input_config.get('multiline') is not None:
                                multiline = "multiline" if input_config.get('multiline') else "single line"
                                print(f"  Text input: {multiline}")
                        print()

            if 'explanations' in survey_dict:
                print("=== Explanations ===")
                explanations = survey_dict['explanations']
                if isinstance(explanations, dict):
                    for key, explanation in explanations.items():
                        print(f"{key}: {explanation}")
                print()

            elif 'original_with_comments' in survey_dict:
                annotated = survey_dict.get('original_with_comments', {})
                survey = annotated.get('survey', {})
                print("=== Original Survey (with comments) ===")
                print(f"Theme: {survey.get('theme', 'N/A')}")
                print(f"Purpose: {survey.get('purpose', 'N/A')}\\n")

                comments = annotated.get('question_comments', [])
                for q in survey.get('questions', []):
                    if isinstance(q, dict):
                        qid = q.get('question_id', q.get('id', 'unknown'))
                        qtext = q.get('question_text', q.get('question', q.get('text', 'N/A')))
                        print(f"Question {qid}: {qtext}")

                        comment = next((c.get('comment', '') for c in comments if c.get('question_id') == qid), None)
                        if comment:
                            print(f"  Comment: {comment}")

                        input_config = q.get('input_config', {})
                        if isinstance(input_config, dict):
                            opts = input_config.get('options', [])
                            if opts:
                                print("  Options:")
                                for o in opts:
                                    print(f"    - {o}")
                        print()

                overall = annotated.get('overall_comment')
                if overall:
                    print(f"Overall comment: {overall}\\n")

            elif 'survey' in survey_dict:
                survey = survey_dict.get('survey', {})
                print(f"Theme:   {survey.get('theme', survey.get('title', 'N/A'))}")
                print(f"Purpose: {survey.get('purpose', survey.get('description', 'N/A'))}\\n")

                for q in survey.get('questions', []):
                    if isinstance(q, dict):
                        qid = q.get('question_id', q.get('id', 'unknown'))
                        qtext = q.get('question_text', q.get('question', q.get('text', 'N/A')))
                        print(f"Q{qid}: {qtext}")

                        if 'options' in q and isinstance(q['options'], list):
                            opts = q['options']
                            print("  Options:")
                            for o in opts:
                                if isinstance(o, dict) and ('label' in o or 'text' in o):
                                    print(f"    - {o.get('label', o.get('text', ''))}")
                                else:
                                    print(f"    - {o}")
                        print()

            elif 'title' in survey_dict and 'fields' in survey_dict:
                print(f"Title: {survey_dict.get('title', 'N/A')}\\n")
                for i, field in enumerate(survey_dict.get('fields', [])):
                    print(f"Q{i+1}: {field.get('title', 'N/A')}")
                    opts = field.get('options', [])
                    if opts:
                        print("  Options:")
                        for o in opts:
                            print(f"    - {o}")
                    print()

            elif 'theme' in survey_dict and 'questions' in survey_dict:
                print(f"Theme: {survey_dict.get('theme', 'N/A')}")
                if 'purpose' in survey_dict:
                    print(f"Purpose: {survey_dict.get('purpose', 'N/A')}")
                print()

                for q in survey_dict.get('questions', []):
                    if isinstance(q, dict):
                        qid = q.get('question_id', q.get('id', 'unknown'))
                        qtext = q.get('question_text', q.get('question', q.get('text', 'N/A')))
                        print(f"Q{qid}: {qtext}")
                        if 'options' in q and isinstance(q['options'], list):
                            opts = q['options']
                            print("  Options:")
                            for o in opts:
                                if isinstance(o, dict) and ('label' in o or 'text' in o):
                                    print(f"    - {o.get('label', o.get('text', ''))}")
                                else:
                                    print(f"    - {o}")
                        print()
            else:
                print("Survey structure doesn't match expected format. Raw structure:")
                print(json.dumps(survey_dict, indent=2))

        except Exception as e:
            print(f"Error parsing survey structure: {str(e)}")
            print("Raw survey data:")
            print(json.dumps(survey_dict, indent=2))

# ========== Survey to Qualtrics Conversion ==========
def survey_dict_to_qualtrics_payload(survey_dict: dict) -> dict:
    """
    将自定义 survey_dict 转成 Qualtrics v3 API 的 survey-definitions payload
    支持题型：multiple_choice, single_choice, slider, text_input
    """
    survey_meta = survey_dict.get("revised_survey", survey_dict.get("survey", {}))
    payload = {
        "SurveyName":      survey_meta.get("theme", "New Survey"),
        "Language":        "EN",
        "ProjectCategory": "CORE",
        "Questions":       {}
    }

    for q in survey_meta.get("questions", []):
        raw_id = q["question_id"]
        num    = re.sub(r'\D+', '', raw_id)
        qid    = f"QID{num}"

        qt  = q["question_text"]
        it  = q["input_type"]
        cfg = q["input_config"]

        qobj = {
            "QuestionText":      qt,
            "DataExportTag":     qid,
            "Configuration":     {"QuestionDescriptionOption": "UseText"},
            "Validation":        {"Settings": {"ForceResponse": "OFF", "Type": "None"}}
        }

        if it in ("multiple_choice", "single_choice"):
            choices = {}
            for opt in cfg.get("options", []):
                if "=" in opt:
                    idx, txt = opt.split("=", 1)
                    idx, txt = idx.strip(), txt.strip()
                else:
                    idx = str(len(choices) + 1)
                    txt = opt.strip()
                choices[idx] = {"Display": txt}

            qobj.update({
                "QuestionType": "MC",
                "Selector":     "SAVR" if it == "multiple_choice" else "SINGLE",
                "SubSelector":  "TX",
                "Choices":      choices
            })


        elif it == "slider":
            start = cfg.get("min", cfg.get("start", 0))
            end   = cfg.get("max", cfg.get("end", 100))
            step  = cfg.get("step", cfg.get("stepSize", 1))

            qobj.update({
                "QuestionType": "SL",
                "Selector":     "Slider",
                "SubSelector":  "SL",
                "SliderStart":  start,
                "SliderEnd":    end,
                "SliderStep":   step
            })


        elif it == "text_input":
            qobj.update({
                "QuestionType": "TE",
                "Selector":     "ML"               })

        else:
            raise ValueError(f"Unsupported input_type: {it!r}")

        payload["Questions"][qid] = qobj

    return payload

# ========== Qualtrics API Client ==========
class QualtricsClient:
    """Handles all Qualtrics API interactions"""

    def __init__(self):
        """Initialize Qualtrics API client with credentials from .env file"""
        print(f"Current working directory: {os.getcwd()}")

        if os.path.exists('.env'):
            print("Found .env file in current directory")
        else:
            print("WARNING: No .env file found in current directory!")

        load_dotenv(verbose=True)

        self.api_token = os.getenv('QUALTRICS_API_TOKEN')
        self.data_center = os.getenv('QUALTRICS_DATA_CENTER')
        self.directory_id = os.getenv('QUALTRICS_DIRECTORY_ID')

        if self.api_token:
            token_length = len(self.api_token)
            masked_token = self.api_token[:4] + '*' * (token_length - 8) + self.api_token[-4:] if token_length > 8 else "****"
            print(f"API Token loaded (masked): {masked_token}")
        else:
            print("WARNING: No API token found in environment variables!")

        if self.data_center:
            print(f"Data center: {self.data_center}")
        else:
            print("WARNING: No data center found in environment variables!")

        if not self.api_token or not self.data_center:
            raise ValueError("Missing Qualtrics API credentials in .env file")

        self.base_url = f"https://{self.data_center}.qualtrics.com/API/v3/"
        self.headers = {
            "X-API-Token": self.api_token,
            "Content-Type": "application/json"
        }

        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.verify = certifi.where()

        print("Testing Qualtrics API connection...")
        try:
            test_url = f"{self.base_url}whoami"
            response = self.session.get(test_url, headers=self.headers, verify=self.verify, timeout=10)
            if response.status_code == 200:
                user_info = response.json()["result"]
                print(f"Connection successful! Authenticated as: {user_info.get('firstName', '')} {user_info.get('lastName', '')}")
            else:
                print(f"Connection test failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error testing connection: {str(e)}")

    def get_survey_questions(self, survey_id: str) -> dict:
        """
        Retrieves the questions and their export tags for a given survey.
        
        Args:
            survey_id (str): The ID of the survey.
            
        Returns:
            dict: A dictionary mapping question export tags (e.g., 'QID1') to question text.
        """
        print(f"Retrieving question definitions for survey: {survey_id}")
        url = f"{self.base_url}survey-definitions/{survey_id}"
        response = self.session.get(url, headers=self.headers, verify=self.verify, timeout=10)
        response.raise_for_status()
        
        survey_definition = response.json().get("result", {})
        questions = survey_definition.get("Questions", {})
        
        question_map = {}
        for qid, q_data in questions.items():
            export_tag = q_data.get("DataExportTag", qid)
            question_text = q_data.get("QuestionText", "N/A")
            # Remove HTML tags from question text for cleaner CSV
            clean_text = re.sub('<[^<]+?>', '', question_text).strip()
            question_map[export_tag] = clean_text
            
        print(f"Found {len(question_map)} questions.")
        return question_map

    def get_survey_responses(self, survey_id: str, file_format="csv") -> pd.DataFrame:
        """
        Exports and downloads survey responses from Qualtrics.
        
        Args:
            survey_id (str): The ID of the survey to get responses from.
            file_format (str): The format for the exported file ('csv', 'json', etc.).
            
        Returns:
            pd.DataFrame: A pandas DataFrame containing the survey responses.
        """
        export_url = f"{self.base_url}surveys/{survey_id}/export-responses"

        # Step 1: Request the export
        print(f"Starting response export for survey: {survey_id}")
        resp = self.session.post(
            export_url,
            headers=self.headers,
            json={"format": file_format, "useLabels": True},
            verify=self.verify, timeout=10
        )
        resp.raise_for_status()
        progress_id = resp.json()["result"]["progressId"]

        # Step 2: Poll for completion
        status = ""
        while status != "complete" and status != "failed":
            check_url = f"{export_url}/{progress_id}"
            check = self.session.get(
                check_url,
                headers=self.headers,
                verify=self.verify, timeout=10
            )
            check.raise_for_status()
            result = check.json()["result"]
            status = result["status"]
            print(f"Export status: {status} ({result.get('percentComplete', 0)}%)")
            if status == 'failed':
                raise Exception(f"Export failed. Reason: {result.get('errorMessage')}")
            time.sleep(2)

        # Step 3: Download the file
        file_id = result["fileId"]
        download_url = f"{export_url}/{file_id}/file"
        print(f"Downloading file with ID: {file_id}")
        dl = self.session.get(
            download_url,
            headers=self.headers,
            verify=self.verify, timeout=60
        )
        dl.raise_for_status()

        # Step 4: Unzip and read into DataFrame
        with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
            fname = next((n for n in zf.namelist() if n.endswith(f".{file_format}")), None)
            if not fname:
                raise FileNotFoundError(f"No '.{file_format}' file found in the downloaded zip archive.")

            with zf.open(fname) as f:
                # Qualtrics CSVs have 2 header rows, we skip the second one which contains question details
                df = pd.read_csv(f, skiprows=[1])

        print(f"Successfully downloaded and parsed {len(df)} records.")
        return df

    def create_survey(self, survey_name, survey_template=None):
        """
        Create a new survey in Qualtrics
        
        Args:
            survey_name (str): Name of the survey
            survey_template (dict, optional): Survey template JSON
            
        Returns:
            str: Survey ID of the created survey
        """
        print(f"Creating survey: {survey_name}")

        if not survey_template:
            # The default survey payload has been removed as per the request.
            # You would typically raise an error or handle this case.
            raise ValueError("A survey_template must be provided to create a survey.")

        survey_payload = survey_template
        if "ProjectCategory" not in survey_payload:
            survey_payload["ProjectCategory"] = "CORE"

        url = f"{self.base_url}survey-definitions"
        payload = json.dumps(survey_payload)

        print(f"Sending payload to Qualtrics: {payload[:200]}...")

        response = requests.post(url, headers=self.headers, data=payload)

        if response.status_code != 200:
            print(f"Error response: {response.text}")
            raise Exception(f"Failed to create survey: {response.text}")

        result = response.json()
        survey_id = result["result"]["SurveyID"]
        print(f"Survey created successfully with ID: {survey_id}")

        return survey_id

    def add_questions(self, survey_id: str, questions: List[dict]):

        for q in questions:
            q_payload = {
                 "QuestionID":   q["QuestionID"],
                 "QuestionText": q["QuestionText"],
                 "QuestionType": q["QuestionType"],
                 "DataExportTag": q["QuestionID"],
                 "Configuration": {"QuestionDescriptionOption": "UseText"},
                 "Validation":    {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
             }
            if "Selector" in q:
                q_payload["Selector"] = q["Selector"]
            if "SubSelector" in q:
                q_payload["SubSelector"] = q["SubSelector"]
            if "Choices" in q:
                q_payload["Choices"] = q["Choices"]

            url = f"{self.base_url}survey-definitions/{survey_id}/questions"
            resp = requests.post(url, headers=self.headers, json=q_payload)
            print(f"POST questions → {resp.status_code}", resp.json())

    def add_block(self, survey_id: str, block_payload: dict):
        url = f"{self.base_url}survey-definitions/{survey_id}/blocks"
        resp = requests.post(url, headers=self.headers, json=block_payload)
        print(f"POST blocks → {resp.status_code}", resp.json())

    def update_flow(self, survey_id: str, flow_payload: dict):
        url = f"{self.base_url}survey-definitions/{survey_id}/flow"
        resp = requests.put(url, headers=self.headers, json=flow_payload)
        print("PUT flow →", resp.status_code, resp.json())

    def activate_survey(self, survey_id):
        """
        Activate a survey to make it available for distribution
        """
        print(f"Activating survey: {survey_id}")

        url = f"{self.base_url}surveys/{survey_id}"
        payload = json.dumps({"isActive": True})

        response = requests.put(url, headers=self.headers, data=payload)

        if response.status_code != 200:
            raise Exception(f"Failed to activate survey: {response.text}")

        print(f"Survey activated successfully")
        return True

    def create_distribution_link(self, survey_id, link_type="Anonymous"):
        """
        Create a distribution link for a survey
        """
        print(f"Creating distribution link for survey: {survey_id}")

        if link_type == "Anonymous":
            survey_link = f"https://{self.data_center}.qualtrics.com/jfe/form/{survey_id}"
            print(f"Anonymous survey link created: {survey_link}")
            return survey_link
        else:
            raise NotImplementedError(f"Distribution type '{link_type}' is not yet supported")

# ========== MTurk API Client ==========
class MTurkClient:
    """Handles all MTurk API interactions"""
    def __init__(self,
                 aws_access_key_id: str = None,
                 aws_secret_access_key: str = None,
                 use_sandbox: bool = True):
        """
        Initialize MTurk client
        
        Args:
            aws_access_key_id: Optional override for AWS access key
            aws_secret_access_key: Optional override for AWS secret key
            use_sandbox: Boolean for using sandbox (defaults to True for safety)
        """
        load_dotenv()

        self.aws_access_key_id     = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')

        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("Missing AWS credentials in .env file or constructor parameters")

        self.use_sandbox = use_sandbox

        region = os.getenv('AWS_REGION', 'us-east-1')
        endpoint = (
            'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
            if self.use_sandbox else
            'https://mturk-requester.us-east-1.amazonaws.com'
        )

        try:
            self.client = boto3.client(
                'mturk',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=region,
                endpoint_url=endpoint
            )
            print(f"MTurk client initialized in {'Sandbox' if self.use_sandbox else 'Production'} mode")
            self.get_account_balance()
        except Exception as e:
            print(f"Error initializing MTurk client: {str(e)}")
            self.client = None
            self.connection_error = str(e)

    def get_account_balance(self):
        """Get the available MTurk account balance"""
        if not self.client:
            print(f"Cannot check balance: {self.connection_error}")
            return 0.0

        try:
            response = self.client.get_account_balance()
            balance = response['AvailableBalance']
            print(f"MTurk account balance: ${balance}")
            return float(balance)
        except Exception as e:
            print(f"Error checking balance: {str(e)}")
            return 0.0

    def create_hit_with_survey_link(self, survey_link, hit_config=None):
        """
        Create an MTurk HIT with a link to a Qualtrics survey
        
        Args:
            survey_link (str): URL to the Qualtrics survey
            hit_config (dict, optional): Custom configuration for the HIT
            
        Returns:
            str: HIT ID
        """
        print("Creating MTurk HIT with survey link")

        if not hit_config:
            hit_config = {
                'Title': 'Complete a short survey',
                'Description': 'We need your input for a quick survey that should take less than 10 minutes',
                'Keywords': 'survey, research, opinion, feedback',
                'Reward': '0.50',
                'MaxAssignments': 100,
                'LifetimeInSeconds': 86400,
                'AssignmentDurationInSeconds': 1800,
                'AutoApprovalDelayInSeconds': 86400,
                'QualificationRequirements': []
            }

        question_html = f"""
        <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
            <HTMLContent><![CDATA[
                <!DOCTYPE html>
                <html>
                <head>
                    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
                    <script type='text/javascript' src='https://s3.amazonaws.com/mturk-public/externalHIT_v1.js'></script>
                </head>
                <body>
                    <form name='mturk_form' method='post' id='mturk_form' action='https://www.mturk.com/mturk/externalSubmit'>
                        <input type='hidden' value='' name='assignmentId' id='assignmentId'/>
                        <h1>Survey Task</h1>
                        <p>Please complete the survey at the following link:</p>
                        <p><a href='{survey_link}' target='_blank'>{survey_link}</a></p>
                        <p>After completing the survey, you will receive a completion code. Enter the code below:</p>
                        <p><input type='text' name='completion_code' id='completion_code' size='40'/></p>
                        <p><input type='submit' id='submitButton' value='Submit' /></p>
                    </form>
                    <script language='Javascript'>
                        turkSetAssignmentID();
                    </script>
                </body>
                </html>
            ]]></HTMLContent>
            <FrameHeight>600</FrameHeight>
        </HTMLQuestion>
        """

        response = self.client.create_hit(
            Title=hit_config['Title'],
            Description=hit_config['Description'],
            Keywords=hit_config['Keywords'],
            Reward=hit_config['Reward'],
            MaxAssignments=hit_config['MaxAssignments'],
            LifetimeInSeconds=hit_config['LifetimeInSeconds'],
            AssignmentDurationInSeconds=hit_config['AssignmentDurationInSeconds'],
            AutoApprovalDelayInSeconds=hit_config['AutoApprovalDelayInSeconds'],
            Question=question_html,
            QualificationRequirements=hit_config['QualificationRequirements']
        )

        hit_id = response['HIT']['HITId']
        hit_type_id = response['HIT']['HITTypeId']

        print(f"HIT created successfully with ID: {hit_id}")

        if self.use_sandbox:
            worker_url = f"https://workersandbox.mturk.com/mturk/preview?groupId={hit_type_id}"
        else:
            worker_url = f"https://worker.mturk.com/mturk/preview?groupId={hit_type_id}"

        print(f"Workers can access the HIT at: {worker_url}")

        return hit_id

    def get_hit_assignments(self, hit_id):
        """
        Get all assignments for a HIT
        
        Args:
            hit_id (str): ID of the HIT
            
        Returns:
            list: List of assignment dictionaries
        """
        print(f"Getting assignments for HIT: {hit_id}")

        all_assignments = []
        next_token = None

        while True:
            if next_token:
                response = self.client.list_assignments_for_hit(
                    HITId=hit_id,
                    NextToken=next_token,
                    MaxResults=100
                )
            else:
                response = self.client.list_assignments_for_hit(
                    HITId=hit_id,
                    MaxResults=100
                )

            all_assignments.extend(response['Assignments'])

            if 'NextToken' in response:
                next_token = response['NextToken']
            else:
                break

        print(f"Found {len(all_assignments)} assignments")
        return all_assignments

    def approve_assignments(self, assignments, feedback=None):
        """
        Approve multiple assignments
        
        Args:
            assignments (list): List of assignment dictionaries or IDs
            feedback (str, optional): Feedback to workers
            
        Returns:
            int: Number of successfully approved assignments
        """
        approved_count = 0

        for assignment in assignments:
            assignment_id = assignment['AssignmentId'] if isinstance(assignment, dict) else assignment

            try:
                self.client.approve_assignment(
                    AssignmentId=assignment_id,
                    RequesterFeedback=feedback if feedback else "Thank you for your participation!"
                )
                approved_count += 1
            except Exception as e:
                print(f"Error approving assignment {assignment_id}: {str(e)}")

        print(f"Successfully approved {approved_count} assignments")
        return approved_count

    def delete_hit(self, hit_id):
        """
        Delete a HIT
        
        Args:
            hit_id (str): ID of the HIT to delete
            
        Returns:
            bool: True if successful
        """
        try:
            hit = self.client.get_hit(HITId=hit_id)
            status = hit['HIT']['HITStatus']

            if status == 'Reviewable':
                self.client.delete_hit(HITId=hit_id)
                print(f"HIT {hit_id} deleted successfully")
                return True
            elif status == 'Assignable':
                self.client.update_expiration_for_hit(
                    HITId=hit_id,
                    ExpireAt=datetime(2015, 1, 1)
                )
                time.sleep(1)
                self.client.delete_hit(HITId=hit_id)
                print(f"HIT {hit_id} expired and deleted successfully")
                return True
            else:
                print(f"Cannot delete HIT {hit_id}, status is {status}")
                return False

        except Exception as e:
            print(f"Error deleting HIT {hit_id}: {str(e)}")
            return False

# ========== Qualtrics and MTurk Integration ==========
class QualtricsAndMTurkAutomation:
    """Handles the automation of creating Qualtrics surveys and MTurk HITs"""

    def __init__(self, mturk_client: Optional[MTurkClient] = None):
        load_dotenv()
        self.qualtrics = QualtricsClient()
        self.mturk     = mturk_client or MTurkClient()

    def run(self, survey_payload: dict, hit_config: dict) -> dict:
        """
        Run the automation to create a Qualtrics survey and MTurk HIT
        
        Args:
            survey_payload (dict): Survey definition for Qualtrics
            hit_config (dict, optional): Configuration for MTurk HIT
            
        Returns:
            dict: Results including survey ID, survey link, and HIT ID
        """
        survey_id = self.qualtrics.create_survey(
            survey_name=survey_payload["SurveyName"],
            survey_template=survey_payload
        )

        questions = []
        for qid, qobj in survey_payload["Questions"].items():
            num = qid.lstrip("Q")
            real_qid = f"QID{num}"

            q_data = {
                "question_id":   real_qid,
                "QuestionID":    real_qid,
                "QuestionText":  qobj["QuestionText"],
                "QuestionType":  qobj["QuestionType"],
                "Selector":      qobj["Selector"]
            }

            if "SubSelector" in qobj:
                q_data["SubSelector"] = qobj["SubSelector"]

            if "Choices" in qobj:
                q_data["Choices"] = qobj["Choices"]

            questions.append(q_data)


        comp_qid = f"QID{len(questions)+1}"
        questions.append({
            "question_id":   comp_qid,
            "QuestionID":    comp_qid,
            "QuestionText":  "Thank you for completing the survey!\\nYour completion code is: ${e://Field/ResponseID}",
            "QuestionType":  "DB",  # Descriptive Text
            "Selector":      "TB",  # Text/Graphic Block
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "Validation":    {"Settings": {"ForceResponse": "OFF", "Type": "None"}}
        })

        self.qualtrics.add_questions(survey_id, questions)
        self.qualtrics.activate_survey(survey_id)
        survey_link = self.qualtrics.create_distribution_link(survey_id)

        hit_id = self.mturk.create_hit_with_survey_link(survey_link, hit_config)

        return {
            "survey_id":   survey_id,
            "survey_link": survey_link,
            "hit_id":      hit_id
        }
    
    def _format_df_with_interleaved_questions(self, responses_df: pd.DataFrame, question_map: dict) -> pd.DataFrame:
        """
        Helper method to reformat a DataFrame to have interleaved question and answer columns.
        
        Args:
            responses_df (pd.DataFrame): The DataFrame with survey responses.
            question_map (dict): A dictionary mapping question export tags to question text.
            
        Returns:
            pd.DataFrame: A reformatted DataFrame.
        """
        if not question_map or responses_df.empty:
            return responses_df

        question_cols_in_df = [col for col in responses_df.columns if col in question_map]
        meta_cols = [col for col in responses_df.columns if col not in question_cols_in_df]
        
        try:
            # Sort columns numerically (e.g., QID1, QID2, QID10)
            question_cols_in_df.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        except (IndexError, TypeError):
            # Fallback to alphanumeric sort if no numbers are found in tags
            question_cols_in_df.sort()

        # Build the list of new interleaved columns and their data
        interleaved_data = {}
        for i, q_col in enumerate(question_cols_in_df, 1):
            interleaved_data[f'Question {i}'] = question_map.get(q_col, "Unknown Question")
            interleaved_data[f'Answer {i}'] = responses_df[q_col]
        
        # Create the new DataFrame from this data
        final_df = pd.DataFrame(interleaved_data)

        # Join the metadata back in, using the original index to ensure alignment
        # This will add metadata columns at the end.
        final_df = final_df.join(responses_df[meta_cols])

        return final_df

    def collect_and_process_results(self, survey_id: str, hit_id: str, auto_approve: bool = False):
        """
        Fetches data from Qualtrics and MTurk, merges them, formats with questions, and optionally approves.
        
        Args:
            survey_id (str): The Qualtrics Survey ID.
            hit_id (str): The MTurk HIT ID.
            auto_approve (bool): Whether to automatically approve valid assignments.
            
        Returns:
            dict: A dictionary containing the formatted merged responses and other processing info.
        """
        print(f"Collecting results for Survey ID: {survey_id} and HIT ID: {hit_id}")

        # 1. Get Qualtrics Questions
        try:
            question_map = self.qualtrics.get_survey_questions(survey_id)
        except Exception as e:
            print(f"Warning: Could not retrieve survey questions: {e}. The output will not contain question text.")
            question_map = {}

        # 2. Get Qualtrics Responses
        qualtrics_df = self.qualtrics.get_survey_responses(survey_id)

        # 3. Get MTurk Assignments
        assignments = self.mturk.get_hit_assignments(hit_id)
        if not assignments:
            print("No MTurk assignments found for this HIT. Formatting Qualtrics data only.")
            formatted_df = self._format_df_with_interleaved_questions(qualtrics_df, question_map)
            return {
                "responses": formatted_df,
                "assignments": [],
                "approved_count": 0
            }

        # 4. Process MTurk Assignments to extract completion codes
        mturk_data = []
        for a in assignments:
            try:
                answer_xml = a['Answer']
                root = ET.fromstring(answer_xml)
                namespace_match = re.match(r'\{.*\}', root.tag)
                namespace = namespace_match.group(0) if namespace_match else ''
                
                code_element = root.find(f".//{namespace}Answer[{namespace}QuestionIdentifier='completion_code']/{namespace}FreeText")
                completion_code = code_element.text if code_element is not None else None

                mturk_data.append({
                    'AssignmentId': a['AssignmentId'],
                    'WorkerId': a['WorkerId'],
                    'SubmitTime': a['SubmitTime'],
                    'AssignmentStatus': a['AssignmentStatus'],
                    'completion_code': completion_code
                })
            except Exception as e:
                print(f"Warning: Could not parse answer for assignment {a['AssignmentId']}: {e}")

        if not mturk_data:
            print("Could not parse any MTurk assignments. Returning formatted Qualtrics data only.")
            formatted_df = self._format_df_with_interleaved_questions(qualtrics_df, question_map)
            return {
                "responses": formatted_df,
                "assignments": assignments,
                "approved_count": 0
            }

        mturk_df = pd.DataFrame(mturk_data)
        print("\nMTurk Data Preview:")
        print(mturk_df.head())

        # 5. Merge Qualtrics and MTurk data
        if 'ResponseID' not in qualtrics_df.columns:
            raise KeyError("The column 'ResponseID' was not found in the Qualtrics data. Cannot merge with MTurk results.")
        if mturk_df.empty or 'completion_code' not in mturk_df.columns:
            print("Warning: No valid completion codes found in MTurk data. Returning only formatted Qualtrics data.")
            formatted_df = self._format_df_with_interleaved_questions(qualtrics_df, question_map)
            return {"responses": formatted_df, "assignments": assignments, "approved_count": 0}

        qualtrics_df['ResponseID'] = qualtrics_df['ResponseID'].astype(str)
        mturk_df['completion_code'] = mturk_df['completion_code'].astype(str)
        
        merged_df = pd.merge(
            qualtrics_df,
            mturk_df,
            left_on='ResponseID',
            right_on='completion_code',
            how='inner'
        )
        print(f"\nSuccessfully merged {len(merged_df)} records from Qualtrics and MTurk.")
        if merged_df.empty:
            print("Warning: No records could be matched between Qualtrics and MTurk based on the completion code.")

        # 6. Format the final DataFrame with interleaved questions and answers
        formatted_df = self._format_df_with_interleaved_questions(merged_df, question_map)

        # 7. Auto-approve assignments if requested
        approved_count = 0
        if auto_approve and not formatted_df.empty:
            assignments_to_approve = formatted_df[formatted_df['AssignmentStatus'] == 'Submitted']['AssignmentId'].tolist()
            if assignments_to_approve:
                print(f"Auto-approving {len(assignments_to_approve)} assignments...")
                approved_count = self.mturk.approve_assignments(assignments_to_approve)
            else:
                print("No assignments in 'Submitted' status to approve.")

        return {
            "responses": formatted_df,
            "assignments": assignments,
            "approved_count": approved_count
        }


# ========== Simulated Data Collection ==========
# This section is added based on run_simulation.py

def collect_simulated_data():
    """
    Collects simulated data for an existing survey by running a simulation.
    This function encapsulates the logic from run_simulation.py and asks the user for file paths.
    """
    print("\n====== Collect Simulated Data from Existing Survey ======")

    print("\nPlease provide the paths for the simulation files.")
    print("Press Enter to use the default value shown in parentheses.")

    default_template_path = "simulate_response/survey_response_template.txt"
    template_path = input(f"Enter the path for the survey response template (default: {default_template_path}): ") or default_template_path

    default_survey_context_path = "simulate_response/test_survey.json"
    survey_context_path = input(f"Enter the path for the survey content JSON file (default: {default_survey_context_path}): ") or default_survey_context_path

    default_participant_csv_path = "simulate_response/participant_pool.csv"
    participant_csv_path = input(f"Enter the path for the participant pool CSV file (default: {default_participant_csv_path}): ") or default_participant_csv_path

    # Check for the existence of required files
    for f_path in [template_path, survey_context_path, participant_csv_path]:
        if not os.path.exists(f_path):
            print(f"\nError: Required file not found: {f_path}")
            print("Please ensure the file exists at the specified path and try again.")
            return

    try:
        # response template
        print(f"Reading survey template from {template_path}...")
        with open(template_path, "r") as f:
            survey_template = f.read()

        # test survey
        print(f"Reading survey context from {survey_context_path}...")
        with open(survey_context_path, "r") as f:
            survey_context_string = f.read()

        simulation_context_dict = None
        survey_json = json.loads(survey_context_string)

        # Check for the 'enhanced.json' format (questions inside 'revised_survey')
        if 'revised_survey' in survey_json and 'questions' in survey_json['revised_survey']:
            print("Detected enhanced survey format. Using 'revised_survey' for simulation.")
            simulation_context_dict = survey_json['revised_survey']
        # Check for the 'test_survey.json' format (questions at the top level)
        elif 'questions' in survey_json:
            print("Detected unenhanced survey format.")
            simulation_context_dict = survey_json
        else:
            # If neither format is found, raise a clear error.
            raise ValueError("Could not find a 'questions' list. The file must have a top-level 'questions' key, or a 'revised_survey' object containing a 'questions' key.")

        survey_context_for_simulation = json.dumps(simulation_context_dict)

        print("Initializing LLM...")
        llm = openai_llm

        # ---- run simulation ----
        print("Running survey simulation...")
        responses_df = run_all_survey_responses_json(
            llm=llm,
            participant_csv_path=participant_csv_path,
            survey_prompt_template=survey_template,
            survey_context=survey_context_for_simulation
        )

        # ---- parse the Response column into a DataFrame qdf ----
        if 'Response' not in responses_df.columns:
            raise KeyError("Expected 'Response' column in simulation output")
        parsed = []
        for resp in responses_df['Response']:
            if isinstance(resp, str):
                try:
                    parsed.append(json.loads(resp))
                except json.JSONDecodeError:
                    # Handle cases where the response is not valid JSON
                    print(f"Warning: Could not parse JSON from LLM response: {resp}")
                    parsed.append({}) # Append empty dict to avoid errors later
            else:
                parsed.append(resp)  # already dict
        qdf = pd.DataFrame(parsed)

        # ---- load question definitions ----
        question_defs = simulation_context_dict.get("questions")
        if not question_defs:
            # This is a fallback check; the logic above should have already handled this.
            raise ValueError("No questions found in the processed survey context JSON.")

        # ---- build the output list ----
        output_list = []
        total_llms = len(qdf)
        for idx, q in enumerate(question_defs):
            q_text = q["question_text"].strip()
            opts   = q.get("input_config", {}).get("options", [])
            choices_str = ", ".join(f"{i+1}-{opt}" for i, opt in enumerate(opts))
            prompt = f"{q_text} Choices: {choices_str}"

            col = f"Q{idx+1}"
            if col not in qdf.columns:
                print(f"Warning: Column '{col}' not found in simulated responses. Skipping question.")
                continue # Skip this question if the column doesn't exist
            answers = qdf[col].tolist()

            output_list.append({
                "Question": prompt,
                "num_llms": total_llms,
                "llm_resp": answers
            })

        # ---- write to JSON file ----
        out_json = "simulated_survey_responses.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)

        # ---- print confirmation + the full JSON for inspection ----
        print(f" Simulation complete. Saved JSON to {out_json}\n")
        print("===== Simulation Output JSON START =====")
        print(json.dumps(output_list, ensure_ascii=False, indent=2))
        print("===== Simulation Output JSON END =====\n")

        debiased_json = "simulated_survey_responses_debiased.json"
        print("Starting debiasing step...")
        run_debias_pipeline(
            input_json=out_json,
            output_json=debiased_json,
            variance_threshold=0.90,
            penalty_weight=15.0,
            lr=1e-3,
            epochs=500,
            embed_model="text-embedding-3-small"
        )
        with open(debiased_json, 'r', encoding='utf-8') as f:
            debiased_data = json.load(f)

        df_debiased = pd.DataFrame(debiased_data)

        csv_path = "simulated_survey_responses_debiased.csv"
        df_debiased.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Debiased data also saved as CSV to {csv_path}")
        print(f"Debias complete. Saved debiased JSON to {debiased_json}")


    except Exception as e:
        print(f"\nAn error occurred during the simulation: {e}")
        import traceback
        traceback.print_exc()

# ========== Main Function for Survey Processing ==========
def generate_research_paper():
    """Orchestrates AI agents to generate a research paper from a CSV file."""
    print("\n====== Generate Research Paper from CSV Data ======")

    # 1. Get CSV file path from user
    csv_path = input("Enter the path to your CSV data file: ")
    if not os.path.exists(csv_path):
        print(f"Error: File not found at '{csv_path}'. Please check the path and try again.")
        return

    print(f"Reading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading or parsing the CSV file: {e}")
        return

    # Prepare data summary for the agent
    data_summary = f"""
    File Name: {os.path.basename(csv_path)}
    Number of Rows: {len(df)}
    Number of Columns: {len(df.columns)}
    Column Names: {', '.join(df.columns)}

    Data Preview (first 5 rows):
    {df.head().to_string()}
    """
    print("\n--- Data Summary ---")
    print(data_summary)

    # 2. Get research hypothesis from user (make it optional)
    print("\n--- Research Hypothesis (Optional) ---")
    print("If you have a central research hypothesis, please enter it below.")
    print("This will guide the AI's analysis and writing.")
    print("If not, just press Enter to proceed with an exploratory analysis.\n")
    
    hypothesis_lines = []
    # Use a different prompt for the first line to make it clearer
    first_line = input("Enter hypothesis (or press Enter to skip): ")
    if first_line:
        hypothesis_lines.append(first_line)
        while True:
            line = input()
            if not line:
                break
            hypothesis_lines.append(line)
    
    hypothesis = "\n".join(hypothesis_lines).strip()

    if hypothesis:
        print("\nHypothesis received. The crew will focus on testing this hypothesis.")
    else:
        print("\nNo hypothesis provided. The crew will perform an exploratory data analysis to find key insights.")
    
    # 3. Conditionally define goals and tasks based on hypothesis
    if hypothesis:
        # --- Prompts for a hypothesis-driven paper ---
        analyst_goal = f"""Analyze the provided CSV data to find insights related to the research hypothesis: '{hypothesis}'.
        Your analysis must be thorough, statistically sound, and directly address the hypothesis."""
        
        analysis_task_description = f"""Conduct a comprehensive data analysis of the provided data with a focus on testing the hypothesis.
        Your final report must be in markdown format and include:
        1.  **Descriptive Statistics:** A table summarizing key variables.
        2.  **Hypothesis Testing:** Conduct statistical tests (like t-tests, ANOVA, or correlations) that directly address the research hypothesis. State the results clearly (e.g., t(df) = value, p = value).
        3.  **Key Findings:** A bulleted list summarizing whether the findings support or refute the hypothesis, with supporting data points.

        **Research Hypothesis:**
        {hypothesis}

        **Data Summary:**
        {data_summary}
        """

        methodology_task_description = """Using the provided data analysis and research hypothesis, write a complete 'Methodology' section for a research paper.
        Your section must be plausible and detailed. It should include the following subsections:
        1.  **Participants:** Describe the sample size, demographics, and recruitment method based on the data.
        2.  **Materials & Procedure:** Describe the hypothetical experiment or data collection process designed to test the hypothesis. What were the variables? How were they measured?
        3.  **Data Analysis Plan:** State how the data was analyzed to test the hypothesis, referencing the statistical tests from the analysis report.
        """

        writing_task_description = """Write a full, high-quality academic research paper that tests the provided hypothesis.
        Use the provided data analysis and methodology. The paper must be structured professionally and include all the following sections:
        1.  **Title:** A concise title related to the hypothesis.
        2.  **Abstract:** A summary of the study's purpose, methods, key results, and conclusion regarding the hypothesis.
        3.  **Introduction:** Provide background, state the problem, and clearly present the research hypothesis.
        4.  **Methodology:** Integrate the 'Methodology' section provided.
        5.  **Results:** Present the findings from the data analysis, focusing on the hypothesis test results.
        6.  **Discussion:** Interpret the results. Do they support the hypothesis? Discuss implications and limitations.
        7.  **Conclusion:** A summary of the key takeaways related to the hypothesis.
        """

    else:
        # --- Prompts for an exploratory paper ---
        analyst_goal = """Conduct an exploratory data analysis on the provided CSV data to uncover interesting patterns,
        correlations, and key insights. Your goal is to identify potential relationships that could form the basis of a future study."""

        analysis_task_description = f"""Conduct a comprehensive exploratory data analysis of the provided data.
        Your final report must be in markdown format and include:
        1.  **Descriptive Statistics:** A table summarizing all key variables.
        2.  **Correlation Analysis:** A correlation matrix or a list of the most significant correlations between variables.
        3.  **Key Patterns & Insights:** A bulleted list of the most interesting and unexpected patterns or group differences found in the data.
        4.  **Potential Hypotheses:** Based on your analysis, suggest 2-3 potential research hypotheses that this data might support.

        **Data Summary:**
        {data_summary}
        """

        methodology_task_description = """Based on the exploratory data analysis, write a complete 'Methodology' section for a research paper.
        Since there was no initial hypothesis, frame this as a descriptive or exploratory study. Your section must be plausible and include:
        1.  **Participants:** Describe the sample size, demographics, and data source.
        2.  **Measures:** Detail the variables that were collected from the dataset.
        3.  **Data Analysis Plan:** Describe the exploratory analysis approach taken (e.g., descriptive stats, correlation analysis).
        """

        writing_task_description = """Write a full, high-quality exploratory research paper based on the provided data analysis and methodology.
        The paper should be framed as an exploratory study. It must be structured professionally and include all the following sections:
        1.  **Title:** A title reflecting the exploratory nature of the study.
        2.  **Abstract:** A summary of the study's purpose, the data analyzed, the most significant findings, and their potential implications.
        3.  **Introduction:** Provide background on the topic area, state that the study is exploratory, and pose the main research questions that guided the analysis.
        4.  **Methodology:** Integrate the 'Methodology' section provided.
        5.  **Results:** Present the key findings from the exploratory data analysis, including descriptive statistics and any significant correlations or patterns discovered. Present the potential hypotheses generated by the analyst.
        6.  **Discussion:** Interpret the key findings. What do these patterns mean? Discuss the implications, limitations of the dataset, and suggest which of the generated hypotheses are most promising for future research.
        7.  **Conclusion:** A brief summary of the key discoveries.
        """

    # 4. Define CrewAI Agent (Econometrician – handles all writing tasks)
    try:
        with open('config/agents/econometrician_agent.yaml', 'r', encoding='utf-8') as f:
            econ_cfg_all = yaml.safe_load(f) or {}
    except FileNotFoundError:
        econ_cfg_all = {}

    econ_cfg = (econ_cfg_all.get('econometrician_agent') or {})
    econometrician_agent = Agent(
        name=econ_cfg.get('name', 'Econometrician Agent'),
        role=econ_cfg.get('role', 'Econometrician & Research Writer'),
        goal=econ_cfg.get('goal', analyst_goal),
        backstory=econ_cfg.get('backstory', 'A seasoned econometrician proficient in methodology, modeling, and academic writing.'),
        verbose=econ_cfg.get('verbose', True),
        allow_delegation=econ_cfg.get('allow_delegation', False),
        allow_code_execution=econ_cfg.get('allow_code_execution', False)
    )

    # 5. Define Tasks
    # Load paper task templates from YAML and format
    paper_cfg = flow = None
    try:
        paper_cfg = yaml.safe_load(open('config/tasks/paper_tasks.yaml','r',encoding='utf-8')) or {}
    except Exception:
        paper_cfg = {}

    if hypothesis:
        analysis_focus = f"Hypothesis-driven analysis focusing on: {hypothesis}"
    else:
        analysis_focus = "Exploratory analysis to discover patterns and relationships."

    a_t = (paper_cfg.get('analysis_task') or {})
    m_t = (paper_cfg.get('methodology_task') or {})
    w_t = (paper_cfg.get('writing_task') or {})

    analysis_task = Task(
        description=(a_t.get('description','').replace('{analysis_focus}', analysis_focus).replace('{data_summary}', data_summary) or analysis_task_description),
        agent=econometrician_agent,
        expected_output=a_t.get('expected_output', "A markdown report detailing the analysis.")
    )

    methodology_task = Task(
        description=m_t.get('description', methodology_task_description),
        agent=econometrician_agent,
        context=[analysis_task],
        expected_output=m_t.get('expected_output', "A complete, well-written 'Methodology' section in markdown format.")
    )

    writing_task = Task(
        description=w_t.get('description', writing_task_description),
        agent=econometrician_agent,
        context=[analysis_task, methodology_task],
        expected_output=w_t.get('expected_output', "A single markdown document containing the full paper.")
    )

    # 6. Instantiate and Run the Crew
    _paper_kwargs = dict(
        agents=[econometrician_agent],
        tasks=[analysis_task, methodology_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )
    try:
        paper_crew = Crew(knowledge=paper_crew_knowledge, **_paper_kwargs)
    except TypeError:
        paper_crew = Crew(**_paper_kwargs)
        attach_knowledge_to_crew(paper_crew, paper_crew_knowledge)

    print("\n Kicking off the Research Paper Generation Crew... This may take several minutes.")
    result = paper_crew.kickoff()

    # 7. Final Output
    print("\n\n Crew finished the task! Here is the final research paper:")
    print("="*80)
    print(result)
    print("="*80)

    # 8. Save the output to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"research_paper_{timestamp}.md"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(result.raw)
    print(f"\n Research paper has been saved to '{output_filename}'")


def main():
    """Main function to run the enhanced survey processing and deployment flow"""
    load_dotenv()

    print("=========================================")
    print("⚠️  INPUT REQUIREMENTS:")
    print("- You must include a line starting with 'Topic:'")
    print("- You must include at least one line starting with 'Questions:'")
    print("Otherwise, the survey cannot be processed.")
    print("=========================================")

    print("\nPlease enter your survey content. Press Enter twice on an empty line when finished.")
    survey_lines = []
    while True:
        line = input()
        if not line and survey_lines and not survey_lines[-1]:
            break
        survey_lines.append(line)
    survey_to_process = "\n".join(survey_lines).strip()

    if not survey_to_process:
        print("No survey text provided. Returning to main menu.")
        return

    if 'Topic:' not in survey_to_process:
        print("Error: Survey must include a line starting with 'Topic:'")
        return

    if 'Questions:' not in survey_to_process:
        print("Error: Survey must include at least one line starting with 'Questions:'")
        return

    enhancement_flow = SurveyEnhancementFlow()

    try:
        initial_result = enhancement_flow.run(survey_to_process)
        if initial_result:
            enhancement_flow.interactive_enhancement()
            
    except Exception as e:
        print(f"An error occurred in the survey processing flow: {str(e)}")
        import traceback
        traceback.print_exc()

# Function to collect data from a previously deployed survey
def collect_survey_data():
    """
    Collects and formats data from a completed survey in Qualtrics and MTurk.
    The output CSV will contain interleaved questions and answers.
    
    Returns:
        dict: Collected data including formatted responses and assignment information.
    """
    survey_id = input("Enter your Qualtrics Survey ID: ")
    hit_id = input("Enter your MTurk HIT ID (leave blank if not using MTurk): ")

    if not survey_id:
        print("Error: Qualtrics Survey ID is required")
        return None

    print(f"Ready to collect data for Survey ID: {survey_id}" +
          (f" and HIT ID: {hit_id}" if hit_id else ""))

    automation = QualtricsAndMTurkAutomation()

    try:
        collected_data = None
        if hit_id:
            # The collect_and_process_results method now handles question fetching and formatting
            collected_data = automation.collect_and_process_results(
                survey_id=survey_id,
                hit_id=hit_id,
                auto_approve=input("Auto-approve MTurk assignments? (y/n): ").lower() == 'y'
            )
        else:
            # Handle Qualtrics-only data collection and formatting
            print("Collecting Qualtrics data only...")
            
            # 1. Get Questions
            try:
                question_map = automation.qualtrics.get_survey_questions(survey_id)
            except Exception as e:
                print(f"Warning: Could not retrieve survey questions: {e}. Output will not include question text.")
                question_map = {}

            # 2. Get Responses
            responses_df = automation.qualtrics.get_survey_responses(survey_id)
            
            # 3. Format with interleaved questions and answers
            final_df = automation._format_df_with_interleaved_questions(responses_df, question_map)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"survey_{survey_id}_responses_{timestamp}.csv"
            final_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
            print(f"\nSaved formatted responses to {csv_filename}")

            collected_data = {
                "responses": final_df,
                "csv_filename": csv_filename,
                "assignments": [],
                "approved_count": 0
            }

        if 'responses' in collected_data and not collected_data['responses'].empty:
            print("\nFormatted Response Preview:")
            print(collected_data['responses'].head())

            # Save to a final detailed CSV file
            detailed_csv = f"survey_{survey_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            collected_data['responses'].to_csv(detailed_csv, index=False, encoding="utf-8-sig")
            print(f"Detailed formatted responses also saved to: {detailed_csv}")
        else:
            print("No responses were collected or the result is empty.")

        return collected_data

    except Exception as e:
        print(f"An error occurred while collecting data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Entry point with options menu
if __name__ == "__main__":
    while True:
        print("\n====== Survey Processing System ======")
        print("1. Create and enhance a new survey")
        print("2. Collect human data from existing survey")
        print("3. Collect simulated data from existing survey")
        print("4. Generate research paper from CSV data")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            main()
        elif choice == "2":
            collect_survey_data()
        elif choice == "3":
            collect_simulated_data()
        elif choice == "4":
            generate_research_paper()
        elif choice == "5":
            print("Exiting...")
            break 
        else:
            print("Invalid choice. Please try again.")
