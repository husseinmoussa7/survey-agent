# Field-Experiment-AI-Agent

AI Agent framework for conducting consumer behavior experiments, and the
replication package for the accompanying manuscript on debiasing LLM-simulated
survey responses.

Two things live here, and they are useful independently:

| | What it is | Start at |
|---|---|---|
| **Replication** | The code, data and committed artifacts behind every reported figure | [Reproducing the paper's results](#reproducing-the-papers-results) |
| **Application** | The interactive survey agent: convert, enhance, deploy, simulate, analyse | [The survey agent application](#the-survey-agent-application) |

---

## Reproducing the paper's results

You do **not** need the full application stack for this. Install the minimal set:

```bash
conda create -n debias python=3.12 -y && conda activate debias
pip install -r debias/panel/requirements-panel.txt

# Only if you want to FIT the model rather than evaluate committed coefficients.
# Not available on every platform - see debias/README.md "Environment" first.
pip install 'torch>=2.2,<2.3'
```

Then, from the repository root:

```bash
# Primary result: the GSS anchor set (prints every published figure beside
# the value this repository produces, and writes the per-item table)
python debias/reproduce_gss_result.py

# Robustness: the four-wave panel, 200 resampled splits
python debias/panel/code/simulate_response/step3_make_averages.py   # averages, from committed per-seed summaries
python debias/panel/code/simulate_response/step2_twin_debias_models.py  # optional: refit from scratch (needs torch, ~10 min)
```

What each backs:

- `debias/reproduce_gss_result.py` → held-out MSE 0.639 → 0.422 (34.0%), correct bias
  direction in 9 of 11 items, and appendix table `tab:full_results`.
- `step3_make_averages.py` → the robustness table: 68.2% mean error reduction at
  `lambda=20` against 56.0% for OLS and 56.9% for Lasso, with directional accuracy
  77.0% against 70.9%.

`step2` and `step3` run from committed artifacts and need **no API key**. Only `step1`
and `run_waves_simulations_original.py` call the OpenAI API. Fitting the penalty model
needs `torch`, whose installability is platform-dependent — read the "Environment"
section of [`debias/README.md`](debias/README.md) first.

Full detail, including what is and is not committed and which configurations do *not*
reproduce, is in [`debias/README.md`](debias/README.md).

---

## Debiasing LLM responses

Two implementations of the same correction, kept separate because they back different numbers:

- **`debias/`** is the GSS path and the primary reported result: 112 General Social Survey anchor
  items over 105 variables, collapsed to 110 after removing two identical-text repeat draws, split
  99 training / 11 held out. Held-out MSE improves from 0.639 to 0.422 (34.0%), with the correct
  bias direction recovered in 9 of 11 items. Reproduce it with
  `python debias/reproduce_gss_result.py`; the holdout is pinned by item identity, not by a seed,
  so `factor-based-debias.py` does not reproduce these figures.
- **`debias/panel/`** is the robustness path: a four-wave survey panel of 330 questions with 2,059
  human respondents per item, evaluated across 200 resampled splits and three estimators. It supplies
  the expected-variation figures (SD 4.2 and 3.9 percentage points on the two bias targets) that a
  single split cannot give.

See `debias/README.md` for how to run each, what is and is not committed, and the data and key
handling. Both read `OPENAI_API_KEY` from the environment; no key is stored in this repository.

Agents
- survey_convert_agent.yaml: Converts raw text to minimal JSON using a cost-efficient model; avoids content rewriting.
- survey_editor.yaml: Enriches context via brief research and enhances/annotates surveys to meet academic standards.
- econometrician_agent.yaml: Executes analysis, methodology, and writing for research papers with journal-level rigor.

Tasks
- convert_survey_to_json.yaml: Converts raw text to a structured survey JSON schema.
- apply_survey_enhancements.yaml: Includes research_task (context enrichment) and improve_survey (annotated + revised survey).
- enhance_survey_iteratively.yaml: Iterative enhancement based on user feedback; outputs revised_survey and explanations.
- paper_tasks.yaml: Templates for analysis_task, methodology_task, and writing_task for end-to-end paper generation.

**Configuration Files:**

* **`config/agents/*.yaml`**: Defines roles and goals for each AI agent.
* **`config/tasks/*.yaml`**: Describes tasks for CrewAI (survey conversion, research, improvement).


---

## The survey agent application

### Overview

This project provides an interactive, AI-powered survey enhancement and deployment system. First-time users can:

* Convert raw text surveys into structured JSON
* Iteratively enhance surveys with AI feedback
* Deploy surveys to Qualtrics and optionally create MTurk HITs
* Collect both human and simulated survey responses
* Generate research papers from CSV data

The system uses Pydantic models, CrewAI agents, OpenAI, Qualtrics API, and MTurk API, offering a seamless end-to-end workflow for academic and market research.

### Prerequisites

* **Python 3.12** — not 3.13 if you need `torch` on an Intel Mac: the last x86_64
  macOS torch wheels are 2.2.2 (cp38-cp312). See `debias/README.md` "Environment".
* A Qualtrics account with API token & data center information
* AWS credentials (if using MTurk) 
* Claude and OpenAI API accounts

---

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Six-Persimmon/Field-Experiment-AI-Agent.git
   cd Field-Experiment-AI-Agent
   ```

2. **Create & activate a virtual environment**

   ```bash
   conda create -n venv python=3.12
   conda activate venv
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

### Quick Start (Recommended Entry & Flow)

- Terminal entry: `python survey.py`
  - Interactive menu to create/enhance surveys, deploy to Qualtrics/MTurk, collect simulated/human data, and generate papers. See “Terminal Usage” below.
  - API keys via `.env` (see “API Keys”). For simulation only, set `OPENAI_API_KEY`.

- Web UI: `python server.py`
  - Starts a local web app (URL shown in terminal) to process/enhance surveys, deploy, simulate responses, and debias via browser.

- Simulation only (quick demo): `python simulate_response/run_simulation.py`
  - Uses `simulate_response/participant_pool.csv`, `simulate_response/test_survey.json`, and `simulate_response/survey_response_template.txt` to produce `simulate_response/simulated_survey_responses.csv`.

---

### Terminal Usage

#### API Keys

   Create a `.env` file in the root folder:

   ```
   # Qualtrics
   QUALTRICS_API_TOKEN=your_token_here
   QUALTRICS_DATA_CENTER=your_datacenter_id
   QUALTRICS_DIRECTORY_ID=your_directory_id

   # AWS (MTurk)
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   MTURK_SANDBOX=True  # Set to True if you want to run a simulation; Set to False if you want to run a real experiment 

   #LLM Models
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_claude_key
   ```

#### Running the Code

Run the following command in your terminal:

```bash
python survey.py
```

##### Input File Formats

* **Copy and Pasting Your Original Survey Text**: Must include:

  ```text
  Topic: Your Survey Topic
  Purpose: Brief description
  Questions:
    - QID1: First question text...
    - QID2: Second question text...
  ```

* **Simulation**:

  * `participant_pool.csv`: CSV with simulated participant metadata. Eg:
  ```ParticipantID,Race,Gender,Age```
  * `test_survey.json`: Can be a default survey in JSON format or your newly created survey (please ensure that you save it as a JSON file)
  * `survey_response_template.txt`: A prompt template telling the LLM model to roleplay the specific demographics found in  `participant_pool.csv` for generating responses.

* **CSV for paper generation**: Any tabular data as long as it is in `.csv` file format; first row is the header.

##### Menu Options

1. **Create and enhance a new survey**

   * Prompt: Enter raw survey text (must include `Topic:` and at least one `Questions:` line).
   * Result: Creates a survey in JSON format, allows AI and human enhancements, and deployable to Qualtrics and MTurk.

2. **Collect human data from existing survey**

   * Input: Qualtrics Survey ID and optional MTurk HIT ID.
   * Output: Formatted CSV with questions and responses.

3. **Collect simulated data from existing survey**

   * Input relative file paths to:

     * `survey_response_template.txt` (LLM template)
     * `test_survey.json` (survey content)
     * `participant_pool.csv` (participant metadata)
   * Output: JSON + CSV of simulated responses; optional debiasing.

4. **Generate research paper from CSV data**

   * Input: Path to CSV file.
   * Optional: Provide a research hypothesis.
   * Output: Markdown-formatted paper saved as `.md` file.

5. **Exit**

---

#### Research Paper Agent

- An econometrician agent executes the full paper workflow end-to-end: analysis, methodology, and writing.
- Focuses on econometric rigor, clear exposition, appropriate visualization, and journal-ready structure aligned with top Economics (Top 5) and Management (UTD 24) venues.
- Tasks remain modular (analysis → methodology → writing) but are handled by a single, specialized agent for coherence and consistency.


### HTML Usage

#### Running the Website

1. Run the following command in your terminal:

```bash
python server.py
```

2. You should see something of the following:
```
 * Serving Flask app 'server'
 * Debug mode: on
2025-07-11 20:09:13,297 - werkzeug - INFO - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://{...}:5001
```

Please click the last link that says "Running on ..."

3. Now, the website will have opened up on your default browser for your usage.

##### API Keys

   Enter the following information onto the website when prompted:

   ```
   # Qualtrics
   QUALTRICS_API_TOKEN=your_token_here
   QUALTRICS_DATA_CENTER=your_datacenter_id
   QUALTRICS_DIRECTORY_ID=your_directory_id

   # AWS (MTurk)
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   MTURK_SANDBOX=True  # Set to True if you want to run a simulation; Set to False if you want to run a real experiment 

   #LLM Models
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_claude_key
   ```

##### Input File Formats

* **Copy and Pasting Your Original Survey Text**: Must include:

  ```text
  Topic: Your Survey Topic
  Purpose: Brief description
  Questions:
    - QID1: First question text...
    - QID2: Second question text...
  ```

* **Simulation**:

  * `participant_pool.csv`: CSV with simulated participant metadata. Eg:
  ```ParticipantID,Race,Gender,Age```
  * `test_survey.json`: Can be a default survey in JSON format or your newly created survey (please ensure that you save it as a JSON file)
  * `survey_response_template.txt`: A prompt template telling the LLM model to roleplay the specific demographics found in  `participant_pool.csv` for generating responses.

* **CSV for paper generation**: Any tabular data as long as it is in `.csv` file format; first row is the header.

##### Menu Options

1. **Create and enhance a new survey**

   * Prompt: Enter raw survey text (must include `Topic:` and at least one `Questions:` line).
   * Result: Creates a survey in JSON format, allows AI and human enhancements, and deployable to Qualtrics and MTurk.

2. **Collect human data from existing survey**

   * Input: Qualtrics Survey ID and optional MTurk HIT ID.
   * Output: Formatted CSV with questions and responses.

3. **Collect simulated data from existing survey**

   * Input relative file paths to:

     * `survey_response_template.txt` (LLM template)
     * `test_survey.json` (survey content)
     * `participant_pool.csv` (participant metadata)
   * Output: JSON + CSV of simulated responses; optional debiasing.

4. **Generate research paper from CSV data**

   * Input: Path to CSV file.
   * Optional: Provide a research hypothesis.
   * Output: Markdown-formatted paper saved as `.md` file.

---

## Key Directories & Files

```plaintext
config/                        # YAML configs for agents and tasks
  agents/
    survey_convert_agent.yaml  # Convert raw text → minimal JSON (cost-efficient agent)
    survey_editor.yaml         # Research enrichment + survey enhancement/editor
    econometrician_agent.yaml  # End-to-end paper agent (analysis/methodology/writing)
  tasks/
    convert_survey_to_json.yaml        # Conversion task template
    apply_survey_enhancements.yaml     # Research + improve survey task templates
    enhance_survey_iteratively.yaml    # Iterative enhancement task template (with placeholders)
    paper_tasks.yaml                   # Paper tasks (analysis, methodology, writing)
debias/                        # Debiasing: GSS path (primary result) - see debias/README.md
  debias.py                    #   reusable CLI tool, also imported by survey.py
  reproduce_gss_result.py      #   reproduces the reported GSS numbers (start here)
  factor-based-debias.py       #   original exploratory script; does NOT reproduce them
  panel/                       #   four-wave panel path (robustness, 200 resamples)
knowledge/                     # Reference materials
simulate_response/             # Survey simulation scripts and templates
test_survey/                   # Sample survey JSON files
```

**Key files:**

```plaintext
survey.py                      # Final production code entry point
survey.html                    # Final HTML product
server.py                      # Backend server to run API calls
survey_logic.py                # Necessary logic from survey.py used for backend calls
requirements.txt               # Python dependencies list
README.md                      # Project overview and instructions
```
