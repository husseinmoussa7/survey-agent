> Original standalone README for this pipeline, preserved as shipped.
> Paths below are relative to this `panel/` directory. For running it inside the
> combined repository, and for what is and is not committed, see `../README.md`.

# Debias Pipeline Replication (Twins)

## Folder structure
- code/simulate_response/
  - step1_twin_build_master.py
  - step2_twin_debias_models.py
  - step3_make_averages.py
  - run_waves_simulations.py (LLM simulation)
  - llm_openai.py
  - simulate_response.py
- data_raw/waves/                  (raw wave files used by Step 1)
- data_questions/                  (cleaned question CSVs used by simulation)
- outputs/
  - combined_waves_1_to_4_master.pkl   (created by Step 1)
  - step2/                             (created by Step 2)
  - step3/                             (created by Step 3)


## Environment setup (recommended)

### Option A: Conda + pip (most reliable)
```bash
conda create -n debias python=3.11 -y
conda activate debias
pip install -r requirements.txt


## API setup (only needed for LLM simulation)
1) Create `.env` in project root:
   OPENAI_API_KEY=YOUR_KEY_HERE

## Step 1: Build master dataset
From project root:
  python code/simulate_response/step1_twin_build_master.py

Expected output:
  outputs/combined_waves_1_to_4_master.pkl

## Step 2: Debias models + seed hunting
  python code/simulate_response/step2_twin_debias_models.py

Expected outputs:
  outputs/step2/waves_1_to_4_combined_results_same_seeds_{d,norm}/...

## Step 3: Average metrics across seeds
  python code/simulate_response/step3_make_averages.py

Expected outputs:
  outputs/step3/waves_1_to_4_combined_AVG_METRICS_ALL_PENALTIES.csv
  outputs/step3/waves_1_to_4_combined_AVG_METRICS_pen_*.csv

## LLM Simulation (optional)
Edit `waves_to_process` and file naming inside `run_waves_simulations.py`, then run:
  python code/simulate_response/run_waves_simulations.py

Expected output:
  data_llm/llm_pooled_responses_wave_<wave>_v7_FINAL.csv

## Troubleshooting
- ModuleNotFoundError: sklearn
  -> activate conda env (`conda activate debias`) or install scikit-learn
- OPENAI_API_KEY not found
  -> create .env in project root and add OPENAI_API_KEY
- Files not found
  -> confirm you ran Step 1 and the outputs/ paths match the scripts

