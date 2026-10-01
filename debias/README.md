# Debiasing LLM survey responses

Two implementations of the same idea: learn the systematic bias between LLM-simulated and human
survey responses from question text, then subtract the predicted bias from new LLM responses. They
are kept separate because they back different numbers.

| | `debias/` (top level) | `debias/panel/` |
|---|---|---|
| Human benchmark | General Social Survey published averages | Four-wave panel, 330 questions / 388 question-wave rows |
| Respondents per item | 50 LLM simulated | 42 LLM simulated, 2,059 human |
| Anchor set | 110 rows over 105 GSS variables | 388 rows |
| Validation | one holdout of 11 items, pinned by identity | 200 resampled 80/20 splits from one master seed |
| Estimators | directional-penalty model | OLS, Lasso, directional-penalty model |
| Bias target | raw difference | Cohen's d and range-normalized |
| Backs | the primary reported result | the robustness result and replication tolerances |

## Reproducing the GSS result

```bash
python debias/reproduce_gss_result.py
```

It prints each published figure beside the value this repository actually produces:

| quantity | reproduced here | published |
|---|---|---|
| baseline (uncorrected) MSE | **0.639** | 0.639 |
| debiased MSE | **0.422** | 0.444 |
| improvement | **34.0%** | 30.5% |
| moved toward human mean | **9 of 11** | 9 of 11 |
| training MSE | **0.515** | 0.903 |

The baseline and the directional accuracy reproduce exactly. The debiased MSE and the improvement
do not match the published values and the training MSE differs substantially. This is the closest
reproducible configuration found; what remains unmatched is documented below so a reader can judge
it rather than discover it.

The baseline and the holdout check need only numpy, pandas and scikit-learn, since the baseline
involves no model. Fitting the correction additionally requires torch.

### Why this script exists rather than `factor-based-debias.py`

**The holdout is pinned by item identity, not by a seed.** `factor-based-debias.py` selects its
holdout with `train_test_split(df, train_size=100, random_state=8566)`. Over the shipped anchor set
that returns a different 12-item holdout whose uncorrected baseline MSE is 0.173, so it cannot
reproduce the published figures, and no seed can: a search over 100,000 seed and split-size
combinations found none selecting the 11 reported items. Those items are recorded only in appendix
table `tab:full_results`, so the script pins them explicitly.

**Three of the four published figures were computed nowhere.** `factor-based-debias.py` prints only
the debiased MSE. The baseline, the percentage improvement, and the directional accuracy are
computed here for the first time.

`factor-based-debias.py` is retained as the original exploratory script, including the
cumulative-explained-variance plot behind the choice of 50 factors.

### What remains unreproduced

The published debiased MSE of 0.444, the 30.5% improvement, and the 0.903 training MSE are not
recovered by any configuration tested:

- As shipped, with the holdout pinned: 0.536 debiased, 16.1%, 8 of 11, training MSE 0.600.
- Excluding any single training row, all 101 possibilities: none yields 0.444 or 0.903.
- Collapsing identical-text repeats, the configuration now used: 0.422, 34.0%, 9 of 11, 0.515.

Per-item debiased values in the current configuration track the appendix table closely (`libhomo`
1.883 against 1.877, `discaffm` 3.061 against 3.062, `discaff` 2.808 against 2.812), so the method
and data are right and some unrecorded detail of the original run is not. Candidates include a
different epoch count, a different embedding snapshot, or a different torch version. The reported
figures should be updated to the reproducible ones.

### The seven repeated variable names

Seven GSS variables appear twice in the shipped anchor set. They are not all the same thing, and an
earlier version of this file described them wrongly.

- **`aged` and `libhomo`** repeat with **identical question text**, so they are second LLM draws of
  one stimulus. These two rows are collapsed, leaving 110.
- **`abnomore`, `conbus`, `fehire`, `helpblk`, `natsci`** share a variable name but are asked with
  **different wording**, so they are distinct stimuli sharing a published human average. `natsci`
  appears once with the full survey preamble and once as a short direct question, giving LLM
  averages of 2.000 and 1.333 against the same human value of 1.688.

The five wording pairs are deliberately kept. Collapsing them discards a prompt-wording comparison
and degrades the correction: averaging every repeated name drops the held-out improvement from
34.0% to 18.2% and directional accuracy from 9 of 11 to 8 of 10.

The consequence to disclose: for those five variables the same human average appears once in
training and once in the holdout. The embeddings differ, so this is not a direct label leak, but it
is a dependency between the splits.

Counts to state consistently: **110 rows** after collapsing identical-text repeats, over **105 GSS
variables**, with **11** held out and **99** used for training. The manuscript's 100 training items
implies one row more than this set; which row is not recorded.

## Files

- `debias.py` is the reusable tool; every hyperparameter is a flag:
  ```bash
  python debias.py --input_json test_new_questions.json --output_json out.json --lambda_ 20 --alpha 0.90
  ```
  It is imported by the application (`from debias.debias import run_debias_pipeline`), so it must
  stay at this path.
- `reproduce_gss_result.py` reproduces the reported result, as above.
- `factor-based-debias.py` is the original exploratory script.
- `gss_with_llm_responses_{1,2,3}.csv` are the anchor items (37 + 37 + 38 = 112 rows before
  collapsing), each with a published human average and an LLM average.
- `survey_with_embeddings.pkl` caches the `text-embedding-3-small` embeddings, so reruns cost no API
  calls.
- `gss_deduplicated_holdout_results.csv` holds the per-item holdout values from the current run.
- The LLM averages were generated by `simulate_response/run_simulation_gss.ipynb`.

## `debias/panel/` — the four-wave panel (robustness)

Run in order from the repository root:

```bash
python debias/panel/code/simulate_response/step1_twin_build_master.py   # build master table
python debias/panel/code/simulate_response/step2_twin_debias_models.py  # fit + 200-seed evaluation
python debias/panel/code/simulate_response/step3_make_averages.py       # average across seeds
```

Step 1 embeds question text and merges human and LLM responses into
`outputs/combined_waves_1_to_4_master.pkl`, computing both bias targets. Step 2 fits each estimator
and evaluates it across 200 shared resamples. Step 3 averages the per-seed metrics.
`run_waves_simulations_original.py` regenerates the LLM responses themselves and costs API calls;
its output is committed so the analysis runs without them.

### Results

Directional-penalty model at `lambda=20`, 200 resamples, mean held-out MSE reduction:

| Estimator | Cohen's d | Normalized |
|---|---|---|
| Directional penalty | 68.2% (SD 4.2, range 54.3 to 78.8), direction 77.0% | 71.7% (SD 3.9, range 60.2 to 81.6), direction 83.9% |
| OLS | 56.0% (SD 7.5), direction 70.9% | 61.3% (SD 5.8), direction 81.0% |
| Lasso | 56.9% (SD 6.8), direction 71.4% | |

The penalty grid collapses to two distinct solutions on the Cohen's d target: `lambda >= 1` gives
68.2% with 77.0% direction, `lambda <= 0.1` gives 68.6% with 75.4%. The higher-penalty branch trades
a little error for better direction, which is what the penalty is for. All five penalties converge
on the normalized target. The penalty earns its gain over OLS and Lasso once present, largely
regardless of its exact weight in this range.

Per-seed numbers are in `outputs/step2/**/…_ALL_SEEDS_SUMMARY.csv`; averages in `outputs/step3/`.
These figures reproduce exactly: an independent evaluation of the committed coefficient vectors
returns 68.196% and 60.035/78 for Cohen's d, 71.728% and 65.445/78 normalized.

### What is not committed

Step 2 also writes per-seed prediction dumps and raw split copies, roughly 260 MB, excluded by
`.gitignore` because step 2 regenerates them deterministically from the same master seed. Committed
instead: the per-seed summaries, the fitted coefficient vectors (`.npy`), the best-epoch tables, and
the cached embeddings.

## Environment

`torch` is needed only to **fit** the directional-penalty model. It is imported defensively in both
`debias.py` and `step2_twin_debias_models.py`, so the application starts and every other path runs
without it; attempting a fit without it raises a clear error.

That matters because torch is not installable everywhere the root README's "Python 3.12+" implies.
There are no torch wheels for Intel macOS past 2.2.2 and none for Python 3.13 on that platform. Two
working routes on x86_64 macOS: Python 3.12 or lower with `torch==2.2.2` from pip, or a conda
environment from conda-forge, which does provide osx-64 builds for Python 3.13
(`conda create -n debias -c conda-forge python=3.13 pytorch scikit-learn pandas`). Apple Silicon and
Linux are unaffected.

Reproducible without torch: `step3_make_averages.py`, the 200-seed evaluation of committed
coefficient vectors, and the GSS baseline MSE. Requiring torch: refitting coefficients, i.e.
`step2_twin_debias_models.py`, `factor-based-debias.py`, and the fit inside
`reproduce_gss_result.py`.

## Data and keys

`panel/data_raw/waves/*.csv` are the human responses, de-identified and keyed by `TWIN_ID`. Each
file's first row is the Qualtrics question-text header, so a wave of 2,059 respondents reads as
2,060 rows. The Qualtrics timing columns (`StartDate`, `EndDate`, `RecordedDate`,
`Duration (in seconds)`) are unused by every step; consider dropping or coarsening them before
public release, since precise timestamps alongside income, household size, religion and education
are a re-identification vector.

Both paths read `OPENAI_API_KEY` from the environment. Copy `panel/.env.example` to `.env` and fill
it in. No key is stored in this repository.
