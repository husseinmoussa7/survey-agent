"""
Implements Algorithm 1 (Factor-Model–Based Debiasing) using purely functions.
Reads:
  - a pickle/CSV of pre‐existing data with columns:
      'Question' (str)
      'Embedding' (list[array]),
      'Average_Human_Response' (float),
      'Average_LLM_Response'   (float)
  - an input JSON of new questions:
      [
        {
          "Question": (str),
          "num_llms": int,
          "llm_resp": (list[float])
        },
        ...
      ]
Writes:
  - an output JSON with an added "debiased_llm_resp" field per question.
"""
import os
import json
import argparse
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
import openai

# torch is only needed to FIT the directional-penalty model (fit_beta_with_penalty).
# It is imported defensively so that importing this module, and therefore starting
# survey.py / server.py, does not require torch to be installed. Same pattern as
# debias/panel/code/simulate_response/step2_twin_debias_models.py.
try:
    import torch
    import torch.optim as optim
except ImportError:  # pragma: no cover
    torch = None
    optim = None


def get_embedding(text, model="text-embedding-3-small"):
    """
    text: str
    model: str, OpenAI embedding model name
    returns: list[float] embedding vector
    """
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def choose_components(X, variance_threshold):
    """
    X: array-like, shape (n_samples, n_features)
    variance_threshold: float in (0, 1]
    returns: (k, pca)
      k: int, minimum # components whose cumulative explained variance ≥ threshold
      pca: fitted sklearn.decomposition.PCA instance
    """
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_threshold) + 1)
    return k, pca

def fit_factor_analysis(X, n_components):
    """
    X: array-like, shape (n_samples, n_features)
    n_components: int
    returns: (F, fa)
      F: ndarray, shape (n_samples, n_components), the factor scores
      fa: fitted sklearn.decomposition.FactorAnalysis instance
    """
    fa = FactorAnalysis(n_components=n_components)
    F = fa.fit_transform(X)
    return F, fa


def fit_beta_with_penalty(F, delta, penalty_weight, lr, epochs, device=None):
    """
    F: array-like of shape (n_samples, n_components)
    delta: array-like of shape (n_samples,)
    penalty_weight: float, weight for the sign‐mismatch penalty
    lr: float, Adam learning rate
    epochs: int, number of training iterations
    device: str or torch.device, optional ("cuda" or "cpu")

    Returns
    -------
    beta: ndarray of shape (n_components,)
        The fitted coefficient vector.
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required to fit the directional-penalty model. "
            "Install it with `pip install torch` (see debias/README.md for platform notes)."
        )

    # select device
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev)

    # prepare tensors (cast to float32)
    F_t     = torch.from_numpy(np.asarray(F, dtype=np.float32)).to(dev)
    delta_t = torch.from_numpy(np.asarray(delta, dtype=np.float32)).to(dev)
    k       = F_t.shape[1]

    # initialize beta as float32
    beta = torch.zeros(k, dtype=torch.float32, device=dev, requires_grad=True)
    optimizer = optim.Adam([beta], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        pred    = F_t @ beta               # both F_t & beta are float32
        mse     = (delta_t - pred).pow(2).mean()
        mask    = (delta_t * pred < 0).float()
        penalty = (mask * (delta_t - pred).abs()).mean()
        loss    = mse + penalty_weight * penalty
        loss.backward()
        optimizer.step()

    return beta.detach().cpu().numpy()


def debias_llm_responses(embedding, beta, fa, raw_llm_resps):
    """
    embedding: array-like of shape (n_features,)
    beta: array-like of shape (n_components,)
    fa: fitted FactorAnalysis instance
    raw_llm_resps: list of float

    Returns
    -------
    debiased: list of float
        Each raw response minus the single bias estimate.
    """
    # compute factor score for this single embedding
    x = np.asarray(embedding, dtype=float).reshape(1, -1)
    F_new = fa.transform(x)            # shape (1, k)
    delta_hat = float(F_new.dot(beta)) # scalar

    # subtract the same bias from every LLM response
    return [resp - delta_hat for resp in raw_llm_resps]


def run_debias_pipeline(
    input_json: str,
    output_json: str,
    variance_threshold: float = 0.90,
    penalty_weight: float = 15.0,
    lr: float = 1e-3,
    epochs: int = 500,
    embed_model: str = "text-embedding-3-small"
):
    """
    input_json: path to the JSON file containing new questions
    output_json: path where the debiased JSON will be written
    variance_threshold: PCA cumulative variance cutoff α (default 0.90)
    penalty_weight: directional penalty λ (default 15.0)
    lr: learning rate for β optimization (default 1e-3)
    epochs: number of training epochs (default 500)
    embed_model: OpenAI embedding model to use (default "text-embedding-3-small")
    """

    # 1) Load the fixed pre‐existing pickle
    # df = pd.read_pickle("survey_with_embeddings.pkl")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(base_dir, "survey_with_embeddings.pkl")
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Cannot find embeddings pickle at {pickle_path}")
    df = pd.read_pickle(pickle_path)

    # 2) Stack embeddings and responses
    embeddings = np.vstack(df["Embedding"].tolist())
    human_avg   = np.array(df["Average_Human_Response"], dtype=float)
    llm_avg     = np.array(df["Average_LLM_Response"], dtype=float)

    # 3) Determine k and fit FA
    k, _ = choose_components(embeddings, variance_threshold)
    F, fa = fit_factor_analysis(embeddings, k)

    # 4) Fit beta on the average bias δ = llm_avg − human_avg
    delta = llm_avg - human_avg
    beta  = fit_beta_with_penalty(
        F, delta,
        penalty_weight=penalty_weight,
        lr=lr,
        epochs=epochs
    )

    # 5) Read new questions JSON
    with open(input_json, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    # 6) Debias each question
    for item in data:
        q       = item["Question"]
        raw_llm = item["llm_resp"]
        emb     = get_embedding(q, model=embed_model)
        # returns a single scalar δ̂ and subtracts from every response
        item["debiased_llm_resp"] = debias_llm_responses(
            emb, beta, fa, raw_llm
        )

    # 7) Write back out
    result = data[0] if len(data)==1 else data
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Debias LLM responses via factor-model correction"
    )
    parser.add_argument(
        "--input_json", "-i", required=True,
        help="Path to JSON with new questions"
    )
    parser.add_argument(
        "--output_json", "-o", required=True,
        help="Where to write the debiased JSON"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.90,
        help="PCA variance threshold α"
    )
    parser.add_argument(
        "--lambda_", type=float, default=15.0,
        help="Directional penalty weight λ"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for β fitting"
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--embed_model", type=str, default="text-embedding-3-small",
        help="OpenAI embedding model"
    )
    args = parser.parse_args()

    # set your API key in the environment beforehand
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai.api_key:
        raise RuntimeError("Please set OPENAI_API_KEY")

    # run it
    run_debias_pipeline(
        input_json=args.input_json,
        output_json=args.output_json,
        variance_threshold=args.alpha,
        penalty_weight=args.lambda_,
        lr=args.lr,
        epochs=args.epochs,
        embed_model=args.embed_model
    )

# User Example
# export OPENAI_API_KEY="sk-YOUR_REAL_KEY_HERE"
# python debias.py --input_json test_new_questions.json --output_json test_debiased_output.json
