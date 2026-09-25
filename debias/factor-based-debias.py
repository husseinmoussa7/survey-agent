import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch

import matplotlib.pyplot as plt
import secrets

import os


def get_embedding(text, model="text-embedding-3-small"):
    # Key is read here, not at import: the reported run below reads cached embeddings
    # from survey_with_embeddings.pkl and needs no API access.
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or export it.")
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# # Load historical and new data
# df_1 = pd.read_csv("gss_with_llm_responses_1.csv")
# df_2 = pd.read_csv("gss_with_llm_responses_2.csv")
# df_3 = pd.read_csv("gss_with_llm_responses_3.csv")
# df = pd.concat([df_1, df_2, df_3], ignore_index=True)
# df.rename(columns={
#     'Average_Human': 'Average_Human_Response',
#     'Average_LLM_response': 'Average_LLM_Response'
# }, inplace=True)

# # Delet missing data
# df = df[df['Average_Human_Response'].notna()]
# df.reset_index(drop=True, inplace=True)

# # Add embedding
# tqdm.pandas()
# df["Embedding"] = df["Question"].progress_apply(get_embedding)

# # Save to new CSVs with embeddings
# df.to_pickle("survey_with_embeddings.pkl")  # Use pickle to keep list structure

def fit_beta_factor_penalty(
    df,
    n_components: int = 50,
    penalty_coef: float = 1.0,
    lr: float = 1e-2,
    epochs: int = 200,
):
    """
    Same as fit_beta_pca but uses FactorAnalysis instead of PCA.
    Returns beta_hat, mse_train, and fitted FA.
    """
    X_orig = np.vstack(df["Embedding"].values)
    llm   = df["Average_LLM_Response"].to_numpy()
    human = df["Average_Human_Response"].to_numpy()
    y     = llm - human

    fa = FactorAnalysis(n_components=n_components, random_state=0)
    X = fa.fit_transform(X_orig)                              # (n, k)

    X_t     = torch.from_numpy(X).float()
    y_t     = torch.from_numpy(y).float()
    llm_t   = torch.from_numpy(llm).float()
    human_t = torch.from_numpy(human).float()

    beta = torch.zeros(n_components, requires_grad=True)
    opt  = torch.optim.Adam([beta], lr=lr)

    for _ in range(epochs):
        pred = X_t @ beta
        debiased = llm_t - pred

        mse_loss = torch.mean((debiased - human_t)**2)
        mask     = (y_t * pred < 0).float()
        pen      = torch.mean(mask * torch.abs(y_t - pred))
        loss     = mse_loss + penalty_coef * pen

        opt.zero_grad()
        loss.backward()
        opt.step()

    beta_hat = beta.detach().numpy()
    df["Debiased_Response"] = llm - X.dot(beta_hat)
    mse_train = np.mean((df["Debiased_Response"] - human)**2)

    return beta_hat, mse_train, fa


# Load the pickle file (with embeddings as lists)
df = pd.read_pickle("survey_with_embeddings.pkl")

# this will randomly select 100 rows for train, and the other 12 for valid
train_df, valid_df = train_test_split(
    df,
    train_size=100,
    random_state=8566,   # for reproducibility
    shuffle=True
)
print(train_df.shape)  # (100, )
print(valid_df.shape)  # (12, )
print("Embedding shape:", len(df["Embedding"].iloc[0]))

# 1) Stack your training embeddings into an (N × D) matrix
X_train = np.vstack(train_df["Embedding"].values)   # shape: (100, D)

# 2) Fit an “untruncated” PCA to get all eigenvalues
pca_full = PCA(random_state=0)
pca_full.fit(X_train)

# 3) Extract the raw eigenvalues (not the explained-variance ratios)
eigenvalues = pca_full.explained_variance_          # length = min(N,D)

# 4) Cumulative explained variance
cum_ev = np.cumsum(pca_full.explained_variance_ratio_)
plt.figure(figsize=(6,4))
plt.plot(
    np.arange(1, len(cum_ev)+1),
    cum_ev,
    marker='o', linestyle='-'
)
plt.axhline(0.90, color='r', linestyle='--', label='90% explained')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.show()

# FA‐based
# list the row‐indices you want to keep
keep_idx = [35, 2, 8, 43, 22, 18, 70, 12, 17, 82]
beta_fa, mse_fa, fa_model = fit_beta_factor_penalty(train_df.copy(), n_components=50, penalty_coef=20.0)
Xv_fa = fa_model.transform(np.vstack(valid_df["Embedding"].values))
valid_df["Debiased_Response"] = valid_df["Average_LLM_Response"].to_numpy() - Xv_fa.dot(beta_fa)
mse_val_fa = np.mean((valid_df["Debiased_Response"] - valid_df["Average_Human_Response"].to_numpy())**2)
print("Valid MSE (FA):", mse_val_fa)
selected_df = valid_df.loc[keep_idx].reset_index(drop=True)
selected_df.head(10)