#!/usr/bin/env python
"""Monte-Carlo: how likely is a random 22-23 drug test fold to have a low
discriminability ratio (<1)? If the probability is non-trivial, the three
collapsed folds in scaffold_cold are simply unlucky random draws.

disc_ratio = std(test drug means) / mean(within-drug std)
Pearson on per-pair predictions collapses when disc_ratio < 1 (drug-level
mean separation is smaller than within-drug noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

df = pd.read_pickle("data/gdsc2.pkl")
drug_means = df.groupby("ID1")["Y"].mean()
drug_stds = df.groupby("ID1")["Y"].std().fillna(0.0)
n_drugs = len(drug_means)

rng = np.random.RandomState(0)
N = 20000
fold_size = 23
disc = np.empty(N)
for i in range(N):
    idx = rng.choice(n_drugs, fold_size, replace=False)
    disc[i] = drug_means.iloc[idx].std() / max(drug_stds.iloc[idx].mean(), 1e-6)

print(f"n_drugs={n_drugs}  fold_size={fold_size}  n_sims={N}")
print(f"disc_ratio: mean={disc.mean():.3f}  p10={np.percentile(disc,10):.3f} "
      f"p25={np.percentile(disc,25):.3f}  p50={np.percentile(disc,50):.3f}")
print(f"P(disc_ratio < 1.0) = {np.mean(disc < 1.0)*100:.2f}%")
print(f"P(disc_ratio < 1.3) = {np.mean(disc < 1.3)*100:.2f}%")
print(f"P(disc_ratio < 1.4) = {np.mean(disc < 1.4)*100:.2f}%")

# Probability that at least 3 of 6 folds have disc_ratio < 1.4 (scaffold_cold had 3)
p_low = np.mean(disc < 1.4)
prob_3of6 = sum(
    (p_low ** k) * ((1 - p_low) ** (6 - k)) * len(list(__import__("itertools").combinations(range(6), k)))
    for k in range(3, 7)
)
print(f"P(>=3 of 6 random folds have disc_ratio<1.4) = {prob_3of6*100:.2f}%")

# Also: how does disc_ratio relate to fold-level pearson? Use oracle-free proxy:
# simulate 6-fold CV repeatedly, count folds with disc<1.4
n_trials = 2000
low_fold_counts = []
for _ in range(n_trials):
    cnt = 0
    for _f in range(6):
        idx = rng.choice(n_drugs, fold_size, replace=False)
        d = drug_means.iloc[idx].std() / max(drug_stds.iloc[idx].mean(), 1e-6)
        if d < 1.4:
            cnt += 1
    low_fold_counts.append(cnt)
c = np.array(low_fold_counts)
print(f"\nPer-CV (6 folds): P(>=2 low folds)={np.mean(c>=2)*100:.2f}%  "
      f"P(>=3 low folds)={np.mean(c>=3)*100:.2f}%  mean low folds={c.mean():.2f}")
