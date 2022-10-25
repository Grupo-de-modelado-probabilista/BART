#!/usr/bin/env python
# coding: utf-8

# Freidman function with another interaction
# Test the values of $m$ and $\alpha$ with a modified Friedman function and more samples.

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pymc_bart as pmb
from pymc_bart.pgbart import compute_prior_probability
import pandas as pd

# Configuration
RANDOM_SEED = 8457
rng = np.random.RandomState(RANDOM_SEED)
az.style.use("arviz-white")
plt.rcParams["figure.dpi"] = 300


# Data generation
X = rng.uniform(low=0, high=1.0, size=(10000, 6))
f_x = (
    10 * np.sin(np.pi * X[:, 0] * X[:, 1] * X[:, 5])
    + 20 * (X[:, 2] - 0.5) ** 2
    + 10 * X[:, 3]
    + 5 * X[:, 4]
)
Y = rng.normal(f_x, 1)

trees = [10, 20, 50, 100, 200]
alphas = [0.1, 0.25, 0.5]
idatas_at = {
    "10": {"0.1": {}, "0.25": {}, "0.5": {}},
    "20": {"0.1": {}, "0.25": {}, "0.5": {}},
    "50": {"0.1": {}, "0.25": {}, "0.5": {}},
    "100": {"0.1": {}, "0.25": {}, "0.5": {}},
    "200": {"0.1": {}, "0.25": {}, "0.5": {}},
}


# Run model
for m in trees:
    for alpha in alphas:
        with pm.Model() as model:
            μ = pmb.BART("μ", X, Y, m=m, alpha=alpha)
            σ = pm.HalfNormal("σ", 1)
            y = pm.Normal("y", μ, σ, observed=Y)
            idata = pm.sample(chains=4, random_seed=RANDOM_SEED)
            idatas_at[str(m)][str(alpha)] = idata

# Boxplots
fig, axes = plt.subplots(1, 5, figsize=(10, 3), sharey=True)

axes = axes.ravel()

for m, ax in zip(trees, axes):
    means = [
        idatas_at[str(m)][str(alpha)]["posterior"]["μ"].mean(("chain", "draw")) - f_x
        for alpha in alphas
    ]
    box = ax.boxplot(
        means,
        notch=True,
        patch_artist=True,
        widths=0.5,
        labels=alphas,
        showfliers=False,
        medianprops=dict(color="k"),
    )
    for patch, color in zip(box["boxes"], ["C0", "C1", "C2", "C3"]):
        patch.set_facecolor(color)
        ax.set_title(f"m = {m}")

fig.supxlabel(r"α", fontsize=16)
fig.supylabel(r"μ - $f_{(x)}$", fontsize=16)

plt.savefig("boxplots_friedman_i3samp.png")


# PSIS-LOO-CV
model_compare = az.compare(
    {
        "m10": idatas_at["10"]["0.25"],
        "m20": idatas_at["20"]["0.25"],
        "m50": idatas_at["50"]["0.25"],
        "m100": idatas_at["100"]["0.25"],
        "m200": idatas_at["200"]["0.25"],
    }
)

az.plot_compare(model_compare, figsize=(11, 5))
plt.savefig("loo_friedman_i3samp.png")


# Tree extraction
trees_length = {
    "10": {"0.1": {}, "0.25": {}, "0.5": {}},
    "20": {"0.1": {}, "0.25": {}, "0.5": {}},
    "50": {"0.1": {}, "0.25": {}, "0.5": {}},
    "100": {"0.1": {}, "0.25": {}, "0.5": {}},
    "200": {"0.1": {}, "0.25": {}, "0.5": {}},
}

for m in trees:
    for alpha in alphas:
        tmp_list = []
        idata = idatas_at[f"{m}"][f"{alpha}"].sample_stats.bart_trees
        for chain in idata:
            for sample in chain:
                for tree in sample:
                    index = max(tree.item().tree_structure.keys())
                    tmp_list.append(pmb.tree.BaseNode(index).depth)
        trees_length[f"{m}"][f"{alpha}"] = pd.Series(tmp_list)

# Trees' depth probabilities based on alpha values
prob_alphas = []
for alpha in alphas:
    q = compute_prior_probability(alpha)
    p = 1 - np.array(q)
    p = p / p.sum()
    prob_alphas.append(p)

# Frequency of trees depths
colors = ["C0", "C1", "C2"]
wd = 0.33
wd_lst = [0, wd, wd * 2]

# Individual plots
for m in trees:
    fig = plt.figure()
    for i in range(0, len(alphas)):
        # Trees Depth Frequencies
        frequency = (
            trees_length[f"{m}"][f"{alphas[i]}"]
            .value_counts(normalize=True)
            .sort_index(ascending=True)
        )
        plt.bar(
            frequency.index + wd_lst[i],
            frequency.values,
            color=colors[i],
            width=wd,
            edgecolor="k",
            alpha=0.9,
            label=rf"$\alpha$ = {alphas[i]}",
        )
        # Probabilities
        x = np.array(range(1, len(prob_alphas[i]) + 1)) + wd_lst[i]
        plt.scatter(
            x,
            prob_alphas[i],
            facecolor=colors[i],
            edgecolor="k",
            marker="o",
            s=80,
            zorder=2,
        )

    major_ticks = np.arange(0, 7, 1)
    ax.set_xticks(major_ticks)
    plt.ylim(0, 1)
    plt.xlim(0.5, 6.9)
    plt.title(f"m={m}")
    plt.legend()

    plt.savefig(f"friedman_i3samp_hist_m{m}.png")


# All frequencies in one plot
fig, axes = plt.subplots(1, 5, figsize=(25, 6), sharey=True)

for m, ax in zip(trees, axes.ravel()):
    for i in range(0, len(alphas)):
        # Trees Depth Frequencies
        frequency = (
            trees_length[f"{m}"][f"{alphas[i]}"]
            .value_counts(normalize=True)
            .sort_index(ascending=True)
        )
        ax.bar(
            frequency.index + wd_lst[i],
            frequency.values,
            color=colors[i],
            width=wd,
            edgecolor="k",
            alpha=0.9,
            label=rf"$\alpha$ = {alphas[i]}",
        )
        # Probabilities
        x = np.array(range(1, len(prob_alphas[i]) + 1)) + wd_lst[i]
        ax.scatter(
            x,
            prob_alphas[i],
            facecolor=colors[i],
            edgecolor="k",
            marker="o",
            s=80,
            zorder=2,
        )

    major_ticks = np.arange(0, 7, 1)
    ax.set_xticks(major_ticks)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, 6.9)
    ax.set_title(f"m={m}")
    if m == 200:
        ax.legend()

plt.savefig("friedman_i3samp_hist.png")
