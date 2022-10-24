#!/usr/bin/env python
# coding: utf-8

# Packages imports
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
from pymc_bart.pgbart import compute_prior_probability


# General settings
RANDOM_SEED = 8457
rng = np.random.RandomState(RANDOM_SEED)
az.style.use("arviz-white")
plt.rcParams["figure.dpi"] = 300

# Simple function

# Data generation
funcs = [
    lambda x: 10 * x,
    lambda x: 10 * np.sin(x * 2 * np.pi),
    lambda x: 2 - 4 * np.where(x < 0.5, 1, 0),
]

X_lin = np.linspace(0, 1, 200)
Y_lin = rng.normal(funcs[0](X_lin), 2, size=200)
X_lin = X_lin[:, None]

X_sin = np.linspace(0, 1, 200)
Y_sin = rng.normal(funcs[1](X_sin), 2, size=200)
X_sin = X_sin[:, None]

X_stp = np.linspace(0, 1, 200)
Y_stp = np.random.normal(funcs[2](X_stp), 0.2, size=200)

X_stp = X_stp[:, None]

XS = [X_lin, X_sin, X_stp]
YS = [Y_lin, Y_sin, Y_stp]

# Run model
idatas = []
for X, Y in zip(XS, YS):
    for m in [10, 50, 200]:
        with pm.Model() as functions:
            σ = pm.HalfNormal("σ", Y.std())
            μ = pmb.BART("μ", X, Y, m=m)
            y = pm.Normal("y", μ, σ, observed=Y)
            idata = pm.sample(random_seed=RANDOM_SEED)
            idatas.append(idata)

# Plot function
_, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True)

for idata, ax, X, Y, f in zip(
    idatas,
    np.ravel(axes),
    np.repeat(XS, 3, 0),
    np.repeat(YS, 3, 0),
    np.repeat(funcs, 3),
):
    mean = idata.posterior["μ"].mean(dim=["draw", "chain"])
    ax.plot(X[:, 0], mean, lw=3)
    az.plot_hdi(X[:, 0], idata.posterior["μ"], color="C0", smooth=False, ax=ax)
    ax.plot(X[:, 0], Y, ".", zorder=1)
    ax.plot(X[:, 0], f(X), "k--")
    ax.set_title(f"m={len(idata.sample_stats.bart_trees_dim_0)}")
    plt.savefig("step_function.png")

# LOO compare
comp = az.compare(dict(zip(["m=10", "m=50", "m=200"], idatas[:3])))
ax = az.plot_compare(
    comp, plot_ic_diff=False, insample_dev=False, figsize=(10, 2.5), legend=False
)
plt.savefig("LOO_lin.png")

#
comp = az.compare(dict(zip(["m=10", "m=50", "m=200"], idatas[3:6])))
ax = az.plot_compare(
    comp, plot_ic_diff=False, insample_dev=False, figsize=(10, 2.5), legend=False
)
plt.savefig("LOO_sin.png")

#
comp = az.compare(dict(zip(["m=10", "m=50", "m=200"], idatas[6:])))
ax = az.plot_compare(
    comp, plot_ic_diff=False, insample_dev=False, figsize=(10, 2.5), legend=False
)
plt.savefig("LOO_stp.png")

# Free memory
del comp, idata, idatas


# Bikes example
# data
bikes = pd.read_csv("bikes.csv")

X = bikes[["hour", "temperature", "humidity", "windspeed"]]
Y = bikes["count"]

# run model
with pm.Model() as model_bikes:
    σ = pm.HalfNormal("σ", Y.std())
    μ = pmb.BART("μ", X, Y, m=50)
    y = pm.Normal("y", μ, σ, observed=Y)
    idata_bikes = pm.sample(random_seed=RANDOM_SEED)

# Trace
az.plot_trace(idata_bikes)
plt.savefig("trace_bikes.png", bbox_inches="tight")

# Partial dependence plot
pmb.plot_dependence(idata_bikes, X=X, Y=Y, grid=(2, 2))
plt.savefig("partial_dependence_plot_bikes.png", bbox_inches="tight")

# Variable Importance
labels = ["hour", "temperature", "humidity", "windspeed"]
pmb.utils.plot_variable_importance(idata_bikes, X.values, labels, samples=100)
plt.savefig("bikes_VI-correlation.png")


# Free memory
del idata_bikes


# Bikes example with different and m
trees = [20, 50, 100]
idatas_bikes = {}
VIs = []

# Run model
for m in trees:
    with pm.Model() as model_bikes:
        σ = pm.HalfNormal("σ", Y.std())
        μ = pmb.BART("μ", X, Y, m=m)
        y = pm.Normal("y", μ, σ, observed=Y)
        idata = pm.sample(chains=4, random_seed=RANDOM_SEED)
        idatas_bikes[str(m)] = idata
        # Variable importance
        VI = idata.sample_stats["variable_inclusion"].mean(("chain", "draw")).values
        VIs.append(VI / VI.sum())

# Plot Variable importance
fig, ax = plt.subplots(sharey=True, figsize=(12, 4))

for tree, vi in zip(trees, VIs):
    plt.plot(vi, label=f"m={tree}", lw=3, marker="o")

plt.axhline(1 / X.shape[1], ls="--", color="k")

plt.legend()
plt.ylabel("Relative variable importance")
plt.xticks(
    ticks=list(range(X.shape[1])), labels=[label for label in X.columns], fontsize=15
)

plt.savefig("bart_vi_bikes.png")

# Free memory
del idata, idatas_bikes


# Approximation to Friedman's five dimension function

# Data generation
X = rng.uniform(low=0, high=1.0, size=(100, 1000))
f_x = (
    10 * np.sin(np.pi * X[:, 0] * X[:, 1])
    + 20 * (X[:, 2] - 0.5) ** 2
    + 10 * X[:, 3]
    + 5 * X[:, 4]
)
Y = rng.normal(f_x, 1)

# Different number of variables
num_covariables = ["5", "10", "100", "1000"]
idatas = {}
VIs = []
for num_covariable, tn, pn in zip(
    num_covariables, (1000, 1000, 2000, 2000), (60, 60, 60, 60)
):
    with pm.Model() as model:
        μ = pmb.BART("μ", X[:, : int(num_covariable)], Y, m=200)
        σ = pm.HalfNormal("σ", 1)
        y = pm.Normal("y", mu=μ, sigma=σ, observed=Y)
        idata = pm.sample(
            tune=tn,
            chains=4,
            random_seed=RANDOM_SEED,
            step=[pmb.PGBART([μ], num_particles=pn)],
        )
        idatas[num_covariable] = idata
        VI = idata.sample_stats["variable_inclusion"].mean(("chain", "draw")).values
        VIs.append(VI / VI.sum())

# Out-of-sample data
X_new = np.random.uniform(low=0.0, high=1.0, size=(100, 1000))
f_x_new = (
    10 * np.sin(np.pi * X_new[:, 0] * X_new[:, 1])
    + 20 * (X_new[:, 2] - 0.5) ** 2
    + 10 * X_new[:, 3]
    + 5 * X_new[:, 4]
)
Y_new = rng.normal(f_x_new, 1)

# plot
fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)

for i, (name, ax) in enumerate(zip(num_covariables * 2, axes.ravel())):
    ax.axline([0, 0], [1, 1], color="0.5")
    if i <= 3:  # in-sample
        ax.set_title(f"p={name}")
        μs = idatas[name].posterior["μ"].stack(samples=["chain", "draw"])
        mean = μs.mean("samples")
        hdi = az.hdi(μs.T.values, hdi_prob=0.9)
        yerr = np.vstack([mean - hdi[:, 0], hdi[:, 1] - mean])
        ax.errorbar(f_x, mean, yerr, linestyle="None", marker=".", alpha=0.5)
    elif i > 3:  # Out-of-sample
        μ_pred = pmb.predict(
            idatas[name], rng, X_new[:, : int(name)], size=500
        ).squeeze()
        mean = μ_pred.mean(0)
        hdi = az.hdi(μ_pred, hdi_prob=0.9)
        yerr = np.vstack([mean - hdi[:, 0], hdi[:, 1] - mean])
        ax.errorbar(
            f_x_new,
            mean,
            yerr,
            linestyle="None",
            marker=".",
            alpha=0.5,
        )

    fig.text(0.42, 1.0, "in-sample")
    fig.text(0.42, 0.49, "out-of-sample")
    fig.text(0.42, -0.02, "observed (f_x)")
    fig.text(-0.02, 0.42, "predicted (f_x)", rotation=90)

fig.tight_layout(h_pad=2)
plt.savefig("friedman_covar_in_out_sample.png", bbox_inches="tight")

# Variable Importance
fig = plt.figure()
for idx, (VI, num_covariable) in enumerate(zip(VIs, num_covariables)):
    plt.plot(VI[:5], label=num_covariable, color=f"C{idx}")
    plt.plot(5, np.mean(VI[5:]), ".")
plt.legend()
plt.savefig("friedman_VI.png")

# Partial dependence plots
lim = [5, 10, 10, 10]

fig = plt.figure()

for num, l in zip(num_covariables, lim):
    var_id = range(min(10, int(num)))
    pmb.plot_dependence(
        idatas[num], X, Y, rug=False, var_idx=var_id, grid=(1, l), figsize=(10, 2)
    )
    plt.ylim(5, 20)
    plt.savefig(f"pdps_friedman_{num}.png", bbox_inches="tight")

# Free memory
del idata, idatas


# Friedman trees test

# Data generation
X = rng.uniform(low=0, high=1.0, size=(100, 10))
f_x = (
    10 * np.sin(np.pi * X[:, 0] * X[:, 1])
    + 20 * (X[:, 2] - 0.5) ** 2
    + 10 * X[:, 3]
    + 5 * X[:, 4]
)
Y = rng.normal(f_x, 1)

idatas = {}
trees = [10, 20, 50, 100, 200]
VIs = []

# Run model
for m in trees:
    with pm.Model() as model:
        μ = pmb.BART("μ", X, Y, m=m)
        σ = pm.HalfNormal("σ", 1)
        y = pm.Normal("y", μ, σ, observed=Y)
        idata = pm.sample(chains=4, random_seed=RANDOM_SEED)
        idatas[str(m)] = idata
        # Variable importance
        VI = idata.sample_stats["variable_inclusion"].mean(("chain", "draw")).values
        VIs.append(VI / VI.sum())

# Plot VIs
fig = plt.figure()
for tree, vi in zip(trees, VIs):
    plt.plot(vi, label=f"trees = {tree}", marker="o", linestyle="dashed")

plt.axhline(1 / X.shape[1], ls="--", color="k")
plt.legend()
plt.savefig("var_importance.png")

# Free memory
del idata, idatas


# Friedman test trees and alphas

trees = [10, 20, 50, 100, 200]
alphas = [0.1, 0.25, 0.5]
idatas_at = {
    "10": {"0.1": {}, "0.25": {}, "0.5": {}},
    "20": {"0.1": {}, "0.25": {}, "0.5": {}},
    "50": {"0.1": {}, "0.25": {}, "0.5": {}},
    "100": {"0.1": {}, "0.25": {}, "0.5": {}},
    "200": {"0.1": {}, "0.25": {}, "0.5": {}},
}

# run model
for m in trees:
    for alpha in alphas:
        with pm.Model() as model:
            μ = pmb.BART("μ", X, Y, m=m, alpha=alpha)
            σ = pm.HalfNormal("σ", 1)
            y = pm.Normal("y", μ, σ, observed=Y)
            idata = pm.sample(chains=4, random_seed=RANDOM_SEED)
            idatas_at[str(m)][str(alpha)] = idata

# plot
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

plt.savefig("boxplots_friedman_i2.png")

# Model compare
model_compare = az.compare(
    {
        "m10": idatas_at["10"]["0.25"],
        "m20": idatas_at["20"]["0.25"],
        "m50": idatas_at["50"]["0.25"],
        "m100": idatas_at["100"]["0.25"],
        "m200": idatas_at["200"]["0.25"],
    }
)

ax = az.plot_compare(
    model_compare,
    plot_ic_diff=False,
    insample_dev=False,
    figsize=(10, 2.5),
    legend=False,
)

plt.savefig("loo_friedman_i2.png")

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

plt.savefig("friedman_i2_hist.png")

# Free memory
del idata, idatas_at, model_compare


# Coal mining disaster

# Load data
coal = np.loadtxt("coal.csv")
# discretize data
years = int(coal.max() - coal.min())
bins = years // 4
hist, x_edges = np.histogram(coal, bins=bins)
# compute the location of the centers of the discretized data
x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
# xdata needs to be 2D for BART
x_data = x_centers[:, None]
# express data as the rate number of disaster per year
y_data = hist / 4

# run model
with pm.Model() as model_coal:
    μ_ = pmb.BART("μ_", X=x_data, Y=y_data, m=20)
    μ = pm.Deterministic("μ", np.abs(μ_))
    y_pred = pm.Poisson("y_pred", mu=μ, observed=y_data)
    idata_coal = pm.sample(random_seed=RANDOM_SEED)

# plot
_, ax = plt.subplots(figsize=(10, 6))

rates = idata_coal.posterior["μ"]
rate_mean = idata_coal.posterior["μ"].mean(dim=["draw", "chain"])
ax.plot(x_centers, rate_mean, "C0", lw=3)
az.plot_hdi(x_centers, rates, smooth=False, color="C0")
az.plot_hdi(
    x_centers, rates, hdi_prob=0.5, smooth=False, color="C0", plot_kwargs={"alpha": 0}
)
ax.plot(coal, np.zeros_like(coal) - 0.5, "k|")
ax.set_xlabel("years")
ax.set_ylabel("rate")
plt.savefig("coal_mining.png")

# Free memory
del idata_coal


# Variance

# Load data
data = np.loadtxt("marketing.csv", skiprows=1, delimiter=",")
X = data[:, 0, None]
Y = data[:, 1]

# run model
with pm.Model() as model:
    α = pm.HalfNormal("α", 50)
    β = pm.HalfNormal("β", 5)
    μ = pm.Deterministic("μ", np.sqrt(α + β * X[:, 0]))
    σ_ = pmb.BART("σ_", X, Y, m=50)
    σ = pm.Deterministic("σ", np.abs(σ_))
    y = pm.Normal("y", μ, σ, observed=Y)
    idata = pm.sample(target_accept=0.99, random_seed=RANDOM_SEED)

# plot
_, ax = plt.subplots(figsize=(10, 6))

mean = idata.posterior["μ"].mean(dim=["draw", "chain"]).values
mean_s = idata.posterior["μ"].stack(samples=["draw", "chain"]).values
idx = np.argsort(X[:, 0])
ax.plot(X[:, 0][idx], mean[idx], "k", lw=3)
az.plot_hdi(X[:, 0], mean_s.T, color="k")
sigma = idata.posterior["σ"].mean(dim=["draw", "chain"]).values
ax.fill_between(X[:, 0][idx], mean[idx] - sigma[idx], mean[idx] + sigma[idx], alpha=0.5)

ax.plot(X[:, 0], Y, "C1o", zorder=0)
ax.set_xlabel("budget")
ax.set_ylabel("sales")
plt.savefig("marketing.png")
