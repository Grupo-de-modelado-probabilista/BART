#!/usr/bin/env python
# coding: utf-8

# Packages imports
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# General settings
RANDOM_SEED = 4579
np.random.seed(RANDOM_SEED)
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
Y_lin = np.random.normal(funcs[0](X_lin), 2, size=200)
X_lin = X_lin[:, None]

X_sin = np.linspace(0, 1, 200)
Y_sin = np.random.normal(funcs[1](X_sin), 2, size=200)
X_sin = X_sin[:, None]

X_stp = np.linspace(0, 1, 200)
Y_stp = np.random.normal(funcs[2](X_stp), 0.2, size=200)
X_stp = X_stp[:, None]

XS = [X_lin, X_sin, X_stp]
YS = [Y_lin, Y_sin, Y_stp]

# Run model
idatas = []
m_trees = []
for X, Y in zip(XS, YS):
    for m in [10, 50, 200]:
        with pm.Model() as functions:
            σ = pm.HalfNormal("σ", Y.std())
            μ = pmb.BART("μ", X, Y, m=m)
            y = pm.Normal("y", μ, σ, observed=Y)
            idata = pm.sample(
                chains=4,
                random_seed=RANDOM_SEED,
                compute_convergence_checks=False,
                idata_kwargs={"log_likelihood": True},
            )
            idatas.append(idata)
            m_trees.append(μ.owner.op.m)

# Plot function
_, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True)

for idata, m, ax, X, Y, f in zip(
    idatas,
    m_trees,
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
    ax.set_title(f"{m=}")
    plt.savefig("simple_functions.png")

# LOO compare
comp = az.compare(dict(zip(["m=10", "m=50", "m=200"], idatas[:3])))
ax = az.plot_compare(
    comp, plot_ic_diff=False, insample_dev=False, figsize=(10, 2.5), legend=False
)
plt.savefig("LOO_lin.png")

# Free memory
del comp, idata, idatas, μ


# Bikes example
# data
bikes = pd.read_csv("bikes.csv")

X = bikes[["hour", "temperature", "humidity", "windspeed"]]
Y = bikes["count"]

# run model
with pm.Model() as model_bikes:
    α = pm.Exponential("α", 0.1)
    μ_ = pmb.BART("μ_", X, np.log(Y), m=50)
    μ = pm.Deterministic("μ", np.exp(μ_))
    y = pm.NegativeBinomial("y", mu=μ, alpha=α, observed=Y)
    idata_bikes = pm.sample(
        chains=4, random_seed=RANDOM_SEED, compute_convergence_checks=False
    )

# Trace
az.plot_trace(idata_bikes, compact=False, var_names=["α"], kind="rank_bars")
plt.savefig("trace_bikes.png", bbox_inches="tight")

# Convergence
pmb.plot_convergence(idata_bikes, var_name="μ")
plt.savefig("bikes_diagnostics_bart_rv.png")

# Partial dependence plot
pmb.plot_pdp(μ_, X=X, Y=Y, grid=(2, 2), func=np.exp)
plt.savefig("bikes_pdp.png", bbox_inches="tight")

# Variable Importance
labels = ["hour", "temperature", "humidity", "windspeed"]
pmb.utils.plot_variable_importance(idata_bikes, μ_, X, samples=100)
plt.savefig("bikes_VI-correlation.png")

# Free memory
del idata_bikes, μ, μ_


# Bikes example with different number of trees
trees = [20, 50, 100]
idatas_bikes = {}
VIs = []

# Run model
for m in trees:
    with pm.Model() as model_bikes:
        α = pm.Exponential("α", 0.1)
        μ_ = pmb.BART("μ_", X, np.log(Y), m=m)
        μ = pm.Deterministic("μ", np.exp(μ_))
        y = pm.NegativeBinomial("y", mu=μ, alpha=α, observed=Y)
        idata = pm.sample(tune=2000, draws=2000, chains=4, random_seed=RANDOM_SEED)
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
X = np.random.uniform(low=0, high=1.0, size=(100, 1000))
f_x = (
    10 * np.sin(np.pi * X[:, 0] * X[:, 1])
    + 20 * (X[:, 2] - 0.5) ** 2
    + 10 * X[:, 3]
    + 5 * X[:, 4]
)
Y = np.random.normal(f_x, 1)

# Different number of variables
num_covariables = ["5", "10", "100", "1000"]
idatas = {}
all_trees = {}
VIs = []
for num_covariable in num_covariables:
    with pm.Model() as model:
        σ = pm.HalfNormal("σ", 1)
        μ = pmb.BART("μ", X[:, : int(num_covariable)], Y, m=200)
        y = pm.Normal("y", mu=μ, sigma=σ, observed=Y)
        idata = pm.sample(
            chains=4, compute_convergence_checks=False, random_seed=RANDOM_SEED
        )
        idatas[num_covariable] = idata
        all_trees[num_covariable] = μ
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
Y_new = np.random.normal(f_x_new, 1)

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
        μ_pred = pmb.utils._sample_posterior(
            all_trees[name].owner.op.all_trees,
            X_new[:, : int(name)],
            np.random.default_rng(RANDOM_SEED),
            size=500,
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
    pmb.plot_pdp(all_trees[num], X, Y, var_idx=var_id, grid=(1, l), figsize=(10, 2))
    plt.ylim(10, 20)
    plt.savefig(f"pdps_friedman_{num}.png", bbox_inches="tight")

# Free memory
del idata, idatas, all_trees


# Friedman trees test

# Data generation
X = np.random.uniform(low=0, high=1.0, size=(100, 10))
f_x = (
    10 * np.sin(np.pi * X[:, 0] * X[:, 1])
    + 20 * (X[:, 2] - 0.5) ** 2
    + 10 * X[:, 3]
    + 5 * X[:, 4]
)
Y = np.random.normal(f_x, 1)

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
del idata, idatas, μ


# Friedman test trees, alphas and betas

trees = [10, 20, 50, 100, 200]
alphas = [0.1, 0.45, 0.95]
betas = [1, 2, 10]
idatas_at = {
    "10": {
        "0.1": {"1": {}, "2": {}, "10": {}},
        "0.45": {"1": {}, "2": {}, "10": {}},
        "0.95": {"1": {}, "2": {}, "10": {}},
    },
    "20": {
        "0.1": {"1": {}, "2": {}, "10": {}},
        "0.45": {"1": {}, "2": {}, "10": {}},
        "0.95": {"1": {}, "2": {}, "10": {}},
    },
    "50": {
        "0.1": {"1": {}, "2": {}, "10": {}},
        "0.45": {"1": {}, "2": {}, "10": {}},
        "0.95": {"1": {}, "2": {}, "10": {}},
    },
    "100": {
        "0.1": {"1": {}, "2": {}, "10": {}},
        "0.45": {"1": {}, "2": {}, "10": {}},
        "0.95": {"1": {}, "2": {}, "10": {}},
    },
    "200": {
        "0.1": {"1": {}, "2": {}, "10": {}},
        "0.45": {"1": {}, "2": {}, "10": {}},
        "0.95": {"1": {}, "2": {}, "10": {}},
    },
}

all_trees_at = {
    "10": {
        "0.1": {"1": {}, "2": {}, "10": {}},
        "0.45": {"1": {}, "2": {}, "10": {}},
        "0.95": {"1": {}, "2": {}, "10": {}},
    },
    "20": {
        "0.1": {"1": {}, "2": {}, "10": {}},
        "0.45": {"1": {}, "2": {}, "10": {}},
        "0.95": {"1": {}, "2": {}, "10": {}},
    },
    "50": {
        "0.1": {"1": {}, "2": {}, "10": {}},
        "0.45": {"1": {}, "2": {}, "10": {}},
        "0.95": {"1": {}, "2": {}, "10": {}},
    },
    "100": {
        "0.1": {"1": {}, "2": {}, "10": {}},
        "0.45": {"1": {}, "2": {}, "10": {}},
        "0.95": {"1": {}, "2": {}, "10": {}},
    },
    "200": {
        "0.1": {"1": {}, "2": {}, "10": {}},
        "0.45": {"1": {}, "2": {}, "10": {}},
        "0.95": {"1": {}, "2": {}, "10": {}},
    },
}

# run model
for m in trees:
    for alpha in alphas:
        for beta in betas:
            with pm.Model() as model:
                μ = pmb.BART("μ", X, Y, m=m, alpha=alpha, beta=beta)
                σ = pm.HalfNormal("σ", 1)
                y = pm.Normal("y", μ, σ, observed=Y)
                idata = pm.sample(
                    chains=4,
                    compute_convergence_checks=False,
                    idata_kwargs={"log_likelihood": True},
                    random_seed=RANDOM_SEED,
                )
                idatas_at[str(m)][str(alpha)][str(beta)] = idata
                all_trees_at[str(m)][str(alpha)][str(beta)] = list(μ.owner.op.all_trees)

# boxplot
fig, axes = plt.subplots(
    len(alphas), len(trees), figsize=(10, 9), sharey=True, sharex=True
)
axes = axes.ravel()
i = 0
for alpha in alphas:
    for m in trees:
        ax = axes[i]
        means = [
            idatas_at[str(m)][str(alpha)][str(beta)]["posterior"]["μ"].mean(
                ("chain", "draw")
            )
            - Y
            for beta in betas
        ]
        box = ax.boxplot(
            means,
            notch=True,
            patch_artist=True,
            widths=0.5,
            labels=betas,
            showfliers=False,
            medianprops=dict(color="k"),
        )
        for patch, color in zip(box["boxes"], ["C0", "C1", "C2"]):
            patch.set_facecolor(color)
            ax.set_title(f"m = {m}")
        i += 1

axes[0].set_ylabel("α = 0.1")
axes[5].set_ylabel("α = 0.45")
axes[10].set_ylabel("α = 0.95")
fig.supxlabel(r"β", fontsize=16)
fig.supylabel(r"μ - Y$_{(data)}$", fontsize=16)

plt.savefig("boxplots_friedman_i2.png")

# Free memory
del idata, idatas_at, all_trees_at, μ


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
y_data = hist

# run model
with pm.Model() as model_coal:
    μ_ = pmb.BART("μ_", X=x_data, Y=np.log(y_data), m=20)
    μ = pm.Deterministic("μ", np.exp(μ_))
    y_pred = pm.Poisson("y_pred", mu=μ, observed=y_data)
    idata_coal = pm.sample(
        chains=4, random_seed=RANDOM_SEED, compute_convergence_checks=False
    )

# plot
_, ax = plt.subplots(figsize=(10, 6))

rates = idata_coal.posterior["μ"] / 4
rate_mean = idata_coal.posterior["μ"].mean(dim=["draw", "chain"]) / 4

ax.plot(x_centers, y_data / 4, "k.")
az.plot_hdi(x_centers, rates, smooth=True, color="k", hdi_prob=0.05)
az.plot_hdi(x_centers, rates, smooth=True, color="C0")
az.plot_hdi(
    x_centers, rates, hdi_prob=0.5, smooth=True, color="C0", plot_kwargs={"alpha": 0}
)

ax.set_xlabel("years")
ax.set_ylabel("rate")
plt.savefig("coal_mining.png")

# Free memory
del idata_coal, μ_, μ


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
    σ_ = pmb.BART("σ_", X, np.log(Y), m=50)
    σ = pm.Deterministic("σ", np.exp(σ_))
    y = pm.Normal("y", μ, σ, observed=Y)
    idata = pm.sample(
        chains=4, compute_convergence_checks=False, random_seed=RANDOM_SEED
    )

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
