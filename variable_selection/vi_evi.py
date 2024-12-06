import numpy as np
from arviz import hdi
from pymc_bart.utils import (
    _sample_posterior,
    compute_variable_importance,
    plot_variable_importance,
)
from scipy.stats import pearsonr


def vi_evi(bart_rvs, idatas, X, indices, method, samples, seed, figsize):
    """Compare the explicit vs implicit variable importance computation.

    Parameters
    ----------
    bart_rvs : list
        List of BART random variables.
    idatas : list
        List of inferencedatas with posterior samples.
    X : DataFrame
        Covariates matrix.
    indices : array-like
        The indices of the variables as computed by the implicit method.
    method: str
        The method used to compute the variable importance.
    samples : int
        The number of posterior samples to compute the R².
    seed : int
        The seed for the random number generator.
    figsize : tuple
        The figure size.
    """
    rng = np.random.default_rng(seed)
    pruning_results = compute_variable_importance(
        idatas[-1],
        bart_rvs[-1],
        X.iloc[:, indices],
        method=method,
        samples=samples,
        random_seed=rng,
    )

    ax = plot_variable_importance(
        pruning_results,
        figsize=figsize,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)

    predicted_all = _sample_posterior(
        bart_rvs[-1].owner.op.all_trees,
        X=X.iloc[:, indices].to_numpy(),
        rng=rng,
        size=samples,
    )
    ev_mean = np.zeros(X.shape[1])
    ev_hdi = np.zeros((X.shape[1], 2))

    for idx in range(X.shape[1]):
        predicted_subset = _sample_posterior(
            bart_rvs[idx].owner.op.all_trees,
            X=X.iloc[:, indices[: idx + 1]].to_numpy(),
            rng=rng,
            size=samples,
        )
        pearson = np.zeros(samples)
        for j in range(samples):
            pearson[j] = (
                (pearsonr(predicted_all[j].flatten(), predicted_subset[j].flatten())[0])
                ** 2
            )
        ev_mean[idx] = np.mean(pearson)
        ev_hdi[idx] = hdi(pearson)

    ticks = np.arange(X.shape[1], dtype=int)
    ax.errorbar(
        ticks,
        ev_mean,
        np.array((ev_mean - ev_hdi[:, 0], ev_hdi[:, 1] - ev_mean)),
        color="C1",
        alpha=0.8,
    )
    children = ax.get_children()
    ax.legend(
        [children[0], children[4]], ["Pruned trees", "Refitted model"], fontsize=11
    )

    return ax
