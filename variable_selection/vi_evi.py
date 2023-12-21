import numpy as np
from scipy.stats import pearsonr
from arviz import hdi

from pymc_bart.utils import plot_variable_importance, _sample_posterior


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

    _, ax = plot_variable_importance(
        idatas[-1],
        bart_rvs[-1],
        X.iloc[:, indices],
        method=method,
        samples=samples,
        xlabel_angle=45,
        figsize=figsize,
        random_seed=seed,
    )

    predicted_all = _sample_posterior(
        bart_rvs[-1].owner.op.all_trees,
        X=X.iloc[:, indices].values,
        rng=rng,
        size=samples,
    )
    ev_mean = np.zeros(X.shape[1])
    ev_hdi = np.zeros((X.shape[1], 2))

    for idx in range(X.shape[1]):
        predicted_subset = _sample_posterior(
            bart_rvs[idx].owner.op.all_trees,
            X=X.iloc[:, indices[: idx + 1]].values,
            rng=rng,
            size=samples,
        )
        pearson = np.zeros(samples)
        for j in range(samples):
            pearson[j] = (
                pearsonr(predicted_all[j].flatten(), predicted_subset[j].flatten())[0]
            ) ** 2
        ev_mean[idx] = np.mean(pearson)
        ev_hdi[idx] = hdi(pearson)

    ticks = np.arange(X.shape[1], dtype=int)
    ax.errorbar(
        ticks,
        ev_mean,
        np.array((ev_mean - ev_hdi[:, 0], ev_hdi[:, 1] - ev_mean)),
        color="C1",
        alpha=0.5,
    )

    return ax
