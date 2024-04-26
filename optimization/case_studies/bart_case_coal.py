# This script is an adaptation from
# https://github.com/Grupo-de-modelado-probabilista/BART/blob/master/experiments/Bikes.ipynb

import argparse

from pathlib import Path

import numpy as np
import pymc as pm
import pymc_bart as pmb


def main(args):
    coal = np.loadtxt(Path("../..", "experiments", "coal.csv").resolve())

    # Discretize data
    years = int(coal.max() - coal.min())
    bins = years // 4
    hist, x_edges = np.histogram(coal, bins=bins)
    # Compute the location of the centers of the discrete data
    x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
    # X data needs to be 2D for BART
    x_data = x_centers[:, None]
    # Express data as the rate number of disaster per year
    y_data = hist / 4

    try:
        with pm.Model() as model_coal:
            μ_ = pmb.BART("μ_", X=x_data, Y=y_data, m=args.trees)
            μ = pm.Deterministic("μ", np.abs(μ_))
            y_pred = pm.Poisson("y_pred", mu=μ, observed=y_data)
            step = pmb.PGBART([μ_], num_particles=args.particle)
    except Exception as e:
        raise RuntimeError("Issue running model") from e

    for iter in range(args.iters):
        step.astep(iter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", type=int, default=50, help="Number of trees")
    parser.add_argument("--particle", type=int, default=20, help="Number of particles")
    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations")
    args = parser.parse_args()
    main(args)
