# This script is an adaptation from
# https://github.com/Grupo-de-modelado-probabilista/BART/blob/master/experiments/friedman.ipynb

import argparse

import numpy as np
import pymc as pm
import pymc_bart as pmb


def main(args):

    X = np.random.uniform(low=0, high=1.0, size=(100, 5))
    f_x = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    Y = np.random.normal(f_x, 1)

    try:
        with pm.Model() as model:
            μ = pmb.BART("μ", X, Y, m=args.trees)
            σ = pm.HalfNormal("σ", 1)
            y = pm.Normal("y", mu=μ, sigma=σ, observed=Y)
            step = pmb.PGBART([μ], num_particles=args.particle)
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