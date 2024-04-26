# This script is an adaptation from
# https://github.com/Grupo-de-modelado-probabilista/BART/blob/master/experiments/Bikes.ipynb

import argparse

from pathlib import Path

import pandas as pd
import pymc as pm
import pymc_bart as pmb


def main(args):
    bikes = pd.read_csv(Path("../..", "experiments", "bikes.csv"))

    X = bikes[["hour", "temperature", "humidity", "windspeed"]]
    Y = bikes["count"]

    try:
        with pm.Model() as model:
            σ = pm.HalfNormal("σ", Y.std())
            μ = pmb.BART("μ", X, Y, m=args.trees)
            y = pm.Normal("y", μ, σ, observed=Y)
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