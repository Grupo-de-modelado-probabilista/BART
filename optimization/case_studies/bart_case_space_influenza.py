# This script is an adaptation from
# https://github.com/Grupo-de-modelado-probabilista/BART/blob/master/experiments/space_influenza.ipynb

import argparse

import numpy as np
import pymc as pm
from pathlib import Path

import pymc_bart as pmb


def main(args):

    sin = np.loadtxt(
        Path("../..",  "experiments", "space_influenza.csv"), 
        skiprows=1, 
        delimiter=","
    )
    X = sin[:, 1][:, None]
    Y = sin[:, 2]

    try:
        with pm.Model() as model:
            μ = pmb.BART("μ", X, Y, m=args.trees)
            p = pm.Deterministic("p", pm.math.sigmoid(μ))
            y = pm.Bernoulli("y", p=p, observed=Y)
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
