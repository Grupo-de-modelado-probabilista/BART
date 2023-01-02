# This script is an adaptation from
# https://github.com/Grupo-de-modelado-probabilista/BART/blob/master/experiments/space_influenza.ipynb

import numpy as np
import pymc as pm
from pathlib import Path

import pymc_bart as pmb

import helper

args = helper.parse_args()

RANDOM_SEED = 8457
rng = np.random.RandomState(RANDOM_SEED)

sin = np.loadtxt(Path("data", "space_influenza.csv"), skiprows=1, delimiter=",")
X = sin[:, 1][:, None]
Y = sin[:, 2]

try:
    with pm.Model() as model:
        μ = pmb.BART("μ", X, Y, m=args.trees)
        p = pm.Deterministic("p", pm.math.sigmoid(μ))
        y = pm.Bernoulli("y", p=p, observed=Y)
        idata = pm.sample(
            step=[pmb.PGBART([μ], num_particles=args.particle)], random_seed=RANDOM_SEED
        )
except Exception as e:
    raise RuntimeError("Issue running model") from e
