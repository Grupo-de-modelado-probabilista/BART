# This script is an adaptation from
# https://github.com/Grupo-de-modelado-probabilista/BART/blob/master/experiments/Bikes.ipynb

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb

import helper

args = helper.parse_args()

RANDOM_SEED = 8457
rng = np.random.RandomState(RANDOM_SEED)

bikes = pd.read_csv(Path("case_studies", "bikes.csv"))

X = bikes[["hour", "temperature", "humidity", "windspeed"]]
Y = bikes["count"]

try:
    with pm.Model() as model:
        σ = pm.HalfNormal("σ", Y.std())
        μ = pmb.BART("μ", X, Y, m=args.trees)
        y = pm.Normal("y", μ, σ, observed=Y)
        pm.sample(
            step=[pmb.PGBART([μ], num_particles=args.particle)],
            random_seed=RANDOM_SEED,
            cores=args.cores,
        )
except Exception as e:
    raise RuntimeError("Issue running model") from e
