# This script is an adaptation from
# https://github.com/Grupo-de-modelado-probabilista/BART/blob/master/experiments/friedman.ipynb

import numpy as np
import pymc as pm
import pymc_bart as pmb

import helper

args = helper.parse_args()

RANDOM_SEED = 4579
np.random.seed(RANDOM_SEED)

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
        μ = pmb.BART("μ", X[:, :100], Y, m=args.trees)
        σ = pm.HalfNormal("σ", 1)
        y = pm.Normal("y", mu=μ, sigma=σ, observed=Y)
        idata = pm.sample(
            step=[pmb.PGBART([μ], num_particles=args.particle)], random_seed=RANDOM_SEED
        )
except Exception as e:
    raise RuntimeError("Issue running model") from e
