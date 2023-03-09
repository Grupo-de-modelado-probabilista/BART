# This script is an adaptation from
# https://github.com/Grupo-de-modelado-probabilista/BART/blob/master/experiments/Bikes.ipynb

from pathlib import Path

import numpy as np
import pymc as pm
import pymc_bart as pmb

import helper

args = helper.parse_args()

RANDOM_SEED = 8457
rng = np.random.RandomState(RANDOM_SEED)

coal = np.loadtxt(Path("case_studies", "coal.csv"))

# discretize data
years = int(coal.max() - coal.min())
bins = years // 4
hist, x_edges = np.histogram(coal, bins=bins)
# compute the location of the centers of the discrete data
x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
# xdata needs to be 2D for BART
x_data = x_centers[:, None]
# express data as the rate number of disaster per year
y_data = hist / 4

try:
    with pm.Model() as model_coal:
        μ_ = pmb.BART("μ_", X=x_data, Y=y_data, m=args.trees)
        μ = pm.Deterministic("μ", np.abs(μ_))
        y_pred = pm.Poisson("y_pred", mu=μ, observed=y_data)
        idata_coal = pm.sample(
            step=[pmb.PGBART([μ_], num_particles=args.particle)],
            random_seed=RANDOM_SEED,
            cores=args.cores,
        )
except Exception as e:
    raise RuntimeError("Issue running model") from e
