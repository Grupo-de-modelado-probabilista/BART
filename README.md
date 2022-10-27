# BART Experiments

## Environment set up
To set up the environments we used [miniconda](https://docs.conda.io/en/latest/miniconda.html).

Once you have miniconda installed, there is an `environment.yml` file in each folder to create an environment designed to run the tests in each one of the folders. These environments can be created from within each folder with:

```bash
conda env create -f environment.yml
```

## Citation

To cite PyMC-BART and/or the results in the folder `experiments` please use [![arXiv](https://img.shields.io/badge/arXiv-2206.03619-b31b1b.svg)](https://arxiv.org/abs/2206.03619)

Here is the citation in BibTeX format

```
@misc{quiroga2022bart,
title={Bayesian additive regression trees for probabilistic programming},
author={Quiroga, Miriana and Garay, Pablo G and Alonso, Juan M. and Loyola, Juan Martin and Martin, Osvaldo A},
year={2022},
doi={10.48550/ARXIV.2206.03619},
archivePrefix={arXiv},
primaryClass={stat.CO}
}
```

## Contributing

If you want to contribute to this repository, we recommend you set up `pre-commit`.
Note that the `environment.yml` already installs it, but we need to set it up in the repository for each environment:
```bash
pre-commit install
```

Now, for every new commit you make, pre-commit will fix some errors and will notify you of others that you need to resolve.
