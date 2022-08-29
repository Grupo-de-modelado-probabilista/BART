# BART Experiments

## Environment set up
To set up the environment we used [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Once you have miniconda installed, run:
```bash
conda env create -f environment.yml
```

## Contributing

If you want to contribute to this repository, we recommend you set up `pre-commit`.
Note that the `environment.yml` already installs it, but we need to set it up in the repository:
```bash
pre-commit install
```

Now, for every new commit you make, pre-commit will fix some errors and will notify you of others that you need to resolve.
