# BART Experiments

## Environment set up
To set up the environment we used [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Once you have miniconda installed, run:
```bash
conda env create -f environment.yml
```

Finally, we need to clone the [pymc-experimental](https://github.com/pymc-devs/pymc-experimental) repository where the implementation of BART is and install it in our environment in editable mode.
For that, run:

```bash
# Go the the directory where you want to clone the pymc-experimental repository
cd ~
# Clone the repository
git clone https://github.com/pymc-devs/pymc-experimental.git
# Move to the repository
cd pymc-experimental
# Activate the environment
conda activate bart-experiments
# Install pymc-experimental in editable mode
pip install --editable .
```

## Contributing

If you want to contribute to this repository, we recommend you set up `pre-commit`.
Note that the `environment.yml` already installs it, but we need to set it up in the repository:
```bash
pre-commit install
```

Now, for every new commit you make, pre-commit will fix some errors and will notify you of others that you need to resolve.
