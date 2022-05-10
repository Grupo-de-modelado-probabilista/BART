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
pip install --no-build-isolation --editable .
```
