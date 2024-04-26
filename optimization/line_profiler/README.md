# PyMC-BART Code Profiling

Welcome to the PyMC-BART line profiler.

## Getting started

Miniconda is used to setup the environments.

Once miniconda is installed, there is an `environment.yml` file to create an environment with all the dependencies needed to run the line profiling tests.

In order to provide line-by-line profiling, specific functions and methods (see the results section to see what was decorated) in the PyMC-BART codebase have been annotated with the `@profile` decorator. These annotations have been done on the branch `git+https://github.com/GStechschulte/pymc-bart.git@line-profiler`. Thus, the `environment.yml` installs PyMC-BART from this branch.

```bash
conda env create -f environment.yml
conda activate bart-line-profiler
```

To run the benchmarks

```bash
bash benchmark.sh
```

This will run line profiling benchmarks on the models in the `case_studies` directory with a variety of different hyperparameter values, and save the results in the `results` directory.

## Results

To view the results of any one result file, run

```bash
python -m line_profiler results/<filename>.lprof
```

There are several results within one line profile file. This is because several methods and functions have been decorated with the `@profile` decorator. In our case, we are interested in profiling the `.astep` method (and the subsequent call stack) of the `PGBART` sampler as this method represents the main entry point of the particle sampler.

Overall, eight functions and or methods are profiled.
- `astep`
    - `sample_tree`
        - `grow_tree`
            - `draw_leaf_value`
    - `update_weight`
        - `logp`
    - `resample`
        - `systematic`