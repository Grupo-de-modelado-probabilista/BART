# PyMC-BART Code Profiling

Welcome to the PyMC-BART memory profiler.

## Getting started

Miniconda is used to setup the environment.

Once miniconda is installed, there is an `environment.yml` file to create an environment with all the dependencies needed to run the line profiling tests.

```bash
conda env create -f environment.yml
conda activate bart-mem-profiler
```

`memray` is used to profile the memory usage of the `PGBART` sampler. To run the memory benchmarks

The memory profile is obtained by running

```bash
bash benchmark.sh
```

This will run memory profiling benchmarks on the models in the `case_studies` directory with a variety of different hyperparameter values, and save the results in the `results` directory.

There are multiple ways of analyzing the memray profile data. To visualize the memory profile data in a flame or icicle graph, run

```bash
python -m memray flame results/<filename>.prof
```

and then open the generated `<memray-flamegraph>.html` file in a browser.

Another meaningful way to analyze the memory profile data is via a tree reporter. A tree reporter provides a simplified representation of the call hierarchy of the tracked process at the time when its memory usage was at its peak. 

```bash
python -m memray tree results/<filename>.prof
```

When analyzing the memray profile data, it is often useful to hide "irrelevant" frames such as import system frames. 

## Results

Below, some brief notes are given on the memory allocation when using the memray tree reporter on the case study models.

### BART bikes

`results/biking_200_60.prof`:

 - `step = pmb.PGBART([mu])`: About 19.5MB of memory utilized
    - `jitter_duplicated(...)`: Accounts for 13.5/19.5MB of memory utilized
 - `step.astep(iter)`: About 7.5MB of memory utilized
    - `self.running_sd[odim].update(new)`: Accounts for 6.0/7.5MB of memory utilized

The initialization of PGBART in this example accounts for the majority of the PGBART memory usage.