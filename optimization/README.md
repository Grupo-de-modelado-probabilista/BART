# BART Optimization

Welcome to the BART optimization module.

The purpose of this module is to perform detailed code profiling on the BART codebase to identify areas of high memory usage and performance bottlenecks. Thus, there are two directories:

1. `line_profiler` - contains line-by-line timing profiles of specific methods and function calls of the PGBART sampler.
2. `memray` - contains reports on memory allocations and sizes of the PGBART sampler.

## Getting started

Please refer to the `README` files within each of the `line_profiler` and `memray` directories for instructions on the environment setup and how to run the benchmarks. The [markdown](docs/pgbart_improvements.md) in `docs/` contains a summary of the code profiling and suggests potential methods to resolve the identified bottlenecks.