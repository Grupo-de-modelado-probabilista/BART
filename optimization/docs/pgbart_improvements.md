# PG-BART Improvements

This document outlines: (1) areas subject to performance and memory allocation improvements for the PyMC-BART codebase, and (2) potential methods to address these areas of improvement. First, an overview of the code profiling results is provided. Following the profiling results, several methods and languages will be discussed that could be utilized to improve the performance and memory allocation of the PyMC-BART codebase.

## Code profiling results

Code profiling was conducted using the `line_profiler` and `memray` packages for time and memory profiling, respectively. Below, a summary of the profiling results is provided. For specific details, refer to the `README.md` file within each `line_profiler` and `memray` directory of this module.

### Line profiling

In order to provide line-by-line profiling, specific functions and methods have been decorated with `@profile` to obtain a line-by-line timing profile within _that_ function or method. In our case, we are interested in profiling the `.astep` method (and the subsequent call stack) of the `PGBART` sampler as this method represents the main entry point of the particle sampler. Overall, eight functions and or methods are profiled:

└── astep
    ├── sample_tree
    │   └── grow_tree
    │       └── draw_leaf_value
    ├── update_weight
    │   └── logp
    └── resample
        └── systematic

Below are the main takeaways from the line profiling results.

As expected, the majority of the time is spent within loops of the `astep` method. Within the loops, there are several calls to other functions or methods performing computations:

└── astep (354s)
    ├── sample_tree (34% or 106s)
    │   └── grow_tree (92% or 80s)
    │       └── draw_leaf_value (35% or 15s)
    │           └── fast_mean (66%)
    ├── update_weight (59% or 205s)
    │   └── logp (90% or 0.30s)
    │       └── pytensor_function (99%)
    └── resample (4% or 12s)
        ├── control_flow (87%)
        └── systematic (5.2%)

Where `%` indicates percent of time spent executing that line of code, and `s` indicates seconds spent executing that line of code. The tree-diagram represents the call stack and only the **time intensive** lines are shown for brevity. The values in parentheses represent the average across all benchmark case studies.

An interesting observation is that even though there are _several_ operations and or function calls within the loops, typically only a _few_ of them are responsible for the percentage of total time.

#### Notes

* If `logp` takes the majority of the time in `update_weight`, why is the total time of 0.30s so low? I would expect this number to be around 185s. 

### Memory profiling

**TODO**

The memory allocation of PG-BART comes from the initialization `pmb.PGBART` and the `astep` method with the `astep` method being the entry point of the particle sampler......

#### Notes

Currently, at each step PG-BART fits $n$ out of $m$ trees, but all $m$ trees are stored in memory. This makes it easier to track state and for computing the sum of trees but it is a waste of memory.

## Solutions

The profiling results above indicate that our problem is CPU bound, i.e. the majority of time is spent doing computational work within for loop(s) with minimal idle time waiting for I/O or other external operations to complete. 

Due to this, our focus should be on improving the speed of loops and the computations happening within them. Within the loops, there are repetitive operations where the same computation is performed, albeit on different data with the same data types. This leads us to consider ahead of time (AOT) and just in time (JIT) compilation, vectorization, and parallelization as potential solutions to improve the performance of PG-BART.

### Ahead of time and just in time compilation

_Ahead of time compilation_ (AOT): Refers to compiling code prior to execution.

- AOT compilation eliminates the need for runtime environment or a virtual machine (VM), reducing memory usage and startup times.
- AOT compiled code may not be as optimized as JIT compiled code since it cannot leverage runtime information, e.g. to gain information about CPU intensive code.

_Just in time compilation_: Refers to compilation that happens dynamically during program execution rather than ahead of time.

- JIT compiler translates bytecode into machine instructions at runtime, just before executing it.
- Allows the JIT compiler to optimize the machine code using runtime information and only compile code paths that are actually executed.

If we re-write parts of PG-BART in Cython and or the entire codebase in Rust, we would get the benefits of AOT compilation. This would improve the speed of CPU bound loops and reduce memory usage.


### Vectorization and parallelization

Another area of improvement in PG-BART is vectorization and parallelization.


### Comparison

| Framework | Rewrite? | Compilation | Impl. Difficulty | Vectorization | Parallelization | Accelerator Support |
|-----------|----------|-------------|------------------|---------------|-----------------|---------------------|
| JAX       | Yes      | JIT         | Medium           | `vmap`        | `pmap`          | Yes                 |
| Cython    | Partial  | AOT         | Medium           | ?             | OpenMP          | ?                   |
| Rust      | Yes      | AOT         | High             | ?             | Native?         | ?                   |            

#### JAX

Rewriting the PG-BART codebase in JAX would allow:
- JIT compilation
- Automatic vectorization `vmap` and parallelization `pmap`
- Ability to run BART on accelerators (e.g. GPU/TPUs) in addition to CPU as JAX uses the XLA compiler.

An implementation of BART in JAX already [exists](https://github.com/Gattocrucco/bartz).

#### Cython

Rewriting the PG-BART codebase in Cython would allow:
- AOT compilation
- OpenMP support for parallelization

#### Rust

An implementation of BART in Rust already [exists](https://github.com/elanmart/rust-pgbart/tree/main).
