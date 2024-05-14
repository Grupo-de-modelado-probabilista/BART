"""

Cython: fast_mean_cython(10) = 0.6382475071698492, time = 0.011557 seconds
Cython: fast_mean_cython(20) = 0.521222002357218, time = 0.021190 seconds
Cython: fast_mean_cython(30) = 0.5330306095345402, time = 0.031117 seconds
Cython: fast_mean_cython(40) = 0.5054503839346485, time = 0.040722 seconds
Cython: fast_mean_cython(50) = 0.4940089242616287, time = 0.050622 seconds
Cython: fast_mean_cython(60) = 0.48341880877922017, time = 0.060313 seconds
Cython: fast_mean_cython(70) = 0.4638653586753624, time = 0.069726 seconds
Cython: fast_mean_cython(80) = 0.5296347270863913, time = 0.079789 seconds
Cython: fast_mean_cython(90) = 0.50822358120088, time = 0.089261 seconds
Cython: fast_mean_cython(100) = 0.5174994999411932, time = 0.099116 seconds
Python: fast_mean(10) = 0.6382475071698492, time = 0.230622 seconds
Python: fast_mean(20) = 0.521222002357218, time = 0.002942 seconds
Python: fast_mean(30) = 0.5330306095345402, time = 0.002922 seconds
Python: fast_mean(40) = 0.5054503839346485, time = 0.003163 seconds
Python: fast_mean(50) = 0.4940089242616287, time = 0.003159 seconds
Python: fast_mean(60) = 0.48341880877922017, time = 0.003181 seconds
Python: fast_mean(70) = 0.4638653586753624, time = 0.003279 seconds
Python: fast_mean(80) = 0.5296347270863913, time = 0.003383 seconds
Python: fast_mean(90) = 0.50822358120088, time = 0.003469 seconds
Python: fast_mean(100) = 0.5174994999411932, time = 0.003555 seconds
Numba >> best, we are not using the good things of cython memview
"""


import timeit
from functools import lru_cache

import numba
import numpy
import numpy as np

# Define the Cython function
import pyximport
from numba import njit

pyximport.install(
    language_level=3,
    setup_args={"include_dirs": numpy.get_include()},
)

from fast_mean_cython import fast_mean_cython


# Define the Python function
@njit
def fast_mean(ari):
    """Use Numba to speed up the computation of the mean."""

    if ari.ndim == 1:
        count = ari.shape[0]
        suma = 0
        for i in range(count):
            suma += ari[i]
        return suma / count
    else:
        res = np.zeros(ari.shape[0])
        count = ari.shape[1]
        for j in range(ari.shape[0]):
            for i in range(count):
                res[j] += ari[j, i]
        return res / count


# Define the range of sizes and number of arrays to generate
min_size = 5
max_size = 20
num_arrays = 10
sizes = [i * 10 for i in range(1, 11)]

# Generate random arrays of different sizes
arrays = []
for size in sizes:
    arr = np.random.rand(size)
    arrays.append(arr)

# Define the input sizes to test
input_sizes = [1, 2, 3, 4]

# Measure the execution time of the Cython function for each input size
for arr in arrays:
    cython_time = timeit.timeit(lambda: fast_mean_cython(arr), number=10000)
    print(
        f"Cython: fast_mean_cython({len(arr)}) = {fast_mean_cython(arr)}, time = {cython_time:.6f} seconds"
    )

# Measure the execution time of the Python function for each input size
for arr in arrays:
    python_time = timeit.timeit(lambda: fast_mean(arr), number=10000)
    print(
        f"Python: fast_mean({len(arr)}) = {fast_mean(arr)}, time = {python_time:.6f} seconds"
    )
