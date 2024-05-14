##python new_idx_data_points.py
"""
Cython: get_new_idx_data_points_c(300) time = 0.034172 seconds
Cython: get_new_idx_data_points_c(600) time = 0.038320 seconds
Cython: get_new_idx_data_points_c(900) time = 0.040737 seconds
Cython: get_new_idx_data_points_c(1200) time = 0.042911 seconds
Python: get_new_idx_data_points(300) time = 0.030682 seconds
Python: get_new_idx_data_points(600) time = 0.039361 seconds
Python: get_new_idx_data_points(900) time = 0.042282 seconds
Python: get_new_idx_data_points(1200) time = 0.046805 seconds
Numba: get_new_idx_data_points_njit(300) time = 0.023319 seconds
Numba: get_new_idx_data_points_njit(600) time = 0.028893 seconds
Numba: get_new_idx_data_points_njit(900) time = 0.035102 seconds
Numba: get_new_idx_data_points_njit(1200) time = 0.039763 seconds

Numba wins
"""
##
import timeit
from functools import lru_cache
import random

import numpy as np

# Define the Cython function
import pyximport
from numba import njit

import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

pyximport.install(
    language_level=3, setup_args={"include_dirs": np.get_include()}, reload_support=True
)

from new_idx_data_points_cython import get_new_idx_data_points_c


# Define the Python function
@njit
def get_new_idx_data_points_njit(
    available_splitting_values, split_value, idx_data_points
):
    split_idx = available_splitting_values <= split_value
    return idx_data_points[split_idx], idx_data_points[~split_idx]


def get_new_idx_data_points(available_splitting_values, split_value, idx_data_points):
    split_idx = available_splitting_values <= split_value
    return idx_data_points[split_idx], idx_data_points[~split_idx]


# Define the input sizes to test
arrays = [
    np.random.rand(300),
    np.random.rand(600),
    np.random.rand(900),
    np.random.rand(1200),
]
get_new_idx_data_points_njit(np.random.rand(300), random.randint(1, 5), np.arange(300))
# Measure the execution time of the Cython function for each input size
for arr in arrays:
    size = len(arr)
    min, max = np.min(arr), np.max(arr)
    random_number = random.randint(int(min), int(max))
    idx = arr = np.arange(size)
    cython_time = timeit.timeit(
        lambda: get_new_idx_data_points_c(arr, random_number, idx), number=10000
    )
    print(f"Cython: get_new_idx_data_points_c({size}) time = {cython_time:.6f} seconds")

# Measure the execution time of the Python function for each input size
for arr in arrays:
    size = len(arr)
    min, max = np.min(arr), np.max(arr)
    random_number = random.randint(int(min), int(max))
    idx = arr = np.arange(size)
    python_time = timeit.timeit(
        lambda: get_new_idx_data_points(arr, random_number, idx), number=10000
    )
    print(f"Python: get_new_idx_data_points({size}) time = {python_time:.6f} seconds")

# Measure the execution time of the Python function for each input size
for arr in arrays:
    size = len(arr)
    min, max = np.min(arr), np.max(arr)
    random_number = random.randint(int(min), int(max))
    idx = arr = np.arange(size)
    get_new_idx_data_points_njit(arr, random_number, idx)
    python_time = timeit.timeit(
        lambda: get_new_idx_data_points_njit(arr, random_number, idx), number=10000
    )
    print(
        f"Numba: get_new_idx_data_points_njit({size}) time = {python_time:.6f} seconds"
    )
