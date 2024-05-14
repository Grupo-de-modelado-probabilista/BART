import timeit
from functools import lru_cache

"""
Cython: bit_length(1) = 2, time = 0.000749 seconds
Cython: bit_length(3) = 3, time = 0.000752 seconds
Cython: bit_length(7) = 4, time = 0.000758 seconds
Cython: bit_length(15) = 5, time = 0.000784 seconds
Python: bit_length(1) = 1, time = 0.001117 seconds
Python: bit_length(3) = 2, time = 0.001068 seconds
Python: bit_length(7) = 3, time = 0.001068 seconds
Python: bit_length(15) = 4, time = 0.001065 seconds
Numba: bit_length_numba(1) = 0, time = 0.001452 seconds
Numba: bit_length_numba(3) = 1, time = 0.001567 seconds
Numba: bit_length_numba(7) = 2, time = 0.001450 seconds
Numba: bit_length_numba(15) = 3, time = 0.001407 seconds
Cache: bit_length_cache(1) = 1, time = 0.000829 seconds
Cache: bit_length_cache(3) = 2, time = 0.000797 seconds
Cache: bit_length_cache(7) = 3, time = 0.000799 seconds
Cache: bit_length_cache(15) = 4, time = 0.000808 seconds


"""
# Define the Cython function
import pyximport
from numba import njit

pyximport.install(language_level=3)

from bit_length_cython import bit_length_cython


# Define the Python function
def bit_length_python(x):
    return x.bit_length()


@lru_cache
def bit_length_cache(x):
    return x.bit_length()


# Define the Python function
@njit
def bit_length_numba(v):
    r = (v > 0xFFFF) << 4
    v >>= r
    shift = (v > 0xFF) << 3
    v >>= shift
    r |= shift
    shift = (v > 0xF) << 2
    v >>= shift
    r |= shift
    shift = (v > 0x3) << 1
    v >>= shift
    r |= shift
    return r | (v >> 1)


# Define the input sizes to test
input_sizes = [1, 2, 3, 4]
bit_length_numba(2**4 - 1)

# Measure the execution time of the Cython function for each input size
for size in input_sizes:
    x = 2**size - 1
    cython_time = timeit.timeit(lambda: bit_length_cython(x), number=10000)
    print(
        f"Cython: bit_length({x}) = {bit_length_cython(x)}, time = {cython_time:.6f} seconds"
    )

# Measure the execution time of the Python function for each input size
for size in input_sizes:
    x = 2**size - 1
    python_time = timeit.timeit(lambda: bit_length_python(x), number=10000)
    print(
        f"Python: bit_length({x}) = {bit_length_python(x)}, time = {python_time:.6f} seconds"
    )

# Measure the execution time of the Numba function for each input size
for size in input_sizes:
    x = 2**size - 1
    python_time = timeit.timeit(lambda: bit_length_numba(x), number=10000)
    print(
        f"Numba: bit_length_numba({x}) = {bit_length_numba(x)}, time = {python_time:.6f} seconds"
    )


# Measure the execution time of the Numba function for each input size
for size in input_sizes:
    x = 2**size - 1
    bit_length_cache(x)

# Measure the execution time of the Numba function for each input size
for size in input_sizes:
    x = 2**size - 1
    python_time = timeit.timeit(lambda: bit_length_cache(x), number=10000)
    print(
        f"Cache: bit_length_cache({x}) = {bit_length_cache(x)}, time = {python_time:.6f} seconds"
    )
