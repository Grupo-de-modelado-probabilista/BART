import cython
import numpy as np
cimport numpy as np
np.import_array()
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fast_mean_cython(ari):
    """Use Cython to speed up the computation of the mean."""

    cdef int count = ari.shape[0]
    cdef double suma = 0

    if ari.ndim == 1:
        for i in range(count):
            suma += ari[i]
        return suma / count

    cdef np.ndarray[double, ndim=1] res = np.zeros(count)
    cdef int n = ari.shape[1]
    for j in range(count):
        for i in range(n):
            res[j] += ari[j, i]
    return res / n
