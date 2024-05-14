cimport cython

@cython.boundscheck(False)
def bit_length_cython(int x):
    cdef int  count = 0
    while x:
        count += 1
        x >>= 1
    return count + 1
