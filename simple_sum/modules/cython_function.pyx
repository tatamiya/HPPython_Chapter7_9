#cython: boundscheck=False
from cython.parallel import prange
#import numpy as np
#cimport numpy as np

def sum_from_one_cython(int N):
    cdef unsigned int sum
    cdef unsigned int i

    sum = 0

    for i in range(1, N+1):
        sum += i

    return sum

def sum_from_one_openmp(int N):
    cdef unsigned int sum
    cdef unsigned int i

    sum = 0
    for i in prange(1, N+1, nogil=True):
        sum += i

    return sum


