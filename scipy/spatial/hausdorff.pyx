"""
Directed Hausdorff Code

.. versionadded:: 0.19.0

"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

__all__ = ['directed_hausdorff']

@cython.boundscheck(False)
def directed_hausdorff(np.ndarray[np.float64_t, ndim =2] ar1,
                       np.ndarray[np.float64_t, ndim =2] ar2):
    """ Calculates the directed Hausdorff distance between point sets.
    
    Notes
    ----------
    Uses the early break technique and the random sampling approach described
    by Taha and Hanbury (2015) IEEE Transactions On Pattern Analysis And Machine
    Intelligence 37. Although worst-case performance is polynomial (as with the
    brute force algorithm), this is exceedingly unlikely in practice, and
    almost-linear time complexity performance can normally be expected for the
    average case.
    """

    cdef double cmax, cmin
    cdef int break_occurred
    cdef int N1 = ar1.shape[0]
    cdef int N2 = ar2.shape[0]
    cdef int data_dims = ar1.shape[1]
    cdef double square_distance = 0
    cdef np.float64_t d

    # shuffling the points in each array generally increases the likelihood of
    # an advantageous break in the inner search loop and never decreases the
    # performance of the algorithm
    np.random.shuffle(ar1)
    np.random.shuffle(ar2)
                                                                                                                                                                                                     
    cmax = 0 
    for i in range(N1):
        break_occurred = 0
        cmin = np.inf
        for j in range(N2):
            d = 0
	    # faster performance with square of distance
	    # avoid sqrt until very end
            for k in range(data_dims):
                d += ((ar1[<unsigned int> i,<unsigned int> k] - 
                       ar2[<unsigned int> j,<unsigned int> k]) * 
                       (ar1[<unsigned int> i,<unsigned int> k] -
                       ar2[<unsigned int> j,<unsigned int> k]))
            if d < cmax: # early break
                break_occurred += 1
                break
            if d < cmin:
                cmin = d
        if cmin > cmax and cmin != np.inf and break_occurred == 0:
            cmax = cmin
    return sqrt(cmax)
