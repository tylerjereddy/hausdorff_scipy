"""
Spherical Voronoi Code

.. versionadded:: 0.19.0

"""

import numpy as np
import scipy

def directed_hausdorff(ar1, ar2):
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

    # shuffling the points in each array generally increases the likelihood of
    # an advantageous break in the inner search loop and never decreases the
    # performance of the algorithm
    for array in [ar1, ar2]:
        np.random.shuffle(array)

    cmax = 0
    for outer_point in ar1:
        break_occurred = 0
        cmin = np.inf
        for inner_point in ar2:
            d = scipy.spatial.distance.euclidean(outer_point, inner_point)
            if d < cmax: # early break
                break_occurred += 1
                break
            if d < cmin:
                cmin = d
        if cmin > cmax and cmin != np.inf and break_occurred == 0:
            cmax = cmin
    return cmax
