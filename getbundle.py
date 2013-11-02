"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import numpy as np


def getbundle(func, grad, x0, g0=None, samprad=1e-4, n=None):
    """
    Get bundle of n-1 gradients at points near x, in addition to g,
    which is gradient at x and goes in first column
    intended to be called by gradsampfixed

    Parameters
    ----------
    func: callable function on 1D arrays of length nvar
        function being optimized

    grad: callable function
        gradient of func

    x0: 1D array of len nvar, optional (default None)
        intial points, one per column

    g0: 1D array of len nvar, optional (default None)
        gradient at point x0

    samprad: float, optional (default 1e-4)
        sampling radius; this should be a small positive float

    n: int, optional (default min(100, 2 * nvar, nvar + 10))
        number of points and gradients to sample

    Returns
    -------
    xbundle: 2D array of shape (nvar, n)
        bundle of n points sampled in the samprand-ball around x0

    xbundle: 2D array of shape (nvar, n)
        bundle of n gradients sampled in the samprand-ball around x0

    """

    x0 = np.ravel(x0)
    nvar = len(x0)
    n = min(100, min(2 * nvar, nvar + 10)) if n is None else n

    xbundle = np.ndarray((nvar, n))
    gbundle = np.ndarray((nvar, n))
    xbundle[..., 0] = x0
    gbundle[..., 0] = g0 if not g0 is None else grad(x0)
    for k in xrange(1, n):  # note the 1
        xpert = x0 + samprad * (np.random.rand(nvar) - 0.5
                               )  # uniform distribution
        f, g = func(xpert), grad(xpert)
        count = 0
        # in particular, disallow infinite function values
        while np.isnan(f) or np.isinf(f) or np.any(
            np.isnan(g)) or np.any(np.isinf(g)):
            xpert = (x0 + xpert) / 2.     # contract back until feasible
            f, g = func(xpert), grad(xpert)
            count = count + 1
            if count > 100:  # should never happen, but just in case
                raise RuntimeError(
                    'getbundle: too many contractions needed to find finite'
                    ' func and grad values')

        xbundle[..., k] = xpert
        gbundle[..., k] = g

    return xbundle, gbundle


if __name__ == '__main__':
    from example_functions import (l1 as func,
                                   gradl1 as grad)
    xbundle, gbundle = getbundle(func, grad, [1e-6, -1e-6], samprad=1, n=100)
    import matplotlib.pyplot as plt
    plt.scatter(*gbundle)
    plt.show()
