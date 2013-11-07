"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import time
import numpy as np
from gradsampfixed import gradsampfixed


def gradsamp1run(func, x0, grad=None, f0=None, g0=None,
                 samprad=[1e-4, 1e-5, 1e-6], cpumax=np.inf, **kwargs):
    """
    Repeatedly run gradient sampling minimization, for various sampling radii
    return info only from final sampling radius; intended to be called by
    gradsamp only

    Parameters
    ----------
    func : callable func(x)
        function to minimise.

    x0: 1D array of len nvar, optional (default None)
        intial point

    grad : callable grad(x, *args)
        the gradient of `func`.  If None, then `func` returns the function
        value and the gradient (``f, g = func(x, *args)``), unless
        `approx_grad` is True in which case `func` returns only ``f``.

    f0: float, optional (default None)
        function value at x0

    g0: 1D array of length nvar = len(x0), optional (default None)
        gradient at x0

    samprad: 1D array of floats, optional (default [1e-4, 1e-5, 1e-6])
        radius around x0, for sampling gradients

    See for example bfgs1run for the meaning of the other params.

    See Also
    --------
    `gradsampfixed` and gradsamp`

    """

    cpufinish = time.time() + cpumax

    for choice in  xrange(len(samprad)):
        cpumax = cpufinish - time.time()  # time left
        x, f, g, dnorm, X, G, w, quitall = gradsampfixed(
            func, x0, grad=grad, f0=f0, g0=g0, samprad=samprad[choice],
            cpumax=cpumax, **kwargs)

        # it's not always the case that x = X(:,1), for example when the max
        # number of iterations is exceeded: this is mentioned in the
        # comments for gradsamp
        if quitall:  # terminate early
            return  x, f, g, dnorm, X, G, w

        # get ready for next run, with lower sampling radius
        # start from where previous one finished,
        # because this is lowest function value so far
        x0 = x
        f0 = f
        g0 = g

    return x, f, g, dnorm, np.array(X), np.array(G), w


if __name__ == '__main__':
    from example_functions import (l1 as func,
                                   grad_l1 as grad)
    x, f, g, dnorm, X, G, w = gradsamp1run(func, [1e-6, -1e-6], grad=grad)
    print "fmin:", f
    print "xopt:", x
    assert X.shape[0] == 2
    assert G.shape[0] == 2
