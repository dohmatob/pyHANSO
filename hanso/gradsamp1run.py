"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import time
import numpy as np
from gradsampfixed import gradsampfixed


def gradsamp1run(func, grad, x0, f0=None, g0=None, samprad=[1e-4, 1e-5, 1e-6],
                 cpumax=np.inf, **kwargs):
    """
    Repeatedly run gradient sampling minimization, for various sampling radii
    return info only from final sampling radius; intended to be called by
    gradsamp only

    """

    cpufinish = time.time() + cpumax

    for choice in  xrange(len(samprad)):
        cpumax = cpufinish - time.time()  # time left
        x, f, g, dnorm, X, G, w, quitall = gradsampfixed(
            func, grad, x0, f0=f0, g0=g0, samprad=samprad[choice],
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
    x, f, g, dnorm, X, G, w = gradsamp1run(func, grad, [1e-6, -1e-6])
    print "fmin:", f
    print "xopt:", x
    assert X.shape[0] == 2
    assert G.shape[0] == 2
