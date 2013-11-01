import time
import numpy as np
import numpy.linalg
from gradsamp1run import gradsamp1run


def gradsamp(func, grad, x0, maxit=10, cpumax=np.inf, verbose=1, **kwargs):
    """
    GRADSAMP Gradient sampling algorithm for nonsmooth, nonconvex
    minimization.

    Intended for nonconvex functions that are continuous everywhere and for
    which the gradient can be computed at most points, but which are known
    to be nondifferentiable at some points, typically including minimizers.

    Reference
    ---------
    J.V. Burke, A.S. Lewis and M.L. Overton,
    A Robust Gradient Sampling Algorithm for Nonsmooth, Nonconvex Optimization
    SIAM J. Optimization, 2005

    """

    def _log(msg, level=0):
        if verbose > level:
            print msg

    _, nstart = x0.shape
    cpufinish = time.time() + cpumax

    f = []
    g = []
    x = []
    dnorm = []
    X = []
    G = []
    w = []
    for run in xrange(nstart):
        if verbose > 0 & nstart > 1:
            _log('gradsamp: starting point %d ' % run)
        f0, g0 = func(x0[..., run]), grad(x0[..., run])
        if np.isnan(f0) or f0 == np.inf or maxit == 0:
            if np.isnan(f0) and verbose > 0:
                _log('gradsamp: function is NaN at initial point')

            elif f0 == np.inf and verbose > 0:
                _log('gradsamp: function is infinite at initial point')

            # useful if just want to evaluate func
            elif maxit == 0 and verbose > 0:
                _log('gradsamp: max iteration limit is 0, returning '
                     'initial point')
            f.append(f0)
            x.append(x0[..., run])
            g.append(g0)
            dnorm.append(numpy.linalg.norm(g0, 2))
            X.append(x[..., run])
            G.append(g0)
            w.append(1)
        else:
            cpumax = cpufinish - time.time()  # time left
            xtmp, ftmp, gtmp, dnormtmp, Xtmp, Gtmp, wtmp = \
                gradsamp1run(func, grad, x0[..., run], f0=f0, g0=g0, **kwargs)
            x.append(xtmp)
            f.append(ftmp)
            g.append(gtmp)
            dnorm.append(dnormtmp)
            X.append(Xtmp)
            G.append(Gtmp)
            w.append(wtmp)
        if time.time() > cpufinish:
            break

    raise RuntimeError("I'm still coding this thing, pal!")
    print np.array(g).shape
    return x, f, np.array(g).T, dnorm, np.array(X)[0], np.array(G)[0], w

if __name__ == '__main__':
    from setx0 import setx0
    from example_functions import (l1 as func,
                                   gradl1 as grad)
    x0 = setx0(20, 10)
    x, f, g, dnorm, X, G, w = gradsamp(func, grad, x0)
    print "fmin:", f
    print "xopt:", x
    assert X.shape[0] == 2, X.shape
    assert G.shape[0] == 2
