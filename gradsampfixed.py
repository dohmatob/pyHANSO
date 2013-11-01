import time
import numpy as np
import numpy.linalg
from linesch_ww import linesch_ww
from getbundle import getbundle
from qpspecial import qpspecial


def gradsampfixed(func, grad, x0, f0=None, g0=None, samprad=1e-4, maxit=10,
                  normtol=1e-6, fvalquit=-np.inf, cpumax=np.inf,
                  verbose=2, ngrad=None, **kwargs):
    """"
    Gradient sampling minimization with fixed sampling radius
    intended to be called by gradsamp1run only


    Parameters
    ----------
    func: callable function on 1D arrays of length nvar
        function being optimized

    grad: callable function
        gradient of func

    x0: 2D array of shape (nvar, nstart), optional (default None)
        intial points, one per column

    See Also
    --------
    `bfgs` and `bfgs1run`

    """

    def _log(msg, level=0):
        if verbose > level:
            print msg

    _log('gradsamp: sampling radius = %7.1e' % samprad)

    x = np.array(x0)
    f0 = func(x0) if f0 is None else f0
    g0 = grad(x0) if g0 is None else g0
    f = f0
    g = g0
    X = x
    G = np.array([g]).T
    w = 1
    quitall = 0
    cpufinish = time.time() + cpumax
    dnorm = np.inf
    for it in xrange(maxit):
        # evaluate gradients at randomly generated points near x
        # first column of Xnew and Gnew are respectively x and g
        Xnew, Gnew = getbundle(func, grad, x, g0=g, samprad=samprad, n=ngrad)

        # solve QP subproblem
        wnew, dnew, _, _ = qpspecial(Gnew, verbose=verbose)
        dnew = -dnew  # this is a descent direction
        gtdnew = np.dot(g.T, dnew)   # gradient value at current point
        dnormnew = numpy.linalg.norm(dnew, 2)
        if dnormnew < dnorm:  # for returning, may not be the final one
            dnorm = dnormnew
            X = Xnew
            G = Gnew
            w = wnew
        if dnormnew < normtol:
            # since dnormnew is first to satisfy tolerance, it must equal dnorm
            _log('  tolerance met at iter %d, f = %g, dnorm = %5.1e' % (
                    it, f, dnorm))
            return x, f, g, dnorm, x, G, w, w, quitall
        elif gtdnew >= 0 or np.isnan(gtdnew):
            # dnorm, not dnormnew, which may be bigger
            _log('  not descent direction, quit at iter %d, f = %g, '
                 'dnorm = %5.1e' % (it, f, dnorm))
            return x, f, g, dnorm, X, G, w, quitall

        # note that dnew is NOT normalized, but we set second Wolfe
        # parameter to 0 so that sign of derivative must change
        # and this is accomplished by expansion steps when necessary,
        # so it does not seem necessary to normalize d
        wolfe1 = 0
        wolfe2 = 0
        alpha, x, f, g, fail, _, _, _ = linesch_ww(
            x, func, grad, f, g, dnew, wolfe1=wolfe1, wolfe2=wolfe2,
            fvalquit=fvalquit, verbose=verbose)
        _log('  iter %d: step = %5.1e, f = %g, dnorm = %5.1e' % (
                it, alpha, f, dnormnew), level=1)

        if f < fvalquit:
            _log('  reached target objective, quit at iter %d ' % iter)
            quitall = 1
            return x, f, g, dnorm, X, G, w, quitall

        # if fail == 1 # Wolfe conditions not both satisfied, DO NOT quit,
        # because this typically means gradient set not rich enough and we
        # should continue sampling
        if fail == -1:  # function apparently unbounded below
            _log('  f may be unbounded below, quit at iter %d, f = %g' % (
                    it, f))
            quitall = 1
            return x, f, g, dnorm, X, G, w, quitall

        if time.time() > cpufinish:
            _log('  cpu time limit exceeded, quit at iter #d' % it)
            quitall = 1
            return x, f, g, dnorm, G, w, quitall

    _log('  %d iters reached, f = %g, dnorm = %5.1e' % (maxit, f, dnorm))
    return x, f, g, dnorm, X, G, w, quitall


if __name__ == '__main__':
    from example_functions import (l1 as func,
                                   gradl1 as grad)
    print gradsampfixed(func, grad, [1e-6, -1e-6])[-1]
