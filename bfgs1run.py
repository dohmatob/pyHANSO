"""
:Synopsis: Python implementation bfgs1run.m (HANSO): BFGS for non-smooth
nonconvex functions via inexact line search

:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import numpy as np
import numpy.linalg
import time
from hgprod import hgprod
from qpspecial import qpspecial
from linesch_ww import linesch_ww


def bfgs1run(x0, func, grad, maxit=100, nvec=0, verbose=1, funcrtol=1e-6,
             gradnormtol=1e-4, fvalquit=-np.inf, xnormquit=np.inf,
             cpumax=np.inf, strongwolfe=False, wolfe1=0, wolfe2=.5,
             quitLSfail=1, ngrad=None, evaldist=1e-4, H0=None, scale=1):
    """
    Make a single run of BFGS (with inexact line search) from one starting
    point. Intended to be called from bfgs.

    Parameters
    ----------
    func: callable function on 1D arrays of length nvar
        function being optimized

    grad: callable function
        gradient of func

    x0: 1D array of len nvar, optional (default None)
        intial point

    nvar: int, optional (default None)
        number of dimensions in the problem (exclusive x0)

    maxit: int, optional (default 100)
        maximum number of BFGS iterates we are ready to pay for

    wolfe1: float, optional (default 0)
        param passed to linesch_ww[sw] function

    wolfe2: float, optional (default .5)
        param passed to linesch_ww[sw] function

    strongwolfe: boolean, optional (default 1)
        0 for weak Wolfe line search (default)
        1 for strong Wolfe line search
        Strong Wolfe line search is not recommended for use with
        BFGS; it is very complicated and bad if f is nonsmooth;
        however, it can be useful to simulate an exact line search

    fvalquit: float, optional (default -inf)
        param passed to bfgs1run function

    gradnormtol: float, optional (default 1e-6)
        termination tolerance on d: smallest vector in convex hull of up
        to ngrad gradients

    xnormquit: float, optional (default inf)
        quit if norm(x) exceeds this value

    evaldist: float, optional default (1e-4)
        the gradients used in the termination test qualify only if
        they are evaluated at points  approximately  within
        distance evaldist of x

    H0: 2D array of shape (nvar, nvar), optional (default identity matrix)
        for full BFGS: initial inverse Hessian approximation (must be
        positive definite, but this is not checked), this could be draw
        drawn from a Wishart distribution;
        for limited memory BFGS: same, but applied every iteration
        (must be sparse in this case)

    scale: boolean, optional (default True)
        for full BFGS: 1 to scale H0 at first iteration, 0 otherwise
        for limited memory BFGS: 1 to scale H0 every time, 0 otherwise

    cpumax: float, optional (default inf)
        quit if cpu time in secs exceeds this (applies to total running
        time)

    verbose: int, optional (default 1)
        param passed to bfgs1run function

    quitLSfail: int, optional (default 1)
        1 if quit when line search fails, 0 (potentially useful if func
        is not numerically continuous)

    ngrad: int, optional (default min(100, 2 * nvar, nvar + 10))
        number of gradients willing to save and use in solving QP to check
        optimality tolerance on smallest vector in their convex hull;
        see also next two options

    Returns
    -------
    x: 1D array of same length nvar = len(x0)
        final iterate

    f: float
        final function value

    d: 1D array of same length nvar
       final smallest vector in convex hull of saved gradients

    H: 2D array of shape (nvar, nvar)
       final inverse Hessian approximation

    iter: int
       number of iterations

    info: int
        reason for termination
         0: tolerance on smallest vector in convex hull of saved gradients met
         1: max number of iterations reached
         2: f reached target value
         3: norm(x) exceeded limit
         4: cpu time exceeded limit
         5: f or g is inf or nan at initial point
         6: direction not a descent direction (because of rounding)
         7: line search bracketed minimizer but Wolfe conditions not satisfied
         8: line search did not bracket minimizer: f may be unbounded below
         9: relative tolerance on function value met on last iteration

    X: 2D array of shape (iter, nvar)
        iterates where saved gradients were evaluated

    G: 2D array of shape (nvar, nvar)
        gradients evaluated at these points

    w: 1D array
        weights defining convex combination d = G*w

    fevalrec: 1D array of length iter
        record of all function evaluations in the line searches

    xrec: 2D array of length (iter, nvar)
        record of x iterates

    Hrec: 2D array of shape (iter, nvar)
       record of H (Hessian) iterates

    times: list of floats
        time consumed in each iteration

    Raises
    ------
    ImportError

    """

    def _log(msg, level=0):
        if verbose > level:
            print msg

    # sanitize input
    x0 = np.array(x0).ravel()
    nvar = np.prod(x0.shape)
    H0 = np.eye(nvar) if H0 is None else H0
    ngrad = min(100, min(2 * nvar, nvar + 10)) if ngrad is None else ngrad
    x = np.array(x0)
    H = np.array(H0)

    # initialize auxiliary variables
    S = []
    Y = []
    xrec = []
    fevalrec = []
    Hrec = []
    X = np.array([x]).T
    nG = 1
    w = 1

    # prepare for timing
    cpufinish = time.time() + cpumax
    time0 = time.time()
    times = []

    # first evaluation
    f, g = func(x), grad(x)
    times.append((time.time() - time0, f))

    # check that all is still well
    d = np.array(g)
    G = np.array([g]).T
    if np.isnan(f) or np.isinf(f):
        _log('bfgs1run: f is infinite or nan at initial iterate')
        info = 5
        return  x, f, d, H, 0, info, X, G, w, fevalrec, xrec, Hrec, times
    if np.any(np.isnan(g)) or np.any(np.isinf(g)):
        _log('bfgs1run: grad is infinite or nan at initial iterate')
        info = 5
        return  x, f, d, H, 0, info, X, G, w, fevalrec, xrec, Hrec, times

    # enter: main loop
    dnorm = numpy.linalg.norm(g, 2)  # initialize dnorm stopping creteria
    f_old = f
    for it in xrange(maxit):
        p = -np.dot(H, g) if nvec == 0 else -hgprod(H, g, S, Y)
        gtp = np.dot(g.T, p)
        if gtp >= 0 or np.any(np.isnan(gtp)):
            _log(
                'bfgs1run: not descent direction, quitting after %d '
                'iteration(s), f = %g, dnorm = %5.1e, gtp=%s' % (
                    it + 1, f, dnorm, gtp))
            info = 6
            times.append((time.time() - time0, f))
            return x, f, d, H, it, info, X, G, w, fevalrec, xrec, Hrec, times

        gprev = np.array(g)  # for BFGS update
        if strongwolfe:
            # strong Wolfe line search is not recommended except to simulate
            # exact line search
            _log("Starting inexact line search (strong Wolfe) ...")

            # have we coded strong Wolfe line search ?
            try:
                from linesch_sw import linesch_sw
            except ImportError:
                raise ImportError(
                    '"linesch_sw" is not in path: it can be obtained from the'
                    ' NLCG distribution')

            alpha, x, f, g, fail, _, _, fevalrecline = linesch_sw(
                x, func, grad, p, wolfe1=wolfe1, wolfe2=wolfe2,
                fvalquit=fvalquit, verbose=verbose)

            # function values are not returned in strongwolfe, so set
            # fevalrecline to nan
            # fevalrecline = np.nan

            _log("... done.")
            # exact line search: increase alpha slightly to get to other side
            # of an discontinuity in nonsmooth case
            if wolfe2 == 0:
                increase = 1e-8 * (1 + alpha)
                x = x + increase * p
                _log(' exact line sch simulation: slightly increasing step '
                     'from %g to %g' % (alpha, alpha + increase), level=1)

                f, g = func(x), grad(x)
        else:
            _log("Starting inexact line search (weak Wolfe) ...")
            alpha, x, f, g, fail, _, _, fevalrecline = linesch_ww(
                x, func, grad, p, wolfe1=wolfe1, wolfe2=wolfe2,
                fvalquit=fvalquit, verbose=verbose)
            _log("... done.")

        # for the optimal check: discard the saved gradients iff the
        # new point x is not sufficiently close to the previous point
        # and replace them with new gradient
        if alpha * numpy.linalg.norm(p, 2) > evaldist:
            nG = 1
            G = np.array([g]).T
            X = np.array([x]).T
        # otherwise add new gradient to set of saved gradients,
        # discarding oldest
        # if alread have ngrad saved gradients
        elif nG < ngrad:
            nG += 1
            G = np.vstack((g, G.T)).T
            X = np.vstack((x, X.T)).T
        else:  # nG = ngrad
            G = np.vstack((g, G[..., :ngrad - 1].T)).T
            X = np.vstack((x, X[..., :ngrad - 1].T)).T
        # optimality check: compute smallest vector in convex hull
        # of qualifying gradients: reduces to norm of latest gradient
        # if ngrad = 1, and the set
        # must always have at least one gradient: could gain efficiency
        # here by updating previous QP solution
        if nG > 1:
            _log("Computing shortest l2-norm vector in convex hull of "
                 "cached gradients: G = %s ..." % G.T)
            w, d, _, _ = qpspecial(G, verbose=verbose)
            _log("... done.")
        else:
            w = 1
            d = np.array(g)

        dnorm = numpy.linalg.norm(d, 2)

        # XXX this recordings shoud be optional!
        xrec.append(x)
        fevalrec.append(fevalrecline)
        Hrec.append(H)

        if verbose > 1:
            nfeval = len(fevalrecline)
            _log(
                'bfgs1run: iter %d: nfevals = %d, step = %5.1e, f = %g, '
                'nG = %d, dnorm = %5.1e' % (it, nfeval, alpha, f, nG, dnorm),
                level=1)
        if f < fvalquit:  # this is checked inside the line search
            _log('bfgs1run: reached target objective, quitting after'
                 ' %d iteration(s)' % (it + 1))
            info = 2
            times.append((time.time() - time0, f))
            return x, f, d, H, it, info, X, G, w, fevalrec, xrec, Hrec, times

        # this is not checked inside the line search
        elif numpy.linalg.norm(x, 2) > xnormquit:
            _log('bfgs1run: norm(x) exceeds specified limit, quitting after'
                 ' %d iteration(s)' % (it + 1))
            info = 3
            times.append(time.time() - time0, f)
            return  x, f, d, H, it, info, X, G, w, fevalrec, xrec, Hrec, times

        # line search failed (Wolfe conditions not both satisfied)
        if fail == 1:
            if not quitLSfail:
                _log('bfgs1run: continue although line search failed',
                     level=1)
            else:  # quit since line search failed
                _log('bfgs1run: quitting after %d iteration(s), f = %g, '
                     'dnorm = %5.1e' % (it + 1, f, dnorm))
                info = 7
                times.append((time.time() - time0, f))
                return  (x, f, d, H, it, info, X, G, w, fevalrec, xrec,
                         Hrec, times)

        # function apparently unbounded below
        elif fail == -1:
            _log('bfgs1run: f may be unbounded below, quitting after %d '
                 'iteration(s), f = %g' % (it + 1, f))
            info = 8
            times.append((time.time() - time0, f))
            return  x, f, d, H, it, info, X, G, w, fevalrec, xrec, Hrec

        # are we trapped in a local minimum ?
        relative_change = np.abs(1 - 1. * f_old / f) if f != f_old else 0
        if relative_change < funcrtol:
            _log('bfgs1run: relative change in func over last iteration (%g)'
                 ' below tolerance (%g) , quiting after %d iteration(s),'
                 ' f = %g' % (relative_change, funcrtol, it + 1, f))
            info = 9
            times.append((time.time() - time0, f))
            return  (x, f, d, H, it, info, X, G, w, fevalrec, xrec,
                     Hrec, times)

        # check near-stationarity
        if dnorm <= gradnormtol:
            if nG == 1:
                _log('bfgs1run: gradient norm below tolerance, quiting '
                     'after %d iteration(s), f = %g' % (it + 1, f))
            else:
                _log(
                    'bfgs1run: norm of smallest vector in convex hull of'
                    ' gradients below tolerance, quitting after '
                    '%d iteration(s), f = %g' % (it + 1, f))
            info = 0
            times.append((time.time() - time0, f))
            return  x, f, d, H, it, info, X, G, w, fevalrec, xrec, Hrec, times

        if time.time() > cpufinish:
            _log('bfgs1run: cpu time limit exceeded, quitting after %d '
                 'iteration(s) %d' % (it + 1))
            info = 4
            times.append((time.time() - time0, f))
            return  x, f, d, H, it, info, X, G, w, fevalrec, xrec, Hrec, times
        s = (alpha * p).reshape((-1, 1))
        y = g - gprev
        sty = np.dot(s.T, y)  # successful line search ensures this is positive
        assert sty > 0
        if nvec == 0:  # perform rank two BFGS update to the inverse Hessian H
            if sty > 0:
                if it == 0 and scale:
                    # for full BFGS, Nocedal and Wright recommend
                    # scaling I before the first update only
                    H = (1. * sty / np.dot(y.T, y)) * H
                # for formula, see Nocedal and Wright's book
                # M = I - rho*s*y', H = M*H*M' + rho*s*s', so we have
                # H = H - rho*s*y'*H - rho*H*y*s' + rho^2*s*y'*H*y*s'
                # + rho*s*s' note that the last two terms combine:
                # (rho^2*y'Hy + rho)ss'
                rho = 1. / sty
                Hy = np.dot(H, y).reshape((-1, 1))
                rhoHyst = rho * np.dot(Hy, s.T)
                # old version: update may not be symmetric because of rounding
                # H = H - rhoHyst' - rhoHyst + rho*s*(y'*rhoHyst) + rho*s*s';
                # new in version 2.02: make H explicitly symmetric
                # also saves one outer product
                # in practice, makes little difference, except H=H' exactly
                ytHy = np.dot(y.T,
                              Hy)  # could be < 0 if H not numerically pos def
                sstfactor = np.max([rho * rho * ytHy + rho, 0])
                sscaled = np.sqrt(sstfactor) * s
                H = H - (rhoHyst.T + rhoHyst) + np.dot(sscaled, sscaled.T)
                # alternatively add the update terms together first: does
                # not seem to make significant difference
                # update = sscaled*sscaled' - (rhoHyst' + rhoHyst);
                # H = H + update;
            # should not happen unless line search fails, and in that
            # case should normally have quit
            else:
                _log('bfgs1run: sty <= 0, skipping BFGS update at iteration '
                     '%d ' % it, level=1)
        else:  # save s and y vectors for limited memory update
            s = alpha * p
            y = g - gprev
            if it + 1 <= nvec:
                S = np.hstack((S, s))
                Y = np.hstack((Y, y))
            # could be more efficient here by avoiding moving the columns
            else:
                S = np.hstack((S[..., 1:nvec], s))
                Y = np.hstack((Y[..., 1:nvec], y))
            if scale:
                # recommended by Nocedal-Wright
                H = np.dot(np.dot(s.T, y), np.dot(np.dot(y.T, y), H0))

        f_old = f
        times.append((time.time() - time0, f))
    # end of 'for loop'

    _log('bfgs1run: %d iteration(s) reached, f = %g, dnorm = %5.1e' % (
            maxit, f, dnorm))

    info = 1  # quit since max iterations reached
    return  x, f, d, H, it, info, X, G, w, fevalrec, xrec, Hrec, times

if __name__ == '__main__':
    nvar = 300
    nstart = 20
    func_name = 'Rosenbrock "Banana" function in %i dimensions' % nvar
    import os
    from example_functions import (l1, grad_l1)
    from setx0 import setx0
    import scipy.io
    if os.path.isfile("/tmp/x0.mat"):
        x0 = scipy.io.loadmat("/tmp/x0.mat", squeeze_me=True,
                              struct_as_record=False)['x0']
    else:
        x0 = setx0(nvar, nstart)

    if x0.ndim == 1:
        x0 = x0.reshape((-1, 1))

    _x = None
    _f = np.inf
    for j in xrange(x0.shape[1]):
        print ">" * 100, "(j = %i)" % j
        x, f = bfgs1run(x0[..., j], l1, grad_l1,
                        strongwolfe=0,
                        maxit=100,
                        verbose=2,
                        gradnormtol=1e-6,
                        xnormquit=np.inf,
                        fvalquit=-np.inf,
                        cpumax=np.inf,
                        wolfe1=0,
                        wolfe2=.5,
                        nvec=0,
                        scale=1,
                        evaldist=1e-6
                        )[:2]
        if f < _f:
            _f = f
            _x = x
        print "<" * 100, "(j = %i)" % j

    print _x
    print _f
