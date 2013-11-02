"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import numpy as np
import numpy.linalg
import time
from setx0 import setx0
from bfgs import bfgs
from gradsamp import gradsamp
from postprocess import postprocess


def hanso(func, grad, x0=None, nvar=None, nstart=None, sampgrad=False,
          normtol=1e-6, verbose=2, fvalquit=-np.inf, cpumax=np.inf,
          maxit=100, **kwargs):
    """
    HANSO: Hybrid Algorithm for Nonsmooth Optimization

    The algorithm is two-fold. Viz,
    BFGS phase: BFGS is run from multiple starting points, taken from
    the columns of x0 parameter, if provided, and otherwise 10 points
    generated randomly. If the termination test was satisfied at the
    best point found by BFGS, or if nvar > 100, HANSO terminates;
    otherwise, it continues to:

    Gradient sampling phases: 3 gradient sampling phases are run from
    lowest point found, using sampling radii:
    10*evaldist, evaldist, evaldist/10
    Termination takes place immediately during any phase if
    cpumax CPU time is exceeded.

    References
    ----------
    A.S. Lewis and M.L. Overton, Nonsmooth Optimization via Quasi-Newton
    Methods, Math Programming, 2012

    J.V. Burke, A.S. Lewis and M.L. Overton, A Robust Gradient Sampling
    Algorithm for Nonsmooth, Nonconvex Optimization
    SIAM J. Optimization 15 (2005), pp. 751-779

    Parameters
    ----------
    func: callable function on 1D arrays of length nvar
        function being optimized

    grad: callable function
        gradient of func

    fvalquit: float, optional (default -inf)
        param passed to bfgs1run function

    normtol: float, optional (default 1e-4)
        termination tolerance for smallest vector in convex hull of saved
        gradients

    verbose: int, optional (default 1)
        param passed to bfgs1run function

    cpumax: float, optional (default inf)
        quit if cpu time in secs exceeds this (applies to total running time)

    sampgrad: boolean, optional (default False)
        if set, the gradient-sampling will be used to continue the algorithm
        in case the BFGS fails

    **kwargs: param-value dict
        optional parameters passed to bfgs backend. Possible key/values are:
        x0: 2D array of shape (nvar, nstart), optional (default None)
            intial points, one per column

        nvar: int, optional (default None)
            number of dimensions in the problem (exclusive x0)

        nstart: int, optional (default None)
            number of starting points for BFGS algorithm (exclusive x0)

        maxit: int, optional (default 100)
            param passed to bfgs1run function
        wolfe1: float, optional (default 0)
            param passed to bfgs1run function

        wolfe2: float, optional (default .5)
            param passed to bfgs1run function

    Returns
    -------
    x: D array of same length nvar = len(x0)
        final iterate

    f: list of nstart floats
        final function values, one per run of bfgs1run

    d: list of nstart 1D arrays, each of same length as input nvar
        final smallest vectors in convex hull of saved gradients,
        one array per run of bfgs1run

    H: list of nstarts 2D arrays, each of shape (nvar, nvar)
        final inverse Hessian approximations, one array per run of bfgs1run

    itrecs: list of nstart int
        numbers of iterations, one per run of bfgs1run; see bfgs1run
        for details

    inforecs: list of int
        reason for termination; see bfgs1run for details

    pobj: list of tuples of the form (duration of iteration, final func value)
        trajectory for best starting point (i.e of the starting point that
        led to the greatest overall decrease in the cost function.
        Note that the O(1) time consumed by the gradient-sampling stage is not
        counted.

    Optional Outputs (in case output_records is True):
    Xrecs: list of nstart 2D arrays, each of shape (iter, nvar)
        iterates where saved gradients were evaluated; one array per run
        of bfgs1run; see bfgs1run
        for details

    Grecs: ist of nstart 2D arrays, each of shape (nvar, nvar)
        gradients evaluated at these points, one per run of bfgs1run;
        see bfgs1run for details

    wrecs: list of nstart 1D arrays, each of length iter
        weights defining convex combinations d = G*w; one array per
        run of bfgs1run; see bfgs1run for details

    fevalrecs: list of nstart 1D arrays, each of length iter
        records of all function evaluations in the line searches;
        one array per run of bfgs1run; see bfgs1run for details

    xrecs: list of nstart 2D arrays, each of length (iter, nvar)
        record of x iterates

    Hrecs: list of nstart 2D arrays, each of shape (iter, nvar)
       record of H (Hessian) iterates; one array per run of bfgs1run;
       see bfgs1run for details

    Raises
    ------
    RuntimeError

    """

    def _log(msg, level=0):
        if verbose > level:
            print msg

    # sanitize x0
    if x0 is None:
        assert not nvar is None, (
            "No value specified for x0, expecting a value for nvar")
        assert not nstart is None, (
            "No value specified for x0, expecting a value for nstart")

        x0 = setx0(nvar, nstart)
    else:
        assert nvar is None, (
            "Value specified for x0, expecting no value for nvar")

        assert nstart is None, (
            "Value specified for x0, expecting no value for nstart")

        x0 = np.array(x0)
        if x0.ndim == 1:
            x0 = x0.reshape((-1, 1))

        nvar, nstart = x0.shape

    cpufinish = time.time() + cpumax

    # run BFGS step
    kwargs['output_records'] = 1
    x, f, d, H, _, info, X, G, w, pobj = bfgs(
        func, grad, x0=x0, fvalquit=fvalquit, normtol=normtol, cpumax=cpumax,
        maxit=maxit, verbose=verbose, **kwargs)

    # throw away all but the best result
    assert len(f) == np.array(x).shape[1], np.array(x).shape
    indx = np.argmin(f)
    f = f[indx]
    x = x[..., indx]
    d = d[..., indx]
    H = H[indx]  # bug if do this when only one start point: H already matrix
    X = X[indx]
    G = G[indx]
    w = w[indx]
    pobj = pobj[indx]

    dnorm = numpy.linalg.norm(d, 2)
    # the 2nd argument will not be used since x == X(:,1) after bfgs
    loc, X, G, w = postprocess(x, np.nan, dnorm, X, G, w, verbose=verbose)

    if np.isnan(f) or np.isinf(f):
        _log('hanso: f is infinite or nan at all starting points')
        return x, f, loc, X, G, w, H, pobj

    if time.time() > cpufinish:
        _log('hanso: cpu time limit exceeded')
        _log('hanso: best point found has f = %g with local optimality '
             'measure: dnorm = %5.1e, evaldist = %5.1e' % (
                f, loc['dnorm'], loc['evaldist']))
        return x, f, loc, X, G, w, H, pobj

    if f < fvalquit:
        _log('hanso: reached target objective')
        _log('hanso: best point found has f = %g with local optimality'
             ' measure: dnorm = %5.1e, evaldist = %5.1e' % (
                f, loc['dnorm'], loc['evaldist']))
        return x, f, loc, X, G, w, H, pobj

    if dnorm < normtol:
        _log('hanso: verified optimality within tolerance in bfgs phase')
        _log('hanso: best point found has f = %g with local optimality '
             'measure: dnorm = %5.1e, evaldist = %5.1e' % (
                f, loc['dnorm'], loc['evaldist']))
        return x, f, loc, X, G, w, H, pobj

    if sampgrad:
        # launch gradient sampling
        # time0 = time.time()
        f_BFGS = f
        # save optimality certificate info in case gradient sampling cannot
        # improve the one provided by BFGS
        dnorm_BFGS = dnorm
        loc_BFGS = loc
        d_BFGS = d
        X_BFGS = X
        G_BFGS = G
        w_BFGS = w
        x0 = x.reshape((-1, 1))

        # otherwise gradient sampling is too expensivea
        if maxit > 100:
            maxit = 100

        # # otherwise grad sampling will augment with random starts
        # x0 = x0[..., :1]
        # assert 0, x0.shape

        cpumax = cpufinish - time.time()  # time left

        # run gradsamp proper
        x, f, g, dnorm, X, G, w = gradsamp(func, grad, x0, maxit,
                                           cpumax=cpumax)

        if f == f_BFGS:  # gradient sampling did not reduce f
            _log('hanso: gradient sampling did not reduce f below best point'
                 ' found by BFGS\n')
            # use the better optimality certificate
            if dnorm > dnorm_BFGS:
                loc = loc_BFGS
                d = d_BFGS
                X = X_BFGS
                G = G_BFGS
                w = w_BFGS
        elif f < f_BFGS:
            loc, X, G, w = postprocess(x, g, dnorm, X, G, w)
            _log('hanso: gradient sampling reduced f below best point found'
                 ' by BFGS\n')
        else:
            raise RuntimeError(
                'hanso: f > f_BFGS: this should never happen'
                )  # this should never happen

        x = x[0]
        f = f[0]
        # pobj.append((time.time() - time0, f))
        return x, f, loc, X, G, w, H, pobj
    else:
        return x, f, loc, X, G, w, H, pobj


if __name__ == '__main__':
    import os
    import scipy.io
    import matplotlib.pyplot as plt
    func_names = [
        "tv",
        'Nesterov',
        'Rosenbrock "banana"',
        'l2-norm',  # this is smooth and convex, we're only being ironic here
        'l1-norm',
        ]
    wolfe_kinds = [0,  # weak
                   # 1 # strong
                   ]

    for func_name, j in zip(func_names, xrange(len(func_names))):
        nstart = 20
        nvar = 300
        if func_name == "tv":
            from example_functions import (tv as func,
                                           grad_tv as grad)
        if "l1-norm" in func_name:
            from example_functions import (l1 as func,
                                           grad_l1 as grad)
        if "l2-norm" in func_name:
            from example_functions import (l2 as func,
                                           gradl2 as grad)
        elif "banana" in func_name:
            nvar = 2
            from example_functions import (rosenbrock_banana as func,
                                           grad_rosenbrock_banana as grad)
        elif "esterov" in func_name:
            from example_functions import (nesterov as func,
                                           grad_nesterov as grad)
        if os.path.exists("/tmp/x0.mat"):
                x0 = scipy.io.loadmat("/tmp/x0.mat", squeeze_me=True,
                                      struct_as_record=False)['x0']
                if x0.ndim == 1:
                    x0 = x0.reshape((-1, 1), order='F')
        else:
            x0 = setx0(nvar, nstart)

        if "banana" in func_name:
            x0 = x0[:nvar, ...]

        nvar, nstart = x0.shape

        func_name = func_name + " in %i dimensions" % nvar
        print "Running HANSO for %s ..." % func_name

        for strongwolfe in wolfe_kinds:
            # run BFGS
            results = hanso(func, grad,
                            x0=x0,
                            sampgrad=True,
                            strongwolfe=strongwolfe,
                            maxit=1000,
                            normtol=2 * 1e-3,
                            fvalquit=1e-4,
                            verbose=2
                            )
            xmin, fmin = results[:2]
            pobj = results[-1]

            assert fmin == func(xmin)

            print "xopt:", xmin
            print "fmin:", fmin

            times, pobj = map(np.array, zip(*pobj))

            plt.plot(times, pobj, label="%swolfe" % (
                    ['weak', 'strong'][strongwolfe]))
            plt.xlabel('Time')
            plt.ylabel('Primal')
            plt.gca().set_xscale('log')
            plt.gca().set_yscale('log')
            plt.legend()

        plt.title(func_name)
        plt.show()
        print "... done (%s).\r\n" % func_name
