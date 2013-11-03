"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import time

import numpy as np
from scipy import linalg

from bfgs1run import bfgs1run
from setx0 import setx0


def bfgs(func, grad, x0=None, nvar=None, nstart=None, maxit=100, nvec=0,
         verbose=1, normtol=1e-6, fvalquit=-np.inf, xnormquit=np.inf,
         cpumax=np.inf, strongwolfe=False, wolfe1=0, wolfe2=.5, quitLSfail=1,
         ngrad=None, evaldist=1e-6, H0=None, scale=1, output_records=2
         ):
    """
    Make a single run of BFGS from one starting point. Intended to be
    called from bfgs.

    Parameters
    ----------
    func: callable function on 1D arrays of length nvar
        function being optimized

    grad: callable function
        gradient of func

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

    normtol: float, optional (default 1e-6)
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

    fvalquit: float, optional (default -inf)
        param passed to bfgs1run function

    quitLSfail: int, optional (default 1)
        1 if quit when line search fails, 0 (potentially useful if func
        is not numerically continuous)

    ngrad: int, optional (default max(100, 2 * nvar))
        number of gradients willing to save and use in solving QP to check
        optimality tolerance on smallest vector in their convex hull;
        see also next two options

    verbose: int, optional (default 1)
        param passed to bfgs1run function

    output_records: int, optional (default 2)
        Which low-level execution records to return from low-level
        bfgs1run calls ? Possible values are:
        0: don't return execution records from low-level bfgs1run calls
        1: return H and w records from low-level bfgs1run calls
        2: return all execution records from low-level bfgs1run calls

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

    pobj: list of lists of tuples of the form (duration of iteration,
    final func value)
        for each starting point, the energy trajectory for each iteration
        of the iterates therefrom

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

    Notes
    -----
    if there is more than one starting vector, then:
    f, iter, info are vectors of length nstart
    x, d are matrices of size pars.nvar by nstart
    H, X, G, w, xrec, Hrec are cell arrays of length nstart, and
    fevalrec is a cell array of cell arrays
    Thus, for example, d[:,i] = G[i] * w[i], for i = 0,...,nstart - 1

    BFGS is normally used for optimizing smooth, not necessarily convex,
    functions, for which the convergence rate is generically superlinear.
    But it also works very well for functions that are nonsmooth at their
    minimizers, typically with a linear convergence rate and a final
    inverse Hessian approximation that is very ill conditioned, as long
    as a weak Wolfe line search is used. This version of BFGS will work
    well both for smooth and nonsmooth functions and has a stopping
    criterion that applies for both cases, described above.
    Reference:  A.S. Lewis and M.L. Overton, Nonsmooth Optimization via
    Quasi-Newton Methods, Math Programming, 2012

    See Also
    --------
    `gradsamp`

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
    pobj = []
    _f = []
    itrecs = []
    inforecs = []
    _d = []
    _x = []
    _H = []
    if output_records:
        xrecs = []
        fevalrecs = []
        Hrecs = []
        Xrecs = []
        Grecs = []
        wrecs = []
    for run in xrange(nstart):
        _log("Staring bfgs1run %i/%i..." % (
                run + 1, nstart))
        if verbose > 0 & nstart > 1:
            _log('bfgs: starting point %d' % (run + 1))
        cpumax = cpufinish - time.time()
        if output_records > 1:
            x, f, d, HH, it, info, X, G, w, fevalrec, xrec, Hrec, times = \
                bfgs1run(x0[..., run], func, grad, maxit=maxit, wolfe1=wolfe1,
                         wolfe2=wolfe2, normtol=normtol, fvalquit=fvalquit,
                         xnormquit=xnormquit, cpumax=cpumax,
                         strongwolfe=strongwolfe, verbose=verbose,
                         quitLSfail=quitLSfail, ngrad=ngrad, evaldist=evaldist,
                         H0=H0, scale=scale)
            _x.append(x)
            _f.append(x)
            _d.append(d)
            itrecs.append(it)
            inforecs.append(info)
            Xrecs.append(X)
            Grecs.append(G)
            wrecs.append(w)
            fevalrecs.append(fevalrec)
            xrecs.append(xrec)
            Hrecs.append(Hrec)
        elif output_records > 0:
            x, f, d, HH, it, info, X, G, w, _, _, _, times = bfgs1run(
                x0[..., run], func, grad, maxit=maxit, wolfe1=wolfe1,
                wolfe2=wolfe2, normtol=normtol, fvalquit=fvalquit,
                xnormquit=xnormquit, cpumax=cpumax, strongwolfe=strongwolfe,
                verbose=verbose, quitLSfail=quitLSfail, ngrad=ngrad,
                evaldist=evaldist, H0=H0, scale=scale)
            _x.append(x)
            _f.append(f)
            _d.append(d)
            itrecs.append(it)
            inforecs.append(info)
            Xrecs.append(X)
            Grecs.append(G)
            wrecs.append(w)
        else:  # avoid computing unnecessary arrays
            x, f, d, HH, it, info, _, _, _, _, _, _, times = bfgs1run(
                x0[..., run], func, grad, maxit=maxit, wolfe1=wolfe1,
                wolfe2=wolfe2, normtol=normtol, fvalquit=fvalquit,
                xnormquit=xnormquit, cpumax=cpumax, strongwolfe=strongwolfe,
                verbose=verbose, quitLSfail=quitLSfail, ngrad=ngrad,
                evaldist=evaldist, H0=H0, scale=scale)
            _x.append(x)
            _f.append(f)
            _d.append(d)
            itrecs.append(it)
            inforecs.append(info)

        _log('... done (bfgs1run %i/%i).' % (run + 1, nstart))
        _log("\r\n")

        # HH should be exactly symmetric as of version 2.02, but does no harm
        _H.append((HH + HH.T) / 2.)

        # commit times
        run_pobj = []
        for duration, f in times:
            run_pobj.append((duration, f))
        pobj.append(run_pobj)

        # check that we'ven't exploded the time budget
        if time.time() > cpufinish or f < fvalquit or linalg.norm(
            x, 2) > xnormquit:
            break
    # end of for loop

    # we're done: now collect and return outputs to caller
    _x = np.array(_x).T
    _f = np.array(_f)
    _d = np.array(_d).T
    if output_records > 1:
        return (_x, _f, _d, _H, itrecs, inforecs, Xrecs, Grecs, wrecs,
                fevalrecs, xrecs, Hrecs, pobj)
    elif output_records > 0:
        return _x, _f, _d, _H, itrecs, inforecs, Xrecs, Grecs, wrecs, pobj
    else:
        return _x, _f, _d, _H, itrecs, inforecs, pobj


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    func_names = [
        'Nesterov',
        'Rosenbrock "banana"',
        'l1-norm',
        'l2-norm'
        ]
    wolfe_kinds = [0,  # weak
                   # 1 # strong
                   ]

    for func_name, j in zip(func_names, xrange(len(func_names))):
        nstart = 20
        nvar = 500
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

        for strongwolfe in wolfe_kinds:
            # run BFGS
            fevalrecs = bfgs(func, grad, nvar=nvar, nstart=nstart,
                             strongwolfe=strongwolfe,
                             maxit=10,
                             verbose=2
                             )[-4]

            # plot results
            ax = plt.subplot2grid((len(func_names), len(wolfe_kinds)),
                                  (j, strongwolfe))
            for fevalrec in fevalrecs:
                for fevalrecline in fevalrec:
                    ax.plot(np.log10(fevalrecline), '*-',
                            c='b')
                    ax.hold('on')

            if j == 0:
                ax.set_title("HANSO-BFGS: %s Wolfe (y axis in logscale)" % (
                        'strong' if strongwolfe else 'weak'))

            if j + 1 == len(func_names):
                ax.set_xlabel('Iteration number')

            if strongwolfe == 0:
                ax.set_ylabel(func_name)

    plt.show()
