import time
import numpy as np
import numpy.linalg
from bfgs1run import bfgs1run
from setx0 import setx0


def bfgs(func, grad, x0=None, nvar=None, nstart=None, maxit=100, nvec=0,
         verbose=1, normtol=1e-6, fvalquit=-np.inf, xnormquit=np.inf,
         cpumax=np.inf, strongwolfe=False, wolfe1=0, wolfe2=.5, quitLSfail=1,
         ngrad=None, evaldist=1e-6, H0=None, scale=1, output_records=2,
         **kwargs
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

    _f: list of nstart floats
        final function values, one per run of bfgs1run

    _d: list of nstart 1D arrays, each of same length as input nvar
       final smallest vectors in convex hull of saved gradients,
       one array per run of bfgs1run

    _H: list of nstarts 2D arrays, each of shape (nvar, nvar)
       final inverse Hessian approximations, one array per run of bfgs1run

    itrecs: list of nstart int
       numbers of iterations, one per run of bfgs1run; see bfgs1run
       for details

    inforecs: list of int
        reason for termination; see bfgs1run for details

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
        nvar, nstart = x0.shape

    cpufinish = time.time() + cpumax
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
        _log(">" * 80)
        _log("Staring bfgs1run %i/%i with x0 = %s ..." % (
                run + 1, nstart, x0[..., run]))
        if verbose > 0 & nstart > 1:
            _log('bfgs: starting point %d' % (run + 1))
        cpumax = cpufinish - time.time()
        if output_records > 1:
            x, f, d, HH, it, info, X, G, w, fevalrec, xrec, Hrec = bfgs1run(
                x0[..., run], func, grad, maxit=maxit, wolfe1=wolfe1,
                wolfe2=wolfe2, normtol=normtol, fvalquit=fvalquit,
                xnormquit=xnormquit, cpumax=cpumax, strongwolfe=strongwolfe,
                verbose=verbose, quitLSfail=quitLSfail, ngrad=ngrad,
                evaldist=evaldist, H0=H0, scale=scale)
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
            x, f, d, HH, it, info, X, G, w, _, _, _ = bfgs1run(
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
            x, f, d, HH, it, info, _, _, _, _, _, _ = bfgs1run(
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
        _log("<" * 80)

        # HH should be exactly symmetric as of version 2.02, but does no harm
        _H.append((HH + HH.T) / 2.)
        if time.time() > cpufinish or f < fvalquit or numpy.linalg.norm(
            x, 2) > xnormquit:
            break

    # we're done: now collect and return outputs to caller
    _x = np.array(_x).T
    _f = np.array(_f)
    _d = np.array(_d).T
    if output_records > 1:
        return (_x, _f, _d, _H, itrecs, inforecs, Xrecs, Grecs, wrecs,
                fevalrecs, xrecs, Hrecs)
    elif output_records > 0:
        return _x, _f, _d, _H, itrecs, inforecs, Xrecs, Grecs, wrecs
    else:
        return _x, _f, _d, _H, itrecs, inforecs


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    func_names = [
        'Nesterov',
        'Rosenbrock "banana"',
        'l1-norm',
        'l2-norm',
        ]
    wolfe_kinds = [0,  # weak
                   # 1 # strong
                   ]

    for func_name, j in zip(func_names, xrange(len(func_names))):
        nstart = 20
        nvar = 500
        if "l1-norm" in func_name:
            from example_functions import (l1 as func,
                                           gradl1 as grad)
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
                             )[-3]

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
