"""
:Synopsis: Python implementation of linesch_ww.m (HANSO)
Author: DOHMATOB Elvis Dopgima

"""

import numpy as np
import numpy.linalg


def linesch_ww(x0, func, grad, d, func0=None, grad0=None, wolfe1=0, wolfe2=.5,
               fvalquit=-np.inf, verbose=1, **kwargs):
    """
    LINESCH_WW Line search enforcing weak Wolfe conditions, suitable
    for minimizing both smooth and nonsmooth functions
    Version 2.0 for HANSO 2.0

    The weak Wolfe line search is far less complicated that the standard
    strong Wolfe line search that is discussed in many texts. It appears
    to have no disadvantages compared to strong Wolfe when used with
    Newton or BFGS methods on smooth functions, and it is essential for
    the application of BFGS or bundle to nonsmooth functions as done in HANSO.
    However, it is NOT recommended for use with conjugate gradient methods,
    which require a strong Wolfe line search for convergence guarantees.
    Weak Wolfe requires two conditions to be satisfied: sufficient decrease
    in the objective, and sufficient increase in the directional derivative
    (not reduction in its absolute value, as required by strong Wolfe).

    There are some subtleties for nonsmooth functions.  In the typical case
    that the directional derivative changes sign somewhere along d, it is
    no problem to satisfy the 2nd condition, but descent may not be possible
    if the change of sign takes place even when the step is tiny. In this
    case it is important to return the gradient corresponding to the positive
    directional derivative even though descent was not obtained. On the other
    hand, for some nonsmooth functions the function decrease is steady
    along the line until at some point it jumps to infinity, because an
    implicit constraint is violated.  In this case, the first condition is
    satisfied but the second is not. All cases are covered by returning
    the end points of an interval [alpha, beta] and returning the function
    value at alpha, but the gradients at both alpha and beta.

    The assertion that [alpha,beta] brackets a point satisfying the
    weak Wolfe conditions depends on an assumption that the function
    f(x + td) is a continuous and piecewise continuously differentiable
    function of t, and that in the unlikely event that f is evaluated at
    a point of discontinuity of the derivative, g'*d, where g is the
    computed gradient, is either the left or right derivative at the point
    of discontinuity, or something in between these two values.

    For functions that are known to be nonsmooth, setting the second Wolfe
    parameter to zero makes sense, especially for a bundle method, and for
    the Shor R-algorithm, for which it is essential.  However, it's not
    a good idea for BFGS, as for smooth functions this may prevent superlinear
    convergence, and it can even make trouble for BFGS on, e.g.,
    f(x) = x_1^2 + eps |x_2|, when eps is small.

    Line search quits immediately if f drops below fvalquit.

    Parameters
    ----------
    x0: 1D array of length nvar
    intial point

    d: 1D array of length nvar
       search direction

    wolfe1: float, optional (default 0)
       Wolfe parameter for the sufficient decrease condition
       f(x0 + t d) ** < ** f0 + wolfe1*t*grad0'*d     (DEFAULT 0)

    wolfe2: float, optional (default .5)
        Wolfe parameter for the WEAK condition on directional derivative
        (grad f)(x0 + t d)'*d ** > ** wolfe2*grad0'*d  (DEFAULT 0.5)
        where 0 <= wolfe1 <= wolfe2 <= 1.
        For usual convergence theory for smooth functions, normally one
        requires 0 < wolfe1 < wolfe2 < 1, but wolfe1=0 is fine in practice.
        May want wolfe1 = wolfe2 = 0 for some nonsmooth optimization
        algorithms such as Shor or bundle, but not BFGS.
        Setting wolfe2=0 may interfere with superlinear convergence of
        BFGS in smooth case.

    fvalquit: float, optional (default -inf)
        quit immediately if f drops below this value, regardless
        of the Wolfe conditions (default -inf)

    verbose: int, optional (default 1)
        for no printing, 1 minimal (default), 2 verbose

    Returns
    -------
    alpha: float
        steplength satisfying weak Wolfe conditions if one was found,
        otherwise left end point of interval bracketing such a point
        (possibly 0)

    xalpha: 1D array of length nvar
        x0 + alpha*d

    falpha: float
        f at the point x0 + alpha * d

    gradalpha: 1D array of length nvar
        grad f at the point x0 + alpha * d

    fail: int
        0 if both Wolfe conditions satisfied, or falpha < fvalquit
        1 if one or both Wolfe conditions not satisfied but an
            interval was found bracketing a point where both satisfied
        -1 if no such interval was found, function may be unbounded
            below

    beta: float
        same as alpha if it satisfies weak Wolfe conditions,
        otherwise right end point of interval bracketing such a point
        (inf if no such finite interval found)

    gradbeta: 1D array of length nvar
        grad f at the point x0 + beta d) (this is important for bundle
        methods; vector of nans if beta is inf

    fevalrec: list
        record of function evaluations

    """

    def _log(msg, level=0):
        if verbose > level:
            print msg

    x0 = np.array(x0)
    d = np.array(d)
    func0 = func(x0) if func0 is None else func0
    grad0 = grad(x0) if grad0 is None else grad0

    if (wolfe1 < 0 or wolfe1 > wolfe2 or wolfe2 > 1
        ):  # allows wolfe1 = 0, wolfe2 = 0 and wolfe2 = 1
        _log('linesch_ww_mod: Wolfe parameters do not satisfy'
             ' 0 <= wolfe1 <= wolfe2 <= 1')

    alpha = 0  # lower bound on steplength conditions
    xalpha = np.array(x0)
    falpha = func0

    # need to pass g0, not g0'*d, in case line search fails
    galpha = grad0
    #  upper bound on steplength satisfying weak Wolfe conditions
    beta = np.inf
    gbeta = np.nan * np.ones(x0.shape)
    g0 = np.dot(grad0.T, d)
    print grad0, d, g0
    if g0 >= 0:
        # error('linesch_ww_mod: g0 is nonnegative, indicating d not
        # a descent direction')
        _log('linesch_ww: WARNING, not a descent direction')

    dnorm = numpy.linalg.norm(d, 2)
    if dnorm == 0:
        raise RuntimeError('linesch_ww_mod: d is zero')
    t = 1  # important to try steplength one first
    nfeval = 0
    nbisect = 0
    nexpand = 0
    # the following limits are rather arbitrary
    # nbisectmax = 30  # 50 is TOO BIG, because of rounding errors
    nbisectmax = max(30,
                     np.round(np.log2(
                1e5 * dnorm)))  # allows more if ||d|| big
    nexpandmax = max(10,
                     np.round(np.log2(
                1e5 / dnorm)))  # allows more if ||d|| small
    done = 0
    fevalrec = []
    while not done:
        x = x0 + t * d
        nfeval = nfeval + 1
        f, g = func(x, **kwargs), grad(x, **kwargs)
        fevalrec.append(f)
        if f < fvalquit:  # nothing more to do, quit
            fail = 0
            alpha = t  # normally beta is inf
            xalpha = x
            falpha = f
            galpha = g
            return (alpha, xalpha, falpha, galpha, fail, beta,
                    gbeta, fevalrec)

        gtd = np.dot(g.T, d)

        # the first condition must be checked first. NOTE THE >=.
        if f >= func0 + wolfe1 * t * g0 or np.isnan(f):
            # first condition violated, gone too far
            beta = t
            gbeta = g  # discard f
        # now the second condition.  NOTE THE <=
        elif gtd <= wolfe2 * g0 or np.isnan(
            gtd):  # second condition violated, not gone far enough
            alpha = t
            xalpha = x
            falpha = f
            galpha = g
        else:  # quit, both conditions are satisfied
            fail = 0
            alpha = t
            xalpha = x
            falpha = f
            galpha = g
            beta = t
            gbeta = g
            return (alpha, xalpha, falpha, galpha, fail, beta,
                    gbeta, fevalrec)

        # setup next function evaluation
        if beta < np.inf:
            if nbisect < nbisectmax:
                nbisect = nbisect + 1
                t = (alpha + beta) / 2.  # bisection
            else:
                done = 1
        else:
            if nexpand < nexpandmax:
                nexpand = nexpand + 1
                t = 2 * alpha  # still in expansion mode
            else:
                done = 1

    # end loop
    # Wolfe conditions not satisfied: there are two cases
    if beta == np.inf:  # minimizer never bracketed
        fail = -1
        if verbose > 1:
            _log('Line search failed to bracket point satisfying weak '
                 'Wolfe conditions; function may be unbounded below')
    else:  # point satisfying Wolfe conditions was bracketed
        fail = 1
        if verbose > 1:
            _log('Line search failed to satisfy weak Wolfe conditions'
                 ' although point satisfying conditions was bracketed')

    return alpha, xalpha, falpha, galpha, fail, beta, gbeta, fevalrec

if __name__ == '__main__':
    from example_functions import l1, gradl1
    print linesch_ww([1, 1], l1, gradl1, [-1, -2])
