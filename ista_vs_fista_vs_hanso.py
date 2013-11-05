import time
from math import sqrt
import numpy as np
from scipy import linalg, signal
import pylab as pl
from pyHANSO.example_functions import l1, grad_l1, tv, grad_tv

penalty_model = "tv"

rng = np.random.RandomState(42)  # pseudo-random number generator
m, n, k = 100, 300, 10
design = "convolutional"
alpha = 1.  # param for tv+l1
rho = .7  # param for tv+l1
lambd = 0.01  # param for tv or l1 only

# random design
if design.lower() == "random":
    A = rng.randn(m, n)  # random design
    A = np.tile(rng.randn(m), (n, 1)).T + 0.0001 * rng.randn(m, n)
# convolutional design
elif design.lower() == "convolutional":
    h = signal.gaussian(50, 5)
    A = signal.convolve2d(np.eye(n), h[:, np.newaxis], 'same')
    A = A[::n // m]
else:
    raise ValueError("Unknown design: %s" % design)

# l1-sparse betamap
x0 = rng.rand(n)
x0[x0 < 0.3] = 0

# tv+l1 sparse betamap
# if penalty_model == "tv":
#     x0 = signal.waveforms.square(rng.randn(n))
if penalty_model in ['tv', 'tv+l1']:
    x0 = np.zeros(n)
    x0[10:80] = 1.
    x0[n // 2: n // 2 + 51] = 2.
    x0_true = np.array(x0)
    x0 += rng.randn(n) * 1e-1  # add noise

# observed signal
b = np.dot(A, x0)


def concat_times(times1, times2):
    if len(times2) == 0:
        return times1
    elif len(times1) > 0:
        return np.hstack((times1, times1[-1] + np.array(times2)))
    else:
        return times2


def unpenalized_loss_func(A, b, x):
    return .5 * linalg.norm(np.dot(A, x) - b) ** 2


def l1_pernalized_loss_function(A, b, x):
    """
    Frobenius + l1 penalty

    """

    return .5 * linalg.norm(np.dot(A, x) - b) ** 2 + lambd * l1(x)


def tv_pernalized_loss_function(A, b, x):
    """
    Frobenius + TV penalty

    """

    return .5 * linalg.norm(np.dot(A, x) - b) ** 2 + lambd * tv(x)


def tv_plus_l1_pernalized_loss_function(A, b, x):
    """
    Frobenius + TV penalty + l1 penalty

    """

    return .5 * linalg.norm(np.dot(A, x) - b) ** 2 + alpha * (
        rho * tv(x) + (1 - alpha) * l1(x))


# define loss function according to penalty model (l1, tv, tv+l1, etc.)
if penalty_model == 'l1':
    loss_function = l1_pernalized_loss_function
elif penalty_model == "tv":
    # l = .05
    loss_function = tv_pernalized_loss_function
elif penalty_model == 'tv+l1':
    loss_function = tv_plus_l1_pernalized_loss_function
else:
    raise ValueError("Unknown penalty model: %s" % penalty_model)


def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)


def ista(A, b, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    L = linalg.norm(A) ** 2
    t0 = time.time()
    for _ in xrange(maxit):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, lambd / L)
        pobj.append((time.time() - t0,
                     loss_function(A, b, x)))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times


def fista(A, b, maxit, stop_if_energy_rises=False):
    x = np.zeros(A.shape[1])
    pobj = []
    t = 1
    z = x.copy()
    L = linalg.norm(A) ** 2
    time0 = time.time()
    old_energy = np.inf
    for _ in xrange(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z, lambd / L)
        t0 = t
        t = (1 + sqrt(1 + 4 * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        energy = loss_function(A, b, x)

        if old_energy < energy and stop_if_energy_rises:
            break
        else:
            old_energy = energy

        pobj.append((time.time() - time0,
                     energy))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times


def hanso(A, b, maxit, x0_init="random"):
    """
    Runs a flovor HANSO (customizable via the x0_init parameter).

    Parameters
    ----------
    x0_init: string, optional (defautl "random")
        initialization mode for HANSO: Possible values are:
        "fista": run FISTA, and then switch to HANSO once energy starts
        to increase
        "random": initialize as usual (multivariate standard normal)

    Raises
    ------
    ValueError

    """

    from pyHANSO.hanso import hanso
    from pyHANSO.setx0 import setx0

    nstart = 1

    def func(x):
        return loss_function(A, b, x)

    def grad_with_l1_penalty(x):
        """
        Gradient of loss function when penalty model is l1

        """
        return np.dot(A.T, np.dot(A, x) - b) + lambd * grad_l1(x)

    def grad_with_tv_penalty(x):
        """
        Gradient of loss function when penalty model is tv

        """

        return np.dot(A.T, np.dot(A, x) - b) + lambd * grad_tv(x)

    def grad_with_tv_plus_l1_penalty(x):
        """
        Gradient of loss function when penalty model is tv + l1

        """

        return np.dot(A.T, np.dot(A, x) - b) + alpha * (rho * grad_tv(x) + (
                1 - rho) * grad_l1(x))

    pobj = []
    times = []

    # penalty dependent gradient definition
    if penalty_model == "l1":
        grad = grad_with_l1_penalty
    elif penalty_model == "tv":
        grad = grad_with_tv_penalty
    elif penalty_model == "tv+l1":
        grad = grad_with_tv_plus_l1_penalty
    else:
        raise ValueError("Unknown penalty model: %s" % penalty_model)

    if x0_init == "fista":
        x0, pobj, times = fista(A, b, maxit=maxit,
                                stop_if_energy_rises=True)
    elif x0_init == "random":
        x0 = setx0(A.shape[1], nstart)
    else:
        raise ValueError("Unknown value for x0_init parameter: %s" % x0_init)

    maxit = maxit - len(pobj)
    results = hanso(func, grad,
                    x0=x0,
                    sampgrad=True,
                    maxit=maxit,
                    nvec=10,
                    verbose=2
                    )
    x = results[0]
    _pobj = results[-1]

    _times, _pobj = map(np.array, zip(*_pobj))
    _times = concat_times(times, _times)
    _pobj = list(pobj) + list(_pobj)

    return x, _pobj, _times


maxit = 1000

# HANSO
hanso_x0_init_modes = [
    # fista",  # uncomment if penalty is l1 only
    'random'
    ]
pobj_hanso = []
times_hanso = []
for x0_init in hanso_x0_init_modes:
    optimizer = "HANSO (%s x0 init)" % x0_init
    x_hanso, pobj, times = hanso(A, b, maxit, x0_init=x0_init)
    pobj_hanso.append(pobj)
    times_hanso.append(times)
    fmin_hanso = loss_function(A, b, x_hanso)
    print "fmin %s: %s" % (optimizer, fmin_hanso)
    print

if penalty_model == "l1":
    # ISTA
    x_ista, pobj_ista, times_ista = ista(A, b, maxit)
    fmin_ista = loss_function(A, b, x_ista)
    print "xopt ISTA:", x_ista
    print "fmin ISTA:", fmin_hanso
    print

    # FISTA
    x_fista, pobj_fista, times_fista = fista(A, b, maxit)
    fmin_fista = loss_function(A, b, x_fista)
    print "xopt FISTA:", x_fista
    print "fmin FISTA:", fmin_fista
    print

# repare for reporting
pl.close('all')
if penalty_model in ['l1', 'tv']:
    params = "lambda = %g" % lambd
elif penalty_model == "tv+l1":
    params = "alpha = %g, rho = %g" % (alpha, rho)

# plot time perfomance
pl.figure()
pl.title("Linear regression on %s design with %s penalty model (%s)" % (
        design, penalty_model, params))
for x0_init, times, pobj in zip(hanso_x0_init_modes,
                                times_hanso, pobj_hanso):
    optimizer = "HANSO (%s x0 init)" % x0_init
    pl.plot(times, pobj, label=optimizer)

if penalty_model == "l1":
    pl.plot(times_ista, pobj_ista, label='ista')
    pl.plot(times_fista, pobj_fista, label='fista')

pl.xlabel('Time')
pl.ylabel('Primal')
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.legend()

# plot betamaps
pl.figure()

if penalty_model in ['tv+l1', 'tv']:
    pl.plot(x0_true, label='True betamap')
    pl.plot(x0, label="Corrupt betamap")
else:
    pl.plot(x0, label='True betamap')

pl.plot(x_hanso, label='HANSO estimated betamap')
if penalty_model == "l1":
    pl.plot(x_ista, '*-', label='ISTA estimated betamap')
    pl.plot(x_fista, '^-', label='FISTA estimated betamap')
pl.ylabel("beta values (i.e regression coffients)")
pl.xlabel("conditions")
pl.legend()
pl.title(("Linear regression on %s design with %s penalty model "
          "(%s)" % (design, penalty_model.upper(), params)))

pl.show()
