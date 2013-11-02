import time
from math import sqrt
import numpy as np
from scipy import linalg, signal
import pylab as pl
from example_functions import l1, grad_l1, tv, grad_tv

penalty_model = "tv"

rng = np.random.RandomState(42)
m, n, k = 100, 300, 10
design = "convolutional"

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
l = 0.01

# tv-sparse betamap
if penalty_model == "tv":
    import scipy.signal
    x0 = scipy.signal.waveforms.square(rng.randn(n))

# observed signal
b = np.dot(A, x0)


def unpenalized_loss_func(A, b, lambd, x):
    return .5 * linalg.norm(np.dot(A, x) - b) ** 2


def l1_pernalized_loss_function(A, b, lambd, x):
    """
    Frobenius + l1 penalty

    """

    return .5 * linalg.norm(np.dot(A, x) - b) ** 2 + lambd * l1(x)


def tv_pernalized_loss_function(A, b, lambd, x):
    """
    Frobenius + TV penalty

    """

    return .5 * linalg.norm(np.dot(A, x) - b) ** 2 + lambd * tv(x)


def tv_plus_l1_pernalized_loss_function(A, b, lambd, x):
    """
    Frobenius + TV penalty + l1 penalty

    """

    raise NotImplementedError("TV+L1 penalty model not implmented!")

# define loss function according to penalty model (l1, tv, tv+l1, etc.)
if penalty_model == 'l1':
    loss_function = l1_pernalized_loss_function
elif penalty_model == "tv":
    l = .05
    loss_function = tv_pernalized_loss_function
else:
    raise ValueError("Unknown penalty model: %s" % penalty_model)


def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)


def ista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    L = linalg.norm(A) ** 2
    t0 = time.time()
    for _ in xrange(maxit):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, l / L)
        pobj.append((time.time() - t0,
                     loss_function(A, b, l, x)))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times


def fista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    t = 1
    z = x.copy()
    L = linalg.norm(A) ** 2
    time0 = time.time()
    for _ in xrange(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z, l / L)
        t0 = t
        t = (1 + sqrt(1 + 4 * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        pobj.append((time.time() - time0,
                     loss_function(A, b, l, x)))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times


def hanso(A, b, lambd, maxit, x0_init="random"):
    """
    Runs a flovor HANSO (customizable via the x0_init parameter).

    Parameters
    ----------
    x0_init: string, optional (defautl "random")
        initialization mode for HANSO: Possible values are:
        "soft_thresh_ista": initialize as if ISTA
        "soft_thresh_fista": initialize as if FISTA
        "random": initialize as usual (multivariate standard normal)

    Raises
    ------
    ValueError

    """

    from hanso import hanso
    from setx0 import setx0

    nstart = 1  # 20

    def func(x):
        return loss_function(A, b, lambd, x)

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

    # penalty dependent gradient definition
    if penalty_model == "l1":
        grad = grad_with_l1_penalty
    elif penalty_model == "tv":
        grad = grad_with_tv_penalty
    else:
        raise ValueError("Unknown penalty model: %s" % penalty_model)

    if x0_init == "soft_thresh_ista":
        x0 = np.zeros(A.shape[1])
        L = linalg.norm(A) ** 2
        x0 = soft_thresh(x0 + np.dot(A.T, b - np.dot(A, x0)) / L,
                         l / L)
    elif x0_init == "soft_thresh_fista":
        x0 = np.zeros(A.shape[1])
        L = linalg.norm(A) ** 2
        x0 = np.zeros(A.shape[1])
        x0 = x0 + np.dot(A.T, b - A.dot(x0)) / L
        x0 = soft_thresh(x0, l / L)
    elif x0_init == "random":
        x0 = setx0(A.shape[1], nstart)
    else:
        raise ValueError("Unknown value for x0_init parameter: %s" % x0_init)

    results = hanso(func, grad,
                    x0=x0,
                    sampgrad=True,
                    maxit=maxit,
                    verbose=2
                    )
    x0 = results[0]
    pobj = results[-1]

    times, pobj = map(np.array, zip(*pobj))
    return x0, pobj, times


maxit = 1000  # 100000

# HANSO
hanso_x0_init_modes = [
    'random',
    'soft_thresh_ista',
    'soft_thresh_fista'
    ]
pobj_hanso = []
times_hanso = []
for x0_init in hanso_x0_init_modes:
    optimizer = "HANSO (%s x0 init)" % x0_init
    x_hanso, pobj, times = hanso(A, b, l, maxit, x0_init=x0_init)
    pobj_hanso.append(pobj)
    times_hanso.append(times)
    fmin_hanso = loss_function(A, b, l, x_hanso)
    print "fmin %s: %s" % (optimizer, fmin_hanso)
    print

# ISTA
x_ista, pobj_ista, times_ista = ista(A, b, l, maxit)
fmin_ista = loss_function(A, b, l, x_ista)
print "xopt ISTA:", x_ista
print "fmin ISTA:", fmin_hanso
print

# FISTA
x_fista, pobj_fista, times_fista = fista(A, b, l, maxit)
fmin_fista = loss_function(A, b, l, x_fista)
print "xopt FISTA:", x_fista
print "fmin FISTA:", fmin_fista
print

# repare for reporting
pl.close('all')

# plot time perfomance
pl.figure()
pl.title("Linear regression on %s design with %s penalty model" % (
        design, penalty_model))
for x0_init, times, pobj in zip(hanso_x0_init_modes,
                                times_hanso, pobj_hanso):
    optimizer = "HANSO (%s x0 init)" % x0_init
    pl.plot(times, pobj, label=optimizer)
pl.plot(times_ista, pobj_ista, label='ista')
pl.plot(times_fista, pobj_fista, label='fista')
pl.xlabel('Time')
pl.ylabel('Primal')
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.legend()

# plot betamaps
pl.title(("Linear regression on %s design with %s penalty model "
          "(lambda = %f)" % (design, penalty_model.upper(), l)))
pl.figure()
pl.plot(x0, 'o-', label='True betamap')
pl.plot(x_hanso, 's-', label='HANSO estimated betamap')
pl.plot(x_ista, '*-', label='ISTA estimated betamap')
pl.plot(x_fista, '^-', label='FISTA estimated betamap')
pl.ylabel("beta values (i.e regression coffients)")
pl.xlabel("conditions (index with nonnegative integers)")
pl.legend()

pl.show()
