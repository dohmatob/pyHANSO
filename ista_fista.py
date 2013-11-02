import time
from math import sqrt
import numpy as np
from scipy import linalg, signal
import pylab as pl
from example_functions import l1, gradl1

rng = np.random.RandomState(42)
m, n, k = 100, 300, 10

# random design
# A = rng.randn(m, n)  # random design
# A = np.tile(rng.randn(m),(n,1)).T + 0.0001 * rng.randn(m, n)

# convolutional design
h = signal.gaussian(50, 5)
A = signal.convolve2d(np.eye(n), h[:, np.newaxis], 'same')
A = A[::n // m]

x0 = rng.rand(n)
x0[x0 < 0.3] = 0
b = np.dot(A, x0)
l = 0.01


def loss_function(A, b, lambd, x):
    """
    Frobenius + l1

    """

    return .5 * linalg.norm(np.dot(A, x) - b) ** 2 + lambd * l1(x)


def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)


def ista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    L = linalg.norm(A)**2
    t0 = time.time()
    for _ in xrange(maxit):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, l/L)
        pobj.append((time.time() - t0,
                     0.5 * linalg.norm(A.dot(x) - b)**2 + l * linalg.norm(x, 1)))

    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times


def fista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    t = 1
    z = x.copy()
    L = linalg.norm(A)**2
    time0 = time.time()
    for _ in xrange(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z,l/L)
        t0 = t
        t = (1 + sqrt(1 + 4*t**2)) / 2.
        z = x + ((t0-1.)/t) * (x - xold)
        pobj.append((time.time() - time0,
                    0.5 * linalg.norm(A.dot(x) - b)**2 + l * linalg.norm(x, 1)))

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

    def grad(x):
        return np.dot(A.T, np.dot(A, x) - b) + lambd * gradl1(x)

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
                    nvec=1,

                    # tolerance threshold for l2-norm of gradient
                    normtol=2 * 1e-3,

                    verbose=2,
                    )
    x0 = results[0]
    pobj = results[-1]

    times, pobj = map(np.array, zip(*pobj))
    return x0, pobj, times

    # import pylab as pl
    # print "xopt:", x
    # print "fmin:", f
    # pl.plot(np.array([x, x0]).T, '*-')
    # pl.legend(("xopt from HANSO", "True x"))
    # pl.show()


maxit = 100000

# HANSO
hanso_x0_init_modes = ['random', 'soft_thresh_ista', 'soft_thresh_fista']
pobj_hanso = []
times_hanso = []
for x0_init in hanso_x0_init_modes:
    optimizer = "HANSO (%s x0 init)" % x0_init
    x_hanso, pobj, times = hanso(A, b, l, maxit, x0_init=x0_init)
    pobj_hanso.append(pobj)
    times_hanso.append(times)
    print "xopt %s: %s" % (optimizer, x_hanso)
    print "fmin %s: %s" % (optimizer, loss_function(A, b, l, x_hanso))
    print

# ISTA
x_ista, pobj_ista, times_ista = ista(A, b, l, maxit)
print "xopt ISTA:", x_ista
print "fmin ISTA:", loss_function(A, b, l, x_ista)
print

# FISTA
x_fista, pobj_fista, times_fista = fista(A, b, l, maxit)
print "xopt FISTA:", x_fista
print "fmin FISTA:", loss_function(A, b, l, x_fista)
print

# plot results
pl.close('all')
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
pl.show()
