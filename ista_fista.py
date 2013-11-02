import time
from math import sqrt
import numpy as np
from scipy import linalg, signal

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


def hanso(A, b, lambd, maxit):
    from hanso import hanso
    from example_functions import l1, gradl1

    nstart = 200

    def func(x):
        z = np.dot(A, x) - b
        return .5 * np.dot(z.T, z) + lambd * l1(x)

    def grad(x):
        return np.dot(A.T, np.dot(A, x) - b) + lambd * gradl1(x)

    x, f = hanso(func, grad,
                 nvar=A.shape[1],
                 nstart=nstart,
                 sampgrad=True,
                 maxit=maxit // nstart,
                 verbose=2,
                 nvec=1
                 )[:2]
    import pylab as pl
    print "xopt:", x
    print "fmin:", f
    pl.plot(np.array([x, x0]).T, '*-')
    pl.show()


import pylab as pl
maxit = 100000
hanso(A, b, l, maxit)
x_ista, pobj_ista, times_ista = ista(A, b, l, maxit)
print x_ista
pl.plot(np.array([x_ista, x0]).T, '*-')
pl.show()

x_fista, pobj_fista, times_fista = fista(A, b, l, maxit)
print x_fista

pl.close('all')
pl.plot(times_ista, pobj_ista, label='ista')
pl.plot(times_fista, pobj_fista, label='fista')
pl.xlabel('Time')
pl.ylabel('Primal')
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
pl.legend()
pl.show()
