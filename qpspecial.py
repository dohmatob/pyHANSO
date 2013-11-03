"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import numpy as np
from scipy import linalg


def qpspecial(G, maxit=100, x=None, verbose=1):
    """
    Solves the QP Problem:
    min q(x) = || G * x ||_2^2 = x' * (G' * G) * x
    s.t. sum(X) = 1
             x >= 0

    """

    def _log(msg, level=0):
        if verbose > level:
            print msg

    G = np.array(G)
    if G.ndim == 1:
        G = G.reshape((-1, 1))

    [m, n] = G.shape
    if not m * n > 0:
        print "qpspecial: G is empty!"
        return [2, 0], [], [], np.inf

    e = np.ones((n, 1)) * 1.

    if x is None:
        x = np.array(e)
    else:
        x = np.array(x) * 1.
        if np.any(x < 0) or x.shape[0] != n:
            x = np.array(e)

    idx = np.arange(0, n ** 2, n + 1)
    Q = np.dot(G.T, G)
    z = np.array(x)
    y = 0
    eta = .9995
    delta = 3
    mu0 = np.dot(x.T, z) / n
    tolmu = 1e-5
    tolrs = 1e-5
    kmu = tolmu * mu0
    nQ = linalg.norm(Q, np.inf) + 2
    krs = tolrs * nQ
    ap = 0
    ad = 0
    _log("k     mu       stpsz      res")
    _log("---------------------------------")
    aborted_loop = False
    for k in xrange(maxit):
        r1 = -np.dot(Q, x) + e * y + z
        r2 = -1 + np.sum(x)
        r3 = -x * z
        rs = linalg.norm(np.vstack((r1, r2)), np.inf)
        mu = -np.sum(r3) / n

        _log('%-3.1i %9.2e %9.2e %9.2e' % (
                k, mu / mu0, max(ap, ad), rs / nQ))

        if mu < kmu:
            if rs < krs:
                info = [0, k - 1]
                aborted_loop = True
                break

        zdx = z / x
        QD = np.array(Q).ravel() * 1.
        QD[idx] = QD[idx] + zdx.ravel()
        QD = QD.reshape(Q.shape)
        C = linalg.cholesky(QD, lower=False)
        KT = linalg.solve(C.T, e, lower=True)
        M = np.dot(KT.T, KT)
        r4 = r1 + r3 / x
        r5 = np.dot(KT.T, linalg.solve(C.T, r4, lower=True))
        r6 = r2 + r5
        dy = -r6 / M
        r7 = r4 + e * dy
        dx = linalg.solve(QD, r7, sym_pos=True)
        dz = (r3 - z * dx) / x

        p = -x / dx
        ap = np.min(np.hstack((p[p > 0.], 1.)))

        p = -z / dz
        ad = np.min(np.hstack((p[p > 0.], 1.)))

        muaff = np.dot((x + ap * dx).T, z + ad * dz) / n
        sig = (muaff / mu) ** delta

        r3 = r3 + sig * mu
        r3 = r3 - dx * dz
        r4 = r1 + r3 / x
        r5 = np.dot(KT.T, linalg.solve(C.T, r4, lower=True))
        r6 = r2 + r5
        dy = -r6 / M
        r7 = r4 + e * dy
        dx = linalg.solve(QD, r7, sym_pos=True)
        dz = (r3 - z * dx) / x

        p = -x / dx
        ap = np.min(np.hstack((p[p > 0.], 1.)))

        p = -z / dz
        ad = np.min(np.hstack((p[p > 0.], 1.)))

        x = x + eta * ap * dx
        y = y + eta * ad * dy
        z = z + eta * ad * dz

    if not aborted_loop:
        info = [1, maxit]
    x = np.maximum(x, 0)
    x = x / np.sum(x)

    d = np.dot(G, x).ravel()
    q = np.dot(d.T, d)

    if verbose > 0:
        reason = "result: optimal."
        if info[0] == 1:
            reason = 'maxit reached.'
        elif info[0] == 2:
            reason = "Failed."
        _log("---------------------------------")
        _log(reason)
        _log("---------------------------------")

    return x, d, q, info


if __name__ == '__main__':
    print qpspecial(np.random.randn(100, 100))[0].T
