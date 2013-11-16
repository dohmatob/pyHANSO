"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import numpy as np


def hgprod(H0, g, S, Y):
    """
    Computes the product required by the LM-BFGS method
    See Nocedal and Wright

    """

    q = g
    r = H0.dot(q)

    if len(S) == 0:
        return r

    S = S[:, np.newaxis] if S.ndim == 1 else S
    Y = Y[:, np.newaxis] if Y.ndim == 1 else Y

    N = S.shape[1]  # number of saved vector pairs (x, y)
    alpha = np.ndarray(N)
    rho = np.ndarray(N)
    for i in xrange(N - 1, 0, -1):
        s = S[..., i]
        y = Y[..., i]
        rho[i] = 1. / s.T.dot(y)
        alpha[i] = rho[i] * s.T.dot(q)
        q -= alpha[i] * y

    r = H0.dot(q)
    for i in xrange(N):
        s = S[..., i]
        y = Y[..., i]
        beta = rho[i] * y.T.dot(r)
        r += (alpha[i] - beta) * s

    return r


if __name__ == '__main__':
    H0 = np.eye(2)
    g = np.array([1, -1])
    Y = np.eye(2)
    S = np.eye(2)
    S[1, 1] = np.pi

    print hgprod(H0, g, S, Y)
