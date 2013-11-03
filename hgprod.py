"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import numpy as np


def hgprod(H0, g, S, Y):
    """
    Computes the product required by the LM-BFGS method
    See Nocedal and Wright

    """

    N = S.shape[1]  # number of saved vector pairs (x, y)
    q = np.array(g)
    alpha = np.ndarray(N)
    rho = np.ndarray(N)
    for i in xrange(N - 1, 0, -1):
        s = S[..., i]
        y = Y[..., i]
        rho[i] = 1. / np.dot(s.T, y)
        alpha[i] = rho[i] * np.dot(s.T, q)
        q -= alpha[i] * y

    r = np.dot(H0, q)
    for i in xrange(N):
        s = S[..., i]
        y = Y[..., i]
        beta = rho[i] * np.dot(y.T, r)
        r += (alpha[i] - beta) * s

    return r
