"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import numpy as np


def hgprod(H0, g, S, Y):
    """
    Computes the product required by the LM-BFGS method
    See Nocedal and Wright

    """

    q = np.array(g)
    r = np.dot(H0, q)

    if len(S) == 0:
        return r

    S = np.array(S)
    Y = np.array(Y)
    S = S.reshape((-1, 1)) if S.ndim == 1 else S
    Y = Y.reshape((-1, 1)) if Y.ndim == 1 else Y

    N = S.shape[1]  # number of saved vector pairs (x, y)
    alpha = np.ndarray(N)
    rho = np.ndarray(N)
    for i in xrange(N - 1, 0, -1):
        s = S[..., i]
        y = Y[..., i]
        rho[i] = 1. / np.dot(s.T, y)
        alpha[i] = rho[i] * np.dot(s.T, q)
        q -= alpha[i] * y

    for i in xrange(N):
        s = S[..., i]
        y = Y[..., i]
        beta = rho[i] * np.dot(y.T, r)
        r += (alpha[i] - beta) * s

    return r


if __name__ == '__main__':
    H0 = np.eye(2)
    g = np.array([1, -1])
    Y = np.eye(2)
    S = np.eye(2)
    S[1, 1] = np.pi

    print hgprod(H0, g, S, Y)
