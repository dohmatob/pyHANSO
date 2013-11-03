"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob.inria.fr>

"""

import numpy as np
import numpy.linalg


def l2(x):
    """
    l2-norm squared

    """

    return .5 * numpy.linalg.norm(x, 2) ** 2


def gradl2(x):
    return x


def l1(x):
    """
    l1-norm

    """

    return 1. * numpy.linalg.norm(x, 1)


def grad_l1(x):
    return np.array(x) / np.abs(x)


def rosenbrock_banana(x, **kwargs):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def grad_rosenbrock_banana(x, **kwargs):
    return np.array([-400 * x[0] - 2 * (
                1 - x[0]),
                      200 * (x[1] - x[0] ** 2)])


def nesterov(x, **kwargs):
    x = np.ravel(x)
    assert len(x) > 1, "1D Nesterov function makes non sense!"

    X2 = x[1:]
    X1 = x[:-1]

    return (x[0] - 1) ** 2 + np.sum(np.abs(X2 - 2 * X1 ** 2 + 1))


def grad_nesterov(x, **kwargs):
    """
    XXX this computation seems incorrect!

    """

    x = np.ravel(x)
    assert len(x) > 1, "1D Nesterov function makes non sense!"

    X2 = x[1:]
    X1 = x[:-1]
    Z = X2 - X1 ** 2 + 1.
    z = x[-1] - x[-2] ** 2 + 1.
    I = np.eye(len(x))

    return np.hstack((-2 * X1 * Z / np.abs(Z) + 2 * (X1 - 1) * I[0, :-1],
                       z / np.abs(z)))


def tv(x):
    """
    Total Variation: l1-norm of gradient

    """

    return l1(np.diff(x))


def grad_tv(x):
    """
    Gradient of Total Variation

    """

    n = len(x)  # length of signal
    index_mask = np.arange(n)  # mask for index positions
    diff1 = np.hstack((np.diff(x), 0.))  # gradient of signal
    diff2 = diff1[(index_mask - 1) % n]  # unit rightward shift of gradient

    # regularize a lil' bit (this can save lives!)
    diff1 += 1e-30 * np.random.randn()
    diff2 += 1e-30 * np.random.randn()

    # finally
    return (0 < index_mask  # all but first index
            ) * diff2 / np.abs(diff2) - (
        index_mask < n - 1  # all but last index
        ) * diff1 / np.abs(diff1)
