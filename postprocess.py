"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import numpy as np
import numpy.linalg
from qpspecial import qpspecial


def postprocess(x, g, dnorm, X, G, w, verbose=1):
    """
    postprocessing of set of sampled or bundled gradients
    if x is not one of the columns of X, prepend it to X and
    g to G and recompute w and dnorm: this can only reduce dnorm
    also set loc.dnorm to dnorm and loc.evaldist to the
    max distance from x to columns of X
    note: w is needed as input argument for the usual case that
    w is not recomputed but is just passed back to output

    """

    dist = [numpy.linalg.norm(x - X[..., j], 2) for j in xrange(X.shape[1])]

    evaldist = np.max(dist)  # for returning
    indx = np.argmin(dist)  # for checking if x is a column of X
    mindist = dist[indx]

    if mindist == 0 and indx == 1:
        # nothing to do
        pass
    elif mindist == 0 and indx > 1:
        # this should not happen in HANSO 2.0
        # swap x and g into first positions of X and G
        # might be necessary after local bundle, which is not used in HANSO 2.0
        X[..., [1, indx]] = X[..., [indx, 1]]
        G[..., [1, indx]] = G[..., [indx, 1]]
        w[..., [1, indx]] = w[..., [indx, 1]]
    else:
        # this cannot happen after BFGS, but it may happen after gradient
        # sampling, for example if max iterations exceeded: line search found a
        # lower point but quit before solving new QP
        # prepend x to X and g to G and recompute w
        X = np.vstack((x, X.T)).T
        if not np.any(np.isnan(g)):
            G = np.hstack((g, G))
        w, d, _, _ = qpspecial(G, verbose=verbose)  # Anders Skajaa's QP code
        dnorm = numpy.linalg.norm(d, 2)

    return {"dnorm": dnorm, "evaldist": evaldist}, X, G, w
