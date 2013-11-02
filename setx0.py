"""
:Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com> <elvis.dohmatob@inria.fr>

"""

import numpy.random


def setx0(nvar, nstart, **kwargs):
    """
    set columns of x0 randomly if not provided by user

    Returns
    -------
    x0: 2D array of shape (nvar, nstart)
        Starting point for BFGS; one point percolumn

   Raises
   ------
   RuntimeError

   """

    if not nstart > 0:
        raise RuntimeError(
            'setx0: input "options.nstart" must be a positive integer '
            'when "options.x0" is not provided')
    else:
        return numpy.random.randn(nvar, nstart)
