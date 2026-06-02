"""Asymmetric 1D double-well potential and its Gibbs (Boltzmann) density.

The potential

    U(x) = x^4 - 2 x^2 + 0.3 x

has two minima of unequal depth (the linear term breaks the symmetry),
which makes it a nice toy target: a *bimodal* distribution whose two modes
have different weights.

The associated Gibbs density at temperature ``tau`` is

    p(x) = (1 / Z) * exp(-U(x) / tau),     Z = \\int exp(-U(x)/tau) dx.
"""

import numpy as np
from scipy.integrate import quad


def potential(x):
    """Double-well potential U(x) = x^4 - 2 x^2 + 0.3 x.

    Works on scalars or numpy arrays (elementwise).
    """
    x = np.asarray(x, dtype=float)
    return x**4 - 2.0 * x**2 + 0.3 * x


def grad_potential(x):
    """Exact gradient U'(x) = 4 x^3 - 4 x + 0.3."""
    x = np.asarray(x, dtype=float)
    return 4.0 * x**3 - 4.0 * x + 0.3


def gibbs_unnormalized(x, tau=1.0):
    """Unnormalized Gibbs density exp(-U(x) / tau)."""
    return np.exp(-potential(x) / tau)


def partition_function(tau=1.0, bounds=(-3.0, 3.0)):
    """Normalizing constant Z = \\int exp(-U/tau) dx by numerical integration."""
    z, _ = quad(lambda x: np.exp(-potential(x) / tau), bounds[0], bounds[1])
    return z


def gibbs_density(x, tau=1.0, bounds=(-3.0, 3.0)):
    """Normalized Gibbs density p(x) = exp(-U(x)/tau) / Z.

    The normalization constant is computed once via numerical integration
    over ``bounds`` (the potential grows like x^4, so the density is
    negligible outside a small interval).
    """
    z = partition_function(tau=tau, bounds=bounds)
    return gibbs_unnormalized(x, tau=tau) / z
