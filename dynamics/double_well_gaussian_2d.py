"""A factorized 2D double well: the 1D well in x, a Gaussian trap in y.

This lifts the asymmetric 1D double well into the plane the simplest way
possible -- keep it along x and add an independent quadratic (harmonic) trap in
y:

    U(x, y) = u(x) + 1/2 * k * y**2,      u(x) = x**4 - 2 x**2 + 0.3 x.

Because the potential is a *sum* of an x-part and a y-part, the Gibbs density
factorizes,

    p(x, y) = p_x(x) * p_y(y),

with ``p_x`` the familiar bimodal 1D double-well density and ``p_y`` a centered
Gaussian of variance ``tau / k``. So the 2D target is two blobs side by side
along x -- the heavy left well near ``x = -1.05`` and the light right well near
``x = +0.95`` -- each a Gaussian ridge in y.

This is the cleanest 2D picture for *watching* Langevin dynamics: the score is a
genuine 2D vector field you can draw as arrows, yet the bimodality we care about
still lives entirely along one axis. Unlike the *radial* well in
``double_well_2d.py`` (concentric rings), the two modes here are separated along
x and split by a barrier at ``x = 0``, so a single chain's metastability -- the
difficulty of hopping wells -- is plainly visible as the particles animate.

All functions are vectorized over a leading batch axis: pass points of shape
``(..., 2)`` and get back the matching shape. ``grad_potential`` returns
``(..., 2)``, exactly the layout the (dimension-agnostic) Langevin sampler in
``sampling/langevin.py`` expects, so we reuse it unchanged to draw samples. The
``(xy, tau=...)`` signatures of :func:`gibbs_density` and :func:`score` mirror
``double_well_2d`` so the same helpers in ``plot.py`` work without changes.
"""

import numpy as np
from scipy.integrate import quad

# Stiffness of the harmonic trap in y. At temperature ``tau`` the y-marginal is
# N(0, tau / KY); KY = 2.0 gives std ~ 0.71 at tau = 1 -- tight enough that the
# two x-wells read as two separate blobs rather than smearing together.
KY = 2.0


def _u(x):
    """1D asymmetric double well u(x) = x^4 - 2 x^2 + 0.3 x."""
    return x**4 - 2.0 * x**2 + 0.3 * x


def _du(x):
    """Derivative u'(x) = 4 x^3 - 4 x + 0.3."""
    return 4.0 * x**3 - 4.0 * x + 0.3


def _split(xy):
    """Return the x and y components of points ``xy`` of shape ``(..., 2)``."""
    xy = np.asarray(xy, dtype=float)
    return xy[..., 0], xy[..., 1]


def potential(xy):
    """Potential U(x, y) = u(x) + 1/2 k y^2, shape ``(...)``.

    Parameters
    ----------
    xy : array_like, shape ``(..., 2)``
        Points in the plane; the last axis holds ``(x, y)``.
    """
    x, y = _split(xy)
    return _u(x) + 0.5 * KY * y**2


def grad_potential(xy):
    """Exact gradient ``∇U = (u'(x), k y)``, shape ``(..., 2)``."""
    x, y = _split(xy)
    return np.stack([_du(x), KY * y], axis=-1)


def gibbs_unnormalized(xy, tau=1.0):
    """Unnormalized Gibbs density exp(-U(x, y) / tau)."""
    return np.exp(-potential(xy) / tau)


def _z_x(tau=1.0, bounds=(-3.0, 3.0)):
    """1D partition function of the x-well, Z_x = ∫ exp(-u(x)/tau) dx."""
    z, _ = quad(lambda x: np.exp(-_u(x) / tau), bounds[0], bounds[1])
    return z


def _z_y(tau=1.0):
    """1D partition function of the y-trap, Z_y = sqrt(2 pi tau / k) (Gaussian)."""
    return np.sqrt(2.0 * np.pi * tau / KY)


def partition_function(tau=1.0, bounds=(-3.0, 3.0)):
    """Normalizer Z = Z_x * Z_y (the density factorizes across x and y).

    The x-part is integrated numerically over ``bounds`` (the well grows like
    x^4, so the density is negligible outside a small interval); the Gaussian
    y-part is known in closed form.
    """
    return _z_x(tau=tau, bounds=bounds) * _z_y(tau=tau)


def gibbs_density(xy, tau=1.0, bounds=(-3.0, 3.0)):
    """Normalized 2D Gibbs density p(x, y) = exp(-U/tau) / Z."""
    return gibbs_unnormalized(xy, tau=tau) / partition_function(tau=tau, bounds=bounds)


def x_marginal(x, tau=1.0, bounds=(-3.0, 3.0)):
    """Marginal density of x, p_x(x) = exp(-u(x)/tau) / Z_x (the 1D double well).

    This is the bimodal curve to compare a histogram of the samples' x-coordinate
    against; integrating out y contributes a factor of 1.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-_u(x) / tau) / _z_x(tau=tau, bounds=bounds)


def y_marginal(y, tau=1.0):
    """Marginal density of y, a centered Gaussian N(0, tau / k)."""
    y = np.asarray(y, dtype=float)
    var = tau / KY
    return np.exp(-(y**2) / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def score(xy, tau=1.0):
    """Analytic score of the data density, ``∇ log p = -∇U / tau``, shape ``(..., 2)``.

    This is also (up to the factor ``tau``) the *drift* direction of the Langevin
    dynamics: ``x_{k+1} = x_k - eta ∇U + noise = x_k + eta * tau * score + noise``,
    so the arrows drawn in the animation point the way the deterministic part of
    each step pushes a particle -- uphill in probability, into the two wells.
    """
    return -grad_potential(xy) / tau
