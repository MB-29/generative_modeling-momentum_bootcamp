"""A *radial* 2D double well: the 1D well, applied to the radius.

We lift the 1D asymmetric double well into the plane through the **radial**
coordinate ``r = ||(x, y)||``. Writing the 1D well as

    u(s) = s**4 - 2 s**2 + 0.3 s,

we shift it by ``R0`` and evaluate it on the radius:

    U(x, y) = u(r - R0),      r = sqrt(x**2 + y**2).

The shift ``R0`` is the key ingredient. The 1D well's two minima sit near
``s = -1.04`` (deep, heavy) and ``s = +0.96`` (shallow, light); without a shift
the heavy basin would land at a *negative* radius, which is unreachable, and the
density would be clipped at the origin. Translating by ``R0`` slides the whole
well to the right so that **both** basins fall at positive radius -- pushing the
mass onto the positive half-line where the radial coordinate lives.

Because ``U`` depends only on ``r``, the Gibbs density ``p propto exp(-U/tau)``
is **rotationally invariant**: its two modes are two concentric **circles**,

    r_inner = R0 - 1.04   (close to the origin),
    r_outer = R0 + 0.96   (further out),

separated by a circular barrier at ``r = R0 + 0.08``. The two rings carry
*different weights*. Two effects combine: the deeper inner basin has the larger
Boltzmann weight per unit area, but the outer ring has the larger circumference
(the ``2 pi r`` area element), and here the geometric factor wins -- so the outer
ring ends up heavier (~2:1 at ``tau = 1``). The asymmetry is genuine bimodality
with unequal mass, exactly the feature we want a generative model to reproduce.

All functions are vectorized over a leading batch axis: pass points of shape
``(..., 2)`` and get back the matching shape. ``grad_potential`` returns
``(..., 2)``, exactly what the (dimension-agnostic) Langevin sampler in
``sampling/langevin.py`` expects, so we reuse it unchanged to draw training data.
"""

import numpy as np
from scipy.integrate import quad

# Radial shift of the well. R0 > 1.04 places BOTH basins at positive radius;
# R0 = 1.5 keeps the inner ring close to the origin (r ~ 0.46) while the outer
# ring (r ~ 2.46) sits clearly further out.
R0 = 1.5


def _u(s):
    """1D asymmetric double well u(s) = s^4 - 2 s^2 + 0.3 s."""
    return s**4 - 2.0 * s**2 + 0.3 * s


def _du(s):
    """Derivative u'(s) = 4 s^3 - 4 s + 0.3."""
    return 4.0 * s**3 - 4.0 * s + 0.3


# Critical radii (inner min, barrier, outer min), found once from u'(s) = 0.
_roots = np.sort(np.roots([4.0, 0.0, -4.0, 0.3]).real)
R_INNER, R_BARRIER, R_OUTER = (_roots + R0)


def _radius(xy):
    """Euclidean radius of points ``xy`` of shape ``(..., 2)``; shape ``(...)``."""
    xy = np.asarray(xy, dtype=float)
    return np.sqrt(xy[..., 0] ** 2 + xy[..., 1] ** 2)


def radial_potential(r):
    """The 1D well evaluated on the (shifted) radius, ``u(r - R0)``."""
    return _u(np.asarray(r, dtype=float) - R0)


def potential(xy):
    """Radial double-well potential U(x, y) = u(r - R0), shape ``(...)``.

    Parameters
    ----------
    xy : array_like, shape ``(..., 2)``
        Points in the plane; the last axis holds ``(x, y)``.
    """
    return radial_potential(_radius(xy))


def grad_potential(xy):
    """Exact gradient ``∇U = u'(r - R0) * (x, y) / r``, shape ``(..., 2)``.

    The gradient is purely radial. At the origin ``(x, y) / r`` is replaced by
    zero (a safe value: the origin carries no mass and the area element vanishes
    there), which keeps the Langevin sampler from dividing by zero.
    """
    xy = np.asarray(xy, dtype=float)
    r = _radius(xy)
    r_safe = np.maximum(r, 1e-12)
    radial = _du(r - R0) / r_safe          # scalar coefficient u'(r-R0)/r
    return radial[..., None] * xy           # (..., 2)


def gibbs_unnormalized(xy, tau=1.0):
    """Unnormalized Gibbs density exp(-U(x, y) / tau)."""
    return np.exp(-potential(xy) / tau)


def partition_function(tau=1.0, r_max=6.0):
    """Normalizer Z = ∫∫ exp(-U/tau) dx dy.

    Rotational invariance collapses the 2D integral to a 1D radial quadrature
    with the polar area element ``2 pi r``::

        Z = ∫_0^∞ exp(-u(r - R0)/tau) * 2 pi r dr.
    """
    z, _ = quad(lambda r: np.exp(-_u(r - R0) / tau) * 2.0 * np.pi * r, 0.0, r_max)
    return z


def gibbs_density(xy, tau=1.0, r_max=6.0):
    """Normalized 2D Gibbs density p(x, y) = exp(-U/tau) / Z."""
    return gibbs_unnormalized(xy, tau=tau) / partition_function(tau=tau, r_max=r_max)


def radial_marginal(r, tau=1.0, r_max=6.0):
    """Marginal density of the radius, p_R(r) = (2 pi r / Z) exp(-u(r-R0)/tau).

    This is the bimodal distribution one sees in a histogram of the samples'
    radii; its two peaks are the inner and outer rings. The ``2 pi r`` factor
    (the polar Jacobian) is what makes the outer ring heavier despite the inner
    basin being deeper.
    """
    r = np.asarray(r, dtype=float)
    z = partition_function(tau=tau, r_max=r_max)
    return 2.0 * np.pi * r * np.exp(-_u(r - R0) / tau) / z


def score(xy, tau=1.0):
    """Analytic score of the *data* density, ``∇ log p(x, y) = -∇U / tau``.

    Shape ``(..., 2)``. This is the ``t -> 0`` target the trained score network
    should reproduce; we overlay it on the learned field for comparison.
    """
    return -grad_potential(xy) / tau
