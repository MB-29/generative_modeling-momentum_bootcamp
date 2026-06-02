"""Lorenz '63 system, integrated with ``scipy.integrate.odeint``.

This is a pure-NumPy rewrite of the original (JAX-based) module. The classic
Lorenz '63 ODE

    x1' = sigma * (x2 - x1)
    x2' = x1 * (rho - x3) - x2
    x3' = x1 * x2 - beta * x3

with the standard chaotic parameters (sigma, rho, beta) = (10, 28, 8/3) has a
strange attractor whose invariant measure is the famous two-winged
"butterfly". We treat points sampled along a long trajectory as samples from
that invariant measure -- this is the dataset our 3D diffusion model learns.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Lorenz63:
    """Lorenz '63 dynamical system."""

    d = 3
    rho, sigma, beta = 28.0, 10.0, 8.0 / 3.0

    def __init__(self):
        self.x0 = np.array([10.0, 10.0, 30.0])

    def dynamics(self, x, t=0.0):
        """Vector field for a single state ``x`` of shape ``(3,)``.

        Signature ``(x, t)`` matches what ``scipy.integrate.odeint`` expects.
        """
        x1, x2, x3 = x
        return np.array([
            self.sigma * (x2 - x1),
            x1 * (self.rho - x3) - x2,
            x1 * x2 - self.beta * x3,
        ])

    def field(self, x):
        """Batched vector field. ``x`` has shape ``(n, 3)`` -> ``(n, 3)``."""
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        v1 = self.sigma * (x2 - x1)
        v2 = x1 * (self.rho - x3) - x2
        v3 = x1 * x2 - self.beta * x3
        return np.stack([v1, v2, v3], axis=1)

    def jacobian(self, x):
        """Analytical Jacobian of the vector field, shape ``(n, 3, 3)``."""
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        ones = np.ones_like(x1)
        zeros = np.zeros_like(x1)
        # Rows are gradients of (x1', x2', x3') w.r.t. (x1, x2, x3).
        row1 = np.stack([-self.sigma * ones, self.sigma * ones, zeros], axis=1)
        row2 = np.stack([self.rho - x3, -ones, -x1], axis=1)
        row3 = np.stack([x2, x1, -self.beta * ones], axis=1)
        return np.stack([row1, row2, row3], axis=1)

    def __repr__(self):
        return "lorenz63"


def integrate_trajectories(dynamics, x0, time_values):
    """Integrate a batch of trajectories with ``scipy.integrate.odeint``.

    Parameters
    ----------
    dynamics : callable
        Single-state vector field ``dynamics(x, t) -> dx/dt`` of shape ``(d,)``
        (e.g. :meth:`Lorenz63.dynamics`).
    x0 : np.ndarray
        Initial condition(s), shape ``(d,)`` or ``(n, d)``.
    time_values : np.ndarray
        Times at which to report the solution, shape ``(nt,)``.

    Returns
    -------
    np.ndarray
        Trajectories of shape ``(n, nt, d)``.
    """
    x0 = np.atleast_2d(np.asarray(x0, dtype=float))
    return np.stack(
        [odeint(dynamics, ic, time_values) for ic in x0], axis=0
    )


def simulate_attractor(n_steps=20000, dt=0.01, burn_in=2000, x0=None, seed=0):
    """Generate one long trajectory and return its points (after burn-in).

    The returned cloud of shape ``(n_steps - burn_in, 3)`` is our empirical
    sample of the Lorenz invariant measure (the butterfly).
    """
    model = Lorenz63()
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = model.x0 + rng.standard_normal(3)
    time_values = dt * np.arange(n_steps)
    traj = integrate_trajectories(model.dynamics, x0, time_values)[0]
    return traj[burn_in:]


if __name__ == "__main__":
    data = simulate_attractor()
    ax3d = plt.figure().add_subplot(projection="3d")
    ax3d.scatter(*data.T, color="black", s=0.5)
    ax3d.set_title("Lorenz '63 attractor")
    plt.show()
