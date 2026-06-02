
import numpy as np
from tqdm import tqdm


def langevin_sample(grad_f, x0, eta, n_steps):
    """Run Langevin Monte Carlo and return final states.

    Parameters
    ----------
    grad_f : callable
        Gradient of the potential, ``grad_f(x) -> array`` (vectorized).
    x0 : array_like
        Initial state(s). Shape ``(n_chains,)`` for independent 1D chains
        (a scalar is promoted to a single chain), or ``(n_chains, d)`` for
        ``d``-dimensional chains -- ``grad_f`` just has to match that shape.
    eta : float
        Step size.
    n_steps : int
        Number of Langevin steps.

    Returns
    -------
    x : np.ndarray
        Final states, same shape as ``x0``.
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    noise_scale = np.sqrt(2.0 * eta)

    for index in tqdm(range(n_steps)):
        noise = np.random.randn(*x.shape)
        x = x - eta * grad_f(x) + noise_scale * noise

    return x


def langevin_trajectory(grad_f, x0, eta, n_steps):
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    noise_scale = np.sqrt(2.0 * eta)

    traj = np.empty((n_steps + 1,) + x.shape)
    traj[0] = x

    for k in tqdm(range(n_steps)):
        eps = np.random.randn(*x.shape)
        x = x - eta * grad_f(x) + noise_scale * eps
        traj[k + 1] = x

    return traj
