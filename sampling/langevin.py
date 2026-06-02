"""Unadjusted Langevin Algorithm (ULA) / Langevin Monte Carlo.

ULA samples from a Gibbs density p(x) propto exp(-U(x)/tau) by simulating the
overdamped Langevin diffusion with an Euler-Maruyama discretization:

    x_{k+1} = x_k - eta * grad_U(x_k) + sqrt(2 * eta * tau) * eps,
    eps ~ N(0, 1).

"Unadjusted" means we skip the Metropolis accept/reject step, so the chain has
a small O(eta) bias but is cheap and simple. The implementation is fully
vectorized over a batch of independent chains, which lets us populate *both*
wells of the double well regardless of how often a single chain hops the
barrier.
"""

import numpy as np
from tqdm import tqdm


def langevin_sample(grad_U, x0, eta, n_steps, tau=1.0, return_trajectory=False,
                    rng=None):
    """Run ULA chains.

    Parameters
    ----------
    grad_U : callable
        Gradient of the potential, ``grad_U(x) -> array`` (vectorized).
    x0 : array_like
        Initial state(s). Shape ``(n_chains,)`` for independent 1D chains
        (a scalar is promoted to a single chain), or ``(n_chains, d)`` for
        ``d``-dimensional chains -- ``grad_U`` just has to match that shape.
    eta : float
        Step size.
    n_steps : int
        Number of Langevin steps.
    tau : float
        Temperature.
    return_trajectory : bool
        If True, also return the full trajectory, shape ``(n_steps + 1,) +
        x.shape`` (so ``(n_steps + 1, n_chains)`` in 1D, ``(n_steps + 1,
        n_chains, d)`` in d dimensions).
    rng : np.random.Generator, optional
        Random generator (defaults to ``np.random.default_rng()``).

    Returns
    -------
    x : np.ndarray
        Final states, shape ``(n_chains,)``.
    traj : np.ndarray, optional
        Full trajectory if ``return_trajectory`` is True.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    noise_scale = np.sqrt(2.0 * eta * tau)

    traj = np.empty((n_steps + 1,) + x.shape) if return_trajectory else None
    if return_trajectory:
        traj[0] = x

    for k in tqdm(range(n_steps)):
        eps = rng.standard_normal(size=x.shape)
        x = x - eta * grad_U(x) + noise_scale * eps
        if return_trajectory:
            traj[k + 1] = x

    if return_trajectory:
        return x, traj
    return x
