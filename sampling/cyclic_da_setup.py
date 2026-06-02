"""Shared setup for the cyclic Lorenz '63 data-assimilation experiment.

Both the diffusion DA (``scripts/lorenz_cyclic_da.py``) and the 3D-Var baseline
(``scripts/lorenz_cyclic_3dvar.py``) -- and the notebook -- run on the *same*
truth trajectory and the *same* noisy observations. Centralizing the operator
``H_PHYS``, the observed-component indices ``OBS_IDX`` and the
``build_truth_and_obs`` factory here guarantees a single ground truth across
all three call sites, so any RMSE difference reflects the method, not the
experimental setup.

Observation model
-----------------
We observe the second and third components of the Lorenz state and leave the
first one hidden::

    H_PHYS = [[0, 1, 0],
              [0, 0, 1]],     y = H_PHYS @ x + nu,   nu ~ N(0, sigma_obs^2 I).

The hidden ``x_1`` is what makes the posterior bimodal -- it lives on either
butterfly wing, and the diffusion prior is needed to capture that.
"""

import numpy as np

from dynamics.lorenz import Lorenz63, integrate_trajectories


# Observation operator: select (x_2, x_3). x_1 stays hidden.
H_PHYS = np.array([[0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

# Indices of the observed components in (x_1, x_2, x_3).
OBS_IDX = np.array([1, 2])


def build_truth_and_obs(n_cycles, dt_cycle, sigma_obs_phys, *,
                        seed=0, spinup_T=20.0, spinup_steps=2000):
    """Truth trajectory + noisy observations at every cycle time.

    A single long ``odeint`` spin-up from ``Lorenz63().x0`` lands a point on
    the attractor; that point is then re-integrated at the cycle times so the
    truth is exactly the attractor trajectory we will assimilate against.

    Observations are ``y_k = H_PHYS @ truth_k + sigma_obs_phys * eps_k`` with
    ``eps_k ~ N(0, I_2)`` drawn from ``np.random.default_rng(seed)`` -- so two
    calls with the same ``seed`` and ``(n_cycles, dt_cycle, sigma_obs_phys)``
    return *exactly* the same ``(cycle_times, truth, y_obs)``. That property is
    the whole point of this helper.

    Parameters
    ----------
    n_cycles : int
        Number of assimilation cycles.
    dt_cycle : float
        Spacing between successive cycles in Lorenz time units.
    sigma_obs_phys : float
        Observation-noise std in physical space (same on both observed
        components).
    seed : int, default 0
        RNG seed for the observation noise.
    spinup_T : float, default 20.0
        Length of the burn-in integration onto the attractor.
    spinup_steps : int, default 2000
        Number of ``odeint`` points during burn-in.

    Returns
    -------
    cycle_times : ndarray (n_cycles,)
    truth       : ndarray (n_cycles, 3)
    y_obs       : ndarray (n_cycles, 2)
    """
    rng = np.random.default_rng(seed)
    dyn = Lorenz63()
    spinup_t = np.linspace(0.0, spinup_T, spinup_steps)
    x_truth0 = integrate_trajectories(dyn.dynamics, dyn.x0, spinup_t)[0, -1]
    cycle_times = dt_cycle * np.arange(n_cycles)
    truth = integrate_trajectories(dyn.dynamics, x_truth0, cycle_times)[0]
    y_obs = (truth @ H_PHYS.T
             + sigma_obs_phys * rng.standard_normal((n_cycles, 2)))
    return cycle_times, truth, y_obs
