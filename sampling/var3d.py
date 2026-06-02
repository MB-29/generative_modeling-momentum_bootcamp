"""3D-Var deterministic baseline for the cyclic Lorenz '63 experiment.

Variational data assimilation finds the analysis state by minimizing the
classical 3D-Var cost function

    J(x) = 1/2 (x - x_b)^T B^{-1} (x - x_b)
         + 1/2 (y - H x)^T R^{-1} (y - H x),                              (1)

i.e. a Gaussian prior centered on the forecast ``x_b`` (covariance ``B``)
combined with a Gaussian observation likelihood (covariance ``R``). For a
linear observation operator and Gaussian errors the minimizer is closed-form

    x_a = x_b + K (y - H x_b),
    K   = B H^T (H B H^T + R)^{-1}.                                       (2)

This is the standard "best-linear-unbiased" analysis -- equivalent to one step
of a Kalman filter with a static ``B``. We use it here as the deterministic,
Gaussian-prior baseline against which the diffusion conditional sampler in
``sampling/inverse_problem.py`` is compared on the cyclic Lorenz '63 setup of
``scripts/lorenz_cyclic_da.py``.

Conceptual differences w.r.t. the diffusion baseline
----------------------------------------------------
1. **Prior**. 3D-Var uses a *Gaussian* climatology prior with a fixed mean
   ``x_b`` (the forecast) and a fixed covariance ``B`` (e.g. the climatological
   covariance of the long Lorenz trajectory). The diffusion baseline uses the
   *learned* non-Gaussian climatology score and ignores the forecast as a prior
   mean -- so a 3D-Var update is sensitive to ``x_b`` whereas the diffusion
   update is not.
2. **Output**. 3D-Var produces a single best-estimate state (the posterior
   mean = mode of the Gaussian posterior). The diffusion sampler produces an
   ensemble that captures non-Gaussian / multi-modal posterior structure -- in
   the cyclic Lorenz experiment this is exactly the bimodality across the two
   butterfly wings that 3D-Var cannot represent.
3. **Cost**. Both updates are O(d^3) here. 3D-Var is essentially free; the
   diffusion sampler integrates a reverse SDE for ``n_steps * ensemble_size``
   network evaluations.

API
---
The free-function API mirrors ``sampling.inverse_problem`` so both baselines
can be swapped in the notebook / scripts:

    >>> B = climatological_B(lorenz_data)
    >>> R = sigma_obs**2 * np.eye(H.shape[0])
    >>> res = run_cyclic_3dvar(truth, y_obs, cycle_times, H, R, B,
    ...                        dynamics=Lorenz63().dynamics,
    ...                        x_b0=lorenz_data.mean(0))

The result dict matches ``scripts/lorenz_cyclic_da.run_cyclic_da`` so every
``plot.cyclic_da.*`` helper plots it unchanged (the ensemble axis is simply of
size 1 here, and ``spread_*`` is identically zero -- 3D-Var is deterministic).
"""

import numpy as np

from dynamics.lorenz import integrate_trajectories


# --------------------------------------------------------------------------- #
# 3D-Var primitives. Everything is float64 -- B is small (3x3 for Lorenz) but
# its conditioning matters for the Kalman gain solve.
# --------------------------------------------------------------------------- #

def climatological_B(data, inflation=1.0):
    """Static background covariance from a long climatology cloud.

    ``data`` has shape ``(N, d)`` (e.g. the Lorenz attractor cloud from
    ``simulate_attractor``). The empirical sample covariance is the standard
    "NMC" / climatological estimator of ``B`` when no flow-dependent
    information is available, and an ``inflation`` factor lets the user widen
    it if the assimilation diverges (a common 3D-Var knob).
    """
    data = np.asarray(data, dtype=np.float64)
    B = np.cov(data, rowvar=False)
    return float(inflation) * B


def gain_matrix(B, H, R):
    """Kalman gain ``K = B H^T (H B H^T + R)^{-1}`` for linear ``H``.

    Solved with ``np.linalg.solve`` rather than an explicit inverse for
    numerical stability. Shapes: ``B (d,d), H (m,d), R (m,m) -> K (d,m)``.
    """
    B = np.asarray(B, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    S = H @ B @ H.T + R                      # (m, m) innovation covariance
    # K^T solves  S^T K^T = (B H^T)^T  ->  K = (B H^T) S^{-1}.
    return np.linalg.solve(S.T, (B @ H.T).T).T


def analysis_3dvar(x_b, y, H, R, B, K=None):
    """Closed-form 3D-Var analysis ``x_a = x_b + K (y - H x_b)``.

    Pass a precomputed ``K`` when cycling so the gain is built once.
    Shapes: ``x_b (d,), y (m,) -> x_a (d,)``.
    """
    x_b = np.asarray(x_b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if K is None:
        K = gain_matrix(B, H, R)
    innovation = y - H @ x_b                 # (m,)  -- "obs minus background"
    return x_b + K @ innovation


def cost_3dvar(x, x_b, y, H, B_inv, R_inv):
    """3D-Var cost ``J(x)`` from eq. (1). Useful as a sanity check and as the
    starting point if one ever needs an iterative minimizer (e.g. nonlinear
    H, where (2) no longer applies)."""
    dx = x - x_b
    dy = y - H @ x
    return 0.5 * dx @ B_inv @ dx + 0.5 * dy @ R_inv @ dy


# --------------------------------------------------------------------------- #
# Cyclic loop. Same data flow as scripts/lorenz_cyclic_da.run_cyclic_da, so
# the same truth / y_obs / cycle_times can drive either baseline.
# --------------------------------------------------------------------------- #

def run_cyclic_3dvar(truth, y_obs, cycle_times, H, R, B, dynamics, x_b0,
                     n_substeps=50, inflation=1.0, verbose=False):
    """Run the cyclic 3D-Var baseline and return a ``plot.cyclic_da``-shaped dict.

    Parameters
    ----------
    truth : ndarray (n_cycles, d)
        True state at each cycle time -- used only for RMSE diagnostics.
    y_obs : ndarray (n_cycles, m)
        Noisy observations ``y_k = H truth_k + nu_k``.
    cycle_times : ndarray (n_cycles,)
        Cycle times; used for forecast step lengths and as the time axis of
        the output diagnostics.
    H : ndarray (m, d)
        Linear observation operator (e.g. ``[[0,1,0],[0,0,1]]`` for the
        cyclic-DA experiment where ``x_2, x_3`` are observed).
    R : ndarray (m, m)
        Observation error covariance (typically ``sigma_obs**2 * I``).
    B : ndarray (d, d)
        Static background covariance -- usually ``climatological_B(data)``.
    dynamics : callable
        Single-state ODE vector field ``dynamics(x, t) -> dx/dt`` compatible
        with ``scipy.integrate.odeint`` (e.g. ``Lorenz63().dynamics``).
    x_b0 : array_like, shape (d,)
        Initial background state at cycle 0. The climatology mean
        ``lorenz_data.mean(0)`` is a sensible default -- no prior info.
    n_substeps : int
        ``odeint`` sub-steps per forecast leg; matches the diffusion baseline.
    inflation : float
        Multiplicative inflation applied to ``B`` after construction. Apply
        externally (via :func:`climatological_B`) if you prefer.
    verbose : bool
        Per-cycle one-line summary on stdout, like the diffusion script.

    Returns
    -------
    dict
        Same keys as ``run_cyclic_da`` so ``plot.cyclic_da.*`` works unchanged.
        ``background``/``analysis`` have ensemble axis of length 1 and
        ``spread_*`` is identically zero (deterministic baseline). The gain
        ``K`` and prior covariance ``B_eff = inflation * B`` are also returned
        for inspection.
    """
    truth = np.asarray(truth, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    cycle_times = np.asarray(cycle_times, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    B_eff = float(inflation) * np.asarray(B, dtype=np.float64)

    n_cycles, d = truth.shape
    K = gain_matrix(B_eff, H, R)

    background = np.zeros((n_cycles, 1, d))
    analysis = np.zeros((n_cycles, 1, d))
    rmse_a = np.zeros(n_cycles)
    rmse_b = np.zeros(n_cycles)

    x_b = np.asarray(x_b0, dtype=np.float64).copy()

    if verbose:
        print(f"3D-Var: {n_cycles} cycles, gain trace(K H) = "
              f"{np.trace(K @ H):.3f}  (d.o.f. used by the update)")

    for k in range(n_cycles):
        background[k, 0] = x_b
        x_a = analysis_3dvar(x_b, y_obs[k], H, R, B_eff, K=K)
        analysis[k, 0] = x_a

        rmse_b[k] = np.sqrt(((x_b - truth[k]) ** 2).mean())
        rmse_a[k] = np.sqrt(((x_a - truth[k]) ** 2).mean())
        if verbose:
            print(f"  cycle {k:2d}  t={cycle_times[k]:5.2f}  "
                  f"rmse_b={rmse_b[k]:.3f} -> rmse_a={rmse_a[k]:.3f}")

        # Forecast: propagate the single analysis state to the next cycle.
        if k < n_cycles - 1:
            dt_cycle = cycle_times[k + 1] - cycle_times[k]
            t_grid = np.linspace(0.0, dt_cycle, n_substeps + 1)
            x_b = integrate_trajectories(dynamics, x_a, t_grid)[0, -1]

    return {
        "times": cycle_times,
        "truth": truth,
        "y_obs": y_obs,
        "background": background,
        "analysis": analysis,
        "rmse_a": rmse_a,
        "rmse_b": rmse_b,
        "spread_a": np.zeros(n_cycles),
        "spread_b": np.zeros(n_cycles),
        "B": B_eff,
        "K": K,
    }
