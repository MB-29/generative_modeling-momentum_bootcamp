"""Conditional sampling for a linear-Gaussian inverse problem.

We observe only the z-component (last coordinate) of the Lorenz state through

    y = H x + nu,     H = [0, 0, 1],     nu ~ N(0, sigma_obs**2).

Bayes' rule gives the posterior  p(x | y) propto p(x) p(y | x), so its score
decomposes as a sum

    grad_x log p(x | y) = grad_x log p(x)  +  grad_x log p(y | x).

The prior score is exactly the trained diffusion model s_theta(x, t). For the
likelihood we add a closed-form Gaussian score (no DPS / no Tweedie denoising of
x_t, no back-prop through the network -- we literally add the two scores).

We use the generative-time convention of ``sde.py`` (t=1 is the data, t=0 is
noise), so the conditional sampler runs forward in t exactly like the
unconditional one, only with the posterior score.

Two guidance modes are supported:

(1) **Likelihood only** -- pass ``y`` and ``observation`` (the original Section-4
    inverse problem). The posterior score is

        s_post(x_t, t) = s_theta(x_t, t) + grad log N(y; H x, sigma_obs**2 I).

(2) **3D-Var-style Gaussian guidance** -- pass ``y``, ``observation``, a single
    background mean ``x_b``, and the prior covariance ``B``. The Gaussian piece
    is then the *exact* 3D-Var posterior

        N(x_a, P_a),   P_a^{-1} = B^{-1} + H^T R^{-1} H,
                       x_a     = x_b + K (y - H x_b),
                       K       = B H^T (H B H^T + R)^{-1},

    whose score is ``P_a^{-1} (x_a - x)``. The posterior score becomes

        s_post(x_t, t) = s_theta(x_t, t) + P_a^{-1} (x_a - x_t).

    With s_theta = 0 the sampler reproduces the 3D-Var posterior (mean and
    covariance); with s_theta enabled the diffusion score adds an attractor
    regularizer on top. This is the form used by the cyclic Lorenz DA experiment
    in ``scripts/lorenz_cyclic_da.py`` -- it guarantees the diffusion sampler is
    a strict improvement over 3D-Var on the Gaussian part.

Everything operates in the standardized data space the score model was trained
on -- standardize ``y``, ``sigma_obs`` (or ``R``), ``x_b`` and ``B`` accordingly.
"""

import numpy as np
import torch

from sampling.schedule import beta as vp_beta


def default_H(device=None):
    """Observation matrix selecting the z-component (last coord), shape ``(1, 3)``."""
    return torch.tensor([[.0, .0, 1.0]], device=device)


class ObservationOperator:
    """Linear-Gaussian observation model y = H x + nu."""

    def __init__(self, H=None, sigma_obs=0.1, device=None):
        self.H = default_H(device) if H is None else torch.as_tensor(
            H, dtype=torch.float32, device=device)
        self.sigma_obs = float(sigma_obs)

    def observe(self, x, seed=None):
        """Generate a noisy observation of state(s) ``x``.

        ``x`` has shape ``(d,)`` or ``(B, d)``; returns ``(m,)`` or ``(B, m)``.
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.H.device)
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        if seed is not None:
            torch.manual_seed(seed)
        Hx = x @ self.H.T  # (B, m)
        y = Hx + self.sigma_obs * torch.randn_like(Hx)
        return y.squeeze(0) if single else y


def likelihood_score(x, y, H, sigma_obs):
    """Clean-data Gaussian likelihood score ``H^T R^{-1} (y - H x)`` with
    isotropic ``R = sigma_obs**2 I``.

    Used by the Section-4 (single-observation) inverse problem, which does not
    have a Gaussian background term. ``x : (B, d), y : (m,), H : (m, d) -> (B, d)``.
    """
    residual = y - x @ H.T                       # (B, m)
    return residual @ H / (sigma_obs ** 2)        # (B, d)


def var3d_components(x_b, y, H, R, B):
    """Pre-compute the 3D-Var guidance building blocks.

    Returns ``(x_b, B_inv, HtRinvH, HtRinvY)`` as torch tensors matching
    ``H``'s dtype/device. With weights ``w_bg = w_obs = 1`` the sum

        w_bg * B^{-1} (x_b - x)  +  w_obs * (H^T R^{-1} y - H^T R^{-1} H x)
       = P_a^{-1} (x_a - x)

    is the exact 3D-Var posterior score, with ``P_a^{-1} = B^{-1} + H^T R^{-1} H``
    and ``x_a = x_b + K(y - H x_b)``. Reducing either weight widens the implied
    Gaussian (smaller precision) -- a stability knob for stiff R, and a way to
    let the diffusion regularizer do relatively more work. Computed in float64
    for conditioning.
    """
    H_np = H.detach().cpu().numpy().astype(np.float64)
    R_np = np.asarray(R, dtype=np.float64)
    B_np = np.asarray(B, dtype=np.float64)
    x_b_np = np.asarray(x_b, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)

    B_inv_np = np.linalg.inv(B_np)                               # (d, d)
    R_inv_H_np = np.linalg.solve(R_np, H_np)                     # (m, d)
    HtRinvH_np = H_np.T @ R_inv_H_np                             # (d, d)
    HtRinvY_np = H_np.T @ np.linalg.solve(R_np, y_np)            # (d,)

    dtype, device = H.dtype, H.device

    def t(a): return torch.as_tensor(a, dtype=dtype, device=device)
    return t(x_b_np), t(B_inv_np), t(HtRinvH_np), t(HtRinvY_np)


def var3d_analysis(x_b, B_inv, HtRinvH, HtRinvY):
    """3D-Var analysis mean ``x_a = P_a (B^{-1} x_b + H^T R^{-1} y)`` from the
    pre-computed components. Useful for diagnostics; the sampler itself does
    not need ``x_a`` explicitly."""
    P_a_inv = B_inv + HtRinvH
    rhs = B_inv @ x_b + HtRinvY
    return torch.linalg.solve(P_a_inv, rhs), P_a_inv


def compute_posterior_score(model, x, t, *,
                            y=None, H=None, sigma_obs=None,
                            x_b=None, B_inv=None, HtRinvH=None, HtRinvY=None,
                            weight_bg=1.0, weight_obs=1.0):
    """Posterior score = prior score s_theta(x, t) + Gaussian guidance.

    Two guidance modes:
      - 3D-Var Gaussian (``B_inv`` & friends provided): adds
            weight_bg * B^{-1}(x_b - x) + weight_obs * (H^T R^{-1} y - H^T R^{-1} H x).
        With ``weight_bg = weight_obs = 1`` the guidance score equals the exact
        3D-Var posterior score ``P_a^{-1}(x_a - x)``. Reducing the weights is a
        stability knob -- the resulting Gaussian is wider, so a stiff R does not
        blow up the Euler step.
      - Likelihood only (``y``, ``H``, ``sigma_obs`` provided): adds
        ``H^T R^{-1}(y - H x)`` with ``R = sigma_obs^2 I``.

    ``t`` is a scalar tensor; expanded to a batch for the network.
    """
    t_batch = t.expand(x.shape[0])
    score = model(x, t_batch)
    if B_inv is not None:
        # Symmetric matrices: A @ v.T == v @ A for batched v of shape (N, d).
        score = score + weight_bg * (x_b - x) @ B_inv
        score = score + weight_obs * (HtRinvY - x @ HtRinvH)
    else:
        score = score + likelihood_score(x, y, H, sigma_obs)
    return score


@torch.no_grad()
def conditional_sample(model, y, observation, n_samples, *,
                       background=None, B=None, R=None,
                       weight_bg=1.0, weight_obs=1.0,
                       dim=3, n_steps=1000, t_eps=1e-3,
                       device=None, return_trajectory=False, seed=None):
    """Sample from p(x | y) by integrating the generative VP-SDE forward in time
    with the posterior score.

    Parameters
    ----------
    model : nn.Module
        Trained prior score network s_theta(x, t).
    y : array_like, shape (m,)
        Observation in *standardized* space.
    observation : ObservationOperator
        Provides ``H`` and (optionally) the scalar ``sigma_obs``.
    n_samples, dim, n_steps, t_eps : sampler settings.
    background : array_like, optional
        3D-Var prior mean ``x_b``, shape ``(d,)`` (or ``(N, d)``; the ensemble
        mean is used). Required to enable 3D-Var Gaussian guidance.
    B : array_like, optional
        3D-Var prior covariance ``(d, d)`` in standardized space. When provided
        together with ``background``, the Gaussian guidance term is the 3D-Var
        posterior score ``weight_bg B^{-1}(x_b - x) + weight_obs H^T R^{-1}(y - Hx)``
        and replaces the simple likelihood term. With both weights = 1 this is
        the exact 3D-Var posterior; the diffusion score adds an attractor
        regularizer on top.
    R : array_like, optional
        Observation-error covariance ``(m, m)``. Defaults to
        ``observation.sigma_obs ** 2 * I``.
    weight_bg, weight_obs : float
        Multiplicative weights on the background ``B^{-1}(x_b - x)`` and
        observation ``H^T R^{-1}(y - Hx)`` terms. Defaults of 1 give the exact
        3D-Var posterior covariance ``P_a``; smaller values widen the implied
        Gaussian (a stability knob when ``R`` is small / precision is stiff).

    Returns
    -------
    torch.Tensor
        Conditional samples, shape ``(n_samples, dim)`` (standardized space).
    """
    if device is None:
        device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    H = observation.H.to(device)
    sigma_obs = observation.sigma_obs
    y = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1)

    # ---- Pre-compute the 3D-Var components once if B is provided ----------- #
    x_b = B_inv = HtRinvH = HtRinvY = None
    if B is not None:
        if background is None:
            raise ValueError("3D-Var guidance requires `background` (x_b).")
        bg_np = np.asarray(background, dtype=np.float64)
        x_b_np = bg_np.mean(axis=0) if bg_np.ndim == 2 else bg_np
        m = H.shape[0]
        R_eff = (sigma_obs ** 2 * np.eye(m)
                 ) if R is None else np.asarray(R, dtype=np.float64)
        x_b, B_inv, HtRinvH, HtRinvY = var3d_components(
            x_b_np, y.detach().cpu().numpy(), H=H, R=R_eff, B=B,
        )

    times = torch.linspace(0.0, 1.0 - t_eps, n_steps + 1, device=device)
    dt = (1.0 - t_eps) / n_steps

    x = torch.randn(n_samples, dim, device=device)  # start from noise at t = 0
    traj = [x.clone()] if return_trajectory else None

    for i in range(n_steps):
        t = times[i]
        beta_t = vp_beta(t)

        score = compute_posterior_score(
            model, x, t, y=y, H=H, sigma_obs=sigma_obs,
            x_b=x_b, B_inv=B_inv, HtRinvH=HtRinvH, HtRinvY=HtRinvY,
            weight_bg=weight_bg, weight_obs=weight_obs,
        )
        drift = 0.5 * beta_t * x + beta_t * score
        x = x + drift * dt

        if i < n_steps - 1:
            x = x + torch.sqrt(beta_t * dt) * torch.randn_like(x)

        if return_trajectory:
            traj.append(x.clone())

    if return_trajectory:
        return x, torch.stack(traj, dim=0)
    return x
