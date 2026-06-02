"""Generative VP-SDE sampler (Euler-Maruyama), run forward in time.

Given a trained score model ``s_theta(x, t) ~= grad_x log p_t(x)`` we generate
samples by integrating the generative SDE **forward in time**, from noise at
``t = 0`` up to the data at ``t = 1`` (the generative-time convention of
``sde.py``):

    dx = [ 0.5 beta(t) x + beta(t) s_theta(x, t) ] dt + sqrt(beta(t)) dw,
    x(0) ~ N(0, I).

Discretizing with a uniform grid of N steps and ``dt = (1 - t_eps)/N``:

    x <- x + [0.5 beta(t) x + beta(t) s_theta(x, t)] dt + sqrt(beta(t) dt) z,
    z ~ N(0, I),

stepping t upward from 0. The noise term is dropped on the final step (standard
practice, gives a cleaner final sample), and we stop at ``t = 1 - t_eps`` to stay
off the singular data end ``t = 1`` where sigma_t = 0.

The sampler is dimension-agnostic: pass a ``ScoreNet1D`` with ``dim=1`` or a
``ScoreNet3D`` with ``dim=3``.
"""

import torch

from sampling.schedule import beta as vp_beta


# ============================================================================
# QUESTION 2.3 -- Generative Euler-Maruyama step
# ============================================================================
# Student version (currently commented out -- the working implementation
# follows below).
#
# Complete the body of the for-loop. Each iteration should:
#   1. Query the score network:    score = model(x, t_batch)
#   2. Compute the deterministic drift of the generative SDE:
#        drift = 0.5 * beta_t * x + beta_t * score
#      Intuition: ``0.5 * beta_t * x`` is the *time-reversed* OU contraction
#      toward 0; the ``beta_t * score`` term pushes back *uphill* in probability
#      toward the data manifold.
#   3. Take a deterministic Euler step:    x = x + drift * dt
#   4. On every step EXCEPT THE LAST, also add a Brownian increment
#        x = x + sqrt(beta_t * dt) * z,    z = torch.randn_like(x).
#      Dropping the noise on the last step is a common trick to denoise the
#      final sample.
# ----------------------------------------------------------------------------
# @torch.no_grad()
# def generate_samples(model, n_samples, dim, n_steps=1000, t_eps=1e-3,
#                      device=None, return_trajectory=False, seed=None):
#     """Generate samples by integrating the generative VP-SDE forward in time. (Student.)"""
#     if device is None:
#         device = next(model.parameters()).device
#     if seed is not None:
#         torch.manual_seed(seed)
#
#     model.eval()
#     # Time grid running forward from noise (t=0) to (just short of) data (t=1).
#     times = torch.linspace(0.0, 1.0 - t_eps, n_steps + 1, device=device)
#     dt = (1.0 - t_eps) / n_steps
#
#     x = torch.randn(n_samples, dim, device=device)  # start from noise at t = 0
#     traj = [x.clone()] if return_trajectory else None
#
#     for i in range(n_steps):
#         t = times[i]
#         t_batch = t.expand(n_samples)
#         beta_t = vp_beta(t)
#
#         # ---------------------------------------------------------------- #
#         # >>> QUESTION 2.3: implement the generative Euler-Maruyama step.  #
#         #                                                                  #
#         #     score = ...                       # model(x, t_batch)        #
#         #     drift = ...                       # 0.5*beta_t*x + beta_t*s  #
#         #     x     = x + drift * dt            # deterministic step       #
#         #                                                                  #
#         #     if i < n_steps - 1:               # drop noise on last step  #
#         #         x = x + ... * torch.randn_like(x)                        #
#         # ---------------------------------------------------------------- #
#         raise NotImplementedError("Q2.3: implement the generative SDE step.")
#
#         if return_trajectory:
#             traj.append(x.clone())
#
#     if return_trajectory:
#         return x, torch.stack(traj, dim=0)
#     return x
# ============================================================================


@torch.no_grad()
def generate_samples(model, n_samples, dim, n_steps=1000, t_eps=1e-3,
                     device=None, return_trajectory=False, seed=None):
    """Generate samples by integrating the generative VP-SDE forward in time.

    Parameters
    ----------
    model : nn.Module
        Trained score network s_theta(x, t).
    n_samples : int
        Number of samples to draw.
    dim : int
        Data dimension (1 for the double well, 3 for Lorenz).
    n_steps : int
        Number of Euler-Maruyama steps.
    t_eps : float
        How far short of the data end (t = 1) to stop, avoiding the sigma_t = 0
        singularity.
    return_trajectory : bool
        If True, also return the ``(n_steps + 1, n_samples, dim)`` trajectory.

    Returns
    -------
    x : torch.Tensor
        Samples of shape ``(n_samples, dim)`` (in standardized data space).
    """
    if device is None:
        device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    # Time grid running forward from noise (t=0) to (just short of) data (t=1).
    times = torch.linspace(0.0, 1.0 - t_eps, n_steps + 1, device=device)
    dt = (1.0 - t_eps) / n_steps

    x = torch.randn(n_samples, dim, device=device)  # start from noise at t = 0
    traj = [x.clone()] if return_trajectory else None

    for i in range(n_steps):
        t = times[i]
        t_batch = t.expand(n_samples)
        beta_t = vp_beta(t)

        score = model(x, t_batch)
        drift = 0.5 * beta_t * x + beta_t * score
        x = x + drift * dt

        if i < n_steps - 1:  # no noise on the last step
            x = x + torch.sqrt(beta_t * dt) * torch.randn_like(x)

        if return_trajectory:
            traj.append(x.clone())

    if return_trajectory:
        return x, torch.stack(traj, dim=0)
    return x


# Backwards-compatible alias for older scripts that imported the previous name.
reverse_sde_sample = generate_samples
