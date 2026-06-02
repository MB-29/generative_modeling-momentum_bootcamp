import torch
from tqdm import tqdm

from sampling.schedule import beta_schedule, mu_schedule, sigma_schedule


@torch.no_grad()
def tweedie_denoise(model, x_t, t):
    model.eval()
    B = x_t.shape[0]
    device = x_t.device

    t_scalar = float(t)
    t_batch = torch.full((B,), t_scalar, device=device)
    mu = float(mu_schedule(t_scalar))
    sigma = float(sigma_schedule(t_scalar))

    score = model(x_t, t_batch)
    return (x_t + sigma**2 * score) / mu


# ============================================================================
# QUESTION 5 -- Generative discrete-time sampling step
# ============================================================================
# Student version (currently commented out -- the working implementation
# follows below).
#
# Complete the body of the for-loop. Each iteration should:
#   1. Query the score network:        score = model(x, t_batch)
#   2. Compute the effective step size: eta = 0.5 * beta_t * dt
#   3. Apply the deterministic part of the update:
#        x = (1 + eta) * x + 2 * eta * score
#      Intuition: ``eta * x`` un-contracts the OU shrinkage (the data is being
#      uncovered as t grows); ``2 * eta * score`` pushes uphill in probability
#      toward the data manifold -- the analogue of Section 1's ``eta * s(x_k)``
#      Langevin drift, with a factor-of-two that comes from the time-reversed
#      reading of the schedule.
#   4. On every step EXCEPT THE LAST, also add Gaussian noise
#        x = x + sqrt(2 * eta) * z,    z = torch.randn_like(x).
#      Dropping the noise on the last step is a common "final-denoise" trick.
# ----------------------------------------------------------------------------
# @torch.no_grad()
# def generate_samples(model, n_samples, dim, n_steps=1000, t_eps=1e-3,
#                      device=None, seed=None):
#     """Generate samples with the discrete-time diffusion update. (Student.)"""
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
#
#     for i in range(n_steps):
#         t = times[i]
#         t_batch = t.expand(n_samples)
#         beta_t = beta_schedule(t)
#
#         # ---------------------------------------------------------------- #
#         # >>> QUESTION 5: implement the discrete-time generative step.    #
#         #                                                                  #
#         #     score = ...                        # model(x, t_batch)       #
#         #     eta   = ...                        # 0.5 * beta_t * dt       #
#         #     x     = ...                        # (1+eta)*x + 2*eta*score #
#         #                                                                  #
#         #     if i < n_steps - 1:                # drop noise on last step #
#         #         x = x + ... * torch.randn_like(x)   # sqrt(2 * eta)      #
#         # ---------------------------------------------------------------- #
#         raise NotImplementedError("Q5: implement the discrete-time generative step.")
#
#     return x
# ============================================================================


@torch.no_grad()
def generate_samples(model, n_samples, dim, n_steps=1000, t_eps=1e-3,
                     device=None, seed=None):
    """Generate samples with the discrete-time diffusion update, forward in time.

    Parameters
    ----------
    model : nn.Module
        Trained score network s_theta(x, t).
    n_samples : int
        Number of samples to draw.
    dim : int
        Data dimension (1 for the double well, 3 for Lorenz).
    n_steps : int
        Number of discrete update steps.
    t_eps : float
        How far short of the data end (t = 1) to stop, avoiding the sigma_t = 0
        singularity.

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

    for i in tqdm(range(n_steps)):
        t = times[i]
        t_batch = t.expand(n_samples)
        beta_t = beta_schedule(t)

        score = model(x, t_batch)
        eta = 0.5 * beta_t * dt
        x = (1.0 + eta) * x + 2.0 * eta * score

        if i < n_steps - 1:  # no noise on the last step
            x = x + torch.sqrt(2.0 * eta) * torch.randn_like(x)

    return x


@torch.no_grad()
def generate_samples_trajectory(model, n_samples, dim, n_steps=1000, t_eps=1e-3,
                                device=None, seed=None):
    if device is None:
        device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    times = torch.linspace(0.0, 1.0 - t_eps, n_steps + 1, device=device)
    dt = (1.0 - t_eps) / n_steps

    x = torch.randn(n_samples, dim, device=device)
    traj = [x.clone()]

    for i in range(n_steps):
        t = times[i]
        t_batch = t.expand(n_samples)
        beta_t = beta_schedule(t)

        score = model(x, t_batch)
        eta = 0.5 * beta_t * dt
        x = (1.0 + eta) * x + 2.0 * eta * score

        if i < n_steps - 1:
            x = x + torch.sqrt(2.0 * eta) * torch.randn_like(x)

        traj.append(x.clone())

    return torch.stack(traj, dim=0)
