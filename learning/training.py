"""Denoising Score Matching (DSM) training for the VP-SDE.

We use the generative-time convention of ``sde.py`` (t=1 is the data, t=0 is
noise). For a clean sample x_0 we form a noisy sample at a random time t:

    x_t = mu_t * x_0 + sigma_t * eps,      eps ~ N(0, I),  t ~ U(0, 1).

By Tweedie's formula the score of the Gaussian transition kernel is

    grad_{x_t} log p(x_t | x_0) = -(x_t - mu_t x_0) / sigma_t**2 = -eps / sigma_t,

so the DSM objective trains s_theta(x_t, t) to match -eps / sigma_t:

    L(theta) = E || s_theta(x_t, t) + eps / sigma_t ||**2.            (CLAUDE.md)

NUMERICAL NOTE -- weighting.
The target -eps/sigma_t blows up as t -> 0 (sigma_t -> 0), which makes the raw
loss extremely ill-conditioned. We therefore train the *likelihood-weighted*
version obtained by multiplying the integrand by lambda(t) = sigma_t**2:

    L_w(theta) = E || sigma_t * s_theta(x_t, t) + eps ||**2.

This has the SAME minimizer (the true score) but is well-scaled at every t --
it is exactly the standard "predict the noise" parameterization. Set
``weighted=False`` to recover the literal CLAUDE.md loss.
"""

import torch
from tqdm import tqdm

from sampling.schedule import mu_t as vp_mu, sigma_t as vp_sigma

# Avoid sampling exactly t = 1 (the data end) where sigma_t = 0 and the score
# target -eps/sigma_t is singular.
T_EPS = 1e-3


# ============================================================================
# QUESTION 2.2 -- Denoising score matching loss
# ============================================================================
# Student version (currently commented out -- the working implementation
# follows below).
#
# Complete the body of ``dsm_loss``. You should:
#   1. Sample noise ``eps`` of the same shape as ``x0`` with ``torch.randn_like``.
#   2. Build the noised sample ``x_t = mu * x0 + sigma * eps``
#      (``mu``, ``sigma`` of shape ``(B, 1)`` already pre-computed).
#   3. Evaluate the score network at ``(x_t, t)``.
#   4. Return the squared-norm residual:
#        - if ``weighted=True`` (default, stable):  sigma * score + eps
#        - if ``weighted=False`` (literal DSM):      score + eps / sigma
#      averaged over the batch (sum over feature dim, mean over batch).
# ----------------------------------------------------------------------------
# def dsm_loss(model, x0, weighted=True, t_eps=T_EPS):
#     """Denoising score-matching loss on a batch of clean samples ``x0``. (Student.)"""
#     B = x0.shape[0]
#     device = x0.device
#
#     # t ~ U(0, 1 - t_eps): cover the whole schedule but stay off the data end
#     # t = 1 where sigma_t = 0.
#     t = torch.rand(B, device=device) * (1.0 - t_eps)
#     mu = vp_mu(t).unsqueeze(-1)        # (B, 1)
#     sigma = vp_sigma(t).unsqueeze(-1)  # (B, 1)
#
#     # ---------------------------------------------------------------- #
#     # >>> QUESTION 2.2: implement the DSM loss.                        #
#     #                                                                  #
#     #     eps   = ...                  # Gaussian noise, shape as x0   #
#     #     x_t   = ...                  # mu * x0 + sigma * eps         #
#     #     score = ...                  # model(x_t, t)                 #
#     #                                                                  #
#     #     if weighted:                                                  #
#     #         residual = sigma * score + eps                            #
#     #     else:                                                         #
#     #         residual = score + eps / sigma                            #
#     #                                                                  #
#     #     return (residual ** 2).sum(dim=-1).mean()                     #
#     # ---------------------------------------------------------------- #
#     raise NotImplementedError("Q2.2: implement the DSM loss.")
# ============================================================================


def dsm_loss(model, x0, weighted=True, t_eps=T_EPS):
    """Denoising score-matching loss on a batch of clean samples ``x0``.

    Parameters
    ----------
    model : nn.Module
        Score network s_theta(x, t).
    x0 : torch.Tensor
        Clean data batch, shape ``(B, dim)``.
    weighted : bool
        If True (default) use the sigma_t**2-weighted (stable) loss; otherwise
        the literal ``|| s_theta + eps/sigma_t ||**2``.
    """
    B = x0.shape[0]
    device = x0.device

    # t ~ U(0, 1 - t_eps): cover the whole schedule but stay off the data end
    # t = 1 where sigma_t = 0. Broadcast over feature dimensions below.
    t = torch.rand(B, device=device) * (1.0 - t_eps)
    mu = vp_mu(t).unsqueeze(-1)        # (B, 1)
    sigma = vp_sigma(t).unsqueeze(-1)  # (B, 1)

    eps = torch.randn_like(x0)
    x_t = mu * x0 + sigma * eps

    score = model(x_t, t)  # (B, dim)

    if weighted:
        # || sigma_t * s_theta + eps ||^2
        residual = sigma * score + eps
    else:
        # || s_theta + eps / sigma_t ||^2
        residual = score + eps / sigma

    return (residual**2).sum(dim=-1).mean()


def train_score_model(model, data, n_epochs=2000, batch_size=512, lr=1e-3,
                      weighted=True, device=None, log_every=200, seed=0):
    """Train a score network by DSM.

    Parameters
    ----------
    model : nn.Module
        Score network.
    data : array_like or torch.Tensor
        Clean training samples, shape ``(N, dim)`` (already standardized).
    n_epochs : int
        Number of optimization steps (each on a fresh random minibatch).
    batch_size, lr : training hyper-parameters.
    weighted : bool
        Passed through to :func:`dsm_loss`.

    Returns
    -------
    model : nn.Module
        The trained model (also modified in place).
    losses : list of float
        Loss value at each step.
    """
    if device is None:
        device = next(model.parameters()).device

    torch.manual_seed(seed)
    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    if data.dim() == 1:
        data = data.unsqueeze(-1)
    N = data.shape[0]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for step in tqdm(range(n_epochs)):
        idx = torch.randint(0, N, (batch_size,), device=device)
        x0 = data[idx]

        opt.zero_grad()
        loss = dsm_loss(model, x0, weighted=weighted)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        # if log_every and (step % log_every == 0 or step == n_epochs - 1):
        #     print(f"step {step:5d} | loss {loss.item():.4f}")

    model.eval()
    return model, losses
