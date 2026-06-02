
import torch
from tqdm import tqdm

from sampling.schedule import mu_schedule, sigma_schedule

# Avoid sampling exactly t = 1 (the data end) where sigma_t = 0 and the score
# target -eps/sigma_t is singular.
T_EPS = 1e-3


# ============================================================================
# QUESTION 2.2 -- Denoising score matching loss
# ============================================================================
# Student version (currently commented out -- the working implementation
# follows below).
#
# Complete the body of ``score_loss``. You should:
#   1. Sample noise ``eps`` of the same shape as ``x0`` with ``torch.randn_like``.
#   2. Build the noised sample ``x_t = mu * x0 + sigma * eps``
#      (``mu``, ``sigma`` of shape ``(B, 1)`` already pre-computed).
#   3. Evaluate the score network at ``(x_t, t)``.
#   4. Return the squared-norm residual ``sigma * score + eps`` (the
#      sigma**2-weighted DSM target), averaged over the batch (sum over the
#      feature dim, mean over batch).
# ----------------------------------------------------------------------------
# def score_loss(model, x0):
#     """Denoising score-matching loss on a batch of clean samples ``x0``. (Student.)"""
#     B = x0.shape[0]
#     device = x0.device
#
#     # t ~ U(0, 1 - T_EPS): cover the whole schedule but stay off the data end
#     # t = 1 where sigma_t = 0.
#     t = torch.rand(B, device=device) * (1.0 - T_EPS)
#     mu = mu_schedule(t).unsqueeze(-1)        # (B, 1)
#     sigma = sigma_t(t).unsqueeze(-1)  # (B, 1)
#
#     # ---------------------------------------------------------------- #
#     # >>> QUESTION 2.2: implement the DSM loss.                        #
#     #                                                                  #
#     #     eps      = ...                  # Gaussian noise, shape as x0#
#     #     x_t      = ...                  # mu * x0 + sigma * eps      #
#     #     score    = ...                  # model(x_t, t)              #
#     #     residual = sigma * score + eps  # sigma**2-weighted target   #
#     #                                                                  #
#     #     return (residual ** 2).sum(dim=-1).mean()                    #
#     # ---------------------------------------------------------------- #
#     raise NotImplementedError("Q2.2: implement the DSM loss.")
# ============================================================================


def score_loss(model, x0):
    """Sigma**2-weighted DSM loss on a batch of clean samples ``x0``.

    Parameters
    ----------
    model : nn.Module
        Score network s_theta(x, t).
    x0 : torch.Tensor
        Clean data batch, shape ``(B, dim)``.
    """
    B = x0.shape[0]
    device = x0.device

    # t ~ U(0, 1 - T_EPS): cover the whole schedule but stay off the data end
    # t = 1 where sigma_t = 0. Broadcast over feature dimensions below.
    t = torch.rand(B, device=device) * (1.0 - T_EPS)
    mu = mu_schedule(t).unsqueeze(-1)        # (B, 1)
    sigma = sigma_schedule(t).unsqueeze(-1)  # (B, 1)

    eps = torch.randn_like(x0)
    x_t = mu * x0 + sigma * eps

    score = model(x_t, t)  # (B, dim)

    residual = sigma * score + eps
    return (residual**2).sum(dim=-1).mean()


def train_score_model(model, data, n_epochs=2000, batch_size=512, lr=1e-3):
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

    Returns
    -------
    model : nn.Module
        The trained model (also modified in place).
    losses : list of float
        Loss value at each step.
    """
    device = next(model.parameters()).device

    torch.manual_seed(0)
    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    if data.dim() == 1:
        data = data.unsqueeze(-1)
    N = data.shape[0]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in tqdm(range(n_epochs)):
        idx = torch.randint(0, N, (batch_size,), device=device)
        x0 = data[idx]

        opt.zero_grad()
        loss = score_loss(model, x0)
        loss.backward()
        opt.step()

        losses.append(loss.item())

    model.eval()
    return model, losses
