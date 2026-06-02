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
