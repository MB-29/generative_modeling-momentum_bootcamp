import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps

from sampling.schedule import sigma_schedule, time_from_sigma

rc('font', size=20)


def noise_schedule(times, mu, sigma):
    plt.plot(times, mu, label=r"$\mu_t$ (signal)")
    plt.plot(times, sigma, label=r"$\sigma_t$ (noise)")
    plt.xlabel(r"$t$   ($0$ = noise,  $1$ = data)")
    plt.legend()
    plt.show()


def loss_curve(losses, title="training loss", log=False):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("loss")
    if log:
        plt.yscale("log")
    plt.show()


def score_1d(x, learned, analytic):
    plt.plot(x, learned, color="red",
             label=r"learned $s_\theta(x, t\!\approx\!1)$")
    plt.plot(x, analytic, color="blue", linestyle="--",
             label=r"analytic $\nabla_x\log p(x)$")
    plt.axhline(0, color="gray", lw=0.5)
    plt.title(r"Learned $s_\theta$ vs. analytic $\nabla_x\log p$ (standardized)")
    plt.xlabel(r"$x$ (standardized)")
    plt.ylabel("score")
    plt.legend()
    plt.show()


def two_hists_vs_density(data, gen, grid, p_vals,
                         title=r"Diffusion model reproduces the bimodal density $p(x)$"):
    plt.hist(data, bins=80, density=True, alpha=0.5, color="blue",
             label="data ")
    plt.hist(gen, bins=80, density=True, alpha=0.5, color="red",
             label="diffusion samples")
    plt.plot(grid, p_vals, "k-", lw=2, label=r"analytic $p(x)$")
    plt.title(title)
    plt.xlabel(r"$x$")
    plt.legend()
    plt.show()


def overlay_histograms(a, b, bins=80):
    """Overlay two empirical 1D histograms on a shared binning.
    """
    raise NotImplementedError(
        "Question 3: histogram `a` and `b` on a shared binning and show them.")


def samples_2d(samples, title="data distribution (empirical)", bins=120,
               extent=4.0):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.hist2d(samples[:, 0], samples[:, 1], bins=bins, cmap="magma",
              range=[[-extent, extent], [-extent, extent]])
    ax.set(title=title, xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def noised_samples_across_sigma(data, sigma_levels=(0.1, 0.3, 0.6, 0.9),
                                bins=120, extent=3.5, seed=0):
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(sigma_levels),
                             figsize=(3.2 * len(sigma_levels), 3.4),
                             sharex=True, sharey=True)
    for ax, sigma in zip(axes, sigma_levels):
        sigma = float(sigma)
        mu = float(np.sqrt(max(1.0 - sigma ** 2, 0.0)))
        eps = rng.standard_normal(size=data.shape)
        xt = mu * data + sigma * eps
        ax.hist2d(xt[:, 0], xt[:, 1], bins=bins, cmap="magma",
                  range=[[-extent, extent], [-extent, extent]])
        ax.set_title(rf"$\sigma = {sigma:.2f}$")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Private helpers for the score-field panels
# --------------------------------------------------------------------------- #
def _learned_field(net, Z, t):
    """Learned score s_theta(Z, t) on grid points Z of shape (M, 2)."""
    tb = torch.full((Z.shape[0],), float(t))
    with torch.no_grad():
        return net(torch.from_numpy(Z.astype(np.float32)), tb).numpy()


def _draw_field(ax, X, Y, S, spacing, length=0.42, vmax=None):
    """Quiver of S=(M, 2) as unit-direction arrows colored by |S|."""
    mag = np.linalg.norm(S, axis=1)
    d = S / (mag[:, None] + 1e-8) * (length * spacing)
    q = ax.quiver(X, Y, d[:, 0], d[:, 1], mag, cmap="inferno",
                  angles="xy", scale_units="xy", scale=.5, width=0.006, pivot="mid")
    if vmax is not None:
        q.set_clim(0.0, vmax)
    return q


def _std_grids(extent, n_arrows, n_dens):
    """Coarse arrow grid + fine density grid over [-extent, extent]^2."""
    zc = np.linspace(-extent, extent, n_arrows)
    h = zc[1] - zc[0]
    ZX, ZY = np.meshgrid(zc, zc)
    Zc = np.stack([ZX.ravel(), ZY.ravel()], -1)
    zf = np.linspace(-extent, extent, n_dens)
    FX, FY = np.meshgrid(zf, zf)
    Zf = np.stack([FX.ravel(), FY.ravel()], -1)
    return (ZX, ZY, Zc, h), (FX, FY, Zf)


def score_fields(net, score_function, density_fn, mean, std, tau=1.0,
                 extent=2.4, n_arrows=30, n_dens=100):
    """Analytic vs. learned score field at the data end (t≈1), over the density.
    """
    (ZX, ZY, Zc, h), (FX, FY, Zf) = _std_grids(extent, n_arrows, n_dens)
    dens_std = density_fn(mean + std * Zf, tau=tau).reshape(FX.shape)

    analytic_S = std * score_function(mean + std * Zc, tau=tau)
    learned_S = _learned_field(net, Zc, 1.0 - 1e-3)
    vmax = np.percentile(np.linalg.norm(analytic_S, axis=1), 95)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    panels = [(axes[0], analytic_S, r"analytic  $\nabla\log p$"),
              (axes[1], learned_S, r"learned  $s_\theta(\cdot,\,t\approx1)$")]
    for ax, S, ttl in panels:
        ax.contourf(FX, FY, dens_std, levels=25, cmap="Blues")
        q = _draw_field(ax, ZX, ZY, S, h, vmax=vmax)
        ax.set_title(ttl)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    fig.colorbar(q, ax=axes, shrink=0.8, label="score magnitude")
    plt.show()


def field_across_noise(net, data_std, sigma_levels=(0.2, 0.4, 0.6, 0.85),
                       t_levels=None, extent=2.4, n_arrows=30, n_dens=100,
                       n_sub=1500, seed=0):
    if t_levels is not None:
        sigma_levels = [float(sigma_schedule(t)) for t in t_levels]

    (ZX, ZY, Zc, h), (FX, FY, Zf) = _std_grids(extent, n_arrows, n_dens)

    def noised_density(grid_pts, data, sigma):
        # mu = sqrt(1-sigma^2)
        m = float(np.sqrt(max(1.0 - sigma ** 2, 1e-20)))
        diff = grid_pts[:, None, :] - m * data[None, :, :]      # (G, N, 2)
        sq = (diff ** 2).sum(-1)
        return (np.exp(-sq / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)).mean(1)

    idx = np.random.default_rng(seed).choice(
        len(data_std), n_sub, replace=False)
    sub = data_std[idx]

    fig, axes = plt.subplots(1, len(sigma_levels),
                             figsize=(4 * len(sigma_levels), 6.),
                             sharex=True, sharey=True)
    for ax, sigma in zip(axes, sigma_levels):
        # query the network at t(sigma)
        t = float(time_from_sigma(sigma))
        pt = noised_density(Zf, sub, sigma).reshape(FX.shape)
        ax.contourf(FX, FY, pt, levels=25, cmap="Blues")
        _draw_field(ax, ZX, ZY, _learned_field(net, Zc, t), h)
        ax.set_title(rf"$\sigma = {sigma:.2f}$")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    fig.suptitle(r"Noise-dependent $s_\theta(x,\sigma)$ ")
    plt.tight_layout()
    # plt.show()


def samples_comparison_2d(density_grid, data, gen,
                          titles=("data (ULA)", "diffusion samples")):
    """Two scatter panels (data vs. generated) over shared density contours.

    Color convention: data → blue, generated samples → red.
    """
    GX, GY, dens = density_grid
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, pts, ttl, c in [(a1, data, titles[0], "blue"),
                            (a2, gen, titles[1], "red")]:
        ax.contour(GX, GY, dens, levels=8, colors="k",
                   linewidths=0.6, alpha=0.5)
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c=c, alpha=0.25)
        ax.set_title(ttl)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$")
    a1.set_ylabel(r"$x_2$")
    plt.tight_layout()
    plt.show()


def denoiser_panels_2d(clean, noisy, denoised, density_grid=None, extent=4.0):
    clean = np.asarray(clean)
    noisy = np.asarray(noisy)
    denoised = np.asarray(denoised)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
    panels = [(axes[0], clean, "data", "blue"),
              (axes[1], noisy, "noisy", "gray"),
              (axes[2], denoised, "denoised", "red")]
    for ax, pts, ttl, c in panels:
        if density_grid is not None:
            GX, GY, dens = density_grid
            ax.contour(GX, GY, dens, levels=6, colors="k",
                       linewidths=0.5, alpha=0.4)
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c=c, alpha=0.3)
        ax.set_title(ttl)
        ax.set_aspect("equal")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
    plt.tight_layout()
    plt.show()


def mode_weight_bars(labels, heavy_frac, target,
                     title="Langevin mixing weights depend on initialization; "
                           "diffusion does not",
                     mode_names=("heavy mode", "light mode")):
    heavy = np.asarray(heavy_frac, dtype=float)
    light = 1.0 - heavy
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, heavy, width=0.6, color="red", label=mode_names[0])
    ax.bar(x, light, width=0.6, bottom=heavy, color="blue",
           label=mode_names[1])
    ax.axhline(target, color="k", ls="--", lw=1, label="target split")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of samples in each mode")
    ax.set_title(title)
    ax.legend(loc="center right", fontsize=8)
    plt.tight_layout()
    plt.show()
