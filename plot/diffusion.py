"""Section 2 — diffusion model plots (shared + 1D + 2D)."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps

from sde import sigma_t, time_from_sigma

rc('font', size=20)


def noise_schedule(times, mu, sigma):
    """The VP-SDE marginal coefficients mu_t (signal) and sigma_t (noise).

    In generative time the signal grows toward the data end (t = 1) while the
    noise vanishes there, so generation flows left-to-right into the data.
    """
    plt.plot(times, mu, label=r"$\mu_t$ (signal)")
    plt.plot(times, sigma, label=r"$\sigma_t$ (noise)")
    plt.title(r"VP-SDE marginal coefficients $\mu_t,\,\sigma_t$ (generative time)")
    plt.xlabel(r"$t$   ($0$ = noise,  $1$ = data)")
    plt.legend()
    plt.show()


def loss_curve(losses, title="training loss (DSM)", log=False):
    """Training-loss curve; set ``log=True`` for a log-scaled y-axis."""
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("loss")
    if log:
        plt.yscale("log")
    plt.show()


def score_1d(x, learned, analytic):
    """Learned vs. analytic 1D score (standardized coordinates)."""
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
    """Data histogram + generated histogram overlaid on an analytic density."""
    plt.hist(data, bins=80, density=True, alpha=0.5, color="blue",
             label="data (ULA)")
    plt.hist(gen, bins=80, density=True, alpha=0.5, color="red",
             label="diffusion samples")
    plt.plot(grid, p_vals, "k-", lw=2, label=r"analytic $p(x)$")
    plt.title(title)
    plt.xlabel(r"$x$")
    plt.legend()
    plt.show()


def two_hists(a, b, bins=80, labels=("data", "Langevin"),
              colors=("blue", "red"),
              title="Empirical histograms: data vs. Langevin",
              xlabel="projection onto the mode-connecting axis"):
    """Overlay two empirical 1D histograms on a shared binning (no analytic curve).

    Used to *see* a sampler's failure mode: if the two histograms put different
    mass on the two peaks, the sampler is misweighting the modes.
    """
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    plt.hist(a, bins=edges, density=True, alpha=0.5,
             color=colors[0], label=labels[0])
    plt.hist(b, bins=edges, density=True, alpha=0.5,
             color=colors[1], label=labels[1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.show()


def samples_2d(samples, title="data distribution (empirical)", bins=120,
               extent=4.0):
    """2D histogram (heatmap) of empirical samples -- 'the data distribution'."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.hist2d(samples[:, 0], samples[:, 1], bins=bins, cmap="magma",
              range=[[-extent, extent], [-extent, extent]])
    ax.set(title=title, xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# 2D radial double well
# --------------------------------------------------------------------------- #
def ring_density_and_marginal(density_grid, samples, radial_marginal_fn,
                              tau=1.0, r_max=3.5):
    """Left: 2D density contour + samples. Right: radial histogram vs p(r).

    Parameters
    ----------
    density_grid : (GX, GY, dens)
        Meshgrid coordinates and the density evaluated on them.
    samples : (N, 2) array
        Points in the plane (physical coordinates).
    radial_marginal_fn : callable
        ``radial_marginal_fn(r, tau=tau) -> p_R(r)`` (e.g. ``dw2.radial_marginal``).
    """
    GX, GY, dens = density_grid
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

    a1.contourf(GX, GY, dens, levels=30, cmap="Blues")
    a1.scatter(samples[:, 0], samples[:, 1], s=2, c="k", alpha=0.10)
    a1.set_title(r"density $p(x_1,x_2)$ + Langevin samples (two rings)")
    a1.set_xlabel(r"$x_1$")
    a1.set_ylabel(r"$x_2$")
    a1.set_aspect("equal")

    r = np.linalg.norm(samples, axis=1)
    rgrid = np.linspace(0, r_max, 300)
    a2.hist(r, bins=80, density=True, alpha=0.5, color="blue", label="samples")
    a2.plot(rgrid, radial_marginal_fn(rgrid, tau=tau),
            "k-", lw=2, label=r"$p(r)$")
    a2.set_title(r"radial marginal $p(r)$ (bimodal)")
    a2.set_xlabel(r"$r$")
    a2.legend()
    plt.tight_layout()
    plt.show()


def density_and_samples_2d(density_grid, samples=None, means=None,
                           title="density + samples"):
    """Filled density contour, optionally with samples scattered on top.

    A generic 2D overview (used for the Gaussian mixture). Pass ``samples=None``
    to show the density alone. ``means`` (optional, shape ``(K, 2)``) marks the
    component centers.
    """
    GX, GY, dens = density_grid
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.contourf(GX, GY, dens, levels=30, cmap="Blues")
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], s=2, c="k", alpha=0.08)
    if means is not None:
        means = np.asarray(means)
        ax.scatter(means[:, 0], means[:, 1], c="red", marker="x", s=80,
                   label="component means")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
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


def score_fields(net, score_fn, density_fn, mean, std, tau=1.0,
                 extent=2.4, n_arrows=30, n_dens=100):
    """Analytic vs. learned score field at the data end (t≈1), over the density.

    Grids are built in *standardized* coordinates; ``mean``/``std`` map them
    back to physical space to evaluate the analytic ``score_fn`` and
    ``density_fn``. The analytic score in standardized coordinates is
    ``std * score_fn(mean + std * z)`` (diagonal chain rule).
    """
    (ZX, ZY, Zc, h), (FX, FY, Zf) = _std_grids(extent, n_arrows, n_dens)
    dens_std = density_fn(mean + std * Zf, tau=tau).reshape(FX.shape)

    analytic_S = std * score_fn(mean + std * Zc, tau=tau)
    # data end of generative time
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
    fig.suptitle(r"Score field at the data end ($t\approx1$): "
                 "the network recovers the data score")
    plt.show()


def field_across_noise(net, data_std, sigma_levels=(0.2, 0.4, 0.6, 0.85),
                       t_levels=None, extent=2.4, n_arrows=30, n_dens=100,
                       n_sub=1500, seed=0):
    """Learned score field over the noised density at several noise levels sigma.

    We index the panels by the *added-noise level* sigma rather than by time: at
    noise level sigma the data is blurred with a Gaussian of std sigma (and
    shrunk by mu = sqrt(1 - sigma**2) under the VP schedule), giving the
    closed-form noised density (a mixture of Gaussians, one per sub-sampled
    standardized data point). The network is queried at the matching time
    t(sigma). As sigma grows the two blobs contract and **merge into one**.

    ``t_levels`` is accepted for backward compatibility; if given, the panels use
    the noise levels ``sigma_t(t)`` of those times instead of ``sigma_levels``.
    """
    if t_levels is not None:
        sigma_levels = [float(sigma_t(t)) for t in t_levels]

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
    fig.suptitle(r"Noise-dependent $s_\theta(x,\sigma)$ over the noised density"
                 r"  —  the two blobs merge as $\sigma$ grows")
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


def mode_weight_bars(labels, heavy_frac, target,
                     title="Langevin mixing weights depend on initialization; "
                           "diffusion does not",
                     mode_names=("heavy mode", "light mode")):
    """Stacked heavy/light bars per method, with a dashed target-split line.

    ``heavy_frac`` is the fraction of each method's samples in the heavy mode
    (one number per ``labels`` entry); the light fraction is the complement.
    """
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
