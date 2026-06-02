
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# --------------------------------------------------------------------------- #
# 1D — the original double well plots
# --------------------------------------------------------------------------- #
def potential_and_density(grid, potential_values, density_values):
    """Side-by-side 1D potential f(x) and its Gibbs density p(x)."""
    fig, (ax_potential, ax_density) = plt.subplots(1, 2, figsize=(11, 4))
    ax_potential.plot(grid, potential_values, color="darkred")
    ax_potential.set(title=r"Potential $f(x)$",
                     xlabel=r"$x$", ylabel=r"$f(x)$")
    ax_potential.axhline(0, color="gray", lw=0.5)
    ax_density.plot(grid, density_values, color="navy")
    ax_density.set(title=r"Gibbs density $p(x)\propto e^{-f(x)}$",
                   xlabel=r"$x$", ylabel=r"$p(x)$")
    plt.tight_layout()
    plt.show()


def hist_vs_density(samples, grid, p_vals, sample_label="ULA samples",
                    title=r"Langevin samples vs. analytic Gibbs density $p(x)$"):
    """Histogram of 1D samples overlaid on an analytic density curve."""
    plt.hist(samples, bins=80, density=True, alpha=0.6, color="blue",
             label=sample_label)
    plt.plot(grid, p_vals, "k-", lw=2, label=r"analytic $p(x)$")
    plt.title(title)
    plt.xlabel(r"$x$")
    plt.legend()
    plt.show()


# --------------------------------------------------------------------------- #
# 2D — potential / density alone
# --------------------------------------------------------------------------- #
def potential_and_density_2d(grid_x, grid_y, potential_grid, density_grid):
    fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(16, 7))

    cf = ax_f.contourf(grid_x, grid_y, potential_grid,
                       levels=30, cmap="coolwarm")
    ax_f.set(title=r"Potential $f(x)$",
             xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax_f.set_aspect("equal")
    fig.colorbar(cf, ax=ax_f, shrink=0.8)

    cp = ax_p.contourf(grid_x, grid_y, density_grid, levels=30, cmap="Blues")
    ax_p.set(title=r"Target density",
             xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax_p.set_aspect("equal")
    fig.colorbar(cp, ax=ax_p, shrink=0.8)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Shared 2D-density + 1D-marginal layout
# --------------------------------------------------------------------------- #
def density_grid_2d(density_fn, xlim, ylim, n=240, **density_kwargs):
    """Evaluate a 2D ``density_fn`` on a regular ``n x n`` grid over the box.

    Parameters
    ----------
    density_fn : callable
        Vectorized density, ``density_fn(x, **density_kwargs)``, accepting an
        array of shape ``(..., 2)`` and returning the matching ``(...)`` shape
        (the convention of :mod:`data_distributions.double_well_gaussian_2d`).
    xlim, ylim : 2-tuples of floats
        Plot extent.
    n : int
        Grid resolution per axis.
    **density_kwargs
        Forwarded to ``density_fn`` (e.g. ``tau=1.0``).

    Returns
    -------
    GX, GY, Z : arrays of shape ``(n, n)``
        Meshgrid coordinates and the density evaluated on them, ready for
        :func:`matplotlib.axes.Axes.contourf`.
    """
    gx = np.linspace(xlim[0], xlim[1], n)
    gy = np.linspace(ylim[0], ylim[1], n)
    GX, GY = np.meshgrid(gx, gy)
    Z = density_fn(np.stack([GX, GY], -1), **density_kwargs)
    return GX, GY, Z


def build_density_marginal_figure(GX, GY, Z, grid_x, p_x, *, xlim, ylim,
                                  hist_ylim=None,
                                  title_2d="distribution",
                                  title_marginal=r"$x_1$-marginal",
                                  figsize=(6.6, 7.4),
                                  fontsize=None,
                                  marginal_size="55%",
                                  marginal_pad=0.4):
    """Two stacked panels with the static layers drawn once.

    Top panel: filled contour of the 2D density. Bottom panel: the analytic
    x-marginal curve. The caller (animation frame loop or one-shot snapshot)
    adds the dynamic artists on top — a scatter cloud, a single particle dot
    and trail, or a histogram of the ensemble's x-coordinate.

    The bottom panel is appended *to the top panel* via
    :func:`make_axes_locatable`, so the two share both the data x-range and the
    physical x-axis position — even though the top panel uses
    ``set_aspect('equal')`` and shrinks below the figure width.

    Parameters
    ----------
    figsize : tuple
        Figure size in inches.
    fontsize : int or None
        If given, applied uniformly to titles, axis labels, tick labels (at
        ``fontsize - 2``), and the legend.
    marginal_size, marginal_pad : str / float
        Height of the bottom panel relative to the top, and the vertical gap
        between them, forwarded to :meth:`AxesDivider.append_axes`.

    Returns
    -------
    fig, ax2d, axm : the figure and the two axes.
    """
    if hist_ylim is None:
        hist_ylim = (0.0, 1.25 * float(np.max(p_x)))

    fig, ax2d = plt.subplots(figsize=figsize)

    ax2d.contourf(GX, GY, Z, levels=25, cmap="Blues", zorder=0)
    ax2d.set(xlim=xlim, ylim=ylim, ylabel=r"$x_2$",
             title='particles',
             yticks=[-2, 0, 2])
    ax2d.set_aspect("equal")
    # x ticks live on the bottom (marginal) panel only.
    ax2d.tick_params(labelbottom=False)

    divider = make_axes_locatable(ax2d)
    axm = divider.append_axes("bottom", size=marginal_size, pad=marginal_pad,
                              sharex=ax2d)

    axm.plot(grid_x, p_x, color="navy", lw=2.0, label=r"target",
             zorder=3)
    axm.fill_between(grid_x, 0, p_x, color="navy", alpha=0.08, zorder=2)
    axm.set(ylim=hist_ylim, xlabel=r"$x_1$", yticks=[],
            title=title_marginal)
    # axm.legend(loc="upper right", framealpha=0.9)

    if fontsize is not None:
        for ax in (ax2d, axm):
            ax.title.set_fontsize(fontsize)
            ax.xaxis.label.set_fontsize(fontsize)
            ax.yaxis.label.set_fontsize(fontsize)
            ax.tick_params(labelsize=fontsize - 2)
        leg = axm.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(fontsize - 2)

    return fig, ax2d, axm


def density_marginal_snapshot(density_grid, grid_x, p_x, samples, *,
                              xlim, ylim, n_bins=70, scatter_max=2500,
                              title_2d="Generated samples and  target density ",
                              title_marginal=(r"Histogram of $x_1$"
                                              r" and target marginal density"),
                              seed=0):
    GX, GY, Z = density_grid
    fig, ax2d, axm = build_density_marginal_figure(
        GX, GY, Z, grid_x, p_x, xlim=xlim, ylim=ylim,
        title_2d=title_2d, title_marginal=title_marginal)

    # Subsample the scatter so the cloud does not oversaturate the contour.
    if len(samples) > scatter_max:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(samples), size=scatter_max, replace=False)
        scatter_pts = samples[idx]
    else:
        scatter_pts = samples
    ax2d.scatter(scatter_pts[:, 0], scatter_pts[:, 1], s=4, c="crimson",
                 alpha=0.35, zorder=2)

    # The full ensemble drives the histogram, regardless of the scatter cap.
    bin_edges = np.linspace(xlim[0], xlim[1], n_bins + 1)
    axm.hist(samples[:, 0], bins=bin_edges, density=True,
             color="crimson", alpha=0.55, edgecolor="white",
             linewidth=0.3, zorder=1, label="ensemble")
    # axm.legend(loc="upper right", framealpha=0.9)
    plt.show()
