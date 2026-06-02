"""Cyclic data assimilation diagnostics for the Lorenz '63 experiment.

Used by ``scripts/lorenz_cyclic_da.py`` and reusable from the notebook. Every
function takes the ``res`` dict produced by ``run_cyclic_da`` and renders one
figure.

Observation model used by the cyclic-DA experiment: only the second and third
coordinates (``x_2``, ``x_3``) are observed; ``x_1`` is hidden. The panel that
plots ``x_1`` therefore has no observation markers -- the goal of those plots
is to show how the diffusion prior plus the partial observation localize the
hidden component.

Color convention: when two ensembles (background vs. analysis) are compared,
background is drawn in blue and analysis in red.
"""

import matplotlib.pyplot as plt

# Indices of the observed coordinates in the (x_1, x_2, x_3) state vector.
OBS_IDX = (1, 2)
COORD_LABELS = (r"$x_1$ (unobserved)",
                r"$x_2$ (observed)", r"$x_3$ (observed)")


def time_series(res):
    """Per-coordinate truth vs. analysis-ensemble mean ±1σ vs. observations.

    Observation markers are placed on the panels for the observed components
    only (``x_2`` and ``x_3``); the ``x_1`` panel shows only truth and
    posterior.
    """
    t = res["times"]
    truth = res["truth"]
    mean_a = res["analysis"].mean(1)
    std_a = res["analysis"].std(1)
    mean_b = res["background"].mean(1)
    y_obs = res["y_obs"]  # (n_cycles, m=2)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, truth[:, i], "k-", lw=1.8, label="truth")
        ax.plot(t, mean_b[:, i], color="blue", linestyle="--", lw=1.0, alpha=0.7,
                label="background mean")
        ax.plot(t, mean_a[:, i], color="red", lw=1.4, label="analysis mean")
        ax.fill_between(t, mean_a[:, i] - std_a[:, i],
                        mean_a[:, i] + std_a[:, i],
                        color="red", alpha=0.2, label=r"analysis $\pm 1\sigma$")
        if i in OBS_IDX:
            j = OBS_IDX.index(i)
            ax.scatter(t, y_obs[:, j], color="gold", edgecolor="k", s=30,
                       zorder=5, label="observations")
        ax.set_ylabel(COORD_LABELS[i])
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(r"$t$")
    fig.suptitle(
        r"Cyclic DA — Lorenz '63  ($x_2,\,x_3$ observed; $x_1$ hidden)")
    plt.tight_layout()
    plt.show()


def rmse(res):
    """Per-cycle RMSE of background / analysis mean state + analysis spread."""
    fig, ax = plt.subplots(figsize=(8, 4))
    t = res["times"]
    ax.plot(t, res["rmse_b"], color="blue", linestyle="--", marker="o", ms=4,
            label="background RMSE")
    ax.plot(t, res["rmse_a"], color="red", marker="o", ms=4,
            label="analysis RMSE")
    ax.plot(t, res["spread_a"], "k:", marker=".", label="analysis spread")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("RMSE / spread")
    ax.set_title(r"Cyclic DA diagnostics (RMSE vs. spread)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_rmse(res_diffusion, res_3dvar, *, title=None):
    """Overlay analysis-RMSE curves of the diffusion DA and the 3D-Var baseline.

    Each argument is the ``res`` dict produced by ``run_cyclic_da`` (diffusion)
    or ``var3d.run_cyclic_3dvar`` (Gaussian baseline). Only the analysis RMSE
    is overlaid; the ensemble spread of the diffusion run is added as a
    reference band so the reader sees how each method weighs accuracy against
    uncertainty.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    t = res_diffusion["times"]
    ax.plot(t, res_diffusion["rmse_a"], color="red", marker="o", ms=4,
            label="diffusion analysis RMSE")
    ax.plot(t, res_diffusion["spread_a"], color="red", linestyle=":",
            marker=".", alpha=0.7, label="diffusion analysis spread")
    ax.plot(res_3dvar["times"], res_3dvar["rmse_a"], color="C0", marker="s",
            ms=4, label="3D-Var analysis RMSE")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("RMSE / spread")
    ax.set_title(
        title or r"Cyclic DA: diffusion prior vs. 3D-Var (Gaussian prior)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def attractor(res, lorenz_cloud):
    """Faint attractor + truth path + analysis ensemble clouds at each cycle."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(lorenz_cloud[::20, 0], lorenz_cloud[::20, 1],
               lorenz_cloud[::20, 2],
               s=0.3, color="lightgray", alpha=0.3)
    truth = res["truth"]
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2],
            "k-", lw=1.5, label="truth path")
    ax.scatter(truth[:, 0], truth[:, 1], truth[:, 2],
               c="k", s=18, label="truth states")
    A = res["analysis"]                                # (K, N, 3)
    A_flat = A.reshape(-1, 3)
    ax.scatter(A_flat[:, 0], A_flat[:, 1], A_flat[:, 2],
               s=2, color="red", alpha=0.3, label="analysis ensemble")
    ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$x_3$",
           title=r"Cyclic DA — truth path \& analysis ensembles")
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_time_series(res_diffusion, res_3dvar, *, title=None):
    """Per-component overlay of the two methods' reconstructed trajectories.

    Three stacked panels (one per coordinate). On each panel we plot
        - the truth                            (black line),
        - the 3D-Var analysis                  (blue line + markers, deterministic),
        - the diffusion analysis ensemble mean (red line + markers)
          with a ±1σ shaded band,
        - the observations                     (gold dots, observed components only).
    Both res dicts must share ``times``, ``truth`` and ``y_obs`` -- that is the
    whole point of the comparison setup (``build_truth_and_obs``).
    """
    t = res_diffusion["times"]
    truth = res_diffusion["truth"]
    mean_d = res_diffusion["analysis"].mean(1)
    std_d = res_diffusion["analysis"].std(1)
    mean_v = res_3dvar["analysis"].mean(1)            # ensemble size 1 here
    y_obs = res_diffusion["y_obs"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, truth[:, i], "k-", lw=1.8, label="truth")
        ax.plot(t, mean_v[:, i], color="C0", marker="s", ms=4, lw=1.2,
                label="3D-Var analysis")
        ax.plot(t, mean_d[:, i], color="red", marker="o", ms=4, lw=1.2,
                label="diffusion analysis mean")
        ax.fill_between(t, mean_d[:, i] - std_d[:, i],
                        mean_d[:, i] + std_d[:, i],
                        color="red", alpha=0.2,
                        label=r"diffusion $\pm 1\sigma$")
        if i in OBS_IDX:
            j = list(OBS_IDX).index(i)
            ax.scatter(t, y_obs[:, j], color="gold", edgecolor="k", s=30,
                       zorder=5, label="observations")
        ax.set_ylabel(COORD_LABELS[i])
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(r"$t$")
    fig.suptitle(title or
                 r"Reconstructed trajectories — diffusion vs. 3D-Var")
    plt.tight_layout()
    plt.show()


def compare_attractor(res_diffusion, res_3dvar, lorenz_cloud=None, *,
                      show_ensemble=True, title=None):
    """3D overlay of the two reconstructed trajectories and the truth path.

    - Optional faint Lorenz climatology cloud as a background reference.
    - Truth path (black line + markers).
    - 3D-Var analysis path (blue line + markers).
    - Diffusion analysis ensemble mean path (red line + markers); when
      ``show_ensemble`` the full ensemble at every cycle is also drawn as a
      faint red point cloud so the reader sees where the diffusion posterior
      spreads (e.g. across both butterfly wings on the hidden ``x_1``).
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if lorenz_cloud is not None:
        ax.scatter(lorenz_cloud[::20, 0], lorenz_cloud[::20, 1],
                   lorenz_cloud[::20, 2],
                   s=0.3, color="lightgray", alpha=0.3)

    truth = res_diffusion["truth"]
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2],
            "k-", lw=1.6, label="truth path")
    ax.scatter(truth[:, 0], truth[:, 1], truth[:, 2],
               c="k", s=18, label="truth states")

    mean_v = res_3dvar["analysis"].mean(1)
    ax.plot(mean_v[:, 0], mean_v[:, 1], mean_v[:, 2],
            color="C0", lw=1.4, label="3D-Var analysis path")
    ax.scatter(mean_v[:, 0], mean_v[:, 1], mean_v[:, 2],
               color="C0", marker="s", s=20)

    mean_d = res_diffusion["analysis"].mean(1)
    ax.plot(mean_d[:, 0], mean_d[:, 1], mean_d[:, 2],
            color="red", lw=1.4, label="diffusion analysis mean path")
    ax.scatter(mean_d[:, 0], mean_d[:, 1], mean_d[:, 2],
               color="red", marker="o", s=20)

    if show_ensemble:
        A = res_diffusion["analysis"]                  # (K, N, 3)
        A_flat = A.reshape(-1, 3)
        ax.scatter(A_flat[:, 0], A_flat[:, 1], A_flat[:, 2],
                   s=2, color="red", alpha=0.25,
                   label="diffusion analysis ensemble")

    ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$x_3$",
           title=title or
           r"Cyclic DA — reconstructed trajectories vs. truth")
    ax.legend()
    plt.tight_layout()
    plt.show()
