"""Section 4 — single-time inverse problem on the Lorenz attractor."""

import matplotlib.pyplot as plt


def posterior_ensemble(lorenz_data, cond, wing, x_obs):
    """Posterior ensemble over the faint attractor + x-marginal histogram.

    Only the z-coordinate is observed; ``wing`` is a boolean mask splitting
    the conditional samples into the two lobes by the sign of x (the natural
    wing discriminator for the Lorenz butterfly). ``x_obs`` is the observed
    z-value, despite the legacy parameter name.

    The posterior ensemble is drawn in a single color (blue) so the geometric
    split into both wings comes through visually without color cueing.
    """
    fig = plt.figure(figsize=(13, 5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(lorenz_data[:, 0], lorenz_data[:, 1], lorenz_data[:, 2],
               s=0.3, color="lightgray", alpha=0.3)
    ax.scatter(cond[:, 0], cond[:, 1], cond[:, 2], s=6, color="blue",
               label="posterior samples")
    ax.set(title=rf"Posterior ensemble  (observed $x_3={x_obs:.0f}$)",
           xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$x_3$")
    ax.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(cond[:, 0], bins=30, color="blue", alpha=0.8,
             label="posterior samples")
    ax2.set(title=r"Posterior of unobserved $x_1$ (bimodal)",
            xlabel=r"$x_1$", ylabel="count")
    ax2.legend()
    plt.tight_layout()
    plt.show()
