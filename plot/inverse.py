"""Section 4 — single-time inverse problem on the Lorenz attractor."""

import matplotlib.pyplot as plt


def posterior_ensemble(lorenz_data, cond, wing, x_obs):
    """Posterior ensemble over the faint attractor + x-marginal histogram.

    Only the z-coordinate is observed; ``wing`` is a boolean mask splitting
    the conditional samples into the two lobes by the sign of x (the natural
    wing discriminator for the Lorenz butterfly). ``x_obs`` is the observed
    z-value, despite the legacy parameter name.

    Color convention: the two posterior modes (wings A/B) are drawn in blue
    and red.
    """
    fig = plt.figure(figsize=(13, 5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(lorenz_data[:, 0], lorenz_data[:, 1], lorenz_data[:, 2],
               s=0.3, color="lightgray", alpha=0.3)
    ax.scatter(cond[wing, 0], cond[wing, 1], cond[wing, 2], s=6, color="red",
               label=r"wing A ($x_1>0$)")
    ax.scatter(cond[~wing, 0], cond[~wing, 1], cond[~wing, 2], s=6,
               color="blue", label=r"wing B ($x_1<0$)")
    ax.set(title=rf"Posterior ensemble  (observed $x_3={x_obs:.0f}$)",
           xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$x_3$")
    ax.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    # Split the histogram by sign of x_1 so the two posterior modes inherit
    # the blue/red wing colors used in the 3D scatter on the left.
    ax2.hist(cond[wing, 0], bins=30, color="red", alpha=0.7,
             label=r"wing A ($x_1>0$)")
    ax2.hist(cond[~wing, 0], bins=30, color="blue", alpha=0.7,
             label=r"wing B ($x_1<0$)")
    ax2.set(title=r"Posterior of unobserved $x_1$ (bimodal)",
            xlabel=r"$x_1$", ylabel="count")
    ax2.legend()
    plt.tight_layout()
    plt.show()
