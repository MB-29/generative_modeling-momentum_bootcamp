import numpy as np
import matplotlib.pyplot as plt


def posterior_ensemble(lorenz_data, posterior_samples, x_obs, truth=None):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(lorenz_data[:, 0], lorenz_data[:, 1], lorenz_data[:, 2],
               s=0.3, color="lightgray", alpha=0.3)
    ax.scatter(posterior_samples[:, 0], posterior_samples[:, 1],
               posterior_samples[:, 2],
               s=6, color="blue", alpha=0.6, label="posterior samples")
    if truth is not None:
        truth = np.asarray(truth).reshape(3)
        ax.scatter([truth[0]], [truth[1]], [truth[2]],
                   s=80, color="black", marker="*", depthshade=False,
                   label="true state")

    ax.set(title=rf"Posterior ensemble  (observed $x_3={x_obs:.0f}$)",
           xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$x_3$")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
