"""Section 3 — Lorenz '63 unconditional diffusion model."""

import matplotlib.pyplot as plt


def comparison_3d(data, gen, titles=("Lorenz data", "Diffusion samples")):
    """Two 3D scatter panels (data vs. generated).

    Color convention: data → blue, diffusion samples → red.
    """
    fig = plt.figure(figsize=(12, 5))
    for k, (pts, title, color) in enumerate(
            [(data, titles[0], "blue"), (gen, titles[1], "red")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.4,
                   color=color, alpha=0.4)
        ax.set(title=title, xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$x_3$")
    plt.tight_layout()
    plt.show()
