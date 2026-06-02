import os
import numpy as np
import matplotlib.pyplot as plt


def comparison_3d(data, gen, titles=("Lorenz data", "Diffusion samples")):
    fig = plt.figure(figsize=(12, 5))
    for k, (pts, title, color) in enumerate(
            [(data, titles[0], "blue"), (gen, titles[1], "red")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.4,
                   color=color, alpha=0.4)
        ax.set(title=title, xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$x_3$")
    plt.tight_layout()
    plt.show()


def denoiser_panels_3d(clean, noisy, denoised):
    clean = np.asarray(clean)
    noisy = np.asarray(noisy)
    denoised = np.asarray(denoised)

    pooled = np.concatenate([clean, noisy, denoised], axis=0)
    lo, hi = pooled.min(0), pooled.max(0)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(12, 5))
    panels = [(clean, "data", "black"),
              (noisy, "noisy", "blue"),
              (denoised, "denoised", "red")]
    for k, (pts, ttl, c) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.4, color=c, alpha=0.4)
        ax.set_title(ttl)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    plt.tight_layout()
    # plt.show()
