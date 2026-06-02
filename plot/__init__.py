"""Plotting helpers for the bootcamp.

Split per notebook section so the modules stay small and easy to skim. The
package re-exports every public helper at the top level, so existing
``import plot; plot.posterior_ensemble(...)`` calls keep working unchanged.

Submodules:

    plot.langevin   : Section 1 (Langevin dynamics)
    plot.diffusion  : Section 2 (1D / 2D diffusion)
    plot.lorenz     : Section 3 (Lorenz '63 unconditional)
    plot.inverse    : Section 4 (single-time inverse problem)
    plot.cyclic_da  : Cyclic data assimilation experiment

Call :func:`use_latex` once (notebook or script) to render text and math with a
real LaTeX engine — same typesetting in the notebook and in the animation
frames.
"""

from matplotlib import rc

from .langevin import (
    potential_and_density,
    potential_and_density_2d,
    hist_vs_density,
    density_grid_2d,
    build_density_marginal_figure,
    density_marginal_snapshot,
)
from .diffusion import (
    noise_schedule,
    loss_curve,
    score_1d,
    two_hists_vs_density,
    two_hists,
    samples_2d,
    ring_density_and_marginal,
    density_and_samples_2d,
    score_fields,
    field_across_noise,
    samples_comparison_2d,
    mode_weight_bars,
)
from .lorenz import comparison_3d
from .inverse import posterior_ensemble
from . import cyclic_da


def use_latex(preamble=r"\usepackage{amsmath}\usepackage{amssymb}"):
    """Render every matplotlib text/math string with a real LaTeX engine.

    Call once near the top of a notebook or script (before drawing). Requires a
    working LaTeX installation on the system path; without one matplotlib will
    raise when it tries to render. The default ``preamble`` is enough for the
    bootcamp's math (``\\mathbf``, ``\\nabla``, ``\\mathcal``, ``\\propto``, …).
    """
    rc("text", usetex=True)
    rc("text.latex", preamble=preamble)


__all__ = [
    # Setup
    "use_latex",
    # Section 1
    "potential_and_density",
    "potential_and_density_2d",
    "hist_vs_density",
    "density_grid_2d",
    "build_density_marginal_figure",
    "density_marginal_snapshot",
    # Section 2
    "noise_schedule",
    "loss_curve",
    "score_1d",
    "two_hists_vs_density",
    "two_hists",
    "samples_2d",
    "ring_density_and_marginal",
    "density_and_samples_2d",
    "score_fields",
    "field_across_noise",
    "samples_comparison_2d",
    "mode_weight_bars",
    # Section 3
    "comparison_3d",
    # Section 4
    "posterior_ensemble",
    # Cyclic DA (namespaced; access as plot.cyclic_da.*)
    "cyclic_da",
]
