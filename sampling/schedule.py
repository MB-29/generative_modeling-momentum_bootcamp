"""Variance-Preserving (VP) noise schedule, written in *generative* time.

We use a single time variable ``t`` in ``[0, 1]`` oriented along the direction
the generative model runs, so that generation moves **forward in time**:

    t = 0   ->   pure noise,   x ~ N(0, I)
    t = 1   ->   the data,      x ~ p_data   (the learned target density)

Generation therefore starts from noise at ``t = 0`` and integrates forward up to
the data at ``t = 1`` -- there is no "reverse" pass. The corresponding
corruption process is just this read backwards (data at ``t = 1`` blurred into
noise as ``t`` decreases to 0); we never need to simulate it explicitly because
its Gaussian marginals are known in closed form.

For a data point ``x0`` the marginal at time ``t`` is Gaussian,

    x_t = mu_t * x0 + sigma_t * eps,        eps ~ N(0, I),

with a signal coefficient that *grows* toward the data end and a noise level
that *shrinks*:

    mu_t    = exp(-0.5 * B(1 - t)),
    sigma_t = sqrt(1 - exp(-B(1 - t))) = sqrt(1 - mu_t**2),

where ``B(s) = beta_min * s + 0.5 * (beta_max - beta_min) * s**2`` is the
integrated noise rate after running the corruption for a duration ``s = 1 - t``.
At ``t = 1`` (no corruption) ``mu_t = 1, sigma_t = 0`` and ``x_t = x0``; at
``t = 0`` (full corruption) ``mu_t ~ 0, sigma_t ~ 1`` and ``x_t ~ N(0, I)``.

The instantaneous rate the generative SDE sees at time ``t`` is

    beta(t) = beta_min + (1 - t) * (beta_max - beta_min),

high near the noisy end (``t = 0``) and low near the data end (``t = 1``).

The schedule is "variance preserving": if ``Var(x0) = 1`` then ``Var(x_t) = 1``
for all ``t``, which is why we standardize data to unit variance before training.

These functions accept python floats, numpy arrays, or torch tensors, so they
can be reused directly inside the torch training loop and the numpy plotting code.
"""

import numpy as np

# Default schedule hyper-parameters (Song et al. VP-SDE defaults).
BETA_MIN = 0.1
BETA_MAX = 20.0


def beta(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Instantaneous noise rate at generative time t (high at t=0, low at t=1)."""
    return beta_min + (1.0 - t) * (beta_max - beta_min)


def _corruption_integral(s, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """B(s) = int_0^s rate(u) du, the noise accumulated over a corruption time s."""
    return beta_min * s + 0.5 * (beta_max - beta_min) * s**2


def _exp(x):
    """exp that works for both numpy and torch inputs."""
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return np.exp(x)
    return x.exp()  # torch tensor


# ============================================================================
# QUESTION 2.1 -- VP schedule (mu_t, sigma_t)
# ============================================================================
# Student version (currently commented out -- the working implementations
# follow below).
#
# Complete the two functions ``mu_t`` and ``sigma_t`` using
#     mu_t    = exp(-0.5 * B(1 - t))
#     sigma_t = sqrt(1 - mu_t**2)
# The helper ``_corruption_integral(s) = B(s)`` is already implemented above.
# Use the ``_exp`` wrapper for the exponential so the same function works for
# numpy arrays and torch tensors. For the square root in ``sigma_t``, clip the
# argument to a small positive number (~1e-20) before taking the root (it can
# be very slightly negative at t=1 because of floating-point round-off).
#
# Sanity check: ``mu_t(0.0)`` should be ~0 and ``mu_t(1.0) = 1``; ``sigma_t(1.0)``
# should be ~0 and ``sigma_t(0.0)`` should be ~1.
# ----------------------------------------------------------------------------
# def mu_t(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
#     """Signal coefficient mu_t = exp(-0.5 * B(1 - t)): mu_1 = 1 (data), mu_0 ~ 0."""
#     # ---------------------------------------------------------------- #
#     # >>> QUESTION 2.1 (a): implement mu_t.                            #
#     #                                                                  #
#     #     Use ``_corruption_integral(1 - t, beta_min, beta_max)`` and  #
#     #     the wrapper ``_exp(...)`` so it works with numpy AND torch.  #
#     # ---------------------------------------------------------------- #
#     raise NotImplementedError("Q2.1(a): implement mu_t.")
#
#
# def sigma_t(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
#     """Noise level sigma_t = sqrt(1 - mu_t**2): sigma_1 = 0 (data), sigma_0 ~ 1."""
#     # ---------------------------------------------------------------- #
#     # >>> QUESTION 2.1 (b): implement sigma_t.                         #
#     #                                                                  #
#     #     Build ``var = 1 - mu_t(t, ...)**2``, then return its square  #
#     #     root after clipping to a small positive number (eg 1e-20) to #
#     #     avoid sqrt of a tiny negative round-off.                     #
#     #                                                                  #
#     #     For numpy:                                                    #
#     #         return np.sqrt(np.clip(var, 1e-20, None))                #
#     #     For torch:                                                    #
#     #         return var.clamp_min(1e-20).sqrt()                       #
#     # ---------------------------------------------------------------- #
#     raise NotImplementedError("Q2.1(b): implement sigma_t.")
# ============================================================================


def mu_t(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Signal coefficient mu_t = exp(-0.5 * B(1 - t)): mu_1 = 1 (data), mu_0 ~ 0."""
    return _exp(-0.5 * _corruption_integral(1.0 - t, beta_min, beta_max))


def sigma_t(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Noise level sigma_t = sqrt(1 - mu_t**2): sigma_1 = 0 (data), sigma_0 ~ 1."""
    m = mu_t(t, beta_min, beta_max)
    var = 1.0 - m**2
    if isinstance(var, np.ndarray) or np.isscalar(var):
        return np.sqrt(np.clip(var, 1e-20, None))
    return var.clamp_min(1e-20).sqrt()  # torch tensor


def time_from_sigma(sigma, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Invert ``sigma_t``: the generative time ``t`` whose noise level is ``sigma``.

    Lets us label a plot by the *noise level* sigma it was evaluated at rather
    than by the (less interpretable) time t. From
    ``sigma**2 = 1 - exp(-B(1 - t))`` we get ``B(1 - t) = -log(1 - sigma**2)``,
    then solve the quadratic ``B(s) = beta_min*s + 0.5*(beta_max-beta_min)*s**2``
    for ``s = 1 - t`` (taking the positive root).
    """
    sigma = np.asarray(sigma, dtype=float)
    target = -np.log(np.clip(1.0 - sigma**2, 1e-20, None))   # = B(1 - t)
    a = 0.5 * (beta_max - beta_min)
    b = beta_min
    s = (-b + np.sqrt(b**2 + 4.0 * a * target)) / (2.0 * a)  # positive root for 1 - t
    return 1.0 - s
