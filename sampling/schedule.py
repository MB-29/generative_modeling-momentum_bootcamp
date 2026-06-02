
import numpy as np

# Default schedule hyper-parameters (Song et al. VP-SDE defaults).
BETA_MIN = 0.1
BETA_MAX = 20.0


def beta_schedule(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Instantaneous noise rate at generative time t (high at t=0, low at t=1)."""
    return beta_min + (1.0 - t) * (beta_max - beta_min)


def mu_integral(s, beta_min=BETA_MIN, beta_max=BETA_MAX):
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
# The helper ``mu_integral(s) = B(s)`` is already implemented above.
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
#     #     Use ``mu_integral(1 - t, beta_min, beta_max)`` and  #
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


def mu_schedule(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Signal coefficient mu_t = exp(-0.5 * B(1 - t)): mu_1 = 1 (data), mu_0 ~ 0."""
    return _exp(-0.5 * mu_integral(1.0 - t, beta_min, beta_max))


def sigma_schedule(t, beta_min=BETA_MIN, beta_max=BETA_MAX):
    """Noise level sigma_t = sqrt(1 - mu_t**2): sigma_1 = 0 (data), sigma_0 ~ 1."""
    m = mu_schedule(t, beta_min, beta_max)
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
    s = (-b + np.sqrt(b**2 + 4.0 * a * target)) / \
        (2.0 * a)  # positive root for 1 - t
    return 1.0 - s
