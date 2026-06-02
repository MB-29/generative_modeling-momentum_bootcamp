
import numpy as np
from scipy.integrate import dblquad
from scipy.special import logsumexp

# --- Mixture parameters --------------------------------------------------- #
# Two well-separated components of UNEQUAL weight (the bimodal-with-unequal-mass
# feature we want a generative model to reproduce), with tilted covariances so
# the score field is anisotropic rather than purely radial.
WEIGHTS = np.array([0.65, 0.35])
MEANS = np.array([[-1.5, -1.0],
                  [1.6,  1.2]])
COVS = np.array([[[0.55, 0.25],
                  [0.25, 0.35]],
                 [[0.50, -0.20],
                  [-0.20, 0.45]]])

N_COMPONENTS = len(WEIGHTS)

# Precomputed per-component quantities (inverse covariance, Cholesky factor, and
# the Gaussian log-normalizer log[1 / (2 pi sqrt|Sigma|)]).
_INV_COVS = np.linalg.inv(COVS)                      # (K, 2, 2)
_CHOLS = np.linalg.cholesky(COVS)                    # (K, 2, 2), lower
_LOG_NORMS = -np.log(2.0 * np.pi) - 0.5 * np.log(np.linalg.det(COVS))  # (K,)
_LOG_WEIGHTS = np.log(WEIGHTS)                        # (K,)


def _diffs(x):
    """Per-component displacements ``x - mu_k``; shape ``(..., K, 2)``."""
    x = np.asarray(x, dtype=float)
    return x[..., None, :] - MEANS


def _log_components(x):
    """Log Gaussian densities ``log N(x; mu_k, Sigma_k)``; shape ``(..., K)``."""
    diff = _diffs(x)
    maha = np.einsum("...ki,kij,...kj->...k", diff, _INV_COVS, diff)
    return _LOG_NORMS - 0.5 * maha


def _log_pdf_and_resp(x):
    """Return ``(log p(x), responsibilities)`` with shapes ``(...)``, ``(..., K)``.

    Computed with log-sum-exp so it stays finite far from both modes, where the
    component densities underflow (there the responsibilities are well-defined
    in the limit and ``log p`` is just very negative).
    """
    log_weighted = _LOG_WEIGHTS + _log_components(x)        # (..., K)
    log_p = logsumexp(log_weighted, axis=-1)                # (...)
    resp = np.exp(log_weighted - log_p[..., None])          # (..., K)
    return log_p, resp


def pdf(x):
    """Normalized mixture density p(x), shape ``(...)``."""
    log_p, _ = _log_pdf_and_resp(x)
    return np.exp(log_p)


def responsibilities(x):
    """Posterior over components ``r_k(x) = w_k N_k(x) / sum_j w_j N_j(x)``.

    Shape ``(..., K)``. ``responsibilities(x).argmax(-1)`` gives a hard
    component assignment, handy for measuring the per-mode mass of a sample set.
    """
    _, resp = _log_pdf_and_resp(x)
    return resp


def potential(x):
    """Energy ``U(x) = -log p(x)``, shape ``(...)`` (temperature-independent)."""
    log_p, _ = _log_pdf_and_resp(x)
    return -log_p


def grad_log_pdf(x):
    """Score of the (untempered) mixture ``grad log p(x)``, shape ``(..., 2)``.

    Responsibility-weighted blend of the per-component linear scores
    ``-Sigma_k^{-1}(x - mu_k)``.
    """
    diff = _diffs(x)                                         # (..., K, 2)
    _, resp = _log_pdf_and_resp(x)                           # (..., K)
    inv_cov_diff = np.einsum(
        "kij,...kj->...ki", _INV_COVS, diff)  # (..., K, 2)
    return -np.einsum("...k,...ki->...i", resp, inv_cov_diff)


def grad_potential(x):
    """Gradient of the energy ``grad U = -grad log p``, shape ``(..., 2)``.

    This is what the Langevin sampler expects; ``score = -grad_potential / tau``.
    """
    return -grad_log_pdf(x)


def score(x, tau=1.0):
    """Analytic score of the (tempered) data density ``grad log p_tau = grad log p / tau``.

    Shape ``(..., 2)``. This is the ``t -> 0`` target the trained score network
    should reproduce; we overlay it on the learned field for comparison.
    """
    return grad_log_pdf(x) / tau


def gibbs_unnormalized(x, tau=1.0):
    """Unnormalized tempered density ``exp(-U/tau) = p(x)**(1/tau)``."""
    log_p, _ = _log_pdf_and_resp(x)
    return np.exp(log_p / tau)


def partition_function(tau=1.0, bound=6.0):
    """Normalizer ``Z = int int p(x)**(1/tau) dx_1 dx_2``.

    At ``tau = 1`` the mixture is already normalized, so ``Z = 1`` exactly and we
    skip the (slow) 2D quadrature. For other temperatures we integrate over the
    box ``[-bound, bound]^2``.
    """
    if np.isclose(tau, 1.0):
        return 1.0
    z, _ = dblquad(lambda x2, x1: gibbs_unnormalized([x1, x2], tau=tau),
                   -bound, bound, -bound, bound)
    return z


def gibbs_density(x, tau=1.0, bound=6.0):
    """Normalized tempered density ``p_tau(x) = exp(-U/tau) / Z``.

    At ``tau = 1`` this is exactly the mixture density :func:`pdf`.
    """
    return gibbs_unnormalized(x, tau=tau) / partition_function(tau=tau, bound=bound)


def sample(n_samples, rng=None):
    """Draw ``n_samples`` exact samples from the mixture; shape ``(n_samples, 2)``.

    Pick a component index with probability ``w_k``, then draw from
    ``N(mu_k, Sigma_k)`` via its Cholesky factor. Exact for the plain mixture
    (``tau = 1``).
    """
    if rng is None:
        rng = np.random.default_rng()
    comp = rng.choice(N_COMPONENTS, size=n_samples, p=WEIGHTS)   # (n,)
    z = rng.standard_normal((n_samples, 2))
    x = MEANS[comp] + np.einsum("nij,nj->ni", _CHOLS[comp], z)
    return x
