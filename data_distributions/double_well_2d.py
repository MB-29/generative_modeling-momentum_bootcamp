
import numpy as np
from scipy.integrate import quad

k = 2.0


def f1(x):
    """1D asymmetric double-well component f1(x) = x^4 - 2 x^2 + 0.3 x."""
    return x**4 - 2.0 * x**2 + 0.3 * x


def df1(x):
    """Derivative f1'(x) of the double-well component."""
    raise NotImplementedError("Question 1: implement f1'(x).")


def f2(x):
    """1D harmonic (Gaussian-trap) component f2(x) = 1/2 k x^2."""
    return 0.5 * k * x**2


def df2(x):
    """Derivative f2'(x) of the harmonic component."""
    raise NotImplementedError("Question 1: implement f2'(x).")


def potential(x):
    """Potential f(x_1, x_2) = f1(x_1) + f2(x_2), shape ``(N,)``.

    Parameters
    ----------
    x : array_like, shape ``(N, 2)``
        Batch of points; column 0 is x_1, column 1 is x_2.
    """
    x = np.asarray(x, dtype=float)
    x1, x2 = x[:, 0], x[:, 1]
    return f1(x1) + f2(x2)


def grad_potential(x):
    """Exact gradient ``(f1'(x_1), f2'(x_2))``, shape ``(N, 2)``."""
    x = np.asarray(x, dtype=float)
    x1, x2 = x[:, 0], x[:, 1]
    return np.stack([df1(x1), df2(x2)], axis=1)


def gibbs_unnormalized(x):
    """Unnormalized Gibbs density exp(-f(x_1, x_2) )."""
    return np.exp(-potential(x))


def _z_x(tau=1.0, bounds=(-3.0, 3.0)):
    """1D partition function of the x_1-well, Z_x = ∫ exp(-f1(x)/tau) dx."""
    z, _ = quad(lambda x: np.exp(-f1(x) / tau), bounds[0], bounds[1])
    return z


def _z_y(tau=1.0):
    """1D partition function of the x_2-trap, Z_y = sqrt(2 pi tau / k) (Gaussian)."""
    return np.sqrt(2.0 * np.pi * tau / k)


def partitionfunction(bounds=(-3.0, 3.0)):
    """Normalizer Z = Z_x * Z_y (the density factorizes across x_1 and x_2).
    """
    return _z_x(bounds=bounds) * _z_y()


def gibbs_density(x, bounds=(-3.0, 3.0)):
    """Normalized 2D Gibbs density p(x_1, x_2) = exp(-f/tau) / Z."""
    return gibbs_unnormalized(x) / partitionfunction(bounds=bounds)


def x_marginal(x, bounds=(-3.0, 3.0)):
    """Marginal density of x_1, p_{x_1}(x) = exp(-f1(x)/tau) / Z_x.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-f1(x)) / _z_x(bounds=bounds)


def score(x):
    """Analytic score of the data density, ``∇ log p = -∇f / tau``, shape ``(N, 2)``.
    """
    return -grad_potential(x)
