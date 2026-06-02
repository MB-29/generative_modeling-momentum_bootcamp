
import torch
import torch.nn as nn


class ScoreNet(nn.Module):
    """Generic time-conditioned MLP score network for ``dim``-dimensional data."""

    def __init__(self, dim, hidden=128, depth=3):
        super().__init__()
        self.dim = dim

        # Input is the state x (dim) plus the scalar time t (1).
        layers = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        """x : (B, dim), t : (B,) or (B, 1) -> score (B, dim)."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        t = t.reshape(-1, 1).to(x.dtype)  # (B, 1)
        h = torch.cat([x, t], dim=-1)
        return self.net(h)


class ScoreNet1D(ScoreNet):
    """Score network for scalar (1D) data, e.g. the double well."""

    def __init__(self, hidden=128, depth=3):
        super().__init__(dim=1, hidden=hidden, depth=depth)


class ScoreNet2D(ScoreNet):
    """Score network for 2D data, e.g. the planar double well.

    Used to *visualize* the learned score as a vector field over the plane.
    """

    def __init__(self, hidden=128, depth=3):
        super().__init__(dim=2, hidden=hidden, depth=depth)


class ScoreNet3D(ScoreNet):
    """Score network for 3D state vectors, e.g. the Lorenz attractor."""

    def __init__(self, hidden=256, depth=4):
        super().__init__(dim=3, hidden=hidden, depth=depth)
