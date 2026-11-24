"""Black-box MLP and Deep Lagrangian Network (DeLaN) for the double cart-pole.

Both map (q, qd, u) -> qdd, with q = [x, th1, th2].  Angles enter through sin/cos.

DeLaN (Lutter et al., 2019) learns the Lagrangian L = 1/2 qd^T H(q) qd - V(q) with
    H(q) = L(q) L(q)^T,   L lower-triangular with a strictly positive diagonal  (=> H > 0)
and obtains Coriolis/centrifugal and gravity terms by automatic differentiation:
    c(q, qd) = dH/dt qd - dT/dq,      g(q) = dV/dq
    H qdd + c + g + D qd = B u
Prior knowledge used by DeLaN: the Euler-Lagrange structure, positive-definite inertia, the known
actuator map B = [gear, 0, 0]^T, and translation invariance (H and V do not depend on the cart
position x, since the rail is not modelled).  Damping D = diag(d) is learned (d > 0).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, jvp, vmap


def mlp(n_in, n_out, hidden=128, layers=3):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]      # smooth activation: needed for 2nd derivatives
        d = hidden
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)


class BlackBox(nn.Module):
    """qdd = f_theta(x, sin/cos th, qd, u) -- no physics."""

    def __init__(self, hidden=128, layers=3, in_mu=None, in_sd=None, out_sd=None):
        super().__init__()
        self.net = mlp(9, 3, hidden, layers)
        self.register_buffer("in_mu", torch.zeros(9) if in_mu is None else torch.as_tensor(in_mu, dtype=torch.float32))
        self.register_buffer("in_sd", torch.ones(9) if in_sd is None else torch.as_tensor(in_sd, dtype=torch.float32))
        self.register_buffer("out_sd", torch.ones(3) if out_sd is None else torch.as_tensor(out_sd, dtype=torch.float32))

    @staticmethod
    def features(q, qd, u):
        return torch.cat([q[..., :1], torch.sin(q[..., 1:]), torch.cos(q[..., 1:]), qd, u[..., None]], -1)

    def forward(self, q, qd, u):
        z = (self.features(q, qd, u) - self.in_mu) / self.in_sd
        return self.net(z) * self.out_sd


class DeLaN(nn.Module):
    def __init__(self, hidden=128, layers=3, gear=10.0, out_sd=None, init_damp=1e-3):
        super().__init__()
        self.Hnet = mlp(4, 6, hidden, layers)           # 3 diagonal + 3 off-diagonal entries of L
        self.Vnet = mlp(4, 1, hidden, layers)
        self.log_damp = nn.Parameter(torch.full((3,), float(torch.log(torch.tensor(init_damp)))))
        self.register_buffer("B", torch.tensor([gear, 0.0, 0.0]))
        self.register_buffer("out_sd", torch.ones(3) if out_sd is None else torch.as_tensor(out_sd, dtype=torch.float32))
        self.eps = 1e-2                                  # minimum diagonal of L

    # --- Lagrangian ingredients (single sample: q [3]) -------------------------------------
    @staticmethod
    def ang(q):
        return torch.stack([torch.sin(q[1]), torch.cos(q[1]), torch.sin(q[2]), torch.cos(q[2])])

    def H(self, q):
        o = self.Hnet(self.ang(q))
        d = F.softplus(o[:3]) + self.eps
        z = torch.zeros_like(d[0])
        L = torch.stack([torch.stack([d[0], z, z]),
                         torch.stack([o[3], d[1], z]),
                         torch.stack([o[4], o[5], d[2]])])
        return L @ L.T

    def V(self, q):
        return self.Vnet(self.ang(q)).squeeze(-1)

    def _terms(self, q, qd):
        H, Hdot = jvp(self.H, (q,), (qd,))               # H and dH/dt = sum_k dH/dq_k qd_k
        dT_dq = grad(lambda q_: 0.5 * qd @ self.H(q_) @ qd)(q)
        g = grad(self.V)(q)
        return H, Hdot @ qd - dT_dq, g

    # --- batched API -----------------------------------------------------------------------
    @property
    def damping(self):
        return torch.exp(self.log_damp)

    def terms(self, q, qd):
        return vmap(self._terms)(q, qd)

    def mass_matrix(self, q):
        return vmap(self.H)(q)

    def energy(self, q, qd):
        """(kinetic, potential) of the learned Lagrangian (potential up to an additive constant)."""
        H = self.mass_matrix(q)
        return 0.5 * torch.einsum("bi,bij,bj->b", qd, H, qd), vmap(self.V)(q)

    def inverse(self, q, qd, qdd):
        H, c, g = self.terms(q, qd)
        return torch.einsum("bij,bj->bi", H, qdd) + c + g + self.damping * qd

    def forward(self, q, qd, u):
        H, c, g = self.terms(q, qd)
        rhs = self.B * u[..., None] - c - g - self.damping * qd
        return torch.linalg.solve(H, rhs[..., None])[..., 0]
