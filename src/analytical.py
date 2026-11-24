"""Analytical Euler-Lagrange model of the dm_control cart-pole with two poles.

Generalized coordinates q = [x, th1, th2]
    x   : cart position (slide along +x)
    th1 : angle of pole 1 w.r.t. vertical (0 = upright), rotation about +y
    th2 : angle of pole 2 *relative* to pole 1
Only the cart is actuated: tau = [gear * u, 0, 0], u in [-1, 1].

Equations of motion:   M(q) qdd + c(q, qd) + g(q) + D qd = tau
All parameters are read from the MuJoCo model, so the model is the one MuJoCo
simulates.  Potential energy is measured relative to the cart height.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
import sympy as sp

NV = 3


@dataclass
class Params:
    mc: float      # cart mass
    m1: float      # pole masses
    m2: float
    l1: float      # length of pole 1 (joint 1 -> joint 2)
    lc1: float     # joint -> centre of mass distances
    lc2: float
    I1: float      # inertia about the hinge axis (y) through the COM
    I2: float
    damp: tuple    # joint damping (cart, hinge1, hinge2)
    gear: float    # actuator gear (force = gear * u)
    g: float = 9.81


def params_from_physics(physics) -> Params:
    m = physics.model
    ids = {n: m.name2id(n, "body") for n in ("cart", "pole_1", "pole_2")}

    def inertia_yy(b):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(m.body_iquat[b], dtype=float))
        R = R.reshape(3, 3)
        return float((R @ np.diag(m.body_inertia[b]) @ R.T)[1, 1])

    return Params(
        mc=float(m.body_mass[ids["cart"]]),
        m1=float(m.body_mass[ids["pole_1"]]),
        m2=float(m.body_mass[ids["pole_2"]]),
        l1=float(m.body_pos[ids["pole_2"]][2]),
        lc1=float(m.body_ipos[ids["pole_1"]][2]),
        lc2=float(m.body_ipos[ids["pole_2"]][2]),
        I1=inertia_yy(ids["pole_1"]),
        I2=inertia_yy(ids["pole_2"]),
        damp=tuple(float(d) for d in m.dof_damping),
        gear=float(m.actuator_gear[0, 0]),
        g=float(-m.opt.gravity[2]),
    )


def _stack(nested, shape):
    """Assemble a nested list of scalars/arrays into an array [*shape, r, c]."""
    r, c = len(nested), len(nested[0])
    out = np.empty(shape + (r, c))
    for i in range(r):
        for j in range(c):
            out[..., i, j] = nested[i][j]
    return out


class AnalyticalModel:
    """Symbolically derived Euler-Lagrange model, vectorised over leading dims."""

    def __init__(self, p: Params):
        self.p = p
        x, t1, t2, xd, t1d, t2d = sp.symbols("x t1 t2 xd t1d t2d")
        q = sp.Matrix([x, t1, t2])
        qd = sp.Matrix([xd, t1d, t2d])
        P = {k: sp.Float(v) for k, v in vars(p).items() if k not in ("damp",)}

        # centre-of-mass positions in the (x, z) plane (z relative to the cart)
        x1 = x + P["lc1"] * sp.sin(t1)
        z1 = P["lc1"] * sp.cos(t1)
        x2 = x + P["l1"] * sp.sin(t1) + P["lc2"] * sp.sin(t1 + t2)
        z2 = P["l1"] * sp.cos(t1) + P["lc2"] * sp.cos(t1 + t2)

        def d(e):  # total time derivative
            return sum(sp.diff(e, qi) * qdi for qi, qdi in zip(q, qd))

        T = (
            P["mc"] * xd**2 / 2
            + P["m1"] * (d(x1) ** 2 + d(z1) ** 2) / 2 + P["I1"] * t1d**2 / 2
            + P["m2"] * (d(x2) ** 2 + d(z2) ** 2) / 2 + P["I2"] * (t1d + t2d) ** 2 / 2
        )
        V = P["g"] * (P["m1"] * z1 + P["m2"] * z2)

        M = sp.hessian(T, qd)
        Mdot = sum((M.diff(qk) * qdk for qk, qdk in zip(q, qd)), sp.zeros(3, 3))
        c = Mdot * qd - sp.Matrix([sp.diff(T, qi) for qi in q])   # Coriolis/centrifugal
        g = sp.Matrix([sp.diff(V, qi) for qi in q])

        args = (x, t1, t2, xd, t1d, t2d)
        mk = lambda e: sp.lambdify(args, e, modules="numpy", cse=True)
        self._M = mk(M.tolist())
        self._c = mk(c.tolist())
        self._g = mk(g.tolist())
        self._T = mk(T)
        self._V = mk(V)
        self.D = np.diag(p.damp)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _split(q, qd):
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        return (q[..., 0], q[..., 1], q[..., 2], qd[..., 0], qd[..., 1], qd[..., 2]), q.shape[:-1]

    def mass_matrix(self, q):
        a, sh = self._split(q, np.zeros_like(q))
        return _stack(self._M(*a), sh)

    def coriolis(self, q, qd):
        a, sh = self._split(q, qd)
        return _stack(self._c(*a), sh)[..., 0]

    def gravity(self, q):
        a, sh = self._split(q, np.zeros_like(q))
        return _stack(self._g(*a), sh)[..., 0]

    def energy(self, q, qd):
        """Return (kinetic, potential)."""
        a, sh = self._split(q, qd)
        return np.broadcast_to(self._T(*a), sh).copy(), np.broadcast_to(self._V(*a), sh).copy()

    # --------------------------------------------------------------- dynamics
    def tau(self, u):
        u = np.asarray(u, float)
        t = np.zeros(u.shape + (NV,))
        t[..., 0] = self.p.gear * u
        return t

    def inverse(self, q, qd, qdd, include_damping=True):
        """Generalized force needed to produce qdd (what mj_inverse returns)."""
        out = (self.mass_matrix(q) @ np.asarray(qdd)[..., None])[..., 0] \
            + self.coriolis(q, qd) + self.gravity(q)
        if include_damping:
            out = out + np.asarray(qd) @ self.D
        return out

    def forward(self, q, qd, u, damping=True):
        rhs = self.tau(u) - self.coriolis(q, qd) - self.gravity(q)
        if damping:
            rhs = rhs - np.asarray(qd) @ self.D
        return np.linalg.solve(self.mass_matrix(q), rhs[..., None])[..., 0]

    def step_rk4(self, q, qd, u, dt, damping=True):
        f = lambda q_, qd_: (qd_, self.forward(q_, qd_, u, damping))
        k1q, k1v = f(q, qd)
        k2q, k2v = f(q + dt / 2 * k1q, qd + dt / 2 * k1v)
        k3q, k3v = f(q + dt / 2 * k2q, qd + dt / 2 * k2v)
        k4q, k4v = f(q + dt * k3q, qd + dt * k3v)
        return (q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q),
                qd + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))

    def rollout(self, q0, qd0, u_seq, dt=0.01, damping=True):
        """u_seq [..., T] -> q, qd of shape [..., T+1, 3]."""
        u_seq = np.asarray(u_seq, float)
        q, qd = np.asarray(q0, float), np.asarray(qd0, float)
        Q, V = [q], [qd]
        for t in range(u_seq.shape[-1]):
            q, qd = self.step_rk4(q, qd, u_seq[..., t], dt, damping)
            Q.append(q); V.append(qd)
        return np.stack(Q, -2), np.stack(V, -2)
