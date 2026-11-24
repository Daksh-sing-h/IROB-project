"""Data generation for the cart-pole with two poles (dm_control `cartpole/two_poles`).

Every episode is simulated directly on the MuJoCo physics (dt = 0.01 s, RK4).
Per episode we store the states q, qd at t = 0..T and, for t = 0..T-1,
the action u, the applied generalized force tau = qfrc_actuator[0],
and the MuJoCo acceleration qdd (the training target).

    q, qd : [E, T+1, 3]    u, tau : [E, T]    qdd : [E, T, 3]

Distributions
  * "train" : x in [-0.6, 0.6], cart speed <= 1.5 m/s, joint speeds ~ N(0, 2^2) clipped to +-4 rad/s,
              theta_1 uniform over the full circle (so swing-up is covered).
  * "ood"   : larger velocities (4-8 rad/s joints, 1.5-3 m/s cart) and larger cart offsets.
Actions: zero-order-hold random, sinusoidal, chirp, mixtures, or zero (passive), optionally with a
weak cart-centering feedback so the cart stays clear of the +-1.8 m rail limit (not modelled by the
Lagrangian). Episodes that touch the limit are rejected.
"""
import numpy as np
from dm_control import suite

T_DEFAULT = 200          # steps per episode (2 s)
X_MAX = 1.7              # reject episodes whose cart goes beyond this (rail limit is 1.8)


def make_physics():
    return suite.load("cartpole", "two_poles").physics


def sample_initial_state(rng, dist="train"):
    th1 = rng.uniform(-np.pi, np.pi)
    th2 = rng.uniform(-np.pi, np.pi) if rng.random() < 0.4 else np.clip(rng.normal(0, 0.6), -np.pi, np.pi)
    if dist == "train":
        x = rng.uniform(-0.6, 0.6)
        xd = rng.uniform(-1.5, 1.5)
        thd = np.clip(rng.normal(0, 2.0, 2), -4, 4)
    elif dist == "ood":
        x = rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
        xd = rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
        thd = rng.choice([-1, 1], 2) * rng.uniform(4, 8, 2)
    else:
        raise ValueError(dist)
    return np.array([x, th1, th2]), np.array([xd, *thd])


def sample_actions(rng, T, dt=0.01, kind=None):
    t = np.arange(T) * dt
    kind = kind or rng.choice(["random", "sine", "chirp", "mix", "zero"], p=[0.35, 0.2, 0.2, 0.15, 0.1])
    if kind == "zero":
        return np.zeros(T), kind
    if kind in ("random", "mix"):
        hold = int(rng.choice([1, 5, 10, 20]))
        u = rng.uniform(-1, 1, T // hold + 1).repeat(hold)[:T]
    else:
        u = np.zeros(T)
    if kind in ("sine", "mix"):
        u = u + rng.uniform(0.3, 1) * np.sin(2 * np.pi * rng.uniform(0.2, 3) * t + rng.uniform(0, 2 * np.pi))
    if kind == "chirp":
        f0, f1 = rng.uniform(0.1, 0.5), rng.uniform(2, 4)
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t[-1]))
        u = rng.uniform(0.4, 1) * np.sin(phase)
    return np.clip(u, -1, 1), kind


def collect_episode(physics, rng, T=T_DEFAULT, dist="train", kind=None):
    """Returns a dict of arrays, or None if the episode left the allowed cart range."""
    q0, qd0 = sample_initial_state(rng, dist)
    u_open, kind = sample_actions(rng, T, kind=kind)
    centre = kind != "zero" and rng.random() < 0.5      # weak feedback keeps the cart on the rail
    kp, kd = 0.4, 0.4

    physics.reset()
    physics.data.qpos[:] = q0
    physics.data.qvel[:] = qd0
    q, qd, u, tau, qdd = [q0.copy()], [qd0.copy()], [], [], []
    for t in range(T):
        a = u_open[t]
        if centre:
            a = a - kp * physics.data.qpos[0] - kd * physics.data.qvel[0]
        a = float(np.clip(a, -1, 1))
        physics.set_control(a)
        physics.forward()                              # acceleration at the current state and action
        if np.abs(physics.data.qfrc_constraint).max() > 1e-9:
            return None
        u.append(a); tau.append(physics.data.qfrc_actuator[0]); qdd.append(physics.data.qacc.copy())
        physics.step()
        q.append(physics.data.qpos.copy()); qd.append(physics.data.qvel.copy())
        if abs(q[-1][0]) > X_MAX:
            return None
    return dict(q=np.array(q), qd=np.array(qd), u=np.array(u), tau=np.array(tau),
                qdd=np.array(qdd), kind=kind)


def generate(n_episodes, seed, dist="train", T=T_DEFAULT):
    rng = np.random.default_rng(seed)
    physics = make_physics()
    eps, tries = [], 0
    while len(eps) < n_episodes:
        tries += 1
        ep = collect_episode(physics, rng, T, dist)
        if ep is not None:
            eps.append(ep)
    out = {k: np.stack([e[k] for e in eps]) for k in ("q", "qd", "u", "tau", "qdd")}
    out["kind"] = np.array([e["kind"] for e in eps])
    out["acceptance"] = np.array(n_episodes / tries)
    return out


def load(name, data_dir="data"):
    return dict(np.load(f"{data_dir}/{name}.npz"))


def flatten(ds, noise_std=0.0, seed=0):
    """Per-sample arrays (q, qd, u, qdd) from episode data; optional Gaussian measurement noise on q, qd."""
    q, qd = ds["q"][:, :-1].reshape(-1, 3), ds["qd"][:, :-1].reshape(-1, 3)
    if noise_std > 0:
        r = np.random.default_rng(seed)
        q = q + r.normal(0, noise_std, q.shape); qd = qd + r.normal(0, noise_std, qd.shape)
    return q, qd, ds["u"].reshape(-1), ds["qdd"].reshape(-1, 3)
