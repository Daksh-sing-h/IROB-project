"""Validate the analytical Euler-Lagrange model against MuJoCo.

Run from the project root:  python -m src.validate_analytical
"""
import mujoco
import numpy as np
from dm_control import suite

from .analytical import AnalyticalModel, params_from_physics

rng = np.random.default_rng(0)
physics = suite.load("cartpole", "two_poles").physics
mj_m, mj_d = physics.model.ptr, physics.data.ptr
p = params_from_physics(physics)
print("Parameters read from MuJoCo:", p)
am = AnalyticalModel(p)


def set_state(q, qd, u):
    with physics.reset_context():
        physics.data.qpos[:] = q
        physics.data.qvel[:] = qd
        physics.data.ctrl[:] = u
    physics.forward()   # reset_context does not apply the control; do it explicitly


err = dict(M=0, qacc=0, inverse=0, kinetic=0, dpot=0)
N = 300
V0 = None
for _ in range(N):
    q = np.array([rng.uniform(-1, 1), rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)])
    qd = rng.normal(0, 3, 3)
    u = rng.uniform(-1, 1)
    set_state(q, qd, u)

    Mmj = np.zeros((3, 3)); mujoco.mj_fullM(mj_m, mj_d, Mmj)
    err["M"] = max(err["M"], np.abs(Mmj - am.mass_matrix(q)).max())
    err["qacc"] = max(err["qacc"], np.abs(physics.data.qacc - am.forward(q, qd, u)).max())

    # inverse dynamics: choose a random qacc, ask MuJoCo for the force producing it
    qdd = rng.normal(0, 5, 3)
    physics.data.qacc[:] = qdd
    mujoco.mj_inverse(mj_m, mj_d)
    err["inverse"] = max(err["inverse"], np.abs(physics.data.qfrc_inverse - am.inverse(q, qd, qdd)).max())

    set_state(q, qd, u)                       # restore (mj_inverse overwrote qacc)
    pot, kin = physics.data.energy
    T, V = am.energy(q, qd)
    err["kinetic"] = max(err["kinetic"], abs(kin - T))
    if V0 is None:
        V0 = (pot, V)
    err["dpot"] = max(err["dpot"], abs((pot - V0[0]) - (V - V0[1])))

print("\nMax abs. errors over", N, "random states (angles over full circle):")
for k, v in err.items():
    print(f"  {k:8s} {v:.3e}")

# multi-step check: same RK4 scheme, same dt, random piecewise-constant forces
T = 200
u_seq = rng.uniform(-0.3, 0.3, T // 10).repeat(10)   # gentle: stay clear of the rail limit
q0 = np.array([0.0, np.pi, 0.1]); qd0 = np.array([0.0, 0.0, 0.0])
set_state(q0, qd0, 0.0)
Q = [q0.copy()]
for t in range(T):
    physics.set_control(u_seq[t]); physics.step(); Q.append(physics.data.qpos.copy())
Qa, _ = am.rollout(q0, qd0, u_seq, dt=physics.model.opt.timestep)
print(f"\n2 s rollout, analytical RK4 vs MuJoCo: max |q error| = {np.abs(np.array(Q) - Qa).max():.3e}")
print(f"(cart range during rollout: {np.array(Q)[:,0].min():.2f} .. {np.array(Q)[:,0].max():.2f}, limit +-1.8)")
