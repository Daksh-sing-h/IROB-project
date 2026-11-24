"""Phase 2 figures:  python -m scripts.plot_phase2   (writes results/figs/*.png)"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analytical import AnalyticalModel, params_from_physics
from src.data import flatten, load, make_physics
from src.train import load_model, predict, to_t

C = dict(bb="#d95f02", dl="#1f77b4", true="#444444")
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
DOF = ["cart  $\\ddot x$ (m/s$^2$)", "pole 1  $\\ddot\\theta_1$ (rad/s$^2$)", "pole 2  $\\ddot\\theta_2$ (rad/s$^2$)"]
bb, _ = load_model("weights/blackbox_f1_s0.pt")
dl, _ = load_model("weights/delan_f1_s0.pt")
rb, rd = (json.load(open(f"results/train_{n}_f1_s0.json")) for n in ("blackbox", "delan"))
am = AnalyticalModel(params_from_physics(make_physics()))

# 1. training curves -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4))
for r, k, lab in ((rb, "bb", "Black-box MLP"), (rd, "dl", "DeLaN")):
    h = r["history"]
    ax.semilogy([e["epoch"] for e in h], [e["val_nmse"] for e in h], color=C[k], lw=2, label=lab + " (validation)")
    ax.semilogy([e["epoch"] for e in h], [e["train_loss"] for e in h], color=C[k], lw=1, ls=":", label=lab + " (train loss)")
ax.set(xlabel="epoch", ylabel="normalised MSE of $\\ddot q$", title="Training curves (full data, seed 0)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("results/figs/p2_training_curves.png", dpi=150); plt.close(fig)

# 2. predicted vs true acceleration on the test set -----------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(11, 6.8))
q, qd, u, qdd = to_t(*flatten(load("test")))
for row, (m, k, lab) in enumerate(((bb, "bb", "Black-box MLP"), (dl, "dl", "DeLaN"))):
    p = predict(m, q, qd, u).numpy(); t = qdd.numpy()
    for j in range(3):
        a = axs[row, j]; lim = np.abs(t[:, j]).max() * 1.05
        a.scatter(t[:, j], p[:, j], s=2, alpha=0.25, color=C[k], rasterized=True)
        a.plot([-lim, lim], [-lim, lim], color=C["true"], lw=1)
        rm = np.sqrt(np.mean((p[:, j] - t[:, j]) ** 2))
        a.set(xlabel="MuJoCo " + DOF[j], ylabel="predicted", title=f"{lab}: RMSE {rm:.3f}")
fig.suptitle("One-step acceleration prediction on the in-distribution test set (20 000 samples)", y=1.0)
fig.tight_layout(); fig.savefig("results/figs/p2_pred_vs_true.png", dpi=150); plt.close(fig)

# 3. error per DOF, in distribution vs out of distribution ------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
w = 0.36
for j, a in enumerate(axs):
    for i, (r, k, lab) in enumerate(((rb, "bb", "Black-box"), (rd, "dl", "DeLaN"))):
        v = [r["test"]["rmse"][j], r["test_ood"]["rmse"][j]]
        b = a.bar(np.arange(2) + (i - 0.5) * w, v, w, color=C[k], label=lab)
        a.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    a.set_xticks([0, 1], ["in-dist. test", "out-of-dist."]); a.set_title(DOF[j]); a.set_ylabel("RMSE")
    a.set_ylim(0, a.get_ylim()[1] * 1.12)
axs[0].legend(); fig.tight_layout(); fig.savefig("results/figs/p2_rmse_bars.png", dpi=150); plt.close(fig)

# 4. what DeLaN learned: inertia matrix and potential energy vs the analytical model ---------------
th2 = np.linspace(-np.pi, np.pi, 400); th1 = 0.5
qs = np.c_[np.zeros_like(th2), np.full_like(th2, th1), th2]
Mt = am.mass_matrix(qs); Ml = dl.mass_matrix(torch.tensor(qs, dtype=torch.float32)).detach().numpy()
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
cols = plt.cm.tab10(np.arange(6))
for c, (i, j) in zip(cols, [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
    axs[0].plot(th2, Mt[:, i, j], color=c, lw=2.5, alpha=0.5)
    axs[0].plot(th2, Ml[:, i, j], color=c, lw=1.2, ls="--", label=f"$H_{{{i+1}{j+1}}}$")
axs[0].set(xlabel="$\\theta_2$ (rad), $\\theta_1$ = 0.5", ylabel="inertia matrix entry", title="Inertia matrix: analytical (thick) vs DeLaN (dashed)")
axs[0].legend(ncol=3, fontsize=8)
err = np.linalg.norm(Ml - Mt, axis=(1, 2)) / np.linalg.norm(Mt, axis=(1, 2)) * 100
axs[1].semilogy(th2, err, color=C["dl"]); axs[1].set(xlabel="$\\theta_2$ (rad)", ylabel="relative error of H (%)", title="Inertia-matrix error")
a1 = np.linspace(-np.pi, np.pi, 400); qv = np.c_[np.zeros_like(a1), a1, np.zeros_like(a1)]
_, Vt = am.energy(qv, np.zeros_like(qv)); Vl = dl.Vnet(torch.tensor(np.c_[np.sin(a1), np.cos(a1), np.zeros_like(a1), np.ones_like(a1)], dtype=torch.float32)).detach().numpy()[:, 0]
axs[2].plot(a1, Vt - Vt.mean(), color=C["true"], lw=3, alpha=0.5, label="analytical"); axs[2].plot(a1, Vl - Vl.mean(), color=C["dl"], ls="--", label="DeLaN")
axs[2].set(xlabel="$\\theta_1$ (rad), $\\theta_2$ = 0", ylabel="potential energy V (J, mean removed)", title="Potential energy"); axs[2].legend()
fig.tight_layout(); fig.savefig("results/figs/p2_learned_physics.png", dpi=150); plt.close(fig)
print("wrote results/figs/p2_*.png")
