"""Train the black-box MLP or DeLaN.

    python -m src.train --model delan --frac 1.0 --seed 0 --epochs 60
Writes weights/<model>_f<frac>_s<seed>.pt and results/train_<model>_f<frac>_s<seed>.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .data import flatten, load
from .models import BlackBox, DeLaN

GEAR = 10.0


def to_t(*arrs):
    return [torch.as_tensor(a, dtype=torch.float32) for a in arrs]


def build_model(kind, cfg, in_mu=None, in_sd=None, out_sd=None):
    if kind == "blackbox":
        return BlackBox(cfg["hidden"], cfg["layers"], in_mu, in_sd, out_sd)
    return DeLaN(cfg["hidden"], cfg["layers"], GEAR, out_sd)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(ck["kind"], ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.eval(), ck


@torch.no_grad()
def predict(model, q, qd, u, bs=4096):
    out = []
    for i in range(0, len(q), bs):
        with torch.enable_grad():       # DeLaN needs autograd internally
            out.append(model(q[i:i + bs], qd[i:i + bs], u[i:i + bs]).detach())
    return torch.cat(out)


def one_step_metrics(model, split, out_sd):
    q, qd, u, qdd = to_t(*flatten(split))
    pred = predict(model, q, qd, u)
    err = pred - qdd
    return dict(nmse=float(((err / out_sd) ** 2).mean()),                  # normalised MSE (all dofs)
                rmse=[float(v) for v in err.pow(2).mean(0).sqrt()])        # raw RMSE per dof [m/s^2, rad/s^2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["blackbox", "delan"], required=True)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of training episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=0, help="0 = parameter-matched default (delan 128, blackbox 180)")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--w_inv", type=float, default=1.0, help="DeLaN: weight of the force-residual (inverse) loss")
    ap.add_argument("--noise", type=float, default=0.0, help="std of Gaussian measurement noise on q, qd")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_batches", type=int, default=0, help="debug: limit batches per epoch")
    a = ap.parse_args()

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tr = load("train")
    n_ep = len(tr["u"]); keep = np.sort(rng.permutation(n_ep)[: max(1, int(round(a.frac * n_ep)))])
    tr = {k: v[keep] if v.ndim and len(v) == n_ep else v for k, v in tr.items()}
    q, qd, u, qdd = to_t(*flatten(tr, a.noise, a.seed))
    val = load("val")
    qv, qdv, uv, qddv = to_t(*flatten(val))

    out_sd = qdd.std(0)
    feats = BlackBox.features(q, qd, u)
    cfg = dict(hidden=a.hidden or (180 if a.model == "blackbox" else 128), layers=a.layers)
    model = build_model(a.model, cfg, feats.mean(0), feats.std(0) + 1e-6, out_sd)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n_batches = int(np.ceil(len(q) / a.bs)) if not a.max_batches else a.max_batches
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * n_batches)
    name = f"{a.model}_f{a.frac:g}_s{a.seed}{a.tag}"
    print(f"{name}: {len(q)} samples, {n_par} parameters")

    hist, best, t0 = [], (1e9, None), time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(q))
        tot = 0.0
        for b in range(n_batches):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            pred = model(q[idx], qd[idx], u[idx])
            loss = (((pred - qdd[idx]) / out_sd) ** 2).mean()
            if a.model == "delan" and a.w_inv > 0:            # equation-of-motion residual in force space
                tau_pred = model.inverse(q[idx], qd[idx], qdd[idx])
                tau_true = model.B * u[idx][:, None]
                loss = loss + a.w_inv * (((tau_pred - tau_true) / GEAR) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step(); sched.step(); tot += loss.item()
        model.eval()
        v = predict(model, qv, qdv, uv)
        vn = float((((v - qddv) / out_sd) ** 2).mean())
        hist.append(dict(epoch=ep, train_loss=tot / n_batches, val_nmse=vn))
        if vn < best[0]:
            best = (vn, {k: x.clone() for k, x in model.state_dict().items()})
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  ep {ep:3d} train {tot / n_batches:.4e} val nMSE {vn:.4e}  ({time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(best[1]); model.eval()
    os.makedirs("weights", exist_ok=True); os.makedirs("results", exist_ok=True)
    torch.save(dict(kind=a.model, cfg=cfg, state=model.state_dict(), args=vars(a)), f"weights/{name}.pt")
    res = dict(args=vars(a), n_params=n_par, best_val_nmse=best[0], history=hist,
               test=one_step_metrics(model, load("test"), out_sd), test_ood=one_step_metrics(model, load("test_ood"), out_sd))
    if a.model == "delan":
        res["damping"] = [float(x) for x in model.damping]
    json.dump(res, open(f"results/train_{name}.json", "w"), indent=1)
    print(f"best val nMSE {best[0]:.4e} | test nMSE {res['test']['nmse']:.4e} | ood nMSE {res['test_ood']['nmse']:.4e}")


if __name__ == "__main__":
    main()
