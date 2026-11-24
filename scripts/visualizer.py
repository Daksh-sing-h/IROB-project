"""Live visualizer: MuJoCo vs black-box MLP vs DeLaN, side by side, in the browser.

    python -m scripts.visualizer            # then open http://localhost:8765

The server is stateless: the browser sends the current state of the three simulators plus the
next N cart forces; the server integrates MuJoCo (ground truth, with the rail limit) and the two
learned models (RK4, dt = 0.01 s, same scheme as MuJoCo) and returns the N new frames.
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from src.analytical import AnalyticalModel, params_from_physics
from src.data import make_physics
from src.train import load_model

DT = 0.01
HERE = os.path.dirname(os.path.abspath(__file__))


class Engine:
    def __init__(self, bb_path, dl_path):
        self.physics = make_physics()
        self.am = AnalyticalModel(params_from_physics(self.physics))
        self.models = {"bb": load_model(bb_path)[0], "dl": load_model(dl_path)[0]}
        self.lock = threading.Lock()

    def _f(self, model, q, qd, u):
        t = lambda a: torch.tensor(np.asarray(a, np.float32))[None]
        with torch.no_grad():
            return model(t(q), t(qd), torch.tensor([float(u)]))[0].numpy().astype(np.float64)

    def _rk4(self, model, q, qd, u):
        f = lambda q_, v_: (v_, self._f(model, q_, v_, u))
        k1q, k1v = f(q, qd)
        k2q, k2v = f(q + DT / 2 * k1q, qd + DT / 2 * k1v)
        k3q, k3v = f(q + DT / 2 * k2q, qd + DT / 2 * k2v)
        k4q, k4v = f(q + DT * k3q, qd + DT * k3v)
        return (q + DT / 6 * (k1q + 2 * k2q + 2 * k3q + k4q), qd + DT / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))

    def _pack(self, Q, V):
        Q, V = np.clip(np.nan_to_num(np.array(Q), nan=0.0), -1e3, 1e3), np.clip(np.nan_to_num(np.array(V), nan=0.0), -1e3, 1e3)
        T, P = self.am.energy(Q, V)          # energy of the *true* Lagrangian evaluated on each trajectory
        return dict(q=Q.tolist(), qd=V.tolist(), E=(T + P).tolist())

    def step(self, req):
        u = np.clip(np.asarray(req["u"], float), -1, 1)
        out = {}
        with self.lock:
            s = req["truth"]; p = self.physics
            p.data.qpos[:] = s["q"]; p.data.qvel[:] = s["qd"]
            Q, V = [], []
            for a in u:
                p.set_control(a); p.step(); Q.append(p.data.qpos.copy()); V.append(p.data.qvel.copy())
            out["truth"] = self._pack(Q, V)
            for k, model in self.models.items():
                q, qd = np.array(req[k]["q"], float), np.array(req[k]["qd"], float)
                Q, V = [], []
                for a in u:
                    q, qd = self._rk4(model, q, qd, a)
                    if not (np.isfinite(q).all() and np.isfinite(qd).all()):
                        q, qd = np.zeros(3), np.zeros(3)
                    Q.append(q); V.append(qd)
                out[k] = self._pack(Q, V)
        return out


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, open(os.path.join(HERE, "viz", "index.html"), "rb").read(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/step":
                return self._send(404, b"not found", "text/plain")
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(200, json.dumps(engine.step(req)).encode(), "application/json")
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bb", default="weights/blackbox_f1_s0.pt")
    ap.add_argument("--dl", default="weights/delan_f1_s0.pt")
    ap.add_argument("--bench", action="store_true", help="time one 20-step chunk and exit")
    a = ap.parse_args()
    eng = Engine(a.bb, a.dl)
    if a.bench:
        s = dict(q=[0, 3.0, 0.1], qd=[0, 0, 0]); req = dict(truth=s, bb=s, dl=s, u=[0.0] * 20)
        eng.step(req); t0 = time.time(); n = 5
        for _ in range(n): eng.step(req)
        print(f"{(time.time() - t0) / n * 1000:.0f} ms per 20-step chunk (real time needs < 200 ms)")
    else:
        print(f"serving on http://localhost:{a.port}")
        ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(eng)).serve_forever()
