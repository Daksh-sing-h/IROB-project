# Physics-informed learning of the double cart-pole (ECE670/CSE633, Group 7)

DeLaN vs black-box MLP vs analytical Euler-Lagrange model on `dm_control` `cartpole/two_poles`,
used in an energy-based swing-up with LQR balance.

## Setup (Python 3.12 -- dm_control needs 3.10-3.12)
    uv venv --python 3.12 .venv && source .venv/bin/activate
    uv pip install -r requirements.txt

## Phase 1: analytical model and data (run from the project root)
    python -m src.validate_analytical   # analytical model vs mj_fullM / mj_inverse / MuJoCo rollout
    python -m scripts.make_data         # writes data/{train,val,test,test_ood}.npz

## Phase 2: models and training (parameter-matched: DeLaN 68.2k, black-box 67.5k)
    python -m src.train --model blackbox --epochs 60            # ~30 s
    python -m src.train --model delan    --epochs 60            # ~6 min on CPU
    # options: --frac 0.1|0.3|0.5|1.0 (data efficiency)  --seed N  --noise STD  --w_inv W
Weights go to `weights/`, training curves and one-step metrics to `results/train_*.json`.

## Live visualizer (for the demo)
    python -m scripts.visualizer        # then open http://localhost:8765
MuJoCo, the black-box MLP and DeLaN are simulated side by side from the same start state and cart force
(arrow keys / buttons push the cart; presets, sine force, speed and a "re-sync to MuJoCo" option are in the page).
Live plots show the state error and energy error of each model against MuJoCo.

## Figures
    python -m scripts.plot_phase2       # results/figs/p2_*.png
