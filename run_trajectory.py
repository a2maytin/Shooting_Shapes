#!/usr/bin/env python3
"""Božič & Svetina vesicle growth: B → D (prolate) → D (pear) → E (+ two-sphere).

Run from this directory::

    python run_trajectory.py

Knobs below: ``ETA``, leave-sphere bumps, ``USE_CACHE``.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Allow `python run_trajectory.py` from Shooting_Shapes/ or repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from seifert_plotting import animate_meridian_trajectory

from bs_functions import (
    cache_dir,
    results_dir,
    save_trajectory,
    step1_point_b,
    step2_prolate_to_D,
    step3_pear_at_D,
    step4_pear_to_E,
    step5_full_track,
)
from bs_plotting import run_two_sphere_finish, save_growth_plots

ETA = 1.83
C0, T_D, KAPPA = 1.0, 1.0, 1.0
DT = 0.001
V_BUMP = 1e-5
PROLATE_U_BUMP, PROLATE_P_BUMP = 3e-5, 3e-5
USE_CACHE = True
CACHE = cache_dir(ETA, T_D)
OUT = results_dir(ETA)


def main() -> None:
    st = step1_point_b(ETA, C0, T_d=T_D, kappa=KAPPA)
    print(
        f"1) B: c₀={st['c0']:.4f}  U₀=U₁={st['U0']:.4f}  "
        f"P̄={st['P_bar']:.4f}  Σ̄={st['sigma_bar']:.4f}  τ_B={st['tau_B']:.4f}"
    )

    track, lm, D = step2_prolate_to_D(
        ETA, C0, st, T_d=T_D, kappa=KAPPA, dt=DT,
        v_bump=V_BUMP,
        prolate_u_bump=PROLATE_U_BUMP, prolate_p_bump=PROLATE_P_BUMP,
        cache=CACHE, use_cache=USE_CACHE, verbose=True,
    )
    tau_D = float(D.constraints["t_growth"]) / T_D
    tau0 = float(track[0].constraints["t_growth"]) / T_D
    n_ok = sum(1 for s in track if s.success)
    d_by = D.constraints.get("D_by", "proxy")
    print(
        f"2) prolate B→D: {n_ok} frames  "
        f"τ={tau0:.4f}→{tau_D:.4f}  v̄={track[0].v:.4f}→{D.v:.4f}  "
        f"c₀={track[0].c0:.4f}→{D.c0:.4f}  "
        f"leave ΔP={track[0].P_bar - st['P_bar']:+.6f}  D_by={d_by}"
    )

    pear_at_D = step3_pear_at_D(
        D, cache=CACHE, T_d=T_D, C0=C0, use_cache=USE_CACHE, verbose=True,
    )
    print(
        f"3) pear→D: v̄={pear_at_D.v:.4f}  c₀={pear_at_D.c0:.4f}  "
        f"(target v̄={D.v:.4f} c₀={D.c0:.4f}  "
        f"A={D.constraints['A_phys']:.4f})"
    )

    past_ok = step4_pear_to_E(
        pear_at_D, D, st, eta=ETA, C0=C0, T_d=T_D, kappa=KAPPA, dt=DT,
        cache=CACHE, use_cache=USE_CACHE,
        max_residual=1e-5, max_step_time=30.0,
    )
    tau_end = float(past_ok[-1].constraints["t_growth"]) / T_D
    print(
        f"4) pear D→cutoff: {len(past_ok)} frames  "
        f"τ={tau_D:.4f}→{tau_end:.4f}  v̄={past_ok[-1].v:.4f}"
    )

    full = step5_full_track(track, past_ok, D, cache=CACHE, use_cache=USE_CACHE)
    shapes_dir = OUT / "shapes"
    save_trajectory(shapes_dir, full, name="trajectory")
    plot_path = save_growth_plots(full, lm, D, eta=ETA, T_d=T_D, out_dir=OUT)
    mp4 = OUT / "trajectory.mp4"
    animate_meridian_trajectory(full, mp4, fps=10, max_frames=1000)
    A0, V0 = lm.A_phys_ref, lm.V_phys_ref
    print(
        f"5) Seifert shoot track → {OUT}/\n"
        f"   shapes: {shapes_dir}\n"
        f"   plots:  {plot_path}\n"
        f"   video:  {mp4}\n"
        f"   A/A₀ {full[0].constraints['A_phys']/A0:.3f}→"
        f"{full[-1].constraints['A_phys']/A0:.3f}  "
        f"V/V₀ {full[0].constraints['V_phys']/V0:.3f}→"
        f"{full[-1].constraints['V_phys']/V0:.3f}"
    )

    print("6) two-sphere finish + full trajectory plots…", flush=True)
    run_two_sphere_finish(eta=ETA, recompute=True)


if __name__ == "__main__":
    main()
