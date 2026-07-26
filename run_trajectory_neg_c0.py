#!/usr/bin/env python3
"""Negative-C₀ Božič & Svetina: A → B → oblate → D_sto → L_sto (nested spheres).

Run from this directory::

    python run_trajectory_neg_c0.py

Leave-sphere energy contest picks the oblate branch, then growth continues until
the approximate oblate↔stomatocyte energy boundary (nested two-sphere energy vs
oblate approx).  At ``D_sto`` the shape switches to the nested sphere-in-sphere
approximation and grows with pressure ``P ≈ −ΔE/ΔV`` (fixed area) until the
analytic limit ``L_sto``.

Seifert stomatocyte shooting is not used on this path.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bs_functions import (
    BS_C0_START_NEG,
    L_sto_reduced_volume,
    _load_track,
    assemble_neg_c0_trajectory,
    cache_dir,
    leave_sphere_by_energy,
    load_sto_track,
    nested_sphere_energy,
    nested_sphere_pressure_from_dEdV,
    results_dir,
    save_trajectory,
    step1_point_b,
    step2_oblate_to_D_sto,
    step4_nested_sphere_to_L_sto,
)
from bs_plotting import movie_full_trajectory, plot_full_trajectory
from seifert_functions import save_solution

ETA = 1.85
C0 = -1.0
T_D, KAPPA = 1.0, 1.0
DT = 0.001
DT_OBLATE = 0.005
DT_STO = 0.005
TAU_D_CEILING = 2.5
# D_sto can already exceed τ=1; grow until L_sto or this ceiling.
TAU_L_CEILING = 2.5
V_BUMP = 1e-5
# Leave-sphere bumps must resolve prolate vs oblate (tiny bumps → numerical ΔE).
U_BUMP, P_BUMP = 5e-3, 5e-3
USE_CACHE = True
MAKE_MOVIE = True
MOVIE_FRAMES = 180
PLOT_ONLY = False  # True → rebuild plots/movie from cache only
CACHE = cache_dir(ETA, T_D, C0=C0)
OUT = results_dir(ETA, C0=C0)


def _plot_and_movie(track, D, sto_track) -> None:
    full = assemble_neg_c0_trajectory(
        track, sto_track, eta=ETA, C0_dim=C0, T_d=T_D, kappa=KAPPA, D=D,
    )
    tau_D = float(D.constraints["t_growth"]) / T_D
    plot_dir = OUT / "full_trajectory"
    png = plot_full_trajectory(
        full, plot_dir, eta=ETA, C0_dim=C0, tau_D=tau_D,
    )
    tip = sto_track[-1]
    payload = {
        "eta": ETA,
        "C0": C0,
        "tau_D_sto": tau_D,
        "tau_L_sto": float(tip.constraints.get("t_growth", float("nan"))) / T_D,
        "v_L_sto": float(tip.constraints.get("v_L_sto", L_sto_reduced_volume(tip.c0))),
        "frames": [fr.to_dict() for fr in full],
    }
    (plot_dir / "full_trajectory.json").write_text(json.dumps(payload, indent=2))
    print(f"6) plot → {png}")
    if MAKE_MOVIE:
        mp4 = movie_full_trajectory(
            full, plot_dir, eta=ETA, target_frames=MOVIE_FRAMES, cache=CACHE,
        )
        print(f"   movie → {mp4}")


def main() -> None:
    if float(C0) >= 0.0:
        raise SystemExit("run_trajectory_neg_c0.py requires C0 < 0")

    if PLOT_ONLY:
        track = _load_track(CACHE, "oblate_growth_track")
        if not track:
            raise SystemExit(f"PLOT_ONLY: missing oblate track under {CACHE}")
        sto_track = load_sto_track(CACHE)
        if not sto_track:
            raise SystemExit(f"PLOT_ONLY: missing nested/sto track under {CACHE}")
        # D_sto = first δ<0 frame, else tip
        D = track[-1]
        for sol in track:
            d = sol.constraints.get("dE_approx_rel")
            if d is not None and float(d) < 0.0:
                D = sol
                break
            if sol.constraints.get("landmark") == "D_sto":
                D = sol
                break
        print(
            f"plot-only: {len(track)} oblate frames, "
            f"{len(sto_track)} nested-sphere frames, D_sto τ="
            f"{float(D.constraints['t_growth'])/T_D:.4f}"
        )
        _plot_and_movie(track, D, sto_track)
        return

    st = step1_point_b(
        ETA, C0, T_d=T_D, kappa=KAPPA, c0_start=BS_C0_START_NEG,
    )
    print(
        f"1) B (analytic): c₀={st['c0']:.4f}  U₀=U₁={st['U0']:.4f}  "
        f"P̄={st['P_bar']:.4f}  Σ̄={st['sigma_bar']:.4f}  "
        f"τ_B={st['tau_B']:.4f}  R={st['R']:.4f}"
    )

    winner, st2, contest = leave_sphere_by_energy(
        ETA, C0,
        T_d=T_D, kappa=KAPPA, c0_start=BS_C0_START_NEG,
        dt=DT, v_bump=V_BUMP, u_bump=U_BUMP, p_bump=P_BUMP,
        verbose=True,
    )
    side = str(contest.get("winner", "?"))
    E_w = float(winner.constraints.get("E_bending", float("nan")))
    print(
        f"2) leave-sphere winner: {side}  "
        f"v̄={winner.v:.6f}  c₀={winner.c0:.4f}  "
        f"U0={winner.U0:.4f}  U1={winner.U1:.4f}  E={E_w:.6g}"
    )
    if side != "oblate":
        print("   warning: expected oblate at B for this (η, C₀); continuing anyway")

    leave_path = save_solution(OUT / "leave_sphere", winner)
    print(f"   saved leave-sphere → {leave_path}")

    track, D = step2_oblate_to_D_sto(
        ETA, C0, winner,
        T_d=T_D, kappa=KAPPA, dt=DT_OBLATE,
        cache=CACHE, use_cache=USE_CACHE,
        tau_ceiling=TAU_D_CEILING, verbose=True,
    )
    tau_D = float(D.constraints["t_growth"]) / T_D
    tau0 = float(track[0].constraints["t_growth"]) / T_D
    n_ok = sum(1 for s in track if s.success)
    delta = D.constraints.get("dE_approx_rel", float("nan"))
    A_D = float(D.constraints.get("A_phys", float("nan")))
    V_D = float(D.constraints.get("V_phys", float("nan")))
    E_n = nested_sphere_energy(A_D, V_D, C0, kappa=KAPPA)
    P_n = nested_sphere_pressure_from_dEdV(A_D, V_D, C0, kappa=KAPPA)
    print(
        f"3) oblate B→D_sto: {n_ok} frames  "
        f"τ={tau0:.4f}→{tau_D:.4f}  v̄={track[0].v:.4f}→{D.v:.4f}  "
        f"c₀={track[0].c0:.4f}→{D.c0:.4f}  "
        f"δ={float(delta):+.4f}  D_by={D.constraints.get('D_by')}"
    )
    print(
        f"   nested@D handoff: E={E_n:.4f}  P=-ΔE/ΔV={P_n:.6f}  "
        f"L_sto(c₀)={L_sto_reduced_volume(D.c0):.6f}"
    )

    sto_track = step4_nested_sphere_to_L_sto(
        D, st, eta=ETA, C0=C0, T_d=T_D, kappa=KAPPA, dt=DT_STO,
        cache=CACHE, use_cache=USE_CACHE,
        tau_stop=TAU_L_CEILING, verbose=True,
    )
    tip = sto_track[-1]
    tau_L = float(tip.constraints["t_growth"]) / T_D
    print(
        f"4) nested-sphere D_sto→L_sto: {len(sto_track)} frames  "
        f"τ={tau_D:.4f}→{tau_L:.4f}  v̄={sto_track[0].v:.4f}→{tip.v:.4f}  "
        f"c₀={sto_track[0].c0:.4f}→{tip.c0:.4f}"
    )

    shapes = OUT / "shapes"
    save_trajectory(shapes, track, name="oblate_to_D_sto")
    save_trajectory(shapes, sto_track, name="nested_sphere_D_to_L")
    save_solution(shapes, D)
    save_solution(shapes, tip)
    print(f"5) outputs → {OUT}/  cache → {CACHE}")
    _ = st2

    _plot_and_movie(track, D, sto_track)


if __name__ == "__main__":
    main()
