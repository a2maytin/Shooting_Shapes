"""Božič–Svetina growth, landmarks B/D, workflow steps, and two-sphere finish.

Shape equilibria come from ``seifert_functions`` (Appendix-B shooting).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from seifert_functions import (
    PACKAGE_ROOT,
    A_STAR,
    DEFAULT_BS_CACHE,
    MeridianSolution,
    V_at_area,
    _continue_report,
    _score_prolate_candidate,
    _shoot_two_leg,
    _track_index,
    area_radius,
    bending_energy,
    c0_reduced,
    integrate_appendix_b_stage2,
    integrate_stage25_pear_at_D,
    load_solution,
    load_solution_list,
    load_trajectory,
    reduced_volume_from_AV,
    save_solution_at,
    save_solution_list,
    save_trajectory,
    solve_seifert,
    solve_stage3_from_appendix_b,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Package paths / pear Appendix-B seed
# ---------------------------------------------------------------------------
PEAR_APPENDIX_B_SEED = PACKAGE_ROOT / "pear_appendix_b_seed.json"


def resolve_pear_seed_path() -> Path:
    """In-package Appendix-B pear warm start (no ``bozic_svetina_cache`` required)."""
    p = PEAR_APPENDIX_B_SEED
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing pear seed {p}. Expected pear_appendix_b_seed.json in Shooting_Shapes/."
        )
    return p


def load_pear_appendix_b_seed() -> MeridianSolution:
    return load_solution(resolve_pear_seed_path())


# ===== BS physics (from bs_solver) =====

BS_ETA_MIN = 1.85  # self-reproduction: η = T_d L_p κ C₀⁴ ≳ 1.85

BS_V_TWO_SPHERE = float(1.0 / np.sqrt(2.0))  # point E: v̄ = 2^{-1/2}

BS_C0_START = 2.0  # point A: c₀(0) = 2 (sphere of zero bending energy, R = 2/C₀)

BS_C0_END = float(2.0 * np.sqrt(2.0))  # point E: c₀ = 2√2 when τ = 1

BS_E_STAR_P_BAR = -0.51  # shooting guess at ε-bumped two-sphere E*

BS_E_STAR_SIGMA_BAR = 0.51  # shooting guess at ε-bumped two-sphere E*

BS_TAU_D_REF = 0.955  # fallback τ_D if energy crossing is unavailable
BS_TAU_D_SEARCH_LO = 0.95  # start comparing prolate/pear bending energies
BS_TAU_D_GROW = 0.97  # grow prolates at least this far when searching for D


def two_sphere_alpha_from_v(v: float) -> Optional[float]:
    """``α = R₁/R₀`` from the two-sphere limit ``α³ + (1-α²)^(3/2) = v``."""
    v = float(v)
    v_lo = 2.0 ** -0.5
    if not (v_lo - 1e-9 <= v <= 1.0 + 1e-9):
        return None
    if abs(v - v_lo) < 1e-12:
        return v_lo
    if abs(v - 1.0) < 1e-12:
        return 1.0

    def _f(alpha: float) -> float:
        if alpha <= 0.0 or alpha >= 1.0:
            return 1e3
        return alpha ** 3 + (1.0 - alpha * alpha) ** 1.5 - v

    a_eq = v_lo
    try:
        if v <= a_eq ** 3 + (1.0 - a_eq * a_eq) ** 1.5 + 1e-12:
            return float(brentq(_f, 1e-8, a_eq - 1e-8))
        return float(brentq(_f, a_eq + 1e-8, 1.0 - 1e-8))
    except ValueError:
        return None


def two_sphere_radii_from_v(v: float, R0: float = 1.0) -> Optional[Tuple[float, float]]:
    """``(R₁, R₂)`` for area ``4π R₀²`` and volume ``(4π/3) v R₀³`` (thin-neck limit)."""
    alpha = two_sphere_alpha_from_v(v)
    if alpha is None:
        return None
    r1 = float(alpha * R0)
    r2 = float(np.sqrt(max(0.0, R0 * R0 - r1 * r1)))
    return r1, r2


def two_sphere_radii_from_AV(A: float, V: float) -> Optional[Tuple[float, float]]:
    """Two-sphere radii from physical ``(A, V)``.

    Thin-neck limit::

        4π (R₁² + R₂²) = A
        (4π/3) (R₁³ + R₂³) = V

    Returns ``(R₁, R₂)`` with ``R₁ ≥ R₂ > 0``, or ``None`` if ``v̄`` is
    outside ``[2^{-1/2}, 1]``.
    """
    A = float(A)
    V = float(V)
    if A <= 0.0 or V <= 0.0:
        return None
    R0 = area_radius(A)
    v = reduced_volume_from_AV(A, V)
    radii = two_sphere_radii_from_v(v, R0)
    if radii is None:
        return None
    r1, r2 = float(radii[0]), float(radii[1])
    if r1 < r2:
        r1, r2 = r2, r1
    return r1, r2


def eta_bozic(
    T_d: float,
    L_p: float,
    C0: float,
    kappa: float = 1.0,
) -> float:
    """Control parameter ``η = T_d L_p κ C₀⁴`` (Božič & Svetina 2004)."""
    return float(T_d * L_p * kappa * (C0 ** 4))

def L_p_from_eta(
    eta: float,
    T_d: float,
    C0: float,
    kappa: float = 1.0,
) -> float:
    """Hydraulic permeability from ``η`` at given ``T_d``, ``C₀``, ``κ``."""
    denom = float(T_d * kappa * (C0 ** 4))
    if denom <= 0.0:
        raise ValueError("T_d, kappa, and C0 must be positive")
    return float(eta / denom)

def alpha_from_T_d(T_d: float) -> float:
    """Area growth rate ``α = ln 2 / T_d`` so ``A(t) = A(0) e^{α t}``."""
    if T_d <= 0.0:
        raise ValueError("T_d must be positive")
    return float(np.log(2.0) / T_d)

def bs_growth_step(
    A: float,
    V: float,
    dt: float,
    *,
    alpha: float,
    L_p: float,
    P: float,
) -> Tuple[float, float]:
    """One Božič–Svetina growth step: ``dA/dt = α A``, ``dV/dt = L_p A P``."""
    dt = float(dt)
    if dt <= 0.0:
        return float(A), float(V)
    A_new = float(A * np.exp(float(alpha) * dt))
    V_new = float(V + float(L_p) * float(A) * float(P) * dt)
    return A_new, V_new

def bs_step_past_b(
    A: float,
    V: float,
    t: float,
    *,
    C0: float,
    T_d: float,
    L_p: float,
    dt: float,
    v_bump: float = 1e-6,
) -> Dict[str, float]:
    """Advance ``(A, V, t)`` one step past point **B** for a prolate shoot.

    Area advances with ``dA/dt = α A``; volume with eq.~(8) ``dV/dt = L_p A ΔP``.
    Because ``A`` grows faster than ``V`` can track a sphere, ``v̄_ode`` drops
    slightly below 1.  The prolate shoot uses ``v̄ = 1 - v_bump`` at the new
    area (same lagging-sphere picture).
    """
    alpha = alpha_from_T_d(T_d)
    P = float(sphere_bs_DeltaP(area_radius(A), T_d, L_p))
    A1, V_ode = bs_growth_step(A, V, dt, alpha=alpha, L_p=L_p, P=P)
    v_ode = float(reduced_volume_from_AV(A1, V_ode))
    v_shoot = float(1.0 - abs(v_bump))
    V_shoot = float(V_at_area(v_shoot, A1))
    return {
        "A": A1,
        "V": V_shoot,
        "V_ode": float(V_ode),
        "t_growth": float(t + dt),
        "R": float(area_radius(A1)),
        "c0": float(c0_reduced(C0, A1)),
        "v": v_shoot,
        "v_ode": v_ode,
        "DeltaP": float(sphere_bs_DeltaP(area_radius(A1), T_d, L_p)),
        "dt_step": float(dt),
    }

def sphere_bs_DeltaP(
    R: float,
    T_d: float,
    L_p: float,
) -> float:
    """Pressure difference for *spherical* growth (Božič & Svetina eq. 8).

    From ``dA/dt = (ln 2 / T_d) A`` and ``dV/dt = L_p A ΔP`` with
    ``V = A^{3/2}/(6√π)``:

        ΔP = (ln 2) R / (2 T_d L_p)

    ``ΔP = P_out − P_in > 0`` ⇒ solvent flows in, volume increases.
    """
    if R <= 0.0 or T_d <= 0.0 or L_p <= 0.0:
        raise ValueError("R, T_d, L_p must be positive")
    return float(np.log(2.0) * R / (2.0 * T_d * L_p))

def c0_critical_sphere(eta: float) -> float:
    """Critical reduced ``c₀`` where spherical growth ends (BS eq. 9 at equality).

    Spherical growth requires

        1 − 4 η (6 − c₀) / ((ln 2) c₀⁴) < 0

    so the critical ``c₀,cr(η)`` solves ``(ln 2) c₀⁴ + 4 η c₀ − 24 η = 0``.
    For ``η = 1.85`` one finds ``c₀,cr ≈ 2.475``.
    """
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    # Scalar polynomial in x = c0: (ln2) x^4 + 4η x - 24η = 0
    ln2 = float(np.log(2.0))

    def f(x: float) -> float:
        return ln2 * x ** 4 + 4.0 * eta * x - 24.0 * eta

    # Root in (0, 6): f(0) = -24η < 0, f(6) = (ln2)*1296 > 0
    lo, hi = 1e-6, 6.0 - 1e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

def R0_from_c0(C0: float, c0: float = BS_C0_START) -> float:
    """Radius of the equivalent sphere: ``R = c₀ / C₀`` (BS: start at ``c₀ = 2``)."""
    if C0 == 0.0:
        raise ValueError("C0 must be nonzero")
    return float(c0 / C0)

def bs_point_a_ref(
    C0: float,
    *,
    c0_start: float = BS_C0_START,
) -> Dict[str, float]:
    """Analytic area/volume at point **A** (``c₀ = c₀(0)``, ``v̄ = 1``, ``R = c₀/C₀``)."""
    R = float(R0_from_c0(C0, c0_start))
    A = float(4.0 * np.pi * R * R)
    V = float(4.0 * np.pi * R ** 3 / 3.0)
    return {
        "R": R,
        "A": A,
        "V": V,
        "c0": float(c0_start),
        "v": 1.0,
    }

def tau_B_from_eta(eta: float, c0_start: float = BS_C0_START) -> float:
    """Reduced time ``t/T_d`` at point B: ``c₀(τ) = c₀(0) 2^{τ/2} = c₀,cr(η)``."""
    c0_cr = c0_critical_sphere(eta)
    if c0_start <= 0.0 or c0_cr <= c0_start:
        raise ValueError(f"need 0 < c0_start < c0_cr (got {c0_start}, {c0_cr})")
    return float(2.0 * np.log2(c0_cr / c0_start))

def bs_point_b_state(
    eta: float,
    C0: float,
    *,
    T_d: float = 1.0,
    kappa: float = 1.0,
    c0_start: float = BS_C0_START,
) -> Dict[str, float]:
    """Analytic Božič–Svetina state at point **B** (eq.~9 equality, ``v̄ = 1``).

    ``c₀ = c₀,cr(η)``, ``R = c₀/C₀``, ``ΔP`` from eq.~(8), ``P̄ = ΔP``,
    ``Σ̄`` from the sphere identity, ``U₀ = U₁ = 1/R``.
    """
    c0_B = float(c0_critical_sphere(eta))
    R_B = float(R0_from_c0(C0, c0_B))
    tau_B = float(tau_B_from_eta(eta, c0_start=c0_start))
    L_p = float(L_p_from_eta(eta, T_d, abs(C0), kappa=kappa))
    A_B = float(4.0 * np.pi * R_B * R_B)
    V_B = float(4.0 * np.pi * R_B ** 3 / 3.0)
    DeltaP = float(sphere_bs_DeltaP(R_B, T_d, L_p))
    P_bar = float(P_bar_from_DeltaP(DeltaP, R_B, kappa=kappa))
    sigma_bar = float(sphere_sigma_bar_at_radius(C0, P_bar, R_B))
    U0 = float(1.0 / R_B)
    U1 = U0
    _, _, S1 = _sphere_geom_at_area(A_B)
    return {
        "c0": c0_B,
        "R": R_B,
        "tau_B": tau_B,
        "t_growth": float(tau_B * T_d),
        "A": A_B,
        "V": V_B,
        "v": 1.0,
        "L_p": L_p,
        "DeltaP": DeltaP,
        "P_bar": P_bar,
        "sigma_bar": sigma_bar,
        "U0": float(U0),
        "U1": float(U1),
        "S1": float(S1),
        "C0_dimensional": float(C0),
        "T_d": float(T_d),
        "kappa": float(kappa),
        "eta": float(eta),
        "eq9_lhs": float(sphere_stability_lhs(c0_B, eta)),
    }


def solve_bs_point_b(
    eta: float,
    C0: float,
    *,
    T_d: float = 1.0,
    kappa: float = 1.0,
    c0_start: float = BS_C0_START,
    dt: Optional[float] = None,
    v_bump: float = 1e-6,
    prolate_u_bump: float = 1e-4,
    prolate_p_bump: float = 1e-4,
    branch: str = "seifert",
    u0_window: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[MeridianSolution, Dict[str, float]]:
    """Point **B** IC: analytic sphere at **B**, then prolate shoot just past **B**.

    The prolate branch does not exist at the critical sphere.  One growth step
    advances ``A`` (and eq.~8 ``V``); ``v̄`` lags below 1, and the shoot uses
    ``v̄ = 1 - v_bump`` with ``U₀↑``, ``P̄↓``.
    """
    st = bs_point_b_state(
        eta, C0, T_d=T_d, kappa=kappa, c0_start=c0_start,
    )
    dt_step = float(dt if dt is not None else T_d / 1000.0)
    post = bs_step_past_b(
        st["A"], st["V"], st["t_growth"],
        C0=C0, T_d=T_d, L_p=st["L_p"], dt=dt_step, v_bump=v_bump,
    )
    warm = build_analytic_bs_sphere(
        post["c0"], area=post["A"], volume=post["V"],
        T_d=T_d, L_p=st["L_p"], C0_dimensional=C0, kappa=kappa, verbose=verbose,
    )
    if verbose:
        print(
            f"  B sphere ref: ΔP={st['DeltaP']:.4f} P̄={st['P_bar']:.4f} "
            f"Σ̄={st['sigma_bar']:.4f} U₀={st['U0']:.4f} "
            f"(R={st['R']:.4g}, c₀={st['c0']:.4g}, τ={st['tau_B']:.4g})",
            flush=True,
        )
        print(
            f"  post-B shoot: Δt={dt_step:.4g}  A/A_B={post['A']/st['A']:.6f}  "
            f"v̄={post['v']:.6f} (ode {post['v_ode']:.6f})  "
            f"c₀={post['c0']:.4g}  R={post['R']:.4g}  τ={post['t_growth']/T_d:.4g}",
            flush=True,
        )
    sol = _shoot_prolate_leave_sphere(
        post["v"], post["c0"], warm,
        A_target=post["A"], branch=branch, u0_window=u0_window, verbose=verbose,
        u_bump=prolate_u_bump, p_bump=prolate_p_bump,
    )
    sol.constraints.update({
        "R0": post["R"],
        "A_phys": post["A"],
        "V_phys": post["V"],
        "v_ode": post["v"],
        "C0_dimensional": float(C0),
        "DeltaP": float(sol.P_bar),
        "P_dimensional": float(growth_pressure_from_shoot(sol, post["R"], kappa=kappa)),
        "P_bar_seifert": float(sol.P_bar),
        "DeltaP_sphere_ref": st["DeltaP"],
        "eq9_lhs": st["eq9_lhs"],
        "eq9_ok": 0.0,
        "t_growth": post["t_growth"],
        "dt_step": post["dt_step"],
        "eta": st["eta"],
        "phase": "shape",
        "T_d": float(T_d),
        "L_p": st["L_p"],
        "kappa": float(kappa),
        "landmark": "B",
        "started_at_B": 1.0,
        "B_ref_A": st["A"],
        "B_ref_V": st["V"],
        "B_ref_t": st["t_growth"],
    })
    return sol, st

def _sphere_geom_at_area(area: float) -> Tuple[float, float, float]:
    """Analytic sphere geometry ``(U₀, U₁, S₁)`` at physical area ``A = 4π R²``.

    At ``A = 4π`` (``R = 1``): ``(1, 1, π)``.  Curvatures scale as ``1/R``,
    arclength as ``R``.
    """
    R = area_radius(area)
    U = 1.0 / R
    return float(U), float(U), float(np.pi * R)

def build_analytic_bs_sphere(
    c0: float,
    *,
    area: float,
    volume: Optional[float] = None,
    v: Optional[float] = None,
    T_d: float,
    L_p: float,
    C0_dimensional: float,
    kappa: float = 1.0,
    n: int = 200,
    branch: str = "bs_sphere",
    verbose: bool = False,
) -> MeridianSolution:
    """Growing sphere without shooting: analytic ``(U₀, U₁, S₁, P̄, Σ̄)`` + two-leg profile.

    ``ΔP = (ln 2) R / (2 T_d L_p)`` from eq.~(8); ``P̄ = ΔP`` and ``Σ̄`` from the
    sphere identity [16].  The meridian is built by integrating the shape equations at
    these fixed multipliers (no optimization).
    """
    c0 = float(c0)
    area = float(area)
    R0 = area_radius(area)
    C0_dim = float(C0_dimensional)
    if volume is None:
        v_hat = 1.0 if v is None else float(v)
        volume = float(V_at_area(v_hat, area))
    else:
        volume = float(volume)
        v_hat = float(reduced_volume_from_AV(area, volume))
    DeltaP = float(sphere_bs_DeltaP(R0, T_d, L_p))
    P_bar = P_bar_from_DeltaP(DeltaP, R0, kappa=kappa)
    sigma_bar = sphere_sigma_bar_at_radius(C0_dim, P_bar, R0)
    U0, U1, S1 = _sphere_geom_at_area(area)
    sol = integrate_appendix_b_stage2(
        U0, U1, S1,
        sigma_bar=sigma_bar, P_bar=P_bar, c0=c0,
        branch=branch, n=n, A_c0=area, A1_hint=area, V1_hint=volume,
    )
    res = float(sol.constraints.get("closure_res", float("nan")))
    v_tgt = float(V_at_area(v_hat, area))
    a_end = float(sol.y[5, -1]) if sol.y.size else float("nan")
    v_end = float(sol.y[6, -1]) if sol.y.size else float("nan")
    constraints = dict(sol.constraints)
    constraints.update({
        "A_phys": area,
        "R0": R0,
        "C0_dimensional": C0_dim,
        "A_target": area,
        "A_end": a_end,
        "V_end": v_end,
        "A_error_rel": float((a_end - area) / area) if np.isfinite(a_end) else float("nan"),
        "V_error_rel": float((v_end - v_tgt) / max(abs(v_tgt), 1e-12))
        if np.isfinite(v_end) else float("nan"),
        "residual": res,
        "DeltaP_enforced": DeltaP,
        "analytic_sphere": 1.0,
    })
    ok = sol.y.size > 0 and np.isfinite(res) and res < 5e-3
    ok = ok and abs(constraints["A_error_rel"]) < 1e-4
    ok = ok and abs(constraints["V_error_rel"]) < 1e-4
    out = replace(
        sol,
        v=v_hat,
        U0=U0,
        U1=U1,
        S1=S1,
        sigma_bar=sigma_bar,
        P_bar=P_bar,
        success=ok,
        message="analytic BS sphere" if ok else "analytic sphere integration failed",
        constraints=constraints,
    )
    if verbose:
        print(
            f"  sphere analytic: ΔP={DeltaP:.4f} P̄={P_bar:.4f} Σ̄={sigma_bar:.4f} "
            f"(R₀={R0:.4g}, v̄={v_hat:.4f}) |res|={res:.2e} ok={ok}",
            flush=True,
        )
    return out

def growth_pressure_from_shoot(
    sol: MeridianSolution,
    R0: float,
    *,
    kappa: float = 1.0,
) -> float:
    """Physical growth pressure from the shoot (same units as eq.~(8) ``ΔP``).

    After the sphere leg, ``P̄`` from the shoot is stored directly as ``ΔP`` in
    ``constraints['DeltaP']``.
    """
    return float(sol.P_bar)

def P_bar_from_DeltaP(DeltaP: float, R0: float = 1.0, kappa: float = 1.0) -> float:
    """Seifert shoot ``P̄`` for Božič–Svetina growth.

    At the physical scale used in the trajectory (``A = 4π R₀²``), the Lagrange
    multiplier ``P̄`` in eqs.~(3.5) equals the growth pressure ``ΔP`` from
    eq.~(8).  ``R₀`` and ``κ`` are accepted for API compatibility but do not
    enter the conversion.
    """
    return float(DeltaP)

def sphere_stability_lhs(c0: float, eta: float) -> float:
    """Left-hand side of BS eq. (9): spherical growth requires this ``< 0``."""
    c0 = float(c0)
    if c0 == 0.0:
        return float("inf")
    return float(1.0 - 4.0 * eta * (6.0 - c0) / (np.log(2.0) * c0 ** 4))

def sphere_sigma_bar_at_radius(
    C0: float,
    P_bar: float,
    R0: float,
) -> float:
    """``Σ̄`` for a sphere of radius ``R₀`` with dimensional ``C₀`` and shoot ``P̄``.

    Božič–Svetina / Seifert sphere identity (small deviations from sphere [16])::

        P̄ R₀² + (2 Σ̄ + C₀²) R₀ − 2 C₀ = 0

    At ``R₀ = 1`` this reduces to ``P̄ + 2 Σ̄ + c₀² − 2 c₀ = 0`` with ``c₀ = C₀``.
    """
    R0 = float(R0)
    C0 = float(C0)
    if R0 <= 0.0:
        raise ValueError("R0 must be positive")
    return float((2.0 * C0 - P_bar * R0 * R0 - C0 * C0 * R0) / (2.0 * R0))

def _bs_asymmetry(sol: MeridianSolution) -> float:
    return float(abs(sol.U0 - sol.U1) / max(abs(sol.U0), abs(sol.U1), 0.05))

def pick_bs_d_target(
    track: Sequence[MeridianSolution],
    landmarks: Optional["BSLandmarks"] = None,
    *,
    T_d: float = 1.0,
    tau_pear: float = BS_TAU_D_REF,
) -> MeridianSolution:
    """Prolate D: prefer ``landmarks.D`` (e.g. energy crossing); else τ proxy."""
    ok = [s for s in track if s.success]
    if not ok:
        raise ValueError("pick_bs_d_target requires a non-empty track")
    if landmarks is not None and landmarks.D is not None:
        return landmarks.D
    prolate = [s for s in ok if _bs_asymmetry(s) < 0.05]
    if not prolate:
        return ok[-1]
    t_end = float(ok[-1].constraints.get("t_growth", T_d))
    inner = [
        s for s in prolate
        if float(s.constraints.get("t_growth", 0.0)) < t_end - 1e-10
    ]
    if inner:
        t_cut = float(tau_pear) * float(T_d)
        early = [
            s for s in inner
            if float(s.constraints.get("t_growth", 0.0)) <= t_cut + 1e-10
        ]
        return early[-1] if early else inner[-1]
    return prolate[-1]


def _tag_energy_D(
    prol: MeridianSolution,
    pear: MeridianSolution,
    Ep: float,
    Ee: float,
) -> Tuple[MeridianSolution, MeridianSolution]:
    """Stamp the bending-energy crossing metadata on the prolate/pear D frames."""
    prol.constraints["phase"] = prol.constraints.get("phase", "shape")
    prol.constraints["D_by"] = "bending_energy"
    prol.constraints["E_prolate"] = Ep
    prol.constraints["E_pear"] = Ee
    pear.constraints["D_by"] = "bending_energy"
    pear.constraints["E_prolate"] = Ep
    pear.constraints["E_pear"] = Ee
    return prol, pear


def find_D_by_bending_energy(
    prolate_track: Sequence[MeridianSolution],
    pear_seed: MeridianSolution,
    *,
    tau_start: float = BS_TAU_D_SEARCH_LO,
    T_d: float = 1.0,
    C0_dim: float = 1.0,
    kappa: float = 1.0,
    verbose: bool = False,
    **stage3_kw: Any,
) -> Tuple[MeridianSolution, MeridianSolution, List[Dict[str, float]]]:
    """Locate D at the prolate↔pear bending-energy crossing.

    ``tau_start`` is only a guideline for where to begin.  If the pear already
    wins there, the scan walks *backward* (decreasing τ) one frame at a time
    until the prolate wins again; D is then the lowest-τ frame where the pear is
    still favorable (the transition).  If the prolate wins at ``tau_start`` the
    scan walks forward as before.  Walks a pear along the prolate ``(v̄, c₀, A)``
    path via stage-3 matching.  Returns ``(D_prolate, pear_at_D, table)``.
    Raises if no crossing is found.
    """
    T_d = float(T_d)
    prolates = sorted(
        (
            s for s in prolate_track
            if s.success and len(s.s) > 0
            and "t_growth" in s.constraints and "A_phys" in s.constraints
        ),
        key=lambda s: float(s.constraints["t_growth"]),
    )
    if not prolates:
        raise RuntimeError("no successful prolates for energy D search")

    t_lo = float(tau_start) * T_d
    start_idx = next(
        (i for i, s in enumerate(prolates)
         if float(s.constraints["t_growth"]) >= t_lo - 1e-12),
        None,
    )
    if start_idx is None:
        raise RuntimeError(
            f"no successful prolates with τ ≥ {tau_start} for energy D search"
        )

    kw = dict(
        u0_window=0.35,
        v_step=0.003,
        min_cv_steps=5,
        verbose=False,
    )
    kw.update(stage3_kw)

    table: List[Dict[str, float]] = []

    def _match(
        prol: MeridianSolution,
        pear_from: MeridianSolution,
        *,
        first_hop: bool = False,
    ) -> Optional[Tuple[MeridianSolution, float, float]]:
        """Match a pear to *prol* warm-started from *pear_from*; log a table row."""
        A = float(prol.constraints["A_phys"])
        local_kw = dict(kw)
        if first_hop:
            # Longer (c₀,v̄) hop from the raw seed → allow more cv sub-steps.
            local_kw["min_cv_steps"] = max(int(local_kw.get("min_cv_steps", 5)), 10)
            local_kw["v_step"] = min(float(local_kw.get("v_step", 0.003)), 0.002)
        pear_try = solve_stage3_from_appendix_b(
            pear_from, float(prol.v), float(prol.c0), A_target=A, **local_kw,
        )
        tau = float(prol.constraints["t_growth"]) / T_d
        if not pear_try.success or len(pear_try.s) < 5:
            if verbose:
                print(
                    f"  energy-D: τ={tau:.6f}  pear match failed "
                    f"({pear_try.message})",
                    flush=True,
                )
            return None
        Ep = bending_energy(prol, kappa=kappa, C0=C0_dim)
        Ee = bending_energy(pear_try, kappa=kappa, C0=C0_dim)
        table.append({
            "tau": tau, "v": float(prol.v), "c0": float(prol.c0),
            "E_prolate": Ep, "E_pear": Ee, "dE": Ee - Ep,
        })
        if verbose:
            print(
                f"  energy-D: τ={tau:.6f}  E_prolate={Ep:.4f}  E_pear={Ee:.4f}  "
                f"Δ={Ee - Ep:+.4f}",
                flush=True,
            )
        return pear_try, Ep, Ee

    # Evaluate the guideline frame first to pick a search direction.
    start = _match(prolates[start_idx], pear_seed, first_hop=True)
    if start is None:
        raise RuntimeError(
            f"pear match failed at guideline τ≈{tau_start} for energy D search"
        )
    pear_start, Ep0, Ee0 = start

    if Ee0 >= Ep0:
        # Prolate still wins at the guideline → march forward to the crossing.
        pear_cur = pear_start
        for prol in prolates[start_idx + 1:]:
            res = _match(prol, pear_cur)
            if res is None:
                continue
            pear_try, Ep, Ee = res
            pear_cur = pear_try
            if Ee < Ep:
                tau = float(prol.constraints["t_growth"]) / T_d
                if verbose:
                    print(
                        f"  energy-D: pear wins at τ={tau:.6f}  "
                        f"(E_pear={Ee:.4f} < E_prolate={Ep:.4f})",
                        flush=True,
                    )
                D_prol, pear_D = _tag_energy_D(prol, pear_try, Ep, Ee)
                return D_prol, pear_D, table
        raise RuntimeError(
            f"no prolate→pear bending-energy crossing for τ ≥ {tau_start} "
            f"(last Δ={table[-1]['dE']:+.4f} at τ={table[-1]['tau']:.6f})"
            if table else
            f"no prolate→pear bending-energy crossing for τ ≥ {tau_start}"
        )

    # Pear already wins at the guideline → walk backward to find the transition.
    best = (prolates[start_idx], pear_start, Ep0, Ee0)
    pear_cur = pear_start
    for prol in reversed(prolates[:start_idx]):
        res = _match(prol, pear_cur)
        if res is None:
            break
        pear_try, Ep, Ee = res
        if Ee < Ep:
            best = (prol, pear_try, Ep, Ee)
            pear_cur = pear_try
            continue
        tau = float(prol.constraints["t_growth"]) / T_d
        best_tau = float(best[0].constraints["t_growth"]) / T_d
        if verbose:
            print(
                f"  energy-D: transition bracketed — prolate wins at τ={tau:.6f}, "
                f"pear wins at τ={best_tau:.6f}",
                flush=True,
            )
        break
    else:
        if verbose and start_idx > 0:
            best_tau = float(best[0].constraints["t_growth"]) / T_d
            print(
                f"  energy-D: pear favored down to the earliest prolate; "
                f"D pinned at τ={best_tau:.6f}",
                flush=True,
            )

    D_prol, pear_at_D, Ep_D, Ee_D = best
    if verbose:
        tau_D = float(D_prol.constraints["t_growth"]) / T_d
        print(
            f"  energy-D: pear wins at τ={tau_D:.6f}  "
            f"(E_pear={Ee_D:.4f} < E_prolate={Ep_D:.4f})",
            flush=True,
        )
    D_prol, pear_at_D = _tag_energy_D(D_prol, pear_at_D, Ep_D, Ee_D)
    return D_prol, pear_at_D, table


def bs_continue_max_iter(n_steps: int, tau_remain: float, dt_nom: float) -> int:
    """Safety cap large enough to reach ``t_final`` when ``max_dv`` shrinks ``dt``."""
    min_dt = dt_nom / (2 ** 24)
    need = int(np.ceil(tau_remain / max(min_dt, 1e-15))) + 5
    return max(int(n_steps) + 1, need)

def bs_continue_n_expect(tau_remain: float, dt_nom: float) -> int:
    """Expected snapshot count at nominal ``dt`` (actual may exceed if ``max_dv`` shrinks ``dt``)."""
    if tau_remain <= 0.0 or dt_nom <= 0.0:
        return 1
    return int(np.ceil(tau_remain / dt_nom)) + 1

def _resolve_bs_dt(
    *,
    T_d: float,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
) -> Tuple[float, int]:
    """Return ``(dt_nom, n_steps_equiv)`` from explicit ``dt`` or ``n_steps``."""
    if dt is not None:
        dt_nom = float(dt)
        if dt_nom <= 0.0:
            raise ValueError("dt must be positive")
        n_equiv = max(int(np.ceil(float(T_d) / dt_nom)), 1)
        return dt_nom, n_equiv
    if n_steps is None:
        n_steps = 64
    n_steps = max(int(n_steps), 1)
    return float(T_d) / n_steps, n_steps

def _prolate_seeds_leave_sphere(
    warm: MeridianSolution,
    c0: float,
    *,
    A_target: Optional[float] = None,
    u_bump: float = 1e-4,
    p_bump: float = 1e-4,
) -> List[np.ndarray]:
    """Shoot seeds for leaving the sphere: ``U₀,U₁↑``, ``P̄↓`` around analytic ICs."""
    P_ref = float(warm.P_bar)
    S1_ref = float(warm.S1)
    sig_ref = float(warm.sigma_bar)
    A_w = float(
        A_target
        if A_target is not None
        else warm.constraints.get("A_phys", warm.constraints.get("A_target", A_STAR))
    )
    R_w = area_radius(A_w)
    U_sphere = float(1.0 / R_w)
    C0_w = float(warm.constraints.get("C0_dimensional", c0 / R_w if R_w > 0 else c0))
    du0 = max(float(u_bump), 0.0)
    dp0 = max(float(p_bump), 0.0)
    du_grid = tuple({du0, 2.0 * du0, 5.0 * du0} - {0.0})
    dp_grid = tuple({dp0, 2.0 * dp0, 5.0 * dp0} - {0.0})
    if not du_grid:
        du_grid = (1e-6,)
    if not dp_grid:
        dp_grid = (1e-6,)
    seeds: List[np.ndarray] = []
    seen: set = set()

    def add_seed(du: float, dp: float) -> None:
        U_seed = float(U_sphere * (1.0 + du))
        P_b = float(P_ref * (1.0 - dp))
        sig = sphere_sigma_bar_at_radius(C0_w, P_b, R_w)
        key = (round(U_seed, 15), round(P_b, 15))
        if key in seen:
            return
        seen.add(key)
        seeds.append(np.array([U_seed, U_seed, S1_ref, sig, P_b], dtype=float))

    for du in du_grid:
        for dp in dp_grid:
            add_seed(float(du), float(dp))
    if not seeds:
        seeds.append(np.array([U_sphere, U_sphere, S1_ref, sig_ref, P_ref], dtype=float))
    return seeds

def _shoot_prolate_leave_sphere(
    v: float,
    c0: float,
    warm: MeridianSolution,
    *,
    A_target: float,
    branch: str = "seifert",
    u0_window: Optional[float] = None,
    u_bump: float = 1e-4,
    p_bump: float = 1e-4,
    verbose: bool = False,
) -> MeridianSolution:
    """First prolate equilibrium after BS eq. (9) fails at point B."""
    u_bump = max(float(u_bump), 0.0)
    p_bump = max(float(p_bump), 0.0)
    seeds = _prolate_seeds_leave_sphere(
        warm, c0, A_target=float(A_target), u_bump=u_bump, p_bump=p_bump,
    )
    shoot_kw = dict(A_target=float(A_target), A_c0=float(A_target))
    U_sphere = float(1.0 / area_radius(float(A_target)))
    u0_min = float(U_sphere * (1.0 + 0.25 * max(u_bump, 1e-8)))
    u_win = float(u0_window) if u0_window is not None else max(0.05 * U_sphere, 0.01)
    u_win = min(u_win, 0.15 * max(U_sphere, 0.1))
    if verbose:
        print(
            f"    prolate-B: U_bump={u_bump:.2e}  P_bump={p_bump:.2e}  "
            f"1/R={U_sphere:.6f}  floor={u0_min:.6f}",
            flush=True,
        )
        print(
            f"    seeds: U₀∈[{seeds[0][0]:.9f},{seeds[-1][0]:.9f}]  "
            f"P̄∈[{min(s[4] for s in seeds):.9f},{max(s[4] for s in seeds):.9f}]  "
            f"u0_window={u_win:.4g}",
            flush=True,
        )
    best: Optional[MeridianSolution] = None
    best_score = float("-inf")

    def _try_seed(x0: np.ndarray) -> None:
        nonlocal best, best_score
        sol = _shoot_two_leg(
            v, c0, x0, branch=branch, n=200, u0_window=u_win, verbose=verbose,
            u0_min=u0_min, u1_min=u0_min,
            **shoot_kw,
        )
        if float(sol.U0) < u0_min or float(sol.U1) < u0_min:
            return
        score = _score_prolate_candidate(sol, u0_min=u0_min)
        if score > best_score:
            best_score = score
            best = sol

    for x0 in seeds:
        _try_seed(x0)

    if best is None:
        P_ref = float(warm.P_bar)
        S1_ref = float(warm.S1)
        R_w = area_radius(float(A_target))
        C0_w = float(warm.constraints.get("C0_dimensional", c0 / R_w if R_w > 0 else c0))
        for scale in (10.0, 20.0, 50.0):
            du = max(u_bump * scale, 1e-6)
            dp = max(p_bump * scale, 1e-6)
            U_seed = float(U_sphere * (1.0 + du))
            P_b = float(P_ref * (1.0 - dp))
            sig = sphere_sigma_bar_at_radius(C0_w, P_b, R_w)
            _try_seed(np.array([U_seed, U_seed, S1_ref, sig, P_b], dtype=float))

    if best is not None:
        best.constraints["branch_side"] = "prolate"
        best.constraints["prolate_u_bump"] = u_bump
        best.constraints["prolate_p_bump"] = p_bump
        return best

    fail = _shoot_two_leg(
        v, c0, seeds[0], branch=branch, n=200, u0_window=u_win, verbose=verbose,
        u0_min=u0_min, u1_min=u0_min,
        **shoot_kw,
    )
    fail = replace(
        fail,
        success=False,
        message="prolate leave-sphere failed (U₀ < 1/R — oblate rejected)",
        constraints={**fail.constraints, "branch_side": "oblate_rejected"},
    )
    return fail

def continue_bozic_svetina(
    prev: MeridianSolution,
    *,
    eta: Optional[float] = None,
    T_d: float = 1.0,
    kappa: float = 1.0,
    alpha: Optional[float] = None,
    L_p: Optional[float] = None,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    t_final: Optional[float] = None,
    C0_dim: Optional[float] = None,
    max_dv: float = 0.03,
    u0_window: float = 0.35,
    lock_branch_steps: Optional[int] = None,
    u0_window_late: Optional[float] = None,
    allow_deflation: bool = True,
    prolate_u_bump: float = 1e-4,
    prolate_p_bump: float = 1e-4,
    branch: str = "seifert",
    max_residual: Optional[float] = None,
    max_step_time: Optional[float] = None,
    verbose: bool = False,
    progress: bool = True,
) -> List[MeridianSolution]:
    """Area/volume growth at fixed dimensional ``C₀`` (Božič & Svetina 2004).

    Area grows as ``dA/dt = α A``; volume as ``dV/dt = L_p A ΔP`` with eq.~(8)
    ``ΔP`` on the sphere leg and shoot ``P̄`` thereafter.  The sphere leg is analytic
    while eq.~(9) holds **and** ``v̄ ≈ 1``; after **B**, ``A`` outruns ``V`` so
    ``v̄ < 1`` and the prolate shoot uses the ODE ``v̄`` (never clamped back to 1).

    Optional pear/late cutoffs (drop the offending frame and stop)::

        max_residual  — shoot ``|res|`` above this (e.g. ``1e-5``)
        max_step_time — wall-clock seconds per step (e.g. ``30``)
    """
    if prev is None or not prev.success:
        raise ValueError("continue_bozic_svetina requires a successful prev solution")

    C0_fixed = float(
        C0_dim
        if C0_dim is not None
        else prev.constraints.get("C0_dimensional", prev.c0)
    )
    if C0_fixed == 0.0 and eta is not None:
        raise ValueError("eta-based L_p requires C0 ≠ 0; pass L_p explicitly for C0=0")

    if alpha is None:
        alpha = alpha_from_T_d(T_d)
    else:
        T_d = float(np.log(2.0) / alpha)

    if L_p is None:
        if eta is None:
            eta = float(prev.constraints.get("eta", BS_ETA_MIN))
        L_p = L_p_from_eta(eta, T_d, abs(C0_fixed), kappa=kappa)
    else:
        eta = eta_bozic(T_d, L_p, abs(C0_fixed), kappa=kappa)

    if alpha <= 0.0 or L_p < 0.0:
        raise ValueError("alpha must be positive and L_p non-negative")

    R0_prev = float(prev.constraints.get("R0", 1.0))
    t_growth = float(prev.constraints.get("t_growth", 0.0))
    if t_final is None:
        t_final = t_growth + float(T_d)
    dt_nom, n_steps_eff = _resolve_bs_dt(T_d=T_d, dt=dt, n_steps=n_steps)
    tau_remain = max(float(t_final) - t_growth, 0.0)
    n_expect = bs_continue_n_expect(tau_remain, dt_nom)
    max_iter = bs_continue_max_iter(n_steps_eff, tau_remain, dt_nom)
    c0_cr = c0_critical_sphere(eta)

    out: List[MeridianSolution] = [prev]
    warm: Optional[MeridianSolution] = prev
    A_phys = float(prev.constraints.get("A_phys", A_STAR * R0_prev ** 2))
    V_phys = float(prev.constraints.get("V_phys", V_at_area(prev.v, A_phys)))

    if progress or verbose:
        print(
            f"continue_bozic_svetina: η={eta:.4g}, C₀={C0_fixed:.4g}, "
            f"T_d={T_d:.4g}, L_p={L_p:.4g}, α={alpha:.4g}, "
            f"dt_nom={dt_nom:.4g}, ≈{n_expect} snaps, t∈[{t_growth:.4g},{t_final:.4g}], "
            f"c₀,cr={c0_cr:.4g} (eq. 9)",
            flush=True,
        )

    def _shoot_shape(
        v_tgt: float,
        c0_tgt: float,
        warm_sol: Optional[MeridianSolution],
        *,
        A_target: float,
        shape_shoot_idx: int = 0,
    ) -> MeridianSolution:
        on_prolate = (
            warm_sol is not None
            and warm_sol.constraints.get("phase") == "shape"
            and warm_sol.U0 > 1.02
            and abs(warm_sol.U0 - warm_sol.U1) < 0.08
        )
        pick = "first"
        use_scan = False
        if lock_branch_steps is None:
            lock = on_prolate
        else:
            lock = on_prolate and shape_shoot_idx < int(lock_branch_steps)
        window = u0_window
        if not lock and u0_window_late is not None:
            window = float(u0_window_late)
        elif not lock:
            window = max(float(u0_window), 2.0)
        return solve_seifert(
            v_tgt, c0_tgt,
            prev=warm_sol,
            branch=branch,
            verbose=verbose,
            use_u0_scan=use_scan,
            pick=pick,
            lock_branch=(lock and warm_sol is not None),
            u0_window=window,
            A_target=float(A_target),
            A_c0=float(A_target),
        )

    i = 0
    shape_shoot_idx = 0
    while i < max_iter:
        if t_growth >= t_final - 1e-15:
            break

        t0 = time.perf_counter()
        advanced_this_step = False
        dt_leave = float(min(dt_nom, max(t_final - t_growth, 0.0)))
        if dt_leave <= 0.0:
            dt_leave = float(dt_nom)
        R0 = float(np.sqrt(A_phys / A_STAR))
        c0_red = c0_reduced(C0_fixed, A_phys)
        v_now = float(reduced_volume_from_AV(A_phys, V_phys))
        v_tol = max(1e-4, 0.5 * float(max_dv))
        eq9_lhs = float(sphere_stability_lhs(c0_red, eta))
        eq9_ok = eq9_lhs < 0.0
        on_sphere = eq9_ok and v_now <= 1.0 + v_tol

        v_shoot = float(max(v_now, 0.05))
        leaving_sphere = (
            warm is not None
            and warm.constraints.get("phase") == "sphere"
            and not on_sphere
        )
        if leaving_sphere:
            P_flux_b = float(sphere_bs_DeltaP(R0, T_d, L_p))
            A_phys, V_phys = bs_growth_step(
                A_phys, V_phys, dt_leave,
                alpha=alpha, L_p=L_p, P=P_flux_b,
            )
            t_growth += dt_leave
            advanced_this_step = True
            R0 = float(np.sqrt(A_phys / A_STAR))
            c0_red = c0_reduced(C0_fixed, A_phys)
            v_now = float(reduced_volume_from_AV(A_phys, V_phys))
            v_shoot = v_now
            eq9_lhs = float(sphere_stability_lhs(c0_red, eta))

        if on_sphere:
            if verbose or (progress and i == 0):
                print(
                    f"  sphere v̄={v_now:.4f} c₀={c0_red:.4f} "
                    f"(R₀={R0:.4f}, τ={t_growth / T_d:.4f}, eq9={eq9_lhs:+.3f})  "
                    f"[analytic]",
                    flush=True,
                )
            DeltaP_growth = float(sphere_bs_DeltaP(R0, T_d, L_p))
            sol = build_analytic_bs_sphere(
                c0_red, area=A_phys, volume=V_phys,
                T_d=T_d, L_p=L_p, C0_dimensional=C0_fixed, kappa=kappa,
                verbose=verbose,
            )
            tag = "sphere" if sol.success else "FAIL"
        elif leaving_sphere and warm is not None:
            if verbose or progress:
                print(
                    f"  shooting v̄={v_shoot:.4f} c₀={c0_red:.4f} "
                    f"(R₀={R0:.4f}, τ={t_growth / T_d:.4f}, eq9={eq9_lhs:+.3f})  "
                    f"[eq. (9) violated → prolate; post-B Δt={dt_leave:.4g}]",
                    flush=True,
                )
            warm_post = build_analytic_bs_sphere(
                c0_red, area=A_phys, volume=V_phys,
                T_d=T_d, L_p=L_p, C0_dimensional=C0_fixed, kappa=kappa,
                verbose=False,
            )
            sol = _shoot_prolate_leave_sphere(
                v_shoot, c0_red, warm_post,
                A_target=A_phys,
                branch=branch,
                verbose=verbose,
                u_bump=prolate_u_bump,
                p_bump=prolate_p_bump,
            )
            tag = "prolate-B" if sol.success else "FAIL"
            shape_shoot_idx += 1
        else:
            if verbose or (progress and leaving_sphere):
                print(
                    f"  shooting v̄={v_shoot:.4f} c₀={c0_red:.4f} "
                    f"(R₀={R0:.4f}, τ={t_growth / T_d:.4f})",
                    flush=True,
                )
            sol = _shoot_shape(v_shoot, c0_red, warm, A_target=A_phys, shape_shoot_idx=shape_shoot_idx)
            tag = "ok" if sol.success else "FAIL"
            shape_shoot_idx += 1

        phase = "sphere" if on_sphere else "shape"
        if on_sphere:
            P_flux = DeltaP_growth
        else:
            DeltaP_growth = float(sol.P_bar)
            P_flux = DeltaP_growth
        sol.constraints.update({
            "R0": R0,
            "A_phys": float(A_phys),
            "V_phys": float(V_phys),
            "v_ode": float(v_now),
            "v_shoot": float(v_shoot),
            "C0_dimensional": C0_fixed,
            "DeltaP": DeltaP_growth,
            "P_dimensional": DeltaP_growth if on_sphere else growth_pressure_from_shoot(
                sol, R0, kappa=kappa,
            ),
            "P_bar_seifert": float(sol.P_bar),
            "DeltaP_sphere_ref": float(sphere_bs_DeltaP(R0, T_d, L_p))
            if C0_fixed != 0.0 and L_p > 0.0 else float("nan"),
            "eq9_lhs": eq9_lhs,
            "eq9_ok": float(eq9_ok),
            "t_growth": float(t_growth),
            "dt_step": 0.0,
            "dt_nom": float(dt_nom),
            "alpha": float(alpha),
            "L_p": float(L_p),
            "T_d": float(T_d),
            "eta": float(eta),
            "phase": phase,
            "kappa": float(kappa),
        })

        dt_wall = time.perf_counter() - t0
        out.append(sol)
        if progress or verbose:
            _continue_report(i, n_expect, sol, tag, dt_wall)

        if not sol.success:
            if progress or verbose:
                print(f"  [{i + 1}] growth path stopped (solve failed)", flush=True)
            break

        res_now = float(sol.constraints.get("residual", float("nan")))
        if (
            max_residual is not None
            and np.isfinite(res_now)
            and res_now > float(max_residual)
        ):
            if progress or verbose:
                print(
                    f"  [{i + 1}] two-sphere cutoff: |res|={res_now:.3e} "
                    f"> {float(max_residual):.3e} at τ={t_growth / T_d:.4f} "
                    f"(drop frame; hand off to thin-neck model)",
                    flush=True,
                )
            out.pop()
            break
        if max_step_time is not None and dt_wall > float(max_step_time):
            if progress or verbose:
                print(
                    f"  [{i + 1}] two-sphere cutoff: step {dt_wall:.1f}s "
                    f"> {float(max_step_time):.1f}s at τ={t_growth / T_d:.4f} "
                    f"(drop frame; hand off to thin-neck model)",
                    flush=True,
                )
            out.pop()
            break

        if t_growth >= t_final - 1e-15:
            if progress or verbose:
                print(
                    f"  reached t_final={t_final:.4g} "
                    f"({len(out)} snapshots, τ={t_growth / T_d:.4g})",
                    flush=True,
                )
            break

        warm = sol
        if advanced_this_step:
            sol.constraints["dt_step"] = float(dt_leave)
            i += 1
            continue

        dt = min(dt_nom, t_final - t_growth)
        v_step = v_now
        min_dt = dt_nom / (2 ** 24)
        for _ in range(24):
            A_trial = float(A_phys * np.exp(alpha * dt))
            V_trial = float(V_phys + L_p * A_phys * P_flux * dt)
            if V_trial <= 0.0:
                dt *= 0.5
                continue
            v_trial = float(reduced_volume_from_AV(A_trial, V_trial))
            if abs(v_trial - v_step) <= max_dv:
                break
            dt *= 0.5
        else:
            if progress or verbose:
                A_trial = float(A_phys * np.exp(alpha * dt))
                V_trial = float(V_phys + L_p * A_phys * P_flux * dt)
                v_trial = float(reduced_volume_from_AV(A_trial, V_trial))
                print(
                    f"  [{i + 1}] growth path stopped "
                    f"(could not keep |Δv̄| ≤ {max_dv}; "
                    f"v̄={v_step:.5f}→{v_trial:.5f}, dt={dt:.2e}, "
                    f"shoot v̄={v_shoot:.4f})",
                    flush=True,
                )
            break

        A_old = A_phys
        A_phys = float(A_old * np.exp(alpha * dt))
        V_phys = float(V_phys + L_p * A_old * P_flux * dt)
        t_growth += dt
        sol.constraints["dt_step"] = float(dt)
        i += 1

    if i >= max_iter and t_growth < t_final - 1e-12 and out and out[-1].success:
        if progress or verbose:
            print(
                f"  stopped at iteration cap ({len(out)} snapshots, "
                f"τ={t_growth / T_d:.4g}, target {t_final / T_d:.4g})",
                flush=True,
            )

    return out

@dataclass
class BSLandmarks:
    """Landmarks on a Božič–Svetina growth trajectory."""

    A: Optional[MeridianSolution] = None  # start (sphere)
    B: Optional[MeridianSolution] = None  # leave sphere at c₀ = c₀,cr(η)
    C: Optional[MeridianSolution] = None  # P changes sign; volume starts falling
    D: Optional[MeridianSolution] = None  # prolate → asymmetric pear/dumbbell
    E: Optional[MeridianSolution] = None  # end (ideally two spheres at t = T_d)
    A_phys_ref: Optional[float] = None  # analytic A at point A (normalization)
    V_phys_ref: Optional[float] = None  # analytic V at point A (normalization)
    started_at_B: bool = False

    def summary(self) -> str:
        lines = []
        for name, sol in (
            ("A", self.A), ("B", self.B), ("C", self.C), ("D", self.D), ("E", self.E),
        ):
            if sol is None:
                lines.append(f"  {name}: —")
            else:
                lines.append(
                    f"  {name}: t={sol.constraints.get('t_growth', float('nan')):.4f}  "
                    f"v̄={sol.v:.4f}  c₀={sol.c0:.4f}  R₀={sol.constraints.get('R0', float('nan')):.4f}  "
                    f"U0={sol.U0:+.3f}  U1={sol.U1:+.3f}  P̄={sol.P_bar:+.3f}"
                )
        return "\n".join(lines)

def detect_bs_landmarks(
    track: Sequence[MeridianSolution],
    *,
    c0_B: Optional[float] = None,
    v_leave: float = 0.995,
    asym_D: float = 0.15,
    started_at_B: bool = False,
    C0: Optional[float] = None,
    c0_start: float = BS_C0_START,
) -> BSLandmarks:
    """Label points A–E on a completed BS track."""
    ok = [s for s in track if s.success]
    lm = BSLandmarks(started_at_B=started_at_B)
    if not ok:
        return lm
    C0_dim = C0
    if C0_dim is None:
        stored = ok[0].constraints.get("C0_dimensional")
        if stored is not None:
            C0_dim = float(stored)
    if C0_dim is not None:
        ref = bs_point_a_ref(float(C0_dim), c0_start=c0_start)
        lm.A_phys_ref = ref["A"]
        lm.V_phys_ref = ref["V"]
    if started_at_B:
        lm.B = ok[0]
        rest = ok[1:]
    else:
        lm.A = ok[0]
        rest = ok
    T_d = float(ok[-1].constraints.get("T_d", 1.0))
    t_end = float(ok[-1].constraints.get("t_growth", 0.0))
    tau_end = t_end / T_d if T_d > 0.0 else t_end

    # E: two-sphere endpoint at τ≈1 after pear continuation — not a prolate endpoint.
    if tau_end >= 0.99 * T_d:
        v_e = float(ok[-1].v)
        asym_e = _bs_asymmetry(ok[-1])
        if abs(v_e - BS_V_TWO_SPHERE) < 0.05 or asym_e >= asym_D:
            lm.E = ok[-1]

    # B: first snapshot off the analytic sphere leg (unless started_at_B).
    if lm.B is None:
        for s in ok:
            if s.constraints.get("phase") == "shape" or s.v < v_leave:
                lm.B = s
                break

    # C: first point after B where growth pressure crosses + → − and/or V falls.
    b_t = lm.B.constraints.get("t_growth", -1.0) if lm.B is not None else -1.0
    c_ok = rest if started_at_B else ok
    for i in range(1, len(c_ok)):
        if c_ok[i].constraints.get("t_growth", 0.0) < b_t - 1e-12:
            continue
        p_prev = c_ok[i - 1].constraints.get("DeltaP", c_ok[i - 1].P_bar)
        p_now = c_ok[i].constraints.get("DeltaP", c_ok[i].P_bar)
        V_prev = c_ok[i - 1].constraints.get("V_phys", float("nan"))
        V_now = c_ok[i].constraints.get("V_phys", float("nan"))
        if p_prev > 0.0 >= p_now:
            lm.C = c_ok[i]
            break
        if V_now < V_prev - 1e-12 and p_now <= 0.0 and lm.C is None:
            lm.C = c_ok[i]
            break

    # D: bending-energy crossing preferred (set by step2); else asymmetry / τ proxy.
    saw_symmetric = False
    for s in c_ok:
        asym = _bs_asymmetry(s)
        if asym < 0.05 and s.v < v_leave:
            saw_symmetric = True
        if saw_symmetric and asym >= asym_D:
            lm.D = s
            break
    if lm.D is None:
        lm.D = pick_bs_d_target(ok, lm, T_d=T_d)
    return lm


# ===== Workflow I/O =====

def _bs_workflow_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "workflow.json"

def save_bs_workflow(
    cache_dir: Path = DEFAULT_BS_CACHE,
    *,
    track: Optional[Sequence[MeridianSolution]] = None,
    landmarks: Optional[BSLandmarks] = None,
    stage2_shapes: Optional[Sequence[MeridianSolution]] = None,
    stage25: Optional[MeridianSolution] = None,
    stage3: Optional[MeridianSolution] = None,
    stage4: Optional[MeridianSolution] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Merge Božič–Svetina notebook checkpoints into ``cache_dir/workflow.json``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wf_path = _bs_workflow_path(cache_dir)
    wf: Dict[str, Any] = {}
    if wf_path.exists():
        wf = json.loads(wf_path.read_text())

    if meta:
        wf["meta"] = {**wf.get("meta", {}), **meta}

    if track is not None:
        save_trajectory(cache_dir, list(track), name="growth_track")
        wf["growth_track"] = "growth_track.json"
        if landmarks is not None:
            lm_idx: Dict[str, int] = {}
            for name in ("A", "B", "C", "D", "E"):
                sol = getattr(landmarks, name, None)
                if sol is None:
                    continue
                idx = _track_index(track, sol)
                if idx is not None:
                    lm_idx[name] = idx
            wf["landmark_indices"] = lm_idx
            wf["landmarks_meta"] = {
                "started_at_B": bool(landmarks.started_at_B),
                "A_phys_ref": landmarks.A_phys_ref,
                "V_phys_ref": landmarks.V_phys_ref,
            }

    if stage2_shapes is not None:
        save_solution_list(cache_dir, "pear_stage2", stage2_shapes)
        wf["pear_stage2"] = "pear_stage2.json"

    if stage25 is not None:
        save_solution_at(cache_dir / "pear_stage25.json", stage25)
        wf["pear_stage25"] = "pear_stage25.json"
        wf["pear_stage25_index"] = int(
            stage25.constraints.get("stage2_index", wf.get("pear_stage25_index", 0))
        )

    if stage3 is not None:
        save_solution_at(cache_dir / "pear_stage3.json", stage3)
        wf["pear_stage3"] = "pear_stage3.json"

    if stage4 is not None:
        save_solution_at(cache_dir / "pear_stage4.json", stage4)
        wf["pear_stage4"] = "pear_stage4.json"

    wf_path.write_text(json.dumps(wf, indent=2))
    return wf_path

def load_bs_workflow(
    cache_dir: Path = DEFAULT_BS_CACHE,
    *,
    require: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Reload cached BS workflow artifacts (missing optional sections are omitted)."""
    cache_dir = Path(cache_dir)
    wf_path = _bs_workflow_path(cache_dir)
    if not wf_path.exists():
        raise FileNotFoundError(f"No BS workflow cache at {wf_path}")
    wf = json.loads(wf_path.read_text())
    need = set(require or ())
    out: Dict[str, Any] = {"meta": wf.get("meta", {}), "manifest": wf}

    def _need(key: str) -> bool:
        return key in need

    if "growth_track" in wf:
        gt_path = cache_dir / wf["growth_track"]
        if gt_path.exists():
            track = load_trajectory(gt_path)
            out["track"] = track
            lm = BSLandmarks()
            for name, idx in wf.get("landmark_indices", {}).items():
                if 0 <= idx < len(track):
                    setattr(lm, name, track[idx])
            lm_meta = wf.get("landmarks_meta", {})
            lm.started_at_B = bool(lm_meta.get("started_at_B", "A" not in wf.get("landmark_indices", {})))
            lm.A_phys_ref = lm_meta.get("A_phys_ref")
            lm.V_phys_ref = lm_meta.get("V_phys_ref")
            if lm.A_phys_ref is None or lm.V_phys_ref is None:
                C0_dim = track[0].constraints.get("C0_dimensional") if track else None
                c0_start = float(wf.get("meta", {}).get("c0_start", BS_C0_START))
                if C0_dim is not None:
                    ref = bs_point_a_ref(float(C0_dim), c0_start=c0_start)
                    lm.A_phys_ref = lm.A_phys_ref or ref["A"]
                    lm.V_phys_ref = lm.V_phys_ref or ref["V"]
            out["landmarks"] = lm
        elif _need("growth_track"):
            raise FileNotFoundError(f"growth_track manifest entry missing: {gt_path}")
    elif _need("growth_track"):
        raise KeyError("growth_track not in cache")

    if "pear_stage2" in wf:
        s2_path = cache_dir / wf["pear_stage2"]
        if s2_path.exists():
            out["stage2_shapes"] = load_solution_list(s2_path)
        elif _need("pear_stage2"):
            raise FileNotFoundError(f"pear_stage2 manifest entry missing: {s2_path}")
    elif _need("pear_stage2"):
        raise KeyError("pear_stage2 not in cache")

    if "pear_stage25" in wf:
        s25_path = cache_dir / wf["pear_stage25"]
        if s25_path.exists():
            out["stage25"] = load_solution(s25_path)
        elif _need("pear_stage25"):
            raise FileNotFoundError(f"pear_stage25 manifest entry missing: {s25_path}")
    elif _need("pear_stage25"):
        raise KeyError("pear_stage25 not in cache")

    if "pear_stage3" in wf:
        s3_path = cache_dir / wf["pear_stage3"]
        if s3_path.exists():
            out["stage3"] = load_solution(s3_path)
        elif _need("pear_stage3"):
            raise FileNotFoundError(f"pear_stage3 manifest entry missing: {s3_path}")
    elif _need("pear_stage3"):
        raise KeyError("pear_stage3 not in cache")

    if "pear_stage4" in wf:
        s4_path = cache_dir / wf["pear_stage4"]
        if s4_path.exists():
            out["stage4"] = load_solution(s4_path)
        elif _need("pear_stage4"):
            raise FileNotFoundError(f"pear_stage4 manifest entry missing: {s4_path}")
    elif _need("pear_stage4"):
        raise KeyError("pear_stage4 not in cache")

    return out


# ===== Workflow steps =====


# --- paths & defaults -------------------------------------------------------

RESULTS = PACKAGE_ROOT / "results"


def cache_dir(eta: float, T_d: float, base: Optional[Path] = None) -> Path:
    if base is None:
        base = DEFAULT_BS_CACHE
    d = Path(base) / f"eta{eta:g}_Td{T_d:g}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(eta: float, base: Optional[Path] = None) -> Path:
    """Output folder for a given η (e.g. ``results/eta1.85``)."""
    if base is None:
        base = RESULTS
    d = Path(base) / f"eta{eta:g}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _targets_match_D(sol: MeridianSolution, D: MeridianSolution, *, rtol: float = 5e-3) -> bool:
    return (
        abs(float(sol.v) - float(D.v)) < rtol
        and abs(float(sol.c0) - float(D.c0)) < rtol * max(abs(float(D.c0)), 1.0)
    )


def _stage25_matches_D(stage25: MeridianSolution, D: MeridianSolution, *, rtol: float = 1e-3) -> bool:
    A_t = float(D.constraints["A_phys"])
    A_s = float(stage25.constraints.get("A_phys", stage25.constraints.get("A_end", float("nan"))))
    return np.isfinite(A_s) and abs(A_s - A_t) / A_t < rtol


def _manifest_path(d: Path) -> Path:
    return d / "workflow.json"


def _read_manifest(d: Path) -> Dict[str, Any]:
    p = _manifest_path(d)
    return json.loads(p.read_text()) if p.exists() else {}


def _write_manifest(d: Path, wf: Dict[str, Any]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    _manifest_path(d).write_text(json.dumps(wf, indent=2))


def _ok(track: Sequence[MeridianSolution]) -> List[MeridianSolution]:
    return [s for s in track if s.success]


# --- cache: prolate B→D, pear D→E, optional full splice -----------------------

def _load_track(d: Path, key: str) -> Optional[List[MeridianSolution]]:
    wf = _read_manifest(d)
    rel = wf.get(key)
    if not rel:
        return None
    path = d / rel
    return load_trajectory(path) if path.exists() else None


def _save_track(
    d: Path,
    key: str,
    track: Sequence[MeridianSolution],
    *,
    name: str,
    manifest_extra: Optional[Dict[str, Any]] = None,
) -> None:
    save_trajectory(d, list(track), name=name)
    wf = _read_manifest(d)
    wf[key] = f"{name}.json"
    if manifest_extra:
        wf.update(manifest_extra)
    _write_manifest(d, wf)


def load_prolate_track(d: Path) -> Tuple[Optional[List[MeridianSolution]], Optional[BSLandmarks]]:
    try:
        wf = load_bs_workflow(d)
    except FileNotFoundError:
        return None, None
    return wf.get("track"), wf.get("landmarks")


def save_prolate_track(
    d: Path,
    track: Sequence[MeridianSolution],
    landmarks: BSLandmarks,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    save_bs_workflow(d, track=track, landmarks=landmarks, meta=meta)


def load_pear_track(d: Path) -> Optional[List[MeridianSolution]]:
    return _load_track(d, "pear_growth_track")


def save_pear_track(d: Path, track: Sequence[MeridianSolution]) -> None:
    _save_track(d, "pear_growth_track", track, name="pear_growth_track")


def load_full_track(d: Path) -> Optional[List[MeridianSolution]]:
    return _load_track(d, "full_track")


def save_full_track(d: Path, track: Sequence[MeridianSolution]) -> None:
    _save_track(d, "full_track", track, name="full_track")


def splice_at_D(
    prolate: Sequence[MeridianSolution],
    pear: Sequence[MeridianSolution],
    D: MeridianSolution,
) -> List[MeridianSolution]:
    t_D = float(D.constraints["t_growth"])
    before = [s for s in _ok(prolate) if float(s.constraints.get("t_growth", 0.0)) <= t_D + 1e-10]
    after = [s for s in pear if s.success and len(s.s) > 0]
    if after and abs(float(after[0].constraints["t_growth"]) - t_D) < 1e-10:
        after = after[1:]
    return before + after


# --- five workflow steps ------------------------------------------------------

def step1_point_b(
    eta: float,
    C0: float,
    *,
    T_d: float = 1.0,
    kappa: float = 1.0,
    c0_start: float = BS_C0_START,
) -> Dict[str, float]:
    """Analytic shooting parameters at landmark B (eq. 9)."""
    return bs_point_b_state(eta, C0, T_d=T_d, kappa=kappa, c0_start=c0_start)


def _pear_seed_for_energy_D(
    cache: Path,
    prolates: Sequence[MeridianSolution],
    *,
    T_d: float,
    C0: float,
    pear_cache: Optional[Path] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> MeridianSolution:
    """Warm pear for the energy-crossing scan (package seed → stage-2.5 at τ≈0.95)."""
    t_lo = float(BS_TAU_D_SEARCH_LO) * float(T_d)
    anchor = next(
        (
            s for s in prolates
            if s.success and float(s.constraints.get("t_growth", -1.0)) >= t_lo - 1e-12
        ),
        None,
    )
    if anchor is None:
        raise RuntimeError("no prolate anchor for pear energy-D seed")

    # Prefer cached stage-3 / stage-2.5 on this η if they already match.
    try:
        pwf = load_bs_workflow(cache)
    except FileNotFoundError:
        pwf = {}
    stage3 = pwf.get("stage3")
    if stage3 is not None and stage3.success and len(stage3.s) > 0 and _targets_match_D(stage3, anchor, rtol=0.02):
        return stage3
    pear25 = pwf.get("stage25")
    if pear25 is not None and _stage25_matches_D(pear25, anchor):
        return pear25

    seed = load_pear_appendix_b_seed()
    if verbose:
        print(f"  pear seed ← {resolve_pear_seed_path().name}", flush=True)
    pear25 = integrate_stage25_pear_at_D(seed, anchor, C0=C0, verbose=verbose)
    if use_cache:
        save_bs_workflow(cache, stage25=pear25)
    return pear25


def step2_prolate_to_D(
    eta: float,
    C0: float,
    st: Dict[str, float],
    *,
    T_d: float = 1.0,
    kappa: float = 1.0,
    dt: float = 0.01,
    v_bump: float = 1e-5,
    prolate_u_bump: float = 3e-5,
    prolate_p_bump: float = 3e-5,
    cache: Path,
    use_cache: bool = True,
    tau_D_search_lo: float = BS_TAU_D_SEARCH_LO,
    tau_D_grow: float = BS_TAU_D_GROW,
    pear_cache: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[List[MeridianSolution], BSLandmarks, MeridianSolution]:
    """Bump onto prolate branch at B; grow past D; pick D by bending-energy crossing.

    Point D is the first τ ≥ ``tau_D_search_lo`` where the pear at the same
    ``(v̄, c₀)`` has lower Helfrich energy than the prolate.  Falls back to the
    ``BS_TAU_D_REF`` proxy if the pear seed / crossing is unavailable.

    ``v_bump`` / ``dt`` should match the Luke leave-sphere
    (``τ_B+dt``, ``v̄=1−v_bump``) so Seifert and Luke share the same physical IC.
    """
    if use_cache:
        track, lm = load_prolate_track(cache)
        if track is not None and lm is not None:
            D = _select_D_from_track(
                track, lm, cache=cache, T_d=T_d, C0=C0,
                pear_cache=pear_cache, use_cache=use_cache,
                tau_D_search_lo=tau_D_search_lo, kappa=kappa,
                verbose=verbose,
            )
            t_D = float(D.constraints["t_growth"])
            track = [
                s for s in track
                if float(s.constraints.get("t_growth", -1.0)) <= t_D + 1e-12
            ]
            lm.D = D
            save_prolate_track(
                cache, track, lm,
                meta={
                    "eta": eta, "C0": C0, "T_d": T_d, "dt": dt,
                    "v_bump": v_bump,
                    "prolate_u_bump": prolate_u_bump, "prolate_p_bump": prolate_p_bump,
                    "tau_D": t_D / float(T_d),
                    "D_by": D.constraints.get("D_by", "fallback"),
                    "from_cache": True,
                },
            )
            return track, lm, D

    tip, _ = solve_bs_point_b(
        eta, C0, T_d=T_d, kappa=kappa, c0_start=BS_C0_START, dt=dt,
        v_bump=v_bump,
        prolate_u_bump=prolate_u_bump, prolate_p_bump=prolate_p_bump,
    )
    if not tip.success:
        raise RuntimeError("prolate departure from B failed")
    track = continue_bozic_svetina(
        tip, eta=eta, T_d=T_d, kappa=kappa, dt=dt,
        t_final=float(tau_D_grow) * T_d, C0_dim=C0, max_dv=0.008,
        prolate_u_bump=prolate_u_bump, prolate_p_bump=prolate_p_bump,
    )
    lm = detect_bs_landmarks(track, started_at_B=True, C0=C0, c0_start=BS_C0_START)
    D = _select_D_from_track(
        track, lm, cache=cache, T_d=T_d, C0=C0,
        pear_cache=pear_cache, use_cache=use_cache,
        tau_D_search_lo=tau_D_search_lo, kappa=kappa,
        verbose=verbose,
    )
    t_D = float(D.constraints["t_growth"])
    track = [
        s for s in track
        if float(s.constraints.get("t_growth", -1.0)) <= t_D + 1e-12
    ]
    lm.D = D
    save_prolate_track(
        cache, track, lm,
        meta={
            "eta": eta, "C0": C0, "T_d": T_d, "dt": dt,
            "v_bump": v_bump,
            "prolate_u_bump": prolate_u_bump, "prolate_p_bump": prolate_p_bump,
            "tau_D": t_D / float(T_d),
            "D_by": D.constraints.get("D_by", "fallback"),
        },
    )
    return track, lm, D


def _select_D_from_track(
    track: Sequence[MeridianSolution],
    lm: BSLandmarks,
    *,
    cache: Path,
    T_d: float,
    C0: float,
    pear_cache: Optional[Path],
    use_cache: bool,
    tau_D_search_lo: float,
    kappa: float,
    verbose: bool,
) -> MeridianSolution:
    """Prefer bending-energy crossing; fall back to τ proxy / asymmetry."""
    if lm.D is not None and lm.D.constraints.get("D_by") == "bending_energy":
        return lm.D
    try:
        pear_seed = _pear_seed_for_energy_D(
            cache, track, T_d=T_d, C0=C0, pear_cache=pear_cache,
            use_cache=use_cache, verbose=verbose,
        )
        D, _pear_at_D, _table = find_D_by_bending_energy(
            track, pear_seed,
            tau_start=tau_D_search_lo, T_d=T_d, C0_dim=C0, kappa=kappa,
            verbose=verbose,
        )
        return D
    except Exception as exc:
        if verbose:
            print(f"  energy-D unavailable ({exc}); using τ proxy", flush=True)
        return pick_bs_d_target(_ok(track), lm, T_d=T_d)


def step3_pear_at_D(
    D: MeridianSolution,
    *,
    cache: Path,
    pear_cache: Optional[Path] = None,
    T_d: float = 1.0,
    C0: float = 1.0,
    use_cache: bool = True,
    verbose: bool = False,
) -> MeridianSolution:
    """Rescale package pear seed to D; shoot along (v̄, c₀) to landmark D."""
    try:
        pwf = load_bs_workflow(cache)
    except FileNotFoundError:
        pwf = {}

    pear25 = pwf.get("stage25") if use_cache else None
    if not (pear25 is not None and _stage25_matches_D(pear25, D)):
        seed = load_pear_appendix_b_seed()
        if verbose:
            print(f"  pear seed ← {resolve_pear_seed_path().name} → D", flush=True)
        pear25 = integrate_stage25_pear_at_D(seed, D, C0=C0, verbose=verbose)
        if use_cache:
            save_bs_workflow(cache, stage25=pear25)

    pear_at_D = pwf.get("stage3") if use_cache else None
    if pear_at_D is not None and not _targets_match_D(pear_at_D, D):
        pear_at_D = None
    if pear_at_D is None:
        pear_at_D = solve_stage3_from_appendix_b(
            pear25, float(D.v), float(D.c0),
            A_target=float(D.constraints["A_phys"]),
            u0_window=0.35, v_step=0.004, min_cv_steps=25, verbose=verbose,
        )
        if use_cache:
            save_bs_workflow(cache, stage3=pear_at_D, stage25=pear25)
    if not pear_at_D.success:
        raise RuntimeError("pear stage-3 match at D failed")
    return pear_at_D


def _pear_seed_at_D(
    pear_at_D: MeridianSolution,
    D: MeridianSolution,
    *,
    C0: float,
    eta: float,
    T_d: float,
    kappa: float,
    L_p: float,
) -> MeridianSolution:
    seed = pear_at_D.copy()
    R_D = float(D.constraints.get("R0", np.sqrt(D.constraints["A_phys"] / (4 * np.pi))))
    seed.constraints.update({
        "R0": R_D,
        "A_phys": float(D.constraints["A_phys"]),
        "V_phys": float(D.constraints["V_phys"]),
        "C0_dimensional": C0,
        "DeltaP": float(pear_at_D.P_bar),
        "t_growth": float(D.constraints["t_growth"]),
        "eta": eta, "T_d": T_d, "L_p": L_p, "kappa": kappa,
        "phase": "shape", "landmark": "D",
    })
    return seed


def step4_pear_to_E(
    pear_at_D: MeridianSolution,
    D: MeridianSolution,
    st: Dict[str, float],
    *,
    eta: float,
    C0: float,
    T_d: float = 1.0,
    kappa: float = 1.0,
    dt: float = 0.01,
    cache: Path,
    use_cache: bool = True,
    max_residual: float = 1e-5,
    max_step_time: float = 30.0,
) -> List[MeridianSolution]:
    """Continue pear-branch growth from D toward τ = 1.

    Stops early (two-sphere handoff) when the shoot residual exceeds
    ``max_residual`` or a step exceeds ``max_step_time`` seconds — the bad
    frame is dropped so the tip is the last clean pear.
    """
    if use_cache:
        cached = load_pear_track(cache)
        if cached is not None:
            return [s for s in cached if s.success and len(s.s) > 0]

    seed = _pear_seed_at_D(
        pear_at_D, D, C0=C0, eta=eta, T_d=T_d, kappa=kappa, L_p=st["L_p"],
    )
    past = continue_bozic_svetina(
        seed, eta=eta, T_d=T_d, kappa=kappa, dt=dt,
        t_final=T_d, C0_dim=C0, max_dv=0.002, n_steps=100,
        u0_window=0.35, lock_branch_steps=5, u0_window_late=2.5,
        max_residual=max_residual, max_step_time=max_step_time,
    )
    past_ok = [s for s in past if s.success and len(s.s) > 0]
    if not past_ok:
        raise RuntimeError("pear continuation D→E produced no shapes")
    tip = past_ok[-1]
    tau_tip = float(tip.constraints.get("t_growth", float("nan"))) / T_d
    if tau_tip < 1.0 - 1e-6:
        print(
            f"  pear cutoff @ τ={tau_tip:.6f}  v̄={tip.v:.6f}  P̄={tip.P_bar:.4f}  "
            f"→ hand off to two-sphere finish",
            flush=True,
        )
    # Always persist so two-sphere finish can find the tip even if use_cache=False.
    save_pear_track(cache, past_ok)
    return past_ok


def step5_full_track(
    prolate: Sequence[MeridianSolution],
    pear: Sequence[MeridianSolution],
    D: MeridianSolution,
    *,
    cache: Path,
    use_cache: bool = True,
) -> List[MeridianSolution]:
    """Splice prolate + pear legs; optionally reuse cached full track."""
    if use_cache:
        cached = load_full_track(cache)
        if cached is not None:
            return cached
    full = splice_at_D(prolate, pear, D)
    # Always persist for downstream two-sphere / plotting tools.
    save_full_track(cache, full)
    return full


def save_growth_plots(
    full: Sequence[MeridianSolution],
    landmarks: BSLandmarks,
    D: MeridianSolution,
    *,
    eta: float,
    T_d: float = 1.0,
    out_dir: Path,
) -> Path:
    """Area/volume vs τ and path in (c₀, v̄); save ``growth_path.png`` under *out_dir*."""
    ok = [s for s in full if s.success]
    A0, V0 = landmarks.A_phys_ref, landmarks.V_phys_ref
    ts = np.array([s.constraints["t_growth"] for s in ok]) / T_d
    As = np.array([s.constraints["A_phys"] / A0 for s in ok])
    Vs = np.array([s.constraints["V_phys"] / V0 for s in ok])
    vs = np.array([s.v for s in ok])
    cs = np.array([s.c0 for s in ok])
    phases = [s.constraints.get("phase") for s in ok]
    C0_CR = c0_critical_sphere(eta)
    pear_D = ok[0]
    for s in ok:
        if abs(float(s.constraints.get("t_growth", 0.0)) - float(D.constraints["t_growth"])) < 1e-8:
            pear_D = s
            break
    pear_E = min(ok, key=lambda s: abs(s.constraints["t_growth"] / T_d - 1.0))

    landmark_lines = (
        ("B", landmarks.B), ("C", landmarks.C), ("D", pear_D), ("E", pear_E),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(ts, As, "o-", ms=3, label=r"$A/A_0$")
    ax.plot(ts, Vs, "s-", ms=3, label=r"$V/V_0$")
    ax.axhline(2.0, color="k", ls="--", lw=0.8, alpha=0.4)
    ax.axhline(np.sqrt(2.0), color="k", ls=":", lw=0.8, alpha=0.4)
    ymax = max(As.max(), Vs.max()) * 1.05
    for name, sol in landmark_lines:
        if sol is None:
            continue
        t = sol.constraints["t_growth"] / T_d
        ax.axvline(t, color="C3", ls="--", lw=0.7, alpha=0.7)
        ax.text(t, ymax, name, ha="center", va="bottom", fontsize=8, color="C3")
    ax.set_xlabel(r"$\tau = t/T_d$")
    ax.set_ylabel("normalized")
    ax.set_title(rf"Growth curves ($\eta={eta:g}$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for phase, marker in (("sphere", "o"), ("shape", "s")):
        m = np.array([p == phase for p in phases])
        if m.any():
            ax.plot(cs[m], vs[m], marker + "-", ms=4, lw=1, label=phase)
    for name, sol in landmark_lines:
        if sol is None:
            continue
        ax.plot(sol.c0, sol.v, "*", ms=12)
        ax.annotate(name, (sol.c0, sol.v), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.axvline(C0_CR, color="C3", ls=":", lw=0.8, alpha=0.6, label=rf"$c_{{0,\mathrm{{cr}}}}={C0_CR:.3f}$")
    ax.axhline(BS_V_TWO_SPHERE, color="0.5", ls=":", lw=0.8, alpha=0.6)
    ax.plot([BS_C0_START, BS_C0_END], [1.0, BS_V_TWO_SPHERE], "k--", lw=0.6, alpha=0.3)
    ax.set_xlabel(r"$c_0 = C_0 R$")
    ax.set_ylabel(r"$\bar v$")
    ax.set_title(r"Path in $(c_0, \bar v)$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.65, 1.02)

    fig.tight_layout()
    out_path = Path(out_dir) / "growth_path.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ===== Two-sphere finish + full assembly =====


ETA = 1.85

C0, T_D, KAPPA = 1.0, 1.0, 1.0
DT = 0.00025
P_FIT_WINDOW = 0.02  # use pear P̄ on [τ_fail − window, τ_fail]
N_SPHERE = 80  # analytic sphere samples on [0, τ_B]
CACHE = cache_dir(ETA, T_D)
OUT = results_dir(ETA) / "two_sphere_finish"
TWO_SPHERE_JSON = OUT / "two_sphere_tau_fail_to_E.json"
FULL_JSON = OUT / "full_trajectory.json"


def configure(*, eta: float = ETA, T_d: float = T_D) -> None:
    """Point module-level paths at ``eta`` / ``T_d`` (for CLI / callers)."""
    global ETA, T_D, CACHE, OUT, TWO_SPHERE_JSON, FULL_JSON
    ETA = float(eta)
    T_D = float(T_d)
    CACHE = cache_dir(ETA, T_D)
    OUT = results_dir(ETA) / "two_sphere_finish"
    TWO_SPHERE_JSON = OUT / "two_sphere_tau_fail_to_E.json"
    FULL_JSON = OUT / "full_trajectory.json"


# ---------------------------------------------------------------------------
# Seifert track loading
# ---------------------------------------------------------------------------

def _seifert_tau(sol: MeridianSolution, *, T_d: float = T_D) -> float:
    return float(sol.constraints["t_growth"]) / float(T_d)


def _ok_seifert(track: Sequence[MeridianSolution]) -> List[MeridianSolution]:
    return [s for s in track if s.success and len(s.s) > 0]


def load_seifert_context(
    cache: Optional[Path] = None,
    *,
    T_d: float = T_D,
    results_shapes: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load Seifert prolate/pear/full tracks; return tip, τ_D, τ_fail, B→fail leg.

    Falls back to ``results/eta{η}/shapes/trajectory.json`` when pear/full
    cache keys are missing (e.g. step4/5 run with ``use_cache=False``).
    """
    if cache is None:
        cache = CACHE

    prolate, lm = load_prolate_track(cache)
    pear = load_pear_track(cache)
    full = load_full_track(cache)

    if full is None:
        shape_path = (
            Path(results_shapes)
            if results_shapes is not None
            else results_dir(ETA) / "shapes" / "trajectory.json"
        )
        if shape_path.is_file():
            full = load_trajectory(shape_path)

    D: Optional[MeridianSolution] = None
    tau_D = float(BS_TAU_D_REF)
    prolate_ok = _ok_seifert(prolate) if prolate else []
    if prolate_ok and lm is not None:
        D = pick_bs_d_target(prolate_ok, lm, T_d=T_d)
        tau_D = _seifert_tau(D, T_d=T_d)
    elif full is not None and lm is None:
        # Infer τ_D from first asymmetric frame in the saved full track.
        for s in _ok_seifert(full):
            asym = abs(s.U0 - s.U1) / max(abs(s.U0), abs(s.U1), 0.05)
            if asym > 0.05:
                tau_D = _seifert_tau(s, T_d=T_d)
                D = s
                break

    pear_ok = _ok_seifert(pear) if pear else []
    if not pear_ok and full is not None:
        pear_ok = [
            s for s in _ok_seifert(full)
            if _seifert_tau(s, T_d=T_d) > tau_D - 1e-8
        ]
        # Persist for next time.
        if pear_ok:
            try:
                save_pear_track(cache, pear_ok)
                save_full_track(cache, _ok_seifert(full))
            except Exception:
                pass

    if not pear_ok and prolate_ok and pear is not None:
        pear_raw = _ok_seifert(pear)
        if pear_raw and D is not None:
            spliced = splice_at_D(prolate_ok, pear_raw, D)
            pear_ok = [
                s for s in spliced
                if _seifert_tau(s, T_d=T_d) > tau_D + 1e-10
            ]

    if not pear_ok:
        raise RuntimeError(
            f"no successful Seifert pear frames under {cache} "
            f"(also checked results shapes). Run run_trajectory.py first."
        )

    tip = pear_ok[-1]
    tau_fail = _seifert_tau(tip, T_d=T_d)

    if full is not None:
        seifert_tr = [
            s for s in _ok_seifert(full)
            if _seifert_tau(s, T_d=T_d) <= tau_fail + 1e-12
        ]
    elif prolate_ok:
        if D is not None:
            t_D = float(D.constraints["t_growth"])
            before = [
                s for s in prolate_ok
                if float(s.constraints.get("t_growth", 0.0)) <= t_D + 1e-10
            ]
            after = [s for s in pear_ok if _seifert_tau(s, T_d=T_d) > tau_D + 1e-10]
            if after and abs(float(after[0].constraints["t_growth"]) - t_D) < 1e-10:
                after = after[1:]
            seifert_tr = before + after
        else:
            seifert_tr = prolate_ok + pear_ok
    else:
        seifert_tr = list(pear_ok)

    if not seifert_tr:
        raise RuntimeError(f"empty Seifert B→τ_fail track from {cache}")

    return {
        "prolate": prolate_ok,
        "pear": pear_ok,
        "full": full,
        "D": D,
        "tip": tip,
        "tau_D": float(tau_D),
        "tau_fail": float(tau_fail),
        "seifert_tr": seifert_tr,
    }


def _match_sol_by_tau(
    sols: Sequence[MeridianSolution],
    tau: float,
    *,
    T_d: float = T_D,
    tol: float = 5e-5,
) -> Optional[MeridianSolution]:
    best: Optional[MeridianSolution] = None
    best_dt = float("inf")
    for sol in sols:
        dt = abs(_seifert_tau(sol, T_d=T_d) - float(tau))
        if dt < best_dt:
            best_dt = dt
            best = sol
    return best if best_dt <= tol else None


def attach_seifert_sols(
    frames: Sequence[TrajFrame],
    cache: Optional[Path] = None,
    *,
    T_d: float = T_D,
) -> None:
    """Attach ``MeridianSolution`` profiles for movie/plot-only reload."""
    try:
        ctx = load_seifert_context(cache if cache is not None else CACHE, T_d=T_d)
    except RuntimeError:
        return
    sols = ctx["seifert_tr"]
    for fr in frames:
        if fr.phase not in ("prolate", "pear"):
            continue
        sol = _match_sol_by_tau(sols, fr.tau, T_d=T_d)
        if sol is not None:
            fr.sol = sol


# ---------------------------------------------------------------------------
# Two-sphere continuation
# ---------------------------------------------------------------------------

def fit_P_bar_linear(
    taus: Sequence[float],
    P_bars: Sequence[float],
    *,
    tau_lo: float,
    tau_hi: float,
) -> Tuple[Callable[[float], float], Dict[str, float]]:
    """Least-squares ``P̄(τ) = a + b τ`` on samples in ``[τ_lo, τ_hi]``."""
    t = np.asarray(taus, dtype=float)
    p = np.asarray(P_bars, dtype=float)
    m = (t >= tau_lo - 1e-15) & (t <= tau_hi + 1e-15) & np.isfinite(p)
    if int(np.count_nonzero(m)) < 2:
        raise ValueError(
            f"need ≥2 P̄ samples in [{tau_lo:.4f},{tau_hi:.4f}] "
            f"(got {int(np.count_nonzero(m))})"
        )
    tt, pp = t[m], p[m]
    b, a = np.polyfit(tt, pp, 1)  # P = b*τ + a

    def P_of_tau(tau: float) -> float:
        return float(a + b * float(tau))

    meta = {
        "a": float(a),
        "b": float(b),
        "tau_lo": float(tt.min()),
        "tau_hi": float(tt.max()),
        "n": int(tt.size),
        "P_lo": float(P_of_tau(tt.min())),
        "P_hi": float(P_of_tau(tt.max())),
        "P_fail": float(P_of_tau(tau_hi)),
        "P_E": float(P_of_tau(1.0)),
    }
    return P_of_tau, meta


def two_sphere_meridian(R1: float, R2: float, *, n_per: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """Side-view meridian: two circles stacked on ``z``, touching at the neck.

    Continuous walk: north pole → neck → south pole (no jump).
    ``R1`` = north lobe, ``R2`` = south lobe (not sorted by size).
    """
    R1, R2 = float(R1), float(R2)
    th1 = np.linspace(0.0, np.pi, n_per)
    # North lobe, center at +R1: th=0 north pole (z=2R1), th=π neck (z=0).
    x1 = R1 * np.sin(th1)
    z1 = R1 * np.cos(th1) + R1
    th2 = np.linspace(0.0, np.pi, n_per)
    # South lobe, center at -R2: th=0 neck (z=0), th=π south pole (z=-2R2).
    x2 = R2 * np.sin(th2)
    z2 = R2 * np.cos(th2) - R2
    x = np.concatenate([x1, x2[1:]])
    z = np.concatenate([z1, z2[1:]])
    z = z - 0.5 * (z.max() + z.min())
    return x, z


def orient_two_sphere_radii(
    R_a: float,
    R_b: float,
    *,
    U0: Optional[float] = None,
    U1: Optional[float] = None,
    north_larger: Optional[bool] = None,
) -> Tuple[float, float]:
    """Order ``(R_north, R_south)`` to match pear pole orientation.

    ``two_sphere_radii_from_AV`` returns ``R_big ≥ R_small``.  Pear tips at
    η=1.85 have the *smaller* lobe at the north pole (``|U0| > |U1|``), so we
    flip unless ``north_larger`` / tip curvatures say otherwise.
    """
    ra, rb = float(R_a), float(R_b)
    r_big, r_small = (ra, rb) if ra >= rb else (rb, ra)
    if north_larger is None:
        if U0 is not None and U1 is not None and abs(U0) > 1e-14 and abs(U1) > 1e-14:
            # Larger |U| ⇒ smaller radius at that pole.
            north_larger = abs(float(U0)) < abs(float(U1))
        else:
            north_larger = True  # legacy: R1 ≥ R2
    if north_larger:
        return r_big, r_small
    return r_small, r_big


def sphere_meridian(R: float, *, n: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0.0, np.pi, n)
    x = float(R) * np.sin(th)
    z = float(R) * np.cos(th)
    return x, z


def make_two_sphere_frame(
    *,
    A: float,
    V: float,
    t_growth: float,
    P_bar: float,
    C0_dim: float = C0,
    T_d: float = T_D,
    L_p: float,
    eta: Optional[float] = None,
    kappa: float = KAPPA,
    message: str = "two-sphere",
    north_larger: Optional[bool] = None,
    tip_U0: Optional[float] = None,
    tip_U1: Optional[float] = None,
) -> MeridianSolution:
    """Build a lightweight ``MeridianSolution`` from two-sphere radii."""
    if eta is None:
        eta = ETA
    radii = two_sphere_radii_from_AV(A, V)
    if radii is None:
        raise ValueError(f"two-sphere radii failed at A={A:.4f} V={V:.4f}")
    R1, R2 = orient_two_sphere_radii(
        radii[0], radii[1],
        U0=tip_U0, U1=tip_U1, north_larger=north_larger,
    )
    U0, U1 = 1.0 / R1, 1.0 / R2
    v = reduced_volume_from_AV(A, V)
    c0 = c0_reduced(C0_dim, A)
    X, Z = two_sphere_meridian(R1, R2)
    n = X.size
    y = np.zeros((7, n), dtype=float)
    y[0] = X
    y[1] = Z
    y[3, 0] = U0
    y[3, -1] = U1
    y[5, -1] = A
    y[6, -1] = V
    s = np.linspace(0.0, float(np.pi * (R1 + R2)), n)
    return MeridianSolution(
        v=float(v),
        c0=float(c0),
        U0=float(U0),
        U1=float(U1),
        sigma_bar=float("nan"),
        P_bar=float(P_bar),
        s=s,
        y=y,
        branch="two_sphere",
        success=True,
        message=message,
        constraints={
            "A_phys": float(A),
            "V_phys": float(V),
            "t_growth": float(t_growth),
            "R1": float(R1),
            "R2": float(R2),
            "C0_dimensional": float(C0_dim),
            "T_d": float(T_d),
            "L_p": float(L_p),
            "eta": float(eta),
            "kappa": float(kappa),
            "phase": "two_sphere",
            "DeltaP": float(P_bar),
            "north_larger": bool(R1 >= R2),
        },
        S1=float(np.pi * R1),
        S_bar=float(np.pi * R1),
    )


def continue_two_sphere_to_E(
    *,
    A0: float,
    V0: float,
    t0: float,
    P_of_tau: Callable[[float], float],
    T_d: float = T_D,
    C0_dim: float = C0,
    eta: Optional[float] = None,
    kappa: float = KAPPA,
    dt: float = DT,
    tau_stop: float = 1.0,
    verbose: bool = True,
    tip_U0: Optional[float] = None,
    tip_U1: Optional[float] = None,
) -> List[MeridianSolution]:
    """Advance ``(A,V)`` with BS ODEs + linear ``P̄(τ)``; shapes are two-spheres."""
    if eta is None:
        eta = ETA
    L_p = float(L_p_from_eta(eta, T_d, C0_dim, kappa))
    alpha = alpha_from_T_d(T_d)
    # Lock lobe orientation to the pear tip for the whole finish leg.
    if tip_U0 is not None and tip_U1 is not None:
        north_larger = abs(float(tip_U0)) < abs(float(tip_U1))
    else:
        north_larger = None
    A, V, t = float(A0), float(V0), float(t0)
    track: List[MeridianSolution] = [
        make_two_sphere_frame(
            A=A, V=V, t_growth=t, P_bar=P_of_tau(t / T_d),
            C0_dim=C0_dim, T_d=T_d, L_p=L_p, eta=eta, kappa=kappa,
            message="two-sphere @ τ_fail",
            north_larger=north_larger,
            tip_U0=tip_U0, tip_U1=tip_U1,
        )
    ]
    if verbose:
        fr0 = track[0]
        print(
            f"  two-sphere start: τ={t/T_d:.6f} A={A:.4f} V={V:.4f} v̄={fr0.v:.6f}  "
            f"R1(N)={fr0.constraints['R1']:.4f} R2(S)={fr0.constraints['R2']:.4f}  "
            f"U0={fr0.U0:.4f} U1={fr0.U1:.4f}  P̄={fr0.P_bar:.4f}",
            flush=True,
        )

    i = 0
    while t / T_d < tau_stop - 1e-12:
        dt_cur = min(dt, tau_stop * T_d - t)
        tau_mid = (t + 0.5 * dt_cur) / T_d
        P = float(P_of_tau(tau_mid))
        A_new, V_new = bs_growth_step(A, V, dt_cur, alpha=alpha, L_p=L_p, P=P)
        t_new = t + dt_cur
        v_new = reduced_volume_from_AV(A_new, V_new)
        hit_E = v_new <= BS_V_TWO_SPHERE + 1e-12
        if hit_E:
            R0 = (A_new / (4.0 * np.pi)) ** 0.5
            V_E = (4.0 * np.pi / 3.0) * BS_V_TWO_SPHERE * R0 ** 3
            V_new = float(V_E)
        try:
            fr = make_two_sphere_frame(
                A=A_new, V=V_new, t_growth=t_new, P_bar=P_of_tau(t_new / T_d),
                C0_dim=C0_dim, T_d=T_d, L_p=L_p, eta=eta, kappa=kappa,
                message=(
                    "two-sphere @ E (clamped)"
                    if hit_E
                    else f"two-sphere step {i+1}"
                ),
                north_larger=north_larger,
                tip_U0=tip_U0, tip_U1=tip_U1,
            )
        except ValueError as exc:
            if verbose:
                print(f"  stop: {exc} at τ→{t_new/T_d:.6f}", flush=True)
            break
        track.append(fr)
        A, V, t = A_new, V_new, t_new
        i += 1
        if verbose and (i % 20 == 0 or t / T_d > tau_stop - 0.002 or hit_E):
            print(
                f"  [{i}] τ={t/T_d:.6f} A={A:.4f} v̄={fr.v:.6f}  "
                f"R1(N)={fr.constraints['R1']:.4f} R2(S)={fr.constraints['R2']:.4f}  "
                f"P̄={fr.P_bar:.4f} U0={fr.U0:.4f} U1={fr.U1:.4f}",
                flush=True,
            )
        if hit_E:
            if verbose:
                print(
                    f"  reached equal-sphere limit v̄={BS_V_TWO_SPHERE:.6f} "
                    f"at τ={t/T_d:.6f} (target τ=1); stopping.",
                    flush=True,
                )
            break
    return track


# ---------------------------------------------------------------------------
# Full τ=0→1 assembly
# ---------------------------------------------------------------------------

@dataclass
class TrajFrame:
    """One frame of the assembled full growth trajectory."""
    tau: float
    A: float
    V: float
    v: float
    c0: float
    P: float
    phase: str  # sphere | prolate | pear | two_sphere
    R1: Optional[float] = None
    R2: Optional[float] = None
    message: str = ""
    sol: Optional[MeridianSolution] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("sol", None)
        return d


def build_sphere_A_to_B(
    *,
    eta: Optional[float] = None,
    C0_dim: float = C0,
    T_d: float = T_D,
    kappa: float = KAPPA,
    n: int = N_SPHERE,
) -> List[TrajFrame]:
    """Analytic spherical growth from point A (τ=0) to B."""
    if eta is None:
        eta = ETA
    ref = bs_point_a_ref(C0_dim, c0_start=BS_C0_START)
    A0 = float(ref["A"])
    L_p = float(L_p_from_eta(eta, T_d, C0_dim, kappa))
    tau_B = float(tau_B_from_eta(eta, c0_start=BS_C0_START))
    taus = np.linspace(0.0, tau_B, max(2, int(n)))
    out: List[TrajFrame] = []
    for tau in taus:
        A = A0 * (2.0 ** float(tau))
        R = area_radius(A)
        V = (4.0 * np.pi / 3.0) * R ** 3  # v̄ = 1
        P = float(sphere_bs_DeltaP(R, T_d, L_p))
        out.append(TrajFrame(
            tau=float(tau), A=float(A), V=float(V), v=1.0,
            c0=float(c0_reduced(C0_dim, A)), P=P, phase="sphere",
            R1=float(R), R2=float(R),
            message="analytic sphere",
        ))
    return out


def seifert_to_traj(
    frames: Sequence[MeridianSolution],
    *,
    tau_D: float,
    tau_hi: Optional[float] = None,
    T_d: float = T_D,
) -> List[TrajFrame]:
    out: List[TrajFrame] = []
    for sol in frames:
        tau = _seifert_tau(sol, T_d=T_d)
        if tau_hi is not None and tau > float(tau_hi) + 1e-12:
            continue
        A = float(sol.constraints.get("A_phys", sol.y[5, -1]))
        V = float(sol.constraints.get("V_phys", sol.y[6, -1]))
        phase = "prolate" if tau <= float(tau_D) + 1e-10 else "pear"
        out.append(TrajFrame(
            tau=float(tau), A=A, V=V,
            v=float(sol.v), c0=float(sol.c0), P=float(sol.P_bar),
            phase=phase, message=sol.message, sol=sol,
        ))
    return out


def two_sphere_rows_to_traj(rows: Sequence[Dict[str, Any]]) -> List[TrajFrame]:
    out: List[TrajFrame] = []
    for r in rows:
        out.append(TrajFrame(
            tau=float(r["tau"]), A=float(r["A"]), V=float(r["V"]),
            v=float(r["v"]), c0=float(r["c0"]), P=float(r["P_bar"]),
            phase="two_sphere",
            R1=float(r["R1"]) if r.get("R1") is not None else None,
            R2=float(r["R2"]) if r.get("R2") is not None else None,
            message=str(r.get("message", "two-sphere")),
        ))
    return out


def stitch_traj(
    *tracks: Sequence[TrajFrame],
    tau_eps: float = 1e-6,
) -> List[TrajFrame]:
    out: List[TrajFrame] = []
    for tr in tracks:
        for fr in tr:
            if out and abs(fr.tau - out[-1].tau) <= tau_eps:
                out[-1] = fr
                continue
            if out and fr.tau + tau_eps < out[-1].tau:
                continue
            out.append(fr)
    return out


def assemble_full_trajectory(
    two_sphere_rows: Sequence[Dict[str, Any]],
    *,
    cache: Optional[Path] = None,
    tau_fail: Optional[float] = None,
    tau_D: Optional[float] = None,
) -> List[TrajFrame]:
    """Sphere A→B + Seifert B→τ_fail + two-sphere finish."""
    ctx = load_seifert_context(cache if cache is not None else CACHE)
    if tau_fail is None:
        tau_fail = float(ctx["tau_fail"])
    if tau_D is None:
        tau_D = float(ctx["tau_D"])
    sphere = build_sphere_A_to_B()
    seifert_tr = seifert_to_traj(
        ctx["seifert_tr"], tau_D=tau_D, tau_hi=tau_fail,
    )
    ts = two_sphere_rows_to_traj(two_sphere_rows)
    return stitch_traj(sphere, seifert_tr, ts)


def print_B_handoff(full: Sequence[TrajFrame]) -> None:
    """Print ΔP and Δv between the sphere tip and the first non-sphere frame."""
    sphere = [f for f in full if f.phase == "sphere"]
    non_sphere = [f for f in full if f.phase != "sphere"]
    if not sphere or not non_sphere:
        print("  B handoff: (missing sphere or Seifert leg)", flush=True)
        return
    s_tip = sphere[-1]
    ns0 = non_sphere[0]
    dP = float(ns0.P - s_tip.P)
    dv = float(ns0.v - s_tip.v)
    print(
        f"  B handoff: sphere tip τ={s_tip.tau:.6f}  "
        f"→ first {ns0.phase} τ={ns0.tau:.6f}  "
        f"ΔP={dP:+.6f}  Δv={dv:+.6e}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Persist / load two-sphere leg
# ---------------------------------------------------------------------------

def compute_two_sphere_leg(
    *,
    cache: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    ctx = load_seifert_context(cache if cache is not None else CACHE)
    pear_ok = ctx["pear"]
    tip: MeridianSolution = ctx["tip"]
    tau_fail = float(ctx["tau_fail"])

    if verbose:
        print(
            f"1) τ_fail tip (Seifert): τ={tau_fail:.6f} A={tip.constraints['A_phys']:.4f}  "
            f"V={tip.constraints['V_phys']:.4f}  v̄={tip.v:.6f} P̄={tip.P_bar:.4f}",
            flush=True,
        )

    P_of_tau, meta = fit_P_bar_linear(
        [_seifert_tau(fr) for fr in pear_ok],
        [fr.P_bar for fr in pear_ok],
        tau_lo=tau_fail - P_FIT_WINDOW,
        tau_hi=tau_fail,
    )
    meta["tau_fail"] = tau_fail
    meta["tau_D"] = float(ctx["tau_D"])

    if verbose:
        print(
            f"2) P̄(τ)= {meta['a']:.6f} + {meta['b']:.6f}·τ   "
            f"(n={meta['n']} on [{meta['tau_lo']:.4f},{meta['tau_hi']:.4f}])  "
            f"P̄(fail)={meta['P_fail']:.4f}  P̄(1)={meta['P_E']:.4f}",
            flush=True,
        )
        A_tip = float(tip.constraints["A_phys"])
        V_tip = float(tip.constraints["V_phys"])
        R = two_sphere_radii_from_AV(A_tip, V_tip)
        if R is not None:
            Rn, Rs = orient_two_sphere_radii(R[0], R[1], U0=tip.U0, U1=tip.U1)
            print(
                f"3) two-sphere @ fail: R1(N)={Rn:.4f} R2(S)={Rs:.4f}  "
                f"(U0={1/Rn:.4f} U1={1/Rs:.4f}; pear tip U0={tip.U0:.4f} U1={tip.U1:.4f})",
                flush=True,
            )

    track = continue_two_sphere_to_E(
        A0=float(tip.constraints["A_phys"]),
        V0=float(tip.constraints["V_phys"]),
        t0=float(tip.constraints["t_growth"]),
        P_of_tau=P_of_tau,
        dt=DT,
        verbose=verbose,
        tip_U0=float(tip.U0),
        tip_U1=float(tip.U1),
    )
    rows = []
    for fr in track:
        rows.append({
            "tau": fr.constraints["t_growth"] / T_D,
            "t_growth": fr.constraints["t_growth"],
            "A": fr.constraints["A_phys"],
            "V": fr.constraints["V_phys"],
            "v": fr.v,
            "c0": fr.c0,
            "R1": fr.constraints["R1"],
            "R2": fr.constraints["R2"],
            "U0": fr.U0,
            "U1": fr.U1,
            "P_bar": fr.P_bar,
            "message": fr.message,
        })
    if verbose:
        print(
            f"4) two-sphere track: {len(rows)} frames  "
            f"τ={rows[0]['tau']:.6f}→{rows[-1]['tau']:.6f}  "
            f"v̄={rows[0]['v']:.6f}→{rows[-1]['v']:.6f}",
            flush=True,
        )
    return rows, meta


def save_two_sphere_json(rows: Sequence[Dict[str, Any]], meta: Dict[str, float]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    TWO_SPHERE_JSON.write_text(json.dumps({"P_fit": meta, "frames": list(rows)}, indent=2))
    return TWO_SPHERE_JSON


def load_two_sphere_json(path: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    raw = json.loads(Path(path if path is not None else TWO_SPHERE_JSON).read_text())
    return list(raw["frames"]), dict(raw.get("P_fit", {}))


def ensure_two_sphere_leg(
    *,
    recompute: bool = False,
    cache: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if (not recompute) and TWO_SPHERE_JSON.is_file():
        rows, meta = load_two_sphere_json()
        if verbose:
            print(
                f"  loaded two-sphere cache → {TWO_SPHERE_JSON}  "
                f"({len(rows)} frames, τ={rows[0]['tau']:.6f}→{rows[-1]['tau']:.6f})",
                flush=True,
            )
        return rows, meta
    rows, meta = compute_two_sphere_leg(cache=cache, verbose=verbose)
    save_two_sphere_json(rows, meta)
    if verbose:
        print(f"saved {TWO_SPHERE_JSON}", flush=True)
    return rows, meta


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    "sphere": "#2e7d32",
    "prolate": "#1f4e79",
    "pear": "#8b3a3a",
    "two_sphere": "#c45c26",
}


def _phase_segments(full: Sequence[TrajFrame]) -> List[Tuple[str, List[TrajFrame]]]:
    if not full:
        return []
    segs: List[Tuple[str, List[TrajFrame]]] = []
    cur_phase = full[0].phase
    cur: List[TrajFrame] = [full[0]]
    for fr in full[1:]:
        if fr.phase != cur_phase:
            segs.append((cur_phase, cur))
            cur = [cur[-1], fr]
            cur_phase = fr.phase
        else:
            cur.append(fr)
    segs.append((cur_phase, cur))
    return segs


def plot_full_trajectory(
    full: Sequence[TrajFrame],
    out_dir: Path,
    *,
    eta: Optional[float] = None,
    C0_dim: float = C0,
    tau_D: Optional[float] = None,
    tau_fail: Optional[float] = None,
    stem: str = "full_trajectory",
) -> Path:
    """``(c₀,v̄)``, pressure, and doubling for the assembled τ=0→1 track."""
    if eta is None:
        eta = ETA
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = bs_point_a_ref(C0_dim, c0_start=BS_C0_START)
    A0, V0 = float(ref["A"]), float(ref["V"])
    c0_cr = float(c0_critical_sphere(eta))
    tau_B = float(tau_B_from_eta(eta))

    if tau_D is None:
        try:
            tau_D = float(load_seifert_context()["tau_D"])
        except RuntimeError:
            tau_D = float(BS_TAU_D_REF)
    if tau_fail is None:
        two_ph = [f for f in full if f.phase == "two_sphere"]
        pear_ph = [f for f in full if f.phase == "pear"]
        if two_ph and pear_ph:
            tau_fail = float(pear_ph[-1].tau)
        elif two_ph:
            tau_fail = float(two_ph[0].tau)
        else:
            tau_fail = float(full[-1].tau)

    ts = np.array([fr.tau for fr in full])
    As = np.array([fr.A / A0 for fr in full])
    Vs = np.array([fr.V / V0 for fr in full])
    segs = _phase_segments(full)
    marks = [
        ("A", full[0]),
        ("B", min(full, key=lambda f: abs(f.tau - tau_B))),
        ("D", min(full, key=lambda f: abs(f.tau - tau_D))),
        ("fail", min(full, key=lambda f: abs(f.tau - tau_fail))),
        ("E", full[-1]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))

    def _phase_lines(ax, xattr: str, yattr: str) -> None:
        for phase, seg in segs:
            ax.plot(
                [getattr(f, xattr) for f in seg],
                [getattr(f, yattr) for f in seg],
                "-", color=PHASE_COLORS.get(phase, "k"), lw=2.0,
                label=phase.replace("_", "-"),
            )

    ax = axes[0, 0]
    _phase_lines(ax, "tau", "c0")
    ax.axhline(c0_cr, color="0.45", ls=":", lw=0.9, alpha=0.7)
    ax.axhline(BS_C0_END, color="0.45", ls="--", lw=0.8, alpha=0.5)
    for tmark in (tau_B, tau_D, tau_fail):
        ax.axvline(tmark, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$c_0$"); ax.set_title(r"$c_0(\tau)$")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3); ax.set_xlim(0.0, 1.0)

    ax = axes[0, 1]
    _phase_lines(ax, "tau", "v")
    ax.axhline(BS_V_TWO_SPHERE, color="0.45", ls=":", lw=0.9, alpha=0.7,
               label=rf"$2^{{-1/2}}$")
    for tmark in (tau_B, tau_D, tau_fail):
        ax.axvline(tmark, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$\bar v$"); ax.set_title(r"$\bar v(\tau)$")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0.0, 1.0); ax.set_ylim(0.65, 1.02)

    ax = axes[0, 2]
    _phase_lines(ax, "tau", "P")
    for tmark in (tau_B, tau_D, tau_fail):
        ax.axvline(tmark, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(0.0, color="0.4", ls="--", lw=0.7, alpha=0.5)
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$P$"); ax.set_title(r"Pressure $P(\tau)$")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0.0, 1.0)

    ax = axes[1, 0]
    _phase_lines(ax, "c0", "v")
    ax.plot([BS_C0_START, BS_C0_END], [1.0, BS_V_TWO_SPHERE],
            "k--", lw=0.7, alpha=0.35, label="sphere→E line")
    ax.axvline(c0_cr, color="0.45", ls=":", lw=0.9, alpha=0.7)
    ax.axhline(BS_V_TWO_SPHERE, color="0.45", ls=":", lw=0.9, alpha=0.5)
    for name, fr in marks:
        ax.plot(fr.c0, fr.v, "*", ms=11, color="k", zorder=5)
        ax.annotate(name, (fr.c0, fr.v), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel(r"$c_0$"); ax.set_ylabel(r"$\bar v$")
    ax.set_title(r"Path in $(c_0,\bar v)$")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(0.65, 1.02)

    ax = axes[1, 1]
    ax.plot(ts, As, "-", color="#1f4e79", lw=2.0, label=r"$A/A_0$")
    ax.plot(ts, Vs, "--", color="#1f4e79", lw=1.5, alpha=0.75, label=r"$V/V_0$")
    ax.axhline(2.0, color="k", ls="--", lw=0.9, alpha=0.45)
    ax.axhline(np.sqrt(2.0), color="k", ls=":", lw=0.9, alpha=0.4)
    for tmark, lab in ((tau_B, "B"), (tau_D, "D"), (tau_fail, "fail")):
        ax.axvline(tmark, color="0.5", ls=":", lw=0.8, alpha=0.5)
        ax.text(tmark, 0.02, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="0.35")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("normalized")
    ax.set_title(r"Area / volume doubling")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3); ax.set_xlim(0.0, 1.0)

    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        rf"$\eta={eta:g}$" + "\n\n"
        + f"frames: {len(full)}\n"
        + f"τ: {full[0].tau:.4f} → {full[-1].tau:.4f}\n\n"
        + "phases:\n"
        + "  green  sphere A→B\n"
        + "  blue   Seifert prolate B→D\n"
        + "  maroon Seifert pear D→fail\n"
        + "  orange two-sphere →E\n\n"
        + f"A/A₀(end) = {full[-1].A/A0:.4f}\n"
        + f"V/V₀(end) = {full[-1].V/V0:.4f}\n"
        + f"v̄(end)   = {full[-1].v:.4f}\n"
        + f"c₀(end)   = {full[-1].c0:.4f}\n"
        + f"P(end)    = {full[-1].P:.4f}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f4ef", edgecolor="0.8"))

    fig.suptitle(
        rf"Full BS trajectory $\tau=0\to 1$ "
        rf"(sphere + Seifert + two-sphere, $\eta={eta:g}$)",
        fontsize=13,
    )
    fig.tight_layout()
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def plot_two_sphere_detail(
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    eta: Optional[float] = None,
) -> Path:
    if eta is None:
        eta = ETA
    out_dir = Path(out_dir)
    ref = bs_point_a_ref(C0, c0_start=BS_C0_START)
    A0ref, V0ref = float(ref["A"]), float(ref["V"])
    ts = np.array([r["tau"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(ts, [r["c0"] for r in rows], "-", color="#c45c26", lw=2)
    axes[0, 0].axhline(BS_C0_END, color="0.5", ls=":", lw=0.8)
    axes[0, 0].set_title(r"$c_0(\tau)$ (two-sphere)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ts, [r["v"] for r in rows], "-", color="#c45c26", lw=2)
    axes[0, 1].axhline(BS_V_TWO_SPHERE, color="0.5", ls=":", lw=0.8)
    axes[0, 1].set_title(r"$\bar v(\tau)$")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ts, [r["A"] / A0ref for r in rows], "-", lw=2, label=r"$A/A_0$")
    axes[1, 0].plot(ts, [r["V"] / V0ref for r in rows], "--", lw=1.6, label=r"$V/V_0$")
    axes[1, 0].axhline(2.0, color="k", ls="--", lw=0.8, alpha=0.4)
    axes[1, 0].legend()
    axes[1, 0].set_title("doubling")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ts, [r["R1"] for r in rows], "-", lw=2, label=r"$R_1$ (N)")
    axes[1, 1].plot(ts, [r["R2"] for r in rows], "-", lw=2, label=r"$R_2$ (S)")
    axes[1, 1].plot(ts, [r["P_bar"] for r in rows], ":", lw=1.5, label=r"$P̄$")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title(r"$R_{1,2}$, $P̄$")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(rf"Two-sphere finish $\tau_{{\mathrm{{fail}}}}\to$E ($\eta={eta:g}$)")
    fig.tight_layout()
    png = out_dir / "two_sphere_finish.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Movie
# ---------------------------------------------------------------------------

def _profile_for_frame(fr: TrajFrame) -> Tuple[np.ndarray, np.ndarray]:
    if fr.phase == "sphere":
        return sphere_meridian(area_radius(fr.A))
    if fr.phase == "two_sphere":
        if fr.R1 is not None and fr.R2 is not None:
            return two_sphere_meridian(fr.R1, fr.R2)
        radii = two_sphere_radii_from_AV(fr.A, fr.V)
        if radii is None:
            return np.array([0.0]), np.array([0.0])
        return two_sphere_meridian(radii[0], radii[1])
    if fr.sol is not None and fr.sol.y.shape[1] > 0:
        x = np.asarray(fr.sol.X, dtype=float)
        z = np.asarray(fr.sol.Z, dtype=float)
        z = z - 0.5 * (z.max() + z.min())
        return x, z
    return np.array([0.0]), np.array([0.0])


def movie_full_trajectory(
    full: Sequence[TrajFrame],
    out_dir: Path,
    *,
    eta: Optional[float] = None,
    target_frames: int = 180,
    fps: int = 16,
    dpi: int = 110,
    stem: str = "full_trajectory_shapes",
    cache: Optional[Path] = None,
) -> Path:
    """Animate meridian profiles along the assembled full trajectory."""
    if eta is None:
        eta = ETA
    if cache is None:
        cache = CACHE
    from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not full:
        raise ValueError("empty trajectory")

    frames = list(full)
    attach_seifert_sols(frames, cache)

    n = len(frames)
    if target_frames > 0 and n > target_frames:
        idx = np.linspace(0, n - 1, int(target_frames), dtype=int)
        frames = [frames[i] for i in idx]
        if frames[-1] is not full[-1]:
            frames.append(full[-1])

    print(
        f"  movie: {len(frames)} profiles (from {n} full frames)…",
        flush=True,
    )
    profiles: List[Tuple[np.ndarray, np.ndarray, TrajFrame]] = []
    for i, fr in enumerate(frames):
        r, z = _profile_for_frame(fr)
        profiles.append((r, z, fr))
        if (i + 1) % 40 == 0 or i == 0 or i == len(frames) - 1:
            print(
                f"    [{i+1}/{len(frames)}] τ={fr.tau:.4f}  "
                f"{fr.phase}  v̄={fr.v:.4f}",
                flush=True,
            )

    r_max = max(float(np.max(np.abs(r))) for r, _, _ in profiles if r.size > 1)
    z_min = min(float(z.min()) for _, z, _ in profiles if z.size > 1)
    z_max = max(float(z.max()) for _, z, _ in profiles if z.size > 1)
    pad = 0.08 * max(r_max, z_max - z_min, 1.0)

    fig, ax = plt.subplots(figsize=(5.2, 5.8))
    line_r, = ax.plot([], [], lw=2.1)
    line_l, = ax.plot([], [], lw=2.1)
    subtitle = ax.set_title("")
    ax.set_aspect("equal")
    ax.set_xlim(-r_max - pad, r_max + pad)
    ax.set_ylim(z_min - pad, z_max + pad)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$z$")
    ax.grid(True, alpha=0.25)
    fig.suptitle(rf"Full Seifert trajectory shapes ($\eta={eta:g}$)", fontsize=12)

    def init():
        line_r.set_data([], [])
        line_l.set_data([], [])
        subtitle.set_text("")
        return line_r, line_l, subtitle

    def update(k: int):
        r, z, fr = profiles[k]
        color = PHASE_COLORS.get(fr.phase, "#333333")
        line_r.set_data(r, z)
        line_l.set_data(-r, z)
        line_r.set_color(color)
        line_l.set_color(color)
        subtitle.set_text(
            rf"$\tau={fr.tau:.4f}$  {fr.phase.replace('_', '-')}  "
            rf"$c_0={fr.c0:.3f}$  $\bar v={fr.v:.4f}$"
        )
        return line_r, line_l, subtitle

    anim = FuncAnimation(
        fig, update, init_func=init, frames=len(profiles), blit=True,
    )
    mp4 = out_dir / f"{stem}.mp4"
    try:
        writer = FFMpegWriter(fps=fps)
        anim.save(str(mp4), writer=writer, dpi=dpi)
        out_path = mp4
    except Exception as exc:
        print(f"  ffmpeg failed ({exc}); writing GIF…", flush=True)
        gif = out_dir / f"{stem}.gif"
        anim.save(str(gif), writer=PillowWriter(fps=max(8, fps // 2)), dpi=dpi)
        out_path = gif
    plt.close(fig)
    print(f"saved {out_path}  ({len(profiles)} frames @ {fps} fps)", flush=True)
    return out_path


def save_full_json(
    full: Sequence[TrajFrame],
    meta: Optional[Dict[str, float]] = None,
) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "eta": ETA,
        "tau_fail": float(meta.get("tau_fail", float("nan"))) if meta else None,
        "tau_D": float(meta.get("tau_D", float("nan"))) if meta else None,
        "P_fit": meta or {},
        "frames": [fr.to_dict() for fr in full],
    }
    FULL_JSON.write_text(json.dumps(payload, indent=2))
    return FULL_JSON


def load_full_json(path: Path = FULL_JSON) -> Tuple[List[TrajFrame], Dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    frames = [TrajFrame(**{k: v for k, v in d.items() if k != "sol"}) for d in raw["frames"]]
    meta = dict(raw.get("P_fit", {}))
    if raw.get("tau_fail") is not None:
        meta.setdefault("tau_fail", raw["tau_fail"])
    if raw.get("tau_D") is not None:
        meta.setdefault("tau_D", raw["tau_D"])
    return frames, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Two-sphere finish past τ_fail + full τ=0→1 plots/movie (Seifert)",
    )
    ap.add_argument(
        "--plot-only", action="store_true",
        help="Skip two-sphere recompute; load cache and (re)plot/movie",
    )
    ap.add_argument(
        "--recompute", action="store_true",
        help="Force recompute of the two-sphere leg",
    )
    ap.add_argument("--no-movie", action="store_true", help="Skip shape movie")
    ap.add_argument(
        "--movie-frames", type=int, default=180,
        help="Target number of movie frames (default 180)",
    )
    ap.add_argument(
        "--eta", type=float, default=None,
        help=f"Growth η (default {ETA})",
    )
    args = ap.parse_args(argv)
    if args.eta is not None:
        configure(eta=float(args.eta))

    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only and not TWO_SPHERE_JSON.is_file():
        raise SystemExit(f"--plot-only requires {TWO_SPHERE_JSON}")

    rows, meta = ensure_two_sphere_leg(
        recompute=bool(args.recompute) and not args.plot_only,
        verbose=True,
    )

    tau_fail = float(meta.get("tau_fail", rows[0]["tau"]))
    tau_D = float(meta.get("tau_D", BS_TAU_D_REF))

    if args.plot_only and FULL_JSON.is_file():
        full, full_meta = load_full_json()
        meta = {**meta, **{k: v for k, v in full_meta.items() if k not in meta}}
        tau_fail = float(meta.get("tau_fail", tau_fail))
        tau_D = float(meta.get("tau_D", tau_D))
    else:
        full = assemble_full_trajectory(
            rows, tau_fail=tau_fail, tau_D=tau_D,
        )
        save_full_json(full, meta)

    print_B_handoff(full)

    print(
        f"5) full trajectory: {len(full)} frames  "
        f"τ={full[0].tau:.4f}→{full[-1].tau:.4f}  "
        f"phases={sorted({f.phase for f in full})}",
        flush=True,
    )
    print(f"saved {FULL_JSON}", flush=True)

    png = plot_full_trajectory(
        full, OUT, tau_D=tau_D, tau_fail=tau_fail,
    )
    print(f"saved {png}", flush=True)

    if not args.no_movie:
        movie_full_trajectory(
            full, OUT, target_frames=int(args.movie_frames),
        )

    ref = bs_point_a_ref(C0, c0_start=BS_C0_START)
    A0ref = float(ref["A"])
    print(
        f"6) endpoint: τ={full[-1].tau:.6f}  "
        f"A/A0={full[-1].A/A0ref:.6f}  "
        f"v̄={full[-1].v:.6f} (E {BS_V_TWO_SPHERE:.6f})  "
        f"c0={full[-1].c0:.4f} (E {BS_C0_END:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
