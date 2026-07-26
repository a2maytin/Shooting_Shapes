"""Božič–Svetina growth, landmarks B/D, workflow steps, and two-sphere finish.

Shape equilibria come from ``seifert_functions`` (two-leg Seifert shooting).
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
    _shoot_two_leg,
    area_radius,
    bending_energy,
    c0_reduced,
    integrate_two_leg,
    integrate_pear_at_landmark,
    load_solution,
    load_solution_list,
    load_trajectory,
    reduced_volume_from_AV,
    save_solution_at,
    save_solution_list,
    save_trajectory,
    solve_seifert,
    match_AV_from_seed,
)

from shapes import (
    L_sto_reduced_volume,
    NestedSphere,
    TwoSphere,
    approx_oblate_energy,
    approx_pear_energy_two_sphere,
    approx_prolate_energy,
    nested_sphere_energy,
    nested_sphere_meridian,
    nested_sphere_pressure_from_dEdV,
    nested_sphere_radii_from_AV,
    nested_sphere_radii_from_v,
    nested_sphere_signed_radii_from_AV,
    orient_two_sphere_radii,
    sphere_meridian,
    two_sphere_alpha_from_v,
    two_sphere_meridian,
    two_sphere_radii_from_AV,
    two_sphere_radii_from_v,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Package paths / seed shapes
# ---------------------------------------------------------------------------
SEED_SHAPES_DIR = PACKAGE_ROOT / "seed_shapes"
PEAR_SEED = SEED_SHAPES_DIR / "pear_seed.json"
STOMATOCYTE_SEED = SEED_SHAPES_DIR / "stomatocyte_seed.json"
STOMATOCYTE_UNSTABLE_SEED = SEED_SHAPES_DIR / "stomatocyte_unstable_seed.json"


def resolve_pear_seed_path() -> Path:
    """In-package pear warm-start shape (no cache required)."""
    p = PEAR_SEED
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing pear seed {p}. Expected pear_seed.json in seed_shapes/."
        )
    return p


def load_pear_seed() -> MeridianSolution:
    return load_solution(resolve_pear_seed_path())


def resolve_stomatocyte_seed_path() -> Path:
    """In-package stomatocyte warm-start shape (no cache required)."""
    p = STOMATOCYTE_SEED
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing stomatocyte seed {p}. Expected stomatocyte_seed.json "
            f"in seed_shapes/."
        )
    return p


def load_stomatocyte_seed() -> MeridianSolution:
    return load_solution(resolve_stomatocyte_seed_path())


def resolve_stomatocyte_unstable_seed_path() -> Path:
    """Archived unstable stomatocyte at D_sto (wrong branch; kept for reference)."""
    p = STOMATOCYTE_UNSTABLE_SEED
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing unstable stomatocyte seed {p}. Expected "
            f"stomatocyte_unstable_seed.json in seed_shapes/."
        )
    return p


def load_stomatocyte_unstable_seed() -> MeridianSolution:
    return load_solution(resolve_stomatocyte_unstable_seed_path())


def _continue_report(
    i: int,
    n_tot: int,
    sol: MeridianSolution,
    tag: str,
    dt: float,
) -> None:
    c = sol.constraints
    res = c.get("residual", float("nan"))
    T_d = float(c.get("T_d", 1.0))
    tau = float(c.get("t_growth", float("nan"))) / T_d if T_d > 0 else float("nan")
    dt_step = float(c.get("dt_step", float("nan")))
    dt_nom = float(c.get("dt_nom", float("nan")))
    step_msg = ""
    if np.isfinite(dt_step) and dt_step > 0 and T_d > 0:
        d_tau = dt_step / T_d
        step_msg = f"  Δτ={d_tau:.4g}"
        if np.isfinite(dt_nom) and dt_nom > 0:
            step_msg += f" (dt/dt_nom={dt_step / dt_nom:.3f})"
    tau_msg = f"τ={tau:.4f}  " if np.isfinite(tau) else ""
    v_ode = c.get("v_ode")
    v_msg = f"v̄={sol.v:.4f}"
    if v_ode is not None and abs(float(v_ode) - sol.v) > 5e-4:
        v_msg = f"v̄={sol.v:.4f}(ode={float(v_ode):.4f})"
    print(
        f"  [{i + 1}/{n_tot}] {tau_msg}{v_msg}  c₀={sol.c0:.4f}  {tag}  "
        f"U0={sol.U0:+.4f}  U1={sol.U1:+.4f}  S1={sol.S1:.3f}  "
        f"Σ̄={sol.sigma_bar:.3f}  P̄={sol.P_bar:.3f}  "
        f"|res|={res:.2e}{step_msg}  ({dt:.2f}s)",
        flush=True,
    )


def _track_index(track: Sequence[MeridianSolution], sol: MeridianSolution) -> Optional[int]:
    """Best-effort index of ``sol`` inside ``track`` (for landmark manifests)."""
    for i, s in enumerate(track):
        if s is sol:
            return i
    key = (
        round(float(sol.v), 8),
        round(float(sol.c0), 8),
        round(float(sol.constraints.get("t_growth", float("nan"))), 8),
    )
    for i, s in enumerate(track):
        sk = (
            round(float(s.v), 8),
            round(float(s.c0), 8),
            round(float(s.constraints.get("t_growth", float("nan"))), 8),
        )
        if sk == key:
            return i
    return None


# Plotting: see bs_plotting.plot_seed_shape

# ===== BS physics (from bs_solver) =====

BS_ETA_MIN = 1.85  # self-reproduction: η = T_d L_p κ C₀⁴ ≳ 1.85

BS_V_TWO_SPHERE = float(1.0 / np.sqrt(2.0))  # point E: v̄ = 2^{-1/2}

BS_C0_START = 2.0  # point A: c₀(0) = 2 (sphere of zero bending energy, R = 2/C₀)

BS_C0_START_NEG = -2.0  # point A for C₀ < 0: c₀(0) = −2 (R = c₀/C₀ > 0)

BS_C0_END = float(2.0 * np.sqrt(2.0))  # point E: c₀ = 2√2 when τ = 1

BS_E_STAR_P_BAR = -0.51  # shooting guess at ε-bumped two-sphere E*

BS_E_STAR_SIGMA_BAR = 0.51  # shooting guess at ε-bumped two-sphere E*

BS_TAU_D_REF = 0.955  # fallback τ_D if energy crossing is unavailable
BS_TAU_D_GROW_CEILING = 1.0  # safety ceiling if approx/exact crossing is late


# Geometry / energy helpers live in ``shapes/`` (re-exported above).
# approx_pear_energy_two_sphere, approx_prolate_energy, nested/two-sphere radii, …


def approx_D_radii(
    c0: float,
    v: float,
    *,
    C0: float = 1.0,
    A: Optional[float] = None,
    V: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Two-sphere radii for the pear/stomatocyte energy approx at ``(c₀, v̄)``.

    Prefer physical ``(A, V)`` when available; otherwise use ``R₀ = c₀/C₀``
    (same-sign ``c₀``, ``C₀`` so ``R₀ > 0``).

    For ``C₀>0`` (pear): positive external two-sphere radii.
    For ``C₀<0`` (stomatocyte): nested-sphere radii with the **inner**
    radius signed negative (``R_inner < 0``, ``R_outer > 0``) so
    ``2 − C₀ R`` uses both ``C₀<0`` and the inverted lobe correctly.
    """
    C0 = float(C0)
    c0 = float(c0)
    if C0 < 0.0:
        if A is not None and V is not None:
            radii = nested_sphere_radii_from_AV(float(A), float(V))
        else:
            if C0 == 0.0 or c0 == 0.0 or (c0 * C0) <= 0.0:
                return None
            R0 = c0 / C0
            if R0 <= 0.0:
                return None
            radii = nested_sphere_radii_from_v(float(v), R0=R0)
        if radii is None:
            return None
        r_out, r_in = float(radii[0]), float(radii[1])
        # Outer > 0, inner < 0 (stomatocyte / sphere-in-sphere convention).
        if abs(r_out) < abs(r_in):
            r_out, r_in = r_in, r_out
        return abs(r_out), -abs(r_in)

    if A is not None and V is not None:
        return two_sphere_radii_from_AV(float(A), float(V))
    if C0 == 0.0 or c0 == 0.0 or (c0 * C0) <= 0.0:
        return None
    R0 = c0 / C0
    if R0 <= 0.0:
        return None
    return two_sphere_radii_from_v(float(v), R0=R0)


def approx_D_energies(
    c0: float,
    v: float,
    *,
    C0: float = 1.0,
    A: Optional[float] = None,
    V: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Return ``(E_pear/sto_approx, E_prolate/oblate_approx)`` or ``None``.

    For ``C₀>0`` this is the usual pear↔prolate pair (positive radii).
    For ``C₀<0`` the oblate↔stomatocyte pair keeps **signed** ``c₀``, ``C₀``
    and nested radii with ``R_inner < 0`` (no absolute-value mirror).
    """
    C0 = float(C0)
    c0 = float(c0)
    radii = approx_D_radii(c0, v, C0=C0, A=A, V=V)
    if radii is None:
        return None
    E_pear = approx_pear_energy_two_sphere(C0, radii[0], radii[1])
    E_prol = approx_prolate_energy(c0, v)
    return E_pear, E_prol


# Aliases for the negative-C₀ branch (NearSphere / NestedSphere energies).
approx_stomatocyte_energy_two_sphere = approx_pear_energy_two_sphere


def approx_D_energy_mismatch(
    c0: float,
    v: float,
    *,
    C0: float = 1.0,
    A: Optional[float] = None,
    V: Optional[float] = None,
) -> Optional[float]:
    """Relative mismatch ``δ = (E_pear − E_prolate) / max(|E|)``.

    ``δ > 0`` ⇒ approximate prolate/oblate favored;
    ``δ < 0`` ⇒ approximate pear/stomatocyte favored.
    """
    energies = approx_D_energies(c0, v, C0=C0, A=A, V=V)
    if energies is None:
        return None
    E_pear, E_prol = energies
    denom = max(abs(E_pear), abs(E_prol), 1e-12)
    return float((E_pear - E_prol) / denom)


def annotate_approx_D_energies(
    sol: MeridianSolution,
    *,
    C0: float = 1.0,
) -> Optional[float]:
    """Stamp approximate D energies / mismatch onto ``sol.constraints``; return ``δ``."""
    A = sol.constraints.get("A_phys")
    V = sol.constraints.get("V_phys")
    energies = approx_D_energies(
        float(sol.c0), float(sol.v), C0=C0,
        A=None if A is None else float(A),
        V=None if V is None else float(V),
    )
    if energies is None:
        return None
    E_pear, E_prol = energies
    delta = float((E_pear - E_prol) / max(abs(E_pear), abs(E_prol), 1e-12))
    sol.constraints["E_pear_approx"] = E_pear
    sol.constraints["E_prolate_approx"] = E_prol
    sol.constraints["E_stomatocyte_approx"] = E_pear
    sol.constraints["E_oblate_approx"] = E_prol
    sol.constraints["dE_approx_rel"] = delta
    return delta


def approx_D_boundary_v(
    c0: float,
    *,
    C0: float = 1.0,
    delta_target: float = 0.0,
) -> Optional[float]:
    """Solve for ``v̄`` where the approximate mismatch equals ``delta_target``."""
    c0 = float(c0)
    C0 = float(C0)
    # Pear (external): v ∈ (2^{-1/2}, 1]; stomatocyte (nested): v ∈ (0, 1].
    if C0 < 0.0:
        v_lo = 1e-3
    else:
        v_lo = float(BS_V_TWO_SPHERE) + 1e-6
    v_hi = 1.0 - 1e-4

    def _f(v: float) -> float:
        d = approx_D_energy_mismatch(c0, v, C0=C0)
        if d is None:
            return float("nan")
        return float(d) - float(delta_target)

    grid = np.linspace(v_lo, v_hi, 96 if C0 < 0.0 else 64)
    vals = [_f(float(v)) for v in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        if fa == 0.0:
            return float(a)
        if fa * fb < 0.0:
            try:
                return float(brentq(_f, float(a), float(b)))
            except ValueError:
                continue
    return None


def approx_D_boundary_curve(
    c0_grid: Sequence[float],
    *,
    C0: float = 1.0,
    delta_target: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(c₀, v̄)`` samples of an approximate-energy level set ``δ = delta_target``."""
    cs: List[float] = []
    vs: List[float] = []
    for c0 in c0_grid:
        v = approx_D_boundary_v(float(c0), C0=C0, delta_target=delta_target)
        if v is None:
            continue
        cs.append(float(c0))
        vs.append(float(v))
    return np.asarray(cs, dtype=float), np.asarray(vs, dtype=float)



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
    denom = float(T_d * kappa * (float(C0) ** 4))
    if float(T_d) <= 0.0 or float(kappa) <= 0.0 or float(C0) == 0.0 or denom <= 0.0:
        raise ValueError("T_d and kappa must be positive; C0 must be nonzero")
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

def c0_critical_sphere(eta: float, *, side: str = "+") -> float:
    """Critical reduced ``c₀`` where spherical growth ends (BS eq. 9 at equality).

    Spherical growth requires

        1 − 4 η (6 − c₀) / ((ln 2) c₀⁴) < 0

    so the critical ``c₀,cr(η)`` solves ``(ln 2) c₀⁴ + 4 η c₀ − 24 η = 0``.
    For ``η = 1.85`` one finds ``c₀,cr ≈ +2.475`` on the positive branch.

    ``side``: ``"+"`` (default) searches ``(0, 6)``; ``"-"`` searches ``(-6, 0)``.
    """
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    side = str(side).strip()
    if side not in ("+", "-"):
        raise ValueError("side must be '+' or '-'")
    ln2 = float(np.log(2.0))

    def f(x: float) -> float:
        return ln2 * x ** 4 + 4.0 * eta * x - 24.0 * eta

    if side == "+":
        # Root in (0, 6): f(0) = -24η < 0, f(6) = (ln2)*1296 > 0
        lo, hi = 1e-6, 6.0 - 1e-6
    else:
        # Root in (-6, 0): f(-6) > 0 for typical η, f→-24η as x→0⁻
        lo, hi = -6.0 + 1e-6, -1e-6
        if f(lo) * f(hi) > 0.0:
            # Widen slightly if the default bracket misses (rare / extreme η).
            lo = -8.0
            if f(lo) * f(hi) > 0.0:
                raise RuntimeError(
                    f"no negative c₀,cr bracket for η={eta:g} "
                    f"(f({lo})={f(lo):.4g}, f({hi})={f(hi):.4g})"
                )
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        # Keep the sign convention of the positive branch root finder:
        # move lo when f(mid) has the same sign as f(lo) relative to crossing.
        if side == "+":
            if f(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        else:
            # f(lo)>0, f(hi)<0 typically → root where f crosses +→−
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
    return float(0.5 * (lo + hi))


def c0_critical_side_from_start(c0_start: float) -> str:
    """``'+'`` / ``'-'`` branch for eq.~(9) matching the sign of ``c₀(0)``."""
    if float(c0_start) < 0.0:
        return "-"
    return "+"


def R0_from_c0(C0: float, c0: float = BS_C0_START) -> float:
    """Radius of the equivalent sphere: ``R = c₀ / C₀`` (BS: start at ``c₀ = ±2``)."""
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
    if R <= 0.0:
        raise ValueError(
            f"point A radius must be positive (got R={R} from c0={c0_start}, C0={C0})"
        )
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
    """Reduced time ``t/T_d`` at point B: ``c₀(τ) = c₀(0) 2^{τ/2} = c₀,cr(η)``.

    Works for both signs: ``c₀(0)`` and ``c₀,cr`` must share a sign and
    ``|c₀,cr| > |c₀(0)|``.
    """
    side = c0_critical_side_from_start(c0_start)
    c0_cr = c0_critical_sphere(eta, side=side)
    c0_start = float(c0_start)
    if c0_start == 0.0:
        raise ValueError("c0_start must be nonzero")
    if c0_start * c0_cr <= 0.0:
        raise ValueError(
            f"c0_start and c0_cr must share a sign (got {c0_start}, {c0_cr})"
        )
    ratio = float(c0_cr / c0_start)
    if ratio <= 1.0:
        raise ValueError(
            f"need |c0_cr| > |c0_start| (got c0_start={c0_start}, c0_cr={c0_cr})"
        )
    return float(2.0 * np.log2(ratio))


def bs_point_b_state(
    eta: float,
    C0: float,
    *,
    T_d: float = 1.0,
    kappa: float = 1.0,
    c0_start: float = BS_C0_START,
) -> Dict[str, float]:
    """Analytic Božič–Svetina state at point **B** (eq.~9 equality, ``v̄ = 1``).

    ``c₀ = c₀,cr(η)`` on the branch matching ``c0_start``, ``R = c₀/C₀``,
    ``ΔP`` from eq.~(8), ``P̄ = ΔP``, ``Σ̄`` from the sphere identity,
    ``U₀ = U₁ = 1/R``.
    """
    side = c0_critical_side_from_start(c0_start)
    c0_B = float(c0_critical_sphere(eta, side=side))
    R_B = float(R0_from_c0(C0, c0_B))
    if R_B <= 0.0:
        raise ValueError(
            f"point B radius must be positive (got R={R_B} from c0={c0_B}, C0={C0})"
        )
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
    sol = integrate_two_leg(
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


class NeedMoreProlate(RuntimeError):
    """Exact energy search ran out of prolate frames before a crossing."""

    def __init__(
        self,
        message: str,
        *,
        pear_cur: MeridianSolution,
        last_idx: int,
        table: List[Dict[str, float]],
        tau_hint: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.pear_cur = pear_cur
        self.last_idx = int(last_idx)
        self.table = table
        self.tau_hint = None if tau_hint is None else float(tau_hint)


def _energy_D_stride(dE: float, dE_per_frame: Optional[float]) -> int:
    """Frames to skip when ``|ΔE|`` is still large (coarse hop toward the zero)."""
    dE = float(dE)
    if dE_per_frame is not None and abs(float(dE_per_frame)) > 1e-8:
        frames = abs(dE) / abs(float(dE_per_frame))
        return max(1, int(0.4 * frames))
    return max(1, min(120, int(abs(dE) / 0.025)))


def _energy_D_tau_hint(
    table: Sequence[Dict[str, float]],
    *,
    fallback_tau: float,
) -> Optional[float]:
    """Linearly extrapolate τ where ``ΔE = E_pear − E_prolate`` crosses zero."""
    if len(table) < 2:
        return None
    a, b = table[-2], table[-1]
    d0, d1 = float(a["dE"]), float(b["dE"])
    t0, t1 = float(a["tau"]), float(b["tau"])
    if abs(d1 - d0) < 1e-12 or abs(t1 - t0) < 1e-15:
        return None
    if d0 * d1 < 0.0:
        return 0.5 * (t0 + t1)
    if abs(d1) >= abs(d0):
        return None
    tau_x = t0 - d0 * (t1 - t0) / (d1 - d0)
    if not np.isfinite(tau_x):
        return None
    if d1 > 0.0 and tau_x < t1:
        return max(t1 + abs(t1 - t0), fallback_tau)
    if d1 < 0.0 and tau_x > t1:
        return min(t1 - abs(t1 - t0), fallback_tau)
    return float(tau_x)


def find_D_by_bending_energy(
    prolate_track: Sequence[MeridianSolution],
    pear_seed: MeridianSolution,
    *,
    start_idx: Optional[int] = None,
    T_d: float = 1.0,
    C0_dim: float = 1.0,
    kappa: float = 1.0,
    verbose: bool = False,
    **match_kw: Any,
) -> Tuple[MeridianSolution, MeridianSolution, List[Dict[str, float]]]:
    """Locate D at the prolate↔pear bending-energy crossing.

    ``start_idx`` is the guideline frame (typically closest to the approximate
    ``E_pear≈E_prolate`` boundary).  Exact Δ often lags the approx guideline, so
    the scan uses coarse strides toward the extrapolated zero and then a binary
    search once the sign change is bracketed.  Returns
    ``(D_prolate, pear_at_D, table)``.
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

    if start_idx is None:
        start_idx = _approx_D_guideline_index(prolates, C0=C0_dim)
    start_idx = int(np.clip(int(start_idx), 0, len(prolates) - 1))

    kw = dict(
        u0_window=0.35,
        v_step=0.003,
        min_cv_steps=5,
        verbose=False,
    )
    kw.update(match_kw)

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
            local_kw["min_cv_steps"] = max(int(local_kw.get("min_cv_steps", 5)), 10)
            local_kw["v_step"] = min(float(local_kw.get("v_step", 0.003)), 0.002)
        pear_try = match_AV_from_seed(
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

    def _binary_bracket(
        lo: int,
        hi: int,
        pear_lo: MeridianSolution,
        pear_hi: MeridianSolution,
        *,
        pear_wins_at_hi: bool,
    ) -> Tuple[MeridianSolution, MeridianSolution, float, float]:
        """Refine ``[lo, hi]`` to the lowest-τ pear-win frame (D).

        ``pear_wins_at_hi`` True means Δ(hi)<0 and Δ(lo)≥0 (forward march).
        False means the opposite orientation was used while walking backward
        (pear wins at lo, prolate at hi) — unused; we always normalize so
        prolate-wins at lo and pear-wins at hi.
        """
        del pear_wins_at_hi  # API clarity for callers
        while hi - lo > 1:
            mid = (lo + hi) // 2
            warm = pear_lo if (mid - lo) <= (hi - mid) else pear_hi
            res = _match(prolates[mid], warm, first_hop=True)
            if res is None:
                # Skip a failed mid toward the side with a known pear.
                if mid - lo <= hi - mid:
                    lo = mid
                else:
                    hi = mid
                continue
            pear_try, Ep, Ee = res
            if Ee < Ep:
                hi, pear_hi = mid, pear_try
            else:
                lo, pear_lo = mid, pear_try
        prol_D = prolates[hi]
        Ep = bending_energy(prol_D, kappa=kappa, C0=C0_dim)
        Ee = bending_energy(pear_hi, kappa=kappa, C0=C0_dim)
        return prol_D, pear_hi, Ep, Ee

    guide = prolates[start_idx]
    guide_tau = float(guide.constraints["t_growth"]) / T_d
    guide_delta = annotate_approx_D_energies(guide, C0=C0_dim)
    if verbose:
        dtxt = "n/a" if guide_delta is None else f"{guide_delta:+.4f}"
        print(
            f"  energy-D: guideline τ={guide_tau:.6f}  "
            f"approx δ={dtxt}  (frame {start_idx + 1}/{len(prolates)})",
            flush=True,
        )

    start = _match(guide, pear_seed, first_hop=True)
    if start is None:
        raise RuntimeError(
            f"pear match failed at approx-boundary guideline τ≈{guide_tau:.4f}"
        )
    pear_start, Ep0, Ee0 = start
    dE0 = float(Ee0 - Ep0)

    if Ee0 >= Ep0:
        # Prolate still wins at the guideline → coarse forward march + binary search.
        lo = start_idx
        pear_lo = pear_start
        dE_lo = dE0
        prev_idx = start_idx
        dE_per_frame: Optional[float] = None
        while True:
            room = len(prolates) - 1 - lo
            if room <= 0:
                tau_hint = _energy_D_tau_hint(table, fallback_tau=guide_tau)
                raise NeedMoreProlate(
                    f"no exact crossing past approx guideline τ={guide_tau:.4f} "
                    f"(last Δ={table[-1]['dE']:+.4f} at τ={table[-1]['tau']:.6f})"
                    if table else
                    f"no exact crossing past τ={guide_tau:.4f}",
                    pear_cur=pear_lo,
                    last_idx=lo,
                    table=table,
                    tau_hint=tau_hint,
                )
            stride = min(room, _energy_D_stride(dE_lo, dE_per_frame))
            if abs(dE_lo) < 0.05:
                stride = min(stride, max(1, room))
                if stride > 1 and abs(dE_lo) < 0.02:
                    stride = 1
            j = lo + stride
            if verbose and stride > 1:
                print(
                    f"  energy-D: coarse +{stride} frames "
                    f"(Δ={dE_lo:+.4f} → try τ="
                    f"{float(prolates[j].constraints['t_growth']) / T_d:.6f})",
                    flush=True,
                )
            res = _match(prolates[j], pear_lo, first_hop=(stride > 2))
            if res is None:
                if stride == 1:
                    tau_hint = _energy_D_tau_hint(table, fallback_tau=guide_tau)
                    raise NeedMoreProlate(
                        f"pear match failed while marching past τ="
                        f"{float(prolates[lo].constraints['t_growth']) / T_d:.6f}",
                        pear_cur=pear_lo,
                        last_idx=lo,
                        table=table,
                        tau_hint=tau_hint,
                    )
                dE_per_frame = dE_lo / max(stride * 0.5, 1.0)
                continue
            pear_try, Ep, Ee = res
            dE = float(Ee - Ep)
            if j != prev_idx:
                dE_per_frame = (dE - dE_lo) / float(j - prev_idx)
            prev_idx = j
            if Ee < Ep:
                if j == lo + 1:
                    if verbose:
                        tau = float(prolates[j].constraints["t_growth"]) / T_d
                        print(
                            f"  energy-D: pear wins at τ={tau:.6f}  "
                            f"(E_pear={Ee:.4f} < E_prolate={Ep:.4f})",
                            flush=True,
                        )
                    D_prol, pear_D = _tag_energy_D(prolates[j], pear_try, Ep, Ee)
                    return D_prol, pear_D, table
                if verbose:
                    print(
                        f"  energy-D: bracketed crossing in "
                        f"τ∈[{float(prolates[lo].constraints['t_growth']) / T_d:.6f}, "
                        f"{float(prolates[j].constraints['t_growth']) / T_d:.6f}]",
                        flush=True,
                    )
                D_prol, pear_D, Ep_D, Ee_D = _binary_bracket(
                    lo, j, pear_lo, pear_try, pear_wins_at_hi=True,
                )
                if verbose:
                    tau = float(D_prol.constraints["t_growth"]) / T_d
                    print(
                        f"  energy-D: pear wins at τ={tau:.6f}  "
                        f"(E_pear={Ee_D:.4f} < E_prolate={Ep_D:.4f})",
                        flush=True,
                    )
                D_prol, pear_D = _tag_energy_D(D_prol, pear_D, Ep_D, Ee_D)
                return D_prol, pear_D, table
            lo, pear_lo, dE_lo = j, pear_try, dE

    # Pear already wins at the guideline → coarse backward march + binary search.
    hi = start_idx
    pear_hi = pear_start
    dE_hi = dE0
    prev_idx = start_idx
    dE_per_frame = None
    while hi > 0:
        room = hi
        stride = min(room, _energy_D_stride(dE_hi, dE_per_frame))
        if abs(dE_hi) < 0.05:
            stride = min(stride, max(1, room))
            if stride > 1 and abs(dE_hi) < 0.02:
                stride = 1
        j = hi - stride
        if verbose and stride > 1:
            print(
                f"  energy-D: coarse −{stride} frames "
                f"(Δ={dE_hi:+.4f} → try τ="
                f"{float(prolates[j].constraints['t_growth']) / T_d:.6f})",
                flush=True,
            )
        res = _match(prolates[j], pear_hi, first_hop=(stride > 2))
        if res is None:
            if stride == 1:
                break
            dE_per_frame = abs(dE_hi) / max(stride * 0.5, 1.0)
            continue
        pear_try, Ep, Ee = res
        dE = float(Ee - Ep)
        if j != prev_idx:
            dE_per_frame = (dE_hi - dE) / float(prev_idx - j)
        prev_idx = j
        if Ee >= Ep:
            if hi == j + 1:
                Ep_D = bending_energy(prolates[hi], kappa=kappa, C0=C0_dim)
                Ee_D = bending_energy(pear_hi, kappa=kappa, C0=C0_dim)
                D_prol, pear_D = _tag_energy_D(prolates[hi], pear_hi, Ep_D, Ee_D)
                if verbose:
                    tau_D = float(D_prol.constraints["t_growth"]) / T_d
                    print(
                        f"  energy-D: pear wins at τ={tau_D:.6f}  "
                        f"(E_pear={Ee_D:.4f} < E_prolate={Ep_D:.4f})",
                        flush=True,
                    )
                return D_prol, pear_D, table
            if verbose:
                print(
                    f"  energy-D: bracketed crossing in "
                    f"τ∈[{float(prolates[j].constraints['t_growth']) / T_d:.6f}, "
                    f"{float(prolates[hi].constraints['t_growth']) / T_d:.6f}]",
                    flush=True,
                )
            D_prol, pear_D, Ep_D, Ee_D = _binary_bracket(
                j, hi, pear_try, pear_hi, pear_wins_at_hi=True,
            )
            if verbose:
                tau_D = float(D_prol.constraints["t_growth"]) / T_d
                print(
                    f"  energy-D: pear wins at τ={tau_D:.6f}  "
                    f"(E_pear={Ee_D:.4f} < E_prolate={Ep_D:.4f})",
                    flush=True,
                )
            D_prol, pear_D = _tag_energy_D(D_prol, pear_D, Ep_D, Ee_D)
            return D_prol, pear_D, table
        hi, pear_hi, dE_hi = j, pear_try, dE
    else:
        if verbose:
            print(
                "  energy-D: pear favored down to the earliest prolate; "
                f"D pinned at τ={float(prolates[hi].constraints['t_growth']) / T_d:.6f}",
                flush=True,
            )

    D_prol, pear_at_D = prolates[hi], pear_hi
    Ep_D = bending_energy(D_prol, kappa=kappa, C0=C0_dim)
    Ee_D = bending_energy(pear_at_D, kappa=kappa, C0=C0_dim)
    if verbose:
        tau_D = float(D_prol.constraints["t_growth"]) / T_d
        print(
            f"  energy-D: pear wins at τ={tau_D:.6f}  "
            f"(E_pear={Ee_D:.4f} < E_prolate={Ep_D:.4f})",
            flush=True,
        )
    D_prol, pear_at_D = _tag_energy_D(D_prol, pear_at_D, Ep_D, Ee_D)
    return D_prol, pear_at_D, table


def _approx_D_guideline_index(
    prolates: Sequence[MeridianSolution],
    *,
    C0: float = 1.0,
) -> int:
    """First frame after the approximate energy boundary is crossed."""
    best_i = 0
    best_abs = float("inf")
    previous: Optional[Tuple[int, float]] = None
    for i, sol in enumerate(prolates):
        delta = annotate_approx_D_energies(sol, C0=C0)
        if delta is None:
            continue
        delta = float(delta)
        if previous is not None:
            _, previous_delta = previous
            if previous_delta >= 0.0 and delta < 0.0:
                return i
        previous = (i, delta)
        a = abs(delta)
        if a < best_abs:
            best_abs = a
            best_i = i
    if not np.isfinite(best_abs):
        return max(len(prolates) - 1, 0)
    return best_i


def _approx_boundary_progress(
    prolates: Sequence[MeridianSolution],
    *,
    C0: float = 1.0,
) -> Dict[str, Any]:
    """Summarize approximate-boundary progress along a prolate track."""
    deltas: List[Tuple[int, float, float]] = []  # (idx, tau, delta)
    for i, sol in enumerate(prolates):
        if not sol.success or "t_growth" not in sol.constraints:
            continue
        delta = annotate_approx_D_energies(sol, C0=C0)
        if delta is None:
            continue
        tau = float(sol.constraints["t_growth"]) / float(
            sol.constraints.get("T_d", 1.0)
        )
        deltas.append((i, tau, float(delta)))
    if not deltas:
        return {
            "crossed_zero": False,
            "guideline_idx": 0,
            "min_abs_delta": float("inf"),
        }
    guide_i = min(range(len(deltas)), key=lambda k: abs(deltas[k][2]))
    guideline_idx = deltas[guide_i][0]
    min_abs = abs(deltas[guide_i][2])
    crossed = False
    for (_, _, d0), (_, _, d1) in zip(deltas[:-1], deltas[1:]):
        if d0 >= 0.0 and d1 < 0.0:
            crossed = True
            break
    last_delta = deltas[-1][2]
    return {
        "crossed_zero": crossed,
        "guideline_idx": guideline_idx,
        "min_abs_delta": min_abs,
        "last_delta": last_delta,
        "last_tau": deltas[-1][1],
    }


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

def _score_prolate_leave_candidate(
    sol: MeridianSolution,
    *,
    u0_min: Optional[float] = None,
) -> float:
    """Leave-sphere prolate score; no absolute ``U₀≤0.4`` reject (large-R spheres)."""
    if len(sol.s) < 5:
        return float("-inf")
    if u0_min is not None and float(sol.U0) < float(u0_min):
        return float("-inf")
    if u0_min is not None and float(sol.U1) < float(u0_min):
        return float("-inf")
    if float(sol.U0) <= 0.0 or float(sol.U1) <= 0.0:
        return float("-inf")
    res = float(sol.constraints.get("residual", float("inf")))
    if not np.isfinite(res):
        res = float("inf")
    if not sol.success and res > 0.05:
        return float("-inf")
    u_excess = float(sol.U0)
    if u0_min is not None:
        u_excess = float(sol.U0 - u0_min)
    return u_excess - 2.0 * res


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
        score = _score_prolate_leave_candidate(sol, u0_min=u0_min)
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


def _oblate_seeds_leave_sphere(
    warm: MeridianSolution,
    c0: float,
    *,
    A_target: Optional[float] = None,
    u_bump: float = 1e-4,
    p_bump: float = 1e-4,
) -> List[np.ndarray]:
    """Shoot seeds for leaving the sphere toward oblates: ``U₀,U₁↓``, ``P̄↓``."""
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
        # Keep U > 0; clamp du so we stay a fraction of the sphere curvature.
        du_eff = min(float(du), 0.85)
        U_seed = float(U_sphere * (1.0 - du_eff))
        if U_seed <= 1e-8:
            return
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
        U_seed = float(U_sphere * 0.99)
        seeds.append(np.array([U_seed, U_seed, S1_ref, sig_ref, P_ref], dtype=float))
    return seeds


def _score_oblate_candidate(
    sol: MeridianSolution,
    *,
    u0_max: Optional[float] = None,
) -> float:
    """Higher is better for oblates; −inf rejects prolate / failed shoots."""
    if len(sol.s) < 5:
        return float("-inf")
    if u0_max is not None and float(sol.U0) > float(u0_max):
        return float("-inf")
    if u0_max is not None and float(sol.U1) > float(u0_max):
        return float("-inf")
    if float(sol.U0) <= 0.0 or float(sol.U1) <= 0.0:
        return float("-inf")
    res = float(sol.constraints.get("residual", float("inf")))
    if not np.isfinite(res):
        res = float("inf")
    if not sol.success and res > 0.05:
        return float("-inf")
    # Prefer deeper oblate (smaller U) with small residual.
    u_deficit = 0.0
    if u0_max is not None:
        u_deficit = float(u0_max - sol.U0)
    return u_deficit - 2.0 * res


def _shoot_oblate_leave_sphere(
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
    """First oblate equilibrium after BS eq. (9) fails at point B."""
    u_bump = max(float(u_bump), 0.0)
    p_bump = max(float(p_bump), 0.0)
    seeds = _oblate_seeds_leave_sphere(
        warm, c0, A_target=float(A_target), u_bump=u_bump, p_bump=p_bump,
    )
    shoot_kw = dict(A_target=float(A_target), A_c0=float(A_target))
    U_sphere = float(1.0 / area_radius(float(A_target)))
    u0_max = float(U_sphere * (1.0 - 0.25 * max(u_bump, 1e-8)))
    u_win = float(u0_window) if u0_window is not None else max(0.05 * U_sphere, 0.01)
    u_win = min(u_win, 0.15 * max(U_sphere, 0.1))
    if verbose:
        print(
            f"    oblate-B: U_bump={u_bump:.2e}  P_bump={p_bump:.2e}  "
            f"1/R={U_sphere:.6f}  ceiling={u0_max:.6f}",
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
            **shoot_kw,
        )
        if float(sol.U0) > u0_max or float(sol.U1) > u0_max:
            return
        score = _score_oblate_candidate(sol, u0_max=u0_max)
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
            du = min(max(u_bump * scale, 1e-6), 0.85)
            dp = max(p_bump * scale, 1e-6)
            U_seed = float(U_sphere * (1.0 - du))
            if U_seed <= 1e-8:
                continue
            P_b = float(P_ref * (1.0 - dp))
            sig = sphere_sigma_bar_at_radius(C0_w, P_b, R_w)
            _try_seed(np.array([U_seed, U_seed, S1_ref, sig, P_b], dtype=float))

    if best is not None:
        best.constraints["branch_side"] = "oblate"
        best.constraints["oblate_u_bump"] = u_bump
        best.constraints["oblate_p_bump"] = p_bump
        return best

    fail = _shoot_two_leg(
        v, c0, seeds[0], branch=branch, n=200, u0_window=u_win, verbose=verbose,
        **shoot_kw,
    )
    fail = replace(
        fail,
        success=False,
        message="oblate leave-sphere failed (U₀ > 1/R — prolate rejected)",
        constraints={**fail.constraints, "branch_side": "prolate_rejected"},
    )
    return fail


# |c₀| past which, at v̄≈1, the sphere loses to prolates (C₀>0) / oblates (C₀<0).
BS_V1_C0_CROSSOVER = 1.35


def leave_sphere_by_energy(
    eta: float,
    C0: float,
    *,
    T_d: float = 1.0,
    kappa: float = 1.0,
    c0_start: float = BS_C0_START_NEG,
    dt: Optional[float] = None,
    v_bump: float = 1e-6,
    # Tiny bumps (~1e-4) stay near-sphere; ΔE is noise and can pick the wrong side.
    # ~3e-3+ resolves the oblate preference for |c₀| ≳ BS_V1_C0_CROSSOVER at v̄≈1.
    u_bump: float = 5e-3,
    p_bump: float = 5e-3,
    branch: str = "seifert",
    u0_window: Optional[float] = None,
    verbose: bool = True,
    energy_rtol: float = 1e-8,
    energy_atol: float = 1e-5,
) -> Tuple[MeridianSolution, Dict[str, float], Dict[str, Any]]:
    """One step past B; shoot prolate and oblate; keep the lower bending energy.

    Returns ``(winner, B_state, contest)`` where ``contest`` logs both sides.
    Near-ties (``|ΔE|`` below ``energy_atol`` / ``energy_rtol``) prefer the
    phase-diagram side for ``C₀<0``: oblate when ``|c₀| ≥ BS_V1_C0_CROSSOVER``.
    """
    if float(C0) >= 0.0:
        raise ValueError("leave_sphere_by_energy is for C0 < 0")
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

    if verbose:
        print("  leave-sphere contest: prolate bump…", flush=True)
    prol = _shoot_prolate_leave_sphere(
        post["v"], post["c0"], warm,
        A_target=post["A"], branch=branch, u0_window=u0_window, verbose=verbose,
        u_bump=u_bump, p_bump=p_bump,
    )
    if verbose:
        print("  leave-sphere contest: oblate bump…", flush=True)
    obl = _shoot_oblate_leave_sphere(
        post["v"], post["c0"], warm,
        A_target=post["A"], branch=branch, u0_window=u0_window, verbose=verbose,
        u_bump=u_bump, p_bump=p_bump,
    )

    def _stamp(sol: MeridianSolution, *, side: str) -> MeridianSolution:
        sol.constraints.update({
            "R0": post["R"],
            "A_phys": post["A"],
            "V_phys": post["V"],
            "v_ode": post["v"],
            "C0_dimensional": float(C0),
            "DeltaP": float(sol.P_bar) if sol.success else float("nan"),
            "P_dimensional": (
                float(growth_pressure_from_shoot(sol, post["R"], kappa=kappa))
                if sol.success else float("nan")
            ),
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
            "branch_side": side,
        })
        return sol

    prol = _stamp(prol, side="prolate")
    obl = _stamp(obl, side="oblate")

    E_prol = (
        float(bending_energy(prol, kappa=kappa, C0=C0))
        if prol.success and len(prol.s) > 0 else float("nan")
    )
    E_obl = (
        float(bending_energy(obl, kappa=kappa, C0=C0))
        if obl.success and len(obl.s) > 0 else float("nan")
    )
    prol.constraints["E_bending"] = E_prol
    obl.constraints["E_bending"] = E_obl

    contest: Dict[str, Any] = {
        "prolate_ok": bool(prol.success),
        "oblate_ok": bool(obl.success),
        "E_prolate": E_prol,
        "E_oblate": E_obl,
        "prolate": prol,
        "oblate": obl,
    }

    if prol.success and obl.success:
        scale = max(abs(E_prol), abs(E_obl), 1.0)
        tol = max(float(energy_atol), float(energy_rtol) * scale)
        dE_raw = float(E_obl - E_prol)
        if abs(dE_raw) <= tol:
            # Near-degenerate near-sphere shoots: use Seifert v̄=1 branch preference.
            prefer_oblate = abs(float(post["c0"])) >= float(BS_V1_C0_CROSSOVER)
            contest["near_tie"] = True
            contest["energy_tol"] = tol
            if prefer_oblate:
                winner, loser = obl, prol
            else:
                winner, loser = prol, obl
        elif E_obl < E_prol:
            winner, loser = obl, prol
            contest["near_tie"] = False
        else:
            winner, loser = prol, obl
            contest["near_tie"] = False
        contest["winner"] = winner.constraints.get("branch_side")
        contest["dE"] = float(
            winner.constraints["E_bending"] - loser.constraints["E_bending"]
        )
    elif prol.success:
        winner = prol
        contest["winner"] = "prolate"
        contest["dE"] = float("nan")
    elif obl.success:
        winner = obl
        contest["winner"] = "oblate"
        contest["dE"] = float("nan")
    else:
        raise RuntimeError(
            "leave-sphere contest failed: neither prolate nor oblate succeeded "
            f"(prolate: {prol.message}; oblate: {obl.message})"
        )

    if verbose:
        print(
            f"  contest: E_prolate={E_prol:.6g} ({'ok' if prol.success else 'FAIL'})  "
            f"E_oblate={E_obl:.6g} ({'ok' if obl.success else 'FAIL'})  "
            f"→ winner={contest['winner']}",
            flush=True,
        )
    return winner, st, contest


def _grow_oblate_to_approx_boundary(
    tip: MeridianSolution,
    *,
    eta: float,
    C0: float,
    T_d: float,
    kappa: float,
    dt: float,
    tau_ceiling: float = BS_TAU_D_GROW_CEILING,
    u0_window: float = 0.35,
    verbose: bool = False,
) -> List[MeridianSolution]:
    """Grow oblates until approximate stomatocyte energy becomes favored (δ<0)."""
    if float(C0) >= 0.0:
        raise ValueError("_grow_oblate_to_approx_boundary is for C0 < 0")
    prev_delta = annotate_approx_D_energies(tip, C0=C0)

    def _crossed(sol: MeridianSolution) -> bool:
        nonlocal prev_delta
        delta = annotate_approx_D_energies(sol, C0=C0)
        if delta is None:
            return False
        crossed = prev_delta is not None and float(prev_delta) >= 0.0 and float(delta) < 0.0
        prev_delta = float(delta)
        return crossed

    if prev_delta is not None and float(prev_delta) < 0.0:
        if verbose:
            print(
                f"  approx-D_sto: tip already on stomatocyte-favored side "
                f"(δ={float(prev_delta):+.4f})",
                flush=True,
            )
        return [tip]

    tau0 = float(tip.constraints.get("t_growth", 0.0)) / float(T_d)
    if verbose:
        dtxt = "n/a" if prev_delta is None else f"{float(prev_delta):+.4f}"
        print(
            f"  approx-D_sto: growing from τ={tau0:.4f} (δ={dtxt}) toward "
            f"E_sto≈E_oblate crossing (ceiling τ={float(tau_ceiling):g})",
            flush=True,
        )

    track = continue_bozic_svetina(
        tip, eta=eta, T_d=T_d, kappa=kappa, dt=dt,
        t_final=float(tau_ceiling) * float(T_d), C0_dim=C0, max_dv=0.008,
        u0_window=u0_window,
        stop_if=_crossed,
    )
    for sol in track:
        annotate_approx_D_energies(sol, C0=C0)
    last = track[-1]
    tau_last = float(last.constraints.get("t_growth", 0.0)) / float(T_d)
    delta_last = last.constraints.get("dE_approx_rel", float("nan"))
    if verbose:
        print(
            f"  approx-D_sto: ended at τ={tau_last:.4f} "
            f"(last δ={float(delta_last):+.4f})",
            flush=True,
        )
    return track


def _select_D_sto_from_oblate_track(
    track: Sequence[MeridianSolution],
    *,
    C0: float,
) -> MeridianSolution:
    """First successful frame with δ<0; else last frame (growth ceiling)."""
    ok = [s for s in track if s.success and len(s.s) > 0]
    if not ok:
        raise RuntimeError("empty oblate track for D_sto")
    for sol in ok:
        annotate_approx_D_energies(sol, C0=C0)
    for sol in ok:
        d = sol.constraints.get("dE_approx_rel")
        if d is not None and float(d) < 0.0:
            sol.constraints["D_by"] = "approx_energy"
            sol.constraints["landmark"] = "D_sto"
            return sol
    D = ok[-1]
    D.constraints["D_by"] = "tau_ceiling"
    D.constraints["landmark"] = "D_sto"
    return D


def step2_oblate_to_D_sto(
    eta: float,
    C0: float,
    tip: MeridianSolution,
    *,
    T_d: float = 1.0,
    kappa: float = 1.0,
    dt: float = 0.001,
    cache: Optional[Path] = None,
    use_cache: bool = True,
    tau_ceiling: float = BS_TAU_D_GROW_CEILING,
    verbose: bool = True,
) -> Tuple[List[MeridianSolution], MeridianSolution]:
    """Grow the post-B oblate until the approximate oblate↔stomatocyte boundary."""
    if float(C0) >= 0.0:
        raise ValueError("step2_oblate_to_D_sto is for C0 < 0")
    if cache is not None and use_cache:
        cached = _load_track(cache, "oblate_growth_track")
        if cached:
            for sol in cached:
                annotate_approx_D_energies(sol, C0=C0)
            D = _select_D_sto_from_oblate_track(cached, C0=C0)
            return cached, D

    if not tip.success:
        raise RuntimeError("oblate tip required for D_sto growth")
    tip.constraints.setdefault("phase", "shape")
    tip.constraints.setdefault("branch_side", "oblate")
    track = _grow_oblate_to_approx_boundary(
        tip, eta=eta, C0=C0, T_d=T_d, kappa=kappa, dt=dt,
        tau_ceiling=tau_ceiling, verbose=verbose,
    )
    D = _select_D_sto_from_oblate_track(track, C0=C0)
    if cache is not None:
        _save_track(
            cache, "oblate_growth_track", track, name="oblate_growth_track",
            manifest_extra={
                "meta": {
                    "eta": eta, "C0": C0, "T_d": T_d, "dt": dt,
                    "tau_D_sto": float(D.constraints["t_growth"]) / float(T_d),
                    "D_by": D.constraints.get("D_by"),
                },
            },
        )
    return track, D


def step3_stomatocyte_at_D(
    D: MeridianSolution,
    *,
    cache: Optional[Path] = None,
    T_d: float = 1.0,
    C0: float = -1.0,
    use_cache: bool = True,
    verbose: bool = True,
) -> MeridianSolution:
    """Rescale package stomatocyte seed to D; AV-match to landmark D_sto.

    The package seed lives at ``c₀=0``.  Unlike pears (where reduced ``c₀`` is
    preserved by ``C₀→C₀/s``), injecting the growth ``C₀<0`` into the rescaled warm start opens
    the junction.  Here we rescale geometry at **seed** ``c₀``, close the
    junction, then walk carefully in ``(c₀,v̄)`` to D.
    """
    if float(C0) >= 0.0:
        raise ValueError("step3_stomatocyte_at_D is for C0 < 0")
    pwf: Dict[str, Any] = {}
    if cache is not None and use_cache:
        try:
            pwf = load_bs_workflow(cache)
        except FileNotFoundError:
            pwf = {}

    A_D = float(D.constraints["A_phys"])
    sto25 = pwf.get("sto_rescaled") if use_cache else None
    # Require a closed, c₀≈0 warm start (reject old C₀-injected rescaled caches).
    if sto25 is not None:
        res = float(sto25.constraints.get("residual", float("inf")))
        psi = abs(float(sto25.constraints.get("psi_match", 1.0)))
        if (
            abs(float(sto25.c0)) > 0.05
            or not sto25.success
            or (np.isfinite(res) and res > 1e-4)
            or psi > 1e-3
            or not _landmark_area_matches(sto25, D)
        ):
            if verbose:
                print(
                    "  discarding cached sto_rescaled "
                    f"(c₀={sto25.c0:.3f}, |res|={res:.3g}, |ψm|={psi:.3g})",
                    flush=True,
                )
            sto25 = None

    if sto25 is None:
        seed = load_stomatocyte_seed()
        if verbose:
            print(
                f"  stomatocyte seed ← {resolve_stomatocyte_seed_path().name} → D_sto "
                f"(geometry rescale at seed c₀={seed.c0:g})",
                flush=True,
            )
        from seifert_functions import (
            rescale_pear_params_to_landmark,
            integrate_two_leg,
            meridian_area,
            solve_seifert,
        )
        params = rescale_pear_params_to_landmark(
            seed, A_target=A_D, S1_reference=float(D.S1),
        )
        c0_seed = float(seed.c0)
        if verbose:
            print(
                f"  rescale: s={params['length_scale']:.4f}  "
                f"A={params['A_source']:.3f}→{A_D:.3f}  "
                f"S₁={params['S1_source']:.3f}→{params['S1']:.3f}  "
                f"c₀ kept={c0_seed:g}",
                flush=True,
            )
        sto25 = integrate_two_leg(
            params["U0"], params["U1"], params["S1"],
            sigma_bar=params["sigma_bar"], P_bar=params["P_bar"],
            c0=c0_seed, branch="stomatocyte_rescaled", n=160,
            enforce_branch_filters=False, A_c0=A_D,
            A1_hint=A_D,
            V1_hint=float(seed.constraints.get("V_end", float("nan")))
            * float(params["length_scale"]) ** 3,
        )
        # Close junction at fixed (v̄, c₀_seed, A_D).
        closed = solve_seifert(
            float(sto25.v), c0_seed, prev=sto25,
            branch="stomatocyte_rescaled",
            use_u0_scan=False, lock_branch=True, u0_window=0.8,
            A_target=A_D, A_c0=A_D, polish=True, verbose=False, max_nfev=160,
        )
        if closed.success and len(closed.s) > 0:
            sto25 = closed
        sto25 = replace(sto25, branch="stomatocyte_rescaled", c0=c0_seed)
        sto25.constraints.update({
            "A_target": A_D, "A_phys": A_D, "length_scale": params["length_scale"],
            "C0": float(C0), "pipeline": "rescaled", "c0_seed": c0_seed,
        })
        if verbose:
            print(
                f"  rescaled closed: v̄={sto25.v:.4f} c₀={sto25.c0:.4f}  "
                f"U0={sto25.U0:.4f} U1={sto25.U1:.4f}  "
                f"|res|={float(sto25.constraints.get('residual', float('nan'))):.2e}  "
                f"ok={sto25.success}",
                flush=True,
            )
        if cache is not None:
            save_bs_workflow(cache, sto_rescaled=sto25)

    sto_at_D = pwf.get("sto_matched") if use_cache else None
    if sto_at_D is not None and not _targets_match_D(sto_at_D, D):
        sto_at_D = None
    if sto_at_D is not None:
        res = float(sto_at_D.constraints.get("residual", float("inf")))
        if (
            not sto_at_D.success
            or not np.isfinite(res)
            or res > 0.08
            or float(sto_at_D.U0) > 0.05
        ):
            if verbose:
                print(
                    f"  discarding cached sto_matched "
                    f"(success={sto_at_D.success}, |res|={res:.3g}, U0={sto_at_D.U0:.3f})",
                    flush=True,
                )
            sto_at_D = None
    if sto_at_D is None:
        # Careful (c₀,v̄) walk from seed c₀→D with stomatocyte branch gate.
        sto_at_D = match_AV_from_seed(
            sto25, float(D.v), float(D.c0),
            A_target=A_D,
            family="stomatocyte",
            refine=False,
            u0_window=0.75,
            c0_step=0.02,
            v_step=0.002,
            min_cv_steps=80,
            pear_min_v=0.50,
            junction_match_tol=0.08,
            s1_max_rel_jump=0.12,
            cv_bisect_max=8,
            verbose=verbose,
        )
        sto_at_D = replace(sto_at_D, branch="stomatocyte")
        sto_at_D.constraints["landmark"] = "D_sto"
        sto_at_D.constraints["branch_side"] = "stomatocyte"
        if cache is not None and sto_at_D.success:
            save_bs_workflow(cache, sto_matched=sto_at_D, sto_rescaled=sto25)
    if not sto_at_D.success:
        if verbose:
            print(
                "  stomatocyte AV-match failed; "
                "using rescaled warm start at D area "
                f"({sto_at_D.message})",
                flush=True,
            )
        sto25.constraints["landmark"] = "D_sto"
        sto25.constraints["branch_side"] = "stomatocyte"
        sto25.constraints["match_failed"] = 1.0
        sto25.constraints["match_message"] = str(sto_at_D.message)
        if cache is not None:
            save_bs_workflow(cache, sto_rescaled=sto25)
        return sto25
    return sto_at_D


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
    stop_if: Optional[Callable[[MeridianSolution], bool]] = None,
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

    ``stop_if(sol)`` — if provided, keep ``sol`` and stop when it returns True
    (e.g. approximate D-boundary crossing).
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
    c0_cr = c0_critical_sphere(
        eta, side="-" if float(C0_fixed) < 0.0 else "+",
    )

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
        # Lock near-symmetric prolates/oblates, or inverted stomatocyte cups.
        on_stomatocyte = (
            warm_sol is not None
            and float(warm_sol.U0) < 0.0
        )
        on_locked = (
            warm_sol is not None
            and warm_sol.constraints.get("phase") == "shape"
            and float(warm_sol.U0) > 0.0
            and abs(float(warm_sol.U0) - float(warm_sol.U1)) < 0.08
        )
        pick = "first"
        use_scan = False
        if on_stomatocyte:
            # Keep the cup branch for the whole D→L_sto leg.
            lock = True
        elif lock_branch_steps is None:
            lock = on_locked
        else:
            lock = on_locked and shape_shoot_idx < int(lock_branch_steps)
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

        if stop_if is not None and stop_if(sol):
            if progress or verbose:
                print(
                    f"  [{i + 1}] stop_if triggered at τ={t_growth / T_d:.4f} "
                    f"({len(out)} snapshots)",
                    flush=True,
                )
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
    closed_pears: Optional[Sequence[MeridianSolution]] = None,
    pear_rescaled: Optional[MeridianSolution] = None,
    pear_matched: Optional[MeridianSolution] = None,
    pear_finish: Optional[MeridianSolution] = None,
    sto_rescaled: Optional[MeridianSolution] = None,
    sto_matched: Optional[MeridianSolution] = None,
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

    if closed_pears is not None:
        save_solution_list(cache_dir, "pear_closed", closed_pears)
        wf["pear_closed"] = "pear_closed.json"

    if pear_rescaled is not None:
        save_solution_at(cache_dir / "pear_rescaled.json", pear_rescaled)
        wf["pear_rescaled"] = "pear_rescaled.json"
        wf["pear_rescaled_index"] = int(
            pear_rescaled.constraints.get("closed_index", wf.get("pear_rescaled_index", 0))
        )

    if pear_matched is not None:
        save_solution_at(cache_dir / "pear_matched.json", pear_matched)
        wf["pear_matched"] = "pear_matched.json"

    if pear_finish is not None:
        save_solution_at(cache_dir / "pear_finish.json", pear_finish)
        wf["pear_finish"] = "pear_finish.json"

    if sto_rescaled is not None:
        save_solution_at(cache_dir / "sto_rescaled.json", sto_rescaled)
        wf["sto_rescaled"] = "sto_rescaled.json"

    if sto_matched is not None:
        save_solution_at(cache_dir / "sto_matched.json", sto_matched)
        wf["sto_matched"] = "sto_matched.json"

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

    # New checkpoint names with fallbacks for pre-rename caches.
    _CK = {
        "pear_closed": ("closed_pears", "pear_stage2", True),
        "pear_rescaled": ("pear_rescaled", "pear_stage25", False),
        "pear_matched": ("pear_matched", "pear_stage3", False),
        "pear_finish": ("pear_finish", "pear_stage4", False),
        "sto_rescaled": ("sto_rescaled", "sto_stage25", False),
        "sto_matched": ("sto_matched", "sto_stage3", False),
    }
    for new_key, (out_key, old_key, is_list) in _CK.items():
        rel = wf.get(new_key) or wf.get(old_key)
        if rel is None:
            if _need(new_key) or _need(old_key):
                raise KeyError(f"{new_key} not in cache")
            continue
        path = cache_dir / rel
        if not path.exists():
            # Manifest may still name the old file after a partial rewrite.
            alt = cache_dir / f"{old_key}.json"
            path = alt if alt.exists() else path
        if path.exists():
            out[out_key] = load_solution_list(path) if is_list else load_solution(path)
        elif _need(new_key) or _need(old_key):
            raise FileNotFoundError(f"{new_key} manifest entry missing: {path}")

    return out


# ===== Workflow steps =====


# --- paths & defaults -------------------------------------------------------

RESULTS = PACKAGE_ROOT / "results"


def cache_dir(
    eta: float,
    T_d: float,
    base: Optional[Path] = None,
    *,
    C0: Optional[float] = None,
) -> Path:
    if base is None:
        base = DEFAULT_BS_CACHE
    if C0 is None:
        name = f"eta{eta:g}_Td{T_d:g}"
    else:
        name = f"eta{eta:g}_C0{float(C0):g}_Td{T_d:g}"
    d = Path(base) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(
    eta: float,
    base: Optional[Path] = None,
    *,
    C0: Optional[float] = None,
) -> Path:
    """Output folder for a given η (e.g. ``results/eta1.85`` or ``…_C0-1``)."""
    if base is None:
        base = RESULTS
    if C0 is None:
        name = f"eta{eta:g}"
    else:
        name = f"eta{eta:g}_C0{float(C0):g}"
    d = Path(base) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _targets_match_D(sol: MeridianSolution, D: MeridianSolution, *, rtol: float = 5e-3) -> bool:
    return (
        abs(float(sol.v) - float(D.v)) < rtol
        and abs(float(sol.c0) - float(D.c0)) < rtol * max(abs(float(D.c0)), 1.0)
    )


def _landmark_area_matches(pear_rescaled: MeridianSolution, D: MeridianSolution, *, rtol: float = 1e-3) -> bool:
    A_t = float(D.constraints["A_phys"])
    A_s = float(pear_rescaled.constraints.get("A_phys", pear_rescaled.constraints.get("A_end", float("nan"))))
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


def load_sto_track(d: Path) -> Optional[List[MeridianSolution]]:
    """Load nested-sphere D→L_sto track (preferred) or legacy Seifert sto track."""
    nested = _load_track(d, "nested_sphere_growth_track")
    if nested is not None:
        return nested
    return _load_track(d, "sto_growth_track")


def save_sto_track(d: Path, track: Sequence[MeridianSolution]) -> None:
    """Legacy Seifert stomatocyte track cache."""
    _save_track(d, "sto_growth_track", track, name="sto_growth_track")


def save_nested_sphere_track(d: Path, track: Sequence[MeridianSolution]) -> None:
    _save_track(d, "nested_sphere_growth_track", track, name="nested_sphere_growth_track")


def load_nested_sphere_track(d: Path) -> Optional[List[MeridianSolution]]:
    return _load_track(d, "nested_sphere_growth_track")


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
    anchor: Optional[MeridianSolution] = None,
) -> MeridianSolution:
    """Warm pear for the energy-crossing scan (package seed → rescaled at guideline)."""
    if anchor is None:
        ok = [s for s in prolates if s.success and len(s.s) > 0]
        if not ok:
            raise RuntimeError("no prolate anchor for pear energy-D seed")
        idx = _approx_D_guideline_index(ok, C0=C0)
        anchor = ok[idx]
    if anchor is None:
        raise RuntimeError("no prolate anchor for pear energy-D seed")

    # Prefer cached matched / rescaled on this η if they already match.
    try:
        pwf = load_bs_workflow(cache)
    except FileNotFoundError:
        pwf = {}
    pear_matched = pwf.get("pear_matched")
    if pear_matched is not None and pear_matched.success and len(pear_matched.s) > 0 and _targets_match_D(pear_matched, anchor, rtol=0.02):
        return pear_matched
    pear25 = pwf.get("pear_rescaled")
    if pear25 is not None and _landmark_area_matches(pear25, anchor):
        return pear25

    seed = load_pear_seed()
    if verbose:
        print(f"  pear seed ← {resolve_pear_seed_path().name}", flush=True)
    pear25 = integrate_pear_at_landmark(seed, anchor, C0=C0, verbose=verbose)
    if use_cache:
        save_bs_workflow(cache, pear_rescaled=pear25)
    return pear25


def _grow_prolate_to_approx_boundary(
    tip: MeridianSolution,
    *,
    eta: float,
    C0: float,
    T_d: float,
    kappa: float,
    dt: float,
    prolate_u_bump: float,
    prolate_p_bump: float,
    tau_ceiling: float = BS_TAU_D_GROW_CEILING,
    verbose: bool = False,
) -> List[MeridianSolution]:
    """Grow prolates in one pass until the approximate energy difference crosses zero."""
    prev_delta = annotate_approx_D_energies(tip, C0=C0)

    def _crossed(sol: MeridianSolution) -> bool:
        nonlocal prev_delta
        delta = annotate_approx_D_energies(sol, C0=C0)
        if delta is None:
            return False
        crossed = prev_delta is not None and float(prev_delta) >= 0.0 and float(delta) < 0.0
        prev_delta = float(delta)
        return crossed

    if prev_delta is not None and float(prev_delta) < 0.0:
        if verbose:
            print(
                f"  approx-D: tip already on pear-favored side "
                f"(δ={float(prev_delta):+.4f})",
                flush=True,
            )
        return [tip]

    tau0 = float(tip.constraints.get("t_growth", 0.0)) / float(T_d)
    if verbose:
        dtxt = "n/a" if prev_delta is None else f"{float(prev_delta):+.4f}"
        print(
            f"  approx-D: growing from τ={tau0:.4f} (δ={dtxt}) toward "
            f"E_pear≈E_prolate crossing (ceiling τ={float(tau_ceiling):g})",
            flush=True,
        )

    track = continue_bozic_svetina(
        tip, eta=eta, T_d=T_d, kappa=kappa, dt=dt,
        t_final=float(tau_ceiling) * float(T_d), C0_dim=C0, max_dv=0.008,
        prolate_u_bump=prolate_u_bump, prolate_p_bump=prolate_p_bump,
        stop_if=_crossed,
    )
    for sol in track:
        annotate_approx_D_energies(sol, C0=C0)
    prog = _approx_boundary_progress(track, C0=C0)
    last = track[-1]
    tau_last = float(last.constraints.get("t_growth", 0.0)) / float(T_d)
    if verbose:
        if prog["crossed_zero"]:
            print(
                f"  approx-D: crossed E_pear≈E_prolate boundary "
                f"(last δ={prog['last_delta']:+.4f} at τ={tau_last:.4f})",
                flush=True,
            )
        else:
            print(
                f"  approx-D: ended at τ={tau_last:.4f} without crossing "
                f"(min|δ|={prog['min_abs_delta']:.4f}, "
                f"last δ={prog.get('last_delta', float('nan')):+.4f})",
                flush=True,
            )
    return track


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
    tau_ceiling: float = BS_TAU_D_GROW_CEILING,
    pear_cache: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[List[MeridianSolution], BSLandmarks, MeridianSolution]:
    """Bump onto prolate branch at B; cross the approximate boundary; pick exact D.

    Growth continues until the approximate two-sphere/prolate energy mismatch
    ``δ`` changes sign. Exact Helfrich pear/prolate comparison then starts at
    the first frame on the pear-favored side of the approximate boundary.
    If that exact crossing is not yet bracketed, prolate growth resumes until
    the crossing appears or ``tau_ceiling`` is hit.  Falls back to the
    ``BS_TAU_D_REF`` proxy if the pear seed / crossing is unavailable.
    """
    if use_cache:
        track, lm = load_prolate_track(cache)
        if track is not None and lm is not None:
            for sol in track:
                annotate_approx_D_energies(sol, C0=C0)
            D = _select_D_from_track(
                track, lm, cache=cache, T_d=T_d, C0=C0,
                pear_cache=pear_cache, use_cache=use_cache,
                kappa=kappa, tau_ceiling=tau_ceiling, dt=dt, eta=eta,
                prolate_u_bump=prolate_u_bump, prolate_p_bump=prolate_p_bump,
                verbose=verbose, allow_extend=True,
            )
            t_D = float(D.constraints["t_growth"])
            # If extension grew the track, reload from landmarks holder if needed.
            if lm.D is not D:
                pass
            # Extension may have appended frames onto `track` via allow_extend.
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
    track = _grow_prolate_to_approx_boundary(
        tip, eta=eta, C0=C0, T_d=T_d, kappa=kappa, dt=dt,
        prolate_u_bump=prolate_u_bump, prolate_p_bump=prolate_p_bump,
        tau_ceiling=tau_ceiling, verbose=verbose,
    )
    lm = detect_bs_landmarks(track, started_at_B=True, C0=C0, c0_start=BS_C0_START)
    D = _select_D_from_track(
        track, lm, cache=cache, T_d=T_d, C0=C0,
        pear_cache=pear_cache, use_cache=use_cache,
        kappa=kappa, tau_ceiling=tau_ceiling, dt=dt, eta=eta,
        prolate_u_bump=prolate_u_bump, prolate_p_bump=prolate_p_bump,
        verbose=verbose, allow_extend=True,
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
    track: List[MeridianSolution],
    lm: BSLandmarks,
    *,
    cache: Path,
    T_d: float,
    C0: float,
    pear_cache: Optional[Path],
    use_cache: bool,
    kappa: float,
    verbose: bool,
    tau_ceiling: float = BS_TAU_D_GROW_CEILING,
    dt: float = 0.001,
    eta: Optional[float] = None,
    prolate_u_bump: float = 3e-5,
    prolate_p_bump: float = 3e-5,
    allow_extend: bool = True,
) -> MeridianSolution:
    """Prefer bending-energy crossing near the approx boundary; fall back to τ proxy."""
    if lm.D is not None and lm.D.constraints.get("D_by") == "bending_energy":
        return lm.D
    try:
        ok = [s for s in track if s.success and len(s.s) > 0]
        if not ok:
            raise RuntimeError("empty prolate track for energy-D")
        guide_idx = _approx_D_guideline_index(ok, C0=C0)
        anchor = ok[guide_idx]
        pear_warm = _pear_seed_for_energy_D(
            cache, track, T_d=T_d, C0=C0, pear_cache=pear_cache,
            use_cache=use_cache, verbose=verbose, anchor=anchor,
        )
        start_idx = guide_idx
        while True:
            try:
                D, pear_at_D, _table = find_D_by_bending_energy(
                    track, pear_warm,
                    start_idx=start_idx, T_d=T_d, C0_dim=C0, kappa=kappa,
                    verbose=verbose,
                )
                if use_cache and pear_at_D.success and _targets_match_D(pear_at_D, D):
                    save_bs_workflow(cache, pear_matched=pear_at_D)
                    if verbose:
                        print(
                            "  energy-D: cached crossing pear as matched "
                            "(step3 will reuse)",
                            flush=True,
                        )
                return D
            except NeedMoreProlate as need:
                if not allow_extend:
                    raise
                last = track[-1]
                tau_last = float(last.constraints.get("t_growth", 0.0)) / float(T_d)
                if tau_last >= float(tau_ceiling) - 1e-12 or not last.success:
                    raise RuntimeError(
                        f"exact D crossing not found up to τ={tau_last:.4f}"
                    ) from need
                if eta is None:
                    eta = float(last.constraints.get("eta", 1.0))
                # Prefer extrapolated crossing τ; else grow a modest chunk.
                tau_target = float(need.tau_hint) if need.tau_hint is not None else (
                    tau_last + 0.05
                )
                # Pad past the hint so the crossing is on the new track, not at the tip.
                tau_target = min(float(tau_ceiling), max(tau_last + 0.02, tau_target + 0.02))
                if verbose:
                    hint = (
                        f" (hint τ≈{need.tau_hint:.4f})"
                        if need.tau_hint is not None else ""
                    )
                    print(
                        f"  energy-D: need more prolate past τ={tau_last:.4f}; "
                        f"extending to τ={tau_target:.4f}{hint}…",
                        flush=True,
                    )
                t_final = float(tau_target) * float(T_d)
                more = continue_bozic_svetina(
                    last, eta=float(eta), T_d=T_d, kappa=kappa, dt=dt,
                    t_final=t_final, C0_dim=C0, max_dv=0.008,
                    prolate_u_bump=prolate_u_bump, prolate_p_bump=prolate_p_bump,
                )
                added = 0
                for sol in more[1:]:
                    annotate_approx_D_energies(sol, C0=C0)
                    track.append(sol)
                    added += 1
                if added == 0:
                    raise RuntimeError(
                        f"could not extend prolate past τ={tau_last:.4f}"
                    ) from need
                # Resume forward search from the last matched frame.
                pear_warm = need.pear_cur
                start_idx = int(need.last_idx)
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

    pear25 = pwf.get("pear_rescaled") if use_cache else None
    if not (pear25 is not None and _landmark_area_matches(pear25, D)):
        seed = load_pear_seed()
        if verbose:
            print(f"  pear seed ← {resolve_pear_seed_path().name} → D", flush=True)
        pear25 = integrate_pear_at_landmark(seed, D, C0=C0, verbose=verbose)
        if use_cache:
            save_bs_workflow(cache, pear_rescaled=pear25)

    pear_at_D = pwf.get("pear_matched") if use_cache else None
    if pear_at_D is not None and not _targets_match_D(pear_at_D, D):
        pear_at_D = None
    if pear_at_D is None:
        pear_at_D = match_AV_from_seed(
            pear25, float(D.v), float(D.c0),
            A_target=float(D.constraints["A_phys"]),
            u0_window=0.35, v_step=0.004, min_cv_steps=25, verbose=verbose,
        )
        if use_cache:
            save_bs_workflow(cache, pear_matched=pear_at_D, pear_rescaled=pear25)
    if not pear_at_D.success:
        raise RuntimeError("pear AV-match at D failed")
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


def _sto_seed_at_D(
    sto_at_D: MeridianSolution,
    D: MeridianSolution,
    *,
    C0: float,
    eta: float,
    T_d: float,
    kappa: float,
    L_p: float,
) -> MeridianSolution:
    """Stamp growth state from landmark D_sto onto the matched stomatocyte."""
    seed = sto_at_D.copy()
    A_D = float(D.constraints["A_phys"])
    V_D = float(D.constraints.get(
        "V_phys",
        V_at_area(float(D.v), A_D),
    ))
    R_D = float(D.constraints.get("R0", np.sqrt(A_D / (4.0 * np.pi))))
    seed = replace(seed, branch="stomatocyte")
    seed.constraints.update({
        "R0": R_D,
        "A_phys": A_D,
        "V_phys": V_D,
        "C0_dimensional": float(C0),
        "DeltaP": float(sto_at_D.P_bar),
        "t_growth": float(D.constraints["t_growth"]),
        "eta": float(eta), "T_d": float(T_d), "L_p": float(L_p),
        "kappa": float(kappa),
        "phase": "shape",
        "landmark": "D_sto",
        "branch_side": "stomatocyte",
    })
    return seed


def step4_stomatocyte_to_L(
    sto_at_D: MeridianSolution,
    D: MeridianSolution,
    st: Dict[str, float],
    *,
    eta: float,
    C0: float,
    T_d: float = 1.0,
    kappa: float = 1.0,
    dt: float = 0.005,
    cache: Path,
    use_cache: bool = True,
    max_residual: float = 1e-4,
    max_step_time: float = 30.0,
    t_final: Optional[float] = None,
) -> List[MeridianSolution]:
    """Continue stomatocyte growth from D_sto until the L_sto cutoff.

    Legacy Seifert-shoot path. Prefer ``step4_nested_sphere_to_L_sto`` for
    the production neg-C₀ trajectory (nested two-sphere + analytic L_sto).
    """
    if float(C0) >= 0.0:
        raise ValueError("step4_stomatocyte_to_L is for C0 < 0")
    if use_cache:
        cached = _load_track(cache, "sto_growth_track")
        if cached is not None:
            return [s for s in cached if s.success and len(s.s) > 0]

    if not sto_at_D.success or len(sto_at_D.s) < 5:
        raise RuntimeError("stomatocyte at D_sto required for D→L_sto growth")

    L_p = float(st.get("L_p", L_p_from_eta(eta, T_d, abs(float(C0)), kappa=kappa)))
    seed = _sto_seed_at_D(
        sto_at_D, D, C0=C0, eta=eta, T_d=T_d, kappa=kappa, L_p=L_p,
    )
    t_D = float(D.constraints["t_growth"])
    if t_final is None:
        # D_sto can already sit past τ=1 (η=0.7); keep growing past D
        # until the residual/time cutoff defines L_sto.
        t_final = max(float(T_d), t_D + 0.75 * float(T_d))
    if float(t_final) <= t_D + 1e-12:
        raise ValueError(
            f"step4_stomatocyte_to_L needs t_final > t_D "
            f"(got t_final={float(t_final):g}, t_D={t_D:g})"
        )
    past = continue_bozic_svetina(
        seed, eta=eta, T_d=T_d, kappa=kappa, dt=dt,
        t_final=float(t_final), C0_dim=C0, max_dv=0.002, n_steps=200,
        u0_window=0.75, branch="stomatocyte",
        max_residual=max_residual, max_step_time=max_step_time,
        stop_if=_stomatocyte_branch_lost,
    )
    past_ok = [s for s in past if s.success and len(s.s) > 0]
    if not past_ok:
        raise RuntimeError("stomatocyte continuation D_sto→L_sto produced no shapes")
    # Drop a duplicate seed frame if continue_bozic_svetina prepended it.
    if len(past_ok) >= 2:
        t0 = float(past_ok[0].constraints.get("t_growth", -1.0))
        t1 = float(past_ok[1].constraints.get("t_growth", -2.0))
        if abs(t0 - t1) < 1e-12:
            past_ok = past_ok[1:]
    # stop_if keeps the offending frame; drop it so L_sto is the last cup.
    if past_ok and _stomatocyte_branch_lost(past_ok[-1]):
        if len(past_ok) == 1:
            raise RuntimeError("stomatocyte left the cup branch immediately after D_sto")
        past_ok = past_ok[:-1]
    tip = past_ok[-1]
    tip.constraints["landmark"] = "L_sto"
    tip.constraints["branch_side"] = "stomatocyte"
    tau_tip = float(tip.constraints.get("t_growth", float("nan"))) / float(T_d)
    tau_D = float(D.constraints["t_growth"]) / float(T_d)
    if tau_tip >= float(t_final) / float(T_d) - 1e-4:
        print(
            f"  stomatocyte reached τ={tau_tip:.6f}  v̄={tip.v:.6f}  "
            f"P̄={tip.P_bar:.4f}  U0={tip.U0:.4f} U1={tip.U1:.4f}  "
            f"(no early L_sto cutoff)",
            flush=True,
        )
    else:
        print(
            f"  L_sto cutoff @ τ={tau_tip:.6f}  "
            f"(from D_sto τ={tau_D:.6f})  v̄={tip.v:.6f}  "
            f"P̄={tip.P_bar:.4f}  U0={tip.U0:.4f} U1={tip.U1:.4f}",
            flush=True,
        )
    save_sto_track(cache, past_ok)
    return past_ok


def step4_nested_sphere_to_L_sto(
    D: MeridianSolution,
    st: Dict[str, float],
    *,
    eta: float,
    C0: float,
    T_d: float = 1.0,
    kappa: float = 1.0,
    dt: float = 0.005,
    cache: Path,
    use_cache: bool = True,
    tau_stop: float = 2.5,
    dV_frac: float = 1e-4,
    verbose: bool = True,
) -> List[MeridianSolution]:
    """Grow nested sphere-in-sphere from D_sto until analytic ``L_sto``.

    Pressure at each step is ``P ≈ −ΔE/ΔV`` from the nested two-sphere
    energy at fixed area.  No Seifert stomatocyte shoot is used.
    """
    if float(C0) >= 0.0:
        raise ValueError("step4_nested_sphere_to_L_sto is for C0 < 0")
    if use_cache:
        cached = load_nested_sphere_track(cache)
        if cached is not None:
            ok = [s for s in cached if s.success and len(s.s) > 0]
            if ok:
                return ok

    A0 = float(D.constraints.get("A_phys", D.y[5, -1] if D.y.size else float("nan")))
    V0 = float(D.constraints.get("V_phys", D.y[6, -1] if D.y.size else float("nan")))
    t0 = float(D.constraints.get("t_growth", float("nan")))
    if not (np.isfinite(A0) and np.isfinite(V0) and np.isfinite(t0)):
        raise RuntimeError("D_sto frame missing A_phys / V_phys / t_growth")

    # Ensure L_p is available for frame metadata (growth uses L_p_from_eta inside).
    _ = float(st.get("L_p", L_p_from_eta(eta, T_d, abs(float(C0)), kappa=kappa)))

    track = continue_nested_sphere_to_L_sto(
        A0=A0, V0=V0, t0=t0,
        T_d=T_d, C0_dim=C0, eta=eta, kappa=kappa, dt=dt,
        tau_stop=float(tau_stop), dV_frac=dV_frac, verbose=verbose,
    )
    if not track:
        raise RuntimeError("nested-sphere D_sto→L_sto produced no frames")
    tip = track[-1]
    tau_D = t0 / float(T_d)
    tau_L = float(tip.constraints.get("t_growth", float("nan"))) / float(T_d)
    print(
        f"  nested L_sto @ τ={tau_L:.6f}  (from D_sto τ={tau_D:.6f})  "
        f"v̄={tip.v:.6f}  c₀={tip.c0:.4f}  "
        f"L_sto(c₀)={tip.constraints.get('v_L_sto', float('nan')):.6f}  "
        f"P={tip.P_bar:.6f}",
        flush=True,
    )
    save_nested_sphere_track(cache, track)
    return track


def _stomatocyte_branch_lost(sol: MeridianSolution) -> bool:
    """True when the shoot has left the asymmetric cup (→ L_sto handoff)."""
    u0 = float(sol.U0)
    u1 = float(sol.U1)
    if u0 >= -0.02:
        return True
    # Symmetric inverted vesicle — not a stomatocyte cup.
    if abs(u0 - u1) < 0.08 and u1 <= 0.05:
        return True
    return False


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
    if tau_tip >= 1.0 - 1e-4:
        print(
            f"  reached τ={tau_tip:.6f}  v̄={tip.v:.6f}  P̄={tip.P_bar:.4f}  "
            f"U0={tip.U0:.4f} U1={tip.U1:.4f}",
            flush=True,
        )
    else:
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
            # Reject a stale full cache that ends before the pear tip.
            pear_ok = [s for s in pear if s.success and len(s.s) > 0]
            if pear_ok:
                t_full = float(cached[-1].constraints.get("t_growth", -1.0))
                t_pear = float(pear_ok[-1].constraints.get("t_growth", 0.0))
                if t_full + 1e-8 >= t_pear:
                    return cached
    full = splice_at_D(prolate, pear, D)
    # Always persist for downstream two-sphere / plotting tools.
    save_full_track(cache, full)
    return full


# Plotting: see bs_plotting.save_growth_plots

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

    # Prefer prolate+pear splice whenever both exist.  A cached full track can
    # lag behind a freshly recomputed pear leg (e.g. step4 with use_cache=False
    # while step5 still loads an old full_track.json).
    if prolate_ok and D is not None:
        t_D = float(D.constraints["t_growth"])
        before = [
            s for s in prolate_ok
            if float(s.constraints.get("t_growth", 0.0)) <= t_D + 1e-10
        ]
        after = [s for s in pear_ok if _seifert_tau(s, T_d=T_d) > tau_D + 1e-10]
        if after and abs(float(after[0].constraints["t_growth"]) - t_D) < 1e-10:
            after = after[1:]
        seifert_tr = before + after
    elif full is not None:
        seifert_tr = [
            s for s in _ok_seifert(full)
            if _seifert_tau(s, T_d=T_d) <= tau_fail + 1e-12
        ]
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


# Plotting helpers: see bs_plotting.attach_seifert_sols

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
    """Build a lightweight ``MeridianSolution`` via ``shapes.TwoSphere``."""
    if eta is None:
        eta = ETA
    shape = TwoSphere(
        A, V, C0_dim,
        north_larger=north_larger, tip_U0=tip_U0, tip_U1=tip_U1,
    )
    return shape.to_meridian_solution(
        t_growth=t_growth, P_bar=P_bar, L_p=L_p, T_d=T_d,
        eta=eta, kappa=kappa, message=message,
    )


def make_nested_sphere_frame(
    *,
    A: float,
    V: float,
    t_growth: float,
    P_bar: float,
    C0_dim: float,
    T_d: float = T_D,
    L_p: float,
    eta: Optional[float] = None,
    kappa: float = KAPPA,
    message: str = "nested-sphere",
) -> MeridianSolution:
    """Build a lightweight ``MeridianSolution`` via ``shapes.NestedSphere``."""
    if eta is None:
        eta = ETA
    shape = NestedSphere(A, V, C0_dim)
    return shape.to_meridian_solution(
        t_growth=t_growth, P_bar=P_bar, L_p=L_p, T_d=T_d,
        eta=eta, kappa=kappa, message=message,
    )



def continue_nested_sphere_to_L_sto(
    *,
    A0: float,
    V0: float,
    t0: float,
    T_d: float = T_D,
    C0_dim: float,
    eta: Optional[float] = None,
    kappa: float = KAPPA,
    dt: float = DT,
    tau_stop: float = 2.5,
    dV_frac: float = 1e-4,
    verbose: bool = True,
) -> List[MeridianSolution]:
    """Advance ``(A,V)`` with BS ODEs + ``P=−ΔE/ΔV`` until analytic ``L_sto``.

    Shapes are nested sphere-in-sphere.  Stop when ``v̄ ≤ L_sto(c₀)`` (last
    frame clamped onto the limit curve) or ``τ ≥ tau_stop``.
    """
    if eta is None:
        eta = ETA
    if float(C0_dim) >= 0.0:
        raise ValueError("continue_nested_sphere_to_L_sto requires C0 < 0")
    L_p = float(L_p_from_eta(eta, T_d, C0_dim, kappa))
    alpha = alpha_from_T_d(T_d)
    A, V, t = float(A0), float(V0), float(t0)

    def _P(A_cur: float, V_cur: float) -> float:
        return nested_sphere_pressure_from_dEdV(
            A_cur, V_cur, C0_dim, dV_frac=dV_frac, kappa=kappa,
        )

    P0 = _P(A, V)
    if not np.isfinite(P0):
        raise RuntimeError(
            f"nested-sphere pressure failed at start A={A:.4f} V={V:.4f}"
        )
    try:
        first = make_nested_sphere_frame(
            A=A, V=V, t_growth=t, P_bar=P0,
            C0_dim=C0_dim, T_d=T_d, L_p=L_p, eta=eta, kappa=kappa,
            message="nested-sphere @ D_sto",
        )
    except ValueError as exc:
        if verbose:
            print(f"  no nested-sphere leg: {exc}", flush=True)
        return []
    first.constraints["landmark"] = "D_sto"
    track: List[MeridianSolution] = [first]
    if verbose:
        print(
            f"  nested-sphere start: τ={t/T_d:.6f} A={A:.4f} V={V:.4f} "
            f"v̄={first.v:.6f}  c₀={first.c0:.4f}  "
            f"L_sto(c₀)={first.constraints['v_L_sto']:.6f}  "
            f"R_out={first.constraints['R_outer']:.4f}  "
            f"R_in={first.constraints['R_inner']:.4f}  "
            f"P={P0:.6f}  E={first.constraints['E_nested']:.4f}",
            flush=True,
        )

    i = 0
    while t / T_d < float(tau_stop) - 1e-12:
        v_now = reduced_volume_from_AV(A, V)
        c0_now = c0_reduced(C0_dim, A)
        v_L = L_sto_reduced_volume(c0_now)
        if np.isfinite(v_L) and v_now <= v_L + 1e-12:
            break

        dt_cur = min(dt, float(tau_stop) * T_d - t)
        P = _P(A, V)
        if not np.isfinite(P):
            if verbose:
                print(f"  stop: pressure NaN at τ={t/T_d:.6f}", flush=True)
            break
        A_new, V_new = bs_growth_step(A, V, dt_cur, alpha=alpha, L_p=L_p, P=P)
        t_new = t + dt_cur
        v_new = reduced_volume_from_AV(A_new, V_new)
        c0_new = c0_reduced(C0_dim, A_new)
        v_L_new = L_sto_reduced_volume(c0_new)
        hit_L = np.isfinite(v_L_new) and v_new <= v_L_new + 1e-12
        if hit_L:
            # Clamp volume onto L_sto at the new area.
            V_new = float(V_at_area(v_L_new, A_new))
            v_new = float(v_L_new)

        P_new = _P(A_new, V_new)
        if not np.isfinite(P_new):
            if verbose:
                print(f"  stop: pressure NaN after step at τ→{t_new/T_d:.6f}", flush=True)
            break
        try:
            fr = make_nested_sphere_frame(
                A=A_new, V=V_new, t_growth=t_new, P_bar=P_new,
                C0_dim=C0_dim, T_d=T_d, L_p=L_p, eta=eta, kappa=kappa,
                message=(
                    "nested-sphere @ L_sto (clamped)"
                    if hit_L
                    else f"nested-sphere step {i+1}"
                ),
            )
        except ValueError as exc:
            if verbose:
                print(f"  stop: {exc} at τ→{t_new/T_d:.6f}", flush=True)
            break
        track.append(fr)
        A, V, t = A_new, V_new, t_new
        i += 1
        if verbose and (i % 20 == 0 or t / T_d > float(tau_stop) - 0.002 or hit_L):
            print(
                f"  [{i}] τ={t/T_d:.6f} A={A:.4f} v̄={fr.v:.6f}  "
                f"c₀={fr.c0:.4f}  L_sto={fr.constraints['v_L_sto']:.6f}  "
                f"P={fr.P_bar:.6f}  E={fr.constraints['E_nested']:.4f}",
                flush=True,
            )
        if hit_L:
            if verbose:
                print(
                    f"  reached L_sto  v̄={fr.v:.6f}  c₀={fr.c0:.4f}  "
                    f"at τ={t/T_d:.6f}; stopping.",
                    flush=True,
                )
            break

    if track:
        tip = track[-1]
        tip.constraints["landmark"] = "L_sto"
        tip.constraints["branch_side"] = "stomatocyte"
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
    c0_start: Optional[float] = None,
) -> List[TrajFrame]:
    """Analytic spherical growth from point A (τ=0) to B."""
    if eta is None:
        eta = ETA
    if c0_start is None:
        c0_start = BS_C0_START_NEG if float(C0_dim) < 0.0 else BS_C0_START
    c0_start = float(c0_start)
    ref = bs_point_a_ref(C0_dim, c0_start=c0_start)
    A0 = float(ref["A"])
    L_p = float(L_p_from_eta(eta, T_d, C0_dim, kappa))
    tau_B = float(tau_B_from_eta(eta, c0_start=c0_start))
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


def solutions_to_traj(
    frames: Sequence[MeridianSolution],
    *,
    phase: str,
    T_d: float = T_D,
    tau_fallback: Optional[float] = None,
) -> List[TrajFrame]:
    """Map Seifert solutions onto ``TrajFrame`` with a fixed phase label."""
    out: List[TrajFrame] = []
    for sol in frames:
        if not sol.success or len(sol.s) < 2:
            continue
        if "t_growth" in sol.constraints:
            tau = _seifert_tau(sol, T_d=T_d)
        elif tau_fallback is not None:
            tau = float(tau_fallback)
        else:
            continue
        A = float(sol.constraints.get("A_phys", sol.y[5, -1] if sol.y.size else float("nan")))
        V = float(sol.constraints.get("V_phys", sol.y[6, -1] if sol.y.size else float("nan")))
        out.append(TrajFrame(
            tau=float(tau), A=A, V=V,
            v=float(sol.v), c0=float(sol.c0), P=float(sol.P_bar),
            phase=str(phase), message=sol.message, sol=sol,
        ))
    return out


def assemble_neg_c0_trajectory(
    oblate: Sequence[MeridianSolution],
    sto: Sequence[MeridianSolution] | MeridianSolution,
    *,
    eta: float,
    C0_dim: float,
    T_d: float = 1.0,
    kappa: float = 1.0,
    D: Optional[MeridianSolution] = None,
) -> List[TrajFrame]:
    """Sphere A→B + oblate B→D_sto + nested-sphere D_sto→L_sto."""
    if float(C0_dim) >= 0.0:
        raise ValueError("assemble_neg_c0_trajectory requires C0 < 0")
    sphere = build_sphere_A_to_B(
        eta=eta, C0_dim=C0_dim, T_d=T_d, kappa=kappa,
        c0_start=BS_C0_START_NEG,
    )
    obl = solutions_to_traj(oblate, phase="oblate", T_d=T_d)
    tau_D = None
    if D is not None and "t_growth" in D.constraints:
        tau_D = float(D.constraints["t_growth"]) / float(T_d)
    elif obl:
        tau_D = float(obl[-1].tau)
    if isinstance(sto, MeridianSolution):
        sto_list: Sequence[MeridianSolution] = [sto]
    else:
        sto_list = sto
    # Prefer nested_sphere phase when frames are nested; fall back to stomatocyte.
    phase = "nested_sphere"
    if sto_list and getattr(sto_list[0], "branch", "") not in (
        "nested_sphere", "two_sphere",
    ):
        if str(sto_list[0].constraints.get("phase", "")).startswith("nested"):
            phase = "nested_sphere"
        elif sto_list[0].branch == "stomatocyte":
            phase = "stomatocyte"
    sto_fr = solutions_to_traj(
        sto_list, phase=phase, T_d=T_d, tau_fallback=tau_D,
    )
    # Stamp R1/R2 onto traj frames for movie drawing.
    for fr, sol in zip(sto_fr, sto_list):
        if fr.R1 is None:
            fr.R1 = sol.constraints.get("R_outer", sol.constraints.get("R1"))
        if fr.R2 is None:
            fr.R2 = sol.constraints.get("R_inner", sol.constraints.get("R2"))
            if fr.R2 is not None:
                fr.R2 = abs(float(fr.R2))
        if fr.R1 is not None:
            fr.R1 = abs(float(fr.R1))
    return stitch_traj(sphere, obl, sto_fr)


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
        # Low-η trajectories can re-symmetrize after D; label those as prolate.
        if phase == "pear" and abs(float(sol.U0) - float(sol.U1)) < 0.05:
            phase = "prolate"
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

    # Already finished the Seifert leg at/near τ=1 (e.g. re-symmetrized prolates
    # for low η).  No thin-neck two-sphere continuation is needed or possible.
    asym = abs(float(tip.U0) - float(tip.U1))
    if tau_fail >= 1.0 - 1e-3 or (
        float(tip.v) < float(BS_V_TWO_SPHERE) - 1e-6 and asym < 0.05
    ):
        meta = {
            "tau_fail": tau_fail,
            "tau_D": float(ctx["tau_D"]),
            "a": float("nan"),
            "b": float("nan"),
            "n": 0,
            "tau_lo": float("nan"),
            "tau_hi": float("nan"),
            "P_fail": float(tip.P_bar),
            "P_E": float(tip.P_bar),
            "skipped": True,
        }
        if verbose:
            why = (
                f"reached τ≈1 as Seifert shape (U0={tip.U0:.4f} U1={tip.U1:.4f})"
                if tau_fail >= 1.0 - 1e-3
                else (
                    f"tip already below two-sphere window "
                    f"(v̄={tip.v:.6f} < {BS_V_TWO_SPHERE:.6f}, |U0−U1|={asym:.4f})"
                )
            )
            print(f"2) skip two-sphere finish: {why}", flush=True)
            print("4) two-sphere track: 0 frames", flush=True)
        return [], meta

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
        if rows:
            print(
                f"4) two-sphere track: {len(rows)} frames  "
                f"τ={rows[0]['tau']:.6f}→{rows[-1]['tau']:.6f}  "
                f"v̄={rows[0]['v']:.6f}→{rows[-1]['v']:.6f}",
                flush=True,
            )
        else:
            print(
                "4) two-sphere track: 0 frames "
                "(pear cutoff already at/below equal-sphere limit)",
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
            span = (
                f", τ={rows[0]['tau']:.6f}→{rows[-1]['tau']:.6f}" if rows else ""
            )
            print(
                f"  loaded two-sphere cache → {TWO_SPHERE_JSON}  "
                f"({len(rows)} frames{span})",
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

# Plotting: see bs_plotting (plot_full_trajectory, movie_full_trajectory, …)

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

    tau_fail = float(meta.get("tau_fail") if meta.get("tau_fail") is not None
                     else rows[0]["tau"])
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

    from bs_plotting import movie_full_trajectory, plot_full_trajectory

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
