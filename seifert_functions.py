"""Seifert Appendix-B meridian shooting and shape equilibria.

Used by ``bs_functions`` for Božič–Svetina growth trajectories.
"""

from __future__ import annotations

import json
import os
import re
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

PACKAGE_ROOT = Path(__file__).resolve().parent
A_STAR = 4.0 * np.pi

V_STAR = lambda v: (4.0 * np.pi / 3.0) * v

def V_at_area(v: float, area: float = A_STAR) -> float:
    """Target volume for reduced volume ``v̄`` at surface area ``A`` (``V ∝ A^{3/2}``)."""
    return float(V_STAR(v) * (float(area) / A_STAR) ** 1.5)

def reduced_volume_from_AV(area: float, volume: float) -> float:
    """Seifert reduced volume ``v̄ = 3V/(4π R₀³)`` with ``R₀² = A/(4π)`` (eq. 4.1)."""
    return 3.0 * volume * (A_STAR / area) ** 1.5 / A_STAR

def area_radius(area: float) -> float:
    """R₀ from A = 4π R₀²."""
    return float(np.sqrt(area / A_STAR))

def c0_reduced(C0: float, area: float = A_STAR) -> float:
    """Reduced spontaneous curvature c₀ = C₀ R₀ (Seifert eq. 5.1 / phase-diagram axis)."""
    return float(C0) * area_radius(area)

def C0_dimensional(c0: float, area: float = A_STAR) -> float:
    """Spontaneous curvature ``C_0`` for eqs. (3.4)--(3.5c) from reduced ``c_0`` (eq. 5.1).

    ``C_0 = c_0 / R_0`` with ``R_0 = sqrt(A / 4π)``. At ``A = 4π``, ``C_0 = c_0``.
    """
    R = area_radius(area)
    return float(c0) / R if R > 0.0 else float(c0)

def C0_for_shape(c0_reduced: float, area: float) -> float:
    """Dimensional ``C₀`` in the shape equations for reduced ``c₀`` at area ``A``."""
    return C0_dimensional(c0_reduced, area)

POLE_R = 1e-6          # regularized r at the pole (≪ S_POLE so sphere C₂ is clean)

S_POLE = 0.01

PSI_SOUTH = np.pi

JUNCTION_FRAC = 0.45  # S̄ / S₁ (interior junction, Appendix B stage 2)

FIG16_SIGMA_COEFF = -1.1  # Fig. 16: Σ̄ = FIG16_SIGMA_COEFF · P̄^{2/3}  (c₀ = 0 example)

FIG16_SCAN_MAX_STEP = 0.05  # stage-1 scan: RK45 step cap (fast branch map)

FIG16_SCAN_S_MAX = 160.0  # arclength ceiling if ψ=π not reached

FIG16_MAX_NFEV = 160_000  # Seifert Fig. 16 caption integration-step budget

FIG16_SCAN_RTOL = 1e-5

FIG16_SCAN_ATOL = 1e-7

FIG16_REFINE_MAX_STEP = 1e-3  # stage-2 / dense trial plots (paper step size)

FIG16_CROSSING_DEFAULT = 1  # S₁^{(n)} index n (n-th time ψ = π)

FIG16_PSI_MIN = -2.0 * np.pi  # discard if ψ < −2π (Fig. 16 caption)

FIG16_PSI_MAX = 3.0 * np.pi   # discard if ψ > 3π (Fig. 16 caption, n = 1)

FIG16_X_DISCARD = 1e-3  # discard if |X(S₁^{(n)})| P̄^{-1/3} < this (Fig. 16 caption)

FIG16_PSI_SWING_MAX = np.pi + 0.15  # reject legs with Δψ > π (looping meridian)

NEAR_ZERO_X_DEFAULT = 0.15  # |X| P̄^{-1/3} cap for one-sided zero extrapolation

SCAN_U0_TIMEOUT = 0.5  # seconds per U(0) trial before skipping

FIG16_S1_MAX = 120.0  # branch-candidate filter upper bound on S₁

FIG16_U0_SCALED_MIN = -0.8

FIG16_U0_SCALED_MAX = 0.8

FIG16_SCAN_GRID_POINTS = 120  # target count when building a wide U(0) grid

STAGE2_MAX_STEP = 0.04  # two-leg RK45 step cap (matches historical solver)

STAGE2_S1_MAX = 80.0  # upper S₁ bound for stage-2 shoot

STAGE2_U0_MIN = -12.0

STAGE2_U0_MAX = 20.0

STAGE2_U1_MIN = -8.0

STAGE2_U1_MAX = 20.0

STAGE2_S1_MIN = 0.5

STAGE2_U0_ABS = 0.12  # ± absolute U(0)

STAGE2_U0_REL = 0.12  # ± relative U(0); half-width = max(ABS, REL×|U₀*|)

STAGE2_U1_ABS = 0.35  # ± absolute U₁ (pear south pole can differ from north-leg hint)

STAGE2_U1_REL = 0.35  # ± relative U₁

STAGE2_S1_REL = 0.12  # ± relative S₁ (short branches, S₁* ≈ closure length)

STAGE2_S1_REL_LONG = 0.50  # long prolate pears: closure S₁ can exceed north-leg S₁*

STAGE2_RESIDUAL_TIMEOUT = 1.0  # wall-clock cap per stage-2 residual evaluation

STAGE3_RESIDUAL_TIMEOUT = 2.5  # wall-clock cap per stage-3 residual evaluation

STAGE1_ROOT_MARCH_MAX = 500  # max outer march iterations

STAGE1_ROOT_BISECT_MAX = 80  # max bisection refinements per bracket

STAGE1_P_ROOT_STEP_DEFAULT = 5e-3  # initial leftward march in P̄ (E* sided search)

STAGE1_P_ROOT_STEP_MIN = 1e-8  # stop bisect / halving below this in P̄

STAGE1_P_ROOT_X_TOL_DEFAULT = 3e-2  # target |X(S₁)| at P̄ root

DEFAULT_BS_CACHE = PACKAGE_ROOT / "cache"

def C2(X: float, psi: float) -> float:
    """Azimuthal curvature C₂ = sin ψ / X (eq. 3.1d; regularized at the pole)."""
    Xr = np.sqrt(X * X + POLE_R * POLE_R)
    return np.sin(psi) / Xr

def X_reg(X: float) -> float:
    return np.sqrt(X * X + POLE_R * POLE_R)

def meridian_rhs(_S: float, y: np.ndarray, C0: float, sigma_bar: float, P_bar: float) -> np.ndarray:
    """Seifert (1991) eqs. (3.5a)–(3.5d) + (B1a,b).

    ``C0`` is the spontaneous curvature ``C_0`` in eqs. (3.4)–(3.5c).
    γ is the Lagrange multiplier for Ẋ = cos ψ (eq. 3.2b), integrated via (3.5c).
    Pole conditions (3.7–3.8): γ(0) = H(0) = 0 and γ(S₁) = 0 on physical solutions.
    """
    X, _Z, psi, U, gamma, _A, _V = y
    Xr = X_reg(X)
    sinp, cosp = np.sin(psi), np.cos(psi)
    return np.array([
        cosp,  # (3.5d)  Ẋ = cos ψ
        -sinp,  # (3.1b)  Ż = −sin ψ
        U,  # (3.5a)  ψ̇ = U = C₁
        -U / Xr * cosp + cosp * sinp / (Xr * Xr) + gamma / Xr * sinp + 0.5 * P_bar * X * cosp,  # (3.5b)
        (U - C0) ** 2 / 2.0 - sinp ** 2 / (2.0 * Xr * Xr) + P_bar * X * sinp + sigma_bar,  # (3.5c)
        2.0 * np.pi * X,  # (B1a)  Ȧ = 2π X
        np.pi * X * X * sinp,  # (B1b)  V̇ = π X² sin ψ
    ])

def sphere_state(S: float, R: float = 1.0) -> np.ndarray:
    """Exact sphere meridian at arclength S (Seifert X–Z convention, Ż = −sin ψ)."""
    th = S / R
    return np.array([
        R * np.sin(th),
        R * (np.cos(th) - 1.0),
        th,
        1.0 / R,
        0.0,
        2.0 * np.pi * R ** 2 * (1.0 - np.cos(th)),
        np.pi * R ** 3 * (2.0 / 3.0 - np.cos(th) + (np.cos(th) ** 3) / 3.0),
    ])

def gamma_from_H0(
    X: float,
    psi: float,
    U: float,
    C0: float,
    sigma_bar: float,
    P_bar: float,
) -> float:
    """Lagrange multiplier γ from the first integral ``H = 0`` (eq. 3.4).

    Poles have ``γ → 0``, but the regularized start at ``S = S_POLE`` has
    ``ψ = O(S_POLE)`` so ``γ`` is ``O(S_POLE)``.  Forcing ``γ = 0`` there
    breaks the analytic sphere (``Σ̄ = −P̄ R/2``, ``U = 1/R``) at the level of
    the pole cutoff and shifts the apparent root off ``U = 5/11``.
    """
    cosp = float(np.cos(psi))
    if abs(cosp) < 1e-14:
        return 0.0
    c2 = C2(X, psi)
    h_no_gamma = (
        0.5 * X * (U * U - (c2 - C0) ** 2)
        - 0.5 * P_bar * X * X * np.sin(psi)
        - sigma_bar * X
    )
    return float(-h_no_gamma / cosp)

def north_pole_state(
    U0: float,
    *,
    C0: float = 0.0,
    sigma_bar: float = 0.0,
    P_bar: float = 0.0,
) -> np.ndarray:
    """Regularized north pole with ψ ≈ U₀ S and U = U₀ (eqs. 3.5a, 3.6a,c, 3.8).

    ``sphere_state(S, R)`` has ψ = S/R and U = 1/R, so R = 1/U₀ makes the cap
    geometry consistent with the shot meridional curvature.  ``γ`` is set from
    ``H = 0`` (not forced to 0) so the sphere branch is preserved.
    """
    if abs(U0) < 1e-14:
        X, psi, U = S_POLE, 0.0, 0.0
        return np.array([
            X,
            0.0,
            psi,
            U,
            gamma_from_H0(X, psi, U, C0, sigma_bar, P_bar),
            0.0,
            0.0,
        ])
    R = 1.0 / U0
    y0 = sphere_state(S_POLE, R=R).copy()
    y0[4] = gamma_from_H0(y0[0], y0[2], y0[3], C0, sigma_bar, P_bar)
    return y0

def south_pole_state_totals(
    U1: float,
    A1: float,
    V1: float,
    *,
    C0: float = 0.0,
    sigma_bar: float = 0.0,
    P_bar: float = 0.0,
) -> np.ndarray:
    """Regularized south pole: ψ ≈ π − U₁ S, U = U₁ (mirror of :func:`north_pole_state`).

    Appendix B starts the south leg at ``S₁`` with ``U = U₁*``; geometry and ``γ``
    (from ``H = 0``) must be consistent with meridional curvature ``U₁``.
    """
    if abs(U1) < 1e-14:
        X, psi, U = S_POLE, PSI_SOUTH, 0.0
        return np.array([
            X,
            0.0,
            psi,
            U,
            gamma_from_H0(X, psi, U, C0, sigma_bar, P_bar),
            float(A1),
            float(V1),
        ])
    R = 1.0 / U1
    th = np.pi - S_POLE / R  # π − U₁ S_POLE
    X = R * np.sin(th)
    y0 = np.array([
        X,
        R * (np.cos(th) - 1.0),
        th,
        U1,
        0.0,
        float(A1),
        float(V1),
    ])
    y0[4] = gamma_from_H0(X, th, U1, C0, sigma_bar, P_bar)
    return y0

@dataclass
class MeridianSolution:
    v: float
    c0: float
    U0: float
    U1: float
    sigma_bar: float
    P_bar: float
    s: np.ndarray
    y: np.ndarray
    branch: str = "seifert"
    success: bool = False
    message: str = ""
    constraints: Dict[str, float] = field(default_factory=dict)
    S1: float = 0.0
    S_bar: float = 0.0

    @property
    def X(self) -> np.ndarray:
        return self.y[0]

    @property
    def Z(self) -> np.ndarray:
        return self.y[1]

    @property
    def U(self) -> np.ndarray:
        return self.y[3]

    @property
    def gamma(self) -> np.ndarray:
        return self.y[4]

    def copy(self) -> "MeridianSolution":
        return replace(self)

# U₀ lower bound must admit deep stomatocytes at A★ (e.g. free-close U₀ ≲ −7).
TWO_LEG_BOUNDS = ([-20.0, -8.0, 0.5, -80.0, -80.0], [20.0, 20.0, 18.0, 80.0, 80.0])

def sigma_bar_from_P(P_bar: float) -> float:
    """Σ̄ = -1.1 · P̄^{2/3} (Seifert reduced c₀ = 0 example)."""
    if abs(P_bar) < 1e-14:
        return 0.0
    return FIG16_SIGMA_COEFF * (abs(P_bar) ** (2.0 / 3.0))

def _psi_window_max(n: int) -> float:
    """Upper ψ abort for ``S₁^{(n)}`` (caption uses 3π; higher n needs more room)."""
    return float(max(FIG16_PSI_MAX, (2 * n - 1) * np.pi))

def _psi_out_of_window(psi: float, *, n: int = 1) -> bool:
    """Fig.~16 caption: discard when ``ψ`` leaves ``[-2π, ψ_max(n)]``."""
    return psi < FIG16_PSI_MIN or psi > _psi_window_max(n) + 1e-9

def _x_scaled(x: float, P_bar: float) -> float:
    """``|X|`` on Fig.~16 axes: ``|X| · P̄^{-1/3}``."""
    if abs(P_bar) <= 1e-12:
        return abs(x)
    return abs(x) * (abs(P_bar) ** (-1.0 / 3.0))

def _discard_u0_trial(y: np.ndarray, P_bar: float, *, n: int = 1) -> bool:
    """Fig.~16 discard: ψ window, or ``|X(S₁^{(n)})| P̄^{-1/3} < 10^{-3}``.

    Points that fall inside the axis cutoff are numerically unreliable near the
    singular point ``X = 0``; Appendix B recovers the root by extrapolation from
    neighbouring samples, not from the discarded point itself.
    """
    if _psi_out_of_window(float(y[2]), n=n):
        return True
    return _x_scaled(float(y[0]), P_bar) < FIG16_X_DISCARD

def _fig16_psi_window_events(*, n: int = 1) -> List:
    """Fig.~16 ψ-window terminal events only (no ``ψ = π`` hit)."""

    def abort_psi_low(_S: float, y: np.ndarray) -> float:
        return float(y[2] - FIG16_PSI_MIN)

    abort_psi_low.terminal = True
    abort_psi_low.direction = -1

    psi_max = _psi_window_max(n)

    def abort_psi_high(_S: float, y: np.ndarray) -> float:
        return float(y[2] - psi_max)

    abort_psi_high.terminal = True
    abort_psi_high.direction = 1

    return [abort_psi_low, abort_psi_high]

def _fig16_leg_psi_loops(psi: np.ndarray) -> bool:
    """True when ψ swings by more than π along one leg (self-looping meridian)."""
    psi = np.asarray(psi, dtype=float)
    if psi.size < 2:
        return False
    return float(np.max(psi) - np.min(psi)) > FIG16_PSI_SWING_MAX

def _fig16_leg_psi_valid(psi: np.ndarray, *, n: int = 1) -> bool:
    """Fig.~16 leg check: ψ in ``[-2π, ψ_max(n)]`` and no π-swing loop."""
    psi = np.asarray(psi, dtype=float)
    if psi.size == 0:
        return False
    if _fig16_leg_psi_loops(psi):
        return False
    return all(not _psi_out_of_window(float(p), n=n) for p in psi)

def _integrate_leg_fig16(
    rhs,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    *,
    n_crossing: int = 1,
    max_step: float,
    rtol: float,
    atol: float,
    dense: bool = False,
) -> Optional[Any]:
    """Integrate one meridian leg; ``None`` if Fig.~16 ψ window or loop rule fires."""
    events = _fig16_psi_window_events(n=n_crossing)
    ivp = solve_ivp(
        rhs, t_span, np.asarray(y0, dtype=float), method="RK45",
        events=events, dense_output=dense,
        rtol=rtol, atol=atol, max_step=max_step,
    )
    if not ivp.success:
        return None
    for te in ivp.t_events:
        if len(te) > 0:
            return None
    if not _fig16_leg_psi_valid(ivp.y[2, :], n=n_crossing):
        return None
    return ivp

def _resolve_S_bar(
    S1: float,
    *,
    junction_frac: Optional[float] = None,
    S_bar: Optional[float] = None,
    north_frac: Optional[float] = None,
) -> float:
    """Interior junction arclength ``S̄`` from total meridian length ``S₁``.

    Specify exactly one of:

    * ``junction_frac`` — ``S̄ = junction_frac · S₁`` (default :data:`JUNCTION_FRAC`)
    * ``S_bar`` — explicit arclength
    * ``north_frac`` — north leg length ``= S_POLE + north_frac · (S₁ − 2 S_POLE)``
    """
    S1f = float(S1)
    n_set = sum(x is not None for x in (junction_frac, S_bar, north_frac))
    if n_set > 1:
        raise ValueError("pass only one of junction_frac, S_bar, north_frac")
    if S_bar is not None:
        s_bar = float(S_bar)
    elif north_frac is not None:
        interior = max(S1f - 2.0 * S_POLE, 1e-6)
        s_bar = S_POLE + float(north_frac) * interior
    elif junction_frac is not None:
        s_bar = float(junction_frac) * S1f
    else:
        s_bar = JUNCTION_FRAC * S1f
    S_south = S1f - S_POLE
    if not (S_POLE + 0.02 < s_bar < S_south - 0.02):
        raise ValueError(
            f"S̄={s_bar:.4f} outside valid range ({S_POLE + 0.02:.4f}, {S_south - 0.02:.4f}) "
            f"for S₁={S1f:.4f}"
        )
    return s_bar

def _two_leg_lengths(S1: float, S_bar: float) -> Tuple[float, float]:
    """``(north_leg, south_leg)`` arclengths from poles to ``S̄``."""
    S_south = float(S1) - S_POLE
    return float(S_bar) - S_POLE, S_south - float(S_bar)

def _integrate_leg_raw(
    rhs,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    *,
    max_step: float,
    rtol: float,
    atol: float,
    dense: bool = False,
) -> Optional[Any]:
    """Integrate one meridian leg without Fig.~16 discard (for diagnostic plots)."""
    ivp = solve_ivp(
        rhs, t_span, np.asarray(y0, dtype=float), method="RK45",
        dense_output=dense, rtol=rtol, atol=atol, max_step=max_step,
    )
    return ivp if ivp.success else None

def _integrate_leg(
    rhs,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    *,
    max_step: float,
    rtol: float,
    atol: float,
    dense: bool = False,
    enforce_fig16: bool = True,
    n_crossing: int = 1,
) -> Optional[Any]:
    if enforce_fig16:
        return _integrate_leg_fig16(
            rhs, t_span, y0,
            n_crossing=n_crossing, max_step=max_step, rtol=rtol, atol=atol, dense=dense,
        )
    return _integrate_leg_raw(
        rhs, t_span, y0, max_step=max_step, rtol=rtol, atol=atol, dense=dense,
    )

def _junction_closure_residual(yn: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Scaled ``(ψ, U, X)`` mismatch at interior junction ``S̄``."""
    scale_u = max(abs(yn[3]), abs(ys[3]), 1.0)
    scale_x = max(yn[0], ys[0], 0.05)
    return np.array([
        (yn[2] - ys[2]) / np.pi,
        (yn[3] - ys[3]) / scale_u,
        (yn[0] - ys[0]) / scale_x,
    ])

def _u0_scan_events(*, n: int = 1) -> List:
    """Terminal events: next ``ψ = π`` hit, or ψ-window exit."""

    def hit_psi(_S: float, y: np.ndarray) -> float:
        return float(y[2] - PSI_SOUTH)

    hit_psi.terminal = True
    hit_psi.direction = 0  # count every crossing of ψ = π

    def abort_psi_low(_S: float, y: np.ndarray) -> float:
        return float(y[2] - FIG16_PSI_MIN)

    abort_psi_low.terminal = True
    abort_psi_low.direction = -1

    psi_max = _psi_window_max(n)

    def abort_psi_high(_S: float, y: np.ndarray) -> float:
        return float(y[2] - psi_max)

    abort_psi_high.terminal = True
    abort_psi_high.direction = 1

    return [hit_psi, abort_psi_low, abort_psi_high]

def _solve_u0_ivp(
    rhs,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    events: List,
    *,
    max_step: float,
    rtol: float,
    atol: float,
    max_nfev: int = FIG16_MAX_NFEV,
) -> Optional[Tuple[Any, int, float]]:
    """Integrate with chunked ``nfev`` accounting (Fig.~16 step budget).

    Returns ``(ivp, event_index, S_hit)`` on a terminal event, else ``None``.
    """
    S_lo, S_hi = t_span
    S_cur = float(S_lo)
    y = np.asarray(y0, dtype=float)
    nfev_total = 0
    chunk = 8.0

    while S_cur < S_hi - 1e-12:
        S_end = min(S_cur + chunk, S_hi)
        ivp = solve_ivp(
            rhs, (S_cur, S_end), y, method="RK45",
            events=events, rtol=rtol, atol=atol, max_step=max_step,
        )
        nfev_total += int(ivp.nfev)
        if nfev_total > max_nfev:
            return None
        if not ivp.success:
            return None
        for ei, te in enumerate(ivp.t_events):
            if len(te) > 0:
                return ivp, ei, float(te[0])
        y = ivp.y[:, -1]
        S_cur = float(ivp.t[-1])
    return None

def _integrate_to_psi_pi(
    U0: float,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    *,
    S_max: float = FIG16_SCAN_S_MAX,
    rtol: float = FIG16_SCAN_RTOL,
    atol: float = FIG16_SCAN_ATOL,
    n: int = FIG16_CROSSING_DEFAULT,
    crossing: Optional[int] = None,
    max_step: float = FIG16_SCAN_MAX_STEP,
    apply_x_discard: bool = True,
) -> Optional[Tuple[np.ndarray, float]]:
    """Integrate to Appendix B ``S₁^{(n)}``: the ``n``-th time ``ψ(S) = π``.

    Fig.~16 discards when the step budget is exceeded, ``ψ`` leaves the caption
    window, or (if ``apply_x_discard``) ``|X(S₁^{(n)})| P̄^{-1/3} < FIG16_X_DISCARD``.

    Set ``apply_x_discard=False`` for stage-1 root refinement near ``X = 0``.
    """
    if crossing is not None:
        n = int(crossing)
    if n < 1:
        raise ValueError("n must be >= 1")

    C0 = C0_dimensional(c0, A_STAR)
    rhs = lambda S, y: meridian_rhs(S, y, C0, sigma_bar, P_bar)
    y_cur = north_pole_state(U0, C0=C0, sigma_bar=sigma_bar, P_bar=P_bar)
    S_cur = S_POLE
    yf: Optional[np.ndarray] = None
    S_hit = 0.0
    events = _u0_scan_events(n=n)

    for k in range(1, n + 1):
        out = _solve_u0_ivp(
            rhs, (S_cur, S_max), y_cur, events,
            max_step=max_step, rtol=rtol, atol=atol,
        )
        if out is None:
            return None
        ivp, ei, s_hit = out
        if ei != 0:
            return None
        yf = ivp.y[:, -1].copy()
        S_hit = s_hit
        if k < n:
            # Step past this π-crossing along dψ/dS = U so the next root is distinct.
            S_cur = S_hit + 1e-4
            y_cur = yf
            u_dir = float(np.sign(y_cur[3])) or 1.0
            y_cur[2] = PSI_SOUTH + u_dir * 1e-3

    assert yf is not None
    if apply_x_discard and _discard_u0_trial(yf, P_bar, n=n):
        return None
    return yf, S_hit

class _IntegrationTimeout(Exception):
    """Raised when a trial integration exceeds its wall-clock budget."""

def _call_with_timeout(timeout: Optional[float], func, *args, **kwargs):
    """Run ``func``; return ``(result, timed_out)``. Requires Unix ``SIGALRM``."""
    if timeout is None or timeout <= 0 or not hasattr(signal, "SIGALRM"):
        return func(*args, **kwargs), False

    def _handler(_signum, _frame) -> None:
        raise _IntegrationTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        return func(*args, **kwargs), False
    except _IntegrationTimeout:
        return None, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

def _scan_u0_worker(
    job: Tuple[int, float, float, float, float, int, float, float, Optional[float]],
) -> Tuple[int, bool, Optional[Tuple[np.ndarray, float]]]:
    """One ``U(0)`` trial for :func:`scan_u0` (runs in a worker process)."""
    i, U0, sigma_bar, P_bar, c0, n_s1, S_max, max_step, timeout = job
    out, timed_out = _call_with_timeout(
        timeout,
        _integrate_to_psi_pi,
        U0, sigma_bar, P_bar, c0,
        n=n_s1, S_max=S_max, max_step=max_step,
    )
    return i, timed_out, out

def scan_u0(
    c0: float,
    U0_values: Optional[np.ndarray] = None,
    *,
    sigma_bar: float = 0.0,
    P_bar: float = 0.0,
    n: int = FIG16_CROSSING_DEFAULT,
    crossing: Optional[int] = None,
    S_max: float = FIG16_SCAN_S_MAX,
    max_step: float = FIG16_SCAN_MAX_STEP,
    timeout: Optional[float] = SCAN_U0_TIMEOUT,
    workers: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan ``U(0)``: record ``X(S₁^{(n)})`` and ``U(S₁^{(n)})`` (Appendix B).

    ``S₁^{(n)}`` is the ``n``-th time ``ψ(S) = π``.  Trials with
    ``|X| P̄^{-1/3} < FIG16_X_DISCARD`` are omitted (NaN) per Fig.~16.

    Trials run in parallel (``workers`` processes, default = CPU count).
    Each trial is capped by ``timeout`` seconds; slow paths are skipped.
    """
    if crossing is not None:
        n = int(crossing)
    if U0_values is None:
        U0_values = _default_u0_grid(c0, P_bar=P_bar)
    U0_values = np.asarray(U0_values, dtype=float)
    n_pts = len(U0_values)
    X_south = np.full(n_pts, np.nan, dtype=float)
    U_south = np.full(n_pts, np.nan, dtype=float)
    S1_hit = np.full(n_pts, np.nan, dtype=float)
    n_workers = max(1, int(workers if workers is not None else (os.cpu_count() or 4)))
    if verbose:
        to_s = "off" if not timeout else f"{timeout:g}s"
        print(
            f"U(0) scan n={n}: {n_pts} points  ({n_workers} workers, timeout {to_s})",
            flush=True,
        )

    def _record(i: int, timed_out: bool, out: Optional[Tuple[np.ndarray, float]]) -> str:
        if timed_out:
            return "timeout"
        if out is not None:
            yf, s1 = out
            X_south[i] = yf[0]
            U_south[i] = yf[3]
            S1_hit[i] = s1
            return f"X(S₁^({n}))={yf[0]:+.4f}"
        return "discarded"

    n_ok = 0
    n_timeout = 0

    if n_workers == 1:
        for i, U0 in enumerate(U0_values):
            out, timed_out = _call_with_timeout(
                timeout,
                _integrate_to_psi_pi,
                U0, sigma_bar, P_bar, c0,
                n=n, S_max=S_max, max_step=max_step,
            )
            if timed_out:
                n_timeout += 1
            elif out is not None:
                n_ok += 1
            status = _record(i, timed_out, out)
            if verbose:
                print(
                    f"  [{i + 1}/{n_pts}] U(0)={U0:+.4f}  {status}  "
                    f"(valid {n_ok}/{i + 1})",
                    flush=True,
                )
    else:
        jobs = [
            (i, float(U0), float(sigma_bar), float(P_bar), float(c0),
             int(n), float(S_max), float(max_step), timeout)
            for i, U0 in enumerate(U0_values)
        ]
        n_done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_scan_u0_worker, job) for job in jobs]
            for fut in as_completed(futures):
                i, timed_out, out = fut.result()
                n_done += 1
                if timed_out:
                    n_timeout += 1
                elif out is not None:
                    n_ok += 1
                status = _record(i, timed_out, out)
                if verbose:
                    print(
                        f"  [{n_done}/{n_pts}] U(0)={U0_values[i]:+.4f}  {status}  "
                        f"(valid {n_ok}/{n_done})",
                        flush=True,
                    )

    if verbose:
        print(
            f"U(0) scan n={n} done: {n_ok}/{n_pts} valid trials"
            + (f", {n_timeout} timed out" if n_timeout else ""),
            flush=True,
        )
    return U0_values, X_south, U_south, S1_hit

def _scan_p_bar_worker(
    job: Tuple[int, float, float, float, float, int, float, float, Optional[float]],
) -> Tuple[int, bool, Optional[Tuple[np.ndarray, float]]]:
    """One ``P̄`` trial for :func:`scan_p_bar` (runs in a worker process)."""
    i, P_bar, sigma_bar, U0, c0, n_s1, S_max, max_step, timeout = job
    out, timed_out = _call_with_timeout(
        timeout,
        lambda: _integrate_to_psi_pi(
            U0, sigma_bar, P_bar, c0,
            n=n_s1, S_max=S_max, max_step=max_step, apply_x_discard=False,
        ),
    )
    return i, timed_out, out

def scan_p_bar(
    c0: float,
    U0: float,
    P_values: np.ndarray,
    *,
    sigma_bar: float = 0.0,
    n: int = FIG16_CROSSING_DEFAULT,
    crossing: Optional[int] = None,
    S_max: float = FIG16_SCAN_S_MAX,
    max_step: float = FIG16_SCAN_MAX_STEP,
    timeout: Optional[float] = SCAN_U0_TIMEOUT,
    workers: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan ``P̄`` at fixed ``U(0)``: record ``X(S₁^{(n)})`` and ``U(S₁^{(n)})``."""
    if crossing is not None:
        n = int(crossing)
    P_values = np.asarray(P_values, dtype=float)
    n_pts = len(P_values)
    X_south = np.full(n_pts, np.nan, dtype=float)
    U_south = np.full(n_pts, np.nan, dtype=float)
    S1_hit = np.full(n_pts, np.nan, dtype=float)
    n_workers = max(1, int(workers if workers is not None else (os.cpu_count() or 4)))
    if verbose:
        to_s = "off" if not timeout else f"{timeout:g}s"
        print(
            f"P̄ scan n={n}: U(0)={U0:.4f}  {n_pts} points  "
            f"({n_workers} workers, timeout {to_s})",
            flush=True,
        )

    def _record(i: int, timed_out: bool, out: Optional[Tuple[np.ndarray, float]]) -> str:
        if timed_out:
            return "timeout"
        if out is not None:
            yf, s1 = out
            X_south[i] = yf[0]
            U_south[i] = yf[3]
            S1_hit[i] = s1
            return f"X(S₁^({n}))={yf[0]:+.4f}"
        return "discarded"

    n_ok = 0
    n_timeout = 0

    if n_workers == 1:
        for i, P_bar in enumerate(P_values):
            out, timed_out = _call_with_timeout(
                timeout,
                lambda pb=P_bar: _integrate_to_psi_pi(
                    U0, sigma_bar, pb, c0,
                    n=n, S_max=S_max, max_step=max_step, apply_x_discard=False,
                ),
            )
            if timed_out:
                n_timeout += 1
            elif out is not None:
                n_ok += 1
            status = _record(i, timed_out, out)
            if verbose:
                print(
                    f"  [{i + 1}/{n_pts}] P̄={P_bar:+.4f}  {status}  "
                    f"(valid {n_ok}/{i + 1})",
                    flush=True,
                )
    else:
        jobs = [
            (i, float(P_bar), float(sigma_bar), float(U0), float(c0),
             int(n), float(S_max), float(max_step), timeout)
            for i, P_bar in enumerate(P_values)
        ]
        n_done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_scan_p_bar_worker, job) for job in jobs]
            for fut in as_completed(futures):
                i, timed_out, out = fut.result()
                n_done += 1
                if timed_out:
                    n_timeout += 1
                elif out is not None:
                    n_ok += 1
                status = _record(i, timed_out, out)
                if verbose:
                    print(
                        f"  [{n_done}/{n_pts}] P̄={P_values[i]:+.4f}  {status}  "
                        f"(valid {n_ok}/{n_done})",
                        flush=True,
                    )

    if verbose:
        print(
            f"P̄ scan n={n} done: {n_ok}/{n_pts} valid trials"
            + (f", {n_timeout} timed out" if n_timeout else ""),
            flush=True,
        )
    return P_values, X_south, U_south, S1_hit

def roots_from_scan(
    param: np.ndarray,
    X_south: np.ndarray,
    U_south: np.ndarray,
    S1_hit: np.ndarray,
    *,
    x_tol: float = 0.03,
) -> List[Tuple[float, float, float]]:
    """``X(S₁)=0`` roots on a 1D scan in ``param`` (``P̄`` or ``U(0)``).

    Returns ``(param*, U₁*, S₁*)`` via linear interpolation at sign changes.
    """
    p = np.asarray(param, dtype=float)
    x = np.asarray(X_south, dtype=float)
    us = np.asarray(U_south, dtype=float)
    s1 = np.asarray(S1_hit, dtype=float)
    roots: List[Tuple[float, float, float]] = []
    for i in range(len(p) - 1):
        if not (np.isfinite(x[i]) and np.isfinite(x[i + 1])):
            continue
        if x[i] * x[i + 1] < 0.0:
            t = -float(x[i]) / (float(x[i + 1]) - float(x[i]))
            roots.append((
                float(p[i] + t * (p[i + 1] - p[i])),
                float(us[i] + t * (us[i + 1] - us[i])),
                float(s1[i] + t * (s1[i + 1] - s1[i])),
            ))
    for i in range(len(p)):
        if np.isfinite(x[i]) and abs(float(x[i])) <= float(x_tol):
            cand = (float(p[i]), float(us[i]), float(s1[i]))
            if not any(abs(cand[0] - r[0]) < 1e-4 for r in roots):
                roots.append(cand)
    roots.sort(key=lambda t: t[0])
    return roots

def _stage1_p_hit(
    P_bar: float,
    *,
    U0: float,
    c0: float,
    sigma_bar: float,
    n_s1: int = 1,
    max_step: Optional[float] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """Signed ``X(S₁^{(n)})``, ``U(S₁)``, ``S₁`` at fixed ``U(0)`` and scanned ``P̄``."""
    specs: List[Tuple[float, float, float]] = []
    if max_step is not None:
        specs.append((
            float(max_step),
            float(rtol if rtol is not None else FIG16_SCAN_RTOL),
            float(atol if atol is not None else FIG16_SCAN_ATOL),
        ))
    else:
        specs.extend([
            (FIG16_SCAN_MAX_STEP, FIG16_SCAN_RTOL, FIG16_SCAN_ATOL),
            (FIG16_REFINE_MAX_STEP, 1e-7, 1e-9),
        ])
    for ms, rt, at in specs:
        out = _integrate_to_psi_pi(
            float(U0), sigma_bar, float(P_bar), c0,
            n=int(n_s1), max_step=ms, rtol=rt, atol=at,
            apply_x_discard=False,
        )
        if out is not None:
            yf, s1 = out
            return float(yf[0]), float(yf[3]), float(s1)
    return None

def _stage1_param_record_history(
    history: Optional[List[Tuple[float, float]]],
    param: float,
    x: float,
) -> None:
    if history is None:
        return
    param, x = float(param), float(x)
    if history and history[-1][0] == param and history[-1][1] == x:
        return
    history.append((param, x))

def _refine_stage1_p_bracket(
    p_lo: float,
    x_lo: float,
    p_hi: float,
    x_hi: float,
    *,
    U0: float,
    c0: float,
    sigma_bar: float,
    n_s1: int,
    max_step: float,
    x_tol: float,
    p_tol: float,
    max_iter: int = STAGE1_ROOT_BISECT_MAX,
    verbose: bool = False,
    history: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[float, float, float, float]:
    """Bisect ``[p_lo, p_hi]`` toward ``X=0`` at fixed ``U(0)``."""
    p_lo, p_hi = float(p_lo), float(p_hi)
    if p_lo > p_hi:
        p_lo, p_hi = p_hi, p_lo
        x_lo, x_hi = x_hi, x_lo

    if abs(x_lo) <= abs(x_hi):
        p_best, x_best = p_lo, x_lo
    else:
        p_best, x_best = p_hi, x_hi
    hit_best = _stage1_p_hit(
        p_best, U0=U0, c0=c0, sigma_bar=sigma_bar, n_s1=n_s1, max_step=max_step,
    )
    if hit_best is None:
        return p_best, x_best, 0.0, 0.0
    _, us_best, s1_best = hit_best
    _stage1_param_record_history(history, p_best, x_best)

    for k in range(int(max_iter)):
        if p_hi - p_lo <= float(p_tol):
            break
        if abs(x_best) <= float(x_tol):
            break

        p_mid = 0.5 * (p_lo + p_hi)
        hit = _stage1_p_hit(
            p_mid, U0=U0, c0=c0, sigma_bar=sigma_bar, n_s1=n_s1, max_step=max_step,
        )
        if hit is None:
            break
        x_mid, us_mid, s1_mid = hit
        _stage1_param_record_history(history, p_mid, x_mid)
        if abs(x_mid) < abs(x_best):
            p_best, x_best, us_best, s1_best = p_mid, x_mid, us_mid, s1_mid
        if abs(x_mid) <= float(x_tol):
            return p_mid, x_mid, us_mid, s1_mid

        if x_mid > 0.0:
            p_hi, x_hi = p_mid, x_mid
        else:
            p_lo, x_lo = p_mid, x_mid

        if verbose and (k < 5 or k == max_iter - 1 or abs(x_best) <= x_tol * 10):
            print(
                f"  stage1 P-bisect [{k + 1}]: P̄={p_best:.8g}  |X|={abs(x_best):.3e}  "
                f"bracket ΔP̄={p_hi - p_lo:.3e}",
                flush=True,
            )

    return p_best, x_best, us_best, s1_best

def find_stage1_p_root_from_right(
    p_start: float,
    *,
    U0: float,
    c0: float,
    sigma_bar: float,
    n_s1: int = 1,
    step: float = STAGE1_P_ROOT_STEP_DEFAULT,
    step_min: float = STAGE1_P_ROOT_STEP_MIN,
    x_tol: float = STAGE1_P_ROOT_X_TOL_DEFAULT,
    max_step: Optional[float] = None,
    max_marches: int = STAGE1_ROOT_MARCH_MAX,
    step_shrink: float = 0.5,
    verbose: bool = False,
    history: Optional[List[Tuple[float, float]]] = None,
) -> Optional[Tuple[float, float, float]]:
    """Lower ``X(S₁^{(n)})`` by marching left in ``P̄`` from the right-hand branch.

    On the wanted branch ``X > 0`` and decreases as ``P̄`` decreases (move left on
    the ``P̄`` vs ``X`` plot).  Overshoots bracket ``X = 0`` and refine by bisection.
    Returns ``(P̄*, U₁*, S₁*)``.
    """
    p_start = float(p_start)
    step = float(step)
    step_min = float(step_min)
    if step <= 0.0 or step_min <= 0.0:
        raise ValueError("step and step_min must be positive")
    p_tol = step_min

    hit0 = _stage1_p_hit(
        p_start, U0=U0, c0=c0, sigma_bar=sigma_bar, n_s1=n_s1, max_step=max_step,
    )
    if hit0 is None:
        return None
    x, u_south, s1 = hit0
    if x <= 0.0:
        if verbose:
            print(f"  stage1 P-sided: p_start={p_start:.8g} has X={x:+.3e} (need X>0)")
        return None

    p = p_start
    step_cur = step
    p_best, x_best, us_best, s1_best = p, x, u_south, s1
    _stage1_param_record_history(history, p, x)

    def _finish_root() -> Tuple[float, float, float]:
        nonlocal p_best, x_best, us_best, s1_best
        if x_best > 0.0:
            p_try = p_best - max(step_min, 10.0 * step_min)
            hit = _stage1_p_hit(
                p_try, U0=U0, c0=c0, sigma_bar=sigma_bar, n_s1=n_s1, max_step=max_step,
            )
            if hit is not None and hit[0] <= 0.0:
                p_b, x_b, us_b, s1_b = _refine_stage1_p_bracket(
                    p_try, hit[0], p_best, x_best,
                    U0=U0, c0=c0, sigma_bar=sigma_bar, n_s1=n_s1,
                    max_step=max_step or FIG16_REFINE_MAX_STEP,
                    x_tol=min(float(x_tol), 1e-4), p_tol=p_tol,
                    verbose=verbose, history=history,
                )
                if abs(x_b) < abs(x_best):
                    p_best, x_best, us_best, s1_best = p_b, x_b, us_b, s1_b
                if verbose:
                    print(
                        f"  stage1 P-polish: P̄={p_best:.8g}  X={x_best:+.3e}",
                        flush=True,
                    )
        return float(p_best), float(us_best), float(s1_best)

    for _n_iter in range(int(max_marches)):
        if abs(x_best) <= float(x_tol):
            if verbose:
                print(
                    f"  stage1 P-sided: |X|={abs(x_best):.3e} <= {x_tol:.3e} "
                    f"at P̄={p_best:.8g}",
                    flush=True,
                )
            return _finish_root()

        if step_cur < step_min:
            break

        p_try = p - step_cur
        hit = _stage1_p_hit(
            p_try, U0=U0, c0=c0, sigma_bar=sigma_bar, n_s1=n_s1, max_step=max_step,
        )
        if hit is None:
            if verbose:
                print(
                    f"  stage1 P-sided: integration failed at P̄={p_try:.8g}  "
                    f"(ψ window / step budget)",
                    flush=True,
                )
            step_cur *= float(step_shrink)
            continue

        x_try, u1_try, s1_try = hit
        _stage1_param_record_history(history, p_try, x_try)
        if abs(x_try) < abs(x_best):
            p_best, x_best, us_best, s1_best = p_try, x_try, u1_try, s1_try

        if x_try <= 0.0:
            if verbose:
                print(
                    f"  stage1 P-sided: overshoot P̄={p_try:.8g} X={x_try:+.3e}  "
                    f"→ bisect [{p_try:.8g}, {p:.8g}]",
                    flush=True,
                )
            p_b, x_b, us_b, s1_b = _refine_stage1_p_bracket(
                p_try, x_try, p, x,
                U0=U0, c0=c0, sigma_bar=sigma_bar, n_s1=n_s1, max_step=max_step or FIG16_REFINE_MAX_STEP,
                x_tol=x_tol, p_tol=p_tol, verbose=verbose, history=history,
            )
            if abs(x_b) < abs(x_best):
                p_best, x_best, us_best, s1_best = p_b, x_b, us_b, s1_b
            p, x, u_south, s1 = p_b, x_b, us_b, s1_b
            step_cur = max(step_min, min(step_cur, p - p_try) * step_shrink)
            continue

        if x_try > 0.0 and abs(x_try) < abs(x):
            p, x = p_try, x_try
            u_south, s1 = u1_try, s1_try
            if verbose:
                print(
                    f"  stage1 P-sided: P̄={p:.8g}  |X|={abs(x):.3e}  step={step_cur:.3e}",
                    flush=True,
                )
            continue

        if verbose:
            print(
                f"  stage1 P-sided: stall at P̄={p_try:.8g} X={x_try:+.3e}  "
                f"step {step_cur:.3e} → {step_cur * step_shrink:.3e}",
                flush=True,
            )
        step_cur *= float(step_shrink)

    if abs(x_best) <= float(x_tol):
        if verbose:
            print(
                f"  stage1 P-sided: |X|={abs(x_best):.3e} <= {x_tol:.3e} "
                f"at P̄={p_best:.8g}",
                flush=True,
            )
        return _finish_root()
    if verbose:
        print(
            f"  stage1 P-sided: best |X|={abs(x_best):.3e} at P̄={p_best:.8g} "
            f"(tol={x_tol:.3e}, step_min={step_min:.3e})",
            flush=True,
        )
    return _finish_root()

def _default_u0_grid(c0: float, P_bar: float = 1.0) -> np.ndarray:
    """Default raw ``U(0)`` grid (scaled range × ``P̄^{1/3}``)."""
    if c0 >= 2.0:
        return u0_grid(0.0, 3.5, n_points=36, P_bar=P_bar)
    return u0_grid(
        FIG16_U0_SCALED_MIN, FIG16_U0_SCALED_MAX, n_points=33, P_bar=P_bar,
    )

def near_zero_seeds(
    U0_values: np.ndarray,
    X_south: np.ndarray,
    U_south: np.ndarray,
    S1_hit: np.ndarray,
    *,
    P_bar: float = 1.0,
    x_tol: float = NEAR_ZERO_X_DEFAULT,
) -> List[Tuple[float, float, float]]:
    """Stage-1 seeds for Appendix B stage 2: ``(U(0)*, U₁*, S₁*)``.

    ``U₁*`` is ``U(S₁^{(n)})`` from the north-leg hit (Appendix B), refined in stage 2.

    Roots of ``X(S₁^{(n)}) = 0`` are recovered by linear extrapolation (paper):
    sign changes between consecutive valid samples, one-sided extrapolation when
    ``|X|`` trends toward zero at a timeout-bounded segment end, and the first
    valid sample after a low-``U(0)`` timeout run (low-``U`` branch cut).  Gap
    neighbours and repeated ``n`` values are merged by scaled ``U(0)``.
    """
    u = np.asarray(U0_values, dtype=float)
    x = np.asarray(X_south, dtype=float)
    us = np.asarray(U_south, dtype=float)
    s1 = np.asarray(S1_hit, dtype=float)
    n_pts = len(u)
    if n_pts == 0:
        return []

    def _extrapolate(i0: int, i1: int) -> Tuple[float, float, float]:
        x0, x1 = float(x[i0]), float(x[i1])
        t = -x0 / (x1 - x0)
        return (
            float(u[i0] + t * (u[i1] - u[i0])),
            float(us[i0] + t * (us[i1] - us[i0])),
            float(s1[i0] + t * (s1[i1] - s1[i0])),
        )

    du = float(np.nanmedian(np.diff(u))) if n_pts > 1 else 0.05
    if not np.isfinite(du) or du <= 0.0:
        du = 0.05
    # Split when samples are non-finite or S₁ jumps (branch change on the scan).
    s1_jump = max(1.0, 5.0 * du)

    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i in range(n_pts):
        ok = np.isfinite(x[i]) and np.isfinite(us[i]) and np.isfinite(s1[i])
        if ok and start is not None and i > 0:
            if abs(float(s1[i]) - float(s1[i - 1])) > s1_jump:
                segments.append((start, i - 1))
                start = i
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, n_pts - 1))

    def _sample_ok(i: int) -> bool:
        return (
            0 <= i < n_pts
            and np.isfinite(x[i])
            and np.isfinite(us[i])
            and np.isfinite(s1[i])
        )

    def _left_open(a: int) -> bool:
        return a == 0 or not _sample_ok(a - 1)

    def _right_open(b: int) -> bool:
        return b == n_pts - 1 or not _sample_ok(b + 1)

    def _branch_cut_right(b: int) -> bool:
        """True when integration fails or ``S₁`` jumps beyond ``b``."""
        if _right_open(b):
            return True
        if _sample_ok(b) and _sample_ok(b + 1):
            return abs(float(s1[b + 1]) - float(s1[b])) > s1_jump
        return False

    def _branch_cut_left(a: int) -> bool:
        if _left_open(a):
            return True
        if _sample_ok(a) and _sample_ok(a - 1):
            return abs(float(s1[a]) - float(s1[a - 1])) > s1_jump
        return False

    def _endpoint(i: int) -> Tuple[float, float, float]:
        return (float(u[i]), float(us[i]), float(s1[i]))

    first_valid = next((i for i in range(n_pts) if _sample_ok(i)), None)

    # (seed, quality): quality = 0 for extrapolated X=0 roots, else |X| on grid.
    candidates: List[Tuple[Tuple[float, float, float], float]] = []
    max_reach = 5.0 * du

    def _add(g: Tuple[float, float, float], *, quality: float = 0.0) -> None:
        candidates.append((g, quality))

    for a, b in segments:
        for i in range(a, b):
            if x[i] * x[i + 1] < 0.0:
                _add(_extrapolate(i, i + 1), quality=0.0)

        if b < a:
            continue
        if b == a:
            i = a
            xq = _x_scaled(float(x[i]), P_bar)
            if _branch_cut_left(i) or _branch_cut_right(i):
                if xq < x_tol:
                    _add(_endpoint(i), quality=xq)
                elif _branch_cut_right(i) and i > 0 and _sample_ok(i - 1):
                    if x[i] * (x[i] - x[i - 1]) < 0.0:
                        g = _extrapolate(i - 1, i)
                        if g[0] > float(u[i]) and g[0] - float(u[i]) <= max_reach:
                            _add(g, quality=0.0)
            continue

        if _branch_cut_left(a):
            xq_a = _x_scaled(float(x[a]), P_bar)
            if first_valid is not None and a == first_valid:
                # Low-U timeout desert: first valid sample (~0.4 scaled).
                _add(_endpoint(a), quality=xq_a)
            elif xq_a < x_tol:
                _add(_endpoint(a), quality=xq_a)
            elif x[a] * (x[a + 1] - x[a]) > 0.0:
                g = _extrapolate(a, a + 1)
                if g[0] < float(u[a]) and float(u[a]) - g[0] <= max_reach:
                    _add(g, quality=0.0)

        if _branch_cut_right(b):
            xq_b = _x_scaled(float(x[b]), P_bar)
            if xq_b < x_tol:
                _add(_endpoint(b), quality=xq_b)
            elif x[b] * (x[b] - x[b - 1]) < 0.0:
                g = _extrapolate(b - 1, b)
                if g[0] > float(u[b]) and g[0] - float(u[b]) <= max_reach:
                    _add(g, quality=0.0)

    p_inv = abs(P_bar) ** (-1.0 / 3.0)
    merge_tol = max(0.12, 8.0 * du * p_inv)

    out: List[Tuple[float, float, float]] = []
    for g, q in sorted(candidates, key=lambda t: (t[0][0] * p_inv, t[1])):
        if not all(np.isfinite(g)):
            continue
        if not (0.2 < g[2] < FIG16_S1_MAX):
            continue
        u_scaled = float(g[0]) * p_inv
        merged = False
        for k, h in enumerate(out):
            if abs(float(h[0]) * p_inv - u_scaled) < merge_tol:
                merged = True
                break
        if not merged:
            out.append(g)
    out.sort(key=lambda t: t[0])
    return out

def scan_branch_seeds(
    c0: float,
    *,
    P_bar_values: Optional[List[float]] = None,
    x_tol: float = NEAR_ZERO_X_DEFAULT,
) -> List[Tuple[float, float, float]]:
    """Run the default ``U(0)`` scan and collect near-zero seeds."""
    if P_bar_values is None:
        P_bar_values = [1.0]
    all_guesses: List[Tuple[float, float, float]] = []
    for P_bar in P_bar_values:
        grid = _default_u0_grid(c0, P_bar=P_bar)
        sigma_bar = sigma_bar_from_P(P_bar)
        u, xs, us, s1 = scan_u0(c0, grid, sigma_bar=sigma_bar, P_bar=P_bar)
        all_guesses.extend(near_zero_seeds(u, xs, us, s1, P_bar=P_bar, x_tol=x_tol))
    out: List[Tuple[float, float, float]] = []
    for g in all_guesses:
        if not all(np.isfinite(g)) or not (0.2 < g[2] < FIG16_S1_MAX):
            continue
        if not any(abs(g[0] - t[0]) < 0.08 for t in out):
            out.append(g)
    out.sort(key=lambda t: t[0])
    return out

def integrate_stage1_profile(
    U0: float,
    *,
    c0: float,
    P_bar: float,
    sigma_bar: Optional[float] = None,
    n: int = 280,
    S_max: Optional[float] = None,
    n_s1: int = 1,
    max_step: float = FIG16_SCAN_MAX_STEP,
    timeout: Optional[float] = 45.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """North-leg profile to ``S₁^{(n)}`` without Fig.~16 ``X`` discard (for root plots)."""
    s_cap = float(S_max if S_max is not None else max(FIG16_SCAN_S_MAX, 120.0))
    out, timed_out = _call_with_timeout(
        timeout,
        _integrate_stage1_profile_impl,
        U0,
        c0=c0,
        P_bar=P_bar,
        sigma_bar=sigma_bar,
        n=n,
        S_max=s_cap,
        n_s1=n_s1,
        max_step=max_step,
    )
    if timed_out:
        return None
    return out

def _integrate_stage1_profile_impl(
    U0: float,
    *,
    c0: float,
    P_bar: float,
    sigma_bar: Optional[float] = None,
    n: int = 280,
    S_max: float = FIG16_SCAN_S_MAX,
    n_s1: int = 1,
    max_step: float = FIG16_SCAN_MAX_STEP,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if sigma_bar is None:
        sigma_bar = sigma_bar_from_P(P_bar)
    if max_step >= FIG16_SCAN_MAX_STEP:
        rtol, atol = FIG16_SCAN_RTOL, FIG16_SCAN_ATOL
    else:
        rtol, atol = 1e-7, 1e-9
    C0 = C0_dimensional(c0, A_STAR)
    rhs = lambda S, y: meridian_rhs(S, y, C0, sigma_bar, P_bar)
    S_cur = S_POLE
    y_cur = north_pole_state(U0, C0=C0, sigma_bar=sigma_bar, P_bar=P_bar)
    s_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    yf: Optional[np.ndarray] = None
    n_per = max(24, int(n // max(n_s1, 1)))
    events = _u0_scan_events(n=n_s1)

    for k in range(1, n_s1 + 1):
        out = _solve_u0_ivp(
            rhs, (S_cur, S_max), y_cur, events,
            max_step=max_step, rtol=rtol, atol=atol,
        )
        if out is None:
            return None
        ivp, ei, s_hit = out
        if ei != 0:
            return None
        yf = ivp.y[:, -1].copy()
        ivp_dense = solve_ivp(
            rhs, (S_cur, s_hit), y_cur, method="RK45", dense_output=True,
            rtol=rtol, atol=atol, max_step=max_step,
        )
        if not ivp_dense.success or ivp_dense.sol is None:
            return None
        s_seg = np.linspace(S_cur, s_hit, n_per)
        y_seg = ivp_dense.sol(s_seg)
        s_parts.append(s_seg)
        y_parts.append(y_seg)
        if k < n_s1:
            S_cur = s_hit + 1e-4
            y_cur = yf
            u_dir = float(np.sign(y_cur[3])) or 1.0
            y_cur[2] = PSI_SOUTH + u_dir * 1e-3

    assert yf is not None and s_parts
    s_full = np.concatenate(s_parts)
    y_full = np.concatenate(y_parts, axis=1)
    return s_full, y_full[0], y_full[1], yf

STAGE2_BOUNDS = (
    [STAGE2_U0_MIN, STAGE2_U1_MIN, STAGE2_S1_MIN],
    [STAGE2_U0_MAX, STAGE2_U1_MAX, STAGE2_S1_MAX],
)

def u0_grid(
    u_min: float,
    u_max: float,
    *,
    step: Optional[float] = None,
    n_points: int = FIG16_SCAN_GRID_POINTS,
    P_bar: float = 1.0,
) -> np.ndarray:
    """Raw ``U(0)`` grid for a scan window (barred units × ``P̄^{1/3}`` if ``P̄≠1``).

    Pass ``step`` for uniform spacing, or omit it to target ``n_points`` samples
    (recommended for wide ranges so runtime stays bounded).
    """
    p_pos = abs(P_bar) ** (1.0 / 3.0)
    scaled = np.linspace(u_min, u_max, n_points) if step is None else np.arange(
        u_min, u_max + 0.5 * step, step,
    )
    return np.asarray(scaled, dtype=float) * p_pos

def _stitch_meridian_profile(
    sn: np.ndarray,
    ss: np.ndarray,
    yn: np.ndarray,
    ys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Join north/south legs; fix Z gauge (shape equations are Z-translation invariant)."""
    ys = np.asarray(ys, dtype=float).copy()
    ys[1, :] += float(yn[1, -1] - ys[1, -1])  # match Z at interior junction S̄

    s_full = np.concatenate([sn, ss[-2::-1]])
    y_full = np.concatenate([np.asarray(yn, dtype=float), ys[:, -2::-1]], axis=1).copy()
    y_full[1, :] -= float(y_full[1, 0])  # north pole Z = 0
    return s_full, y_full

def _integrate_two_legs(
    U0: float,
    U1: float,
    S1: float,
    S_bar: float,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    v: float,
    *,
    dense: bool = False,
    n: int = 200,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: float = STAGE2_MAX_STEP,
    enforce_fig16: bool = True,
    A_c0: float = A_STAR,
    A_target: float = A_STAR,
):
    """North leg S₀→S̄ and south leg S₁→S̄ (paper: integrate backward from south pole)."""
    S_south = S1 - S_POLE
    if S1 <= 2.0 * S_POLE + 0.05 or S_bar <= S_POLE + 0.02 or S_bar >= S_south - 0.02:
        return None

    C0 = C0_for_shape(c0, A_c0)
    rhs = lambda S, y: meridian_rhs(S, y, C0, sigma_bar, P_bar)
    y_n = north_pole_state(U0, C0=C0, sigma_bar=sigma_bar, P_bar=P_bar)
    A1 = float(A_target)
    V1 = float(V_at_area(v, A_target))
    y_s = south_pole_state_totals(
        U1, A1, V1, C0=C0, sigma_bar=sigma_bar, P_bar=P_bar,
    )
    ivp_n = _integrate_leg(
        rhs, (S_POLE, S_bar), y_n,
        max_step=max_step, rtol=rtol, atol=atol, dense=dense,
        enforce_fig16=enforce_fig16,
    )
    ivp_s = _integrate_leg(
        rhs, (S_south, S_bar), y_s,
        max_step=max_step, rtol=rtol, atol=atol, dense=dense,
        enforce_fig16=enforce_fig16,
    )
    if ivp_n is None or ivp_s is None:
        return None

    if dense:
        sn = np.linspace(S_POLE, S_bar, n // 2)
        ss = np.linspace(S_south, S_bar, n // 2)
        if ivp_n.sol is None or ivp_s.sol is None:
            return None
        yn = ivp_n.sol(sn)
        ys = ivp_s.sol(ss)
        if enforce_fig16 and not (
            _fig16_leg_psi_valid(yn[2], n=1) and _fig16_leg_psi_valid(ys[2], n=1)
        ):
            return None
        s_full, y_full = _stitch_meridian_profile(sn, ss, yn, ys)
        return s_full, y_full, ivp_n.y[:, -1], ivp_s.y[:, -1]

    return ivp_n.y[:, -1], ivp_s.y[:, -1]

def _north_state_at_S1(
    U0: float,
    S1: float,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = STAGE2_MAX_STEP,
    A_c0: float = A_STAR,
) -> Optional[np.ndarray]:
    """Integrate north pole → ``S₁``; return endpoint state (for ``A(S₁)``, ``V(S₁)``)."""
    if S1 <= S_POLE + 1e-4:
        return None
    C0 = C0_for_shape(c0, A_c0)
    rhs = lambda S, y: meridian_rhs(S, y, C0, sigma_bar, P_bar)
    ivp = solve_ivp(
        rhs, (S_POLE, S1),
        north_pole_state(U0, C0=C0, sigma_bar=sigma_bar, P_bar=P_bar),
        method="RK45",
        rtol=rtol, atol=atol, max_step=max_step,
    )
    if not ivp.success:
        return None
    return ivp.y[:, -1]

def _south_pole_totals_from_probe(
    U0: float,
    S1: float,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = STAGE2_MAX_STEP,
    A_c0: float = A_STAR,
) -> Optional[Tuple[float, float]]:
    """``(A(S₁), V(S₁))`` by integrating the north leg to arclength ``S₁``."""
    yf = _north_state_at_S1(
        U0, S1, sigma_bar, P_bar, c0, rtol=rtol, atol=atol, max_step=max_step, A_c0=A_c0,
    )
    if yf is None:
        return None
    return float(yf[5]), float(yf[6])

def _integrate_two_legs_stage2(
    U0: float,
    U1: float,
    S1: float,
    S_bar: float,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    *,
    dense: bool = False,
    n: int = 200,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = STAGE2_MAX_STEP,
    enforce_fig16: bool = True,
    A1_hint: Optional[float] = None,
    V1_hint: Optional[float] = None,
    A_c0: float = A_STAR,
):
    """Stage-2 two-leg shoot (Appendix B): north ``S₀→S̄``, south ``S₁→S̄``.

    The regularized south IC is the state a distance ``S_POLE`` before the south
    pole, so the south leg starts at ``S₁ − S_POLE``.  Totals ``A, V`` are taken
    from the north leg at that same arclength.

    When ``enforce_fig16=False``, integrate without ψ-window discard so failed
    boundary conditions can still be plotted.
    """
    S_south = S1 - S_POLE
    if S1 <= 2.0 * S_POLE + 0.05 or S_bar <= S_POLE + 0.02 or S_bar >= S_south - 0.02:
        return None

    if (
        A1_hint is not None
        and V1_hint is not None
        and np.isfinite(A1_hint)
        and np.isfinite(V1_hint)
    ):
        A1, V1 = float(A1_hint), float(V1_hint)
    else:
        totals = _south_pole_totals_from_probe(
            U0, S_south, sigma_bar, P_bar, c0, rtol=rtol, atol=atol, max_step=max_step,
            A_c0=A_c0,
        )
        if totals is None:
            return None
        A1, V1 = totals

    C0 = C0_for_shape(c0, A_c0)
    rhs = lambda S, y: meridian_rhs(S, y, C0, sigma_bar, P_bar)
    y_n = north_pole_state(U0, C0=C0, sigma_bar=sigma_bar, P_bar=P_bar)
    y_s = south_pole_state_totals(
        U1, A1, V1, C0=C0, sigma_bar=sigma_bar, P_bar=P_bar,
    )
    ivp_n = _integrate_leg(
        rhs, (S_POLE, S_bar), y_n,
        max_step=max_step, rtol=rtol, atol=atol, dense=False,
        enforce_fig16=enforce_fig16,
    )
    ivp_s = _integrate_leg(
        rhs, (S_south, S_bar), y_s,
        max_step=max_step, rtol=rtol, atol=atol, dense=dense,
        enforce_fig16=enforce_fig16,
    )
    if ivp_n is None or ivp_s is None:
        return None

    if dense:
        sn = np.linspace(S_POLE, S_bar, n // 2)
        ss = np.linspace(S_south, S_bar, n // 2)
        ivp_n_dense = _integrate_leg(
            rhs, (S_POLE, S_bar), y_n,
            max_step=max_step, rtol=rtol, atol=atol, dense=True,
            enforce_fig16=enforce_fig16,
        )
        if ivp_n_dense is None or ivp_s.sol is None:
            return None
        yn_d = ivp_n_dense.sol(sn)
        ys = ivp_s.sol(ss)
        if enforce_fig16 and not (
            _fig16_leg_psi_valid(yn_d[2], n=1) and _fig16_leg_psi_valid(ys[2], n=1)
        ):
            return None
        s_full, y_full = _stitch_meridian_profile(sn, ss, yn_d, ys)
        return s_full, y_full, ivp_n.y[:, -1], ivp_s.y[:, -1]

    return ivp_n.y[:, -1], ivp_s.y[:, -1]

def _integrate_two_legs_totals(
    U0: float,
    U1: float,
    S1: float,
    S_bar: float,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    *,
    dense: bool = False,
    n: int = 200,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: float = STAGE2_MAX_STEP,
    enforce_fig16: bool = True,
    A1_hint: Optional[float] = None,
    V1_hint: Optional[float] = None,
    A_c0: float = A_STAR,
):
    """Alias kept for callers; stage-2 path uses the probe + short-north integrator."""
    return _integrate_two_legs_stage2(
        U0, U1, S1, S_bar, sigma_bar, P_bar, c0,
        dense=dense, n=n, rtol=rtol, atol=atol, max_step=max_step,
        enforce_fig16=enforce_fig16, A1_hint=A1_hint, V1_hint=V1_hint, A_c0=A_c0,
    )

def _leg_psi_diagnostics(psi: np.ndarray) -> Dict[str, float]:
    psi = np.asarray(psi, dtype=float)
    if psi.size == 0:
        return {"psi_min": float("nan"), "psi_max": float("nan"), "fig16_ok": 0.0}
    return {
        "psi_min": float(np.min(psi)),
        "psi_max": float(np.max(psi)),
        "fig16_ok": float(_fig16_leg_psi_valid(psi, n=1)),
    }

def _integrate_north_leg_profile(
    U0: float,
    S_end: float,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    *,
    n: int = 200,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = STAGE2_MAX_STEP,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Single north-leg profile ``S₀→S_end`` (fallback when two-leg fails)."""
    if S_end <= S_POLE + 0.02:
        return None
    C0 = C0_dimensional(c0, A_STAR)
    rhs = lambda S, y: meridian_rhs(S, y, C0, sigma_bar, P_bar)
    y_n = north_pole_state(U0, C0=C0, sigma_bar=sigma_bar, P_bar=P_bar)
    ivp = _integrate_leg_raw(
        rhs, (S_POLE, S_end), y_n,
        max_step=max_step, rtol=rtol, atol=atol, dense=True,
    )
    if ivp is None or ivp.sol is None:
        return None
    s = np.linspace(S_POLE, S_end, n)
    return s, ivp.sol(s)

def _stage2_closure_residual(
    x: np.ndarray,
    *,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    junction_frac: float = JUNCTION_FRAC,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    A_c0: float = A_STAR,
    enforce_fig16: bool = False,
) -> np.ndarray:
    """Appendix B stage 2: match ``ψ, U, X`` at ``S̄``; ``γ`` via conserved ``H``.

    Fig.~16 ψ-window discard is off by default (same as :func:`_shoot_two_leg`
    residuals): stage-2 only cares about junction closure, and stomatocyte legs
    often leave the Fig.~16 scan window while still being valid shoots.
    """
    U0, U1, S1 = map(float, x)
    S_bar = junction_frac * S1
    out = _integrate_two_legs_totals(
        U0, U1, S1, S_bar, sigma_bar, P_bar, c0,
        rtol=rtol, atol=atol, A_c0=A_c0, enforce_fig16=enforce_fig16,
    )
    if out is None:
        return np.full(3, 1e3)
    yn, ys = out
    return _junction_closure_residual(yn, ys)

def _stage2_closure_residual_safe(
    x: np.ndarray,
    *,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    junction_frac: float = JUNCTION_FRAC,
    timeout: float = STAGE2_RESIDUAL_TIMEOUT,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    A_c0: float = A_STAR,
    enforce_fig16: bool = False,
) -> np.ndarray:
    """Like :func:`_stage2_closure_residual` but skips pathological ``U₁`` trials."""
    out, timed_out = _call_with_timeout(
        timeout,
        _stage2_closure_residual,
        x,
        sigma_bar=sigma_bar,
        P_bar=P_bar,
        c0=c0,
        junction_frac=junction_frac,
        rtol=rtol,
        atol=atol,
        A_c0=A_c0,
        enforce_fig16=enforce_fig16,
    )
    if timed_out or out is None:
        return np.full(3, 1e3)
    return out

def _stage2_local_bounds(
    U0_seed: float,
    U1_seed: float,
    S1_seed: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tight ``(U₀, U₁, S₁)`` box around the stage-1 root."""
    u0_half = max(STAGE2_U0_ABS, STAGE2_U0_REL * abs(U0_seed))
    u1_half = max(STAGE2_U1_ABS, STAGE2_U1_REL * max(abs(U1_seed), abs(U0_seed), 0.05))
    s1_rel = STAGE2_S1_REL_LONG if S1_seed > 8.0 else STAGE2_S1_REL
    lb = np.array([
        U0_seed - u0_half,
        U1_seed - u1_half,
        (1.0 - s1_rel) * S1_seed,
    ], dtype=float)
    ub = np.array([
        U0_seed + u0_half,
        U1_seed + u1_half,
        (1.0 + s1_rel) * S1_seed,
    ], dtype=float)
    lb = np.maximum(lb, np.array(STAGE2_BOUNDS[0], dtype=float))
    ub = np.minimum(ub, np.array(STAGE2_BOUNDS[1], dtype=float))
    for i in range(3):
        if ub[i] <= lb[i]:
            mid = 0.5 * (float(lb[i]) + float(ub[i]))
            eps = max(1e-9, 1e-6 * max(abs(mid), 1.0))
            lb[i], ub[i] = mid - eps, mid + eps
    return lb, ub

def shoot_appendix_b_stage2(
    U0: float,
    U1: float,
    S1: float,
    *,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    branch: str = "appendix_b",
    n: int = 200,
    verbose: bool = False,
    junction_frac: Optional[float] = None,
    A_c0: float = A_STAR,
    max_nfev: int = 80,
) -> MeridianSolution:
    """Appendix B stage 2: fixed ``(Σ̄, P̄, C₀)``; shoot ``(U₀, U₁, S₁)`` for junction closure.

    Seeds from the stage-1 root and refines with a bounded TRF least-squares shoot
    in a small box around ``(U₀*, U₁*, S₁*)``.

    ``junction_frac`` sets ``S̄ = junction_frac · S₁`` (default :data:`JUNCTION_FRAC`).
    """
    j_frac = JUNCTION_FRAC if junction_frac is None else float(junction_frac)
    lb, ub = _stage2_local_bounds(U0, U1, S1)
    x0 = np.clip(np.array([float(U0), float(U1), float(S1)], dtype=float), lb, ub)
    r0n = float(np.linalg.norm(_stage2_closure_residual_safe(
        x0, sigma_bar=sigma_bar, P_bar=P_bar, c0=c0, junction_frac=j_frac, A_c0=A_c0,
    )))
    if verbose:
        print(
            f"  stage2 seed: U₀={x0[0]:.4f} U₁={x0[1]:.4f} S₁={x0[2]:.3f}  |res|₀={r0n:.3e}"
        )
        print(
            f"  stage2 box:  U₀∈[{lb[0]:.3f},{ub[0]:.3f}]  "
            f"U₁∈[{lb[1]:.3f},{ub[1]:.3f}]  "
            f"S₁∈[{lb[2]:.3f},{ub[2]:.3f}]"
        )
    if r0n >= 900.0:
        msg = "integration failed at seed"
        if verbose:
            print(f"  stage2 {msg}")
        return MeridianSolution(
            0.0, c0, U0, U1, sigma_bar, P_bar, np.array([]), np.zeros((7, 0)),
            branch, False, msg, {}, S1, j_frac * S1,
        )

    opt = least_squares(
        lambda x: _stage2_closure_residual_safe(
            x, sigma_bar=sigma_bar, P_bar=P_bar, c0=c0, junction_frac=j_frac, A_c0=A_c0,
        ),
        x0,
        method="trf",
        bounds=(lb, ub),
        ftol=1e-10,
        xtol=1e-10,
        max_nfev=int(max_nfev),
    )
    U0f, U1f, S1f = map(float, opt.x)
    opt_fun = opt.fun
    S_bar = j_frac * S1f
    out = _integrate_two_legs_totals(
        U0f, U1f, S1f, S_bar, sigma_bar, P_bar, c0,
        dense=True, n=n, rtol=1e-8, atol=1e-10, A_c0=A_c0,
        enforce_fig16=False,
    )
    if out is None:
        return MeridianSolution(
            0.0, c0, U0f, U1f, sigma_bar, P_bar, np.array([]), np.zeros((7, 0)),
            branch, False, "stage-2 integration failed", {}, S1f, S_bar,
        )
    s, y, yn, ys = out
    v_hat = float(reduced_volume_from_AV(float(y[5, -1]), float(y[6, -1])))
    constraints = _constraints_from_junction(yn, ys, v_hat, S1f, S_bar)
    constraints["A_end"] = float(y[5, -1])
    constraints["V_end"] = float(y[6, -1])
    ok = float(np.linalg.norm(opt_fun)) < 5e-3
    ok = ok and abs(constraints["psi_match"]) < 0.02 and abs(constraints["X_match"]) < 0.02
    if verbose:
        print(
            f"  stage2 U0={U0f:.3f} U1={U1f:.3f} S1={S1f:.3f} v̄={v_hat:.4f} "
            f"|res|={np.linalg.norm(opt_fun):.2e} ok={ok}"
        )
    return MeridianSolution(
        v_hat, c0, U0f, U1f, sigma_bar, P_bar, s, y, branch, ok,
        "appendix B stage 2" if ok else "stage-2 shoot",
        constraints, S1f, S_bar,
    )

def _two_leg_residual(
    x: np.ndarray,
    v: float,
    c0: float,
    rtol: float,
    atol: float,
    *,
    A_target: float = A_STAR,
    A_c0: float = A_STAR,
    enforce_fig16: bool = True,
) -> np.ndarray:
    """Match ψ, U, X, A, V at interior junction S̄ (γ via conserved H)."""
    U0, U1, S1, sigma_bar, P_bar = x
    S_bar = JUNCTION_FRAC * S1
    out = _integrate_two_legs(
        U0, U1, S1, S_bar, sigma_bar, P_bar, c0, v,
        rtol=rtol, atol=atol, enforce_fig16=enforce_fig16, A_c0=A_c0,
        A_target=A_target,
    )
    if out is None:
        return np.full(5, 1e3)
    yn, ys = out
    scale_u = max(abs(yn[3]), abs(ys[3]), 1.0)
    scale_x = max(yn[0], ys[0], 0.05)
    v_tgt = V_at_area(v, A_target)
    return np.array([
        (yn[2] - ys[2]) / np.pi,
        (yn[3] - ys[3]) / scale_u,
        (yn[0] - ys[0]) / scale_x,
        (yn[5] - ys[5]) / A_target,
        (yn[6] - ys[6]) / max(abs(v_tgt), 1e-12),
    ])

def _two_leg_residual_safe(
    x: np.ndarray,
    v: float,
    c0: float,
    rtol: float,
    atol: float,
    *,
    A_target: float = A_STAR,
    A_c0: float = A_STAR,
    timeout: float = STAGE3_RESIDUAL_TIMEOUT,
    enforce_fig16: bool = True,
) -> np.ndarray:
    out, timed_out = _call_with_timeout(
        timeout, _two_leg_residual, x, v, c0, rtol, atol,
        A_target=A_target, A_c0=A_c0, enforce_fig16=enforce_fig16,
    )
    if timed_out or out is None:
        return np.full(5, 1e3)
    return out

def _constraints_from_junction(
    yn: np.ndarray,
    ys: np.ndarray,
    v: float,
    S1: float,
    S_bar: float,
    *,
    A_target: float = A_STAR,
) -> Dict[str, float]:
    v_tgt = V_at_area(v, A_target)
    return {
        "psi_match": float(yn[2] - ys[2]),
        "U_match": float(yn[3] - ys[3]),
        "X_match": float(yn[0] - ys[0]),
        "A_match": float(yn[5] - ys[5]),
        "V_match": float(yn[6] - ys[6]),
        "A_error_rel": float((yn[5] - A_target) / A_target),
        "V_error_rel": float((yn[6] - v_tgt) / max(abs(v_tgt), 1e-12)),
        "A_target": float(A_target),
        "S1": float(S1),
        "S_bar": float(S_bar),
    }

def integrate_appendix_b_stage2(
    U0: float,
    U1: float,
    S1: float,
    *,
    sigma_bar: float,
    P_bar: float,
    c0: float,
    branch: str = "appendix_b_seed",
    n: int = 200,
    junction_frac: Optional[float] = None,
    S_bar: Optional[float] = None,
    north_frac: Optional[float] = None,
    enforce_fig16: bool = False,
    A1_hint: Optional[float] = None,
    V1_hint: Optional[float] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: Optional[float] = None,
    A_c0: float = A_STAR,
) -> MeridianSolution:
    """Two-leg meridian at stage-1 ``(U₀, U₁, S₁*)`` — no parameter shoot.

    Junction placement (pick one):

    * ``junction_frac`` — ``S̄ = junction_frac · S₁``
    * ``S_bar`` — explicit interior arclength
    * ``north_frac`` — north leg ``= S_POLE + north_frac · (S₁ − 2 S_POLE)``

    By default ``enforce_fig16=False`` so a profile is returned even when ψ loops
    or junction BCs fail (for diagnostic plots).  Junction mismatches and Fig.~16
    leg flags are stored in ``constraints``.
    """
    U0f, U1f, S1f = map(float, (U0, U1, S1))
    try:
        s_bar = _resolve_S_bar(
            S1f, junction_frac=junction_frac, S_bar=S_bar, north_frac=north_frac,
        )
    except ValueError as exc:
        return MeridianSolution(
            0.0, c0, U0f, U1f, sigma_bar, P_bar, np.array([]), np.zeros((7, 0)),
            branch, False, str(exc), {}, S1f, float("nan"),
        )
    north_len, south_len = _two_leg_lengths(S1f, s_bar)
    step = STAGE2_MAX_STEP if max_step is None else float(max_step)

    out = _integrate_two_legs_totals(
        U0f, U1f, S1f, s_bar, sigma_bar, P_bar, c0,
        dense=True, n=n, rtol=rtol, atol=atol, max_step=step,
        enforce_fig16=enforce_fig16,
        A1_hint=A1_hint, V1_hint=V1_hint, A_c0=A_c0,
    )
    partial = False
    if out is None:
        north_only = _integrate_north_leg_profile(
            U0f, min(s_bar, S1f - S_POLE), sigma_bar, P_bar, c0, n=n,
        )
        if north_only is None:
            return MeridianSolution(
                0.0, c0, U0f, U1f, sigma_bar, P_bar, np.array([]), np.zeros((7, 0)),
                branch, False, "integration failed (no profile)", {
                    "S_bar": s_bar,
                    "junction_frac": s_bar / S1f,
                    "north_length": north_len,
                    "south_length": south_len,
                }, S1f, s_bar,
            )
        s, y = north_only
        partial = True
        yn = ys = y[:, -1]
        res_norm = float("inf")
        msg = "partial north leg only (two-leg failed)"
    else:
        s, y, yn, ys = out
        res_norm = float(np.linalg.norm(_junction_closure_residual(yn, ys)))
        msg = "stage-1 seed on two-leg integrator"

    v_hat = float(reduced_volume_from_AV(float(y[5, -1]), float(y[6, -1])))
    constraints = _constraints_from_junction(yn, ys, v_hat, S1f, s_bar)
    constraints["A_end"] = float(y[5, -1])
    constraints["V_end"] = float(y[6, -1])
    constraints["closure_res"] = res_norm
    constraints["junction_frac"] = s_bar / S1f
    constraints["north_length"] = north_len
    constraints["south_length"] = south_len
    constraints["partial_profile"] = float(partial)

    if not partial and y.shape[1] > 1:
        j_idx = int(np.argmin(np.abs(s - s_bar)))
        north_diag = _leg_psi_diagnostics(y[2, : j_idx + 1])
        south_diag = _leg_psi_diagnostics(y[2, j_idx:])
        for key, val in north_diag.items():
            constraints[f"north_{key}"] = val
        for key, val in south_diag.items():
            constraints[f"south_{key}"] = val

    ok = (not partial) and np.isfinite(res_norm) and res_norm < 5e-3
    ok = ok and abs(constraints["psi_match"]) < 0.02 and abs(constraints["X_match"]) < 0.02
    if not ok and not partial:
        msg = "stage-1 seed (junction open)"
    if constraints.get("north_fig16_ok") == 0.0 or constraints.get("south_fig16_ok") == 0.0:
        msg += " [Fig. 16 ψ loop/window on leg]"
    return MeridianSolution(
        v_hat, c0, U0f, U1f, sigma_bar, P_bar, s, y, branch, ok,
        msg, constraints, S1f, s_bar,
    )

def stage2_area(sol: MeridianSolution) -> float:
    """Total area from an Appendix B stage-2 solution (free ``A``, not necessarily ``4π``)."""
    a = sol.constraints.get("A_end")
    if a is not None and np.isfinite(a) and a > 0.0:
        return float(a)
    if sol.y.size and sol.y.shape[1] > 0:
        return float(sol.y[5, -1])
    return float("nan")

def rescale_shoot_params_by_length(
    U0: float,
    U1: float,
    S1: float,
    sigma_bar: float,
    P_bar: float,
    scale: float,
) -> Tuple[float, float, float, float, float]:
    """Uniform meridian length scale ``s`` (thin-neck similarity).

    ``S' = s S``, ``U' = U/s`` (curvature ∝ 1/length),
    ``U₀|P̄|^{-1/3}`` and ``Σ̄|P̄|^{-2/3}`` invariant
    ⇒ ``P̄' = P̄ (U₀'/U₀)³``, ``Σ̄' = Σ̄ (U₀'/U₀)²``.
    """
    s = float(scale)
    if not np.isfinite(s) or s <= 0.0:
        return float(U0), float(U1), float(S1), float(sigma_bar), float(P_bar)
    U0n = float(U0) / s
    U1n = float(U1) / s
    S1n = float(S1) * s
    r_u = U0n / float(U0) if abs(U0) > 1e-14 else s ** -1
    Pn = float(P_bar) * r_u ** 3
    sig_n = float(sigma_bar) * r_u ** 2
    return U0n, U1n, S1n, sig_n, Pn

def rescale_pear_params_to_landmark(
    stage2: MeridianSolution,
    *,
    A_target: float,
    S1_reference: Optional[float] = None,
) -> Dict[str, float]:
    """Rescale stage-2 shoot parameters to target area ``A_target``.

    Uniform similarity with ``s = √(A_target / A_stage2)`` (not ``S₁`` ratio).
    Returns rescaled ``(U₀,U₁,S₁,Σ̄,P̄)`` and diagnostics including implied ``S₁``.
    """
    A_src = stage2_area(stage2)
    if not np.isfinite(A_target) or A_target <= 0.0:
        raise ValueError("rescale_pear_params_to_landmark needs positive finite A_target")
    if not np.isfinite(A_src) or A_src <= 0.0:
        raise ValueError("rescale_pear_params_to_landmark needs positive finite stage-2 area")
    scale = float(np.sqrt(A_target / A_src))
    U0, U1, S1, sig, P = rescale_shoot_params_by_length(
        stage2.U0, stage2.U1, stage2.S1, stage2.sigma_bar, stage2.P_bar, scale,
    )
    S1_src = float(stage2.S1)
    out: Dict[str, float] = {
        "U0": U0, "U1": U1, "S1": S1,
        "sigma_bar": sig, "P_bar": P,
        "length_scale": scale,
        "A_target": float(A_target),
        "A_source": float(A_src),
        "area_scale_sq": float(A_target / A_src),
        "S1_source": S1_src,
        "S1_scaled": float(S1),
    }
    if S1_reference is not None and np.isfinite(S1_reference):
        out["S1_reference"] = float(S1_reference)
    return out

def C0_for_length_scale(C0: float, scale: float) -> float:
    """Dimensional ``C₀`` under meridian length scale ``s`` (``C₀ ∝ 1/s``, like ``U``).

    Reduced ``c₀ = C₀ R₀`` is unchanged when ``R₀ → s R₀`` and ``C₀ → C₀/s``.
    """
    s = float(scale)
    if not np.isfinite(s) or s <= 0.0:
        return float(C0)
    return float(C0) / s

def integrate_stage25_pear_at_D(
    stage2: MeridianSolution,
    d_tip: MeridianSolution,
    *,
    C0: float = 1.0,
    branch: str = "stage25",
    n: int = 120,
    junction_frac: Optional[float] = None,
    enforce_fig16: bool = False,
    verbose: bool = False,
) -> MeridianSolution:
    """Stage 2.5: rescaled pear at D area (no shoot).

    Rescales stage-2 ``(U₀,U₁,S₁,Σ̄,P̄)`` with ``s = √(A_D / A_stage2)``.
    Under that length scale, reduced ``c₀ = C₀ R₀`` is preserved
    (``C₀ → C₀/s``, like ``U``); the integrator uses ``C₀/s``.  Volume is stage 3.

    Uses similarity-scaled ``(A, V)`` hints so the integrator skips the expensive
    full-meridian north-leg probe that otherwise stalls on rescaled pears.
    """
    t0 = time.perf_counter()
    A_target = float(d_tip.constraints.get("A_phys", stage2_area(d_tip)))
    if not np.isfinite(A_target) or A_target <= 0.0:
        raise ValueError("d_tip must carry A_phys in constraints")
    params = rescale_pear_params_to_landmark(
        stage2, A_target=A_target, S1_reference=float(d_tip.S1),
    )
    if junction_frac is None:
        jf = stage2.constraints.get("junction_frac")
        if jf is not None and np.isfinite(jf):
            junction_frac = float(jf)
        elif stage2.S1 > 0.0 and np.isfinite(stage2.S_bar):
            junction_frac = float(stage2.S_bar / stage2.S1)
    scale = float(params["length_scale"])
    A_src = stage2_area(stage2)
    V_src = stage2.constraints.get("V_end")
    if (V_src is None or not np.isfinite(V_src)) and stage2.y.size:
        V_src = float(stage2.y[6, -1])
    A1_hint = V1_hint = None
    if np.isfinite(A_src) and A_src > 0.0 and V_src is not None and np.isfinite(V_src):
        A1_hint = float(A_target)
        V1_hint = float(V_src * scale ** 3)
    max_step = STAGE2_MAX_STEP * max(1.0, float(params["S1"]) / 2.5)
    C0_int = C0_for_length_scale(C0, scale)
    c0_preserved = (
        c0_reduced(C0_int, A_target)
        if np.isfinite(A_target) and A_target > 0.0
        else float("nan")
    )
    if verbose:
        ah = f"{A1_hint:.3f}" if A1_hint is not None else "probe"
        vh = f"{V1_hint:.3f}" if V1_hint is not None else "probe"
        s1_ref = params.get("S1_reference", float("nan"))
        print(
            f"  stage2.5 rescale: s=√(A_D/A₂)={scale:.4f}  "
            f"A={A_src:.3f}→{A_target:.3f}  S₁={params['S1_source']:.3f}→{params['S1']:.3f}"
            f"  (D S₁={s1_ref:.3f})",
            flush=True,
        )
        print(
            f"  stage2.5 integrate: C₀={C0_int:.4f} (={C0:g}/s)  c₀={c0_preserved:.4f}  "
            f"hints A={ah} V={vh}  max_step={max_step:.3f}",
            flush=True,
        )
    sol = integrate_appendix_b_stage2(
        params["U0"], params["U1"], params["S1"],
        sigma_bar=params["sigma_bar"],
        P_bar=params["P_bar"],
        c0=c0_preserved,
        branch=branch,
        n=n,
        junction_frac=junction_frac,
        enforce_fig16=enforce_fig16,
        A1_hint=A1_hint,
        V1_hint=V1_hint,
        rtol=1e-7,
        atol=1e-9,
        max_step=max_step,
        A_c0=A_target,
    )
    if verbose:
        print(f"  stage2.5 integrate done ({time.perf_counter() - t0:.2f}s)", flush=True)
    c0_D = c0_reduced(C0, A_target)
    sol = replace(
        sol,
        c0=c0_preserved,
        branch=branch,
        constraints={
            **sol.constraints,
            **params,
            "A_target": A_target,
            "A_phys": A_target,
            "V_phys_target": float(d_tip.constraints.get("V_phys", float("nan"))),
            "c0_preserved": c0_preserved,
            "c0_reduced_D": c0_D,
            "C0": float(C0),
            "C0_integrate": C0_int,
            "stage": "2.5",
        },
    )
    A_end = stage2_area(sol)
    if np.isfinite(A_end) and A_target > 0.0:
        sol.constraints["A_error_rel_D"] = float((A_end - A_target) / A_target)
    return sol

def _prev_at_target_area(prev: MeridianSolution, A_target: float, tol: float = 0.08) -> bool:
    """True when ``prev`` is an integrated solution at ``A_target``."""
    if len(prev.s) == 0:
        return False
    A = stage2_area(prev)
    if not np.isfinite(A) or A <= 0.0:
        return False
    return abs(A - A_target) / A_target < tol

def _prev_at_unit_area(prev: MeridianSolution, tol: float = 0.08) -> bool:
    """True when ``prev`` is already a stage-3 / phase-diagram solution at ``A = 4π``."""
    return _prev_at_target_area(prev, A_STAR, tol=tol)

def _warm_start_x5(prev: MeridianSolution) -> np.ndarray:
    """``(U₀,U₁,S₁,Σ̄,P̄)`` warm start for stage-3 shooting from ``prev``.

    For a stage-2 solution at free area, use the **native** stage-2 parameters.
    Rescaling ``(U,S,Σ̄,P̄)`` to ``A = 4π`` is the correct continuum map, but the
    rescaled point can sit in a stiff region for the two-leg integrator; shooting
    from the native seed is faster and the optimizer adjusts to ``A = 4π``.
    """
    return np.array(
        [prev.U0, prev.U1, prev.S1, prev.sigma_bar, prev.P_bar], dtype=float,
    )

def _seed_candidates(
    v: float,
    c0: float,
    prev: Optional[MeridianSolution] = None,
    *,
    u0_seeds: Optional[List[Tuple[float, float, float]]] = None,
    fig16: Optional[List[Tuple[float, float, float]]] = None,
) -> List[np.ndarray]:
    seeds: List[np.ndarray] = []
    if prev is not None and len(prev.s) > 0:
        seeds.append(_warm_start_x5(prev))
        if not _prev_at_unit_area(prev):
            seeds.append(np.array(
                [prev.U0, prev.U1, prev.S1, prev.sigma_bar, prev.P_bar], dtype=float,
            ))

    branch_seeds = u0_seeds if u0_seeds is not None else fig16
    if branch_seeds:
        for U0_g, U1_g, S1_g in branch_seeds:
            seeds.append(np.array([U0_g, U1_g, S1_g, 0.0, 0.0], dtype=float))

    if abs(v - 1.0) < 0.05:
        seeds.append(np.array([1.0, 1.0, np.pi, 0.0, 0.0], dtype=float))

    uniq: List[np.ndarray] = []
    for s in seeds:
        if not any(np.linalg.norm(s - t) < 0.06 for t in uniq):
            uniq.append(s)
    return uniq

def _av_errors_from_profile(
    y: np.ndarray,
    v: float,
    *,
    A_target: float = A_STAR,
) -> Tuple[float, float]:
    """Relative ``A``, ``V`` errors at the south-pole end of a stitched profile."""
    if y.size == 0 or y.shape[1] == 0:
        return float("nan"), float("nan")
    a_end = float(y[5, -1])
    v_end = float(y[6, -1])
    v_tgt = V_at_area(v, A_target)
    a_err = (a_end - A_target) / A_target
    v_err = (v_end - v_tgt) / max(abs(v_tgt), 1e-12)
    return float(a_err), float(v_err)

def _shoot_box(
    x0: np.ndarray,
    *,
    u0_window: Optional[float] = None,
    u0_min: Optional[float] = None,
    u1_min: Optional[float] = None,
    u0_rel: float = 0.18,
    u1_rel: float = 0.22,
    s1_window: Tuple[float, float] = (0.82, 1.18),
    sig_rel: float = 0.35,
    p_rel: float = 0.30,
    sig_abs: float = 0.35,
    p_abs: float = 0.25,
    pin_P_sigma: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded TRF box around a 5-vector warm start."""
    x0 = np.asarray(x0, dtype=float).reshape(5)
    lb = list(TWO_LEG_BOUNDS[0])
    ub = list(TWO_LEG_BOUNDS[1])
    u0, u1, s1, sig0, p0 = map(float, x0)
    if u0_window is not None and u0_window > 0.0:
        half = float(u0_window)
        u0_half = u1_half = half
    else:
        u0_half = max(0.08, u0_rel * max(abs(u0), 0.15))
        u1_half = max(0.08, u1_rel * max(abs(u1), abs(u0), 0.15))
    lb[0] = max(lb[0], u0 - u0_half)
    ub[0] = min(ub[0], u0 + u0_half)
    lb[1] = max(lb[1], u1 - u1_half)
    ub[1] = min(ub[1], u1 + u1_half)
    if u0_min is not None:
        lb[0] = max(lb[0], float(u0_min))
    if u1_min is not None:
        lb[1] = max(lb[1], float(u1_min))
    lb[2] = max(lb[2], s1 * s1_window[0])
    ub[2] = min(ub[2], s1 * s1_window[1])
    sig_half = max(sig_abs, sig_rel * max(abs(sig0), 0.5))
    p_half = max(p_abs, p_rel * max(abs(p0), 0.25))
    if pin_P_sigma:
        sig_half = p_half = 1e-14
    lb[3] = max(lb[3], sig0 - sig_half)
    ub[3] = min(ub[3], sig0 + sig_half)
    lb[4] = max(lb[4], p0 - p_half)
    ub[4] = min(ub[4], p0 + p_half)
    for i in range(5):
        if lb[i] >= ub[i]:
            mid = 0.5 * (lb[i] + ub[i])
            eps = max(1e-6, 1e-4 * max(abs(mid), 1.0))
            lb[i], ub[i] = mid - eps, mid + eps
    x_clip = np.clip(x0, lb, ub)
    return x_clip, np.array(lb, dtype=float), np.array(ub, dtype=float)

def _shoot_two_leg(
    v: float,
    c0: float,
    x0: np.ndarray,
    *,
    branch: str,
    n: int,
    verbose: bool,
    A_target: float = A_STAR,
    A_c0: Optional[float] = None,
    u0_window: Optional[float] = None,
    u0_min: Optional[float] = None,
    u1_min: Optional[float] = None,
    s1_window: Tuple[float, float] = (0.82, 1.18),
    residual_rtol: float = 1e-9,
    residual_atol: float = 1e-11,
    integrate_rtol: float = 1e-9,
    integrate_atol: float = 1e-11,
    max_nfev: int = 120,
    polish: bool = True,
    polish_frac: float = 0.07,
    residual_timeout: float = STAGE3_RESIDUAL_TIMEOUT,
    enforce_fig16_residual: bool = False,
    enforce_fig16_profile: bool = True,
    pin_P_sigma: bool = False,
) -> MeridianSolution:
    if A_c0 is None:
        A_c0 = A_target
    x0, lb, ub = _shoot_box(
        x0, u0_window=u0_window, u0_min=u0_min, u1_min=u1_min,
        s1_window=s1_window, pin_P_sigma=pin_P_sigma,
    )

    def residual(x: np.ndarray) -> np.ndarray:
        return _two_leg_residual_safe(
            x, v, c0, residual_rtol, residual_atol, timeout=residual_timeout,
            A_target=A_target, A_c0=A_c0, enforce_fig16=enforce_fig16_residual,
        )

    opt = least_squares(
        residual, x0, method="trf", bounds=(lb, ub),
        ftol=1e-12, xtol=1e-12, max_nfev=max_nfev,
    )
    if polish and float(np.linalg.norm(opt.fun)) < 0.25:
        x_pol = np.asarray(opt.x, dtype=float)
        plb = np.maximum(lb, x_pol * (1.0 - polish_frac))
        pub = np.minimum(ub, x_pol * (1.0 + polish_frac))
        for i in range(5):
            if plb[i] >= pub[i]:
                mid = float(x_pol[i])
                eps = max(1e-6, polish_frac * max(abs(mid), 1.0))
                plb[i], pub[i] = mid - eps, mid + eps
        opt2 = least_squares(
            residual, x_pol, method="trf", bounds=(plb, pub),
            ftol=1e-13, xtol=1e-13, max_nfev=max(40, max_nfev // 2),
        )
        if float(np.linalg.norm(opt2.fun)) <= float(np.linalg.norm(opt.fun)):
            opt = opt2

    U0, U1, S1, sigma_bar, P_bar = map(float, opt.x)
    S_bar = JUNCTION_FRAC * S1
    out = _integrate_two_legs(
        U0, U1, S1, S_bar, sigma_bar, P_bar, c0, v,
        dense=True, n=n, rtol=integrate_rtol, atol=integrate_atol,
        enforce_fig16=enforce_fig16_profile, A_c0=A_c0, A_target=A_target,
    )
    if out is None and enforce_fig16_profile:
        out = _integrate_two_legs(
            U0, U1, S1, S_bar, sigma_bar, P_bar, c0, v,
            dense=True, n=n, rtol=integrate_rtol, atol=integrate_atol,
            enforce_fig16=False, A_c0=A_c0, A_target=A_target,
        )
    if out is None:
        return MeridianSolution(
            v, c0, U0, U1, sigma_bar, P_bar, np.array([]), np.zeros((7, 0)),
            branch, False, "integration failed", {"residual": float(np.linalg.norm(opt.fun))},
            S1, S_bar,
        )

    s, y, yn, ys = out
    constraints = _constraints_from_junction(yn, ys, v, S1, S_bar, A_target=A_target)
    a_err, v_err = _av_errors_from_profile(y, v, A_target=A_target)
    constraints["A_end"] = float(y[5, -1]) if y.shape[1] else float("nan")
    constraints["V_end"] = float(y[6, -1]) if y.shape[1] else float("nan")
    constraints["A_error_rel"] = a_err
    constraints["V_error_rel"] = v_err
    if y.shape[1] > 0:
        constraints["X_max"] = float(y[0].max())
    res_n = float(np.linalg.norm(opt.fun))
    constraints["residual"] = res_n
    if u0_window is not None and u0_window > 0.5:
        res_tol, match_tol, av_tol = 0.08, 0.05, 0.05
    else:
        res_tol, match_tol, av_tol = 5e-3, 0.02, 0.02
    ok = res_n < res_tol
    ok = ok and abs(constraints["psi_match"]) < match_tol
    ok = ok and abs(constraints["X_match"]) < match_tol
    ok = ok and np.isfinite(a_err) and abs(a_err) < av_tol
    ok = ok and np.isfinite(v_err) and abs(v_err) < av_tol
    # Pear / asymmetric shapes can hold A,V while ψ/X junction lags slightly in the ramp.
    if (
        not ok
        and np.isfinite(a_err)
        and np.isfinite(v_err)
        and abs(a_err) < av_tol
        and abs(v_err) < av_tol
        and res_n < 1.05
        and abs(constraints["psi_match"]) < 0.15
        and abs(constraints["X_match"]) < 0.15
    ):
        ok = True

    msg = "junction matched" if ok else opt.message
    if (
        not ok
        and y.shape[1] > 0
        and np.isfinite(a_err)
        and np.isfinite(v_err)
        and abs(a_err) < 1e-4
        and abs(v_err) < 1e-4
    ):
        ok = True
        msg = "A,V matched (ψ/X junction open)"
    sol = MeridianSolution(
        v, c0, U0, U1, sigma_bar, P_bar, s, y, branch, ok,
        msg, constraints, S1, S_bar,
    )
    if verbose:
        print(
            f"  v={v:.4f} c0={c0:.4f} U0={U0:.4f} U1={U1:.4f} S1={S1:.4f} "
            f"Σ̄={sigma_bar:.4f} P̄={P_bar:.4f} "
            f"A_err={a_err:+.2e} V_err={v_err:+.2e} "
            f"X_max={constraints.get('X_max', float('nan')):.3f} "
            f"|res|={res_n:.2e} ok={ok}  {msg}"
        )
    return sol

def _shoot_residual_norm(sol: MeridianSolution) -> float:
    c = sol.constraints
    return float(
        abs(c.get("psi_match", 1.0))
        + abs(c.get("X_match", 1.0))
        + abs(c.get("A_error_rel", 1.0))
        + abs(c.get("V_error_rel", 1.0))
    )

def _score_prolate_candidate(
    sol: MeridianSolution,
    *,
    u0_min: Optional[float] = None,
) -> float:
    """Higher is better; −inf rejects oblate / stomatocyte side."""
    if len(sol.s) < 5:
        return float("-inf")
    if u0_min is not None and float(sol.U0) < float(u0_min):
        return float("-inf")
    if u0_min is not None and float(sol.U1) < float(u0_min):
        return float("-inf")
    if sol.U0 <= 0.4:
        return float("-inf")
    res = _shoot_residual_norm(sol)
    if not sol.success and res > 0.05:
        return float("-inf")
    zs = meridian_z_span(sol)
    if not np.isfinite(zs):
        return float("-inf")
    u_excess = float(sol.U0)
    if u0_min is not None:
        u_excess = float(sol.U0 - u0_min)
    return u_excess + 0.05 * zs - 2.0 * res

def solve_seifert(
    v: float,
    c0: float = 0.0,
    *,
    prev: Optional[MeridianSolution] = None,
    branch: str = "seifert",
    n: int = 240,
    verbose: bool = False,
    use_u0_scan: bool = True,
    pick: str = "first",
    u0_seeds: Optional[List[Tuple[float, float, float]]] = None,
    use_fig16: Optional[bool] = None,
    fig16_guesses: Optional[List[Tuple[float, float, float]]] = None,
    lock_branch: bool = False,
    u0_window: Optional[float] = None,
    A_target: float = A_STAR,
    A_c0: Optional[float] = None,
    polish: bool = True,
    polish_frac: float = 0.07,
    max_nfev: int = 120,
    residual_rtol: float = 1e-9,
    residual_atol: float = 1e-11,
    integrate_rtol: float = 1e-9,
    integrate_atol: float = 1e-11,
) -> MeridianSolution:
    """Appendix B: ``U(0)`` scan + two-leg shoot for target area and ``V = V(v, A)``.

    pick="prolate" tries all seeds and keeps the highest-U₀ prolate candidate
    (needed when continuing down in v from the sphere — warm-start alone lands on oblates).

    lock_branch=True keeps ``(U₀,U₁,S₁)`` near ``prev`` (Božič–Svetina tracks).
    """
    shoot_kw = dict(
        n=n, verbose=verbose, polish=polish, polish_frac=polish_frac, max_nfev=max_nfev,
        residual_rtol=residual_rtol, residual_atol=residual_atol,
        integrate_rtol=integrate_rtol, integrate_atol=integrate_atol,
        A_target=A_target, A_c0=A_c0,
    )
    if use_fig16 is not None:
        use_u0_scan = use_fig16
    if fig16_guesses is not None:
        u0_seeds = fig16_guesses
    if u0_seeds is None and use_u0_scan and (prev is None or pick == "prolate"):
        u0_seeds = scan_branch_seeds(c0)
        if verbose and u0_seeds:
            print(f"  U(0) scan seeds: {[(round(a,2),round(b,2),round(c,2)) for a,b,c in u0_seeds[:4]]}")

    best: Optional[MeridianSolution] = None
    best_ok: Optional[MeridianSolution] = None
    best_ok_score = np.inf
    best_prolate: Optional[MeridianSolution] = None
    best_prolate_score = float("-inf")

    # Branch-locked continuation: only the warm start, with U₀ window.
    if lock_branch and prev is not None:
        x0 = _warm_start_x5(prev)
        return _shoot_two_leg(
            v, c0, x0, branch=branch, u0_window=u0_window, **shoot_kw,
        )

    for i, x0 in enumerate(_seed_candidates(v, c0, prev, u0_seeds=u0_seeds)):
        max_seeds = 4 if pick == "prolate" else 12
        if i >= max_seeds:
            break
        sol = _shoot_two_leg(v, c0, x0, branch=branch, **shoot_kw)
        if pick == "prolate":
            ps = _score_prolate_candidate(sol)
            if ps > best_prolate_score:
                best_prolate_score = ps
                best_prolate = sol
            continue
        if sol.success:
            score = abs(sol.constraints.get("X_match", 1.0)) + abs(sol.constraints.get("psi_match", 1.0))
            if score < best_ok_score:
                best_ok_score = score
                best_ok = sol
            if best_ok_score < 5e-3:
                break
        elif best is None or abs(sol.constraints.get("psi_match", 1.0)) < abs(
            best.constraints.get("psi_match", 1.0)
        ):
            best = sol

    if pick == "prolate" and best_prolate is not None:
        return best_prolate
    if best_ok is not None:
        return best_ok
    return best if best is not None else MeridianSolution(
        v, c0, 0.0, 0.0, 0.0, 0.0, np.array([]), np.zeros((7, 0)),
        branch, False, "no seed converged", {}, 0.0, 0.0,
    )

def _stage3_cv_line_track(
    v0: float,
    c0_0: float,
    v1: float,
    c0_1: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Straight-line samples in ``(c₀, v̄)`` from start to target (excluding start)."""
    n = int(n_steps)
    if n < 1:
        return np.array([]), np.array([])
    t = np.linspace(0.0, 1.0, n + 1, dtype=float)[1:]
    return c0_0 + t * (c0_1 - c0_0), v0 + t * (v1 - v0)

def _stage3_pear_step_ok(
    sol: MeridianSolution,
    *,
    min_v: float = 0.65,
    warm: Optional[MeridianSolution] = None,
    strict_pear: bool = False,
    min_asym: float = 0.08,
    junction_match_tol: float = 0.05,
    s1_max_rel_jump: float = 0.035,
) -> bool:
    """True when a continuation step stays on the pear / dumbbell branch."""
    if len(sol.s) < 5:
        return False
    if float(sol.v) < min_v:
        return False
    if _score_prolate_candidate(sol) == float("-inf"):
        return False
    if not strict_pear:
        return True
    msg = (sol.message or "").lower()
    if "junction open" in msg:
        return False
    c = sol.constraints
    psi = abs(float(c.get("psi_match", 1.0)))
    x_m = abs(float(c.get("X_match", 1.0)))
    if psi > junction_match_tol or x_m > junction_match_tol:
        return False
    if _bs_asymmetry(sol) < min_asym:
        return False
    if warm is not None and warm.S1 > 0.0:
        rel_jump = abs(float(sol.S1) - float(warm.S1)) / float(warm.S1)
        if rel_jump > s1_max_rel_jump:
            return False
    return True


def _stage3_stomatocyte_step_ok(
    sol: MeridianSolution,
    *,
    min_v: float = 0.50,
    warm: Optional[MeridianSolution] = None,
    junction_match_tol: float = 0.08,
    s1_max_rel_jump: float = 0.08,
) -> bool:
    """True when a continuation step stays on a stomatocyte-like branch.

    Unlike the pear checker, this does **not** use ``_score_prolate_candidate``
    (which hard-rejects ``U₀ ≤ 0.4`` and would discard every stomatocyte).
    """
    if len(sol.s) < 5:
        return False
    if float(sol.v) < float(min_v):
        return False
    # North pole should be inverted (cup).
    if float(sol.U0) > 0.05:
        return False
    c = sol.constraints
    res = float(c.get("residual", float("inf")))
    psi = abs(float(c.get("psi_match", 1.0)))
    x_m = abs(float(c.get("X_match", 1.0)))
    if not np.isfinite(res):
        res = float("inf")
    if res > 0.20 and not sol.success:
        return False
    if psi > float(junction_match_tol) or x_m > float(junction_match_tol):
        return False
    if warm is not None and float(warm.S1) > 0.0:
        rel_jump = abs(float(sol.S1) - float(warm.S1)) / float(warm.S1)
        if rel_jump > float(s1_max_rel_jump):
            return False
    return True


def pear_branch_at_E_ok(sol: MeridianSolution, *, rtol: float = 5e-3) -> bool:
    """True when ``sol`` is the asymmetric pear/dumbbell **E**, not prolate **E**."""
    if not (
        abs(float(sol.v) - BS_V_TWO_SPHERE) < rtol
        and abs(float(sol.c0) - BS_C0_END) < rtol * max(BS_C0_END, 1.0)
    ):
        return False
    return _stage3_pear_step_ok(
        sol, strict_pear=True, warm=None, min_asym=0.08,
        junction_match_tol=0.05, s1_max_rel_jump=1.0,
    )


def _stage3_cv_shoot_step(
    warm: MeridianSolution,
    c_try: float,
    v_try: float,
    ramp_kw: Dict[str, Any],
    *,
    strict_pear: bool,
    pear_min_v: float,
    pear_min_asym: float,
    junction_match_tol: float,
    s1_max_rel_jump: float,
    cv_bisect_max: int,
    verbose: bool,
    step_label: str,
    family: str = "pear",
) -> Tuple[MeridianSolution, bool]:
    """Shoot one ``(c₀,v̄)`` increment; bisect when branch checks fail."""
    c_w, v_w = float(warm.c0), float(warm.v)
    c_pt, v_pt = float(c_try), float(v_try)
    sol_try: Optional[MeridianSolution] = None
    ok = False
    for k in range(int(cv_bisect_max) + 1):
        sol_try = solve_seifert(float(v_pt), float(c_pt), prev=warm, **ramp_kw)
        if family == "stomatocyte":
            ok = _stage3_stomatocyte_step_ok(
                sol_try, min_v=pear_min_v, warm=warm,
                junction_match_tol=junction_match_tol,
                s1_max_rel_jump=s1_max_rel_jump,
            )
        else:
            ok = _stage3_pear_step_ok(
                sol_try, min_v=pear_min_v, warm=warm, strict_pear=strict_pear,
                min_asym=pear_min_asym, junction_match_tol=junction_match_tol,
                s1_max_rel_jump=s1_max_rel_jump,
            )
        # Always bisect on failure (do not skip bisect in non-strict pear mode).
        if ok or k == cv_bisect_max:
            break
        c_pt = 0.5 * (c_w + c_pt)
        v_pt = 0.5 * (v_w + v_pt)
        if verbose:
            print(
                f"  {step_label}: branch reject @ c₀={float(c_try):.4f} "
                f"v̄={float(v_try):.4f}  bisect [{k + 1}/{cv_bisect_max}] "
                f"→ c₀={c_pt:.4f} v̄={v_pt:.4f}",
                flush=True,
            )
    assert sol_try is not None
    return sol_try, ok

def solve_stage3_from_appendix_b(
    warm_start: MeridianSolution,
    v: float,
    c0: float,
    *,
    A_target: Optional[float] = None,
    branch: str = "stage3",
    n: int = 240,
    verbose: bool = False,
    u0_window: Optional[float] = None,
    refine: bool = True,
    c0_step: float = 0.06,
    min_cv_steps: int = 25,
    v_step: float = 0.004,
    pear_min_v: float = 0.65,
    strict_pear_branch: bool = False,
    pear_min_asym: float = 0.08,
    junction_match_tol: float = 0.05,
    s1_max_rel_jump: float = 0.035,
    cv_bisect_max: int = 5,
    polish: bool = True,
    family: str = "pear",
    **shoot_kw,
) -> MeridianSolution:
    """Match growth-track ``(A, v̄, c₀)`` at D from a stage-2.5 warm start.

    ``family="pear"`` (default) or ``"stomatocyte"`` selects the branch
    acceptance test along the ``(c₀,v̄)`` line.

    (1) Junction refine via stage-2 shooting (``U₀,U₁,S₁``; ``v̄,c₀`` may shift),
    (2) straight-line continuation in ``(c₀, v̄)`` to D with full two-leg shoots
        at ``A = A_D`` (avoids a c₀-only jump across a branch discontinuity),
    (3) optional final polish at the target.
    """
    v_tgt, c0_tgt = float(v), float(c0)
    family = str(family).lower().strip() or "pear"
    if "min_v_steps" in shoot_kw:
        min_cv_steps = int(shoot_kw.pop("min_v_steps"))
    warm = warm_start
    if A_target is None:
        A_target = float(warm.constraints.get("A_target", float("nan")))
    if not np.isfinite(A_target) or A_target <= 0.0:
        A_target = float(warm.constraints.get("A_phys", stage2_area(warm)))
    if not np.isfinite(A_target) or A_target <= 0.0:
        R0 = area_radius(A_STAR) if abs(c0_tgt) < 1e-14 else abs(c0_tgt)
        A_target = float(A_STAR * R0 * R0)
    A_target = float(A_target)

    sol: Optional[MeridianSolution] = None
    base_kw = dict(
        branch=branch, n=n, verbose=verbose, polish=polish,
        use_u0_scan=False, lock_branch=True, u0_window=u0_window,
        A_target=A_target, A_c0=A_target, **shoot_kw,
    )
    ramp_kw = {
        **base_kw,
        "polish": False,
        "max_nfev": min(50, int(shoot_kw.get("max_nfev", 120))),
        "residual_rtol": float(shoot_kw.get("residual_rtol", 1e-9) * 10),
        "residual_atol": float(shoot_kw.get("residual_atol", 1e-11) * 10),
        "integrate_rtol": float(shoot_kw.get("integrate_rtol", 1e-9) * 10),
        "integrate_atol": float(shoot_kw.get("integrate_atol", 1e-11) * 10),
    }
    final_kw = base_kw

    if verbose:
        print(
            f"  stage3: A_D={A_target:.4f}  warm v̄={warm.v:.4f} c₀={warm.c0:.4f}  "
            f"→ v̄={v_tgt:.4f} c₀={c0_tgt:.4f}",
            flush=True,
        )

    v_cur = float(warm.v)
    c0_cur = float(warm.c0)
    path: List[Dict[str, Any]] = []

    def _path_note(
        phase: str, v_pt: float, c0_pt: float, *, ok: bool = True,
        v_req: Optional[float] = None, c0_req: Optional[float] = None,
    ) -> None:
        path.append({
            "phase": phase,
            "v": float(v_pt),
            "c0": float(c0_pt),
            "v_req": float(v_req if v_req is not None else v_pt),
            "c0_req": float(c0_req if c0_req is not None else c0_pt),
            "ok": bool(ok),
        })

    if refine and len(warm.s) > 0:
        c0_ref = float(warm.c0)
        if verbose:
            print(
                f"  stage3: junction refine (stage-2 shoot)  "
                f"v̄≈{v_cur:.4f} c₀={c0_ref:.4f}  (v̄ may update)",
                flush=True,
            )
        sol = _shoot_pear_junction_from_warm(
            warm, c0_ref, branch=f"{branch}_refine", verbose=verbose, n=n,
            A_c0=A_target,
        )
        if len(sol.s) > 0:
            warm = sol
            v_cur, c0_cur = float(warm.v), float(warm.c0)
            _path_note("refine", v_cur, c0_cur, ok=bool(warm.success))
            if verbose:
                print(
                    f"  stage3: after refine  v̄={v_cur:.4f} c₀={c0_cur:.4f}  "
                    f"A={stage2_area(warm):.3f}  ok={warm.success}",
                    flush=True,
                )

    if abs(c0_tgt - c0_cur) > 1e-9 or abs(v_tgt - v_cur) > 1e-9:
        n_c0 = int(np.ceil(abs(c0_tgt - c0_cur) / c0_step)) if c0_step > 0 else 1
        n_v = int(np.ceil(abs(v_tgt - v_cur) / v_step)) if v_step > 0 else 1
        n_line = max(min_cv_steps, n_c0, n_v)
        c_track, v_track = _stage3_cv_line_track(
            v_cur, c0_cur, v_tgt, c0_tgt, n_line,
        )
        if verbose:
            dc0 = abs(c0_tgt - c0_cur) / max(len(c_track), 1)
            dv = abs(v_tgt - v_cur) / max(len(c_track), 1)
            print(
                f"  stage3: (c₀,v̄) line {c0_cur:.4f},{v_cur:.4f} → "
                f"{c0_tgt:.4f},{v_tgt:.4f}  ({len(c_track)} shoots, "
                f"Δc₀≈{dc0:.4f} Δv̄≈{dv:.4f})",
                flush=True,
            )
        t_line = time.perf_counter()
        for i, (c_try, v_try) in enumerate(zip(c_track, v_track)):
            sol_try, ok = _stage3_cv_shoot_step(
                warm, float(c_try), float(v_try), ramp_kw,
                strict_pear=strict_pear_branch,
                pear_min_v=pear_min_v,
                pear_min_asym=pear_min_asym,
                junction_match_tol=junction_match_tol,
                s1_max_rel_jump=s1_max_rel_jump,
                cv_bisect_max=cv_bisect_max,
                verbose=verbose,
                step_label=f"cv [{i + 1}/{len(c_track)}]",
                family=family,
            )
            _path_note(
                "cv_line", float(sol_try.v), float(sol_try.c0), ok=ok,
                v_req=float(v_try), c0_req=float(c_try),
            )
            if verbose and (i == 0 or i == len(c_track) - 1 or (i + 1) % 5 == 0):
                print(
                    f"  cv [{i + 1}/{len(c_track)}] c₀={float(c_try):.4f} v̄={float(v_try):.4f}  "
                    f"→ v̄={sol_try.v:.4f} ok={ok}",
                    flush=True,
                )
            if ok:
                warm = sol_try
                sol = sol_try
                v_cur, c0_cur = float(warm.v), float(warm.c0)
        if verbose:
            print(
                f"  stage3: line done ({time.perf_counter() - t_line:.1f}s)  "
                f"v̄={v_cur:.4f} c₀={c0_cur:.4f}",
                flush=True,
            )

    if sol is None or abs(sol.v - v_tgt) > 1e-9 or abs(sol.c0 - c0_tgt) > 1e-9:
        sol = solve_seifert(v_tgt, c0_tgt, prev=warm, **final_kw)
    elif sol is not None and not sol.success and len(sol.s) > 0:
        if verbose:
            print("  stage3: final polish on junction residual", flush=True)
        polish_kw = {
            **final_kw,
            "max_nfev": max(160, int(shoot_kw.get("max_nfev", 120))),
            "polish": True,
            "polish_frac": 0.04,
            "u0_window": 0.25 if u0_window is None else min(0.35, float(u0_window)),
        }
        sol_pol = solve_seifert(v_tgt, c0_tgt, prev=sol, **polish_kw)
        if sol_pol.success or float(sol_pol.constraints.get("residual", 1e9)) < float(
            sol.constraints.get("residual", 1e9)
        ):
            sol = sol_pol
    if sol is not None:
        sol.constraints["A_target"] = A_target
        sol.constraints["A_phys"] = A_target
        _path_note("target", v_tgt, c0_tgt)
        sol.constraints["stage3_path"] = path
        sol.constraints["stage3_target_v"] = v_tgt
        sol.constraints["stage3_target_c0"] = c0_tgt
        sol.constraints["stage3_family"] = family
        # Stomatocyte: do not accept a high-residual final as success.
        if family == "stomatocyte":
            res = float(sol.constraints.get("residual", float("inf")))
            psi = abs(float(sol.constraints.get("psi_match", 1.0)))
            xm = abs(float(sol.constraints.get("X_match", 1.0)))
            if (not np.isfinite(res) or res > 0.08 or psi > 0.08 or xm > 0.08
                    or float(sol.U0) > 0.05):
                sol = replace(
                    sol,
                    success=False,
                    message=(
                        sol.message + "; stomatocyte stage-3 residual/junction gate"
                        if sol.message else
                        "stomatocyte stage-3 residual/junction gate"
                    ),
                )
    return sol

def _junction_frac_from_warm(warm: MeridianSolution) -> Optional[float]:
    jf = warm.constraints.get("junction_frac")
    if jf is not None and np.isfinite(jf):
        return float(jf)
    if warm.S1 > 0.0 and np.isfinite(warm.S_bar):
        return float(warm.S_bar / warm.S1)
    return None

def _shoot_pear_junction_from_warm(
    warm: MeridianSolution,
    c0: float,
    *,
    branch: str,
    verbose: bool = False,
    junction_frac: Optional[float] = None,
    n: int = 200,
    A_c0: float = A_STAR,
    max_nfev: int = 80,
) -> MeridianSolution:
    """Close the pear junction with stage-2 shooting (``U₀,U₁,S₁`` only; ``v̄`` free)."""
    sol = shoot_appendix_b_stage2(
        warm.U0, warm.U1, warm.S1,
        sigma_bar=warm.sigma_bar, P_bar=warm.P_bar, c0=float(c0),
        branch=branch, verbose=verbose,
        junction_frac=junction_frac if junction_frac is not None else _junction_frac_from_warm(warm),
        n=n, A_c0=A_c0, max_nfev=max_nfev,
    )
    if len(sol.s) == 0:
        return warm
    return replace(
        sol,
        c0=float(c0),
        constraints={**warm.constraints, **sol.constraints},
    )

def meridian_z_span(sol: MeridianSolution) -> float:
    """Pole-to-pole meridian height (increasing on prolate-1 as v decreases)."""
    if len(sol.s) < 5:
        return float("nan")
    Z = -sol.Z
    return float(Z.max() - Z.min())

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


def bending_energy(
    sol: MeridianSolution,
    *,
    kappa: float = 1.0,
    C0: Optional[float] = None,
) -> float:
    """Spontaneous-curvature Helfrich energy ``(κ/2) ∫ (C₁ + C₂ − C₀)² dA``.

    ``C₁ = U = ψ̇``, ``C₂ = sinψ / X``, ``dA = 2π X dS``.  Dimensional ``C₀``
    defaults to ``constraints['C0_dimensional']``, else ``C0_dimensional(c₀, A)``.
    """
    s = np.asarray(sol.s, dtype=float)
    y = np.asarray(sol.y, dtype=float)
    if s.size < 2 or y.size == 0:
        return float("nan")
    X = y[0]
    psi = y[2]
    U = y[3]
    if C0 is None:
        stored = sol.constraints.get("C0_dimensional")
        if stored is not None:
            C0 = float(stored)
        else:
            A = float(sol.constraints.get("A_phys", A_STAR))
            C0 = float(C0_dimensional(sol.c0, A))
    C0 = float(C0)
    Xr = np.sqrt(X * X + POLE_R * POLE_R)
    C2 = np.sin(psi) / Xr
    dens = 0.5 * float(kappa) * (U + C2 - C0) ** 2 * (2.0 * np.pi * np.maximum(X, 0.0))
    return float(np.trapz(dens, s))


def _cache_path(cache_dir: Path, v: float, c0: float) -> Path:
    return cache_dir / f"v{v:.6f}_c0{c0:.6f}.json"

def save_solution(cache_dir: Path, sol: MeridianSolution) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, sol.v, sol.c0)
    return save_solution_at(path, sol)

def _solution_payload(sol: MeridianSolution) -> Dict[str, Any]:
    return {
        "model": "seifert_spontaneous_curvature",
        "solver": "appendix_b_two_leg",
        "v": sol.v,
        "c0": sol.c0,
        "branch": sol.branch,
        "success": sol.success,
        "message": sol.message,
        "constraints": sol.constraints,
        "params": {
            "U0": sol.U0, "U1": sol.U1, "S1": sol.S1, "S_bar": sol.S_bar,
            "sigma_bar": sol.sigma_bar, "P_bar": sol.P_bar,
        },
        "meridian": {
            "s": sol.s.tolist(),
            "X": sol.X.tolist(),
            "Z": sol.Z.tolist(),
            "psi": sol.y[2].tolist(),
            "U": sol.U.tolist(),
            "gamma": sol.y[4].tolist(),
            "A": sol.y[5].tolist(),
            "V": sol.y[6].tolist(),
        },
    }

def save_solution_at(path: Path, sol: MeridianSolution) -> Path:
    """Write one ``MeridianSolution`` JSON to an explicit path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_solution_payload(sol), indent=2))
    return path

def load_solution(path: Path) -> MeridianSolution:
    data = json.loads(Path(path).read_text())
    m = data["meridian"]
    s = np.array(m["s"])
    # Current keys (Seifert notation)
    if "X" in m:
        y = np.array([m["X"], m["Z"], m["psi"], m["U"], m["gamma"], m["A"], m["V"]])
        p = data["params"]
        U0 = float(p["U0"])
        U1 = float(p.get("U1", 0.0))
        sigma_bar = float(p["sigma_bar"])
        P_bar = float(p["P_bar"])
        S1 = float(p.get("S1", 0.0))
        S_bar = float(p.get("S_bar", 0.0))
    else:
        r = np.array(m["r"])
        z = np.array(m["z"])
        psi = np.array(m["psi"])
        H = np.array(m["H"])
        q = np.array(m["q"])
        y = np.vstack([
            r,
            -z,
            psi,
            2.0 * H - 1.0,
            q,
            np.array(m["A"]),
            np.array(m["V"]),
        ])
        p = data["params"]
        H0 = float(p.get("H0", p.get("U0", 1.0)))
        U0 = 2.0 * H0 - 1.0
        sigma_bar = float(p.get("sigma_bar", p.get("lambda", 0.0)))
        P_bar = float(p.get("P_bar", p.get("p_tilde", 0.0)))
        S1 = float(p.get("S1", float(s[-1]) if len(s) else 0.0))
    return MeridianSolution(
        v=float(data["v"]),
        c0=float(data["c0"]),
        U0=U0,
        U1=U1,
        sigma_bar=sigma_bar,
        P_bar=P_bar,
        s=s,
        y=y,
        branch=str(data.get("branch", "symmetric")),
        success=bool(data["success"]),
        message=str(data.get("message", "")),
        constraints=dict(data.get("constraints", {})),
        S1=S1,
        S_bar=S_bar,
    )

def save_trajectory(
    cache_dir: Path,
    traj: List[MeridianSolution],
    *,
    name: str = "trajectory",
) -> Path:
    """Cache an ordered branch trajectory (one JSON manifest + per-frame solutions)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    for sol in traj:
        files.append(save_solution(cache_dir, sol).name)
    manifest = {
        "model": "seifert_spontaneous_curvature",
        "branch": traj[0].branch if traj else "",
        "c0": float(traj[0].c0) if traj else 0.0,
        "v_values": [float(s.v) for s in traj],
        "z_span": [float(meridian_z_span(s)) for s in traj],
        "files": files,
    }
    path = cache_dir / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path

def load_trajectory(manifest_path: Path) -> List[MeridianSolution]:
    """Reload a cached trajectory from its manifest."""
    data = json.loads(Path(manifest_path).read_text())
    cache_dir = manifest_path.parent
    return [load_solution(cache_dir / fname) for fname in data["files"]]

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

def save_solution_list(
    cache_dir: Path,
    name: str,
    sols: Sequence[MeridianSolution],
) -> Path:
    """Cache an ordered list of solutions under ``{name}.json`` manifest."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    for i, sol in enumerate(sols):
        fname = f"{name}_{i:03d}.json"
        save_solution_at(cache_dir / fname, sol)
        files.append(fname)
    manifest = {"kind": "solution_list", "name": name, "files": files}
    path = cache_dir / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path

def load_solution_list(manifest_path: Path) -> List[MeridianSolution]:
    data = json.loads(Path(manifest_path).read_text())
    cache_dir = manifest_path.parent
    return [load_solution(cache_dir / fname) for fname in data["files"]]

