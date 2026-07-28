"""Božič et al. (Phys. Rev. E 73, 041915, 2006) mobile-inclusion shooting.

Axisymmetric vesicle shapes with laterally mobile inclusions.  The phospholipid
continuum uses spontaneous-curvature bending plus optional nonlocal ADE
(``k_r/k_c``, ``λ_Δa = 2(k_r/k_c)(Δa−Δa₀)`` from paper eqs. 10 and 13).

Relative free energy (paper eq. 10) is ``F / (8π k_c)`` with lengths in units of
``R_s = sqrt(A / 4π)`` so the reduced area is ``a = 1``.

When ``p → 0`` the inclusion sector drops and the EL system reduces to Seifert
(1991) meridian equations under the multiplier map

    λ = γ / 4 ,   λ_v = −P̄ / 6 ,   λ_a = −Σ̄ / 2 .

Euler–Lagrange ODEs (paper 21–22) with ``ν`` from (19) each RHS call;
``Δȧ = (sinψ + χ)/4`` as in the paper Lagrangian (15).
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from seifert_functions import (
    A_STAR,
    JUNCTION_FRAC,
    POLE_R,
    S_POLE,
    MeridianSolution,
    bending_energy,
    gamma_from_H0,
    load_solution,
    meridian_area,
    rescale_shoot_params_by_length,
)

# --- Parameters ----------------------------------------------------------------

INCL_P_MIN = 1e-12  # below this, treat as p = 0 (Seifert reduction)
INCL_MAX_STEP = 0.02
INCL_RTOL = 1e-7
INCL_ATOL = 1e-9
INCL_L_MAX = 40.0
# Paper Fig. 1 (PRE 73, 041915): κ=10^{-4}, h_m=30, shapes at p=1, 395, 790.5.
# p=1 is essentially the bare shape; interesting coupling is O(10²)–O(10³).
INCL_HM_DEFAULT = 30.0
INCL_KAPPA_DEFAULT = 1e-4
INCL_P_DEFAULT = 790.5  # Fig. 1 shapes C/D (near transition)
INCL_KR_OVER_KC_DEFAULT = 3.0  # Fig. 1 ADE ratio k_r / k_c
# Paper Fig. 1: Δa0 = 0.5022 at v=0.6.
INCL_DA0_DEFAULT = 0.5022
# Fig. 1b sketch: Δa ≈ 0.5022 → ~0.60 by p ≈ 200 (used for λ_Δa warms/bounds).
INCL_FIG1_DA_AT_P200 = 0.60
INCL_FIG1_P_REF = 200.0

DEFAULT_SEIFERT_SEED = Path(__file__).resolve().parent / "seed_shapes" / "pear_seed.json"

# Wall-clock cap per warm shoot in p-continuation (avoids silent hangs in LS/IVP).
INCL_SHOOT_TIMEOUT_S = 25.0


class _ShootTimeout(TimeoutError):
    """Raised when a single inclusion shoot exceeds ``INCL_SHOOT_TIMEOUT_S``."""


def _run_with_timeout(fn, timeout_s: float):
    """Run ``fn()`` with a SIGALRM wall-clock limit (main thread, Unix)."""
    if timeout_s is None or timeout_s <= 0:
        return fn()

    def _handler(_signum, _frame):
        raise _ShootTimeout(f"shoot exceeded {timeout_s:.1f}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def fig1_delta_a_estimate(
    p: float, *,
    da0: float = INCL_DA0_DEFAULT,
    da_at_p_ref: float = INCL_FIG1_DA_AT_P200,
    p_ref: float = INCL_FIG1_P_REF,
) -> float:
    """Linear sketch of Fig. 1b: ``Δa(0)=Δa₀``, ``Δa(p_ref)≈da_at_p_ref``."""
    p = max(0.0, float(p))
    frac = min(p, float(p_ref)) / float(p_ref)
    return float(da0) + (float(da_at_p_ref) - float(da0)) * frac


def fig1_lambda_da_estimate(
    p: float, *,
    da0: float = INCL_DA0_DEFAULT,
    kr_over_kc: float = INCL_KR_OVER_KC_DEFAULT,
    da_at_p_ref: float = INCL_FIG1_DA_AT_P200,
    p_ref: float = INCL_FIG1_P_REF,
) -> float:
    """``λ_Δa = 2(k_r/k_c)(Δa_est−Δa₀)`` along the Fig. 1b sketch."""
    return ade_lambda_from_da(
        fig1_delta_a_estimate(p, da0=da0, da_at_p_ref=da_at_p_ref, p_ref=p_ref),
        da0, kr_over_kc,
    )


# --- Solution container --------------------------------------------------------

@dataclass
class InclusionSolution:
    """Closed axisymmetric shape with optional inclusion density profile."""

    v: float
    c0: float
    hm: float
    kappa: float
    p: float
    U0: float
    U1: float
    la: float
    lv: float
    lnu: float
    s: np.ndarray
    y: np.ndarray  # rows: ρ, z, ψ, χ, λ, a, v, n[, Δa]
    nu: np.ndarray
    branch: str = "inclusions"
    success: bool = False
    message: str = ""
    constraints: Dict[str, float] = field(default_factory=dict)
    lda: float = 0.0  # λ_Δa
    da0: float = 0.0
    kr_over_kc: float = 0.0

    @property
    def rho(self) -> np.ndarray:
        return self.y[0]

    @property
    def X(self) -> np.ndarray:
        return self.y[0]

    @property
    def Z(self) -> np.ndarray:
        return self.y[1]

    @property
    def psi(self) -> np.ndarray:
        return self.y[2]

    @property
    def chi(self) -> np.ndarray:
        return self.y[3]

    @property
    def U(self) -> np.ndarray:
        r = np.maximum(np.abs(self.y[0]), POLE_R)
        return self.y[3] / r

    @property
    def lam(self) -> np.ndarray:
        return self.y[4]

    @property
    def S1(self) -> float:
        return float(self.s[-1]) if len(self.s) else 0.0

    @property
    def da(self) -> float:
        """Total relative area difference Δa (last sample), or constraints fallback."""
        if self.y.shape[0] >= 9 and self.y.shape[1] > 0:
            return float(self.y[8, -1])
        return float(self.constraints.get("da", 0.0))

    def copy(self) -> "InclusionSolution":
        return replace(self)


def inclusion_to_dict(sol: InclusionSolution) -> Dict[str, Any]:
    """JSON-serializable payload for one inclusion frame."""
    return {
        "model": "bozic_2006_inclusions",
        "v": float(sol.v),
        "c0": float(sol.c0),
        "hm": float(sol.hm),
        "kappa": float(sol.kappa),
        "p": float(sol.p),
        "kr_over_kc": float(sol.kr_over_kc),
        "da0": float(sol.da0),
        "branch": sol.branch,
        "success": bool(sol.success),
        "message": sol.message,
        "constraints": {k: float(v) for k, v in sol.constraints.items()},
        "params": {
            "U0": float(sol.U0),
            "U1": float(sol.U1),
            "S1": float(sol.constraints.get("S1", sol.S1)),
            "la": float(sol.la),
            "lv": float(sol.lv),
            "lnu": float(sol.lnu),
            "lda": float(sol.lda),
        },
        "meridian": {
            "s": np.asarray(sol.s, dtype=float).tolist(),
            "y": np.asarray(sol.y, dtype=float).tolist(),
            "nu": np.asarray(sol.nu, dtype=float).tolist(),
        },
    }


def inclusion_from_dict(data: Dict[str, Any]) -> InclusionSolution:
    """Rebuild ``InclusionSolution`` from ``inclusion_to_dict`` payload."""
    m = data["meridian"]
    p = data["params"]
    y = np.asarray(m["y"], dtype=float)
    s = np.asarray(m["s"], dtype=float)
    nu = np.asarray(m.get("nu", np.ones(len(s))), dtype=float)
    return InclusionSolution(
        v=float(data["v"]),
        c0=float(data["c0"]),
        hm=float(data.get("hm", 0.0)),
        kappa=float(data.get("kappa", 0.0)),
        p=float(data.get("p", 0.0)),
        U0=float(p["U0"]),
        U1=float(p["U1"]),
        la=float(p["la"]),
        lv=float(p["lv"]),
        lnu=float(p.get("lnu", 0.0)),
        s=s,
        y=y,
        nu=nu,
        branch=str(data.get("branch", "inclusions")),
        success=bool(data.get("success", True)),
        message=str(data.get("message", "")),
        constraints={k: float(v) for k, v in dict(data.get("constraints", {})).items()},
        lda=float(p.get("lda", data.get("lda", 0.0))),
        da0=float(data.get("da0", 0.0)),
        kr_over_kc=float(data.get("kr_over_kc", 0.0)),
    )


def save_inclusion_trajectory(
    path: Path | str,
    traj: Sequence[InclusionSolution],
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a full ``p``-continuation trajectory (all meridians) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "bozic_2006_inclusions_trajectory",
        "n_frames": len(traj),
        "p": [float(s.p) for s in traj],
        "meta": dict(meta or {}),
        "frames": [inclusion_to_dict(s) for s in traj],
    }
    path.write_text(json.dumps(payload))
    return path


def load_inclusion_trajectory(path: Path | str) -> List[InclusionSolution]:
    """Load trajectory written by ``save_inclusion_trajectory``."""
    data = json.loads(Path(path).read_text())
    return [inclusion_from_dict(fr) for fr in data["frames"]]


# --- Multiplier map (Božič ↔ Seifert) -----------------------------------------

def seifert_to_bozic_multipliers(sigma_bar: float, P_bar: float) -> Tuple[float, float]:
    """``(λ_a, λ_v)`` from Seifert ``(Σ̄, P̄)``."""
    return -0.5 * float(sigma_bar), -float(P_bar) / 6.0


def bozic_to_seifert_multipliers(la: float, lv: float) -> Tuple[float, float]:
    """``(Σ̄, P̄)`` from Božič ``(λ_a, λ_v)``."""
    return -2.0 * float(la), -6.0 * float(lv)


def seifert_gamma_to_lam(gamma: float) -> float:
    return 0.25 * float(gamma)


def bozic_lam_to_gamma(lam: float) -> float:
    return 4.0 * float(lam)


# --- Algebraic inclusion density (eq. 19) -------------------------------------

def ln_nu_eq19(rho: float, psi: float, chi: float, *,
    kappa: float, hm: float, p: float, lnu: float,
) -> float:
    """``ln ν`` from paper eq. (19) (isotropic inclusions, ``ξ = ξ*``)."""
    if p <= INCL_P_MIN:
        return 0.0  # ν = 1 (unused)
    rr = rho * rho + POLE_R * POLE_R
    s = np.sin(psi)
    t1 = (chi - s) ** 2
    t2 = (chi - 2.0 * hm * rho + s) ** 2
    return float(lnu / p - 1.0 - 0.125 * kappa * (t1 + t2) / rr)


def nu_from_state(rho: float, psi: float, chi: float, *,
    kappa: float, hm: float, p: float, lnu: float,
) -> float:
    if p <= INCL_P_MIN:
        return 1.0
    lnnu = ln_nu_eq19(rho, psi, chi, kappa=kappa, hm=hm, p=p, lnu=lnu)
    # clamp for ODE robustness away from critical shapes
    lnnu = float(np.clip(lnnu, -40.0, 40.0))
    return float(np.exp(lnnu))


# --- Meridian ODE (eqs. 21–22; ν from 19) -------------------------------------

def _chi_dot(rho: float, psi: float, chi: float, lam: float, nu: float, *,
    kappa: float, p: float, hm: float, lv: float,
) -> float:
    """``χ̇`` from EL (paper eq. 21). Independent of ``λ_Δa``."""
    c = np.cos(psi)
    s = np.sin(psi)
    rr = rho * rho + POLE_R * POLE_R
    rho = np.sign(rho) * np.sqrt(rr) if abs(rho) < POLE_R else rho
    kpn = kappa * nu * p
    num = (
        2.0 * chi * rho * rho * (kpn + 1.0) * c
        - (kappa ** 2) * nu * p * (chi - hm * rho)
        * (chi ** 2 - chi * s - hm * rho * s + s * s) * c
        + rho ** 3 * (8.0 * lam * s - 6.0 * lv * rho * rho * c)
        - 2.0 * rho * rho * (chi * kpn + chi - kpn * s - s) * c
    )
    den = rho * (-(kappa ** 2) * nu * p * (chi - hm * rho) ** 2 + 2.0 * rho * rho * (kpn + 1.0))
    if abs(den) < 1e-30:
        return 0.0
    return float(num / den)


def _lam_dot(rho: float, psi: float, chi: float, nu: float, *,
    kappa: float, p: float, hm: float, c0: float, la: float, lv: float, lnu: float,
    lda: float = 0.0,
) -> float:
    """``λ̇`` from EL for ``ρ`` (paper eq. 22), including ADE ``λ_Δa``."""
    s = np.sin(psi)
    rr = rho * rho + POLE_R * POLE_R
    kpn = kappa * nu * p
    ent = 0.0 if (p <= INCL_P_MIN or nu <= 0.0) else 4.0 * nu * p * rr * np.log(nu)
    lnu_term = 0.0 if p <= INCL_P_MIN else 4.0 * lnu * nu * rr
    # ADE contribution in λ̇: +χ λ_Δa / (4 ρ)  ⇒  numerator +2 χ λ_Δa ρ.
    # With constitutive λ_Δa = 2(k_r/k_c)(Δa−Δa₀) (eq. 13), this sign is the
    # restoring force for w_RE=(k_r/k_c)(Δa−Δa₀)².  The printed Lagrangian (15)
    # has −λ_Δa Δȧ; that combination is anti-restoring in our numerics (positive
    # λ_Δa raised Δa).  Prefer energy consistency over the OCR'd L sign.
    num = (
        c0 * c0 * rr
        - 2.0 * c0 * chi * rho
        + chi * chi * (1.0 + kpn)
        - 2.0 * chi * hm * kpn * rho
        + 2.0 * hm * hm * kpn * rr
        - kpn * s * s
        - 4.0 * la * rr
        - lnu_term
        - 12.0 * lv * (rho ** 3) * s
        + ent
        - s * s
        + 2.0 * chi * lda * rho
    )
    return float(num / (8.0 * rr))


def inclusion_meridian_rhs(_l: float, y: np.ndarray, *,
    c0: float, hm: float, kappa: float, p: float, la: float, lv: float, lnu: float,
    lda: float = 0.0,
) -> np.ndarray:
    """State ``y = (ρ, z, ψ, χ, λ, a, v, n, Δa)``; ``χ = ρ ψ̇``.

    Geometry: ``ρ̇ = cos ψ``, ``ż = −sin ψ``, ``ψ̇ = χ/ρ``.
    Reduced integrals: ``ȧ = ρ/2``, ``v̇ = (3/4) ρ² sin ψ``, ``ṅ = ν ρ / 2``,
    ``Δȧ = (sin ψ + χ)/4`` (paper).
    """
    rho, _z, psi, chi, lam, _a, _v, _n = y[:8]
    s = np.sin(psi)
    c = np.cos(psi)
    rho_r = float(np.sign(rho) * np.sqrt(rho * rho + POLE_R * POLE_R)) if abs(rho) < 10 * POLE_R else float(rho)

    nu = nu_from_state(rho_r, psi, chi, kappa=kappa, hm=hm, p=p, lnu=lnu)
    chid = _chi_dot(rho_r, psi, chi, lam, nu, kappa=kappa, p=p, hm=hm, lv=lv)
    lamd = _lam_dot(
        rho_r, psi, chi, nu,
        kappa=kappa, p=p, hm=hm, c0=c0, la=la, lv=lv, lnu=lnu, lda=lda,
    )
    psid = chi / rho_r
    return np.array([
        c,  # ρ̇
        -s,  # ż
        psid,  # ψ̇
        chid,  # χ̇
        lamd,  # λ̇
        0.5 * rho_r,  # ȧ
        0.75 * rho_r * rho_r * s,  # v̇
        0.5 * nu * rho_r,  # ṅ
        0.25 * (s + chi),  # Δȧ
    ], dtype=float)


# --- Pole initial conditions --------------------------------------------------

def north_pole_state_incl(U0: float, *,
    c0: float = 0.0, la: float = 0.0, lv: float = 0.0,
    kappa: float = 0.0, p: float = 0.0, hm: float = 0.0, lnu: float = 0.0,
) -> np.ndarray:
    """Regularized north pole (``ψ ≈ 0``, ``c_p = c_m = U₀``, ``λ ≈ 0``)."""
    U0 = float(U0)
    if abs(U0) < 1e-14:
        rho, psi, chi = S_POLE, 0.0, 0.0
    else:
        R = 1.0 / U0
        psi = S_POLE / R
        rho = R * np.sin(psi)
        chi = rho * U0
    # (la, lv, c0) unused at the tip: paper requires λ=0 at poles for smoothness.
    # Do NOT use Seifert gamma_from_H0(la,…): with inclusion-scale la∼−p that
    # returns a spurious nonzero pole λ and breaks the II.C reduction.
    _ = (la, lv, c0)
    a0 = 0.5 * rho * S_POLE  # rough; overwritten by integrator from S_POLE
    # Better: use sphere-cap area in reduced units if spherical
    if abs(U0) > 1e-14:
        R = 1.0 / U0
        A_cap = 2.0 * np.pi * R * R * (1.0 - np.cos(psi))
        a0 = A_cap / (4.0 * np.pi)  # = (R²/2)(1-cosψ) since Rs=1 scale if R in Rs units
    V_cap = 0.0
    if abs(U0) > 1e-14:
        R = 1.0 / U0
        V_cap = np.pi * R ** 3 * (2.0 / 3.0 - np.cos(psi) + (np.cos(psi) ** 3) / 3.0)
    v0 = 3.0 * V_cap / (4.0 * np.pi)  # Rs=1
    nu0 = nu_from_state(rho, psi, chi, kappa=kappa, hm=hm, p=p, lnu=lnu)
    n0 = 0.5 * nu0 * rho * S_POLE
    da0 = 0.25 * (np.sin(psi) + chi) * S_POLE  # tip stub
    return np.array([rho, 0.0, psi, chi, 0.0, a0, v0, n0, da0], dtype=float)


def south_pole_state_incl(U1: float, *,
    c0: float = 0.0, la: float = 0.0, lv: float = 0.0,
    kappa: float = 0.0, p: float = 0.0, hm: float = 0.0, lnu: float = 0.0,
) -> np.ndarray:
    """Regularized south pole (``ψ ≈ π``, ``c_p = c_m = U₁``)."""
    U1 = float(U1)
    if abs(U1) < 1e-14:
        rho, psi, chi = S_POLE, np.pi, 0.0
    else:
        R = 1.0 / U1
        th = np.pi - S_POLE / R
        rho = R * np.sin(th)
        psi = th
        chi = rho * U1
    _ = (la, lv, c0)  # λ=0 at poles; see north_pole_state_incl
    if abs(U1) > 1e-14:
        R = 1.0 / U1
        # cap from south: area of small tip
        alpha = S_POLE / R
        A_cap = 2.0 * np.pi * R * R * (1.0 - np.cos(alpha))
        a0 = A_cap / (4.0 * np.pi)
        V_cap = np.pi * R ** 3 * (2.0 / 3.0 - np.cos(alpha) + (np.cos(alpha) ** 3) / 3.0)
        v0 = 3.0 * V_cap / (4.0 * np.pi)
    else:
        a0, v0 = 0.0, 0.0
    nu0 = nu_from_state(rho, psi, chi, kappa=kappa, hm=hm, p=p, lnu=lnu)
    n0 = 0.5 * nu0 * rho * S_POLE
    da0 = 0.25 * (np.sin(psi) + chi) * S_POLE
    # z at south tip measured upward from south: start at 0 for the south-leg frame
    return np.array([rho, 0.0, psi, chi, 0.0, a0, v0, n0, da0], dtype=float)


# --- Two-leg integrator (Seifert-style: north forward, south reverse-arclength) -

def _integrate_inclusion_leg(y0: np.ndarray, L: float, *,
    c0: float, hm: float, kappa: float, p: float, la: float, lv: float, lnu: float,
    lda: float = 0.0,
    reverse: bool = False, n: int = 120,
    max_step: float = INCL_MAX_STEP, rtol: float = INCL_RTOL, atol: float = INCL_ATOL,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Integrate one leg over arclength ``L``.

    ``reverse=True`` uses ``dy/dσ = −f(y)`` (south pole → junction), matching
    Seifert's backward south-leg integration so the stitched meridian satisfies
    the forward ODE.
    """
    L = float(L)
    if not np.isfinite(L) or L <= 1e-8:
        return None
    y_start = np.asarray(y0, dtype=float).copy()
    if y_start.size < 9:
        y_start = np.concatenate([y_start, [0.0]])
    if reverse:
        # accumulate a,v,n,Δa from the tip; signs flipped after integrate
        y_start[5] = 0.0
        y_start[6] = 0.0
        y_start[7] = 0.0
        y_start[8] = 0.0

    def rhs(_s: float, y: np.ndarray) -> np.ndarray:
        r = inclusion_meridian_rhs(
            _s, y, c0=c0, hm=hm, kappa=kappa, p=p, la=la, lv=lv, lnu=lnu, lda=lda,
        )
        return -r if reverse else r

    ivp = solve_ivp(
        rhs, (0.0, L), y_start, method="RK45",
        rtol=rtol, atol=atol, max_step=max_step, dense_output=True,
    )
    if not ivp.success or ivp.sol is None:
        return None
    t = np.linspace(0.0, L, max(n, 8))
    Y = np.asarray(ivp.sol(t), dtype=float)
    if reverse:
        Y = Y.copy()
        Y[5:, :] *= -1.0
    return t, Y


def integrate_inclusion_two_leg(U0: float, U1: float, S1: float, *,
    c0: float = 0.0, hm: float = 0.0, kappa: float = 0.0, p: float = 0.0,
    la: float = 0.0, lv: float = 0.0, lnu: float = 0.0, lda: float = 0.0,
    n: int = 160, junction_frac: float = JUNCTION_FRAC, **ivp_kw: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Two-leg meridian at ``(U₀, U₁, S₁)``; return stitched profile.

    Returns ``(s, y, nu, y_north_end, y_south_end)`` or ``None``.
    """
    S1f = float(S1)
    s_bar = float(junction_frac) * S1f
    north_len = s_bar - S_POLE
    south_len = (S1f - S_POLE) - s_bar
    if north_len <= 0.02 or south_len <= 0.02:
        return None

    y0n = north_pole_state_incl(
        U0, c0=c0, la=la, lv=lv, kappa=kappa, p=p, hm=hm, lnu=lnu,
    )
    y0s = south_pole_state_incl(
        U1, c0=c0, la=la, lv=lv, kappa=kappa, p=p, hm=hm, lnu=lnu,
    )
    north = _integrate_inclusion_leg(
        y0n, north_len, c0=c0, hm=hm, kappa=kappa, p=p, la=la, lv=lv, lnu=lnu, lda=lda,
        reverse=False, n=max(n // 2, 8), **ivp_kw,
    )
    south = _integrate_inclusion_leg(
        y0s, south_len, c0=c0, hm=hm, kappa=kappa, p=p, la=la, lv=lv, lnu=lnu, lda=lda,
        reverse=True, n=max(n // 2, 8), **ivp_kw,
    )
    if north is None or south is None:
        return None
    tn, Yn = north
    ts, Ys = south
    yn = Yn[:, -1]
    ys = Ys[:, -1]

    # South tip starts at z=0; with reverse-arclength, z increases toward the
    # junction.  Match junction z to north and place the tip below.
    z_nj = float(yn[1])
    z_sj = float(Ys[1, -1])
    Ys = Ys.copy()
    Ys[1, :] = z_nj - z_sj + Ys[1, :]

    ts_rev = tn[-1] + (ts[-1] - ts[::-1])
    Ys_rev = Ys[:, ::-1]
    if Ys_rev.shape[1] > 1:
        ts_rev = ts_rev[1:]
        Ys_rev = Ys_rev[:, 1:]

    s = np.concatenate([tn, ts_rev])
    y = np.concatenate([Yn, Ys_rev], axis=1)
    y = y.copy()
    y[1, :] -= float(y[1, 0])  # north pole z = 0
    nu = np.array([
        nu_from_state(y[0, i], y[2, i], y[3, i], kappa=kappa, hm=hm, p=p, lnu=lnu)
        for i in range(y.shape[1])
    ], dtype=float)
    # Cumulative Δa along the stitched meridian (tip-to-tip).
    if y.shape[0] >= 9 and y.shape[1] > 1:
        da_dens = 0.25 * (np.sin(y[2]) + y[3])
        da_cum = np.concatenate([[0.0], np.cumsum(0.5 * (da_dens[1:] + da_dens[:-1]) * np.diff(s))])
        # Match two-leg total (north tip stub + legs + south tip stub ≈ yn[8]+ys[8])
        y = y.copy()
        y[8, :] = da_cum
        # Prefer exact two-leg sum for the endpoint used in residuals
        da_tot = float(yn[8] + ys[8]) if yn.size > 8 and ys.size > 8 else float(da_cum[-1])
        if abs(da_cum[-1]) > 1e-14:
            y[8, :] *= da_tot / da_cum[-1]
        yn = yn.copy(); ys = ys.copy()
    return s, y, nu, yn, ys


# --- Shooting -----------------------------------------------------------------

def meridian_area_difference(yn: np.ndarray, ys: np.ndarray) -> float:
    """Total ``Δa`` from two-leg tip integrals."""
    if yn.size > 8 and ys.size > 8:
        return float(yn[8] + ys[8])
    return float("nan")


def ade_lambda_from_da(da: float, da0: float, kr_over_kc: float) -> float:
    """Paper eq. (13): ``λ_Δa = 2 (k_r / k_c) (Δa − Δa₀)``.

    Matches ``w_RE = (k_r/k_c)(Δa−Δa₀)²`` in eq. (10) and the inversion
    ``Δa₀ = Δa − k_c λ_Δa / (2 k_r)`` stated after eq. (13).
    """
    return 2.0 * float(kr_over_kc) * (float(da) - float(da0))


def _junction_residuals(U0: float, U1: float, S1: float, la: float, lv: float, lnu: float, *,
    v_target: float, c0: float, hm: float, kappa: float, p: float,
    match_n: bool, lda: float = 0.0,
    match_ade: bool = False, da0: float = 0.0, kr_over_kc: float = 0.0,
) -> np.ndarray:
    """Match ψ, c_m, ρ, a=1, v̄ (and n=1 / ADE law when enabled)."""
    out = integrate_inclusion_two_leg(
        U0, U1, S1, c0=c0, hm=hm, kappa=kappa, p=p,
        la=la, lv=lv, lnu=lnu, lda=lda, n=80,
    )
    n_extra = (1 if match_n else 0) + (1 if match_ade else 0)
    n_base = 5
    if out is None:
        return np.full(n_base + n_extra, 1e2)
    _s, _y, _nu, yn, ys = out
    scale_u = max(
        abs(float(yn[3]) / max(abs(float(yn[0])), POLE_R)),
        abs(float(ys[3]) / max(abs(float(ys[0])), POLE_R)),
        1.0,
    )
    scale_x = max(float(yn[0]), float(ys[0]), 0.05)
    # Weight a, v, n strongly so ⟨ν⟩=n/a stays near 1 (a>1 ⇒ ν<1 everywhere).
    w_av = 20.0
    core = np.array([
        (float(yn[2]) - float(ys[2])) / np.pi,
        (
            float(yn[3]) / max(abs(float(yn[0])), POLE_R)
            - float(ys[3]) / max(abs(float(ys[0])), POLE_R)
        ) / scale_u,
        (float(yn[0]) - float(ys[0])) / scale_x,
        w_av * float(yn[5] + ys[5] - 1.0),
        w_av * (float(yn[6] + ys[6]) - float(v_target)),
    ], dtype=float)
    parts = [core]
    if match_n:
        parts.append(np.array([w_av * (float(yn[7] + ys[7]) - 1.0)], dtype=float))
    if match_ade:
        da = meridian_area_difference(yn, ys)
        parts.append(np.array([
            float(lda) - ade_lambda_from_da(da, da0, kr_over_kc)
        ], dtype=float))
    return np.concatenate(parts)


def shoot_inclusion_shape(v: float, c0: float = 0.0, *,
    hm: float = 0.0, kappa: float = 0.0, p: float = 0.0,
    U0: float = 1.0, U1: float = 1.0, S1: float = np.pi,
    la: float = 0.0, lv: float = 0.0, lnu: float = 0.0, lda: float = 0.0,
    kr_over_kc: float = 0.0, da0: float = 0.0,
    n: int = 200, verbose: bool = False, max_nfev: int = 80,
    residual_tol: float = 2e-2,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    free_lda: bool = True,
) -> InclusionSolution:
    """Two-leg shoot at fixed ``(v, c0, hm, κ, p[, k_r/k_c, Δa₀])``.

    Parameters: ``(U₀, U₁, S₁, λ_a, λ_v)``, plus ``λ_ν`` when ``p > 0``,
    plus ``λ_Δa`` when ``k_r/k_c > 0`` and ``free_lda`` (matched to eq. 13).

    If ``free_lda=False``, ``λ_Δa`` is held fixed in the ODEs (Picard / staged ADE)
    and is not a least-squares unknown — free ``λ_Δa`` otherwise collapses to the
    soft ``λ_Δa≈0``, ``Δa≈Δa₀`` root and blocks hard junction convergence.
    """
    t0 = time.perf_counter()
    v = float(v)
    match_n = float(p) > INCL_P_MIN
    lda = float(lda)
    da0 = float(da0)
    kr_over_kc = float(kr_over_kc)
    match_ade = bool(free_lda) and float(kr_over_kc) > 0.0
    lda_fixed = float(lda)

    # Parameter vector layout
    names = ["U0", "U1", "S1", "la", "lv"]
    x0_list = [U0, U1, S1, la, lv]
    if match_n:
        names.append("lnu"); x0_list.append(lnu)
    if match_ade:
        names.append("lda"); x0_list.append(lda)
    x0 = np.array(x0_list, dtype=float)
    idx = {n: i for i, n in enumerate(names)}

    def _unpack(x: np.ndarray):
        U0x, U1x, S1x = float(x[0]), float(x[1]), float(x[2])
        lax, lvx = float(x[3]), float(x[4])
        lnux = float(x[idx["lnu"]]) if match_n else 0.0
        ldax = float(x[idx["lda"]]) if match_ade else lda_fixed
        return U0x, U1x, S1x, lax, lvx, lnux, ldax

    def resid(x: np.ndarray) -> np.ndarray:
        U0x, U1x, S1x, lax, lvx, lnux, ldax = _unpack(x)
        return _junction_residuals(
            U0x, U1x, S1x, lax, lvx, lnux,
            v_target=v, c0=c0, hm=hm, kappa=kappa,
            p=p if match_n else 0.0,
            match_n=match_n, lda=ldax,
            match_ade=match_ade, da0=da0, kr_over_kc=kr_over_kc,
        )

    r0 = resid(x0)
    if verbose:
        print(f"  incl shoot |res|₀={np.linalg.norm(r0):.3e}  x0={x0}", flush=True)

    ls_kw: Dict[str, Any] = dict(
        method="trf", max_nfev=max_nfev,
        xtol=1e-9, ftol=1e-9, gtol=1e-9,
    )
    if bounds is not None:
        lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
        if lo.shape != x0.shape or hi.shape != x0.shape:
            raise ValueError("bounds must match shoot parameter vector")
        eps = 1e-9 * (1.0 + np.abs(x0))
        x0 = np.minimum(np.maximum(x0, lo + eps), hi - eps)
        ls_kw["bounds"] = (lo, hi)
        ls_kw["x_scale"] = np.maximum(np.abs(x0), 1.0)

    opt = least_squares(resid, x0, **ls_kw)
    x = opt.x
    U0f, U1f, S1f, laf, lvf, lnuf, ldaf = _unpack(x)

    out = integrate_inclusion_two_leg(
        U0f, U1f, S1f, c0=c0, hm=hm, kappa=kappa, p=p if match_n else 0.0,
        la=laf, lv=lvf, lnu=lnuf, lda=ldaf, n=n,
    )
    rf = resid(x)
    res_n = float(np.linalg.norm(rf))
    ok = bool(out is not None and res_n < float(residual_tol))
    msg = "inclusion junction matched" if ok else str(opt.message)

    if out is None:
        sol = InclusionSolution(
            v, c0, hm, kappa, p, U0f, U1f, laf, lvf, lnuf,
            np.array([]), np.zeros((9, 0)), np.array([]),
            "inclusions", False, "integration failed",
            {"residual": res_n, "S1": S1f},
            lda=ldaf, da0=da0, kr_over_kc=kr_over_kc,
        )
    else:
        s, y, nu, yn, ys = out
        v_tot = float(yn[6] + ys[6])
        n_tot = float(yn[7] + ys[7])
        da_tot = meridian_area_difference(yn, ys)
        constraints = {
            "residual": res_n,
            "rho_match": float(yn[0] - ys[0]),
            "psi_match": float(yn[2] - ys[2]),
            "chi_match": float(yn[3] - ys[3]),
            "lam_match": float(yn[4] - ys[4]),
            "v_end": v_tot,
            "n_end": n_tot,
            "a_north": float(yn[5]),
            "a_south": float(ys[5]),
            "da": da_tot,
            "lda_target": (
                ade_lambda_from_da(da_tot, da0, kr_over_kc)
                if float(kr_over_kc) > 0.0 else 0.0
            ),
            "S1": S1f,
            "free_lda": float(1.0 if match_ade else 0.0),
            "dt_wall": time.perf_counter() - t0,
        }
        sig, Pb = bozic_to_seifert_multipliers(laf, lvf)
        constraints["sigma_bar"] = sig
        constraints["P_bar"] = Pb
        sol = InclusionSolution(
            v_tot if np.isfinite(v_tot) else v, c0, hm, kappa, float(p),
            U0f, U1f, laf, lvf, lnuf, s, y, nu,
            "inclusions", ok, msg, constraints,
            lda=ldaf, da0=da0, kr_over_kc=kr_over_kc,
        )
    if verbose:
        print(
            f"  incl shoot done ok={sol.success} |res|={res_n:.3e}  "
            f"U0={U0f:.4f} U1={U1f:.4f} S1={S1f:.4f} v̄={sol.v:.4f}  "
            f"lda={ldaf:.4g} da={sol.constraints.get('da', float('nan')):.4f}  "
            f"({sol.constraints.get('dt_wall', 0):.2f}s)",
            flush=True,
        )
    return sol



def continue_inclusions_in_p(
    seed: MeridianSolution, *,
    p_final: float = INCL_P_DEFAULT, hm: float = INCL_HM_DEFAULT,
    kappa: float = INCL_KAPPA_DEFAULT,
    kr_over_kc: float = INCL_KR_OVER_KC_DEFAULT,
    da0: Optional[float] = INCL_DA0_DEFAULT,
    n_steps: int = 800, max_nfev: int = 80, n: int = 140,
    residual_tol: float = 1e-3, verbose: bool = True,
    max_dp: Optional[float] = None,
    u_box: float = 0.35, s_box: float = 0.35,
    prefer_rising_da: bool = True,
    use_fig1_lda_track: bool = True,
    ade_picard: bool = True,
    picard_iters: int = 6,
    checkpoint_path: Optional[Path | str] = None,
    checkpoint_every: int = 5,
    resume: bool = True,
) -> List[InclusionSolution]:
    """Warm-start shoot while ramping ``p`` from 0 to ``p_final``.

    Includes ADE when ``kr_over_kc > 0`` (Fig. 1 uses 3).  By default ADE is
    handled by Picard iteration (``ade_picard=True``): shoot with fixed
    ``λ_Δa`` in the ODEs, then update ``λ_Δa = 2(k_r/k_c)(Δa−Δa₀)``.  Freeing
    ``λ_Δa`` in the least-squares vector collapses onto a soft
    ``λ_Δa≈0``, ``Δa≈Δa₀`` root and stalls near ``p∼5``.

    If ``da0`` is ``None``, it is set from the ``p = 0`` shape.
    """
    v = float(seed.v)
    c0 = float(seed.c0)
    la0, lv0 = seifert_to_bozic_multipliers(seed.sigma_bar, seed.P_bar)
    asym_seed = abs(float(seed.U0) - float(seed.U1))
    seed_U0 = float(seed.U0)
    seed_U1 = float(seed.U1)
    # Stay on the stomatocyte neighborhood of the Fig. 1 seed.
    shape_prox_du = 1.5
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
    use_ade = float(kr_over_kc) > 0.0

    traj: List[InclusionSolution] = []
    if ckpt is not None and (not resume) and ckpt.exists():
        ckpt.unlink()
        if verbose:
            print(f"  p-continue: removed old checkpoint {ckpt} (resume=False)", flush=True)
    if ckpt is not None and resume and ckpt.exists():
        try:
            traj = load_inclusion_trajectory(ckpt)
            if traj:
                # Refuse a checkpoint from a different Seifert branch/seed.
                if abs(float(traj[0].U0) - float(seed.U0)) > 0.05 or abs(
                    float(traj[0].U1) - float(seed.U1)
                ) > 0.05:
                    if verbose:
                        print(
                            f"  p-continue: checkpoint seed mismatch "
                            f"(ckpt U0={traj[0].U0:.4f} U1={traj[0].U1:.4f} vs "
                            f"seed U0={seed.U0:.4f} U1={seed.U1:.4f}); starting fresh",
                            flush=True,
                        )
                    traj = []
                elif verbose:
                    print(
                        f"  p-continue: resumed {len(traj)} frames from {ckpt} "
                        f"(p={traj[-1].p:.4f})",
                        flush=True,
                    )
        except Exception as exc:
            traj = []
            if verbose:
                print(f"  p-continue: resume failed ({exc}); starting fresh", flush=True)

    # Resolve Δa0: prefer checkpoint, else argument, else measure p=0 seed.
    if traj:
        da0_use = float(traj[-1].da0)
        kr_use = float(traj[-1].kr_over_kc) if traj[-1].kr_over_kc > 0 else float(kr_over_kc)
    else:
        kr_use = float(kr_over_kc)
        if da0 is None:
            # Measure Δa on the Seifert seed meridian.
            da0_use = float(meridian_da_from_profile(seed.s, seed.y[2], seed.y[3]))
        else:
            da0_use = float(da0)

    if not traj:
        # p=0: Fig. 1 starts at the SC energy minimum with λ_Δa=0 and Δa=Δa0.
        # Shoot without ADE residual first; then lock Δa0 to that shape's Δa
        # (unless the caller passed an explicit da0).
        sol = shoot_inclusion_shape(
            v, c0, hm=0.0, kappa=0.0, p=0.0,
            U0=float(seed.U0), U1=float(seed.U1), S1=float(seed.S1),
            la=la0, lv=lv0, lnu=0.0, lda=0.0,
            kr_over_kc=0.0, da0=0.0,
            n=n, max_nfev=max_nfev, residual_tol=residual_tol, verbose=verbose,
        )
        if not sol.success:
            out = integrate_inclusion_two_leg(
                seed.U0, seed.U1, seed.S1, c0=c0, la=la0, lv=lv0, lda=0.0, n=n,
            )
            if out is None:
                raise RuntimeError("p=0 seed integrate failed")
            s, y, nu, yn, ys = out
            da_tot = meridian_area_difference(yn, ys)
            sol = InclusionSolution(
                float(yn[6] + ys[6]), c0, 0.0, 0.0, 0.0,
                float(seed.U0), float(seed.U1), la0, lv0, 0.0,
                s, y, nu, "inclusions_p0", True, "seed integrate",
                {"residual": 0.0, "S1": float(seed.S1), "da": da_tot},
                lda=0.0, da0=0.0, kr_over_kc=0.0,
            )
            if verbose:
                print("  p=0: using direct seed integrate", flush=True)
        da_meas = float(sol.constraints.get("da", sol.da))
        if da0 is None:
            da0_use = da_meas
        else:
            da0_use = float(da0)
        sol.da0 = da0_use
        sol.kr_over_kc = kr_use
        sol.lda = 0.0
        sol.constraints = dict(sol.constraints)
        sol.constraints["da"] = da_meas
        sol.constraints["lda_target"] = ade_lambda_from_da(da_meas, da0_use, kr_use)
        if verbose:
            print(
                f"  p-continue ADE: kr/kc={kr_use:g}  Δa0={da0_use:.6f}  "
                f"Δa(p=0)={da_meas:.6f}  λΔa={sol.lda:.6g}",
                flush=True,
            )
        traj = [sol]

    p_cur = float(traj[-1].p)
    lnu = float(traj[-1].lnu)
    lda = float(traj[-1].lda)
    dp_nom = float(p_final) / max(int(n_steps), 1)
    if max_dp is None:
        # Fig. 1 changes slowly in p; keep early steps modest.
        max_dp = min(2.0, max(0.25, dp_nom))
    dp = min(dp_nom, float(max_dp))

    def _checkpoint(force: bool = False) -> None:
        if ckpt is None:
            return
        if (not force) and (len(traj) % max(int(checkpoint_every), 1)) != 0:
            return
        save_inclusion_trajectory(
            ckpt, traj,
            meta={
                "v": v, "c0": c0, "hm": float(hm), "kappa": float(kappa),
                "kr_over_kc": kr_use, "da0": da0_use,
                "p_final": float(p_final), "n_steps": int(n_steps),
                "fig1_lda_track": bool(use_fig1_lda_track),
                "ade_picard": bool(ade_picard),
            },
        )
        if verbose and force:
            print(f"  p-continue: wrote checkpoint {ckpt} ({len(traj)} frames)", flush=True)

    free_lda = bool(use_ade) and (not bool(ade_picard))

    def _local_bounds(prev: InclusionSolution, p_try: float, lnu_guess: float,
                      la_guess: float, lda_guess: float) -> Tuple[np.ndarray, np.ndarray]:
        U0, U1 = float(prev.U0), float(prev.U1)
        S1 = float(prev.constraints.get("S1", prev.S1))
        lv = float(prev.lv)
        du0 = max(float(u_box), 0.08 * abs(U0), 0.15)
        du1 = max(float(u_box), 0.08 * abs(U1), 0.15)
        ds = max(float(s_box), 0.05 * abs(S1))
        # λ_a tracks ≈ λ_a(0) − p from the λ_ν term; keep room to follow.
        dla = max(2.0, 0.35 * abs(la_guess) + abs(p_try - float(prev.p)) + 1.0)
        dlv = max(1.5, 0.25 * abs(lv) + 0.5)
        dlnu = max(1.0, 0.2 * abs(lnu_guess) + 0.2 * abs(p_try))
        lo_list = [U0 - du0, U1 - du1, max(0.5, S1 - ds), la_guess - dla, lv - dlv]
        hi_list = [U0 + du0, U1 + du1, S1 + ds, la_guess + dla, lv + dlv]
        if float(p_try) > INCL_P_MIN:
            lo_list.append(lnu_guess - dlnu)
            hi_list.append(lnu_guess + dlnu)
        if free_lda:
            if use_fig1_lda_track:
                est = fig1_lambda_da_estimate(p_try, da0=da0_use, kr_over_kc=kr_use)
                lo_lda = max(0.0, min(lda_guess, est) - 0.02)
                hi_lda = max(lda_guess, est) + 0.15
                hi_lda = max(hi_lda, lo_lda + 0.05)
                hi_lda = min(hi_lda, max(0.25, est + 0.30))
            else:
                dlda = max(0.5, 0.35 * abs(lda_guess) + 0.2)
                lo_lda = lda_guess - dlda
                hi_lda = lda_guess + dlda
            lo_list.append(lo_lda)
            hi_list.append(hi_lda)
        return np.array(lo_list, dtype=float), np.array(hi_list, dtype=float)

    def _same_branch(trial: InclusionSolution) -> bool:
        if asym_seed < 0.4:
            return True
        asym = abs(float(trial.U0) - float(trial.U1))
        return asym > 0.35 * asym_seed

    def _near_seed_shape(trial: InclusionSolution) -> bool:
        """Reject far morphology jumps (e.g. stomatocyte → near-oblate)."""
        return (
            abs(float(trial.U0) - seed_U0) + abs(float(trial.U1) - seed_U1)
            < shape_prox_du
        )

    def _da_ok_for_fig1(trial: InclusionSolution, p_try: float) -> bool:
        """Reject Δa far below Δa₀; allow upward overshoot vs the Fig. 1 sketch."""
        if not use_fig1_lda_track:
            return True
        da = float(trial.constraints.get("da", trial.da))
        da_est = fig1_delta_a_estimate(p_try, da0=da0_use)
        # Lower floor near Δa₀; upper bound loose (sketch underestimates early rise).
        return (da0_use - 0.01) <= da <= max(da_est + 0.25, da0_use + 0.35)

    def _shoot_with_ade(
        p_try: float, *,
        U0w: float, U1w: float, S1w: float,
        la_w: float, lv_w: float, lnu_w: float, lda_w: float,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> InclusionSolution:
        """One step: free-λ_Δa least squares, or Picard with fixed λ_Δa."""
        if not use_ade:
            return shoot_inclusion_shape(
                v, c0, hm=hm, kappa=kappa, p=p_try,
                U0=U0w, U1=U1w, S1=S1w, la=la_w, lv=lv_w, lnu=lnu_w, lda=0.0,
                kr_over_kc=0.0, da0=da0_use,
                n=n, max_nfev=max_nfev, residual_tol=residual_tol,
                verbose=False, bounds=bounds, free_lda=False,
            )
        if free_lda:
            return shoot_inclusion_shape(
                v, c0, hm=hm, kappa=kappa, p=p_try,
                U0=U0w, U1=U1w, S1=S1w, la=la_w, lv=lv_w, lnu=lnu_w, lda=lda_w,
                kr_over_kc=kr_use, da0=da0_use,
                n=n, max_nfev=max_nfev, residual_tol=residual_tol,
                verbose=False, bounds=bounds, free_lda=True,
            )
        # Fixed-λ_Δa shoot (Picard in p): hold λ_Δa from the previous constitutive
        # value. Re-solving immediately with the updated λ_Δa kicks the junction
        # off the hard root; continuation in p supplies the outer iteration.
        sol_k = shoot_inclusion_shape(
            v, c0, hm=hm, kappa=kappa, p=p_try,
            U0=U0w, U1=U1w, S1=S1w, la=la_w, lv=lv_w, lnu=lnu_w, lda=float(lda_w),
            kr_over_kc=kr_use, da0=da0_use,
            n=n, max_nfev=max_nfev, residual_tol=residual_tol,
            verbose=False, bounds=bounds, free_lda=False,
        )
        da_k = float(sol_k.constraints.get("da", sol_k.da))
        lda_new = ade_lambda_from_da(da_k, da0_use, kr_use)
        sol_k.lda = float(lda_w)
        sol_k.constraints = dict(sol_k.constraints)
        sol_k.constraints["lda_target"] = lda_new
        sol_k.constraints["lda_fixed"] = float(lda_w)
        if verbose:
            print(
                f"      fixed-λΔa={float(lda_w):.4f} → da={da_k:.6f} "
                f"λΔa_const={lda_new:.4f} |res|={sol_k.constraints.get('residual'):.2e} "
                f"ok={sol_k.success}",
                flush=True,
            )
        return sol_k

    soft_streak = 0
    while p_cur + 1e-14 < float(p_final):
        prev = traj[-1]
        attempt = 0
        ok_sol: Optional[InclusionSolution] = None
        best_fail: Optional[InclusionSolution] = None
        step = min(dp, float(max_dp), float(p_final) - p_cur)
        used_soft = False
        while attempt < 8:
            p_try = p_cur + step
            if p_cur <= INCL_P_MIN:
                frozen = apply_frozen_nu(prev, hm=hm, kappa=kappa, p=max(p_try, 1e-3))
                lnu_guess = float(frozen.lnu)
            elif p_cur > INCL_P_MIN:
                lnu_guess = float(lnu) * (p_try / p_cur)
            else:
                lnu_guess = float(lnu)
            # Uniform-ν limit: λ_a shifts by ≈ −Δp to cancel the λ_ν term in λ̇.
            la_guess = float(prev.la) - (p_try - p_cur)
            da_prev = float(prev.constraints.get("da", prev.da))
            lda_prev = float(prev.lda)
            lda_from_da = ade_lambda_from_da(da_prev, da0_use, kr_use) if use_ade else 0.0
            # Cap the per-step λ_Δa advance so the ODE root can follow.
            dlda_max = 0.04 + 0.02 * abs(p_try - p_cur)
            lda_step = float(np.clip(lda_from_da - lda_prev, -dlda_max, dlda_max))
            lda_relax = lda_prev + lda_step
            lda_fig1 = (
                fig1_lambda_da_estimate(p_try, da0=da0_use, kr_over_kc=kr_use)
                if (use_ade and use_fig1_lda_track) else lda_from_da
            )
            warms: List[Tuple[float, float, float]] = [(0.0, 0.0, lda_relax)]
            if use_ade and abs(lda_prev - lda_relax) > 1e-5:
                warms.append((0.0, 0.0, lda_prev))
            if use_ade and abs(lda_fig1 - lda_relax) > 1e-3 and abs(lda_fig1 - lda_prev) > 1e-3:
                warms.append((0.0, 0.0, max(0.0, lda_fig1)))
            if prefer_rising_da and use_ade:
                warms.append((-0.05, 0.0, lda_relax))

            if verbose:
                print(
                    f"  p-continue: {p_cur:.4f} → {p_try:.4f}  "
                    f"(dp={step:.4g}, attempt {attempt + 1}, {len(warms)} warms, "
                    f"Δa0={da0_use:.4f}, λΔa={lda_prev:.4f}→{lda_relax:.4f} "
                    f"(const {lda_from_da:.4f}), picard={ade_picard and use_ade})",
                    flush=True,
                )

            candidates: List[InclusionSolution] = []
            for i_w, (dU0, dU1, lda_guess) in enumerate(warms):
                if verbose:
                    print(
                        f"    warm {i_w + 1}/{len(warms)}: "
                        f"ΔU0={dU0:+.3f} λΔa_guess={lda_guess:.4f} la_guess={la_guess:.4f}",
                        flush=True,
                    )
                bounds = _local_bounds(prev, p_try, lnu_guess, la_guess, lda_guess)
                t_shoot0 = time.perf_counter()
                try:
                    trial = _run_with_timeout(
                        lambda dU0=dU0, dU1=dU1, lda_guess=lda_guess, bounds=bounds: (
                            _shoot_with_ade(
                                p_try,
                                U0w=float(prev.U0) + dU0,
                                U1w=float(prev.U1) + dU1,
                                S1w=float(prev.constraints.get("S1", prev.S1)),
                                la_w=la_guess,
                                lv_w=float(prev.lv),
                                lnu_w=lnu_guess,
                                lda_w=lda_guess,
                                bounds=bounds,
                            )
                        ),
                        INCL_SHOOT_TIMEOUT_S * (1.0 + 0.5 * float(picard_iters if ade_picard else 1)),
                    )
                except _ShootTimeout as exc:
                    if verbose:
                        print(
                            f"      → TIMEOUT after shoot budget; skipping",
                            flush=True,
                        )
                    continue
                if verbose:
                    print(
                        f"      (shoot {time.perf_counter() - t_shoot0:.1f}s)",
                        flush=True,
                    )
                res = float(trial.constraints.get("residual", 1e9))
                branched = _same_branch(trial)
                proximal = _near_seed_shape(trial)
                da_ok = _da_ok_for_fig1(trial, p_try)
                if trial.success and not branched:
                    if verbose:
                        print(
                            f"  p-continue: reject branch jump "
                            f"U0={trial.U0:.3f} U1={trial.U1:.3f}",
                            flush=True,
                        )
                    continue
                if trial.success and branched and not proximal:
                    if verbose:
                        print(
                            f"  p-continue: reject far shape "
                            f"U0={trial.U0:.3f} U1={trial.U1:.3f}",
                            flush=True,
                        )
                    continue
                if trial.success and branched and proximal and not da_ok:
                    if verbose:
                        print(
                            f"  p-continue: reject Δa="
                            f"{trial.constraints.get('da'):.4f} off Fig.1 envelope "
                            f"(est {fig1_delta_a_estimate(p_try, da0=da0_use):.4f})",
                            flush=True,
                        )
                    continue
                if trial.success and branched and proximal and da_ok:
                    candidates.append(trial)
                    da_t = float(trial.constraints.get("da", trial.da))
                    if verbose:
                        print(
                            f"      → ok Δa={da_t:.6f} U0={trial.U0:.4f} "
                            f"λΔa={trial.lda:.4f} |res|={res:.2e}",
                            flush=True,
                        )
                elif branched and proximal and (
                    best_fail is None
                    or res < float(best_fail.constraints.get("residual", 1e9))
                ):
                    best_fail = trial

            if candidates:
                da_est = fig1_delta_a_estimate(p_try, da0=da0_use)

                def _score(s: InclusionSolution) -> Tuple[float, float, float]:
                    da = float(s.constraints.get("da", s.da))
                    # Prefer larger fixed λ_Δa (climbing ADE), then rising Δa.
                    return (float(s.lda), da, -abs(da - da_est))

                ok_sol = max(candidates, key=_score)
                da_new = float(ok_sol.constraints.get("da", ok_sol.da))
                # Bookkeeping: keep λ_Δa moving toward constitutive even if the
                # winning warm used a smaller trial value.
                if use_ade and ade_picard:
                    lda_t = float(ok_sol.constraints.get(
                        "lda_target", ade_lambda_from_da(da_new, da0_use, kr_use)
                    ))
                    dlda_max = 0.04 + 0.02 * abs(p_try - p_cur)
                    lda_adv = float(ok_sol.lda) + float(np.clip(
                        lda_t - float(ok_sol.lda), -dlda_max, dlda_max
                    ))
                    ok_sol.lda = lda_adv
                if verbose:
                    print(
                        f"  p-continue: accepted Δa={da_new:.6f} "
                        f"(est {da_est:.6f}, Δ={da_new - da_prev:+.6f})  "
                        f"U0={ok_sol.U0:.4f} λΔa={ok_sol.lda:.4f} "
                        f"|res|={ok_sol.constraints.get('residual'):.2e}",
                        flush=True,
                    )
                lnu = float(ok_sol.lnu)
                lda = float(ok_sol.lda)
                break
            step *= 0.5
            attempt += 1
            if step < 5e-4:
                break
        if ok_sol is None and best_fail is not None and _same_branch(best_fail):
            res = float(best_fail.constraints.get("residual", 1e9))
            a_err = abs(
                float(best_fail.constraints.get("a_north", 0.0))
                + float(best_fail.constraints.get("a_south", 0.0))
                - 1.0
            )
            n_err = abs(float(best_fail.constraints.get("n_end", 1.0)) - 1.0)
            # No soft-accept crawl: if we cannot hard-converge, end the run.
            if verbose:
                print(
                    f"  p-continue: no hard success (best |res|={res:.3e}, "
                    f"a−1={a_err:.2e}, n−1={n_err:.2e}, step={step:.4g}) — stalling",
                    flush=True,
                )
        if ok_sol is None:
            if verbose:
                print(f"  p-continue stalled at p={p_cur:.4f}", flush=True)
            break
        # Guard accepted steps too (hard success can still leave tiny a drift).
        a_ok = abs(
            float(ok_sol.constraints.get("a_north", 0.0))
            + float(ok_sol.constraints.get("a_south", 0.0))
            - 1.0
        )
        n_ok = abs(float(ok_sol.constraints.get("n_end", 1.0)) - 1.0)
        if a_ok > 5e-3 or n_ok > 5e-3:
            if verbose:
                print(
                    f"  p-continue stalled: area/n drift a−1={a_ok:.2e} n−1={n_ok:.2e} "
                    f"at p={ok_sol.p:.4f}",
                    flush=True,
                )
            break
        traj.append(ok_sol)
        p_cur = float(ok_sol.p)
        soft_streak = 0
        if attempt > 0:
            dp = min(float(max_dp), max(step, 5e-4))
        else:
            dp = min(float(max_dp), step * 1.05)
        # End if shape freezes while residual climbs (continuation lost the branch).
        if len(traj) >= 5:
            recent = traj[-5:]
            dU = sum(
                abs(recent[i + 1].U0 - recent[i].U0)
                + abs(recent[i + 1].U1 - recent[i].U1)
                for i in range(4)
            )
            r0 = float(recent[0].constraints.get("residual", 0.0))
            r1 = float(recent[-1].constraints.get("residual", 0.0))
            if dU < 0.015 and r1 > max(1e-3, 1.5 * r0):
                if verbose:
                    print(
                        f"  p-continue stalled: shape freeze "
                        f"(Σ|ΔU|={dU:.3e}, |res| {r0:.3e}→{r1:.3e}) at p={p_cur:.4f}",
                        flush=True,
                    )
                break
        _checkpoint(force=False)

    _checkpoint(force=True)
    if verbose:
        print(
            f"  p-continue done: {len(traj)} frames, p_final={traj[-1].p:.4f}  "
            f"ok={traj[-1].success} |res|={traj[-1].constraints.get('residual')}  "
            f"Δa={traj[-1].constraints.get('da')} λΔa={traj[-1].lda}",
            flush=True,
        )
    return traj


def meridian_da_from_profile(s: np.ndarray, psi: np.ndarray, chi: np.ndarray) -> float:
    """``Δa = (1/4) ∫ (sin ψ + χ) dl`` along a meridian profile."""
    s = np.asarray(s, dtype=float)
    psi = np.asarray(psi, dtype=float)
    chi = np.asarray(chi, dtype=float)
    if len(s) < 2:
        return float("nan")
    return float(0.25 * np.trapz(np.sin(psi) + chi, s))



# --- Energy (eq. 10, k_r = 0) -------------------------------------------------

def relative_free_energy(sol: InclusionSolution) -> float:
    """Relative free energy ``f = F/(8π k_c)`` with ``k_r = 0`` (eq. 10)."""
    if len(sol.s) < 2:
        return float("nan")
    rho = sol.rho
    psi = sol.psi
    chi = sol.chi
    nu = sol.nu
    s = sol.s
    rho_r = np.sqrt(rho * rho + POLE_R * POLE_R)
    c1 = chi / rho_r
    c2 = np.sin(psi) / rho_r
    da_dl = 0.5 * np.maximum(rho_r, 0.0)
    wb = 0.25 * (c1 + c2 - sol.c0) ** 2 * da_dl
    if sol.p <= INCL_P_MIN:
        fm = 0.0
    else:
        hm = sol.hm
        kap = sol.kappa
        Q1 = (c1 + c2 - 2.0 * hm) ** 2
        Q2 = (c1 - c2) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = nu * np.log(np.maximum(nu, 1e-300))
        fm = sol.p * (0.125 * kap * nu * Q1 + 0.125 * kap * Q2 + ent) * da_dl
        fm = np.where(np.isfinite(fm), fm, 0.0)
    return float(np.trapz(wb + fm, s))


def inclusion_to_meridian_solution(sol: InclusionSolution) -> MeridianSolution:
    """Project to a Seifert-style ``MeridianSolution`` (no inclusion field)."""
    if len(sol.s) < 2:
        return MeridianSolution(
            sol.v, sol.c0, sol.U0, sol.U1, 0.0, 0.0,
            np.array([]), np.zeros((7, 0)), sol.branch, False, sol.message, {},
        )
    sig, Pb = bozic_to_seifert_multipliers(sol.la, sol.lv)
    rho, z, psi, chi, lam, a, v, _n = sol.y
    rho_r = np.sqrt(rho * rho + POLE_R * POLE_R)
    U = chi / rho_r
    gamma = 4.0 * lam
    A = 4.0 * np.pi * a
    V = (4.0 * np.pi / 3.0) * v
    y = np.vstack([rho, z, psi, U, gamma, A, V])
    S1 = float(sol.constraints.get("S1", sol.S1))
    return MeridianSolution(
        float(sol.v), float(sol.c0), float(sol.U0), float(sol.U1),
        float(sig), float(Pb), sol.s, y, sol.branch, sol.success, sol.message,
        dict(sol.constraints), S1, float(JUNCTION_FRAC) * S1,
    )


# --- Regression / demo helpers ------------------------------------------------

def _closed_meridian_xz(sol: InclusionSolution) -> Tuple[np.ndarray, np.ndarray]:
    """Axisymmetric cross-section ``(x, z)`` (upright Z, left+right reflections)."""
    x_r = np.asarray(sol.X, dtype=float)
    z_up = -np.asarray(sol.Z, dtype=float)  # Seifert Ż=−sinψ → upright plot
    x = np.concatenate([x_r, -x_r[::-1]])
    z = np.concatenate([z_up, z_up[::-1]])
    z = z - 0.5 * (z.max() + z.min())
    return x, z


def _seifert_xz(sol: MeridianSolution) -> Tuple[np.ndarray, np.ndarray]:
    x_r = np.asarray(sol.X, dtype=float)
    z_up = -np.asarray(sol.Z, dtype=float)
    x = np.concatenate([x_r, -x_r[::-1]])
    z = np.concatenate([z_up, z_up[::-1]])
    z = z - 0.5 * (z.max() + z.min())
    return x, z


def rescale_meridian_to_unit_area(seed: MeridianSolution, *,
    label: Optional[str] = None,
) -> MeridianSolution:
    """Rescale a stored Seifert meridian to ``A = 4π`` (no re-integration)."""
    A_src = float(meridian_area(seed))
    if not np.isfinite(A_src) or A_src <= 0.0:
        raise ValueError("seed has non-positive area")
    scale = float(np.sqrt(A_STAR / A_src))
    U0, U1, S1, sig, P = rescale_shoot_params_by_length(
        seed.U0, seed.U1, seed.S1, seed.sigma_bar, seed.P_bar, scale,
    )
    y = np.asarray(seed.y, dtype=float).copy()
    y[0, :] *= scale
    y[1, :] *= scale
    y[3, :] /= scale
    y[4, :] /= scale
    y[5, :] *= scale * scale
    y[6, :] *= scale ** 3
    s = np.asarray(seed.s, dtype=float) * scale
    s_bar = float(seed.S_bar) * scale if np.isfinite(getattr(seed, "S_bar", np.nan)) else JUNCTION_FRAC * S1
    constraints = dict(seed.constraints)
    constraints.update({
        "length_scale": scale,
        "A_source": A_src,
        "A_end": float(y[5, -1]) if y.shape[1] else float("nan"),
        "V_end": float(y[6, -1]) if y.shape[1] else float("nan"),
    })
    name = label or str(seed.branch)
    return MeridianSolution(
        float(seed.v), float(seed.c0), float(U0), float(U1),
        float(sig), float(P), s, y, name, True,
        f"rescaled {name}", constraints, float(S1), float(s_bar),
    )


def load_unit_area_seifert_seed(path: Optional[Path] = None) -> MeridianSolution:
    """Load a stored Seifert shape and rescale the meridian to ``A = 4π``."""
    path = Path(path) if path is not None else DEFAULT_SEIFERT_SEED
    seed = load_solution(path)
    sol = rescale_meridian_to_unit_area(seed, label=f"seed:{path.name}")
    sol.constraints["seed_path"] = str(path)
    return sol


def seifert_to_inclusion_solution(seif: MeridianSolution, *,
    hm: float = 0.0, kappa: float = 0.0, p: float = 0.0, lnu: float = 0.0,
) -> InclusionSolution:
    """Convert a Seifert meridian to ``InclusionSolution`` state (χ=ρU, λ=γ/4)."""
    la, lv = seifert_to_bozic_multipliers(seif.sigma_bar, seif.P_bar)
    if len(seif.s) < 2:
        return InclusionSolution(
            seif.v, seif.c0, hm, kappa, p, seif.U0, seif.U1, la, lv, lnu,
            np.array([]), np.zeros((9, 0)), np.array([]),
            "from_seifert", False, "empty", {},
        )
    X, Z, psi, U, gamma, A, V = seif.y
    rho_r = np.sqrt(X * X + POLE_R * POLE_R)
    chi = rho_r * U
    lam = 0.25 * gamma
    a = A / (4.0 * np.pi)
    v = 3.0 * V / (4.0 * np.pi)
    nu = np.array([
        nu_from_state(float(X[i]), float(psi[i]), float(chi[i]),
                      kappa=kappa, hm=hm, p=p, lnu=lnu)
        for i in range(len(seif.s))
    ], dtype=float)
    # cumulative reduced inclusion number along meridian
    n = np.zeros_like(a)
    if len(seif.s) > 1:
        integrand = 0.5 * nu * rho_r
        n = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(seif.s))])
        if len(n) != len(seif.s):
            n = np.interp(seif.s, seif.s[:len(n)], n)
    da_dens = 0.25 * (np.sin(psi) + chi)
    da = np.concatenate([[0.0], np.cumsum(0.5 * (da_dens[1:] + da_dens[:-1]) * np.diff(seif.s))])
    y = np.vstack([X, Z, psi, chi, lam, a, v, n, da])
    return InclusionSolution(
        float(seif.v), float(seif.c0), hm, kappa, p,
        float(seif.U0), float(seif.U1), la, lv, lnu,
        np.asarray(seif.s, dtype=float), y, nu,
        "from_seifert", True, "converted from Seifert seed",
        {"S1": float(seif.S1), "sigma_bar": float(seif.sigma_bar), "P_bar": float(seif.P_bar),
         "da": float(da[-1]) if len(da) else float("nan")},
    )


def _rhs_identity_metrics(seif: MeridianSolution, *, n_sample: int = 80) -> Dict[str, float]:
    """Pointwise ``p=0`` inclusion RHS vs Seifert ``meridian_rhs`` on a stored shape."""
    from seifert_functions import meridian_rhs

    la, lv = seifert_to_bozic_multipliers(seif.sigma_bar, seif.P_bar)
    idx = np.linspace(0, len(seif.s) - 1, min(n_sample, len(seif.s)), dtype=int)
    errs = []
    for i in idx:
        X, Z, psi, U, gamma, A, V = seif.y[:, i]
        chi = float(np.sqrt(X * X + POLE_R * POLE_R) * U)
        lam = 0.25 * float(gamma)
        # reduced a,v unused by geometric RHS
        yi = np.array([X, Z, psi, chi, lam, 0.0, 0.0, 0.0], dtype=float)
        ys = np.array([X, Z, psi, U, gamma, 0.0, 0.0], dtype=float)
        ri = inclusion_meridian_rhs(
            0.0, yi, c0=float(seif.c0), hm=0.0, kappa=0.0, p=0.0, la=la, lv=lv, lnu=0.0,
        )
        rs = meridian_rhs(0.0, ys, float(seif.c0), float(seif.sigma_bar), float(seif.P_bar))
        rho_r = float(np.sqrt(X * X + POLE_R * POLE_R))
        Ud = (ri[3] * rho_r - chi * ri[0]) / (rho_r * rho_r)
        mapped = np.array([ri[0], ri[1], ri[2], Ud, 4.0 * ri[4]])
        errs.append(float(np.linalg.norm(mapped - rs[:5])))
    errs_a = np.asarray(errs, dtype=float)
    return {
        "rhs_abs_err_max": float(np.max(errs_a)),
        "rhs_abs_err_mean": float(np.mean(errs_a)),
        "rhs_abs_err_median": float(np.median(errs_a)),
        "n_sample": float(len(errs_a)),
    }


def _seifert_reintegrate(incl: InclusionSolution) -> Tuple[np.ndarray, Dict[str, float]]:
    """Re-integrate Seifert ``meridian_rhs`` along the inclusion arclength grid."""
    from seifert_functions import meridian_rhs

    if len(incl.s) < 5:
        return np.zeros((7, 0)), {"state_abs_err_max": float("nan")}
    sig, Pb = bozic_to_seifert_multipliers(incl.la, incl.lv)
    y0 = np.array([
        float(incl.X[0]), float(incl.Z[0]), float(incl.psi[0]), float(incl.U[0]),
        float(4.0 * incl.lam[0]), 0.0, 0.0,
    ], dtype=float)
    t_eval = np.asarray(incl.s - incl.s[0], dtype=float)

    def rhs(s: float, y: np.ndarray) -> np.ndarray:
        return meridian_rhs(s, y, float(incl.c0), sig, Pb)

    ivp = solve_ivp(
        rhs, (0.0, float(t_eval[-1])), y0, t_eval=t_eval,
        rtol=1e-9, atol=1e-11, max_step=0.02,
    )
    metrics: Dict[str, float] = {
        "sigma_bar": sig,
        "P_bar": Pb,
        "reintegrate_success": float(ivp.success),
    }
    if not ivp.success or ivp.y.shape[1] != len(incl.s):
        metrics["state_abs_err_max"] = float("nan")
        return np.zeros((7, 0)), metrics

    U_incl = incl.U
    err_X = float(np.max(np.abs(ivp.y[0] - incl.X)))
    err_Z = float(np.max(np.abs(ivp.y[1] - incl.Z)))
    err_psi = float(np.max(np.abs(ivp.y[2] - incl.psi)))
    err_U = float(np.max(np.abs(ivp.y[3] - U_incl)))
    err_g = float(np.max(np.abs(ivp.y[4] - 4.0 * incl.lam)))
    mask = (incl.s > 0.05) & (incl.s < incl.s[-1] - 0.05)
    if np.count_nonzero(mask) > 5:
        err_X_b = float(np.max(np.abs(ivp.y[0, mask] - incl.X[mask])))
        err_psi_b = float(np.max(np.abs(ivp.y[2, mask] - incl.psi[mask])))
        err_U_b = float(np.max(np.abs(ivp.y[3, mask] - U_incl[mask])))
        err_g_b = float(np.max(np.abs(ivp.y[4, mask] - 4.0 * incl.lam[mask])))
    else:
        err_X_b, err_psi_b, err_U_b, err_g_b = err_X, err_psi, err_U, err_g
    metrics.update({
        "X_abs_err_max": err_X,
        "Z_abs_err_max": err_Z,
        "psi_abs_err_max": err_psi,
        "U_abs_err_max": err_U,
        "gamma_abs_err_max": err_g,
        "state_abs_err_max": float(max(err_X, err_Z, err_psi, err_U, err_g)),
        "bulk_X_abs_err_max": err_X_b,
        "bulk_psi_abs_err_max": err_psi_b,
        "bulk_U_abs_err_max": err_U_b,
        "bulk_gamma_abs_err_max": err_g_b,
        "bulk_state_abs_err_max": float(max(err_X_b, err_psi_b, err_U_b, err_g_b)),
    })
    return ivp.y, metrics


def compare_p0_to_seifert(seed_path: Optional[Path] = None, *,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compare ``p=0`` inclusion EL ODEs to a stored Seifert seed (no re-shoot).

    Uses pointwise RHS identity on the stored meridian.  Full Seifert
    reintegration along the pear profile can stall at the thin neck, so it is
    reserved for the analytic sphere check.
    """
    seif = load_unit_area_seifert_seed(seed_path)
    incl = seifert_to_inclusion_solution(seif, p=0.0)
    rhs = _rhs_identity_metrics(seif)
    metrics: Dict[str, float] = {
        "seif_v": float(seif.v),
        "seif_c0": float(seif.c0),
        "seif_U0": float(seif.U0),
        "seif_U1": float(seif.U1),
        "seif_S1": float(seif.S1),
        "rho_max": float(np.max(seif.X)) if len(seif.s) else float("nan"),
        "E_bend": bending_energy(seif, C0=float(seif.c0)),
        "Z_span": float(incl.Z[0] - incl.Z[-1]) if len(incl.s) else float("nan"),
        "X_pole_north": float(incl.X[0]) if len(incl.s) else float("nan"),
        "X_pole_south": float(incl.X[-1]) if len(incl.s) else float("nan"),
    }
    metrics.update(rhs)
    if verbose:
        print(
            f"  Seifert seed: v̄={seif.v:.4f} c₀={seif.c0:.4f}  "
            f"U0={seif.U0:.4f} U1={seif.U1:.4f} S1={seif.S1:.4f}  "
            f"ρ_max={metrics['rho_max']:.4f}",
            flush=True,
        )
        print("  p=0 identity:", {k: metrics[k] for k in (
            "rhs_abs_err_max", "rhs_abs_err_mean", "rhs_abs_err_median",
            "X_pole_north", "X_pole_south", "Z_span",
        ) if k in metrics}, flush=True)
    return metrics


def regress_p0_sphere(*, verbose: bool = True) -> Dict[str, float]:
    """Unit sphere at ``p=0``: integrate inclusions, reintegrate Seifert ODE."""
    out = integrate_inclusion_two_leg(
        1.0, 1.0, np.pi, c0=2.0, la=0.0, lv=0.0, n=200,
    )
    if out is None:
        metrics = {"incl_success": 0.0}
        if verbose:
            print("  sphere integrate failed", flush=True)
        return metrics
    s, y, nu, yn, ys = out
    incl = InclusionSolution(
        float(yn[6] + ys[6]), 2.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0, s, y, nu,
        "sphere", True, "analytic sphere",
        {"residual": float(np.linalg.norm([
            yn[0] - ys[0], yn[2] - ys[2], yn[5] + ys[5] - 1.0, yn[6] + ys[6] - 1.0,
        ])), "S1": float(s[-1])},
    )
    _y, metrics = _seifert_reintegrate(incl)
    metrics.update({
        "incl_success": 1.0,
        "incl_residual": float(incl.constraints["residual"]),
        "incl_v": float(incl.v),
        "a_tot": float(yn[5] + ys[5]),
        "rho_max": float(np.max(incl.X)),
        "Z_span": float(incl.Z[0] - incl.Z[-1]),
    })
    if verbose:
        print("  sphere p=0:", {k: metrics[k] for k in (
            "bulk_state_abs_err_max", "bulk_X_abs_err_max", "incl_v", "a_tot", "Z_span",
        ) if k in metrics}, flush=True)
    return metrics


def _fit_lnu_for_frozen_shape(incl: InclusionSolution, *,
    hm: float, kappa: float, p: float,
) -> Tuple[float, np.ndarray]:
    """Choose ``λ_ν`` on a frozen shape so ``∫ ν da ≈ 1`` (eq. 19)."""
    from scipy.optimize import brentq

    rho = incl.rho
    psi = incl.psi
    chi = incl.chi
    s = incl.s
    rho_r = np.sqrt(rho * rho + POLE_R * POLE_R)

    def n_tot(lnu: float) -> float:
        nu = np.array([
            nu_from_state(float(rho[i]), float(psi[i]), float(chi[i]),
                          kappa=kappa, hm=hm, p=p, lnu=lnu)
            for i in range(len(s))
        ])
        return float(np.trapz(0.5 * nu * rho_r, s))

    # bracket: larger lnu → larger ν
    lo, hi = -20.0, 20.0
    f_lo, f_hi = n_tot(lo) - 1.0, n_tot(hi) - 1.0
    if f_lo * f_hi > 0:
        lnu = 0.0
    else:
        lnu = float(brentq(lambda z: n_tot(z) - 1.0, lo, hi))
    nu = np.array([
        nu_from_state(float(rho[i]), float(psi[i]), float(chi[i]),
                      kappa=kappa, hm=hm, p=p, lnu=lnu)
        for i in range(len(s))
    ])
    return lnu, nu


def apply_frozen_nu(incl: InclusionSolution, *,
    hm: float = INCL_HM_DEFAULT, kappa: float = INCL_KAPPA_DEFAULT,
    p: float = INCL_P_DEFAULT,
) -> InclusionSolution:
    """Return a copy with ``ν(l)`` from eq. (19) normalized to ``∫ ν da = 1``."""
    out = incl.copy()
    lnu, nu = _fit_lnu_for_frozen_shape(out, hm=hm, kappa=kappa, p=p)
    out.hm = float(hm)
    out.kappa = float(kappa)
    out.p = float(p)
    out.lnu = float(lnu)
    out.nu = nu
    out.constraints = dict(out.constraints)
    out.constraints["lnu"] = float(lnu)
    out.constraints["frozen_shape"] = 1.0
    out.constraints["nu_min"] = float(np.min(nu)) if len(nu) else float("nan")
    out.constraints["nu_max"] = float(np.max(nu)) if len(nu) else float("nan")
    out.message = "frozen shape; ν from eq. (19) with ∫ν da = 1"
    return out


def meridian_principal_curvatures(incl: InclusionSolution) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(c_m, c_p, c_m+c_p)`` along the meridian (``c_m=χ/ρ``, ``c_p=sinψ/ρ``)."""
    rho = np.asarray(incl.rho, dtype=float)
    psi = np.asarray(incl.psi, dtype=float)
    chi = np.asarray(incl.chi, dtype=float)
    rho_r = np.sqrt(rho * rho + POLE_R * POLE_R)
    c_m = chi / rho_r
    c_p = np.sin(psi) / rho_r
    return c_m, c_p, c_m + c_p


def meridian_z_upright(incl: InclusionSolution) -> np.ndarray:
    """Meridian ``z`` in the upright plotting frame (matches shape panels)."""
    z_up = -np.asarray(incl.Z, dtype=float)
    return z_up - 0.5 * (float(z_up.max()) + float(z_up.min()))


def plot_delta_a_vs_p(
    traj: Sequence[InclusionSolution] | Path | str, *,
    out_path: Optional[Path | str] = None,
    verbose: bool = True,
) -> Path:
    """Fig. 1(b)-style ``Δa(p)`` for an inclusion trajectory."""
    import matplotlib.pyplot as plt

    if isinstance(traj, (str, Path)):
        traj = load_inclusion_trajectory(traj)
    if not traj:
        raise ValueError("empty trajectory")
    if out_path is None:
        out_path = Path("results") / "inclusions_demo" / "stomatocyte_fig1_delta_a_vs_p.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = np.array([float(s.p) for s in traj], dtype=float)
    da = np.array([float(s.constraints.get("da", s.da)) for s in traj], dtype=float)
    da0 = float(traj[0].da0)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(p, da, color="C0", lw=2.0, marker="o", ms=3.5, label=r"$\Delta a(p)$")
    ax.axhline(da0, color="0.45", ls="--", lw=1.0, label=rf"$\Delta a_0$={da0:.4f}")
    ax.set_xlabel(r"$p = N_T k_B T / (8\pi k_c)$")
    ax.set_ylabel(r"relative area difference $\Delta a$")
    ax.set_title(
        rf"Fig. 1(b)–style  ($h_m$={traj[-1].hm:g}, $\kappa$={traj[-1].kappa:g}, "
        rf"$k_r/k_c$={traj[-1].kr_over_kc:g})"
    )
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  wrote {out_path}  (Δa: {da[0]:.4f} → {da[-1]:.4f}, p≤{p[-1]:.4g})", flush=True)
    return out_path


def demo_stomatocyte_nu_vs_curvature(*,
    seed_path: Optional[Path] = None,
    hm: float = INCL_HM_DEFAULT, kappa: float = INCL_KAPPA_DEFAULT,
    p: float = INCL_P_DEFAULT,
    kr_over_kc: float = INCL_KR_OVER_KC_DEFAULT,
    da0: Optional[float] = INCL_DA0_DEFAULT,
    n_steps: int = 800, shoot: bool = True,
    out_png: Optional[Path] = None,
    traj_path: Optional[Path] = None,
    animate: bool = True,
    resume: bool = True,
    verbose: bool = True,
) -> Path:
    """Stomatocyte: Seifert seed vs shot inclusion shape; ν(l) and curvature.

    If ``shoot=True`` (default), ramps ``p`` from 0 → ``p`` with warm starts,
    including ADE (``kr_over_kc``, default 3 as in Fig. 1).
    Frames are checkpointed to ``traj_path``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root = Path(__file__).resolve().parent
    if seed_path is None:
        seed_path = root / "seed_shapes" / "stomatocyte_starting_seed.json"
        if not seed_path.exists():
            seed_path = root / "seed_shapes" / "stomatocyte_seifert_seed.json"
    seed_path = Path(seed_path)
    if traj_path is None:
        traj_path = Path("results") / "inclusions_demo" / "stomatocyte_fig1_trajectory.json"
    traj_path = Path(traj_path)

    seif = load_unit_area_seifert_seed(seed_path)
    seed_incl = seifert_to_inclusion_solution(seif, p=0.0)

    if shoot:
        traj = continue_inclusions_in_p(
            seif, p_final=p, hm=hm, kappa=kappa,
            kr_over_kc=kr_over_kc, da0=da0,
            n_steps=n_steps, verbose=verbose,
            checkpoint_path=traj_path, checkpoint_every=5, resume=resume,
        )
        warm = traj[-1]
        p0 = traj[0]
    else:
        warm = apply_frozen_nu(seed_incl, hm=hm, kappa=kappa, p=p)
        p0 = seed_incl
        traj = [p0, warm]
        save_inclusion_trajectory(traj_path, traj, meta={"frozen": True})

    _cm, _cp, c_sum = meridian_principal_curvatures(warm)

    xs, zs = _seifert_xz(seif)
    x0, z0 = _closed_meridian_xz(p0)
    # Shape change vs seed (closed contours, resampled)
    n_c = 200
    # use radial distance from centroid as a crude mismatch on overlapping s
    if len(warm.s) > 5 and len(seed_incl.s) > 5:
        s_w = warm.s - warm.s[0]
        s_s = seed_incl.s - seed_incl.s[0]
        s_c = np.linspace(0.0, min(float(s_w[-1]), float(s_s[-1])), n_c)
        Xw = np.interp(s_c, s_w, warm.X)
        Zw = np.interp(s_c, s_w, warm.Z)
        Xs = np.interp(s_c, s_s, seed_incl.X)
        Zs = np.interp(s_c, s_s, seed_incl.Z)
        shape_err = float(np.max(np.hypot(Xw - Xs, Zw - Zs)))
    else:
        shape_err = float("nan")

    if len(warm.nu) > 2 and np.std(warm.nu) > 0 and np.std(c_sum) > 0:
        corr = float(np.corrcoef(warm.nu, c_sum)[0, 1])
    else:
        corr = float("nan")

    if out_png is None:
        out_png = Path("results") / "inclusions_demo" / "stomatocyte_seifert_nu_curvature.png"
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax = axes[0]
    ax.plot(xs, zs, color="0.15", lw=1.8, ls="--", zorder=3, label="Seifert seed (p=0)")
    if len(warm.s) > 5:
        lc = plot_nu_colored_meridian(ax, warm)
    else:
        lc = None
    ax.set_xlabel(r"$x / R_s$")
    ax.set_ylabel(r"$z / R_s$")
    mode = "shot" if shoot else "frozen"
    ax.set_title(
        rf"stomatocyte_seifert  $v̄$={seif.v:.3f}, $c_0$={seif.c0:g}  [{mode}]"
        + "\n"
        + rf"$p$={warm.p:g}, shape $|Δr|_{{\max}}$ vs seed = {shape_err:.2e}"
    )
    handles = [
        Line2D([0], [0], color="0.15", lw=1.8, ls="--", label="Seifert seed (p=0)"),
        Line2D([0], [0], color=plt.cm.cividis(0.7), lw=3.0, label=rf"inclusions $p$={warm.p:g}"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="best")
    if lc is not None:
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$\nu$")

    ax = axes[1]
    z_mer = meridian_z_upright(warm)
    ln_nu, = ax.plot(z_mer, warm.nu, color="C1", lw=1.8, label=r"$\nu(z)$")
    ax.set_xlabel(r"$z / R_s$")
    ax.set_ylabel(r"relative density $\nu$", color="C1")
    ax.tick_params(axis="y", labelcolor="C1")

    ax2 = ax.twinx()
    ln_c, = ax2.plot(z_mer, c_sum, color="C0", lw=1.6, label=r"$c_m + c_p$")
    ax2.set_ylabel(r"total curvature $c_m + c_p$", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")

    ax.set_title(
        rf"$h_m$={hm:g}, $\kappa$={kappa:g}, $p$={warm.p:g}"
        + "\n"
        + rf"corr$(\nu,\,c_m+c_p)$ = {corr:.3f}"
        + (f",  |res|={warm.constraints.get('residual', float('nan')):.2e}" if shoot else "")
    )
    lines = [ln_nu, ln_c]
    ax.legend(lines, [ln.get_label() for ln in lines], frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  wrote {out_png}", flush=True)
        print(f"  trajectory → {traj_path}  ({len(traj)} frames, p≤{traj[-1].p:.4g})", flush=True)
        print(
            f"  mode={mode}  frames={len(traj)}  shape_err={shape_err:.3e}  "
            f"corr(ν, c_m+c_p)={corr:.4f}  "
            f"ν∈[{warm.nu.min():.3f},{warm.nu.max():.3f}]  "
            f"C∈[{c_sum.min():.3f},{c_sum.max():.3f}]",
            flush=True,
        )
    if shoot and len(traj) >= 2:
        da_png = Path(traj_path).with_name("stomatocyte_fig1_delta_a_vs_p.png")
        plot_delta_a_vs_p(traj, out_path=da_png, verbose=verbose)
    if animate and len(traj) >= 2:
        anim_path = Path("results") / "inclusions_demo" / "stomatocyte_fig1_anim.mp4"
        if traj_path is not None:
            # Prefer fig1_anim.mp4 next to the trajectory when name matches.
            stem = Path(traj_path).stem
            if "fig1" in stem or stem.startswith("stomatocyte"):
                anim_path = Path(traj_path).with_name("stomatocyte_fig1_anim.mp4")
            else:
                anim_path = Path(traj_path).with_name(stem + "_anim.mp4")
        animate_inclusion_p_trajectory(
            traj, out_path=anim_path, seed=seif, verbose=verbose,
        )
    return out_png


def animate_inclusion_p_trajectory(
    traj: Sequence[InclusionSolution] | Path | str, *,
    out_path: Optional[Path | str] = None,
    seed: Optional[MeridianSolution | InclusionSolution] = None,
    max_frames: int = 200,
    fps: int = 10,
    dpi: int = 120,
    verbose: bool = True,
) -> Path:
    """Animate like ``stomatocyte_seifert_nu_curvature.png``.

    Left: dashed seed + ν-colored shot shape.  Right: ν(z) and c_m+c_p vs
    upright ``z`` (paper Fig. 1a style; arclength is the parameter).
    ``traj`` may be frames or a JSON path.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    if isinstance(traj, (str, Path)):
        traj = load_inclusion_trajectory(traj)
    traj = [s for s in traj if len(s.s) > 5]
    if len(traj) < 2:
        raise ValueError("need ≥2 valid frames to animate")

    if out_path is None:
        out_path = Path("results") / "inclusions_demo" / "stomatocyte_fig1_anim.mp4"
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(traj) > max_frames:
        idx = np.unique(np.linspace(0, len(traj) - 1, max_frames, dtype=int))
        frames = [traj[i] for i in idx]
    else:
        frames = list(traj)

    # Seed contour (dashed reference).
    if seed is None:
        seed = traj[0]
    if isinstance(seed, MeridianSolution):
        xs_seed, zs_seed = _seifert_xz(seed)
        v_seed, c0_seed = float(seed.v), float(seed.c0)
    else:
        xs_seed, zs_seed = _closed_meridian_xz(seed)
        v_seed, c0_seed = float(seed.v), float(seed.c0)

    xs_all, zs_all, nu_all, csum_all, z_all = [], [], [], [], []
    z_lo, z_hi = np.inf, -np.inf
    c_lo, c_hi = np.inf, -np.inf
    for sol in frames:
        x, z, c = _closed_meridian_xyz_nu(sol)
        _cm, _cp, c_sum = meridian_principal_curvatures(sol)
        z_mer = meridian_z_upright(sol)
        xs_all.append(x); zs_all.append(z); nu_all.append(c)
        csum_all.append(np.asarray(c_sum, dtype=float))
        z_all.append(z_mer)
        z_lo = min(z_lo, float(np.min(z_mer)))
        z_hi = max(z_hi, float(np.max(z_mer)))
        c_lo = min(c_lo, float(np.min(c_sum)))
        c_hi = max(c_hi, float(np.max(c_sum)))
    nu_lo = float(min(np.min(c) for c in nu_all))
    nu_hi = float(max(np.max(c) for c in nu_all))
    if abs(nu_hi - nu_lo) < 1e-9:
        nu_lo, nu_hi = nu_lo - 0.05, nu_hi + 0.05
    if not np.isfinite(c_lo) or abs(c_hi - c_lo) < 1e-9:
        c_lo, c_hi = c_lo - 0.5, c_hi + 0.5
    pad = 0.08 * max(
        max(np.ptp(x) for x in xs_all),
        max(np.ptp(z) for z in zs_all),
        float(np.ptp(xs_seed)) if len(xs_seed) else 0.0,
        float(np.ptp(zs_seed)) if len(zs_seed) else 0.0,
        0.5,
    )
    xlim = (
        min(min(x.min() for x in xs_all), float(xs_seed.min())) - pad,
        max(max(x.max() for x in xs_all), float(xs_seed.max())) + pad,
    )
    ylim = (
        min(min(z.min() for z in zs_all), float(zs_seed.min())) - pad,
        max(max(z.max() for z in zs_all), float(zs_seed.max())) + pad,
    )
    nu_pad = 0.05 * (nu_hi - nu_lo)
    c_pad = 0.05 * (c_hi - c_lo)
    z_pad = 0.05 * max(z_hi - z_lo, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax_sh, ax_nu = axes
    ax_sh.set_aspect("equal")
    ax_sh.set_xlim(*xlim)
    ax_sh.set_ylim(*ylim)
    ax_sh.set_xlabel(r"$x / R_s$")
    ax_sh.set_ylabel(r"$z / R_s$")
    ax_sh.plot(xs_seed, zs_seed, color="0.15", lw=1.8, ls="--", zorder=3)
    title_sh = ax_sh.set_title("")
    ax_sh.legend(
        handles=[
            Line2D([0], [0], color="0.15", lw=1.8, ls="--", label="Seifert seed (p=0)"),
            Line2D([0], [0], color=plt.cm.cividis(0.7), lw=3.0, label="inclusions"),
        ],
        frameon=False, fontsize=9, loc="best",
    )

    pts0 = np.array([xs_all[0], zs_all[0]]).T.reshape(-1, 1, 2)
    segs0 = np.concatenate([pts0[:-1], pts0[1:]], axis=1)
    lc = LineCollection(segs0, cmap="cividis", norm=plt.Normalize(nu_lo, nu_hi), linewidths=3.0)
    lc.set_array(0.5 * (nu_all[0][:-1] + nu_all[0][1:]))
    ax_sh.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax_sh, fraction=0.046, pad=0.04)
    cb.set_label(r"$\nu$")

    ax_nu.set_xlim(z_lo - z_pad, z_hi + z_pad)
    ax_nu.set_ylim(nu_lo - nu_pad, nu_hi + nu_pad)
    ax_nu.set_xlabel(r"$z / R_s$")
    ax_nu.set_ylabel(r"relative density $\nu$", color="C1")
    ax_nu.tick_params(axis="y", labelcolor="C1")
    (nu_line,) = ax_nu.plot([], [], color="C1", lw=1.8, label=r"$\nu(z)$")
    ax_c = ax_nu.twinx()
    ax_c.set_ylim(c_lo - c_pad, c_hi + c_pad)
    ax_c.set_ylabel(r"total curvature $c_m + c_p$", color="C0")
    ax_c.tick_params(axis="y", labelcolor="C0")
    (c_line,) = ax_c.plot([], [], color="C0", lw=1.6, label=r"$c_m + c_p$")
    title_nu = ax_nu.set_title("")
    ax_nu.legend(
        [nu_line, c_line], [r"$\nu(z)$", r"$c_m + c_p$"],
        frameon=False, loc="best",
    )
    fig.tight_layout()

    def _shape_err(sol: InclusionSolution) -> float:
        if len(sol.s) < 5 or len(traj[0].s) < 5:
            return float("nan")
        s_w = sol.s - sol.s[0]
        s_s = traj[0].s - traj[0].s[0]
        s_c = np.linspace(0.0, min(float(s_w[-1]), float(s_s[-1])), 200)
        Xw = np.interp(s_c, s_w, sol.X)
        Zw = np.interp(s_c, s_w, sol.Z)
        Xs = np.interp(s_c, s_s, traj[0].X)
        Zs = np.interp(s_c, s_s, traj[0].Z)
        return float(np.max(np.hypot(Xw - Xs, Zw - Zs)))

    def _update(i: int):
        sol = frames[i]
        x, z, c = xs_all[i], zs_all[i], nu_all[i]
        pts = np.array([x, z]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc.set_segments(segs)
        lc.set_array(0.5 * (c[:-1] + c[1:]))
        nu_line.set_data(z_all[i], sol.nu)
        c_line.set_data(z_all[i], csum_all[i])
        corr = float("nan")
        if len(sol.nu) > 2 and np.std(sol.nu) > 0 and np.std(csum_all[i]) > 0:
            corr = float(np.corrcoef(sol.nu, csum_all[i])[0, 1])
        title_sh.set_text(
            rf"stomatocyte_seifert  $\bar{{v}}$={v_seed:.3f}, $c_0$={c0_seed:g}  [shot]"
            + "\n"
            + rf"$p$={sol.p:g}, shape $|\Delta r|_{{\max}}$ vs seed = {_shape_err(sol):.2e}"
        )
        title_nu.set_text(
            rf"$h_m$={sol.hm:g}, $\kappa$={sol.kappa:g}, $p$={sol.p:g}"
            + "\n"
            + rf"corr$(\nu,\,c_m+c_p)$ = {corr:.3f}"
            + f",  $|$res$|$={sol.constraints.get('residual', float('nan')):.2e}"
        )
        return lc, nu_line, c_line, title_sh, title_nu

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=1000 / max(fps, 1), blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    if verbose:
        print(f"  wrote animation {out_path}  ({len(frames)} frames)", flush=True)
    return out_path


def _closed_meridian_xyz_nu(incl: InclusionSolution) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed contour ``(x, z, ν)`` with left/right reflection (upright Z)."""
    x_r = np.asarray(incl.X, dtype=float)
    z_up = -np.asarray(incl.Z, dtype=float)
    nu = np.asarray(incl.nu, dtype=float)
    x = np.concatenate([x_r, -x_r[::-1]])
    z = np.concatenate([z_up, z_up[::-1]])
    c = np.concatenate([nu, nu[::-1]])
    z = z - 0.5 * (z.max() + z.min())
    return x, z, c


def plot_nu_colored_meridian(ax, incl: InclusionSolution, *,
    cmap: str = "cividis", lw: float = 3.5,
    vmin: Optional[float] = None, vmax: Optional[float] = None,
):
    """Draw a closed meridian colored by ``ν``; return the ``LineCollection``."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    x, z, c = _closed_meridian_xyz_nu(incl)
    if len(x) < 2:
        return None
    points = np.column_stack([x, z]).reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    c_seg = 0.5 * (c[:-1] + c[1:])
    vmin = float(np.min(c)) if vmin is None else float(vmin)
    vmax = float(np.max(c)) if vmax is None else float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-15:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(c_seg)
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    pad = 0.08 * max(float(np.ptp(x)), float(np.ptp(z)), 0.5)
    ax.set_xlim(float(x.min()) - pad, float(x.max()) + pad)
    ax.set_ylim(float(z.min()) - pad, float(z.max()) + pad)
    ax.set_aspect("equal")
    return lc


def demo_nu_colored_gallery(*,
    hm: float = INCL_HM_DEFAULT, kappa: float = INCL_KAPPA_DEFAULT,
    p: float = INCL_P_DEFAULT,
    out_dir: Optional[Path] = None, verbose: bool = True,
) -> Path:
    """Color meridians by ``ν`` for seeds + prolate + oblate; write PNG gallery."""
    import matplotlib.pyplot as plt
    from seifert_functions import load_solution_list

    if out_dir is None:
        out_dir = Path("results") / "inclusions_demo" / "nu_colored"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent

    def _load_labeled(label: str, path: Path) -> Optional[MeridianSolution]:
        if not path.exists():
            if verbose:
                print(f"  skip missing {path}", flush=True)
            return None
        if path.name.endswith("_track.json") or path.name == "oblate_growth_track.json":
            track = load_solution_list(path)
            pick = None
            best_aspect = -1.0
            for sol in track:
                if "nested" in str(sol.branch):
                    continue
                if len(sol.s) < 10:
                    continue
                height = abs(float(sol.Z[0] - sol.Z[-1]))
                aspect = float(sol.X.max()) / max(0.5 * height, 1e-6)
                if aspect > best_aspect:
                    best_aspect = aspect
                    pick = sol
            if pick is None:
                if verbose:
                    print(f"  skip empty track {path}", flush=True)
                return None
            return rescale_meridian_to_unit_area(pick, label=label)
        return load_unit_area_seifert_seed(path)

    entries: List[Tuple[str, Path]] = [
        ("pear", root / "seed_shapes" / "pear_seed.json"),
        ("stomatocyte_seifert", root / "seed_shapes" / "stomatocyte_seifert_seed.json"),
        ("stomatocyte_stable", root / "seed_shapes" / "stomatocyte_stable_seed.json"),
        ("stomatocyte_unstable", root / "seed_shapes" / "stomatocyte_unstable_seed.json"),
        ("prolate", root / "cache" / "eta1.8_Td1" / "v0.801534_c02.767168.json"),
        ("oblate", root / "cache" / "eta1.85_C0-1_Td1" / "oblate_growth_track.json"),
    ]

    panels: List[Tuple[str, InclusionSolution]] = []
    for label, path in entries:
        seif_sol = _load_labeled(label, path)
        if seif_sol is None:
            continue
        base = seifert_to_inclusion_solution(seif_sol, p=0.0)
        warm = apply_frozen_nu(base, hm=hm, kappa=kappa, p=p)
        panels.append((label, warm))
        if verbose:
            print(
                f"  {label}: v̄={warm.v:.3f} c₀={warm.c0:.3f}  "
                f"ν∈[{warm.nu.min():.3f},{warm.nu.max():.3f}]  λν={warm.lnu:.3g}",
                flush=True,
            )

    if not panels:
        raise RuntimeError("no shapes available for ν-colored gallery")

    # Shared color scale across the gallery
    vmin = min(float(np.min(w.nu)) for _, w in panels)
    vmax = max(float(np.max(w.nu)) for _, w in panels)

    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.2 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    last_lc = None
    for ax, (label, warm) in zip(axes_flat, panels):
        last_lc = plot_nu_colored_meridian(ax, warm, vmin=vmin, vmax=vmax)
        ax.set_xlabel(r"$x / R_s$")
        ax.set_ylabel(r"$z / R_s$")
        ax.set_title(
            rf"{label}" + "\n"
            + rf"$v̄$={warm.v:.3f}, $c_0$={warm.c0:g}, $\nu\in[{warm.nu.min():.2f},{warm.nu.max():.2f}]$",
            fontsize=10,
        )
        fig_i, ax_i = plt.subplots(figsize=(4.2, 4.6))
        lc_i = plot_nu_colored_meridian(ax_i, warm, vmin=vmin, vmax=vmax)
        ax_i.set_xlabel(r"$x / R_s$")
        ax_i.set_ylabel(r"$z / R_s$")
        ax_i.set_title(
            rf"{label}: $v̄$={warm.v:.3f}, $c_0$={warm.c0:g}"
            + "\n"
            + rf"$h_m$={hm:g}, $\kappa$={kappa:g}, $p$={p:g}"
        )
        if lc_i is not None:
            cb = fig_i.colorbar(lc_i, ax=ax_i, fraction=0.046, pad=0.04)
            cb.set_label(r"relative density $\nu$")
        fig_i.tight_layout()
        fig_i.savefig(out_dir / f"nu_meridian_{label}.png", dpi=150)
        plt.close(fig_i)

    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(
        rf"Inclusion density on frozen Seifert shapes  ($h_m$={hm:g}, $\kappa$={kappa:g}, $p$={p:g})",
        fontsize=12, y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.08, wspace=0.28, hspace=0.35)
    if last_lc is not None:
        cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        cb = fig.colorbar(last_lc, cax=cax)
        cb.set_label(r"relative density $\nu$")
    gallery = out_dir / "nu_meridian_gallery.png"
    fig.savefig(gallery, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"  wrote {gallery} and {n} single-shape PNGs under {out_dir}", flush=True)
    return gallery


def demo_small_p(*,
    seed_path: Optional[Path] = None,
    hm: float = INCL_HM_DEFAULT, kappa: float = INCL_KAPPA_DEFAULT,
    p: float = INCL_P_DEFAULT,
    out_png: Optional[Path] = None, verbose: bool = True,
) -> InclusionSolution:
    """Demo: closed stored Seifert shape + p=0 ODE identity + frozen-shape ν(l)."""
    import matplotlib.pyplot as plt

    seif = load_unit_area_seifert_seed(seed_path)
    base = seifert_to_inclusion_solution(seif, p=0.0)
    rhs = _rhs_identity_metrics(seif)
    warm = apply_frozen_nu(base, hm=hm, kappa=kappa, p=p)

    if out_png is None:
        out_png = Path("results") / "inclusions_demo" / "nu_profile.png"
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    ax = axes[0]
    xs, zs = _seifert_xz(seif)
    ax.plot(xs, zs, "C0-", lw=2.2, label="Seifert seed (closed)")
    xb, zb = _closed_meridian_xz(base)
    ax.plot(xb, zb, "k--", lw=1.2, alpha=0.85, label="inclusions state (p=0)")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x / R_s$")
    ax.set_ylabel(r"$z / R_s$")
    ax.legend(frameon=False, fontsize=9)
    err = rhs.get("rhs_abs_err_max", float("nan"))
    ax.set_title(
        rf"closed pear  $v̄$={seif.v:.3f}, $c_0$={seif.c0:g}"
        + "\n"
        + rf"p=0 RHS vs Seifert max $|f|$ err = {err:.2e}"
    )

    ax = axes[1]
    ax.plot(warm.s, warm.nu, "C1-", lw=1.4, label=r"$\nu(l)$ (frozen shape)")
    ax.axhline(1.0, color="k", lw=1.0, alpha=0.7, label=r"$p=0$ ($\nu\equiv1$)")
    ax.set_xlabel(r"arclength $l / R_s$")
    ax.set_ylabel(r"relative density $\nu$")
    ax.legend(frameon=False)
    ax.set_title(rf"$h_m={hm:g}$, $\kappa={kappa:g}$, $p={p:g}$, $\lambda_\nu$={warm.lnu:.3g}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    if verbose:
        print(f"  wrote {out_png}", flush=True)
        print("  Seifert identity:", rhs, flush=True)
        print(
            f"  closed poles X=({base.X[0]:.3e},{base.X[-1]:.3e})  "
            f"Zspan={base.Z[0]-base.Z[-1]:.4f}  "
            f"ν∈[{warm.nu.min():.3f},{warm.nu.max():.3f}]",
            flush=True,
        )
    return warm


if __name__ == "__main__":
    print("=== Božič 2006 inclusions: sphere p=0 vs Seifert ODE ===", flush=True)
    regress_p0_sphere(verbose=True)
    print("=== p=0 vs stored Seifert pear seed ===", flush=True)
    compare_p0_to_seifert(verbose=True)
    print("=== small-p demo (pear seed) ===", flush=True)
    demo_small_p(verbose=True)
    print("=== ν-colored meridian gallery ===", flush=True)
    demo_nu_colored_gallery(verbose=True)
    print("=== stomatocyte_seifert: ν vs curvature ===", flush=True)
    demo_stomatocyte_nu_vs_curvature(verbose=True)

