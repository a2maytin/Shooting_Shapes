"""Nested sphere-in-sphere (stomatocyte) geometry."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from seifert_functions import area_radius, reduced_volume_from_AV
from shapes.base import VesicleShape
from shapes.two_sphere import two_sphere_energy


def L_sto_reduced_volume(c0: float) -> float:
    """Analytic stomatocyte limit ``L_sto`` in the ``(c₀, v̄)`` plane.

    ``v̄ = -2 c₀^{-3} + (1 - 2 c₀^{-2}) √(1 + c₀^{-2})``.
    """
    c0 = float(c0)
    if abs(c0) < 1e-12:
        return float("nan")
    inv2 = 1.0 / (c0 * c0)
    return float(-2.0 / (c0 ** 3) + (1.0 - 2.0 * inv2) * np.sqrt(1.0 + inv2))


def nested_sphere_alpha_from_v(v: float) -> Optional[float]:
    """``α = R_outer/R₀`` for ``|α³ − (1−α²)^{3/2}| = v``."""
    v = float(v)
    if not (0.0 < v <= 1.0 + 1e-9):
        return None
    if abs(v - 1.0) < 1e-12:
        return 1.0

    def _f(alpha: float) -> float:
        if alpha <= 0.0 or alpha >= 1.0:
            return 1e3
        return abs(alpha ** 3 - (1.0 - alpha * alpha) ** 1.5) - v

    a_eq = float(2.0 ** -0.5)
    for lo, hi in ((a_eq + 1e-8, 1.0 - 1e-8), (1e-8, a_eq - 1e-8)):
        try:
            flo, fhi = _f(lo), _f(hi)
            if not (np.isfinite(flo) and np.isfinite(fhi)):
                continue
            if flo == 0.0:
                return float(lo)
            if flo * fhi < 0.0:
                return float(brentq(_f, lo, hi))
        except ValueError:
            continue
    return None


def nested_sphere_radii_from_v(v: float, R0: float = 1.0) -> Optional[Tuple[float, float]]:
    alpha = nested_sphere_alpha_from_v(v)
    if alpha is None:
        return None
    R0 = float(R0)
    r_out = float(alpha * R0)
    r_in = float(np.sqrt(max(0.0, R0 * R0 - r_out * r_out)))
    if r_out < r_in:
        r_out, r_in = r_in, r_out
    return r_out, r_in


def nested_sphere_radii_from_AV(A: float, V: float) -> Optional[Tuple[float, float]]:
    A = float(A)
    V = float(V)
    if A <= 0.0 or V <= 0.0:
        return None
    return nested_sphere_radii_from_v(reduced_volume_from_AV(A, V), area_radius(A))


def nested_sphere_signed_radii_from_AV(
    A: float, V: float,
) -> Optional[Tuple[float, float]]:
    """``(R_outer > 0, R_inner < 0)`` for energy with inverted lobe."""
    radii = nested_sphere_radii_from_AV(A, V)
    if radii is None:
        return None
    r_out, r_in = float(radii[0]), float(radii[1])
    if abs(r_out) < abs(r_in):
        r_out, r_in = r_in, r_out
    return abs(r_out), -abs(r_in)


def nested_sphere_meridian(
    R_out: float,
    R_in: float,
    *,
    n_per: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Outer + invaginated inner circles sharing a neck at ``z=0``."""
    Ro = abs(float(R_out))
    Ri = abs(float(R_in))
    th_o = np.linspace(np.pi, 0.0, n_per)
    x_o = Ro * np.sin(th_o)
    z_o = -Ro + Ro * np.cos(th_o)
    th_i = np.linspace(0.0, np.pi, n_per)
    x_i = Ri * np.sin(th_i)
    z_i = -Ri + Ri * np.cos(th_i)
    x = np.concatenate([x_o, x_i[1:]])
    z = np.concatenate([z_o, z_i[1:]])
    z = z - 0.5 * (z.max() + z.min())
    return x, z


class NestedSphere(VesicleShape):
    """Nested sphere-in-sphere (stomatocyte thin-neck limit)."""

    def __init__(self, A: float, V: float, C0: float) -> None:
        super().__init__(A, V, C0)
        signed = nested_sphere_signed_radii_from_AV(A, V)
        if signed is None:
            raise ValueError(f"nested-sphere radii failed at A={A:.4f} V={V:.4f}")
        self.R_outer = float(signed[0])
        self.R_inner = float(signed[1])  # signed negative

    def bending_energy(self, kappa: float = 1.0) -> float:
        return float(kappa) * two_sphere_energy(self.C0, self.R_outer, self.R_inner)

    def pressure_from_dEdV(
        self,
        *,
        dV_frac: float = 1e-4,
        kappa: float = 1.0,
    ) -> float:
        """Growth pressure ``P ≈ −ΔE/ΔV`` at fixed area."""
        dV_frac = abs(float(dV_frac))
        if dV_frac < 1e-10:
            dV_frac = 1e-4
        V2 = self.V * (1.0 - dV_frac)
        if V2 <= 0.0:
            return float("nan")
        E1 = self.bending_energy(kappa=kappa)
        try:
            E2 = NestedSphere(self.A, V2, self.C0).bending_energy(kappa=kappa)
        except ValueError:
            return float("nan")
        dV = V2 - self.V
        if abs(dV) < 1e-18:
            return float("nan")
        return float(-(E2 - E1) / dV)

    def meridian(self, *, n_per: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        return nested_sphere_meridian(self.R_outer, self.R_inner, n_per=n_per)

    def branch_name(self) -> str:
        return "nested_sphere"

    def pole_curvatures(self) -> Tuple[float, float]:
        Ri = abs(self.R_inner)
        Ro = self.R_outer
        U0 = -1.0 / Ri if Ri > 1e-14 else 0.0
        U1 = 1.0 / Ro if Ro > 1e-14 else 0.0
        return U0, U1

    def arc_length(self) -> float:
        return float(np.pi * (self.R_outer + abs(self.R_inner)))

    def north_arc_length(self) -> float:
        return float(np.pi * self.R_outer)

    def geometry_constraints(self) -> Dict[str, Any]:
        return {
            "R_outer": float(self.R_outer),
            "R_inner": float(self.R_inner),
            "R1": float(self.R_outer),
            "R2": float(self.R_inner),
            "v_L_sto": float(L_sto_reduced_volume(self.c0)),
            "E_nested": float(self.bending_energy()),
        }


def nested_sphere_energy(A: float, V: float, C0: float, *, kappa: float = 1.0) -> float:
    try:
        return NestedSphere(A, V, C0).bending_energy(kappa=kappa)
    except ValueError:
        return float("nan")


def nested_sphere_pressure_from_dEdV(
    A: float,
    V: float,
    C0: float,
    *,
    dV_frac: float = 1e-4,
    kappa: float = 1.0,
) -> float:
    try:
        return NestedSphere(A, V, C0).pressure_from_dEdV(dV_frac=dV_frac, kappa=kappa)
    except ValueError:
        return float("nan")
