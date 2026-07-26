"""External two-sphere (pear thin-neck) geometry."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from seifert_functions import area_radius, reduced_volume_from_AV
from shapes.base import VesicleShape

V_EQUAL_SPHERES = float(1.0 / np.sqrt(2.0))


def two_sphere_energy(C0: float, R1: float, R2: float) -> float:
    """Two-sphere Helfrich energy: ``2π[(2−C₀R₁)²+(2−C₀R₂)²]``."""
    C0 = float(C0)
    return float(2.0 * np.pi * ((2.0 - C0 * float(R1)) ** 2 + (2.0 - C0 * float(R2)) ** 2))


# Alias used by D-boundary / pear scans.
approx_pear_energy_two_sphere = two_sphere_energy


def two_sphere_alpha_from_v(v: float) -> Optional[float]:
    """``α = R₁/R₀`` from ``α³ + (1−α²)^{3/2} = v``."""
    v = float(v)
    v_lo = V_EQUAL_SPHERES
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
    alpha = two_sphere_alpha_from_v(v)
    if alpha is None:
        return None
    r1 = float(alpha * R0)
    r2 = float(np.sqrt(max(0.0, R0 * R0 - r1 * r1)))
    return r1, r2


def two_sphere_radii_from_AV(A: float, V: float) -> Optional[Tuple[float, float]]:
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


def orient_two_sphere_radii(
    R_a: float,
    R_b: float,
    *,
    U0: Optional[float] = None,
    U1: Optional[float] = None,
    north_larger: Optional[bool] = None,
) -> Tuple[float, float]:
    """Order ``(R_north, R_south)`` to match pear pole orientation."""
    ra, rb = float(R_a), float(R_b)
    r_big, r_small = (ra, rb) if ra >= rb else (rb, ra)
    if north_larger is None:
        if U0 is not None and U1 is not None and abs(U0) > 1e-14 and abs(U1) > 1e-14:
            north_larger = abs(float(U0)) < abs(float(U1))
        else:
            north_larger = True
    if north_larger:
        return r_big, r_small
    return r_small, r_big


def two_sphere_meridian(R1: float, R2: float, *, n_per: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """North lobe ``R1`` then south lobe ``R2``, neck at ``z=0``."""
    R1, R2 = float(R1), float(R2)
    th1 = np.linspace(0.0, np.pi, n_per)
    x1 = R1 * np.sin(th1)
    z1 = R1 * np.cos(th1) + R1
    th2 = np.linspace(0.0, np.pi, n_per)
    x2 = R2 * np.sin(th2)
    z2 = R2 * np.cos(th2) - R2
    x = np.concatenate([x1, x2[1:]])
    z = np.concatenate([z1, z2[1:]])
    z = z - 0.5 * (z.max() + z.min())
    return x, z


class TwoSphere(VesicleShape):
    """External two-sphere (thin-neck pear limit)."""

    def __init__(
        self,
        A: float,
        V: float,
        C0: float,
        *,
        north_larger: Optional[bool] = None,
        tip_U0: Optional[float] = None,
        tip_U1: Optional[float] = None,
    ) -> None:
        super().__init__(A, V, C0)
        radii = two_sphere_radii_from_AV(A, V)
        if radii is None:
            raise ValueError(f"two-sphere radii failed at A={A:.4f} V={V:.4f}")
        self.R1, self.R2 = orient_two_sphere_radii(
            radii[0], radii[1],
            U0=tip_U0, U1=tip_U1, north_larger=north_larger,
        )

    def bending_energy(self, kappa: float = 1.0) -> float:
        return float(kappa) * two_sphere_energy(self.C0, self.R1, self.R2)

    def meridian(self, *, n_per: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        return two_sphere_meridian(self.R1, self.R2, n_per=n_per)

    def branch_name(self) -> str:
        return "two_sphere"

    def pole_curvatures(self) -> Tuple[float, float]:
        return 1.0 / self.R1, 1.0 / self.R2

    def arc_length(self) -> float:
        return float(np.pi * (self.R1 + self.R2))

    def north_arc_length(self) -> float:
        return float(np.pi * self.R1)

    def geometry_constraints(self) -> Dict[str, Any]:
        return {
            "R1": float(self.R1),
            "R2": float(self.R2),
            "north_larger": bool(self.R1 >= self.R2),
        }
