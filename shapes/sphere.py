"""Sphere geometry helpers."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from shapes.base import VesicleShape


def sphere_meridian(R: float, *, n: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0.0, np.pi, n)
    x = float(R) * np.sin(th)
    z = float(R) * np.cos(th)
    return x, z


class Sphere(VesicleShape):
    """Closed sphere at area ``A`` (volume ignored for meridian; v̄ from ``(A,V)``)."""

    def bending_energy(self, kappa: float = 1.0) -> float:
        # Spontaneous-curvature sphere: 8πκ(1 − c₀/2)² at reduced c₀.
        c0 = self.c0
        return float(kappa) * 8.0 * np.pi * (1.0 - 0.5 * c0) ** 2

    def meridian(self, *, n_per: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        return sphere_meridian(self.R0, n=n_per)

    def branch_name(self) -> str:
        return "sphere"

    def pole_curvatures(self) -> Tuple[float, float]:
        R = self.R0
        if R <= 0.0:
            return 0.0, 0.0
        u = 1.0 / R
        return u, u

    def geometry_constraints(self) -> Dict[str, Any]:
        return {"R": float(self.R0)}
