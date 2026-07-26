"""Near-sphere (prolate / oblate) energy approximation."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from shapes.base import VesicleShape
from shapes.sphere import sphere_meridian


def near_sphere_energy(c0: float, v: float) -> float:
    """Near-sphere prolate/oblate energy: ``8π[(1−c₀/2)²−(6−c₀)(v̄−1)/3]``."""
    c0 = float(c0)
    v = float(v)
    return float(8.0 * np.pi * ((1.0 - 0.5 * c0) ** 2 - (6.0 - c0) * (v - 1.0) / 3.0))


# Historical aliases used by D-boundary scans.
approx_prolate_energy = near_sphere_energy
approx_oblate_energy = near_sphere_energy


class NearSphere(VesicleShape):
    """Near-sphere branch energy at ``(c₀, v̄)``; meridian is the enclosing sphere."""

    def bending_energy(self, kappa: float = 1.0) -> float:
        return float(kappa) * near_sphere_energy(self.c0, self.v)

    def meridian(self, *, n_per: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        return sphere_meridian(self.R0, n=n_per)

    def branch_name(self) -> str:
        return "near_sphere"
