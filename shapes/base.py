"""Abstract vesicle shape interface for analytic geometries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from seifert_functions import (
    MeridianSolution,
    area_radius,
    c0_reduced,
    reduced_volume_from_AV,
)


class VesicleShape(ABC):
    """Analytic vesicle geometry at fixed physical ``(A, V)`` and dimensional ``C₀``."""

    A: float
    V: float
    C0: float

    def __init__(self, A: float, V: float, C0: float) -> None:
        self.A = float(A)
        self.V = float(V)
        self.C0 = float(C0)

    @property
    def R0(self) -> float:
        return float(area_radius(self.A))

    @property
    def v(self) -> float:
        return float(reduced_volume_from_AV(self.A, self.V))

    @property
    def c0(self) -> float:
        """Reduced spontaneous curvature ``c₀ = C₀ R₀``."""
        return float(c0_reduced(self.C0, self.A))

    @abstractmethod
    def bending_energy(self, kappa: float = 1.0) -> float:
        """Helfrich bending energy (κ-scaled)."""

    @abstractmethod
    def meridian(self, *, n_per: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        """Side-view meridian ``(X, Z)`` in the right half-plane."""

    @abstractmethod
    def branch_name(self) -> str:
        """Label stored on ``MeridianSolution.branch`` / phase."""

    def to_meridian_solution(
        self,
        *,
        t_growth: float,
        P_bar: float,
        L_p: float,
        T_d: float = 1.0,
        eta: float = 1.0,
        kappa: float = 1.0,
        message: str = "",
        extra_constraints: Optional[Dict[str, Any]] = None,
    ) -> MeridianSolution:
        """Serialize this analytic shape as a lightweight ``MeridianSolution``."""
        X, Z = self.meridian()
        n = X.size
        y = np.zeros((7, n), dtype=float)
        y[0] = X
        y[1] = Z
        U0, U1 = self.pole_curvatures()
        y[3, 0] = U0
        y[3, -1] = U1
        y[5, -1] = self.A
        y[6, -1] = self.V
        s = np.linspace(0.0, float(self.arc_length()), n)
        constraints: Dict[str, Any] = {
            "A_phys": float(self.A),
            "V_phys": float(self.V),
            "t_growth": float(t_growth),
            "C0_dimensional": float(self.C0),
            "T_d": float(T_d),
            "L_p": float(L_p),
            "eta": float(eta),
            "kappa": float(kappa),
            "phase": self.branch_name(),
            "DeltaP": float(P_bar),
            "E_bend": float(self.bending_energy(kappa=kappa)),
        }
        constraints.update(self.geometry_constraints())
        if extra_constraints:
            constraints.update(extra_constraints)
        return MeridianSolution(
            v=float(self.v),
            c0=float(self.c0),
            U0=float(U0),
            U1=float(U1),
            sigma_bar=float("nan"),
            P_bar=float(P_bar),
            s=s,
            y=y,
            branch=self.branch_name(),
            success=True,
            message=message or self.branch_name(),
            constraints=constraints,
            S1=float(self.north_arc_length()),
            S_bar=float(self.north_arc_length()),
        )

    def pole_curvatures(self) -> Tuple[float, float]:
        """``(U₀, U₁)`` pole curvature guesses for frame metadata."""
        return 0.0, 0.0

    def arc_length(self) -> float:
        return float(np.pi * self.R0)

    def north_arc_length(self) -> float:
        return 0.5 * self.arc_length()

    def geometry_constraints(self) -> Dict[str, Any]:
        return {}
