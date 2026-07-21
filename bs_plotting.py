"""Božič–Svetina trajectory plotting: growth panels and full τ=0→1 figures/movies.

Thin wrappers around plotting helpers defined in ``bs_functions`` (two-sphere
finish assembly lives there too).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from bs_functions import (
    ETA,
    configure,
    ensure_two_sphere_leg,
    main as two_sphere_finish_main,
    movie_full_trajectory,
    plot_full_trajectory,
    plot_two_sphere_detail,
    save_growth_plots,
)


def run_two_sphere_finish(
    *,
    eta: float,
    recompute: bool = True,
    no_movie: bool = False,
    movie_frames: int = 180,
) -> None:
    """Assemble two-sphere finish + full trajectory plots for ``eta``."""
    argv: List[str] = [f"--eta={eta}", f"--movie-frames={movie_frames}"]
    if recompute:
        argv.append("--recompute")
    if no_movie:
        argv.append("--no-movie")
    two_sphere_finish_main(argv)


__all__ = [
    "configure",
    "ensure_two_sphere_leg",
    "movie_full_trajectory",
    "plot_full_trajectory",
    "plot_two_sphere_detail",
    "run_two_sphere_finish",
    "save_growth_plots",
]
