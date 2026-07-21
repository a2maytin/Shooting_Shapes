"""Seifert shape plotting: meridian movies for Appendix-B / growth tracks."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from seifert_functions import MeridianSolution

plt.rcParams["figure.dpi"] = 140


def _meridian_profile(sol: MeridianSolution) -> Tuple[np.ndarray, np.ndarray]:
    """Meridian polyline in plot coordinates (X, upright Z)."""
    if len(sol.s) >= 2:
        return np.asarray(sol.X, dtype=float), np.asarray(-sol.Z, dtype=float)
    R0 = sol.constraints.get("R0")
    if R0 is None:
        C0_dim = sol.constraints.get("C0_dimensional")
        if C0_dim and sol.c0:
            R0 = float(sol.c0) / float(C0_dim)
    if R0 is not None and float(R0) > 0.0:
        theta = np.linspace(0.0, np.pi, 64)
        R0 = float(R0)
        return R0 * np.sin(theta), R0 * np.cos(theta)
    return np.array([]), np.array([])


def animate_meridian_trajectory(
    traj: Sequence[MeridianSolution],
    save_path: Path,
    *,
    fps: int = 8,
    dpi: int = 120,
    max_frames: Optional[int] = None,
    show: bool = False,
) -> Path:
    """MP4 animation of meridian profiles along a growth or v-trajectory."""
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    if not traj:
        raise ValueError("empty trajectory")
    traj_list: List[MeridianSolution] = [s for s in traj if s is not None]
    if max_frames is not None and len(traj_list) > int(max_frames) > 0:
        idx = np.linspace(0, len(traj_list) - 1, int(max_frames), dtype=int)
        traj_list = [traj_list[i] for i in idx]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    profiles = [_meridian_profile(s) for s in traj_list]
    nonempty = [(x, z) for x, z in profiles if x.size >= 2]
    if not nonempty:
        raise ValueError("trajectory has no plottable meridian profiles")
    x_max = max(float(x.max()) for x, _ in nonempty)
    z_max = max(float(z.max()) for _, z in nonempty)
    z_min = min(float(z.min()) for _, z in nonempty)

    fig, ax = plt.subplots(figsize=(4.5, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, x_max * 1.12)
    ax.set_ylim(z_min * 1.05 if z_min < 0 else -0.05, z_max * 1.15)
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$Z$")
    (line,) = ax.plot([], [], lw=2.2, color="C0")
    title = ax.set_title("")

    def init():
        line.set_data([], [])
        return line,

    def update(frame: int):
        sol = traj_list[frame]
        x, z = profiles[frame]
        if x.size < 2:
            x, z = nonempty[0]
        line.set_data(x, z)
        tau = sol.constraints.get("t_growth")
        T_d = float(sol.constraints.get("T_d", 1.0))
        tau_str = f"τ={float(tau) / T_d:.3f}  " if tau is not None and T_d > 0.0 else ""
        phase = sol.constraints.get("phase")
        phase_str = f"  {phase}" if phase else ""
        title.set_text(
            rf"{tau_str}$c_0$={sol.c0:.3f}  $\bar v$={sol.v:.3f}{phase_str}"
        )
        return line,

    anim = FuncAnimation(fig, update, init_func=init, frames=len(traj_list), blit=True)
    writer = FFMpegWriter(fps=fps)
    anim.save(str(save_path), writer=writer, dpi=dpi)
    print(f"Saved {save_path}  ({len(traj_list)} frames @ {fps} fps)")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path
