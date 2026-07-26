"""Božič–Svetina trajectory plotting: growth panels and full τ=0→1 figures/movies.

Plotting implementations live here (not in ``bs_functions``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from seifert_functions import MeridianSolution, area_radius

import bs_functions as _bs
from shapes import (
    nested_sphere_meridian,
    nested_sphere_radii_from_AV,
    sphere_meridian,
    two_sphere_meridian,
    two_sphere_radii_from_AV,
)

# Re-bind names used by moved code to ``bs_functions`` symbols.
C0 = _bs.C0
ETA = _bs.ETA
T_D = _bs.T_D
CACHE = _bs.CACHE
BS_C0_START = _bs.BS_C0_START
BS_C0_START_NEG = _bs.BS_C0_START_NEG
BS_C0_END = _bs.BS_C0_END
BS_V_TWO_SPHERE = _bs.BS_V_TWO_SPHERE
BS_TAU_D_REF = _bs.BS_TAU_D_REF
bs_point_a_ref = _bs.bs_point_a_ref
c0_critical_sphere = _bs.c0_critical_sphere
tau_B_from_eta = _bs.tau_B_from_eta
load_seifert_context = _bs.load_seifert_context
approx_D_boundary_curve = _bs.approx_D_boundary_curve
L_sto_reduced_volume = _bs.L_sto_reduced_volume
_seifert_tau = _bs._seifert_tau
TrajFrame = _bs.TrajFrame
configure = _bs.configure
ensure_two_sphere_leg = _bs.ensure_two_sphere_leg


def plot_seed_shape(
    sol: MeridianSolution,
    out_path: Path,
    *,
    title: Optional[str] = None,
    color: str = "#8b3a3a",
) -> Path:
    """Save a side-view meridian plot next to a seed JSON under ``seed_shapes/``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(sol.X if hasattr(sol, "X") else sol.y[0], dtype=float)
    Z = np.asarray(sol.Z if hasattr(sol, "Z") else sol.y[1], dtype=float)
    if X.size < 2:
        raise ValueError("seed solution has no meridian to plot")
    Zc = Z - 0.5 * (float(Z.max()) + float(Z.min()))
    fig, ax = plt.subplots(figsize=(4.5, 5.0))
    ax.plot(X, Zc, "-", color=color, lw=2.0)
    ax.plot(-X, Zc, "-", color=color, lw=2.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if title is None:
        title = (
            f"{out_path.stem}\n"
            f"U0={sol.U0:.3f} U1={sol.U1:.3f}  v̄={sol.v:.4f} c₀={sol.c0:.3f}"
        )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path




def save_growth_plots(
    full: Sequence[MeridianSolution],
    landmarks: BSLandmarks,
    D: MeridianSolution,
    *,
    eta: float,
    T_d: float = 1.0,
    out_dir: Path,
) -> Path:
    """Area/volume vs τ and path in (c₀, v̄); save ``growth_path.png`` under *out_dir*."""
    ok = [s for s in full if s.success]
    A0, V0 = landmarks.A_phys_ref, landmarks.V_phys_ref
    ts = np.array([s.constraints["t_growth"] for s in ok]) / T_d
    As = np.array([s.constraints["A_phys"] / A0 for s in ok])
    Vs = np.array([s.constraints["V_phys"] / V0 for s in ok])
    vs = np.array([s.v for s in ok])
    cs = np.array([s.c0 for s in ok])
    phases = [s.constraints.get("phase") for s in ok]
    C0_CR = c0_critical_sphere(eta)
    pear_D = ok[0]
    for s in ok:
        if abs(float(s.constraints.get("t_growth", 0.0)) - float(D.constraints["t_growth"])) < 1e-8:
            pear_D = s
            break
    pear_E = min(ok, key=lambda s: abs(s.constraints["t_growth"] / T_d - 1.0))

    landmark_lines = (
        ("B", landmarks.B), ("C", landmarks.C), ("D", pear_D), ("E", pear_E),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(ts, As, "o-", ms=3, label=r"$A/A_0$")
    ax.plot(ts, Vs, "s-", ms=3, label=r"$V/V_0$")
    ax.axhline(2.0, color="k", ls="--", lw=0.8, alpha=0.4)
    ax.axhline(np.sqrt(2.0), color="k", ls=":", lw=0.8, alpha=0.4)
    ymax = max(As.max(), Vs.max()) * 1.05
    for name, sol in landmark_lines:
        if sol is None:
            continue
        t = sol.constraints["t_growth"] / T_d
        ax.axvline(t, color="C3", ls="--", lw=0.7, alpha=0.7)
        ax.text(t, ymax, name, ha="center", va="bottom", fontsize=8, color="C3")
    ax.set_xlabel(r"$\tau = t/T_d$")
    ax.set_ylabel("normalized")
    ax.set_title(rf"Growth curves ($\eta={eta:g}$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for phase, marker in (("sphere", "o"), ("shape", "s")):
        m = np.array([p == phase for p in phases])
        if m.any():
            ax.plot(cs[m], vs[m], marker + "-", ms=4, lw=1, label=phase)
    for name, sol in landmark_lines:
        if sol is None:
            continue
        ax.plot(sol.c0, sol.v, "*", ms=12)
        ax.annotate(name, (sol.c0, sol.v), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.axvline(C0_CR, color="C3", ls=":", lw=0.8, alpha=0.6, label=rf"$c_{{0,\mathrm{{cr}}}}={C0_CR:.3f}$")
    ax.axhline(BS_V_TWO_SPHERE, color="0.5", ls=":", lw=0.8, alpha=0.6)
    ax.plot([BS_C0_START, BS_C0_END], [1.0, BS_V_TWO_SPHERE], "k--", lw=0.6, alpha=0.3)
    ax.set_xlabel(r"$c_0 = C_0 R$")
    ax.set_ylabel(r"$\bar v$")
    ax.set_title(r"Path in $(c_0, \bar v)$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.65, 1.02)

    fig.tight_layout()
    out_path = Path(out_dir) / "growth_path.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path




def _match_sol_by_tau(
    sols: Sequence[MeridianSolution],
    tau: float,
    *,
    T_d: float = T_D,
    tol: float = 5e-5,
) -> Optional[MeridianSolution]:
    best: Optional[MeridianSolution] = None
    best_dt = float("inf")
    for sol in sols:
        dt = abs(_seifert_tau(sol, T_d=T_d) - float(tau))
        if dt < best_dt:
            best_dt = dt
            best = sol
    return best if best_dt <= tol else None


def attach_seifert_sols(
    frames: Sequence[TrajFrame],
    cache: Optional[Path] = None,
    *,
    T_d: float = T_D,
) -> None:
    """Attach ``MeridianSolution`` profiles for movie/plot-only reload."""
    try:
        ctx = load_seifert_context(cache if cache is not None else CACHE, T_d=T_d)
    except RuntimeError:
        return
    sols = ctx["seifert_tr"]
    for fr in frames:
        if fr.phase not in ("prolate", "pear"):
            continue
        sol = _match_sol_by_tau(sols, fr.tau, T_d=T_d)
        if sol is not None:
            fr.sol = sol


# ---------------------------------------------------------------------------
# Two-sphere continuation
# ---------------------------------------------------------------------------



PHASE_COLORS = {
    "sphere": "#2e7d32",
    "prolate": "#1f4e79",
    "pear": "#8b3a3a",
    "two_sphere": "#c45c26",
    "oblate": "#1f4e79",
    "stomatocyte": "#8b3a3a",
    "nested_sphere": "#c45c26",
}


def _phase_segments(full: Sequence[TrajFrame]) -> List[Tuple[str, List[TrajFrame]]]:
    if not full:
        return []
    segs: List[Tuple[str, List[TrajFrame]]] = []
    cur_phase = full[0].phase
    cur: List[TrajFrame] = [full[0]]
    for fr in full[1:]:
        if fr.phase != cur_phase:
            segs.append((cur_phase, cur))
            cur = [cur[-1], fr]
            cur_phase = fr.phase
        else:
            cur.append(fr)
    segs.append((cur_phase, cur))
    return segs


def plot_full_trajectory(
    full: Sequence[TrajFrame],
    out_dir: Path,
    *,
    eta: Optional[float] = None,
    C0_dim: float = C0,
    tau_D: Optional[float] = None,
    tau_fail: Optional[float] = None,
    stem: str = "full_trajectory",
) -> Path:
    """``(c₀,v̄)``, pressure, and doubling for the assembled growth track."""
    if eta is None:
        eta = ETA
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neg = float(C0_dim) < 0.0
    c0_start = BS_C0_START_NEG if neg else BS_C0_START
    ref = bs_point_a_ref(C0_dim, c0_start=c0_start)
    A0, V0 = float(ref["A"]), float(ref["V"])
    c0_cr = float(c0_critical_sphere(eta, side="-" if neg else "+"))
    tau_B = float(tau_B_from_eta(eta, c0_start=c0_start))

    if tau_D is None:
        if neg:
            nested = [f for f in full if f.phase == "nested_sphere"]
            sto = [f for f in full if f.phase == "stomatocyte"]
            obl = [f for f in full if f.phase == "oblate"]
            if nested:
                tau_D = float(nested[0].tau)
            elif sto:
                tau_D = float(sto[0].tau)
            elif obl:
                tau_D = float(obl[-1].tau)
            else:
                tau_D = float(full[-1].tau)
        else:
            try:
                tau_D = float(load_seifert_context()["tau_D"])
            except RuntimeError:
                tau_D = float(BS_TAU_D_REF)
    if tau_fail is None:
        two_ph = [f for f in full if f.phase == "two_sphere"]
        pear_ph = [f for f in full if f.phase == "pear"]
        if two_ph and pear_ph:
            tau_fail = float(pear_ph[-1].tau)
        elif two_ph:
            tau_fail = float(two_ph[0].tau)
        else:
            tau_fail = float(full[-1].tau)

    ts = np.array([fr.tau for fr in full])
    As = np.array([fr.A / A0 for fr in full])
    Vs = np.array([fr.V / V0 for fr in full])
    segs = _phase_segments(full)
    # "fail" = early Seifert handoff to two-sphere; omit when the Seifert leg
    # itself runs to the trajectory endpoint (no thin-neck cutoff).
    early_fail = (
        (not neg)
        and float(tau_fail) < float(full[-1].tau) - 1e-3
        and float(tau_fail) < 1.0 - 1e-3
    )
    d_label = "D" if not neg else r"$D_{\mathrm{sto}}$"
    marks = [
        ("A", full[0]),
        ("B", min(full, key=lambda f: abs(f.tau - tau_B))),
        (d_label, min(full, key=lambda f: abs(f.tau - tau_D))),
    ]
    if early_fail:
        marks.append(("fail", min(full, key=lambda f: abs(f.tau - tau_fail))))
    end_label = r"$L_{\mathrm{sto}}$" if neg else "E"
    marks.append((end_label, full[-1]))

    tau_hi = max(1.0, float(full[-1].tau) * 1.02)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))

    def _phase_lines(ax, xattr: str, yattr: str) -> None:
        for phase, seg in segs:
            ax.plot(
                [getattr(f, xattr) for f in seg],
                [getattr(f, yattr) for f in seg],
                "-", color=PHASE_COLORS.get(phase, "k"), lw=2.0,
                label=phase.replace("_", "-"),
            )

    tau_guides = [tau_B, tau_D] + ([tau_fail] if early_fail else [])

    ax = axes[0, 0]
    _phase_lines(ax, "tau", "c0")
    ax.axhline(c0_cr, color="0.45", ls=":", lw=0.9, alpha=0.7)
    if not neg:
        ax.axhline(BS_C0_END, color="0.45", ls="--", lw=0.8, alpha=0.5)
    for tmark in tau_guides:
        ax.axvline(tmark, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$c_0$"); ax.set_title(r"$c_0(\tau)$")
    ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3); ax.set_xlim(0.0, tau_hi)

    ax = axes[0, 1]
    _phase_lines(ax, "tau", "v")
    ax.axhline(BS_V_TWO_SPHERE, color="0.45", ls=":", lw=0.9, alpha=0.7,
               label=rf"$2^{{-1/2}}$")
    for tmark in tau_guides:
        ax.axvline(tmark, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$v$"); ax.set_title(r"$v(\tau)$")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0.0, tau_hi)
    v_min = min(float(fr.v) for fr in full)
    ax.set_ylim(min(0.65, v_min - 0.03), 1.02)

    ax = axes[0, 2]
    _phase_lines(ax, "tau", "P")
    for tmark in tau_guides:
        ax.axvline(tmark, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(0.0, color="0.4", ls="--", lw=0.7, alpha=0.5)
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel(r"$P$"); ax.set_title(r"Pressure $P(\tau)$")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0.0, tau_hi)

    ax = axes[1, 0]
    _phase_lines(ax, "v", "c0")
    if neg:
        ax.plot(
            [1.0, BS_V_TWO_SPHERE], [BS_C0_START_NEG, -BS_C0_END],
            "k--", lw=0.7, alpha=0.35, label="sphere→E line",
        )
        c0_lo = min(float(fr.c0) for fr in full)
        c0_hi = float(BS_C0_START_NEG)
        # Span past the trajectory tip so the guideline stays visible at L_sto.
        c0_end = min(c0_lo - 0.4, -4.0, float(c0_cr) * 1.5)
        c0_grid = np.linspace(c0_hi, c0_end, 121)
        c_mid, v_mid = approx_D_boundary_curve(c0_grid, C0=C0_dim, delta_target=0.0)
        bound_label = r"approx $E_{\mathrm{sto}}=E_{\mathrm{oblate}}$"
        # Analytic L_sto curve.
        c_L = np.linspace(min(c0_hi, -0.5), c0_end, 161)
        v_L = np.array([L_sto_reduced_volume(c) for c in c_L], dtype=float)
        ok_L = np.isfinite(v_L) & (v_L > 0.0) & (v_L < 1.0)
        if np.any(ok_L):
            ax.plot(
                v_L[ok_L], c_L[ok_L], "-", color="#6b3fa0", lw=1.6, alpha=0.85,
                label=r"$L_{\mathrm{sto}}$", zorder=2,
            )
    else:
        ax.plot([1.0, BS_V_TWO_SPHERE], [BS_C0_START, BS_C0_END],
                "k--", lw=0.7, alpha=0.35, label="sphere→E line")
        c0_grid = np.linspace(BS_C0_START, BS_C0_END, 81)
        c_mid, v_mid = approx_D_boundary_curve(c0_grid, C0=C0_dim, delta_target=0.0)
        bound_label = r"approx $E_{\mathrm{pear}}=E_{\mathrm{prolate}}$"
    if c_mid.size:
        ax.plot(
            v_mid, c_mid, "-", color="#b8860b", lw=1.8, alpha=0.9,
            label=bound_label,
            zorder=2,
        )
    ax.axhline(c0_cr, color="0.45", ls=":", lw=0.9, alpha=0.7)
    ax.axvline(BS_V_TWO_SPHERE, color="0.45", ls=":", lw=0.9, alpha=0.5)
    for name, fr in marks:
        ax.plot(fr.v, fr.c0, "*", ms=11, color="k", zorder=5)
        ax.annotate(name, (fr.v, fr.c0), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel(r"$v$"); ax.set_ylabel(r"$c_0$")
    ax.set_title(r"Path in $(v,c_0)$")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    v_min = min(float(fr.v) for fr in full)
    ax.set_xlim(min(0.65, v_min - 0.03), 1.02)

    ax = axes[1, 1]
    ax.plot(ts, As, "-", color="#1f4e79", lw=2.0, label=r"$A/A_0$")
    ax.plot(ts, Vs, "--", color="#1f4e79", lw=1.5, alpha=0.75, label=r"$V/V_0$")
    ax.axhline(2.0, color="k", ls="--", lw=0.9, alpha=0.45)
    ax.axhline(np.sqrt(2.0), color="k", ls=":", lw=0.9, alpha=0.4)
    av_labels = [("B", tau_B), ("D" if not neg else r"$D_{\mathrm{sto}}$", tau_D)]
    if early_fail:
        av_labels.append(("fail", tau_fail))
    else:
        av_labels.append(
            (r"$L_{\mathrm{sto}}$" if neg else "E", float(full[-1].tau))
        )
    for lab, tmark in av_labels:
        ax.axvline(tmark, color="0.5", ls=":", lw=0.8, alpha=0.5)
        ax.text(tmark, 0.02, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="0.35")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("normalized")
    ax.set_title(r"Area / volume doubling")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3); ax.set_xlim(0.0, tau_hi)

    ax = axes[1, 2]
    ax.axis("off")
    if neg:
        phase_blurb = (
            "phases:\n"
            "  green  sphere A→B\n"
            "  blue   Seifert oblate B→D_sto\n"
            "  maroon stomatocyte D_sto→L_sto\n\n"
        )
        title = (
            rf"Negative-$C_0$ BS trajectory "
            rf"($\eta={eta:g}$, $C_0={C0_dim:g}$)"
        )
    else:
        pear_line = (
            "  maroon Seifert pear D→fail\n"
            if early_fail
            else "  maroon Seifert pear D→E\n"
        )
        phase_blurb = (
            "phases:\n"
            "  green  sphere A→B\n"
            "  blue   Seifert prolate B→D\n"
            + pear_line
            + "  orange two-sphere →E\n\n"
        )
        title = (
            rf"Full BS trajectory $\tau=0\to 1$ "
            rf"(sphere + Seifert + two-sphere, $\eta={eta:g}$)"
        )
    summary = (
        rf"$\eta={eta:g}$" + (rf", $C_0={C0_dim:g}$" if neg else "") + "\n\n"
        + f"frames: {len(full)}\n"
        + f"τ: {full[0].tau:.4f} → {full[-1].tau:.4f}\n\n"
        + phase_blurb
        + f"A/A₀(end) = {full[-1].A/A0:.4f}\n"
        + f"V/V₀(end) = {full[-1].V/V0:.4f}\n"
        + f"v(end)    = {full[-1].v:.4f}\n"
        + f"c₀(end)   = {full[-1].c0:.4f}\n"
        + f"P(end)    = {full[-1].P:.4f}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f4ef", edgecolor="0.8"))

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def plot_two_sphere_detail(
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    eta: Optional[float] = None,
) -> Path:
    if eta is None:
        eta = ETA
    out_dir = Path(out_dir)
    ref = bs_point_a_ref(C0, c0_start=BS_C0_START)
    A0ref, V0ref = float(ref["A"]), float(ref["V"])
    ts = np.array([r["tau"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(ts, [r["c0"] for r in rows], "-", color="#c45c26", lw=2)
    axes[0, 0].axhline(BS_C0_END, color="0.5", ls=":", lw=0.8)
    axes[0, 0].set_title(r"$c_0(\tau)$ (two-sphere)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(ts, [r["v"] for r in rows], "-", color="#c45c26", lw=2)
    axes[0, 1].axhline(BS_V_TWO_SPHERE, color="0.5", ls=":", lw=0.8)
    axes[0, 1].set_title(r"$\bar v(\tau)$")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(ts, [r["A"] / A0ref for r in rows], "-", lw=2, label=r"$A/A_0$")
    axes[1, 0].plot(ts, [r["V"] / V0ref for r in rows], "--", lw=1.6, label=r"$V/V_0$")
    axes[1, 0].axhline(2.0, color="k", ls="--", lw=0.8, alpha=0.4)
    axes[1, 0].legend()
    axes[1, 0].set_title("doubling")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ts, [r["R1"] for r in rows], "-", lw=2, label=r"$R_1$ (N)")
    axes[1, 1].plot(ts, [r["R2"] for r in rows], "-", lw=2, label=r"$R_2$ (S)")
    axes[1, 1].plot(ts, [r["P_bar"] for r in rows], ":", lw=1.5, label=r"$P̄$")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title(r"$R_{1,2}$, $P̄$")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(rf"Two-sphere finish $\tau_{{\mathrm{{fail}}}}\to$E ($\eta={eta:g}$)")
    fig.tight_layout()
    png = out_dir / "two_sphere_finish.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Movie
# ---------------------------------------------------------------------------

def _profile_for_frame(fr: TrajFrame) -> Tuple[np.ndarray, np.ndarray]:
    if fr.phase == "sphere":
        return sphere_meridian(area_radius(fr.A))
    if fr.phase == "two_sphere":
        if fr.R1 is not None and fr.R2 is not None:
            return two_sphere_meridian(fr.R1, fr.R2)
        radii = two_sphere_radii_from_AV(fr.A, fr.V)
        if radii is None:
            return np.array([0.0]), np.array([0.0])
        return two_sphere_meridian(radii[0], radii[1])
    if fr.phase == "nested_sphere":
        if fr.R1 is not None and fr.R2 is not None:
            return nested_sphere_meridian(fr.R1, fr.R2)
        radii = nested_sphere_radii_from_AV(fr.A, fr.V)
        if radii is None:
            return np.array([0.0]), np.array([0.0])
        return nested_sphere_meridian(radii[0], radii[1])
    if fr.sol is not None and fr.sol.y.shape[1] > 0:
        x = np.asarray(fr.sol.X, dtype=float)
        z = np.asarray(fr.sol.Z, dtype=float)
        z = z - 0.5 * (z.max() + z.min())
        return x, z
    return np.array([0.0]), np.array([0.0])


def movie_full_trajectory(
    full: Sequence[TrajFrame],
    out_dir: Path,
    *,
    eta: Optional[float] = None,
    target_frames: int = 180,
    fps: int = 16,
    dpi: int = 110,
    stem: str = "full_trajectory_shapes",
    cache: Optional[Path] = None,
) -> Path:
    """Animate meridian profiles along the assembled full trajectory."""
    if eta is None:
        eta = ETA
    if cache is None:
        cache = CACHE
    from matplotlib.animation import FFMpegWriter, PillowWriter, FuncAnimation

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not full:
        raise ValueError("empty trajectory")

    frames = list(full)
    attach_seifert_sols(frames, cache)

    n = len(frames)
    if target_frames > 0 and n > target_frames:
        idx = np.linspace(0, n - 1, int(target_frames), dtype=int)
        frames = [frames[i] for i in idx]
        if frames[-1] is not full[-1]:
            frames.append(full[-1])

    print(
        f"  movie: {len(frames)} profiles (from {n} full frames)…",
        flush=True,
    )
    profiles: List[Tuple[np.ndarray, np.ndarray, TrajFrame]] = []
    for i, fr in enumerate(frames):
        r, z = _profile_for_frame(fr)
        profiles.append((r, z, fr))
        if (i + 1) % 40 == 0 or i == 0 or i == len(frames) - 1:
            print(
                f"    [{i+1}/{len(frames)}] τ={fr.tau:.4f}  "
                f"{fr.phase}  v̄={fr.v:.4f}",
                flush=True,
            )

    r_max = max(float(np.max(np.abs(r))) for r, _, _ in profiles if r.size > 1)
    z_min = min(float(z.min()) for _, z, _ in profiles if z.size > 1)
    z_max = max(float(z.max()) for _, z, _ in profiles if z.size > 1)
    pad = 0.08 * max(r_max, z_max - z_min, 1.0)

    fig, ax = plt.subplots(figsize=(5.2, 5.8))
    line_r, = ax.plot([], [], lw=2.1)
    line_l, = ax.plot([], [], lw=2.1)
    subtitle = ax.set_title("")
    ax.set_aspect("equal")
    ax.set_xlim(-r_max - pad, r_max + pad)
    ax.set_ylim(z_min - pad, z_max + pad)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$z$")
    ax.grid(True, alpha=0.25)
    fig.suptitle(rf"Trajectory shapes ($\eta={eta:g}$)", fontsize=12)

    def init():
        line_r.set_data([], [])
        line_l.set_data([], [])
        subtitle.set_text("")
        return line_r, line_l, subtitle

    def update(k: int):
        r, z, fr = profiles[k]
        color = PHASE_COLORS.get(fr.phase, "#333333")
        line_r.set_data(r, z)
        line_l.set_data(-r, z)
        line_r.set_color(color)
        line_l.set_color(color)
        subtitle.set_text(
            rf"$\tau={fr.tau:.4f}$  {fr.phase.replace('_', '-')}  "
            rf"$c_0={fr.c0:.3f}$  $\bar v={fr.v:.4f}$"
        )
        return line_r, line_l, subtitle

    anim = FuncAnimation(
        fig, update, init_func=init, frames=len(profiles), blit=True,
    )
    mp4 = out_dir / f"{stem}.mp4"
    try:
        writer = FFMpegWriter(fps=fps)
        anim.save(str(mp4), writer=writer, dpi=dpi)
        out_path = mp4
    except Exception as exc:
        print(f"  ffmpeg failed ({exc}); writing GIF…", flush=True)
        gif = out_dir / f"{stem}.gif"
        anim.save(str(gif), writer=PillowWriter(fps=max(8, fps // 2)), dpi=dpi)
        out_path = gif
    plt.close(fig)
    print(f"saved {out_path}  ({len(profiles)} frames @ {fps} fps)", flush=True)
    return out_path




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
    _bs.main(argv)


__all__ = [
    "PHASE_COLORS",
    "attach_seifert_sols",
    "configure",
    "ensure_two_sphere_leg",
    "movie_full_trajectory",
    "plot_full_trajectory",
    "plot_seed_shape",
    "plot_two_sphere_detail",
    "run_two_sphere_finish",
    "save_growth_plots",
]
