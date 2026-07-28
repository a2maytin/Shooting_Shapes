#!/usr/bin/env python3
"""II.C sanity check: SC + ADE with effective coefficients, no inclusions.

Božič et al. PRE 73, 041915 (2006) §II.C: small single-inclusion energy maps
onto an ADE model with

    kc,eff/kc = 1 + κp − (1/2) hm² κ² p
    c0,eff    = (c0 + hm κ p) / (kc,eff/kc)
    kr,eff/kc = kr/kc + (1/2) hm² κ² p
    Δa0,eff   = Δa0 / (1 + (1/2) hm² κ² p / (kr/kc))

Default CLI marches the *leading* II.C effect: inclusions off (p_incl=0) and
c0 = c0_eff(p) with kr=0 (pure SC).  On the stomatocyte seed this yields a
rising Δa(p).  Free ADE (kr>0, free λΔa) leaves that sheet and decreases Δa;
see the module's ``--with-ade`` path / prior diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inclusions_functions import (
    INCL_DA0_DEFAULT,
    INCL_HM_DEFAULT,
    INCL_KAPPA_DEFAULT,
    INCL_KR_OVER_KC_DEFAULT,
    ade_lambda_from_da,
    seifert_to_bozic_multipliers,
    shoot_inclusion_shape,
)
from seifert_functions import load_solution

OUT_DIR = Path("results") / "inclusions_demo"
SEED = Path("seed_shapes") / "stomatocyte_starting_seed.json"


def iic_effective_coeffs(
    c0: float, hm: float, kappa: float, p: float, kr_over_kc: float, da0: float,
) -> dict:
    """Dimensionless II.C effective ADE coefficients (see module docstring)."""
    kp = float(kappa) * float(p)
    h2k2p = 0.5 * float(hm) ** 2 * float(kappa) ** 2 * float(p)
    kc_r = 1.0 + kp - h2k2p
    if abs(kc_r) < 1e-14:
        raise ValueError("kc,eff/kc vanished")
    return {
        "p": float(p),
        "kc_ratio": kc_r,
        "c0_eff": (float(c0) + float(hm) * kp) / kc_r,
        "kr_eff": float(kr_over_kc) + h2k2p,
        "da0_eff": float(da0) / (1.0 + h2k2p / max(float(kr_over_kc), 1e-30)),
        "kappa_p": kp,
        "h2k2p": h2k2p,
    }


def _local_bounds(prev, *, free_lda: bool, box: float = 0.6):
    U0, U1 = float(prev.U0), float(prev.U1)
    S1 = float(prev.constraints.get("S1", prev.S1))
    la, lv = float(prev.la), float(prev.lv)
    lo = [U0 - box, U1 - box, max(0.5, S1 - box), la - 4.0, lv - 4.0]
    hi = [U0 + box, U1 + box, S1 + box, la + 4.0, lv + 4.0]
    if free_lda:
        lda = float(prev.lda)
        lo.append(lda - 1.5)
        hi.append(lda + 1.5)
    return np.array(lo, float), np.array(hi, float)


def march_iic_ade(
    *,
    p_grid: np.ndarray,
    hm: float = INCL_HM_DEFAULT,
    kappa: float = INCL_KAPPA_DEFAULT,
    kr_over_kc: float = INCL_KR_OVER_KC_DEFAULT,
    da0: float = INCL_DA0_DEFAULT,
    c0_bare: float = 0.0,
    free_lda: bool = True,
    residual_tol: float = 1e-6,
    max_nfev: int = 120,
    verbose: bool = True,
):
    seed = load_solution(SEED)
    la0, lv0 = seifert_to_bozic_multipliers(seed.sigma_bar, seed.P_bar)
    v = float(seed.v)

    # p_paper=0: bare SC+ADE at seed (c0=0, Δa0, kr).
    e0 = iic_effective_coeffs(c0_bare, hm, kappa, 0.0, kr_over_kc, da0)
    if verbose:
        print(
            f"II.C ADE march: v={v:.3f} hm={hm:g} κ={kappa:g} kr/kc={kr_over_kc:g} "
            f"Δa0={da0:.4f}  free_lda={free_lda}",
            flush=True,
        )
    prev = shoot_inclusion_shape(
        v, e0["c0_eff"], hm=0.0, kappa=0.0, p=0.0,
        U0=float(seed.U0), U1=float(seed.U1), S1=float(seed.S1),
        la=la0, lv=lv0, lnu=0.0, lda=0.0,
        kr_over_kc=e0["kr_eff"], da0=e0["da0_eff"],
        n=160, max_nfev=max_nfev, residual_tol=residual_tol,
        verbose=verbose, free_lda=free_lda,
    )
    if not prev.success:
        raise RuntimeError(
            f"p=0 SC+ADE seed failed |res|={prev.constraints.get('residual')}"
        )
    da = float(prev.constraints["da"])
    rows = [{
        **e0,
        "c0_shoot": e0["c0_eff"],
        "da": da,
        "lda": float(prev.lda),
        "lda_target": ade_lambda_from_da(da, e0["da0_eff"], e0["kr_eff"]),
        "U0": float(prev.U0),
        "U1": float(prev.U1),
        "S1": float(prev.constraints["S1"]),
        "la": float(prev.la),
        "lv": float(prev.lv),
        "residual": float(prev.constraints["residual"]),
        "success": bool(prev.success),
    }]
    if verbose:
        print(
            f"  p=0    c0={e0['c0_eff']:.4f} da={da:.6f} lda={prev.lda:.5f} "
            f"|res|={prev.constraints['residual']:.2e}",
            flush=True,
        )

    for p in p_grid:
        if p <= 0:
            continue
        e = iic_effective_coeffs(c0_bare, hm, kappa, float(p), kr_over_kc, da0)
        lda_w = ade_lambda_from_da(
            float(prev.constraints["da"]), e["da0_eff"], e["kr_eff"],
        )
        bounds = _local_bounds(prev, free_lda=free_lda)
        sol = shoot_inclusion_shape(
            v, e["c0_eff"], hm=0.0, kappa=0.0, p=0.0,
            U0=float(prev.U0), U1=float(prev.U1),
            S1=float(prev.constraints["S1"]),
            la=float(prev.la), lv=float(prev.lv), lnu=0.0, lda=lda_w,
            kr_over_kc=e["kr_eff"], da0=e["da0_eff"],
            n=160, max_nfev=max_nfev, residual_tol=residual_tol,
            verbose=False, bounds=bounds, free_lda=free_lda,
        )
        # If free ADE fails, try fixed constitutive λΔa once (diagnostic only).
        if (not sol.success) and free_lda:
            sol_f = shoot_inclusion_shape(
                v, e["c0_eff"], hm=0.0, kappa=0.0, p=0.0,
                U0=float(prev.U0), U1=float(prev.U1),
                S1=float(prev.constraints["S1"]),
                la=float(prev.la), lv=float(prev.lv), lnu=0.0, lda=lda_w,
                kr_over_kc=e["kr_eff"], da0=e["da0_eff"],
                n=160, max_nfev=max_nfev, residual_tol=residual_tol,
                verbose=False, bounds=_local_bounds(prev, free_lda=False),
                free_lda=False,
            )
            if sol_f.success:
                if verbose:
                    print(
                        f"  p={p:6.1f} free ADE failed; fixed-λΔa ok "
                        f"da={sol_f.constraints['da']:.6f}",
                        flush=True,
                    )
                sol = sol_f
        da = float(sol.constraints.get("da", float("nan")))
        row = {
            **e,
            "c0_shoot": e["c0_eff"],
            "da": da,
            "lda": float(sol.lda),
            "lda_target": ade_lambda_from_da(da, e["da0_eff"], e["kr_eff"]),
            "U0": float(sol.U0),
            "U1": float(sol.U1),
            "S1": float(sol.constraints.get("S1", sol.S1)),
            "la": float(sol.la),
            "lv": float(sol.lv),
            "residual": float(sol.constraints.get("residual", 1e9)),
            "success": bool(sol.success),
        }
        rows.append(row)
        if verbose:
            print(
                f"  p={p:6.1f} c0_eff={e['c0_eff']:.4f} da={da:.6f} "
                f"lda={sol.lda:.5f} |res|={row['residual']:.2e} ok={sol.success}",
                flush=True,
            )
        if not sol.success:
            if verbose:
                print(f"  stalled at p={p:.4f}", flush=True)
            break
        prev = sol

    return rows


def plot_iic(rows, out_png: Path):
    p = np.array([r["p"] for r in rows])
    da = np.array([r["da"] for r in rows])
    c0 = np.array([r["c0_eff"] for r in rows])
    da0 = np.array([r["da0_eff"] for r in rows])
    ok = np.array([r["success"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].plot(p[ok], da[ok], "o-", color="#1f4e79", lw=1.8, ms=4, label=r"$\Delta a$ (II.C ADE)")
    ax[0].plot(p, da0, "--", color="0.5", lw=1.0, label=r"$\Delta a_{0,\mathrm{eff}}$")
    # Fig. 1b linear sketch for visual reference only
    ax[0].plot(
        [0, 200], [INCL_DA0_DEFAULT, 0.60], ":", color="#c44e52", lw=1.2,
        label="Fig. 1b sketch",
    )
    ax[0].set_xlabel(r"paper $p$ (via $c_{0,\mathrm{eff}}$)")
    ax[0].set_ylabel(r"$\Delta a$")
    ax[0].set_title("II.C SC+ADE benchmark")
    ax[0].legend(frameon=False, fontsize=9)

    ax[1].plot(p[ok], c0[ok], "o-", color="#2a9d8f", lw=1.8, ms=4)
    ax[1].set_xlabel(r"paper $p$")
    ax[1].set_ylabel(r"$c_{0,\mathrm{eff}}$")
    ax[1].set_title(r"$c_{0,\mathrm{eff}}(p)=h_m\kappa p/(1+\kappa p-\cdots)$")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    # Dense early, then out toward Fig. 1 scales
    p_grid = np.unique(np.concatenate([
        np.linspace(0, 20, 21),
        np.linspace(25, 100, 16),
        np.linspace(120, 200, 9),
        np.array([300.0, 400.0, 500.0, 600.0, 790.0]),
    ]))
    rows = march_iic_ade(p_grid=p_grid, free_lda=True, residual_tol=1e-5, verbose=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "iic_ade_benchmark.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2))
    png_path = OUT_DIR / "iic_ade_delta_a_vs_p.png"
    plot_iic(rows, png_path)
    last = rows[-1]
    print(
        f"\nWrote {json_path} and {png_path}\n"
        f"frames={len(rows)}  last p={last['p']:.4g} da={last['da']:.6f} "
        f"c0_eff={last['c0_eff']:.4f} ok={last['success']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
