#!/usr/bin/env python3
"""kr/kc=3: II.C ADE march, then actual inclusion shoots seeded from II.C.

No Picard.  ADE uses free λ_Δa matched to eq. (13) in the least-squares
residuals (paper-style).  Each actual step is warm-started from the II.C
shape at the same paper-p, with

    λ_a ← λ_a^IIC − λ_ν + (1/2) κ p h_m²
    λ_ν fitted so n=1 on the II.C meridian
    λ_Δa, U0, U1, S1, λ_v from II.C
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from inclusions_functions import (
    INCL_HM_DEFAULT,
    INCL_KAPPA_DEFAULT,
    INCL_KR_OVER_KC_DEFAULT,
    save_inclusion_trajectory,
    seifert_to_bozic_multipliers,
    shoot_inclusion_shape,
    nu_from_state,
)
from run_iic_ade_benchmark import iic_effective_coeffs
from seifert_functions import load_solution

OUT = Path("results") / "inclusions_demo"
SEED = Path("seed_shapes") / "stomatocyte_starting_seed.json"


def _bounds5(prev, box: float = 0.7):
    U0, U1 = float(prev.U0), float(prev.U1)
    S1 = float(prev.constraints["S1"])
    la, lv = float(prev.la), float(prev.lv)
    lda = float(prev.lda)
    lo = [U0 - box, U1 - box, max(0.5, S1 - box), la - 5.0, lv - 5.0, lda - 1.5]
    hi = [U0 + box, U1 + box, S1 + box, la + 5.0, lv + 5.0, lda + 1.5]
    return np.array(lo, float), np.array(hi, float)


def _bounds6(U0, U1, S1, la, lv, lnu, lda, box: float = 0.8):
    dlnu = max(2.0, 0.25 * abs(lnu) + 1.0)
    lo = [U0 - box, U1 - box, max(0.5, S1 - box), la - 8.0, lv - 5.0, lnu - dlnu, lda - 1.5]
    hi = [U0 + box, U1 + box, S1 + box, la + 8.0, lv + 5.0, lnu + dlnu, lda + 1.5]
    return np.array(lo, float), np.array(hi, float)


def fit_lnu_on_shape(sol, *, hm: float, kappa: float, p: float) -> float:
    s = np.asarray(sol.s, float)
    rho, psi, chi = sol.y[0], sol.y[2], sol.y[3]

    def n_of(lnu: float) -> float:
        nu = np.array([
            nu_from_state(
                float(rho[i]), float(psi[i]), float(chi[i]),
                kappa=kappa, hm=hm, p=p, lnu=lnu,
            )
            for i in range(len(s))
        ])
        return float(np.trapz(0.5 * nu * np.abs(rho), s)) - 1.0

    lo, hi = -3.0 * p, 3.0 * p
    for _ in range(40):
        if n_of(lo) * n_of(hi) < 0.0:
            break
        lo, hi = lo * 1.5, hi * 1.5
    return float(brentq(n_of, lo, hi))


def march_iic_free_lda(
    *,
    v: float,
    seed_sol,
    da0: float,
    hm: float,
    kappa: float,
    kr: float,
    p_grid: np.ndarray,
    verbose: bool = True,
):
    """II.C effective ADE with free λ_Δa (no Picard)."""
    prev = seed_sol
    prev.lda = 0.0
    rows = [{
        "p": 0.0,
        "c0_eff": 0.0,
        "kr_eff": float(kr),
        "da0_eff": float(da0),
        "da": float(da0),
        "lda": 0.0,
        "U0": float(prev.U0),
        "U1": float(prev.U1),
        "S1": float(prev.constraints["S1"]),
        "la": float(prev.la),
        "lv": float(prev.lv),
        "residual": float(prev.constraints["residual"]),
        "success": True,
        "sol": prev,
    }]
    if verbose:
        print(f"II.C free-λΔa march  da0={da0:.6f}  kr/kc={kr:g}", flush=True)

    for p in p_grid:
        if p <= 0:
            continue
        e = iic_effective_coeffs(0.0, hm, kappa, float(p), kr, da0)
        sol = shoot_inclusion_shape(
            v, e["c0_eff"], hm=0.0, kappa=0.0, p=0.0,
            U0=float(prev.U0), U1=float(prev.U1),
            S1=float(prev.constraints["S1"]),
            la=float(prev.la), lv=float(prev.lv), lnu=0.0, lda=float(prev.lda),
            kr_over_kc=e["kr_eff"], da0=e["da0_eff"],
            n=140, max_nfev=120, residual_tol=1e-5,
            verbose=False, free_lda=True, bounds=_bounds5(prev),
        )
        row = {
            "p": float(p),
            "c0_eff": e["c0_eff"],
            "kr_eff": e["kr_eff"],
            "da0_eff": e["da0_eff"],
            "da": float(sol.constraints.get("da", float("nan"))),
            "lda": float(sol.lda),
            "U0": float(sol.U0),
            "U1": float(sol.U1),
            "S1": float(sol.constraints.get("S1", sol.S1)),
            "la": float(sol.la),
            "lv": float(sol.lv),
            "residual": float(sol.constraints.get("residual", 1e9)),
            "success": bool(sol.success),
            "sol": sol if sol.success else None,
        }
        rows.append(row)
        if verbose:
            print(
                f"  IIC p={p:6.1f} c0={e['c0_eff']:.4f} da={row['da']:.6f} "
                f"lda={row['lda']:.5f} |res|={row['residual']:.2e} ok={row['success']}",
                flush=True,
            )
        if not sol.success:
            break
        prev = sol
    return rows


def shoot_actual_from_iic(
    iic_sol,
    *,
    v: float,
    p: float,
    hm: float,
    kappa: float,
    kr: float,
    da0: float,
    verbose: bool = True,
):
    """One free-λΔa inclusion shoot seeded from an II.C ADE solution."""
    lnu = fit_lnu_on_shape(iic_sol, hm=hm, kappa=kappa, p=p)
    iso = 0.5 * float(kappa) * float(p) * float(hm) ** 2
    la_w = float(iic_sol.la) - lnu + iso
    S1 = float(iic_sol.constraints["S1"])
    U0, U1 = float(iic_sol.U0), float(iic_sol.U1)
    lv, lda = float(iic_sol.lv), float(iic_sol.lda)

    sol = shoot_inclusion_shape(
        v, 0.0, hm=hm, kappa=kappa, p=p,
        U0=U0, U1=U1, S1=S1, la=la_w, lv=lv, lnu=lnu, lda=lda,
        kr_over_kc=kr, da0=da0,
        n=140, max_nfev=150, residual_tol=1e-5,
        verbose=False, free_lda=True,
        bounds=_bounds6(U0, U1, S1, la_w, lv, lnu, lda),
    )
    if verbose:
        da_i = float(iic_sol.constraints["da"])
        da_a = float(sol.constraints.get("da", float("nan")))
        print(
            f"  act p={p:6.1f} da={da_a:.6f} (IIC {da_i:.6f}, d={da_a - da_i:+.6f}) "
            f"lda={float(sol.lda):.5f} (IIC {lda:.5f}) lnu={float(sol.lnu):.3f} "
            f"|res|={float(sol.constraints.get('residual', 1e9)):.2e} ok={sol.success}",
            flush=True,
        )
    return sol


def closed_xz(sol):
    x = np.asarray(sol.X, float)
    z = -np.asarray(sol.Z, float)
    z = z - 0.5 * (z.max() + z.min())
    return np.concatenate([x, -x[::-1]]), np.concatenate([z, z[::-1]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hm, kap, kr = INCL_HM_DEFAULT, INCL_KAPPA_DEFAULT, INCL_KR_OVER_KC_DEFAULT
    seed = load_solution(SEED)
    la0, lv0 = seifert_to_bozic_multipliers(seed.sigma_bar, seed.P_bar)
    v = float(seed.v)

    print("=== p=0 seed ===", flush=True)
    p0 = shoot_inclusion_shape(
        v, 0.0, hm=0.0, kappa=0.0, p=0.0,
        U0=float(seed.U0), U1=float(seed.U1), S1=float(seed.S1),
        la=la0, lv=lv0, lnu=0.0, lda=0.0,
        kr_over_kc=kr, da0=0.5,
        n=160, max_nfev=80, residual_tol=1e-6, verbose=False, free_lda=False,
    )
    if not p0.success:
        raise RuntimeError(f"p=0 failed |res|={p0.constraints.get('residual')}")
    da0 = float(p0.constraints["da"])
    p0.lda = 0.0
    print(f"da0={da0:.6f}", flush=True)

    p_grid = np.unique(np.concatenate([
        np.linspace(0, 20, 11),
        np.linspace(24, 60, 10),
        np.linspace(80, 200, 7),
        np.linspace(250, 500, 6),
        np.array([600.0, 700.0, 750.0, 790.0, 790.5]),
    ]))

    print("\n=== II.C (free λΔa) ===", flush=True)
    iic_rows = march_iic_free_lda(
        v=v, seed_sol=p0, da0=da0, hm=hm, kappa=kap, kr=kr, p_grid=p_grid,
    )

    print("\n=== actual from II.C seeds (free λΔa, no Picard) ===", flush=True)
    actual = [p0]
    for row in iic_rows[1:]:
        if not row["success"] or row["sol"] is None:
            break
        sol = shoot_actual_from_iic(
            row["sol"], v=v, p=row["p"], hm=hm, kappa=kap, kr=kr, da0=da0,
        )
        actual.append(sol)
        if not sol.success:
            print(f"  stalled at p={row['p']:.4f}", flush=True)
            break

    ok_act = [s for s in actual if s.success]
    save_inclusion_trajectory(OUT / "stomatocyte_kr3_iic_seeded_traj.json", ok_act)

    # plot
    ok_iic = [r for r in iic_rows if r["success"]]
    pi = np.array([r["p"] for r in ok_iic])
    dai = np.array([r["da"] for r in ok_iic])
    ldai = np.array([r["lda"] for r in ok_iic])
    pa = np.array([s.p for s in ok_act])
    daa = np.array([float(s.constraints["da"]) for s in ok_act])
    ldaa = np.array([float(s.lda) for s in ok_act])

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax[0].plot(pi, dai, "-", color="#2a9d8f", lw=2.2, label="II.C ADE")
    ax[0].plot(pa, daa, "o-", color="#c44e52", lw=1.5, ms=4, label="actual (II.C seed)")
    ax[0].axhline(da0, color="0.6", ls=":")
    ax[0].set_xlabel(r"$p$")
    ax[0].set_ylabel(r"$\Delta a$")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(r"$k_r/k_c=3$: free $\lambda_{\Delta a}$, no Picard")

    ax[1].plot(pi, ldai, "-", color="#2a9d8f", lw=2.2, label="II.C")
    ax[1].plot(pa, ldaa, "o-", color="#c44e52", lw=1.5, ms=4, label="actual")
    ax[1].set_xlabel(r"$p$")
    ax[1].set_ylabel(r"$\lambda_{\Delta a}$")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title(r"$\lambda_{\Delta a}(p)$")

    for pt, ls in [(0, "-"), (20, "--"), (60, "-."), (min(pa[-1], 200), ":")]:
        ia = int(np.argmin(np.abs(pa - pt)))
        ii = int(np.argmin(np.abs(pi - pt)))
        if ok_iic[ii]["sol"] is None and pt > 0:
            continue
        Xi, Zi = closed_xz(ok_iic[ii]["sol"] if pt > 0 else p0)
        Xa, Za = closed_xz(ok_act[ia])
        ax[2].plot(Xi, Zi, ls, color="#2a9d8f", lw=1.5, alpha=0.9)
        ax[2].plot(Xa, Za, ls, color="#c44e52", lw=1.5, alpha=0.9)
    ax[2].set_aspect("equal")
    ax[2].set_title("teal=II.C, red=actual")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    fig.tight_layout()
    png = OUT / "kr3_iic_seeded_no_picard.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)
    print(f"\nwrote {png}", flush=True)

    (OUT / "kr3_iic_seeded_no_picard.json").write_text(json.dumps({
        "da0": da0,
        "note": "II.C-seeded actual; free λΔa; no Picard",
        "iic": [{k: v for k, v in r.items() if k != "sol"} for r in iic_rows],
        "actual": [{
            "p": float(s.p),
            "da": float(s.constraints["da"]),
            "lda": float(s.lda),
            "U0": float(s.U0),
            "lnu": float(s.lnu),
            "residual": float(s.constraints.get("residual", float("nan"))),
            "success": bool(s.success),
        } for s in actual],
    }, indent=2))

    print("\nmatched:")
    print(f'{"p":>7} {"da_act":>10} {"da_iic":>10} {"diff":>10} {"lda_a":>9} {"lda_i":>9}')
    for s in ok_act:
        j = int(np.argmin(np.abs(pi - s.p)))
        r = ok_iic[j]
        if abs(r["p"] - s.p) > 2.0 and s.p > 0:
            continue
        print(
            f'{s.p:7.1f} {float(s.constraints["da"]):10.6f} {r["da"]:10.6f} '
            f'{float(s.constraints["da"]) - r["da"]:10.6f} '
            f'{s.lda:9.5f} {r["lda"]:9.5f}'
        )
    print(
        f"done: IIC frames={len(ok_iic)} actual frames={len(ok_act)} "
        f"p_last={ok_act[-1].p:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
