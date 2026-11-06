#!/usr/bin/env python3
"""
SAMS vs PIP comparison for the collisions-only Snow & Hillier (2021) current sheet
(half domain x >= 0, symmetry plane at x = 0). See "SAMS PIP collisions-only comparison handoff.md".

Usage:
  python3 scripts/collonly_compare.py --pip <PIP outdir> [--sams <SAMS SAMS_testdata dir>] \
          [--snap 10] [--out <prefix>] [--window <x_lo> <x_hi>] [--xmax <x>]

  --sams    omit it for a PIP-only check (shock position, widths, boundary disturbance; no plots)
  --snap    snapshot number (PIP tNNNN.h5 <-> SAMS diagnostics_NNNNN_rank0.h5), default 10 (t = 100)
  --out     output prefix for the two PNGs (default collonly): <prefix>_full.png, <prefix>_shock.png
  --window  x-range of the shock plot (default: found automatically around the slow shock)
  --xmax    right edge of the full plot (default: where the disturbance falls below 1e-3)

Prints, for each code: time, cell size, slow-shock position, 10-90% widths in x units and in cells
(neutral density, neutral v_y, neutral T, B_y, plasma v_y), the largest disturbance in the outermost
1% of the domain (must be tiny: nothing may reach the boundary), and plateau values. PIP is the
reference; SAMS differences are given in cells and percent.

SAMS velocities are vertex-centred: they are plotted at their true positions x_c + dx/2, with the
vertex at x = 0 added (v_x = 0 there by symmetry; v_y is not written there). Nothing is averaged,
so any grid-scale structure stays visible.
"""
import argparse, os, sys
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 5.0/3.0

def load_pip(path):
    f = h5py.File(path, "r"); g = lambda k: np.array(f[k]).ravel()[1:-1]   # drop ghost cells
    o = {"time": float(np.array(f["time"]).ravel()[0]), "x": g("xgrid")}
    for s in ("p", "n"):
        rho = g("ro_"+s); v = [g(m+"_"+s)/rho for m in ("mx", "my", "mz")]
        ke = 0.5*rho*sum(c**2 for c in v)
        mag = 0.5*(g("bx")**2+g("by")**2+g("bz")**2) if s == "p" else 0.0
        pr = (G-1)*(g("en_"+s)-ke-mag)                                        # all velocity components
        o.update({"rho_"+s: rho, "vx_"+s: v[0], "vy_"+s: v[1], "pr_"+s: pr,
                  "T_"+s: (0.5 if s == "p" else 1.0)*G*pr/rho})               # T_p = gamma P/(2 rho)
    o["by"] = g("by"); o["xv"] = o["x"]
    return o

def load_sams(path):
    """path: one diagnostics_NNNNN_rank0.h5; files of the other MPI ranks (_rank1, _rank2, ...) are
    found and joined in x order (each rank writes its own x range)."""
    import glob, re
    files = sorted(glob.glob(re.sub(r"_rank\d+\.h5$", "_rank*.h5", path)),
                   key=lambda p: int(re.search(r"_rank(\d+)\.h5$", p).group(1))) or [path]
    fs = [h5py.File(p, "r") for p in files]
    fs.sort(key=lambda f: float(np.array(f["MeshCC/x"])[0]))
    m = lambda k: np.concatenate([np.array(f[k]).mean(axis=(1, 2)) for f in fs])
    x = np.concatenate([np.array(f["MeshCC/x"]) for f in fs]); dx = x[1]-x[0]
    assert np.allclose(np.diff(x), dx), "SAMS rank files do not join into a uniform grid"
    o = {"time": float(fs[0].attrs["time"]), "x": x, "xv": np.concatenate([[0.0], x+dx/2])}
    for s, sfx in (("p", ""), ("n", "_n")):
        rho = m("rho"+sfx); pr = (G-1)*rho*m("energy_ion" if s == "p" else "energy_n")
        o.update({"rho_"+s: rho, "pr_"+s: pr, "T_"+s: (0.5 if s == "p" else 1.0)*G*pr/rho,
                  "vx_"+s: np.concatenate([[0.0], m("vx"+sfx)]),
                  "vy_"+s: np.concatenate([[np.nan], m("vy"+sfx)])})
    o["by"] = m("by")
    return o

VERTEX = {"vx_p", "vx_n", "vy_p", "vy_n"}
def xs(D, k): return D["xv"] if k in VERTEX else D["x"]

def slow_shock(D):
    """Slow shock = steepest INCREASE of neutral density outward, away from the sheet (x > 1)."""
    x, r = D["x"], D["rho_n"]; g = np.gradient(r, x); g[x < 1.0] = 0.0
    return x[np.argmax(g)]

def width(D, k, xs0, half=6.0):
    """10-90% width of field k across the shock at xs0, between the values at xs0 -/+ half."""
    x = xs(D, k); v = D[k]; ok = np.isfinite(v)
    x, v = x[ok], v[ok]
    lo, hi = xs0-half, xs0+half
    vlo, vhi = np.interp(lo, x, v), np.interp(hi, x, v)
    m = (x >= lo) & (x <= hi); xx, f = x[m], (v[m]-vlo)/(vhi-vlo) if vhi != vlo else np.zeros(m.sum())
    try:
        x10 = xx[np.argmax(f > 0.1)]; x90 = xx[len(f)-1-np.argmax(f[::-1] < 0.9)]
    except ValueError:
        return np.nan
    return abs(x90-x10)

def edge_disturbance(D, D0):
    """Largest relative change since t = 0 in the outermost 1% of the domain."""
    xr = D["x"][-1]; out = {}
    for k in ("rho_n", "rho_p", "vx_n", "vx_p", "by", "T_p", "T_n"):
        x = xs(D, k); m = (x > xr-0.01*xr) & np.isfinite(D[k])
        scale = 1.0 if k.startswith("v") else np.abs(D0[k][np.isfinite(D0[k])]).max()
        out[k] = np.abs(D[k][m]-D0[k][m]).max()/scale
    return out

def reach(D, D0, thr=1e-3):
    """Largest x where rho_n, v_x,p or B_y differs from t = 0 by more than thr (relative)."""
    far = 0.0
    for k in ("rho_n", "vx_p", "by"):
        x = xs(D, k); d = np.abs(D[k]-D0[k]); d[~np.isfinite(d)] = 0.0
        scale = 1.0 if k.startswith("v") else np.abs(D0[k]).max()
        big = np.where(d > thr*scale)[0]
        if len(big): far = max(far, x[big.max()])
    return far

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pip", required=True); ap.add_argument("--sams")
    ap.add_argument("--snap", type=int, default=10); ap.add_argument("--out", default="collonly")
    ap.add_argument("--window", type=float, nargs=2); ap.add_argument("--xmax", type=float)
    a = ap.parse_args()
    P = load_pip(os.path.join(a.pip, f"t{a.snap:04d}.h5")); P0 = load_pip(os.path.join(a.pip, "t0000.h5"))
    if not a.sams:   # PIP-only check: shock position, widths and boundary disturbance; no plots
        dx = P["x"][1]-P["x"][0]; x0 = slow_shock(P)
        print(f"PIP  t = {P['time']:.4f}, {len(P['x'])} cells, dx = {dx:.5g}, x in [0, {P['x'][-1]+0.5*dx:.6g}]")
        print(f"slow shock (steepest rise of rho_n) at x = {x0:.4f}; 10-90% widths, x units (cells):")
        for k in ("rho_n", "vy_n", "T_n", "by", "vy_p", "rho_p"):
            w = width(P, k, x0); print(f"   {k:6s} {w:8.3f} ({w/dx:6.0f})")
        e = edge_disturbance(P, P0)
        print("outermost 1% of the domain, max relative change since t = 0: " + ", ".join(f"{k} {v:.1e}" for k, v in e.items())
              + f"   | 1e-3 disturbance reaches x = {reach(P, P0):.1f}")
        return
    S = load_sams(os.path.join(a.sams, f"diagnostics_{a.snap:05d}_rank0.h5"))
    S0 = load_sams(os.path.join(a.sams, "diagnostics_00000_rank0.h5"))

    # ---------------- numbers ----------------
    print(f"PIP  t = {P['time']:.4f}, {len(P['x'])} cells, dx = {P['x'][1]-P['x'][0]:.5g}, x in [0, {P['x'][-1]+0.5*(P['x'][1]-P['x'][0]):.6g}]")
    print(f"SAMS t = {S['time']:.4f}, {len(S['x'])} cells, dx = {S['x'][1]-S['x'][0]:.5g}, x in [0, {S['x'][-1]+0.5*(S['x'][1]-S['x'][0]):.6g}]")
    xsP, xsS = slow_shock(P), slow_shock(S); dxP, dxS = P["x"][1]-P["x"][0], S["x"][1]-S["x"][0]
    print(f"slow shock (steepest rise of rho_n): PIP x = {xsP:.4f}, SAMS x = {xsS:.4f}  (SAMS - PIP = {(xsS-xsP)/dxS:+.1f} SAMS cells)")
    print("10-90% widths across the shock   PIP: x units (cells)   SAMS: x units (cells)")
    for k in ("rho_n", "vy_n", "T_n", "by", "vy_p", "rho_p"):
        wP, wS = width(P, k, xsP), width(S, k, xsS)
        print(f"   {k:6s}  {wP:8.3f} ({wP/dxP:6.0f})   {wS:8.3f} ({wS/dxS:6.0f})")
    for name, D, D0 in (("PIP", P, P0), ("SAMS", S, S0)):
        e = edge_disturbance(D, D0)
        print(f"{name:4s} outermost 1% of the domain, max relative change since t = 0: " +
              ", ".join(f"{k} {v:.1e}" for k, v in e.items()) + f"   | 1e-3 disturbance reaches x = {reach(D, D0):.1f}")
    for lab, (lo, hi) in (("between sheet and shock", (0.3*xsP, 0.6*xsP)), ("just outside the shock", (xsP+3, xsP+8))):
        print(f"plateau {lab} (x in [{lo:.1f}, {hi:.1f}]):")
        for k in ("rho_n", "rho_p", "vx_n", "vx_p", "vy_n", "vy_p", "T_n", "T_p", "by"):
            mP = (xs(P, k) >= lo) & (xs(P, k) <= hi); mS = (xs(S, k) >= lo) & (xs(S, k) <= hi)
            vP, vS = np.nanmean(P[k][mP]), np.nanmean(S[k][mS])
            print(f"   {k:6s} PIP {vP:+.5e}  SAMS {vS:+.5e}  ({100*(vS-vP)/abs(vP) if vP else np.nan:+.2f}%)")

    # ---------------- plots ----------------
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e5e4df", "#fcfcfb"
    CN, CP = "#2a78d6", "#eb6834"          # neutral = blue, plasma (ions) = orange; code = line style
    STY = {"PIP": dict(ls="-", lw=2.2, alpha=0.55), "SAMS": dict(ls="--", lw=1.4, alpha=1.0)}
    def style(ax):
        ax.set_facecolor(surf); ax.grid(color=grid, lw=0.6); ax.tick_params(colors=ink2, labelsize=9)
        [s.set_color(grid) for s in ax.spines.values()]
    def pair(ax, k, lo, hi, logy=False, title=""):
        for code, D in (("PIP", P), ("SAMS", S)):
            for fl, c, lab in (("n", CN, "neutral"), ("p", CP, "plasma")):
                key = f"{k}_{fl}"; x = xs(D, key); m = (x >= lo) & (x <= hi)
                ax.plot(x[m], D[key][m], color=c, label=f"{code} {lab}", **STY[code])
        if logy: ax.set_yscale("log")
        ax.set_title(title, loc="left", color=ink, fontsize=10.5); style(ax)
    def single(ax, k, lo, hi, title, c="#0b0b0b"):
        for code, D in (("PIP", P), ("SAMS", S)):
            x = xs(D, k); m = (x >= lo) & (x <= hi)
            ax.plot(x[m], D[k][m], color=c, label=code, **STY[code])
        ax.set_title(title, loc="left", color=ink, fontsize=10.5); style(ax)

    xmax = a.xmax or min(P["x"][-1], 1.15*max(reach(P, P0), reach(S, S0), 1.5*xsP))
    fig, axs = plt.subplots(3, 2, figsize=(12.5, 10), sharex=True, facecolor=surf)
    pair(axs[0, 0], "rho", 0, xmax, logy=True, title="density")
    pair(axs[0, 1], "T", 0, xmax, logy=True, title="temperature")
    pair(axs[1, 0], "vx", 0, xmax, title="$v_x$ (inflow < 0)")
    pair(axs[1, 1], "vy", 0, xmax, title="$v_y$ (outflow along the sheet)")
    pair(axs[2, 0], "pr", 0, xmax, title="gas pressure")
    single(axs[2, 1], "by", 0, xmax, "$B_y$")
    for ax in axs[-1]: ax.set_xlabel("x", color=ink2)
    for ax in axs.flat: ax.axvline(xsP, color=ink2, lw=0.6, ls=":")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.962), ncol=4, frameon=False, fontsize=9, labelcolor=ink)
    fig.suptitle(f"Collisions-only current sheet, t = {P['time']:.4g}: PIP (solid) vs SAMS (dashed); dotted line = slow shock",
                 x=0.01, ha="left", color=ink, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{a.out}_full.png", dpi=120, facecolor=surf); plt.close(fig)

    wN = width(P, "vy_n", xsP)
    lo, hi = a.window if a.window else (xsP-2.5*max(wN, 1e-6), xsP+1.5*max(wN, 1e-6))
    fig, axs = plt.subplots(3, 2, figsize=(12.5, 10), sharex=True, facecolor=surf)
    pair(axs[0, 0], "rho", lo, hi, logy=True, title="density")
    pair(axs[0, 1], "T", lo, hi, title="temperature")
    pair(axs[1, 0], "vx", lo, hi, title="$v_x$")
    pair(axs[1, 1], "vy", lo, hi, title="$v_y$")
    # ion-neutral separation: drift velocity v_p - v_n, on the cell (PIP) / vertex (SAMS) grid
    for code, D in (("PIP", P), ("SAMS", S)):
        x = xs(D, "vx_p"); m = (x >= lo) & (x <= hi)
        axs[2, 0].plot(x[m], (D["vx_p"]-D["vx_n"])[m], color="#1baf7a", label=f"{code} $v_{{x,p}}-v_{{x,n}}$", **STY[code])
        axs[2, 0].plot(x[m], (D["vy_p"]-D["vy_n"])[m], color="#6250d6", label=f"{code} $v_{{y,p}}-v_{{y,n}}$", **STY[code])
    axs[2, 0].set_title("ion–neutral drift $v_p - v_n$", loc="left", color=ink, fontsize=10.5); style(axs[2, 0])
    axs[2, 0].legend(frameon=False, fontsize=8.5, labelcolor=ink)
    single(axs[2, 1], "by", lo, hi, "$B_y$")
    axs[2, 1].legend(frameon=False, fontsize=8.5, labelcolor=ink)
    for ax in axs[-1]: ax.set_xlabel("x", color=ink2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.962), ncol=4, frameon=False, fontsize=9, labelcolor=ink)
    fig.suptitle(f"Slow shock at t = {P['time']:.4g} (PIP x = {xsP:.2f}, SAMS x = {xsS:.2f}); "
                 f"neutral v_y width {wN:.2f} = {wN/dxP:.0f} PIP cells, {width(S, 'vy_n', xsS)/dxS:.0f} SAMS cells",
                 x=0.01, ha="left", color=ink, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{a.out}_shock.png", dpi=120, facecolor=surf); plt.close(fig)
    print(f"wrote {a.out}_full.png and {a.out}_shock.png")

if __name__ == "__main__":
    main()
