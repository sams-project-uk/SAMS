#!/usr/bin/env python3
"""Exact self-similar solution of the symmetric current-sheet Riemann problem in ideal 1D MHD, and
comparison with SAMS and PIP.

Problem (half domain x >= 0, mirror plane at x = 0): uniform rho0, P0, v = 0, B_x = Bx and
B_y = -B0 for x > 0 (+B0 for x < 0). By symmetry v_x = 0 and B_y = 0 at x = 0. Going outwards from
the sheet, the solution for x > 0 is:

    state 2 (B_y = 0, v_x = 0) | switch-off slow shock | state 1 | fast rarefaction | state 0

A rotational discontinuity is not needed: B_y passes through zero, and only switch-off shocks
reach B_y = 0. The one unknown, the density rho1 behind the rarefaction, is fixed by v_x2 = 0.

Limits of the two-fluid problem:
  - alpha0 = 0 (uncoupled): plasma alone, rho0 = xi_i, P0 = P_p.  --limit uncoupled
  - alpha0 -> infinity: one fluid with the total density and pressure, rho0 = 1, P0 = P_t.  --limit coupled
The neutrals of the uncoupled case stay uniform.

Usage:
  python3 mhd_riemann_exact.py --limit uncoupled --t 10            # print the states
  python3 mhd_riemann_exact.py --limit uncoupled --t 10 \
      --pip runs/scan/pip_a1e-8/out --sams runs/scan/sams_a0/SAMS_testdata --snap 1 --out exact_a0_s1
"""
import argparse, os, sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

G = 5.0/3.0


def equilibrium_xi_n(T0=1.0e4):
    """Neutral fraction at T0, as in SAMS's PIP::get_equilibrium_ion_fraction; 0.99734 at 1e4 K.
    Hard-coded from the SAMS start-up print, so the exact solution uses the same number."""
    return 0.99734


def initial_state(limit, beta=0.1, B0=1.0, Bx=0.1, xi_n=None):
    xi_n = equilibrium_xi_n() if xi_n is None else xi_n
    xi_i = 1.0 - xi_n
    P_t = 0.5*beta*B0**2
    if limit == "uncoupled":
        return dict(rho=xi_i, P=2.0*xi_i/(xi_n + 2.0*xi_i)*P_t, vx=0.0, vy=0.0, Bx=Bx, By=-B0)
    if limit == "coupled":
        return dict(rho=1.0, P=P_t, vx=0.0, vy=0.0, Bx=Bx, By=-B0)
    raise ValueError(limit)


def cfast(rho, P, Bx, By):
    a2, b2, bx2 = G*P/rho, (Bx**2 + By**2)/rho, Bx**2/rho
    return np.sqrt(0.5*(a2 + b2 + np.sqrt((a2 + b2)**2 - 4.0*a2*bx2)))


def rarefaction(s0, rho1, n=400):
    """Right-moving fast rarefaction from state 0 down to density rho1. The ODEs in rho come from the
    fast eigenvector: dvx = c dρ/ρ, dBy = By c² dρ/(ρ(c² - Bx²/ρ)), dvy = -Bx dBy/(ρ c), P isentropic."""
    Bx, P0, r0 = s0["Bx"], s0["P"], s0["rho"]
    P = lambda r: P0*(r/r0)**G
    def f(r, y):
        vx, vy, By = y
        c = cfast(r, P(r), Bx, By)
        dBy = By*c**2/(r*(c**2 - Bx**2/r))
        return [c/r, -Bx*dBy/(r*c), dBy]
    sol = solve_ivp(f, (r0, rho1), [0.0, 0.0, s0["By"]], rtol=1e-11, atol=1e-13, dense_output=True)
    r = np.linspace(r0, rho1, n); vx, vy, By = sol.sol(r)
    return dict(rho=r, P=P(r), vx=vx, vy=vy, By=By, xi=vx + cfast(r, P(r), Bx, By))


def switch_off(s1, Bx):
    """Switch-off slow shock moving right into state 1. In the shock frame the upstream normal velocity
    is u1 = -Bx/sqrt(rho1), the Alfvén speed on B_x. Returns state 2 (B_y = 0) and the shock speed."""
    r1, P1, vx1, vy1, By1 = (s1[k] for k in ("rho", "P", "vx", "vy", "By"))
    u1 = -abs(Bx)/np.sqrt(r1); m = r1*u1; s = vx1 - u1
    vy2 = vy1 - Bx*By1/m                                          # tangential momentum (and induction)
    Pt1 = P1 + 0.5*By1**2 + m*u1                                  # normal momentum flux
    E1 = u1*(0.5*r1*(u1**2 + vy1**2) + G/(G - 1)*P1 + By1**2) - Bx*vy1*By1   # energy flux
    def res(r2):
        u2 = m/r2; P2 = Pt1 - m*u2
        return u2*(0.5*r2*(u2**2 + vy2**2) + G/(G - 1)*P2) - E1
    rr = np.geomspace(r1*(1 + 1e-9), r1*4.0, 4000); v = [res(r) for r in rr]
    roots = [brentq(res, rr[i], rr[i+1], xtol=1e-15*r1) for i in range(len(rr) - 1) if v[i]*v[i+1] < 0]
    if len(roots) != 1:
        raise RuntimeError(f"switch-off shock: {len(roots)} roots for rho2")
    r2 = roots[0]; u2 = m/r2
    return dict(rho=r2, P=Pt1 - m*u2, vx=s + u2, vy=vy2, By=0.0), s


def solve(s0):
    Bx = s0["Bx"]
    def end(rho1):
        R = rarefaction(s0, rho1, n=2)
        s1 = {k: R[k][-1] for k in ("rho", "P", "vx", "vy", "By")}
        s2, s = switch_off(s1, Bx)
        return s1, s2, s
    rho1 = brentq(lambda r: end(r)[1]["vx"], s0["rho"]*0.3, s0["rho"]*(1 - 1e-6), xtol=1e-14*s0["rho"])
    s1, s2, s = end(rho1)
    return dict(s0=s0, s1=s1, s2=s2, shock_speed=s, fan=rarefaction(s0, rho1),
                head=cfast(s0["rho"], s0["P"], Bx, s0["By"]))


def profile(sol, x, t):
    """Exact solution at time t on positions x >= 0; also the vertex-free T = gamma P/(n rho)."""
    s0, s1, s2, F = sol["s0"], sol["s1"], sol["s2"], sol["fan"]
    xi = x/t; out = {}
    for k in ("rho", "P", "vx", "vy", "By"):
        y = np.where(xi < sol["shock_speed"], s2[k], s1[k])
        fan = (xi >= F["xi"][-1]) & (xi < F["xi"][0])
        y = np.where(fan, np.interp(xi, F["xi"][::-1], F[k][::-1]), y)
        out[k] = np.where(xi >= sol["head"], s0[k], y)
    return out


def report(sol, t, Tfac):
    s0, s1, s2 = sol["s0"], sol["s1"], sol["s2"]
    print(f"Bx = {s0['Bx']}, t = {t}; T = {Tfac} gamma P / rho")
    print(f"{'state':10s} {'rho':>12s} {'P':>12s} {'T':>10s} {'vx':>10s} {'vy':>10s} {'By':>10s}")
    for n, s in (("2 (inner)", s2), ("1", s1), ("0", s0)):
        print(f"{n:10s} {s['rho']:12.6e} {s['P']:12.6e} {Tfac*G*s['P']/s['rho']:10.4f} "
              f"{s['vx']:10.5f} {s['vy']:10.5f} {s['By']:10.5f}")
    print(f"slow shock: speed {sol['shock_speed']:.6f}, x = {sol['shock_speed']*t:.4f} at t = {t}; "
          f"compression rho2/rho1 = {s2['rho']/s1['rho']:.5f}")
    print(f"fast rarefaction: tail x = {sol['fan']['xi'][-1]*t:.3f}, head x = {sol['head']*t:.3f}")


def compare(sol, args, Tfac):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from collonly_compare import load_pip, load_sams
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sp = "n" if args.limit == "coupled" else "p"      # coupled: compare the neutrals (≈ the total fluid)
    runs = {}
    if args.pip: runs["PIP"] = load_pip(os.path.join(args.pip, f"t{args.snap:04d}.h5"))
    if args.sams: runs["SAMS"] = load_sams(os.path.join(args.sams, f"diagnostics_{args.snap:05d}_rank0.h5"))
    t = next(iter(runs.values()))["time"]
    xs = sol["shock_speed"]*t
    print(f"\nComparison at t = {t:g} (species {sp}); exact slow shock at x = {xs:.4f}")
    for name, D in runs.items():
        x, r = D["x"], D["rho_"+sp]
        g = np.gradient(r, x); g[x < 0.2*xs] = 0
        xsh = x[np.argmax(np.abs(g))]                       # steepest density jump away from the sheet
        m = (x > 0.2*xs) & (x < 0.8*xs)
        E = profile(sol, x[m], t)
        dr = r[m].mean()/E["rho"].mean() - 1; dT = D["T_"+sp][m].mean()/(Tfac*G*E["P"]/E["rho"]).mean() - 1
        dP = (D["pr_"+sp][m]).mean()/E["P"].mean() - 1
        print(f"  {name:5s} shock x = {xsh:.3f} ({xsh/xs:.4f} of exact) | inner plateau vs exact: "
              f"rho {dr:+.2%}, T {dT:+.2%}, P {dP:+.2%}")
    xmax = args.xmax or 1.5*sol["head"]*t
    xe = np.linspace(0, xmax, 4000); E = profile(sol, xe, t)
    E["T"] = Tfac*G*E["P"]/E["rho"]
    fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex=True); ax = ax.ravel()
    style = {"PIP": dict(color="#e07b39", lw=2.2, alpha=0.6), "SAMS": dict(color="#3b6fd8", lw=1.2, ls="--")}
    for i, (k, lab) in enumerate((("rho", r"$\rho$"), ("T", "T"), ("vx", "$v_x$"), ("vy", "$v_y$"),
                                   ("By", "$B_y$"), ("P", "P"))):
        a = ax[i]; a.plot(xe, E[k], color="k", lw=1.0, label="exact")
        for name, D in runs.items():
            key = {"rho": "rho_"+sp, "T": "T_"+sp, "vx": "vx_"+sp, "vy": "vy_"+sp, "By": "by", "P": "pr_"+sp}[k]
            xx = D["xv"] if k in ("vx", "vy") else D["x"]
            a.plot(xx, D[key], label=name, **style[name])
        a.set_title(lab, loc="left"); a.grid(alpha=0.3)
        if args.window: a.set_xlim(*args.window)
        else: a.set_xlim(0, xmax)
    ax[0].legend(); ax[4].set_xlabel("x"); ax[5].set_xlabel("x")
    fig.suptitle(f"{args.limit} limit, t = {t:g}: exact slow shock x = {xs:.3f}")
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=110)
    print(f"wrote {args.out}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", choices=("uncoupled", "coupled"), default="uncoupled")
    ap.add_argument("--t", type=float, default=10.0)
    ap.add_argument("--pip"); ap.add_argument("--sams"); ap.add_argument("--snap", type=int, default=1)
    ap.add_argument("--out", default="exact"); ap.add_argument("--xmax", type=float)
    ap.add_argument("--window", type=float, nargs=2)
    a = ap.parse_args()
    sol = solve(initial_state(a.limit))
    # T = Tfac gamma P / rho, as in collonly_compare.py. Plasma (ions + electrons): Tfac = 1/2. Coupled
    # fluid (neutrals, ions, electrons at one temperature): number density (xi_n + 2 xi_i) rho.
    xi_n = equilibrium_xi_n()
    Tfac = 0.5 if a.limit == "uncoupled" else 1.0/(xi_n + 2.0*(1.0 - xi_n))
    report(sol, a.t, Tfac)
    if a.pip or a.sams:
        compare(sol, a, Tfac)


if __name__ == "__main__":
    main()
