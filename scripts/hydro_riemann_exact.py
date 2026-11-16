#!/usr/bin/env python3
"""Exact Riemann solver for 1D ideal-gas hydrodynamics (Toro, ch. 4), and comparison with a SAMS pure-hydro run
in which both LARE packages solve the same problem side by side (sams LARE3DST LARE3DNF SodShockTubeNeutral,
built with runs/hydro_build.sh; no coupling, plasma with B = 0).

For each fluid (plasma: rho, energy_ion, vx; neutral: rho_n, energy_n, vx_n) it prints the right-moving shock
position against the exact one, the errors of the state between contact and shock (or between the two shocks for
colliding flows), and the Rankine-Hugoniot balance across the code's own shock: mass, momentum and energy fluxes in
the frame of the shock, with the shock speed from mass conservation. An energy-flux mismatch is energy the scheme
loses (or creates) at the shock.

Usage:
  hydro_riemann_exact.py <run>/SAMS_testdata <snap> --left RHO U P --right RHO U P [--x0 0.5] [--gamma 1.4] [--out f]
  Sod:      --left 1 0 1 --right 0.125 0 0.1 --x0 0.5 --gamma 1.4
  collide:  --left 1 6 0.05 --right 1 -6 0.05 --x0 0 --gamma 1.6666666667
"""
import argparse, glob, os, re, sys
import numpy as np, h5py


class Riemann:
    def __init__(self, WL, WR, g):
        self.WL, self.WR, self.g = WL, WR, g
        self.cL, self.cR = np.sqrt(g*WL[2]/WL[0]), np.sqrt(g*WR[2]/WR[0])
        self.p, self.u = self._star()

    def _f(self, p, W, c):
        r, u, P = W; g = self.g
        if p > P:   # shock
            A, B = 2/((g + 1)*r), (g - 1)/(g + 1)*P
            return (p - P)*np.sqrt(A/(p + B)), np.sqrt(A/(p + B))*(1 - (p - P)/(2*(B + p)))
        q = (p/P)**((g - 1)/(2*g))                       # rarefaction
        return 2*c/(g - 1)*(q - 1), 1/(r*c)*(p/P)**(-(g + 1)/(2*g))

    def _star(self):
        WL, WR = self.WL, self.WR; du = WR[1] - WL[1]
        p = max(1e-12, 0.5*(WL[2] + WR[2]) - 0.125*du*(WL[0] + WR[0])*(self.cL + self.cR))
        for _ in range(200):
            fL, dL = self._f(p, WL, self.cL); fR, dR = self._f(p, WR, self.cR)
            pn = max(1e-14, p - (fL + fR + du)/(dL + dR))
            if abs(pn - p) < 1e-14*(pn + p): p = pn; break
            p = pn
        fL, _ = self._f(p, WL, self.cL); fR, _ = self._f(p, WR, self.cR)
        return p, 0.5*(WL[1] + WR[1]) + 0.5*(fR - fL)

    def right_shock_speed(self):
        r, u, P = self.WR; g = self.g
        if self.p <= P: return None
        return u + self.cR*np.sqrt((g + 1)/(2*g)*self.p/P + (g - 1)/(2*g))

    def star_right(self):
        r, u, P = self.WR; g = self.g; q = self.p/P
        if self.p > P: return r*(q + (g - 1)/(g + 1))/((g - 1)/(g + 1)*q + 1)
        return r*q**(1/g)

    def sample(self, xi):
        """Exact (rho, u, p) at x/t = xi (vectorised)."""
        g = self.g; out = np.zeros((3, len(xi)))
        for k, s in enumerate(xi):
            if s <= self.u:   # left of contact
                r, u, P = self.WL; c = self.cL; ps = self.p
                if ps > P:
                    S = u - c*np.sqrt((g + 1)/(2*g)*ps/P + (g - 1)/(2*g))
                    out[:, k] = (self.WL if s < S else (r*((ps/P) + (g - 1)/(g + 1))/((g - 1)/(g + 1)*ps/P + 1), self.u, ps))
                else:
                    cs = c*(ps/P)**((g - 1)/(2*g)); SH, ST = u - c, self.u - cs
                    if s < SH: out[:, k] = self.WL
                    elif s > ST: out[:, k] = (r*(ps/P)**(1/g), self.u, ps)
                    else:
                        ue = 2/(g + 1)*(c + (g - 1)/2*u + s); ce = ue - s
                        out[:, k] = (r*(ce/c)**(2/(g - 1)), ue, P*(ce/c)**(2*g/(g - 1)))
            else:             # right of contact
                r, u, P = self.WR; c = self.cR; ps = self.p
                if ps > P:
                    S = u + c*np.sqrt((g + 1)/(2*g)*ps/P + (g - 1)/(2*g))
                    out[:, k] = (self.WR if s > S else (r*((ps/P) + (g - 1)/(g + 1))/((g - 1)/(g + 1)*ps/P + 1), self.u, ps))
                else:
                    cs = c*(ps/P)**((g - 1)/(2*g)); SH, ST = u + c, self.u + cs
                    if s > SH: out[:, k] = self.WR
                    elif s < ST: out[:, k] = (r*(ps/P)**(1/g), self.u, ps)
                    else:
                        ue = 2/(g + 1)*(-c + (g - 1)/2*u + s); ce = s - ue
                        out[:, k] = (r*(ce/c)**(2/(g - 1)), ue, P*(ce/c)**(2*g/(g - 1)))
        return out


def load(datadir, snap):
    files = sorted(glob.glob(os.path.join(datadir, f"diagnostics_{snap:05d}_rank*.h5")),
                   key=lambda p: int(re.search(r"_rank(\d+)\.h5$", p).group(1)))
    fs = sorted([h5py.File(p, "r") for p in files], key=lambda f: float(np.array(f["MeshCC/x"])[0]))
    cat = lambda k: np.concatenate([np.array(f[k]).mean(axis=(1, 2)) for f in fs])
    x = np.concatenate([np.array(f["MeshCC/x"]) for f in fs]); dx = x[1] - x[0]
    D = {"t": float(fs[0].attrs["time"]), "x": x, "dx": dx}
    for name, sfx, ek in (("plasma", "", "energy_ion"), ("neutral", "_n", "energy_n")):
        rho = cat("rho" + sfx); e = cat(ek)
        vx = np.interp(x, x + dx/2, cat("vx" + sfx))   # vertex i is at x_i + dx/2 -> cell centres
        D[name] = (rho, vx, e)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir"); ap.add_argument("snap", type=int)
    ap.add_argument("--left", type=float, nargs=3, required=True); ap.add_argument("--right", type=float, nargs=3, required=True)
    ap.add_argument("--x0", type=float, default=0.5); ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--out"); ap.add_argument("--window", type=float, nargs=2)
    a = ap.parse_args(); g = a.gamma
    R = Riemann(np.array(a.left), np.array(a.right), g)
    D = load(a.datadir, a.snap); t = D["t"]; x = D["x"]; dx = D["dx"]
    S = R.right_shock_speed(); xs_ex = a.x0 + S*t
    rs = R.star_right()
    print(f"t = {t:.4f}, gamma = {g:.6g}, dx = {dx:g}: exact p* = {R.p:.6g}, u* = {R.u:.6g}, rho*R = {rs:.6g}, "
          f"right shock x = {xs_ex:.5f} (speed {S:.6g})")
    xc = a.x0 + R.u*t                                                     # contact
    for name in ("plasma", "neutral"):
        rho, u, e = D[name]; P = (g - 1)*rho*e
        sel = x > xc + 0.25*(xs_ex - xc)                                  # right of the contact
        gr = np.gradient(rho, x); gr[~sel] = 0
        i = np.argmin(gr) if rho[sel].max() > a.right[0] else np.argmax(np.abs(gr))   # steepest drop outward
        xs = x[i]
        pl = (x > xc + 0.2*(xs_ex - xc)) & (x < xc + 0.8*(xs_ex - xc))   # plateau between contact and shock
        up = (x > xs + 0.05*abs(xs_ex - a.x0) + 5*dx) & (x < xs + 0.3*abs(xs_ex - a.x0) + 10*dx)
        E = R.sample((x[pl] - a.x0)/t)
        s2 = dict(rho=rho[pl].mean(), u=u[pl].mean(), P=P[pl].mean()); s1 = dict(rho=rho[up].mean(), u=u[up].mean(), P=P[up].mean())
        sp = (s2["rho"]*s2["u"] - s1["rho"]*s1["u"])/(s2["rho"] - s1["rho"])
        def fl(s):
            w = s["u"] - sp; r = s["rho"]
            return np.array([r*w, r*w*w + s["P"], r*w*(g/(g - 1)*s["P"]/r + 0.5*w*w)])
        f1, f2 = fl(s1), fl(s2)
        print(f"  {name:7s}: shock x = {xs:.5f} ({(xs - a.x0)/(xs_ex - a.x0):.4f} of exact distance; {(xs - xs_ex)/dx:+.1f} cells) | "
              f"plateau rho {s2['rho']/E[0].mean() - 1:+.2%}, P {s2['P']/E[2].mean() - 1:+.2%}, "
              f"T {(s2['P']/s2['rho'])/(E[2]/E[0]).mean() - 1:+.2%}, u {s2['u']:+.4f} (exact {E[1].mean():+.4f}) | "
              f"RH (out-in)/in: mom {(f2[1] - f1[1])/abs(f1[1]):+.1e}, energy {(f2[2] - f1[2])/abs(f1[2]):+.2e}")
    if a.out:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xe = np.linspace(x.min(), x.max(), 4000); E = R.sample((xe - a.x0)/t)
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for name, st in (("plasma", dict(color="#3b6fd8", ls="--", lw=1.2)), ("neutral", dict(color="#e07b39", ls=":", lw=1.6))):
            rho, u, e = D[name]; P = (g - 1)*rho*e
            for k, (q, lab) in enumerate(((rho, "rho"), (u, "u"), (P, "P"), (P/rho, "P/rho"))):
                ax[k].plot(x, q, label=f"SAMS {name}", **st)
        for k, (q, lab) in enumerate(((E[0], "rho"), (E[1], "u"), (E[2], "P"), (E[2]/E[0], "P/rho"))):
            ax[k].plot(xe, q, "k", lw=0.8, label="exact"); ax[k].set_title(lab, loc="left"); ax[k].grid(alpha=0.3)
            if a.window: ax[k].set_xlim(*a.window)
        ax[0].legend(); fig.suptitle(f"pure hydro, t = {t:.3f}, dx = {dx:g}"); fig.tight_layout(); fig.savefig(a.out, dpi=100)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
