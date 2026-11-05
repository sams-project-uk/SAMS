/*
 *    Copyright 2025 SAMS Team
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
/*
 * Per-operator energy budget of the plasma (data.energy_budget; diagnostic, off by default).
 *
 * Totals are measured after every operator of the step and the change is charged to that operator:
 *   beforeStep      (Eulerian)  change since the previous step's afterRke -> source half-step 2 (src2)
 *   afterSource1    (Eulerian)  source half-step 1 (src1)
 *   afterCorrector  (Lagrangian) predictor + corrector
 *   afterRemapPass  x, y, z passes separately, in the order they ran
 *   afterRke                    remap kinetic-energy correction (zero when rke is off)
 *
 * Definitions (all per unit y-z area, so directly comparable with energy_budget.py and E0 = 5.8):
 *   internal  sum over cells of rho e V
 *   kinetic   sum over vertices of w m_v |v|^2 / 2, m_v = (1/8) sum of rho V over the 8 neighbouring
 *             cells (LARE's rho_v cv_v), w = 1/2 per domain-boundary direction. The vertex at the
 *             symmetry plane x = 0 is included (half mass): snapshots do not output it.
 *   magnetic  sum over cells of |B_c|^2 V / (2 mu0), B_c the average of the two face values.
 *   V is the cell volume of the state being measured: cv for an Eulerian state, cv1 for the
 *   Lagrangian state after the corrector, until the x-pass has run.
 *
 * LARE's Lagrangian step does not change the face-centred field: the whole induction equation is
 * applied in the remap (constrained transport of fluxes). For the Lagrangian state the field is
 * measured as LARE's remap sees it, flux / Lagrangian face area, so x-compression is counted in the
 * Lagrangian step and the remap passes are only charged for transport. The shear term (B_x dv_y/dx
 * creating B_y) is applied by the y-pass, so the Lorentz work along y (tension) done in the corrector
 * is paid back by the y-pass:  magnetic dissipation of the y-pass = -(dEmag_y + W_Ly).
 * This Lagrangian treatment of B assumes the flow varies in x only (1D, uniform in y and z).
 *
 * Exact identities checked every print (they hold to round-off if the code below is right):
 *   dEkin(Lagrangian) = W_p + W_visc + W_L   (corrector: dv = dt (f + f_visc) / rho_v and the mean of
 *                                             the old and new velocity is the predictor's vx1)
 *   dEint(Lagrangian) = Q_visc - PdV
 * Host-side loops: OpenMP builds only (as noiseCheck).
 */
#include "LARE3DSingleTemperature/shared_data.h"
#include <cstdio>
#include <cmath>

namespace LARE
{
    template<typename T_EOS>
    typename LARE3DST<T_EOS>::budgetTotals LARE3DST<T_EOS>::budgetMeasure(simulationData &data, bool fluxFormB,
                                                                          remapData *rd, double *keLare)
    {
        budgetTotals t;
        const T_indexType nx = data.nx, ny = data.ny, nz = data.nz;
        eb_profEmag.assign(static_cast<size_t>(nx + 2), 0.0);
        eb_profEkin.assign(static_cast<size_t>(nx + 2), 0.0);
        eb_profEint.assign(static_cast<size_t>(nx + 2), 0.0);
        const bool lagX = !eb_xdone;
        auto rx = [&](T_indexType ix) { return lagX ? eb_rx[static_cast<size_t>(ix + 2)] : 1.0; };
        auto vol = [&](T_indexType ix, T_indexType iy, T_indexType iz) { return data.cv(ix, iy, iz) * rx(ix); };

        for (T_indexType iz = 1; iz <= nz; ++iz)
            for (T_indexType iy = 1; iy <= ny; ++iy)
                for (T_indexType ix = 1; ix <= nx; ++ix) {
                    const double V = vol(ix, iy, iz);
                    t.eint += data.rho(ix, iy, iz) * data.energy_ion(ix, iy, iz) * V;
                    eb_profEint[static_cast<size_t>(ix)] += data.rho(ix, iy, iz) * data.energy_ion(ix, iy, iz) * V / eb_area;
                    double bxl = data.bx(ix - 1, iy, iz), bxh = data.bx(ix, iy, iz);
                    double byl = data.by(ix, iy - 1, iz), byh = data.by(ix, iy, iz);
                    double bzl = data.bz(ix, iy, iz - 1), bzh = data.bz(ix, iy, iz);
                    if (fluxFormB) {
                        // Inside the remap B holds face fluxes B * (Eulerian face area)
                        bxl /= data.dxab(ix - 1, iy, iz); bxh /= data.dxab(ix, iy, iz);
                        byl /= data.dyab(ix, iy - 1, iz); byh /= data.dyab(ix, iy, iz);
                        bzl /= data.dzab(ix, iy, iz - 1); bzh /= data.dzab(ix, iy, iz);
                    }
                    // Lagrangian state: y and z faces have been stretched in x by rx
                    const double r = rx(ix);
                    const double bx = 0.5 * (bxl + bxh), by = 0.5 * (byl + byh) / r, bz = 0.5 * (bzl + bzh) / r;
                    const double em = 0.5 * (bx * bx + by * by + bz * bz) * V / data.mu0;
                    t.emag += em;
                    eb_profEmag[static_cast<size_t>(ix)] += em / eb_area;
                }

        double kl = 0.0;
        for (T_indexType iz = 0; iz <= nz; ++iz)
            for (T_indexType iy = 0; iy <= ny; ++iy)
                for (T_indexType ix = 0; ix <= nx; ++ix) {
                    double w = 1.0;
                    if (ix == 0 || ix == nx) w *= 0.5;
                    if (iy == 0 || iy == ny) w *= 0.5;
                    if (iz == 0 || iz == nz) w *= 0.5;
                    double m = 0.0;
                    if (lagX) {
                        // Lagrangian state: vertex mass is invariant, use LARE's own (rho_v cv_v from
                        // lagrangian_prepare). Re-deriving it from cells would need the ghost cells' cv1,
                        // which the corrector does not update.
                        m = data.rho_v(ix, iy, iz) * data.cv_v(ix, iy, iz);
                    } else {
                        for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 2; ++c)
                            m += data.rho(ix + a, iy + b, iz + c) * vol(ix + a, iy + b, iz + c);
                        m *= 0.125;
                    }
                    const double v2 = data.vx(ix, iy, iz) * data.vx(ix, iy, iz) + data.vy(ix, iy, iz) * data.vy(ix, iy, iz)
                                    + data.vz(ix, iy, iz) * data.vz(ix, iy, iz);
                    t.ekin += 0.5 * w * m * v2;
                    eb_profEkin[static_cast<size_t>(ix)] += 0.5 * w * m * v2 / eb_area;
                    if (rd) kl += 0.5 * w * rd->rho_v1(ix, iy, iz) * rd->cv2(ix, iy, iz) * v2;
                }
        if (keLare) *keLare = kl / eb_area;
        t.eint /= eb_area; t.ekin /= eb_area; t.emag /= eb_area;
        return t;
    }

    template<typename T_EOS>
    void LARE3DST<T_EOS>::budgetStage(simulationData &data, budgetPoint point, int pass, remapData *rd)
    {
        const T_indexType nx = data.nx, ny = data.ny, nz = data.nz;
        if (!eb_started) {
            if (point != budgetPoint::beforeStep) return;
            eb_area = (data.yb(ny) - data.yb(0)) * (data.zb(nz) - data.zb(0));
            eb_rx.assign(static_cast<size_t>(nx + 5), 1.0);
            eb_cellEmagX.assign(static_cast<size_t>(nx + 2), 0.0);
            eb_cellEmagY.assign(static_cast<size_t>(nx + 2), 0.0);
            eb_vertEkinX.assign(static_cast<size_t>(nx + 2), 0.0);
            eb_xdone = true;
            eb_start = eb_last = budgetMeasure(data, false);
            eb_E0 = eb_start.total();
            eb_next_print = data.time + data.dt_snapshots;
            eb_started = true;
            printf("ENERGY BUDGET (plasma, per unit y-z area) started at t = %.6e: internal %.10e kinetic %.10e magnetic %.10e total %.10e\n",
                   static_cast<double>(data.time), eb_start.eint, eb_start.ekin, eb_start.emag, eb_E0);
            return;
        }

        // Per-x-index change of the last measurement, accumulated for the x and y passes
        auto accumulate = [&](std::vector<double> &acc, const std::vector<double> &prev, const std::vector<double> &now) {
            for (size_t i = 0; i < acc.size(); ++i) acc[i] += now[i] - prev[i];
        };

        switch (point) {
        case budgetPoint::beforeStep: {
            const budgetTotals t = budgetMeasure(data, false);
            eb.src2 += t.total() - eb_last.total();
            eb_last = t;
            break;
        }
        case budgetPoint::afterSource1: {
            const budgetTotals t = budgetMeasure(data, false);
            eb.src1 += t.total() - eb_last.total();
            eb_last = t;
            break;
        }
        case budgetPoint::afterCorrector: {
            const double dt = data.dt;
            // Work and heating terms of the corrector, before anything is overwritten by the remap
            double Wp = 0, Wv = 0, WL[3] = {0, 0, 0}, Q = 0, PdV = 0;
            for (T_indexType iz = 0; iz <= nz; ++iz)
                for (T_indexType iy = 0; iy <= ny; ++iy)
                    for (T_indexType ix = 0; ix <= nx; ++ix) {
                        const double w = ((ix == 0 || ix == nx) ? 0.5 : 1.0) * ((iy == 0 || iy == ny) ? 0.5 : 1.0) * ((iz == 0 || iz == nz) ? 0.5 : 1.0);
                        const T_indexType ixp = ix + 1, iyp = iy + 1, izp = iz + 1;
                        // Pressure force exactly as in predictor_step
                        const double pp = data.pressure(ix, iy, iz), ppx = data.pressure(ixp, iy, iz), ppy = data.pressure(ix, iyp, iz),
                                     ppz = data.pressure(ix, iy, izp), ppxy = data.pressure(ixp, iyp, iz), ppxz = data.pressure(ixp, iy, izp),
                                     ppyz = data.pressure(ix, iyp, izp), ppxyz = data.pressure(ixp, iyp, izp);
                        const double dhy = data.hyc(ix) * data.dyc(iy), dhz = data.hzc(ix, iy) * data.dzc(iz);
                        const double fpx = ((pp + ppy + ppz + ppyz) - (ppx + ppxy + ppxz + ppxyz)) * 0.25 / data.dxc(ix);
                        const double fpy = ((pp + ppx + ppz + ppxz) - (ppy + ppxy + ppyz + ppxyz)) * 0.25 / dhy;
                        const double fpz = ((pp + ppx + ppy + ppxy) - (ppz + ppxz + ppyz + ppxyz)) * 0.25 / dhz;
                        const double c = w * dt * data.cv_v(ix, iy, iz);
                        const double u = data.vx1(ix, iy, iz), v = data.vy1(ix, iy, iz), s = data.vz1(ix, iy, iz);
                        Wp += c * (fpx * u + fpy * v + fpz * s);
                        Wv += c * (data.fx_visc(ix, iy, iz) * u + data.fy_visc(ix, iy, iz) * v + data.fz_visc(ix, iy, iz) * s);
                        WL[0] += c * (data.fx(ix, iy, iz) - fpx) * u;
                        WL[1] += c * (data.fy(ix, iy, iz) - fpy) * v;
                        WL[2] += c * (data.fz(ix, iy, iz) - fpz) * s;
                    }
            for (T_indexType iz = 1; iz <= nz; ++iz)
                for (T_indexType iy = 1; iy <= ny; ++iy)
                    for (T_indexType ix = 1; ix <= nx; ++ix) {
                        Q += dt * data.visc_heat(ix, iy, iz) * data.cv(ix, iy, iz);
                        PdV += (data.cv1(ix, iy, iz) - data.cv(ix, iy, iz)) * data.pressure(ix, iy, iz);
                    }
            eb.Wp += Wp / eb_area; eb.Wvisc += Wv / eb_area; eb.Qvisc += Q / eb_area; eb.PdV += PdV / eb_area;
            for (int d = 0; d < 3; ++d) eb.WL[d] += WL[d] / eb_area;

            // Lagrangian x-width ratio of every cell (y- and z-uniform flow: one value per ix)
            for (T_indexType ix = -1; ix <= nx + 2; ++ix)
                eb_rx[static_cast<size_t>(ix + 2)] = data.cv1(ix, 1, 1) / data.cv(ix, 1, 1);
            eb_xdone = false;
            const std::vector<double> prevEint = eb_profEint;
            const budgetTotals t = budgetMeasure(data, false);
            {
                // Locate any cell where the internal energy change differs from the corrector's
                // cv (dt visc_heat - dv p): the identity dEint = Q_visc - PdV must hold cell by cell.
                double worst = 0.0; T_indexType wi = -1;
                for (T_indexType ix = 1; ix <= nx; ++ix) {
                    double expect = 0.0;
                    for (T_indexType iz = 1; iz <= nz; ++iz)
                        for (T_indexType iy = 1; iy <= ny; ++iy)
                            expect += (dt * data.visc_heat(ix, iy, iz) * data.cv(ix, iy, iz)
                                       - (data.cv1(ix, iy, iz) - data.cv(ix, iy, iz)) * data.pressure(ix, iy, iz)) / eb_area;
                    const double err = (eb_profEint[static_cast<size_t>(ix)] - prevEint[static_cast<size_t>(ix)]) - expect;
                    if (std::abs(err) > std::abs(worst)) { worst = err; wi = ix; }
                }
                if (std::abs(worst) > 1e-15 * std::abs(t.eint))
                    printf("ENERGY BUDGET identity: step %lld cell %lld dEint - (Q - PdV) = %+.3e\n",
                           static_cast<long long>(data.step), static_cast<long long>(wi), worst);
            }
            eb.lagEint += t.eint - eb_last.eint;
            eb.lagEkin += t.ekin - eb_last.ekin;
            eb.lagEmag += t.emag - eb_last.emag;
            eb_last = t;
            break;
        }
        case budgetPoint::afterRemapPass: {
            const std::vector<double> prevEmag = eb_profEmag, prevEkin = eb_profEkin;
            double keLare = 0.0;
            if (pass == 0) eb_xdone = true;
            const budgetTotals t = budgetMeasure(data, true, pass == 0 ? rd : nullptr, pass == 0 ? &keLare : nullptr);
            eb.passEint[pass] += t.eint - eb_last.eint;
            eb.passEkin[pass] += t.ekin - eb_last.ekin;
            eb.passEmag[pass] += t.emag - eb_last.emag;
            if (pass == 0) {
                eb.xMassRedef += t.ekin - keLare;
                accumulate(eb_cellEmagX, prevEmag, eb_profEmag);
                accumulate(eb_vertEkinX, prevEkin, eb_profEkin);
            } else if (pass == 1) {
                accumulate(eb_cellEmagY, prevEmag, eb_profEmag);
            }
            eb_last = t;
            break;
        }
        case budgetPoint::afterRke: {
            const budgetTotals t = budgetMeasure(data, false);
            eb.rke += t.total() - eb_last.total();
            eb_last = t;
            eb.steps++;
            if (data.time + data.dt >= eb_next_print * (1.0 - 1e-12) ||
                (data.energy_budget_every > 0 && eb.steps % data.energy_budget_every == 0)) {
                budgetPrint(data);
                while (eb_next_print <= data.time + data.dt) eb_next_print += data.dt_snapshots;
            }
            break;
        }
        }
    }

    template<typename T_EOS>
    void LARE3DST<T_EOS>::budgetPrint(simulationData &data)
    {
        const budgetTotals &t = eb_last;
        const double E = t.total();
        const double relMag = eb_start.emag - t.emag; // magnetic energy released so far
        const double lagErr = eb.lagEint + eb.lagEkin + eb.lagEmag - eb.WL[1] - eb.WL[2];
        const double sumParts = eb.src1 + eb.src2 + (eb.lagEint + eb.lagEkin + eb.lagEmag)
                              + eb.passEint[0] + eb.passEkin[0] + eb.passEmag[0]
                              + eb.passEint[1] + eb.passEkin[1] + eb.passEmag[1]
                              + eb.passEint[2] + eb.passEkin[2] + eb.passEmag[2] + eb.rke;
        auto pc = [&](double x) { return relMag != 0.0 ? 100.0 * x / relMag : 0.0; };
        printf("\nENERGY BUDGET t = %.6e after %lld steps (plasma, per unit y-z area; %% of magnetic energy released = %.6e)\n",
               static_cast<double>(data.time + data.dt), static_cast<long long>(eb.steps), relMag);
        printf("  now: internal %.8e kinetic %.8e magnetic %.8e total %.10e   E - E0 = %+.6e (%+.2f%%)\n",
               t.eint, t.ekin, t.emag, E, E - eb_E0, pc(E - eb_E0));
        printf("  change since start: internal %+.6e kinetic %+.6e magnetic %+.6e\n",
               t.eint - eb_start.eint, t.ekin - eb_start.ekin, t.emag - eb_start.emag);
        printf("  sources            : half-step 1 %+.6e, half-step 2 %+.6e\n", eb.src1, eb.src2);
        printf("  Lagrangian step    : dEint %+.6e dEkin %+.6e dEmag(x-compression) %+.6e\n", eb.lagEint, eb.lagEkin, eb.lagEmag);
        printf("    work             : W_p %+.6e W_visc %+.6e W_Lorentz x %+.6e y %+.6e z %+.6e\n",
               eb.Wp, eb.Wvisc, eb.WL[0], eb.WL[1], eb.WL[2]);
        printf("    heating          : Q_visc %+.6e  PdV %+.6e\n", eb.Qvisc, eb.PdV);
        printf("    identities       : dEkin - sum(W) = %+.3e   dEint - (Q_visc - PdV) = %+.3e  (both should be round-off)\n",
               eb.lagEkin - (eb.Wp + eb.Wvisc + eb.WL[0] + eb.WL[1] + eb.WL[2]), eb.lagEint - (eb.Qvisc - eb.PdV));
        printf("    errors           : pressure W_p - PdV = %+.6e (%+.2f%%)  viscous W_visc + Q_visc = %+.6e (%+.2f%%)\n",
               eb.Wp - eb.PdV, pc(eb.Wp - eb.PdV), eb.Wvisc + eb.Qvisc, pc(eb.Wvisc + eb.Qvisc));
        printf("                       x-magnetic dEmag + W_Lx = %+.6e (%+.2f%%)   Lagrangian step total (less y/z tension work) = %+.6e (%+.2f%%)\n",
               eb.lagEmag + eb.WL[0], pc(eb.lagEmag + eb.WL[0]), lagErr, pc(lagErr));
        const char *nm[3] = {"x", "y", "z"};
        for (int d = 0; d < 3; ++d) {
            const double magLoss = eb.passEmag[d] + (d > 0 ? eb.WL[d] : 0.0);
            printf("  remap %s-pass       : dEint %+.6e dEkin %+.6e dEmag %+.6e   magnetic error%s %+.6e (%+.2f%%)  total %+.6e (%+.2f%%)\n",
                   nm[d], eb.passEint[d], eb.passEkin[d], eb.passEmag[d], d > 0 ? " (dEmag + W_L)" : "             ", magLoss, pc(magLoss),
                   eb.passEint[d] + eb.passEkin[d] + magLoss, pc(eb.passEint[d] + eb.passEkin[d] + magLoss));
        }
        printf("    x-pass dEkin from re-deriving vertex mass after the pass: %+.6e (%+.2f%%)\n", eb.xMassRedef, pc(eb.xMassRedef));
        auto band = [&](const std::vector<double> &v, size_t lo, size_t hi) { double x = 0; for (size_t i = lo; i <= hi && i < v.size(); ++i) x += v[i]; return x; };
        const size_t n = static_cast<size_t>(data.nx);
        printf("  where (x index; cells 1..nx, vertices 0..nx; vertex 0 is x = 0):\n");
        printf("    x-pass dEmag     : cells 1-2 %+.4e  3-10 %+.4e  11-50 %+.4e  51-nx %+.4e\n",
               band(eb_cellEmagX, 1, 2), band(eb_cellEmagX, 3, 10), band(eb_cellEmagX, 11, 50), band(eb_cellEmagX, 51, n));
        printf("    y-pass dEmag     : cells 1-2 %+.4e  3-10 %+.4e  11-50 %+.4e  51-nx %+.4e\n",
               band(eb_cellEmagY, 1, 2), band(eb_cellEmagY, 3, 10), band(eb_cellEmagY, 11, 50), band(eb_cellEmagY, 51, n));
        printf("    x-pass dEkin     : vertex 0 %+.4e  1-2 %+.4e  3-10 %+.4e  11-50 %+.4e  51-nx %+.4e\n",
               band(eb_vertEkinX, 0, 0), band(eb_vertEkinX, 1, 2), band(eb_vertEkinX, 3, 10), band(eb_vertEkinX, 11, 50), band(eb_vertEkinX, 51, n));
        printf("    kinetic energy now at vertex 0 (half mass, x = 0): %.6e of %.6e\n", eb_profEkin.empty() ? 0.0 : eb_profEkin[0], t.ekin);
        printf("  rke correction (and end-of-remap field BCs): %+.6e\n", eb.rke);
        printf("  check: sum of all parts - (E - E0) = %+.3e\n", sumParts - (E - eb_E0));
        std::fflush(stdout);
    }

    // Instantiated through the explicit class instantiation at the end of shared_data.h, as the other LARE3DST files
}
