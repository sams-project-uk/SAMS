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
#include "LARE3DNeutralFluid/shared_data.h"

namespace LARE
{

    namespace{

        using T_dataType = SAMS::T_dataType;

    template<typename T_EOS>
    DEVICEPREFIX INLINE T_dataType edge_viscosity(const typename LARE3DNF<T_EOS>::simulationData &data,
                                                  T_dataType dvdots, T_dataType dx, T_dataType dxm, T_dataType dxp, T_dataType cs_edge,
                                                  int i0, int i1, int i2, int i3,
                                                  int j0, int j1, int j2, int j3,
                                                  int k0, int k1, int k2, int k3)
    {
        dvdots = pw::min(0.0, dvdots);
        T_dataType rho_edge = 2.0 * data.rho_v(i1, j1, k1) * data.rho_v(i2, j2, k2) / (data.rho_v(i1, j1, k1) + data.rho_v(i2, j2, k2));

        T_dataType dvx = data.vx(i1, j1, k1) - data.vx(i2, j2, k2);
        T_dataType dvy = data.vy(i1, j1, k1) - data.vy(i2, j2, k2);
        T_dataType dvz = data.vz(i1, j1, k1) - data.vz(i2, j2, k2);
        T_dataType dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
        T_dataType dv = std::sqrt(dv2);

        T_dataType psi = 0.0;
        dvdots = dv * data.dt / dx < 1.e-14 ? 0.0 : dvdots / dv;

        T_dataType dvxm = data.vx(i0, j0, k0) - data.vx(i1, j1, k1);
        T_dataType dvxp = data.vx(i2, j2, k2) - data.vx(i3, j3, k3);
        T_dataType dvym = data.vy(i0, j0, k0) - data.vy(i1, j1, k1);
        T_dataType dvyp = data.vy(i2, j2, k2) - data.vy(i3, j3, k3);
        T_dataType dvzm = data.vz(i0, j0, k0) - data.vz(i1, j1, k1);
        T_dataType dvzp = data.vz(i2, j2, k2) - data.vz(i3, j3, k3);

        T_dataType rl = 1.0, rr = 1.0;

        rl = dv * data.dt / dx < 1.e-14 ? 1.0 : (dvxp * dvx + dvyp * dvy + dvzp * dvz) * dx / (dxp * dv2);
        rr = dv * data.dt / dx < 1.e-14 ? 1.0 : (dvxm * dvx + dvym * dvy + dvzm * dvz) * dx / (dxm * dv2);

        psi = pw::min({0.5 * (rr + rl), 2.0 * rl, 2.0 * rr, 1.0});
        psi = pw::max(0.0, psi);

        // Find q_kur / abs(dv)
        T_dataType q_k_bar = rho_edge *
                             (data.visc2_norm * dv + std::sqrt(data.visc2_norm * data.visc2_norm * dv2 + (data.visc1 * cs_edge) * (data.visc1 * cs_edge)));
        return q_k_bar * (1.0 - psi) * dvdots;
    }
}

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::lagrangian_step(simulationData &data, const LARE3DST<T_EOS>::simulationData & core_data, SAMS::controlFunctions &controlFns)
    {   
        using Range = pw::Range;

        Range xbp = pw::Range(-1, data.nx + 1);
        Range ybp = pw::Range(-1, data.ny + 1);
        Range zbp = pw::Range(-1, data.nz + 1);

        // All of the arrays are deallocated when lagranManager goes out of scope
        //  Initialize p_i, pressure
        volumeArray cvl = data.cv;
        volumeArray energy_il = data.energy;

        pw::applyKernel(LAMBDA(auto ix, auto iy, auto iz) {
            auto &d = data;

            if constexpr (std::is_invocable_v<decltype(&T_EOS::getPressure), T_EOS, eosDensity, eosEnergy>)
            {
                d.pressure(ix, iy, iz) = d.eos.getPressure(eosDensity(d.rho(ix, iy, iz)), eosEnergy(energy_il(ix, iy, iz)));
            } else if constexpr (std::is_invocable_v<decltype(&T_EOS::getPressure), T_EOS, eosDensity, eosEnergy, eosIndex, eosIndex, eosIndex>)
            {
                d.pressure(ix, iy, iz) = d.eos.getPressure(eosDensity(d.rho(ix, iy, iz)), eosEnergy(energy_il(ix, iy, iz)), eosIndex(ix), eosIndex(iy), eosIndex(iz));
            } else
            {
                static_assert(pw::alwaysFalse<T_dataType>::value, "Unsupported EOS pressure interface");
            }
        },
                        data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);

        // Compute rho_v and cv_v
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType izp = iz + 1;
            T_indexType iyp = iy + 1;
            T_indexType ixp = ix + 1;

            T_dataType sum_rho_cv = data.rho(ix, iy, iz) * cvl(ix, iy, iz) +
                                    data.rho(ixp, iy, iz) * cvl(ixp, iy, iz) +
                                    data.rho(ix, iyp, iz) * cvl(ix, iyp, iz) +
                                    data.rho(ixp, iyp, iz) * cvl(ixp, iyp, iz) +
                                    data.rho(ix, iy, izp) * cvl(ix, iy, izp) +
                                    data.rho(ixp, iy, izp) * cvl(ixp, iy, izp) +
                                    data.rho(ix, iyp, izp) * cvl(ix, iyp, izp) +
                                    data.rho(ixp, iyp, izp) * cvl(ixp, iyp, izp);

            T_dataType sum_cv = cvl(ix, iy, iz) +
                                cvl(ixp, iy, iz) +
                                cvl(ix, iyp, iz) +
                                cvl(ixp, iyp, iz) +
                                cvl(ix, iy, izp) +
                                cvl(ixp, iy, izp) +
                                cvl(ix, iyp, izp) +
                                cvl(ixp, iyp, izp);
            data.rho_v(ix, iy, iz) = sum_rho_cv / sum_cv;
            data.cv_v(ix, iy, iz) = 0.125 * sum_cv; // Assuming a constant factor for control volume
        },
                        xbp, ybp, zbp);

        shock_viscosity(data, core_data);
        controlFns.calculateTimestep();

        this->predictor_step(data, core_data);
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::shock_viscosity(simulationData &data, const LARE3DST<T_EOS>::simulationData & core_data)
    {
        using Range = pw::Range;
        data.visc2_norm = 0.25 * (data.gas_gamma + 1.0) * data.visc2;
        pw::portableArrayManager svManager;
        // Temporary arrays for sound speed
        pw::portableArray<T_dataType, 3> cs, cs_v;
        Range xp1 = pw::Range(-1, data.nx + 1);
        Range yp1 = pw::Range(-1, data.ny + 1);
        Range zp1 = pw::Range(-1, data.nz + 1);
        svManager.allocate(cs, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        svManager.allocate(cs_v, xp1, yp1, zp1);

        pw::assign(data.p_visc, 0.0);
        pw::assign(data.visc_heat, 0.0);

        // Compute cs
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_dataType rmin = pw::max(data.rho(ix, iy, iz), data.none_zero);
        T_dataType p = data.pressure(ix, iy, iz);
        cs(ix, iy, iz) = std::sqrt((data.gas_gamma * p) / rmin); }, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        pw::fence();
        // Compute cs_v
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izp = iz + 1;
        T_indexType iyp = iy + 1;
        T_indexType ixp = ix + 1;

        T_dataType sum =
            cs(ix, iy, iz) * data.cv(ix, iy, iz) +
            cs(ixp, iy, iz) * data.cv(ixp, iy, iz) +
            cs(ix, iyp, iz) * data.cv(ix, iyp, iz) +
            cs(ixp, iyp, iz) * data.cv(ixp, iyp, iz) +
            cs(ix, iy, izp) * data.cv(ix, iy, izp) +
            cs(ixp, iy, izp) * data.cv(ixp, iy, izp) +
            cs(ix, iyp, izp) * data.cv(ix, iyp, izp) +
            cs(ixp, iyp, izp) * data.cv(ixp, iyp, izp);
        cs_v(ix, iy, iz) = 0.125 * sum / data.cv_v(ix, iy, iz); }, xp1, yp1, zp1);
        pw::fence();

        // alpha1
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            T_indexType iym = iy - 1;
            T_indexType izm = iz - 1;
            T_indexType ixp = ix + 1;

            T_indexType i1 = ixm, j1 = iym, k1 = izm;
            T_indexType i2 = ix, j2 = iym, k2 = izm;
            T_indexType i0 = i1 - 1, j0 = j1, k0 = k1;
            T_indexType i3 = i2 + 1, j3 = j2, k3 = k2;

            T_dataType dx = core_data.dxb(ix);
            T_dataType dxp = core_data.dxb(ixp);
            T_dataType dxm = core_data.dxb(ixm);
            T_dataType dvdots = -(data.vx(i1, j1, k1) - data.vx(i2, j2, k2));
            T_dataType cs_edge = pw::min(cs_v(i1, j1, k1), cs_v(i2, j2, k2));
            // Edge viscosities from Caramana
            data.alpha1(ix, iy, iz) = edge_viscosity<T_EOS>(data,
                                                       dvdots, dx, dxm, dxp, cs_edge,
                                                       i0, i1, i2, i3, j0, j1, j2, j3, k0, k1, k2, k3);
        },
                        Range(0, data.nx + 1), Range(0, data.ny + 2), Range(0, data.nz + 2));

        // alpha2
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        T_indexType iym = iy - 1, iyp = iy + 1;
        T_indexType i1 = ix, j1 = iym, k1 = izm;
        T_indexType i2 = ix, j2 = iy, k2 = izm;
        T_indexType i0 = i1, j0 = j1 - 1, k0 = k1;
        T_indexType i3 = i2, j3 = j2 + 1, k3 = k2;

        T_dataType dx = core_data.dyb(iy) * core_data.hy(ix);
        T_dataType dxp = core_data.dyb(iyp) * core_data.hy(ix);
        T_dataType dxm = core_data.dyb(iym) * core_data.hy(ix);
        T_dataType dvdots = -(data.vy(i1, j1, k1) - data.vy(i2, j2, k2));
        T_dataType cs_edge = pw::min(cs_v(i1, j1, k1), cs_v(i2, j2, k2));
        data.alpha2(ix, iy, iz) = edge_viscosity<T_EOS>(data,
            dvdots, dx, dxm, dxp,
            cs_edge,
            i0, i1, i2, i3, j0, j1, j2, j3, k0, k1, k2, k3); }, Range(-1, data.nx + 1), Range(0, data.ny + 1), Range(0, data.nz + 2));

        // alpha3
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1, izp = iz + 1;

        T_indexType i1 = ix, j1 = iy, k1 = izm;
        T_indexType i2 = ix, j2 = iy, k2 = iz;
        T_indexType i0 = i1, j0 = j1, k0 = k1 - 1;
        T_indexType i3 = i2, j3 = j2, k3 = k2 + 1;

        T_dataType dx = core_data.dzb(iz) * core_data.hz(ix, iy);
        T_dataType dxp = core_data.dzb(izp) * core_data.hz(ix, iy);
        T_dataType dxm = core_data.dzb(izm) * core_data.hz(ix, iy);
        T_dataType dvdots = -(data.vz(i1, j1, k1) - data.vz(i2, j2, k2));
        T_dataType cs_edge = pw::min(cs_v(i1, j1, k1), cs_v(i2, j2, k2));
        data.alpha3(ix, iy, iz) = edge_viscosity<T_EOS>(data,
            dvdots, dx, dxm, dxp,
            cs_edge,
            i0, i1, i2, i3, j0, j1, j2, j3,
            k0, k1, k2, k3); }, Range(-1, data.nx + 1), Range(-1, data.ny + 1), Range(0, data.nz + 1));

        pw::fence();

        // p_visc and visc_heat
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType izm = iz - 1, izp = iz + 1;
            T_indexType iym = iy - 1, iyp = iy + 1;
            T_indexType ixm = ix - 1;
            T_dataType a1 = pow(data.vx(ixm, iym, izm) - data.vx(ix, iym, izm), 2) + pow(data.vy(ixm, iym, izm) - data.vy(ix, iym, izm), 2) + pow(data.vz(ixm, iym, izm) - data.vz(ix, iym, izm), 2);
            T_dataType a2 = pow(data.vx(ix,  iym, izm) - data.vx(ix, iy, izm), 2) + pow(data.vy(ix, iym, izm) - data.vy(ix, iy, izm), 2) + pow(data.vz(ix, iym, izm) - data.vz(ix, iy, izm), 2);
            T_dataType a3 = pow(data.vx(ix,  iy, izm) - data.vx(ixm, iy, izm), 2) + pow(data.vy(ix, iy, izm) - data.vy(ixm, iy, izm), 2) + pow(data.vz(ix, iy, izm) - data.vz(ixm, iy, izm), 2);
            T_dataType a4 = pow(data.vx(ixm, iy, izm) - data.vx(ixm, iym, izm), 2) + pow(data.vy(ixm, iy, izm) - data.vy(ixm, iym, izm), 2) + pow(data.vz(ixm, iy, izm) - data.vz(ixm, iym, izm), 2);

            T_dataType a5 = pow(data.vx(ixm, iym, iz) - data.vx(ix, iym, iz), 2) + pow(data.vy(ixm, iym, iz) - data.vy(ix, iym, iz), 2) + pow(data.vz(ixm, iym, iz) - data.vz(ix, iym, iz), 2);
            T_dataType a6 = pow(data.vx(ix, iym, iz) - data.vx(ix, iy, iz), 2) + pow(data.vy(ix, iym, iz) - data.vy(ix, iy, iz), 2) + pow(data.vz(ix, iym, iz) - data.vz(ix, iy, iz), 2);
            T_dataType a7 = pow(data.vx(ix, iy, iz) - data.vx(ixm, iy, iz), 2) + pow(data.vy(ix, iy, iz) - data.vy(ixm, iy, iz), 2) + pow(data.vz(ix, iy, iz) - data.vz(ixm, iy, iz), 2);
            T_dataType a8 = pow(data.vx(ixm, iy, iz) - data.vx(ixm, iym, iz), 2) + pow(data.vy(ixm, iy, iz) - data.vy(ixm, iym, iz), 2) + pow(data.vz(ixm, iy, iz) - data.vz(ixm, iym, iz), 2);

            T_dataType a9 = pow(data.vx(ix, iy, izm) - data.vx(ix, iy, iz), 2) + pow(data.vy(ix, iy, izm) - data.vy(ix, iy, iz), 2) + pow(data.vz(ix, iy, izm) - data.vz(ix, iy, iz), 2);
            T_dataType a10 = pow(data.vx(ixm, iy, izm) - data.vx(ixm, iy, iz), 2) + pow(data.vy(ixm, iy, izm) - data.vy(ixm, iy, iz), 2) + pow(data.vz(ixm, iy, izm) - data.vz(ixm, iy, iz), 2);
            T_dataType a11 = pow(data.vx(ixm, iym, izm) - data.vx(ixm, iym, iz), 2) + pow(data.vy(ixm, iym, izm) - data.vy(ixm, iym, iz), 2) + pow(data.vz(ixm, iym, izm) - data.vz(ixm, iym, iz), 2);
            T_dataType a12 = pow(data.vx(ix, iym, izm) - data.vx(ix, iym, iz), 2) + pow(data.vy(ix, iym, izm) - data.vy(ix, iym, iz), 2) + pow(data.vz(ix, iym, izm) - data.vz(ix, iym, iz), 2);

            data.p_visc(ix, iy, iz) = pw::max(data.p_visc(ix, iy, iz), -data.alpha1(ix, iy, iz) * std::sqrt(a1));
            data.p_visc(ix, iy, iz) = pw::max(data.p_visc(ix, iy, iz), -data.alpha2(ix, iy, iz) * std::sqrt(a2));
            data.p_visc(ix, iy, iz) = pw::max(data.p_visc(ix, iy, iz), -data.alpha3(ix, iy, iz) * std::sqrt(a9));

            T_dataType dx = core_data.dxb(ix);
            T_dataType dy = core_data.dyb(iy) * core_data.hyc(ix);
            T_dataType dz = core_data.dzb(iz) * core_data.hz2(ix, iy);

            data.visc_heat(ix, iy, iz) =
                -0.25 * dy * dz * data.alpha1(ix,  iy,  iz) * a1 
                -0.25 * dx * dz * data.alpha2(ix,  iy,  iz) * a2 
                -0.25 * dy * dz * data.alpha1(ix,  iyp, iz) * a3 
                -0.25 * dx * dz * data.alpha2(ixm, iy,  iz) * a4 
                -0.25 * dy * dz * data.alpha1(ix,  iy,  izp) * a5 
                -0.25 * dx * dz * data.alpha2(ix,  iy,  izp) * a6 
                -0.25 * dy * dz * data.alpha1(ix,  iyp, izp) * a7 
                -0.25 * dx * dz * data.alpha2(ixm, iy,  izp) * a8 
                -0.25 * dx * dy * data.alpha3(ix,  iy,  iz) * a9 
                -0.25 * dx * dy * data.alpha3(ixm, iy,  iz) * a10 
                -0.25 * dx * dy * data.alpha3(ixm, iym, iz) * a11 
                -0.25 * dx * dy * data.alpha3(ix,  iym, iz) * a12;

            data.visc_heat(ix, iy, iz) = data.visc_heat(ix, iy, iz) / data.cv(ix, iy, iz);
        },
                        Range(0, data.nx + 1), Range(0, data.ny + 1), Range(0, data.nz + 1));

        pw::assign(data.fx_visc, 0.0);
        pw::assign(data.fy_visc, 0.0);
        pw::assign(data.fz_visc, 0.0);

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType izm = iz - 1, izp = iz + 1;
            T_indexType iym = iy - 1, iyp = iy + 1;
            T_indexType ixm = ix - 1, ixp = ix + 1;

            T_dataType dx = core_data.dxb(ix);
            T_dataType dy = core_data.dyb(iy) * core_data.hyc(ix);
            T_dataType dz = core_data.dzb(iz) * core_data.hz2(ix, iy);

            T_dataType a1 = data.alpha1(ix, iyp, izp) * dy * dz;
            T_dataType a2 = data.alpha1(ixp, iyp, izp) * dy * dz;
            T_dataType a3 = data.alpha2(ix, iy, izp) * dx * dz;
            T_dataType a4 = data.alpha2(ix, iyp, izp) * dx * dz;
            T_dataType a5 = data.alpha3(ix, iy, iz) * dx * dy;
            T_dataType a6 = data.alpha3(ix, iy, izp) * dx * dy;

            data.fx_visc(ix, iy, iz) =
                (a1 * (data.vx(ix, iy, iz) - data.vx(ixm, iy, iz)) +
                 a2 * (data.vx(ix, iy, iz) - data.vx(ixp, iy, iz)) +
                 a3 * (data.vx(ix, iy, iz) - data.vx(ix, iym, iz)) +
                 a4 * (data.vx(ix, iy, iz) - data.vx(ix, iyp, iz)) +
                 a5 * (data.vx(ix, iy, iz) - data.vx(ix, iy, izm)) +
                 a6 * (data.vx(ix, iy, iz) - data.vx(ix, iy, izp))) /
                data.cv_v(ix, iy, iz);

            data.fy_visc(ix, iy, iz) =
                (a1 * (data.vy(ix, iy, iz) - data.vy(ixm, iy, iz)) +
                 a2 * (data.vy(ix, iy, iz) - data.vy(ixp, iy, iz)) +
                 a3 * (data.vy(ix, iy, iz) - data.vy(ix, iym, iz)) +
                 a4 * (data.vy(ix, iy, iz) - data.vy(ix, iyp, iz)) +
                 a5 * (data.vy(ix, iy, iz) - data.vy(ix, iy, izm)) +
                 a6 * (data.vy(ix, iy, iz) - data.vy(ix, iy, izp))) /
                data.cv_v(ix, iy, iz);

            data.fz_visc(ix, iy, iz) =
                (a1 * (data.vz(ix, iy, iz) - data.vz(ixm, iy, iz)) +
                 a2 * (data.vz(ix, iy, iz) - data.vz(ixp, iy, iz)) +
                 a3 * (data.vz(ix, iy, iz) - data.vz(ix, iym, iz)) +
                 a4 * (data.vz(ix, iy, iz) - data.vz(ix, iyp, iz)) +
                 a5 * (data.vz(ix, iy, iz) - data.vz(ix, iy, izm)) +
                 a6 * (data.vz(ix, iy, iz) - data.vz(ix, iy, izp))) /
                data.cv_v(ix, iy, iz);
        },
                        Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::set_dt(simulationData &data, const LARE3DST<T_EOS>::simulationData & core_data)
    {
        using Range = pw::Range;

        int i0 = core_data.geometry == geometryType::Cartesian ? 0 : 1;

        // Now need to do a map and reduction
        T_dataType dt1 = data.dt_multiplier *
                  pw::applyReduction(LAMBDA(auto ix, auto iy, auto iz) {
        T_indexType izm = iz - 1;
        T_indexType iym = iy - 1;
        T_indexType ixm = ix - 1;
        T_dataType dx = core_data.dxb(ix);
        T_dataType dy = core_data.dyb(iy);
        T_dataType dz = core_data.dzb(iz);

        T_dataType dhx = dx;
        T_dataType dhy = dy * core_data.hyc(ix);
        T_dataType dhz = dz * core_data.hzc(ix, iy);

        T_dataType rho0 = pw::max(data.rho(ix, iy, iz), data.none_zero);

        T_dataType cs2;

        if constexpr (std::is_invocable_v<decltype(&T_EOS::getSoundSpeedSquared), T_EOS, eosDensity, eosEnergy>)
        {
            cs2 = data.eos.getSoundSpeedSquared(eosDensity(data.rho(ix, iy, iz)), eosEnergy(data.energy(ix, iy, iz)));
        } else if constexpr (std::is_invocable_v<decltype(&T_EOS::getSoundSpeedSquared), T_EOS, eosDensity, eosEnergy, eosIndex, eosIndex, eosIndex>)
            cs2 = data.eos.getSoundSpeedSquared(eosDensity(data.rho(ix, iy, iz)), eosEnergy(data.energy(ix, iy, iz) ), eosIndex(ix), eosIndex(iy), eosIndex(iz));
         else
        {
            static_assert(pw::alwaysFalse<T_dataType>::value, "Unsupported EOS sound speed interface");
        }

        T_dataType c_visc2 = data.p_visc(ix, iy, iz) / rho0;

        T_dataType length  = pw::min(dhx, dhy, dhz);

        T_dataType dt1  = length / (std::sqrt(c_visc2) + std::sqrt(cs2 + c_visc2));


        T_dataType ax = 0.25 * core_data.dxab(ix,iy,iz);
        T_dataType ay = 0.25 * core_data.dyab(ix,iy,iz);
        T_dataType az = 0.25 * core_data.dzab(ix,iy,iz);

        T_dataType vxbm = (data.vx(ixm,iy ,iz ) + data.vx(ixm,iym,iz ) + data.vx(ixm,iy ,izm) + data.vx(ixm,iym,izm)) * ax;
        T_dataType vxbp = (data.vx(ix ,iy ,iz ) + data.vx(ix ,iym,iz ) + data.vx(ix ,iy ,izm) + data.vx(ix ,iym,izm)) * ax;
        T_dataType vybm = (data.vy(ix ,iym,iz ) + data.vy(ixm,iym,iz ) + data.vy(ix ,iym,izm) + data.vy(ixm,iym,izm)) * ay;
        T_dataType vybp = (data.vy(ix ,iy ,iz ) + data.vy(ixm,iy ,iz ) + data.vy(ix ,iy ,izm) + data.vy(ixm,iy ,izm)) * ay;
        T_dataType vzbm = (data.vz(ix ,iy ,izm) + data.vz(ixm,iy ,izm) + data.vz(ix ,iym,izm) + data.vz(ixm,iym,izm)) * az;
        T_dataType vzbp = (data.vz(ix ,iy ,iz ) + data.vz(ixm,iy ,iz ) + data.vz(ix ,iym,iz ) + data.vz(ixm,iym,iz )) * az;
        T_dataType dvx = abs(vxbp - vxbm);
        T_dataType dvy = abs(vybp - vybm);
        T_dataType dvz = abs(vzbp - vzbm);
        T_dataType avxm = abs(vxbm);
        T_dataType avxp = abs(vxbp);
        T_dataType avym = abs(vybm);
        T_dataType avyp = abs(vybp);
        T_dataType avzm = abs(vzbm);
        T_dataType avzp = abs(vzbp);

        T_dataType volume = data.cv(ix,iy,iz);
        T_dataType dt2 = volume / pw::max(avxm, avxp, dvx, 1.0e-10 * volume);
        T_dataType dt3 = volume / pw::max(avym, avyp, dvy, 1.0e-10 * volume);
        T_dataType dt4 = volume / pw::max(avzm, avzp, dvz, 1.0e-10 * volume);

        return pw::min(dt1, dt2, dt3, dt4);
        }, LAMBDA(T_dataType & a, const T_dataType &b) { a = pw::min(a, b); }, data.largest_number, Range(i0, data.nx), Range(0, data.ny), Range(0, data.nz));

        data.dt = dt1;
        data.dtr = data.dt;
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::predictor_step(simulationData &data, const LARE3DST<T_EOS>::simulationData & core_data)
    {
        using Range = pw::Range;

        // Predictor step for energy and pressure
        pw::applyKernel(LAMBDA(auto ix, auto iy, auto iz) {
            T_dataType dv = data.cv1(ix, iy, iz) / data.cv(ix, iy, iz) - 1.0;
            T_dataType e1_i = data.energy(ix, iy, iz) - data.pressure(ix, iy, iz) * dv / data.rho(ix, iy, iz);
            e1_i += data.visc_heat(ix, iy, iz) * data.dt / 2.0 / data.rho(ix, iy, iz);

            if constexpr(std::is_invocable_v<decltype(&T_EOS::getPressure), T_EOS, eosDensity, eosEnergy>){
                data.pressure(ix,iy,iz) = data.eos.getPressure(eosDensity(data.rho(ix,iy,iz)*data.cv(ix,iy,iz)/data.cv1(ix,iy,iz)), eosEnergy(e1_i));
            } else if constexpr(std::is_invocable_v<decltype(&T_EOS::getPressure), T_EOS, eosDensity, eosEnergy, eosIndex, eosIndex, eosIndex>){
                data.pressure(ix,iy,iz) = data.eos.getPressure(eosDensity(data.rho(ix,iy,iz)*data.cv(ix,iy,iz)/data.cv1(ix,iy,iz)), eosEnergy(e1_i), eosIndex(ix), eosIndex(iy), eosIndex(iz));
            } else {
                static_assert(pw::alwaysFalse<T_dataType>::value, "Unsupported EOS getPressure interface");
            }
        },
                        Range(0, data.nx + 1), Range(0, data.ny + 1), Range(0, data.nz + 1));

        pw::fence();

        // Compute forces and currents
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType izp = iz + 1;
            T_indexType iyp = iy + 1;
            T_indexType ixp = ix + 1;
            T_dataType dx = core_data.dxc(ix);
            T_dataType dy = core_data.dyc(iy);
            T_dataType dz = core_data.dzc(iz);

            T_dataType h2c = core_data.hyc(ix), h3c = core_data.hzc(ix, iy);
            T_dataType dhy = h2c * dy, dhz = h3c * dz;

            T_dataType pp = data.pressure(ix, iy, iz);
            T_dataType ppx = data.pressure(ixp, iy, iz);
            T_dataType ppy = data.pressure(ix, iyp, iz);
            T_dataType ppz = data.pressure(ix, iy, izp);
            T_dataType ppxy = data.pressure(ixp, iyp, iz);
            T_dataType ppxz = data.pressure(ixp, iy, izp);
            T_dataType ppyz = data.pressure(ix, iyp, izp);
            T_dataType ppxyz = data.pressure(ixp, iyp, izp);

            // Add pressure gradient force
            T_dataType w1 = pp + ppy + ppz + ppyz;
            T_dataType w2 = ppx + ppxy + ppxz + ppxyz;
            data.fx(ix, iy, iz) = (w1 - w2) * 0.25 / dx;

            w1 = pp + ppx + ppz + ppxz;
            w2 = ppy + ppxy + ppyz + ppxyz;
            data.fy(ix, iy, iz) = (w1 - w2) * 0.25 / dhy;

            w1 = pp + ppx + ppy + ppxy;
            w2 = ppz + ppxz + ppyz + ppxyz;
            data.fz(ix, iy, iz) = (w1 - w2) * 0.25 / dhz;

            // Add gravity force
            // data.fx(ix, iy, iz) -= data.rho_v(ix, iy, iz) * data.grav_r(ix);
            // data.fz(ix, iy, iz) -= data.rho_v(ix, iy, iz) * data.grav_z(iz);

            // Geometry corrections
            if (core_data.geometry == geometryType::Spherical)
            {
              T_dataType cotantheta = 1.0 / tan(core_data.yb(iy));
              data.fx(ix, iy, iz) += data.rho_v(ix, iy, iz) * (data.vy(ix, iy, iz) * data.vy(ix, iy, iz) + data.vz(ix, iy, iz) * data.vz(ix, iy, iz)) / core_data.xb(ix);
              data.fy(ix, iy, iz) -= data.rho_v(ix, iy, iz) * (data.vy(ix, iy, iz) * data.vx(ix, iy, iz) - cotantheta * data.vz(ix, iy, iz) * data.vz(ix, iy, iz)) / core_data.xb(ix);
              data.fz(ix, iy, iz) -= data.rho_v(ix, iy, iz) * (data.vz(ix, iy, iz) * data.vx(ix, iy, iz) + cotantheta * data.vz(ix, iy, iz) * data.vy(ix, iy, iz)) / core_data.xb(ix);
            } else if (core_data.geometry == geometryType::Cylindrical)
            {
                data.fx(ix, iy, iz) += data.rho_v(ix, iy, iz) * data.vy(ix, iy, iz) * data.vy(ix, iy, iz) / core_data.xb(ix);
                data.fy(ix, iy, iz) -= data.rho_v(ix, iy, iz) * data.vy(ix, iy, iz) * data.vx(ix, iy, iz) / core_data.xb(ix);
            }

            // Update positions
            data.x(ix, iy, iz) += data.vx(ix, iy, iz) * data.dt;
            data.y(ix, iy, iz) += data.vy(ix, iy, iz) * data.dt;
            data.z(ix, iy, iz) += data.vz(ix, iy, iz) * data.dt;

            // Half-step velocities for remap
            data.vx1(ix, iy, iz) = data.vx(ix, iy, iz) + data.dt / 2.0 * (data.fx_visc(ix, iy, iz) + data.fx(ix, iy, iz)) / data.rho_v(ix, iy, iz);
            data.vy1(ix, iy, iz) = data.vy(ix, iy, iz) + data.dt / 2.0 * (data.fy_visc(ix, iy, iz) + data.fy(ix, iy, iz)) / data.rho_v(ix, iy, iz);
            data.vz1(ix, iy, iz) = data.vz(ix, iy, iz) + data.dt / 2.0 * (data.fz_visc(ix, iy, iz) + data.fz(ix, iy, iz)) / data.rho_v(ix, iy, iz);
        },
                        Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();
        // Remap (half timestep) boundary conditions
        this->remap_v_bcs();
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::corrector_step(simulationData &data, const LARE3DST<T_EOS>::simulationData & core_data)
    {

        using Range = pw::Range;

        shock_heating(data, core_data);

        // Correct velocities to final values
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        data.vx(ix, iy, iz) += data.dt * (data.fx_visc(ix, iy, iz) + data.fx(ix, iy, iz)) / data.rho_v(ix, iy, iz);
        data.vy(ix, iy, iz) += data.dt * (data.fy_visc(ix, iy, iz) + data.fy(ix, iy, iz)) / data.rho_v(ix, iy, iz);
        data.vz(ix, iy, iz) += data.dt * (data.fz_visc(ix, iy, iz) + data.fz(ix, iy, iz)) / data.rho_v(ix, iy, iz); }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();
        velocity_bcs();

        // Correct density and energy to final values
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType izm = iz - 1;
            T_indexType iym = iy - 1;
            T_indexType ixm = ix - 1;

            // vx1 at Bx(i,j,k)
            T_dataType vxb = 0.25 * (data.vx1(ix, iy, iz) + data.vx1(ix, iym, iz) + data.vx1(ix, iy, izm) + data.vx1(ix, iym, izm));
            // vx1 at Bx(i-1,j,k)
            T_dataType vxbm = 0.25 * (data.vx1(ixm, iy, iz) + data.vx1(ixm, iym, iz) + data.vx1(ixm, iy, izm) + data.vx1(ixm, iym, izm));
            // vy1 at By(i,j,k)
            T_dataType vyb = 0.25 * (data.vy1(ix, iy, iz) + data.vy1(ixm, iy, iz) + data.vy1(ix, iy, izm) + data.vy1(ixm, iy, izm));
            // vy1 at By(i,j-1,k)
            T_dataType vybm = 0.25 * (data.vy1(ix, iym, iz) + data.vy1(ixm, iym, iz) + data.vy1(ix, iym, izm) + data.vy1(ixm, iym, izm));
            // vz1 at Bz(i,j,k)
            T_dataType vzb = 0.25 * (data.vz1(ix, iy, iz) + data.vz1(ixm, iy, iz) + data.vz1(ix, iym, iz) + data.vz1(ixm, iym, iz));
            // vz1 at Bz(i,j,k-1)
            T_dataType vzbm = 0.25 * (data.vz1(ix, iy, izm) + data.vz1(ixm, iy, izm) + data.vz1(ix, iym, izm) + data.vz1(ixm, iym, izm));

            T_dataType vol = data.cv(ix, iy, iz);
            T_dataType dvxdx = (vxb * core_data.dxab(ix, iy, iz) - vxbm * core_data.dxab(ixm, iy, iz)) / vol;
            T_dataType dvydy = (vyb * core_data.dyab(ix, iy, iz) - vybm * core_data.dyab(ix, iym, iz)) / vol;
            T_dataType dvzdz = (vzb * core_data.dzab(ix, iy, iz) - vzbm * core_data.dzab(ix, iy, izm)) / vol;

            T_dataType dv = (dvxdx + dvydy + dvzdz) * data.dt;

            data.cv1(ix, iy, iz) = vol * (1.0 + dv);

            // Energy at end of Lagrangian step
            data.energy(ix, iy, iz) += (data.dt * data.visc_heat(ix, iy, iz) - dv * data.pressure(ix, iy, iz)) / data.rho(ix, iy, iz);

            // Update density based on volume change
            data.rho(ix, iy, iz) /= (1.0 + dv);

            // total_visc_heating += dt * visc_heat(ix, iy, iz) * cv(ix, iy, iz);
        },
                        Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
        pw::fence();

        this->energy_bcs();
        this->density_bcs();
        this->velocity_bcs();
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::shock_heating(simulationData &data, const LARE3DST<T_EOS>::simulationData & core_data)
    {
        using Range = pw::Range;

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType izm = iz - 1, izp = iz + 1;
            T_indexType iym = iy - 1, iyp = iy + 1;
            T_indexType ixm = ix - 1;

            T_dataType a1 = 1 *
                            ((data.vx(ixm, iym, iz) - data.vx(ix, iym, iz)) *
                                 (data.vx1(ixm, iym, iz) - data.vx1(ix, iym, iz)) +
                             (data.vy(ixm, iym, iz) - data.vy(ix, iym, iz)) *
                                 (data.vy1(ixm, iym, iz) - data.vy1(ix, iym, iz)) +
                             (data.vz(ixm, iym, iz) - data.vz(ix, iym, iz)) *
                                 (data.vz1(ixm, iym, iz) - data.vz1(ix, iym, iz)));
            T_dataType a2 = 1 *
                            ((data.vx(ix, iym, iz) - data.vx(ix, iy, iz)) *
                                 (data.vx1(ix, iym, iz) - data.vx1(ix, iy, iz)) +
                             (data.vy(ix, iym, iz) - data.vy(ix, iy, iz)) *
                                 (data.vy1(ix, iym, iz) - data.vy1(ix, iy, iz)) +
                             (data.vz(ix, iym, iz) - data.vz(ix, iy, iz)) *
                                 (data.vz1(ix, iym, iz) - data.vz1(ix, iy, iz)));
            T_dataType a3 = 1 *
                            ((data.vx(ix, iy, iz) - data.vx(ixm, iy, iz)) *
                                 (data.vx1(ix, iy, iz) - data.vx1(ixm, iy, iz)) +
                             (data.vy(ix, iy, iz) - data.vy(ixm, iy, iz)) *
                                 (data.vy1(ix, iy, iz) - data.vy1(ixm, iy, iz)) +
                             (data.vz(ix, iy, iz) - data.vz(ixm, iy, iz)) *
                                 (data.vz1(ix, iy, iz) - data.vz1(ixm, iy, iz)));
            T_dataType a4 = 1 *
                            ((data.vx(ixm, iy, iz) - data.vx(ixm, iym, iz)) *
                                 (data.vx1(ixm, iy, iz) - data.vx1(ixm, iym, iz)) +
                             (data.vy(ixm, iy, iz) - data.vy(ixm, iym, iz)) *
                                 (data.vy1(ixm, iy, iz) - data.vy1(ixm, iym, iz)) +
                             (data.vz(ixm, iy, iz) - data.vz(ixm, iym, iz)) *
                                 (data.vz1(ixm, iy, iz) - data.vz1(ixm, iym, iz)));

            T_dataType a5 = 1 *
                            ((data.vx(ixm, iym, izp) - data.vx(ix, iym, izp)) *
                                 (data.vx1(ixm, iym, izp) - data.vx1(ix, iym, izp)) +
                             (data.vy(ixm, iym, izp) - data.vy(ix, iym, izp)) *
                                 (data.vy1(ixm, iym, izp) - data.vy1(ix, iym, izp)) +
                             (data.vz(ixm, iym, izp) - data.vz(ix, iym, izp)) *
                                 (data.vz1(ixm, iym, izp) - data.vz1(ix, iym, izp)));
            T_dataType a6 = 1 *
                            ((data.vx(ix, iym, izp) - data.vx(ix, iy, izp)) *
                                 (data.vx1(ix, iym, izp) - data.vx1(ix, iy, izp)) +
                             (data.vy(ix, iym, izp) - data.vy(ix, iy, izp)) *
                                 (data.vy1(ix, iym, izp) - data.vy1(ix, iy, izp)) +
                             (data.vz(ix, iym, izp) - data.vz(ix, iy, izp)) *
                                 (data.vz1(ix, iym, izp) - data.vz1(ix, iy, izp)));
            T_dataType a7 = 1 *
                            ((data.vx(ix, iy, izp) - data.vx(ixm, iy, izp)) *
                                 (data.vx1(ix, iy, izp) - data.vx1(ixm, iy, izp)) +
                             (data.vy(ix, iy, izp) - data.vy(ixm, iy, izp)) *
                                 (data.vy1(ix, iy, izp) - data.vy1(ixm, iy, izp)) +
                             (data.vz(ix, iy, izp) - data.vz(ixm, iy, izp)) *
                                 (data.vz1(ix, iy, izp) - data.vz1(ixm, iy, izp)));
            T_dataType a8 = 1 *
                            ((data.vx(ixm, iy, izp) - data.vx(ixm, iym, izp)) *
                                 (data.vx1(ixm, iy, izp) - data.vx1(ixm, iym, izp)) +
                             (data.vy(ixm, iy, izp) - data.vy(ixm, iym, izp)) *
                                 (data.vy1(ixm, iy, izp) - data.vy1(ixm, iym, izp)) +
                             (data.vz(ixm, iy, izp) - data.vz(ixm, iym, izp)) *
                                 (data.vz1(ixm, iy, izp) - data.vz1(ixm, iym, izp)));

            T_dataType a9 = 1 *
                            ((data.vx(ix, iy, izm) - data.vx(ix, iy, iz)) *
                                 (data.vx1(ix, iy, izm) - data.vx1(ix, iy, iz)) +
                             (data.vy(ix, iy, izm) - data.vy(ix, iy, iz)) *
                                 (data.vy1(ix, iy, izm) - data.vy1(ix, iy, iz)) +
                             (data.vz(ix, iy, izm) - data.vz(ix, iy, iz)) *
                                 (data.vz1(ix, iy, izm) - data.vz1(ix, iy, iz)));
            T_dataType a10 = 1 *
                             ((data.vx(ixm, iy, izm) - data.vx(ixm, iy, iz)) *
                                  (data.vx1(ixm, iy, izm) - data.vx1(ixm, iy, iz)) +
                              (data.vy(ixm, iy, izm) - data.vy(ixm, iy, iz)) *
                                  (data.vy1(ixm, iy, izm) - data.vy1(ixm, iy, iz)) +
                              (data.vz(ixm, iy, izm) - data.vz(ixm, iy, iz)) *
                                  (data.vz1(ixm, iy, izm) - data.vz1(ixm, iy, iz)));
            T_dataType a11 = 1 *
                             ((data.vx(ixm, iym, izm) - data.vx(ixm, iym, iz)) *
                                  (data.vx1(ixm, iym, izm) - data.vx1(ixm, iym, iz)) +
                              (data.vy(ixm, iym, izm) - data.vy(ixm, iym, iz)) *
                                  (data.vy1(ixm, iym, izm) - data.vy1(ixm, iym, iz)) +
                              (data.vz(ixm, iym, izm) - data.vz(ixm, iym, iz)) *
                                  (data.vz1(ixm, iym, izm) - data.vz1(ixm, iym, iz)));
            T_dataType a12 = 1 *
                             ((data.vx(ix, iym, izm) - data.vx(ix, iym, iz)) *
                                  (data.vx1(ix, iym, izm) - data.vx1(ix, iym, iz)) +
                              (data.vy(ix, iym, izm) - data.vy(ix, iym, iz)) *
                                  (data.vy1(ix, iym, izm) - data.vy1(ix, iym, iz)) +
                              (data.vz(ix, iym, izm) - data.vz(ix, iym, iz)) *
                                  (data.vz1(ix, iym, izm) - data.vz1(ix, iym, iz)));

            T_dataType dx = core_data.dxb(ix);
            T_dataType dy = core_data.dyb(iy) * core_data.hyc(ix);
            T_dataType dz = core_data.dzb(iz) * core_data.hz2(ix, iy);

            data.visc_heat(ix, iy, iz) =
                (-0.25 * dy * dz * data.alpha1(ix, iy, iz) * a1 - 0.25 * dx * dz * data.alpha2(ix, iy, iz) * a2 - 0.25 * dy * dz * data.alpha1(ix, iyp, iz) * a3 -
                 0.25 * dx * dz * data.alpha2(ixm, iy, iz) * a4 - 0.25 * dy * dz * data.alpha1(ix, iy, izp) * a5 - 0.25 * dx * dz * data.alpha2(ix, iy, izp) * a6 -
                 0.25 * dy * dz * data.alpha1(ix, iyp, izp) * a7 - 0.25 * dx * dz * data.alpha2(ixm, iy, izp) * a8 - 0.25 * dy * dy * data.alpha3(ix, iy, iz) * a9 -
                 0.25 * dx * dy * data.alpha3(ixm, iy, iz) * a10 - 0.25 * dy * dy * data.alpha3(ixm, iym, iz) * a11 - 0.25 * dx * dy * data.alpha3(ix, iym, iz) * a12);

            data.visc_heat(ix, iy, iz) = pw::max(data.visc_heat(ix, iy, iz) / data.cv(ix, iy, iz), 0.0);
        },
                        Range(0, data.nx + 1), Range(0, data.ny + 1), Range(0, data.nz + 1));
    }
}
