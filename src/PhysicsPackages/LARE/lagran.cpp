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
#include "shared_data.h"

/**
 * Class representing data only needed during the lagrangian step
 */
struct lagranData
{
    volumeArray bx1; // X-magnetic field at half timestep
    volumeArray by1; // Y-magnetic field at half timestep
    volumeArray bz1; // Z-magnetic field at half timestep
    volumeArray alpha1; // Alpha1 coefficient for magnetic field update
    volumeArray alpha2; // Alpha2 coefficient for magnetic field update
    volumeArray alpha3; // Alpha3 coefficient for magnetic field update
    volumeArray visc_heat; // Viscous heating
    volumeArray pressure; // Pressure array
    volumeArray p_e; // Electron pressure
    volumeArray p_i; // Ion pressure
    volumeArray rho_v; // Density at half timestep
    volumeArray cv_v; // Control volume at half timestep
    volumeArray fx; // X-force
    volumeArray fy; // Y-force
    volumeArray fz; // Z-force
    volumeArray fx_visc; // X-viscous force
    volumeArray fy_visc; // Y-viscous force
    volumeArray fz_visc; // Z-viscous force
    volumeArray flux_x; // X-flux
    volumeArray flux_y; // Y-flux
    volumeArray flux_z; // Z-flux
    volumeArray curlb; // Curl of the magnetic field
};


void shock_viscosity(simulationData &data, lagranData &lagran);
void set_dt(simulationData &data, lagranData &lagran);
void resistive_effects(simulation& sim, simulationData &data, lagranData &lagran);
void rkstep(simulation& sim, simulationData &data, lagranData &lagran);
void bstep(simulation &sim, simulationData &data, lagranData &lagran);
void predictor_corrector_step(simulation &sim, simulationData &data, lagranData &lagran);
void b_field_and_cv1_update(simulation &sim, simulationData &data, lagranData &lagran);
void shock_heating(simulationData &data, lagranData &lagran);

DEVICEPREFIX INLINE T_dataType edge_viscosity(simulationData data, lagranData lagran,
    T_dataType dvdots, T_dataType dx, T_dataType dxm, T_dataType dxp, T_dataType cs_edge,
    int i0, int i1, int i2, int i3,
    int j0, int j1, int j2, int j3,
    int k0, int k1, int k2, int k3){

        dvdots = portableWrapper::min(0.0,dvdots);
        T_dataType rho_edge = 2.0 * lagran.rho_v(i1, j1, k1) * lagran.rho_v(i2, j2, k2) / (lagran.rho_v(i1, j1, k1) + lagran.rho_v(i2, j2, k2));

        T_dataType dvx = data.vx(i1, j1, k1) - data.vx(i2, j2, k2);
        T_dataType dvy = data.vy(i1, j1, k1) - data.vy(i2, j2, k2);
        T_dataType dvz = data.vz(i1, j1, k1) - data.vz(i2, j2, k2);
        T_dataType dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
        T_dataType dv = std::sqrt(dv2);

        T_dataType psi = 0.0;
        if (dv * data.dt/dx < 1.e-14) {
            dvdots = 0.0;
        } else {
            dvdots/= dv;
        }

        T_dataType dvxm = data.vx(i0, j0, k0) - data.vx(i1, j1, k1);
        T_dataType dvxp = data.vx(i2, j2, k2) - data.vx(i3, j3, k3);
        T_dataType dvym = data.vy(i0, j0, k0) - data.vy(i1, j1, k1);
        T_dataType dvyp = data.vy(i2, j2, k2) - data.vy(i3, j3, k3);
        T_dataType dvzm = data.vz(i0, j0, k0) - data.vz(i1, j1, k1);
        T_dataType dvzp = data.vz(i2, j2, k2) - data.vz(i3, j3, k3);

        T_dataType rl = 1.0, rr = 1.0;
        if (dv * data.dt / dx >= 1.e-14) {
            rl = (dvxp * dvx + dvyp * dvy + dvzp * dvz) * dx / (dxp * dv2);
            rr = (dvxm * dvx + dvym * dvy + dvzm * dvz) * dx / (dxm * dv2);
        }

        psi = portableWrapper::min({0.5 * (rr + rl), 2.0 * rl, 2.0 * rr, 1.0});
        psi = portableWrapper::max(0.0, psi);

        // Find q_kur / abs(dv)
        T_dataType q_k_bar = rho_edge *
            (data.visc2_norm * dv + std::sqrt(data.visc2_norm * data.visc2_norm * dv2 + (data.visc1 * cs_edge) * (data.visc1 * cs_edge)));
        return q_k_bar * (1.0 - psi) * dvdots;
    }

void simulation::lagrangian_step(simulationData &data) {
    lagranData lagran;
    portableWrapper::portableArrayManager lagranManager;
    using Range = portableWrapper::Range;
    // Allocate arrays using the portableArrayManager
    lagranManager.allocate(lagran.bx1, Range(-1, data.nx + 2), Range(-1, data.ny + 2), Range(-1, data.nz + 2));
    lagranManager.allocate(lagran.by1, Range(-1, data.nx + 2), Range(-1, data.ny + 2), Range(-1, data.nz + 2));
    lagranManager.allocate(lagran.bz1, Range(-1, data.nx + 2), Range(-1, data.ny + 2), Range(-1, data.nz + 2));
    lagranManager.allocate(lagran.alpha1, Range(0,data.nx+1), Range(0,data.ny+2), Range(0,data.nz+2));
    lagranManager.allocate(lagran.alpha2, Range(-1,data.nx+1), Range(0,data.ny+1), Range(0,data.nz+2));
    lagranManager.allocate(lagran.alpha3, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(0,data.nz+1));
    lagranManager.allocate(lagran.visc_heat, Range(0,data.nx+1), Range(0,data.ny+1), Range(0,data.nz+1));
    lagranManager.allocate(lagran.pressure, Range(-1,data.nx+2), Range(-1,data.ny+2), Range(-1,data.nz+2));
    lagranManager.allocate(lagran.p_e, Range(-1,data.nx+2), Range(-1,data.ny+2), Range(-1,data.nz+2));
    lagranManager.allocate(lagran.p_i, Range(-1,data.nx+2), Range(-1,data.ny+2), Range(-1,data.nz+2));
    lagranManager.allocate(lagran.rho_v, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
    lagranManager.allocate(lagran.cv_v, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
    lagranManager.allocate(lagran.fx, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    lagranManager.allocate(lagran.fy, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    lagranManager.allocate(lagran.fz, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    lagranManager.allocate(lagran.fx_visc, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    lagranManager.allocate(lagran.fy_visc, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    lagranManager.allocate(lagran.fz_visc, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    lagranManager.allocate(lagran.flux_x, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    lagranManager.allocate(lagran.flux_y, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    lagranManager.allocate(lagran.flux_z, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    lagranManager.allocate(lagran.curlb, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));


    //All of the arrays are deallocated when lagranManager goes out of scope
    // Initialize bx1, by1, bz1, p_e, p_i, pressure
    T_dataType gas_gamma = data.gas_gamma;
    volumeArray bxl = data.bx;
    volumeArray byl = data.by;
    volumeArray bzl = data.bz;
    volumeArray cvl = data.cv;
    volumeArray energy_el = data.energy_electron;
    volumeArray energy_il = data.energy_ion;
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        T_indexType iym = iy - 1;
        T_indexType ixm = ix - 1;
        lagran.bx1(ix, iy, iz) = 0.5 * (bxl(ix, iy, iz) + bxl(ixm, iy, iz));
        lagran.by1(ix, iy, iz) = 0.5 * (byl(ix, iy, iz) + byl(ix, iym, iz));
        lagran.bz1(ix, iy, iz) = 0.5 * (bzl(ix, iy, iz) + bzl(ix, iy, izm));

        lagran.p_e(ix, iy, iz) = (gas_gamma - 1.0) * data.rho(ix, iy, iz) * energy_el(ix, iy, iz);
        lagran.p_i(ix, iy, iz) = (gas_gamma - 1.0) * data.rho(ix, iy, iz) * energy_il(ix, iy, iz);
        lagran.pressure(ix, iy, iz) = lagran.p_e(ix, iy, iz) + lagran.p_i(ix, iy, iz);

    }, Range(-1,data.nx+2), Range(-1,data.ny+2), Range(-1,data.nz+2));

    // Compute rho_v and cv_v
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
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
        lagran.rho_v(ix, iy, iz) = sum_rho_cv / sum_cv;
        lagran.cv_v(ix, iy, iz) = 0.125 * sum_cv; // Assuming a constant factor for control volume
    }, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));

    shock_viscosity(data, lagran);
    set_dt(data, lagran);
    if (data.resistiveMHD){
        T_dataType dt_sub = data.dtr;
        int substeps = static_cast<int>(data.dt / dt_sub)+1;

        T_dataType actual_dt = data.dt;
        data.dt /= static_cast<T_dataType>(substeps);
        for (int i = 0; i < substeps; ++i) {
            this->eta_calc(data);
            resistive_effects(*this, data, lagran);
        }
        data.dt = actual_dt; // Restore the original dt after sub-stepping
    }

    predictor_corrector_step(*this, data, lagran);

    this->energy_bcs(data);
    this->density_bcs(data);
    this->velocity_bcs(data);
    //Lagrangian step data is automatically deallocated when lagranManager goes out of scope

}

void shock_viscosity(simulationData &data, lagranData &lagran) {
    using Range = portableWrapper::Range;
    data.visc2_norm = 0.25 * (data.gas_gamma + 1.0) * data.visc2;
    portableWrapper::portableArrayManager svManager;
    // Temporary arrays for sound speed
    portableWrapper::portableArray<T_dataType, 3> cs, cs_v;
    svManager.allocate(cs, portableWrapper::Range(-1, data.nx + 2), portableWrapper::Range(-1, data.ny + 2), portableWrapper::Range(-1, data.nz + 2));
    svManager.allocate(cs_v, portableWrapper::Range(-1, data.nx + 1), portableWrapper::Range(-1, data.ny + 1), portableWrapper::Range(-1, data.nz + 1));

    portableWrapper::assign(data.p_visc, 0.0);
    portableWrapper::assign(lagran.visc_heat, 0.0);
    portableWrapper::assign(lagran.fx_visc, 0.0);
    portableWrapper::assign(lagran.fy_visc, 0.0);
    portableWrapper::assign(lagran.fz_visc, 0.0);

    // Compute cs
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_dataType rmin = portableWrapper::max(data.rho(ix, iy, iz), data.none_zero);
        T_dataType b2 = lagran.bx1(ix, iy, iz) * lagran.bx1(ix, iy, iz) +
                        lagran.by1(ix, iy, iz) * lagran.by1(ix, iy, iz) +
                        lagran.bz1(ix, iy, iz) * lagran.bz1(ix, iy, iz);
        cs(ix, iy, iz) = std::sqrt((data.gas_gamma * lagran.pressure(ix, iy, iz) + b2) / rmin);
    }, Range(-1,data.nx+2), Range(-1,data.ny+2), Range(-1,data.nz+2));
    portableWrapper::fence();
    // Compute cs_v
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
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
        cs_v(ix, iy, iz) = 0.125 * sum / lagran.cv_v(ix, iy, iz);
    }, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
    portableWrapper::fence();

    // alpha1
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType ixm = ix - 1;
        T_indexType iym = iy - 1;
        T_indexType izm = iz - 1;
        T_indexType ixp = ix + 1;

        T_indexType i1 = ixm, j1 = iym, k1 = izm;
        T_indexType i2 = ix, j2 = iym, k2 = izm;
        T_indexType i0 = i1-1, j0 = j1, k0 = k1;
        T_indexType i3 = i2+1, j3 = j1, k3 = k1;

        T_dataType dx = data.dxb(ix);
        T_dataType dxp = data.dxb(ixp);
        T_dataType dxm = data.dxb(ixm);
        T_dataType dvdots = -(data.vx(i1, j1, k1) - data.vx(i2, j2, k2));
        T_dataType cs_edge = portableWrapper::min(cs_v(i1, j1, k1), cs_v(i2, j2, k2));
        // Edge viscosities from Caramana
        lagran.alpha1(ix, iy, iz) = edge_viscosity(data, lagran,
            dvdots, dx, dxm, dxp, cs_edge,
            i0, i1, i2, i3, j0, j1, j2, j3, k0, k1, k2, k3);

    }, Range(0,data.nx+1), Range(0,data.ny+2), Range(0,data.nz+2));

    // alpha2
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        T_indexType iym = iy - 1, iyp = iy + 1;
        T_indexType i1 = ix, j1 = iym, k1 = izm;
        T_indexType i2 = ix, j2 = iy, k2 = izm;
        T_indexType i0 = i1, j0 = j1 - 1, k0 = k1;
        T_indexType i3 = i2, j3 = j1 + 1, k3 = k1;

        T_dataType dx = data.dyb(iy) * data.hy(ix);
        T_dataType dxp = data.dyb(iyp) * data.hy(ix);
        T_dataType dxm = data.dyb(iym) * data.hy(ix);
        T_dataType dvdots = -(data.vy(i1, j1, k1) - data.vy(i2, j2, k2));
        T_dataType cs_edge = portableWrapper::min(cs_v(i1, j1, k1), cs_v(i2, j2, k2));
        lagran.alpha2(ix, iy, iz) = edge_viscosity(data, lagran,
            dvdots, dx, dxm, dxp,
            cs_edge,
            i0, i1, i2, i3, j0, j1, j2, j3, k0, k1, k2, k3);
    }, Range(-1,data.nx+1), Range(0,data.ny+1), Range(0,data.nz+2));

    // alpha3
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1, izp = iz + 1;

        T_indexType i1 = ix, j1 = iy, k1 = izm;
        T_indexType i2 = ix, j2 = iy, k2 = iz;
        T_indexType i0 = i1, j0 = j1, k0 = k1 - 1;
        T_indexType i3 = i2, j3 = j1, k3 = k1 + 1;

        T_dataType dx = data.dzb(iz) * data.hz(ix, iy);
        T_dataType dxp = data.dzb(izp) * data.hz(ix, iy);
        T_dataType dxm = data.dzb(izm) * data.hz(ix, iy);
        T_dataType dvdots = -(data.vz(i1, j1, k1) - data.vz(i2, j2, k2));
        T_dataType cs_edge = portableWrapper::min(cs_v(i1, j1, k1), cs_v(i2, j2, k2));
        lagran.alpha3(ix, iy, iz) = edge_viscosity(data, lagran,
            dvdots, dx, dxm, dxp,
            cs_edge,
            i0, i1, i2, i3, j0, j1, j2, j3,
            k0, k1, k2, k3);
    }, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(0,data.nz+1));

    portableWrapper::fence();

    //p_visc and visc_heat
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1, izp = iz + 1;
        T_indexType iym = iy - 1, iyp = iy + 1;
        T_indexType ixm = ix - 1;
        T_dataType a1 = pow(data.vx(ixm, iym, iz) - data.vx(ix, iym, iz), 2) + pow(data.vy(ixm, iym, iz) - data.vy(ix, iym, iz), 2) + pow(data.vz(ixm, iym, iz) - data.vz(ix, iym, iz), 2);
        T_dataType a2 = pow(data.vx(ix, iym, iz) - data.vx(ix, iy, iz), 2) + pow(data.vy(ix, iym, iz) - data.vy(ix, iy, iz), 2) + pow(data.vz(ix, iym, iz) - data.vz(ix, iy, iz), 2);
        T_dataType a3 = pow(data.vx(ix, iy, iz) - data.vx(ixm, iy, iz), 2) + pow(data.vy(ix, iy, iz) - data.vy(ixm, iy, iz), 2) + pow(data.vz(ix, iy, iz) - data.vz(ixm, iy, iz), 2);
        T_dataType a4 = pow(data.vx(ixm, iy, iz) - data.vx(ixm, iym, iz), 2) + pow(data.vy(ixm, iy, iz) - data.vy(ixm, iym, iz), 2) + pow(data.vz(ixm, iy, iz) - data.vz(ixm, iym, iz), 2);

        T_dataType a5 = pow(data.vx(ixm, iym, izp) - data.vx(ix, iym, izp), 2) + pow(data.vy(ixm, iym, izp) - data.vy(ix, iym, izp), 2) + pow(data.vz(ixm, iym, izp) - data.vz(ix, iym, izp), 2);
        T_dataType a6 = pow(data.vx(ix, iym, izp) - data.vx(ix, iy, izp), 2) + pow(data.vy(ix, iym, izp) - data.vy(ix, iy, izp), 2) + pow(data.vz(ix, iym, izp) - data.vz(ix, iy, izp), 2);
        T_dataType a7 = pow(data.vx(ix, iy, izp) - data.vx(ixm, iy, izp), 2) + pow(data.vy(ix, iy, izp) - data.vy(ixm, iy, izp), 2) + pow(data.vz(ix, iy, izp) - data.vz(ixm, iy, izp), 2);
        T_dataType a8 = pow(data.vx(ixm, iy, izp) - data.vx(ixm, iym, izp), 2) + pow(data.vy(ixm, iy, izp) - data.vy(ixm, iym, izp), 2) + pow(data.vz(ixm, iy, izp) - data.vz(ixm, iym, izp), 2);


        T_dataType a9 = pow(data.vx(ix, iy, izm) - data.vx(ix, iy, iz), 2) + pow(data.vy(ix, iy, izm) - data.vy(ix, iy, iz), 2) + pow(data.vz(ix, iy, izm) - data.vz(ix, iy, iz), 2);
        T_dataType a10 = pow(data.vx(ixm, iy, izm) - data.vx(ixm, iy, iz), 2) + pow(data.vy(ixm, iy, izm) - data.vy(ixm, iy, iz), 2) + pow(data.vz(ixm, iy, izm) - data.vz(ixm, iy, iz), 2);
        T_dataType a11 = pow(data.vx(ixm, iym, izm) - data.vx(ixm, iym, iz), 2) + pow(data.vy(ixm, iym, izm) - data.vy(ixm, iym, iz), 2) + pow(data.vz(ixm, iym, izm) - data.vz(ixm, iym, iz), 2);
        T_dataType a12 = pow(data.vx(ix, iym, izm) - data.vx(ix, iym, iz), 2) + pow(data.vy(ix, iym, izm) - data.vy(ix, iym, iz), 2) + pow(data.vz(ix, iym, izm) - data.vz(ix, iym, iz), 2);

        data.p_visc(ix, iy, iz) = portableWrapper::max(data.p_visc(ix, iy, iz), -lagran.alpha1(ix, iy, iz) * std::sqrt(a1));
        data.p_visc(ix, iy, iz) = portableWrapper::max(data.p_visc(ix, iy, iz), -lagran.alpha2(ix, iy, iz) * std::sqrt(a2));
        data.p_visc(ix, iy, iz) = portableWrapper::max(data.p_visc(ix, iy, iz), -lagran.alpha3(ix, iy, iz) * std::sqrt(a9));

        T_dataType dx = data.dxb(ix);
        T_dataType dy = data.dyb(iy) * data.hyc(ix);
        T_dataType dz = data.dzb(iz) * data.hz2(ix, iy);

        lagran.visc_heat(ix, iy, iz) =
            - 0.25 * dy * dz * lagran.alpha1(ix , iy , iz ) * a1 
            - 0.25 * dx * dz * lagran.alpha2(ix , iy , iz ) * a2 
            - 0.25 * dy * dz * lagran.alpha1(ix , iyp, iz ) * a3 
            - 0.25 * dx * dz * lagran.alpha2(ixm, iy , iz ) * a4 
            - 0.25 * dy * dz * lagran.alpha1(ix , iy , izp) * a5 
            - 0.25 * dx * dz * lagran.alpha2(ix , iy , izp) * a6 
            - 0.25 * dy * dz * lagran.alpha1(ix , iyp, izp) * a7 
            - 0.25 * dx * dz * lagran.alpha2(ixm, iy , izp) * a8 
            - 0.25 * dy * dy * lagran.alpha3(ix , iy , iz ) * a9 
            - 0.25 * dx * dy * lagran.alpha3(ixm, iy , iz ) * a10 
            - 0.25 * dy * dy * lagran.alpha3(ixm, iym, iz ) * a11 
            - 0.25 * dx * dy * lagran.alpha3(ix , iym, iz ) * a12;

        lagran.visc_heat(ix, iy, iz) = lagran.visc_heat(ix, iy, iz) / data.cv(ix, iy, iz);

    }, Range(0,data.nx+1), Range(0,data.ny+1), Range(0,data.nz+1));


    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1, izp = iz + 1;
        T_indexType iym = iy - 1, iyp = iy + 1;
        T_indexType ixm = ix - 1, ixp = ix + 1;

        T_dataType dx = data.dxb(ix);
        T_dataType dy = data.dyb(iy) * data.hyc(ix);
        T_dataType dz = data.dzb(iz) * data.hz2(ix, iy);

        T_dataType a1 = lagran.alpha1(ix, iyp, izp) * dy * dz;
        T_dataType a2 = lagran.alpha1(ixp, iyp, izp) * dy * dz;
        T_dataType a3 = lagran.alpha2(ix, iy, izp) * dx * dz;
        T_dataType a4 = lagran.alpha2(ix, iyp, izp) * dx * dz;
        T_dataType a5 = lagran.alpha3(ix, iy, iz) * dx * dy;
        T_dataType a6 = lagran.alpha3(ix, iy, izp) * dx * dy;

        lagran.fx_visc(ix, iy, iz) = 
            (
						a1 * (data.vx(ix, iy, iz) - data.vx(ixm, iy, iz)) + 
						a2 * (data.vx(ix, iy, iz) - data.vx(ixp, iy, iz)) + 
						a3 * (data.vx(ix, iy, iz) - data.vx(ix, iym, iz)) + 
						a4 * (data.vx(ix, iy, iz) - data.vx(ix, iyp, iz)) + 
						a5 * (data.vx(ix, iy, iz) - data.vx(ix, iy, izm)) + 
						a6 * (data.vx(ix, iy, iz) - data.vx(ix, iy, izp))
						) / lagran.cv_v(ix, iy, iz);

        lagran.fy_visc(ix, iy, iz) = 
            (
						a1 * (data.vy(ix, iy, iz) - data.vy(ixm, iy, iz)) + 
						a2 * (data.vy(ix, iy, iz) - data.vy(ixp, iy, iz)) + 
						a3 * (data.vy(ix, iy, iz) - data.vy(ix, iym, iz)) + 
						a4 * (data.vy(ix, iy, iz) - data.vy(ix, iyp, iz)) + 
						a5 * (data.vy(ix, iy, iz) - data.vy(ix, iy, izm)) + 
						a6 * (data.vy(ix, iy, iz) - data.vy(ix, iy, izp))
						) / lagran.cv_v(ix, iy, iz);

        lagran.fz_visc(ix, iy, iz) = 
            (
						a1 * (data.vz(ix, iy, iz) - data.vz(ixm, iy, iz)) + 
						a2 * (data.vz(ix, iy, iz) - data.vz(ixp, iy, iz)) + 
						a3 * (data.vz(ix, iy, iz) - data.vz(ix, iym, iz)) + 
						a4 * (data.vz(ix, iy, iz) - data.vz(ix, iyp, iz)) + 
						a5 * (data.vz(ix, iy, iz) - data.vz(ix, iy, izm)) + 
						a6 * (data.vz(ix, iy, iz) - data.vz(ix, iy, izp))
						) / lagran.cv_v(ix, iy, iz);

    }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    portableWrapper::fence();
}

void set_dt(simulationData &data, lagranData &lagran) {

    using Range = portableWrapper::Range;

    int i0 = data.geometry == geometryType::Cartesian ? 0:1;

    //Now need to do a map and reduction
    data.dt = data.dt_multiplier * 
    portableWrapper::applyReduction(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_dataType dx = data.dxb(ix);
        T_dataType dy = data.dyb(iy);
        T_dataType dz = data.dzb(iz);

        T_dataType dhx = dx;
        T_dataType dhy = dy * data.hyc(ix);
        T_dataType dhz = dz * data.hzc(ix, iy);

        T_dataType rho0 = portableWrapper::max(data.rho(ix, iy, iz), data.none_zero);
        T_dataType cs2 = data.gas_gamma * lagran.pressure(ix, iy, iz) / rho0;

        T_dataType w1 = (data.bx(ix, iy, iz) * data.bx(ix, iy, iz) +
                         data.by(ix, iy, iz) * data.by(ix, iy, iz) +
                         data.bz(ix, iy, iz) * data.bz(ix, iy, iz)) / data.mu0_si / rho0;
        T_dataType c_visc2 = data.p_visc(ix, iy, iz) / rho0;

        T_dataType length  = portableWrapper::min({dhx, dhy, dhz});

        T_dataType t1  = length / (std::sqrt(c_visc2) + std::sqrt(cs2 + w1 + c_visc2));

        return t1;
    }, LAMBDA(T_dataType &a, const T_dataType &b) {
        a=portableWrapper::min(a, b);
    }, data.largest_number,
    Range(i0, data.nx), Range(0, data.ny), Range(0, data.nz));

    data.time += data.dt;
}

void simulation::eta_calc(simulationData &data) {
    using Range = portableWrapper::Range;

    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izp = iz + 1;
        T_indexType iyp = iy + 1;
        T_indexType ixp = ix + 1;

        T_dataType jx = (data.bz(ix, iyp, iz) - data.bz(ix, iy, iz)) / data.dyc(iy) - (data.by(ix, iy, izp) - data.by(ix, iy, iz)) / data.dzc(iz);

        T_dataType jxp = (data.bz(ixp, iyp, iz) - data.bz(ixp, iy, iz)) / data.dyc(iy) - (data.by(ixp, iy, izp) - data.by(ixp, iy, iz)) / data.dzc(iz);

        T_dataType jy = (data.bx(ix, iy, izp) - data.bx(ix, iy, iz)) / data.dzc(iz) - (data.bz(ixp, iy, iz) - data.bz(ix, iy, iz)) / data.dxc(ix);

        T_dataType jyp = (data.bx(ix, iyp, izp) - data.bx(ix, iyp, iz)) / data.dzc(iz) - (data.bz(ixp, iyp, iz) - data.bz(ix, iyp, iz)) / data.dxc(ix);

        T_dataType jz = (data.by(ixp, iy, iz) - data.by(ix, iy, iz)) / data.dxc(ix) - (data.bx(ix, iyp, iz) - data.bx(ix, iy, iz)) / data.dyc(iy);

        T_dataType jzp = (data.by(ixp, iy, izp) - data.by(ix, iy, izp)) / data.dxc(ix) - (data.bx(ix, iyp, izp) - data.bx(ix, iy, izp)) / data.dyc(iy);

        jx = (jx + jxp) * 0.5;
        jy = (jy + jyp) * 0.5;
        jz = (jz + jzp) * 0.5;

        T_dataType j_local = sqrt(pow(jx, 2.0) + pow(jy, 2.0) + pow(jz, 2.0));

        if (j_local > data.j_max)
        {
            data.eta(ix, iy, iz) = data.eta_background + data.eta0;
        }
        else
        {
            data.eta(ix, iy, iz) = data.eta_background;
        }
    }, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
    portableWrapper::fence();
}

void resistive_effects(simulation &sim, simulationData &data, lagranData &lagran) {
    using Range = portableWrapper::Range;

    portableWrapper::assign(lagran.bx1, data.bx);
    portableWrapper::assign(lagran.by1, data.by);
    portableWrapper::assign(lagran.bz1, data.bz);

    rkstep(sim, data, lagran);
    bstep(sim, data, lagran);

    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        T_indexType iym = iy - 1;
        T_indexType ixm = ix - 1;
        T_dataType sum_curlb =
            lagran.curlb(ix, iy, iz) + lagran.curlb(ixm, iy, iz) +
            lagran.curlb(ix, iym, iz) + lagran.curlb(ixm, iym, iz) +
            lagran.curlb(ix, iy, izm) + lagran.curlb(ixm, iy, izm) +
            lagran.curlb(ix, iym, izm) + lagran.curlb(ixm, iym, izm);

        data.energy_electron(ix,iy,iz) += sum_curlb * data.dt/(8.0 * data.rho(ix,iy,iz));
    }, Range(1,data.nx), Range(1,data.ny), Range(1,data.nz));
    portableWrapper::fence();
    sim.energy_bcs(data);
    //TODO add ohmic heating
    //In original code jx_r, jy_r and jz_r are calculated but not used
    rkstep(sim, data, lagran);
}

void rkstep(simulation &sim, simulationData &data, lagranData &lagran) {
    using Range = portableWrapper::Range;

    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izp = iz + 1;
        T_indexType iyp = iy + 1;
        T_indexType ixp = ix + 1;

        T_dataType jx1 = (data.bz(ix, iyp, iz) - data.bz(ix, iy, iz)) / data.dyc(iy) - (data.by(ix, iy, izp) - data.by(ix, iy, iz)) / data.dzc(iz);
        T_dataType jx2 = (data.bz(ixp, iyp, iz) - data.bz(ixp, iy, iz)) / data.dyc(iy) - (data.by(ixp, iy, izp) - data.by(ixp, iy, iz)) / data.dzc(iz);
        T_dataType jy1 = (data.bx(ix, iy, izp) - data.bx(ix, iy, iz)) / data.dzc(iz) - (data.bz(ixp, iy, iz) - data.bz(ix, iy, iz)) / data.dxc(ix);
        T_dataType jy2 = (data.bx(ix, iyp, izp) - data.bx(ix, iyp, iz)) / data.dzc(iz) - (data.bz(ixp, iyp, iz) - data.bz(ix, iyp, iz)) / data.dxc(ix);
        T_dataType jz1 = (data.by(ixp, iy, iz) - data.by(ix, iy, iz)) / data.dxc(ix) - (data.bx(ix, iyp, iz) - data.bx(ix, iy, iz)) / data.dyc(iy);
        T_dataType jz2 = (data.by(ixp, iy, izp) - data.by(ix, iy, izp)) / data.dxc(ix) - (data.bx(ix, iyp, izp) - data.bx(ix, iy, izp)) / data.dyc(iy);

        T_dataType jx = 0.5 * (jx1 + jx2);
        T_dataType jy = 0.5 * (jy1 + jy2);
        T_dataType jz = 0.5 * (jz1 + jz2);

        lagran.flux_x(ix, iy, iz) = -jx * data.eta(ix, iy, iz) * data.dxc(ix) * 0.5;
        lagran.flux_y(ix, iy, iz) = -jy * data.eta(ix, iy, iz) * data.dyc(iy) * 0.5;
        lagran.flux_z(ix, iy, iz) = -jz * data.eta(ix, iy, iz) * data.dzc(iz) * 0.5;
        // This isn't really curlb. It's actually heat flux
        lagran.curlb(ix, iy, iz) = data.eta(ix, iy, iz) * (jx * jx + jy * jy + jz * jz);

    }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    portableWrapper::fence();
}

void bstep(simulation &sim, simulationData &data, lagranData &lagran) {
    using Range = portableWrapper::Range;

    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        T_indexType iym = iy - 1;
        T_dataType area = data.dyb(iy) * data.dzb(iz);
        data.bx(ix, iy, iz) = lagran.bx1(ix, iy, iz) + 
        (
            lagran.flux_x(ix, iy, iz) - 
            lagran.flux_y(ix, iym, iz) + 
            lagran.flux_z(ix, iy, izm) - 
            lagran.flux_z(ix, iym, izm) - 
            lagran.flux_y(ix, iy, iz) + 
            lagran.flux_y(ix, iy, izm) - 
            lagran.flux_y(ix, iym, iz) + 
            lagran.flux_y(ix, iym, izm)
        ) * data.dt / area;
    }, Range(0,data.nx), Range(1,data.ny), Range(1,data.nz));

    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        T_indexType ixm = ix - 1;
        T_dataType area = data.dxb(ix) * data.dzb(iz);
        data.by(ix, iy, iz) = lagran.by1(ix, iy, iz) + 
        (
                lagran.flux_x(ix, iy, iz) - 
                lagran.flux_x(ix, iy, izm) + 
                lagran.flux_x(ixm, iy, iz) - 
                lagran.flux_x(ixm, iy, izm) - 
                lagran.flux_z(ix, iy, iz) + 
                lagran.flux_z(ixm, iy, iz) - 
                lagran.flux_z(ix, iy, izm) + 
                lagran.flux_z(ixm, iy, izm)
        ) * data.dt / area;
    }, Range(1,data.nx), Range(0,data.ny), Range(1,data.nz));

    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType iym = iy - 1;
        T_indexType ixm = ix - 1;
        T_dataType area = data.dxb(ix) * data.dyb(iy);
        data.bz(ix, iy, iz) = lagran.bz1(ix, iy, iz) + 
        (
                lagran.flux_y(ix, iy, iz) - 
                lagran.flux_y(ixm, iy, iz) + 
                lagran.flux_y(ix, iym, iz) - 
                lagran.flux_y(ixm, iym, iz) - 
                lagran.flux_x(ix, iy, iz) + 
                lagran.flux_x(ix, iym, iz) - 
                lagran.flux_x(ixm, iy, iz) + 
                lagran.flux_x(ixm, iym, iz)
        ) * data.dt / area;
    }, Range(1,data.nx), Range(1,data.ny), Range(0,data.nz));
    portableWrapper::fence();
    sim.bfield_bcs(data);
}

void predictor_corrector_step(simulation &sim, simulationData &data, lagranData &lagran) {
    using Range = portableWrapper::Range;
    //Update magnetic field and cell volume at half time step
    b_field_and_cv1_update(sim, data, lagran);

    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        lagran.bx1(ix,iy,iz)*=data.cv1(ix,iy,iz);
        lagran.by1(ix,iy,iz)*=data.cv1(ix,iy,iz);
        lagran.bz1(ix,iy,iz)*=data.cv1(ix,iy,iz);
    }, Range(-1,data.nx+2), Range(-1,data.ny+2), Range(-1,data.nz+2));

    //Predictor step for energy and pressure
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {

        T_dataType dv = data.cv1(ix, iy, iz) / data.cv(ix, iy, iz) - 1.0;
        T_dataType e1_e = data.energy_electron(ix,iy,iz) - lagran.p_e(ix,iy,iz) * dv/data.rho(ix,iy,iz);
        T_dataType e1_i = data.energy_ion(ix,iy,iz) - lagran.p_i(ix,iy,iz) * dv/data.rho(ix,iy,iz);
        e1_i += lagran.visc_heat(ix, iy, iz) * data.dt/2.0 /data.rho(ix,iy,iz);

        lagran.p_e(ix, iy, iz) = e1_e * (data.gas_gamma - 1.0) * data.rho(ix, iy, iz) * data.cv(ix, iy, iz)/ data.cv1(ix, iy, iz);
        lagran.p_i(ix, iy, iz) = e1_i * (data.gas_gamma - 1.0) * data.rho(ix, iy, iz) * data.cv(ix, iy, iz)/ data.cv1(ix, iy, iz);
        lagran.pressure(ix, iy, iz) = lagran.p_e(ix, iy, iz) + lagran.p_i(ix, iy, iz);
        //data.energy_electron(ix,iy,iz) = data.cv(ix,iy,iz);

    }, Range(0,data.nx+1), Range(0,data.ny+1), Range(0,data.nz+1));

    portableWrapper::fence();

    //Compute forces and currents
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izp = iz + 1;
        T_indexType iyp = iy + 1;
        T_indexType ixp = ix + 1;
        T_dataType dx = data.dxc(ix);
        T_dataType dy = data.dyc(iy);
        T_dataType dz = data.dzc(iz);

        T_dataType h2 = data.hy(ix), h2x = data.hy(ixp);
        T_dataType h3 = data.hz(ix, iy), h3x = data.hz(ixp, iy), h3y = data.hz(ix, iyp);
        T_dataType h2c = data.hyc(ix), h3c = data.hzc(ix, iy);
        T_dataType dhy = h2c * dy, dhz = h3c * dz;

        T_dataType pp = lagran.pressure(ix, iy, iz);
        T_dataType ppx = lagran.pressure(ixp, iy, iz);
        T_dataType ppy = lagran.pressure(ix, iyp, iz);
        T_dataType ppz = lagran.pressure(ix, iy, izp);
        T_dataType ppxy = lagran.pressure(ixp, iyp, iz);
        T_dataType ppxz = lagran.pressure(ixp, iy, izp);
        T_dataType ppyz = lagran.pressure(ix, iyp, izp);
        T_dataType ppxyz = lagran.pressure(ixp, iyp, izp);

        //Add pressure gradient force
        T_dataType w1 = pp + ppy + ppz + ppyz;
        T_dataType w2 = ppx + ppxy + ppxz + ppxyz;
        lagran.fx(ix, iy, iz) = (w1 - w2) * 0.25 / dx;

        w1 = pp + ppx + ppz + ppxz;
        w2 = ppy + ppxy + ppyz + ppxyz;
        lagran.fy(ix, iy, iz) = (w1 - w2) * 0.25 / dhy;

        w1 = pp + ppx + ppy + ppxy;
        w2 = ppz + ppxz + ppyz + ppxyz;
        lagran.fz(ix, iy, iz) = (w1 - w2) * 0.25 / dhz;

        // Cell volumes for current calculation
        T_dataType cvx = data.cv1(ix, iy, iz) + data.cv1(ix, iyp, iz) + data.cv1(ix, iy, izp) + data.cv1(ix, iyp, izp);
        T_dataType cvxp = data.cv1(ixp, iy, iz) + data.cv1(ixp, iyp, iz) + data.cv1(ixp, iy, izp) + data.cv1(ixp, iyp, izp);
        T_dataType cvy = data.cv1(ix, iy, iz) + data.cv1(ixp, iy, iz) + data.cv1(ix, iy, izp) + data.cv1(ixp, iy, izp);
        T_dataType cvyp = data.cv1(ix, iyp, iz) + data.cv1(ixp, iyp, iz) + data.cv1(ix, iyp, izp) + data.cv1(ixp, iyp, izp);
        T_dataType cvz =data.cv1(ix, iy, iz) + data.cv1(ixp, iy, iz) + data.cv1(ix, iyp, iz) + data.cv1(ixp, iyp, iz);
        T_dataType cvzp = data.cv1(ix, iy, izp) + data.cv1(ixp, iy, izp) + data.cv1(ix, iyp, izp) + data.cv1(ixp, iyp, izp);

        //Current components
        //Update jx
        //dbz/dy
        w1 = (lagran.bz1(ix, iy, iz) + lagran.bz1(ixp, iy, iz) + lagran.bz1(ix, iy, izp) + lagran.bz1(ixp, iy, izp)) * h3 / cvy;
        w2 = (lagran.bz1(ix, iyp, iz) + lagran.bz1(ixp, iyp, iz) + lagran.bz1(ix, iyp, izp) + lagran.bz1(ixp, iyp, izp)) * h3y / cvyp;
        T_dataType jx = (w2 - w1) / dhy / h3c;

        //dby/dz
        w1 = (lagran.by1(ix, iy, iz) + lagran.by1(ixp, iy, iz) + lagran.by1(ix, iyp, iz) + lagran.by1(ixp, iyp, iz)) / cvz;
        w2 = (lagran.by1(ix, iy, izp) + lagran.by1(ixp, iy, izp) + lagran.by1(ix, iyp, izp) + lagran.by1(ixp, iyp, izp)) / cvzp;
        jx -= (w2 - w1) / dhz;

        //Update jy
        //dbz/dx
        w1 = (lagran.bz1(ix, iy, iz) + lagran.bz1(ix, iyp, iz) + lagran.bz1(ix, iy, izp) + lagran.bz1(ix, iyp, izp)) * h3 / cvx;
        w2 = (lagran.bz1(ixp, iy, iz) + lagran.bz1(ixp, iyp, iz) + lagran.bz1(ixp, iy, izp) + lagran.bz1(ixp, iyp, izp)) * h3x / cvxp;
        T_dataType jy = -(w2 - w1) / dx / h3c;

        //dbx/dz
        w1 = (lagran.bx1(ix, iy, iz) + lagran.bx1(ixp, iy, iz) + lagran.bx1(ix, iyp, iz) + lagran.bx1(ixp, iyp, iz)) / cvz;
        w2 = (lagran.bx1(ix, iy, izp) + lagran.bx1(ixp, iy, izp) + lagran.bx1(ix, iyp, izp) + lagran.bx1(ixp, iyp, izp)) / cvzp;
        jy += (w2 - w1) / dhz;

        //Update jz
        //dby/dx
        w1 = (lagran.by1(ix, iy, iz) + lagran.by1(ix, iyp, iz) + lagran.by1(ix, iy, izp) + lagran.by1(ix, iyp, izp)) * h2 / cvx;
        w2 = (lagran.by1(ixp, iy, iz) + lagran.by1(ixp, iyp, iz) + lagran.by1(ixp, iy, izp) + lagran.by1(ixp, iyp, izp)) * h2x / cvxp;
        T_dataType jz = (w2 - w1) / dx / h2c;

        //dbx/dy
        w1 = (lagran.bx1(ix, iy, iz) + lagran.bx1(ixp, iy, iz) + lagran.bx1(ix, iy, izp) + lagran.bx1(ixp, iy, izp)) / cvy;
        w2 = (lagran.bx1(ix, iyp, iz) + lagran.bx1(ixp, iyp, iz) + lagran.bx1(ix, iyp, izp) + lagran.bx1(ixp, iyp, izp)) / cvyp;
        jz -= (w2 - w1) / dhy;


        // Average B field at cell center
        T_dataType bxv = (lagran.bx1(ix, iy, iz) + lagran.bx1(ixp, iy, iz) + lagran.bx1(ix, iyp, iz) + lagran.bx1(ixp, iyp, iz) + lagran.bx1(ix, iy, izp) + lagran.bx1(ixp, iy, izp) + lagran.bx1(ix, iyp, izp) + lagran.bx1(ixp, iyp, izp)) / (cvx + cvxp);
        T_dataType byv = (lagran.by1(ix, iy, iz) + lagran.by1(ixp, iy, iz) + lagran.by1(ix, iyp, iz) + lagran.by1(ixp, iyp, iz) + lagran.by1(ix, iy, izp) + lagran.by1(ixp, iy, izp) + lagran.by1(ix, iyp, izp) + lagran.by1(ixp, iyp, izp)) / (cvx + cvxp);
        T_dataType bzv = (lagran.bz1(ix, iy, iz) + lagran.bz1(ixp, iy, iz) + lagran.bz1(ix, iyp, iz) + lagran.bz1(ixp, iyp, iz) + lagran.bz1(ix, iy, izp) + lagran.bz1(ixp, iy, izp) + lagran.bz1(ix, iyp, izp) + lagran.bz1(ixp, iyp, izp)) / (cvx + cvxp);

        // Add JxB force
        lagran.fx(ix, iy, iz) += (jy * bzv - jz * byv) / mu0_si;
        lagran.fy(ix, iy, iz) += (jz * bxv - jx * bzv) / mu0_si;
        lagran.fz(ix, iy, iz) += (jx * byv - jy * bxv) / mu0_si;

        //Add gravity force
        //lagran.fx(ix, iy, iz) -= lagran.rho_v(ix, iy, iz) * data.grav_r(ix);
        //lagran.fz(ix, iy, iz) -= lagran.rho_v(ix, iy, iz) * data.grav_z(iz);

        // Geometry corrections
        if (data.geometry == geometryType::Spherical)
        {
          T_dataType cotantheta = 1.0 / tan(data.yb(iy));
          lagran.fx(ix, iy, iz) += lagran.rho_v(ix, iy, iz) * (data.vy(ix, iy, iz) * data.vy(ix, iy, iz) + data.vz(ix, iy, iz) * data.vz(ix, iy, iz)) / data.xb(ix);
          lagran.fy(ix, iy, iz) -= lagran.rho_v(ix, iy, iz) * (data.vy(ix, iy, iz) * data.vx(ix, iy, iz) - cotantheta * data.vz(ix, iy, iz) * data.vz(ix, iy, iz)) / data.xb(ix);
          lagran.fz(ix, iy, iz) -= lagran.rho_v(ix, iy, iz) * (data.vz(ix, iy, iz) * data.vx(ix, iy, iz) + cotantheta * data.vz(ix, iy, iz) * data.vy(ix, iy, iz)) / data.xb(ix);
        } else if (data.geometry == geometryType::Cylindrical)
        {
          lagran.fx(ix, iy, iz) += lagran.rho_v(ix, iy, iz) * data.vy(ix, iy, iz) * data.vy(ix, iy, iz) / data.xb(ix);
          lagran.fy(ix, iy, iz) -= lagran.rho_v(ix, iy, iz) * data.vy(ix, iy, iz) * data.vx(ix, iy, iz) / data.xb(ix);
        }

        // Update positions
        data.x(ix, iy, iz) += data.vx(ix, iy, iz) * data.dt;
        data.y(ix, iy, iz) += data.vy(ix, iy, iz) * data.dt;
        data.z(ix, iy, iz) += data.vz(ix, iy, iz) * data.dt;

        // Half-step velocities for remap
        data.vx1(ix, iy, iz) = data.vx(ix, iy, iz) + data.dt/2.0 * (lagran.fx_visc(ix, iy, iz) + lagran.fx(ix, iy, iz)) / lagran.rho_v(ix, iy, iz);
        data.vy1(ix, iy, iz) = data.vy(ix, iy, iz) + data.dt/2.0 * (lagran.fy_visc(ix, iy, iz) + lagran.fy(ix, iy, iz)) / lagran.rho_v(ix, iy, iz);
        data.vz1(ix, iy, iz) = data.vz(ix, iy, iz) + data.dt/2.0 * (lagran.fz_visc(ix, iy, iz) + lagran.fz(ix, iy, iz)) / lagran.rho_v(ix, iy, iz);

    }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    portableWrapper::fence();
    //Remap (half timestep) boundary conditions
    sim.remap_v_bcs(data);

    //Divide bx1, by1, bz1 by cv1
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        lagran.bx1(ix,iy,iz)/=data.cv1(ix,iy,iz);
        lagran.by1(ix,iy,iz)/=data.cv1(ix,iy,iz);
        lagran.bz1(ix,iy,iz)/=data.cv1(ix,iy,iz);
    }, Range(-1,data.nx+2), Range(-1,data.ny+2), Range(-1,data.nz+2));

    shock_heating(data, lagran);

    //Correct velocities to final values
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        data.vx(ix, iy, iz) += data.dt * (lagran.fx_visc(ix, iy, iz) + lagran.fx(ix, iy, iz)) / lagran.rho_v(ix, iy, iz);
        data.vy(ix, iy, iz) += data.dt * (lagran.fy_visc(ix, iy, iz) + lagran.fy(ix, iy, iz)) / lagran.rho_v(ix, iy, iz);
        data.vz(ix, iy, iz) += data.dt * (lagran.fz_visc(ix, iy, iz) + lagran.fz(ix, iy, iz)) / lagran.rho_v(ix, iy, iz);
    }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
    portableWrapper::fence();
    sim.velocity_bcs(data);

    //Correct density and energy to final values
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
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
        T_dataType dvxdx = (vxb * data.dxab(ix, iy, iz) - vxbm * data.dxab(ixm, iy, iz)) / vol;
        T_dataType dvydy = (vyb * data.dyab(ix, iy, iz) - vybm * data.dyab(ix, iym, iz)) / vol;
        T_dataType dvzdz = (vzb * data.dzab(ix, iy, iz) - vzbm * data.dzab(ix, iy, izm)) / vol;

        T_dataType dv = (dvxdx + dvydy + dvzdz) * data.dt;

        data.cv1(ix, iy, iz) = vol * (1.0 + dv);

        // Energy at end of Lagrangian step
        data.energy_electron(ix, iy, iz) -= dv * lagran.p_e(ix, iy, iz) / data.rho(ix, iy, iz);
        data.energy_ion(ix, iy, iz) += (data.dt * lagran.visc_heat(ix, iy, iz) - dv * lagran.p_i(ix, iy, iz)) / data.rho(ix, iy, iz);

        //Update density based on volume change
        data.rho(ix, iy, iz) /= (1.0 + dv);

        //total_visc_heating += dt * visc_heat(ix, iy, iz) * cv(ix, iy, iz);
    
    }, Range(1,data.nx), Range(1,data.ny), Range(1,data.nz));
    portableWrapper::fence();

}

void b_field_and_cv1_update(simulation &sim, simulationData &data, lagranData &lagran) {
    using Range = portableWrapper::Range;

    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        T_indexType iym = iy - 1;
        T_indexType ixm = ix - 1;

        // vx at Bx(i,j,k)
        T_dataType vxb = 0.25 * (data.vx(ix, iy, iz) + data.vx(ix, iym, iz) + data.vx(ix, iy, izm) + data.vx(ix, iym, izm));
        // vx at Bx(i-1,j,k)
        T_dataType vxbm = 0.25 * (data.vx(ixm, iy, iz) + data.vx(ixm, iym, iz) + data.vx(ixm, iy, izm) + data.vx(ixm, iym, izm));
        // vy at By(i,j,k)
        T_dataType vyb = 0.25 * (data.vy(ix, iy, iz) + data.vy(ixm, iy, iz) + data.vy(ix, iy, izm) + data.vy(ixm, iy, izm));
        // vy at By(i,j-1,k)
        T_dataType vybm = 0.25 * (data.vy(ix, iym, iz) + data.vy(ixm, iym, iz) + data.vy(ix, iym, izm) + data.vy(ixm, iym, izm));
        // vz at Bz(i,j,k)
        T_dataType vzb = 0.25 * (data.vz(ix, iy, iz) + data.vz(ixm, iy, iz) + data.vz(ix, iym, iz) + data.vz(ixm, iym, iz));
        // vz at Bz(i,j,k-1)
        T_dataType vzbm = 0.25 * (data.vz(ix, iy, izm) + data.vz(ixm, iy, izm) + data.vz(ix, iym, izm) + data.vz(ixm, iym, izm));

        T_dataType vol = data.cv(ix,iy,iz);
        T_dataType dvxdx = (vxb * data.dxab(ix, iy, iz) - vxbm * data.dxab(ixm, iy, iz)) / vol;
        T_dataType dvydy = (vyb * data.dyab(ix, iy, iz) - vybm * data.dyab(ix, iym, iz)) / vol;
        T_dataType dvzdz = (vzb * data.dzab(ix, iy, iz) - vzbm * data.dzab(ix, iy, izm)) / vol;

        T_dataType dv = (dvxdx + dvydy + dvzdz) * data.dt/2.0;
        data.cv1(ix, iy, iz) = vol*(1.0 + dv);

        // vx at By(i,j,k)
        vxb = 0.25 * (data.vx(ix, iy, iz) + data.vx(ixm, iy, iz) +data.vx(ix, iy, izm) + data.vx(ixm, iy, izm));
        // vx at By(i,j-1,k)
        vxbm = 0.25 * (data.vx(ix, iym, iz) + data.vx(ixm, iym, iz) + data.vx(ix, iym, izm) + data.vx(ixm, iym, izm));
        // vy at Bx(i,j,k)
        vyb = 0.25 * (data.vy(ix, iy, iz) + data.vy(ix, iym, iz) + data.vy(ix, iy, izm) + data.vy(ix, iym, izm));
        // vy at Bx(i-1,j,k)
        vybm = 0.25 * (data.vy(ixm, iy, iz) + data.vy(ixm, iym, iz) + data.vy(ixm, iy, izm) + data.vy(ixm, iym, izm));

        T_dataType dvxdy = (vxb * data.dyab(ix, iy, iz) - vxbm * data.dyab(ix, iym, iz)) / vol;
        T_dataType dvydx = (vyb * data.dxab(ix, iy, iz) - vybm * data.dxab(ixm, iy, iz)) / vol;

        // vx at Bz(i,j,k)
        vxb = 0.25 * (data.vx(ix, iy, iz) + data.vx(ixm, iy, iz) + data.vx(ix, iym, iz) + data.vx(ixm, iym, iz));
        // vx at Bz(i,j,k-1)
        vxbm = 0.25 * (data.vx(ix, iy, izm) + data.vx(ixm, iy, izm) + data.vx(ix, iym, izm) + data.vx(ixm, iym, izm));
        // vz at Bx(i,j,k)
        vzb = 0.25 * (data.vz(ix, iy, iz) + data.vz(ix, iym, iz) + data.vz(ix, iy, izm) + data.vz(ix, iym, izm));
        // vz at Bx(i-1,j,k)
        vzbm = 0.25 * (data.vz(ixm, iy, iz) + data.vz(ixm, iym, iz) + data.vz(ixm, iy, izm) + data.vz(ixm, iym, izm));

        T_dataType dvxdz = (vxb * data.dzab(ix, iy, iz) - vxbm * data.dzab(ix, iy, izm)) / vol;
        T_dataType dvzdx = (vzb * data.dxab(ix, iy, iz) - vzbm * data.dxab(ixm, iy, iz)) / vol;

        // vy at Bz(i,j,k)
        vyb = 0.25 * (data.vy(ix, iy, iz) + data.vy(ixm, iy, iz) + data.vy(ix, iym, iz) + data.vy(ixm, iym, iz));
        // vy at Bz(i,j,k-1)
        vybm = 0.25 * (data.vy(ix, iy, izm) + data.vy(ixm, iy, izm) + data.vy(ix, iym, izm) + data.vy(ixm, iym, izm));
        // vz at By(i,j,k)
        vzb = 0.25 * (data.vz(ix, iy, iz) + data.vz(ixm, iy, iz) + data.vz(ix, iy, izm) + data.vz(ixm, iy, izm));
        // vz at By(i,j-1,k)
        vzbm = 0.25 * (data.vz(ix, iym, iz) + data.vz(ixm, iym, iz) + data.vz(ix, iym, izm) + data.vz(ixm, iym, izm));

        T_dataType dvydz = (vyb * data.dzab(ix, iy, iz) - vybm * data.dzab(ix, iy, izm)) / vol;
        T_dataType dvzdy = (vzb * data.dyab(ix, iy, iz) - vzbm * data.dyab(ix, iym, iz)) / vol;

        T_dataType w3 = lagran.bx1(ix, iy, iz) * dvxdx + lagran.by1(ix, iy, iz) * dvxdy + lagran.bz1(ix, iy, iz) * dvxdz;
        T_dataType w4 = lagran.bx1(ix, iy, iz) * dvydx + lagran.by1(ix, iy, iz) * dvydy + lagran.bz1(ix, iy, iz) * dvydz;
        T_dataType w5 = lagran.bx1(ix, iy, iz) * dvzdx + lagran.by1(ix, iy, iz) * dvzdy + lagran.bz1(ix, iy, iz) * dvzdz;

        lagran.bx1(ix, iy, iz) = (lagran.bx1(ix, iy, iz) + w3 * data.dt/2.0) / (1.0 + dv);
        lagran.by1(ix, iy, iz) = (lagran.by1(ix, iy, iz) + w4 * data.dt/2.0) / (1.0 + dv);
        lagran.bz1(ix, iy, iz) = (lagran.bz1(ix, iy, iz) + w5 * data.dt/2.0) / (1.0 + dv);

    }, Range(-1,data.nx+2), Range(-1,data.ny+2), Range(-1,data.nz+2));
    portableWrapper::fence();
}

void shock_heating(simulationData &data, lagranData &lagran) {
    using Range = portableWrapper::Range;
    portableWrapper::assign(lagran.visc_heat,0.0);

    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1, izp = iz + 1;
        T_indexType iym = iy - 1, iyp = iy + 1;
        T_indexType ixm = ix -1;

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

        T_dataType dx = data.dxb(ix);
        T_dataType dy = data.dyb(iy) * data.hyc(ix);
        T_dataType dz = data.dzb(iz) * data.hz2(ix, iy);

        lagran.visc_heat(ix, iy, iz) =
            (-0.25 * dy * dz * lagran.alpha1(ix, iy, iz) * a1 - 0.25 * dx * dz * lagran.alpha2(ix, iy, iz) * a2 - 0.25 * dy * dz * lagran.alpha1(ix, iyp, iz) * a3 - 0.25 * dx * dz * lagran.alpha2(ixm, iy, iz) * a4 - 0.25 * dy * dz * lagran.alpha1(ix, iy, izp) * a5 - 0.25 * dx * dz * lagran.alpha2(ix, iy, izp) * a6 - 0.25 * dy * dz * lagran.alpha1(ix, iyp, izp) * a7 - 0.25 * dx * dz * lagran.alpha2(ixm, iy, izp) * a8 - 0.25 * dy * dy * lagran.alpha3(ix, iy, iz) * a9 - 0.25 * dx * dy * lagran.alpha3(ixm, iy, iz) * a10 - 0.25 * dy * dy * lagran.alpha3(ixm, iym, iz) * a11 - 0.25 * dx * dy * lagran.alpha3(ix, iym, iz) * a12);

        lagran.visc_heat(ix, iy, iz) = portableWrapper::max(lagran.visc_heat(ix, iy, iz) / data.cv(ix, iy, iz), 0.0);

    }, Range(0,data.nx+1), Range(0,data.ny+1), Range(0,data.nz+1));
}
