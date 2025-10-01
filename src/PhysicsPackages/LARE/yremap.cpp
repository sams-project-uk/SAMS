#include "shared_data.h"
#include "remapData.h"

void vy_bx_flux(simulationData &data, remapData &remap_data);
void vy_bz_flux(simulationData &data, remapData &remap_data);
void y_mass_flux(simulationData &data, remapData &remap_data);
template<auto mPtr>
void y_energy_flux(simulationData &data, remapData &remap_data);
template<auto mPtr>
void y_mom_flux(simulationData &data, remapData &remap_data);


void simulation::remap_y(simulationData &data, remapData &remap_data) {
    using Range = portableWrapper::Range;
    portableWrapper::portableArrayManager yRemapManager;
	remap_data.flux.nullify();
	assert(remap_data.flux.data()==nullptr);

    yRemapManager.allocate(remap_data.flux, Range(-1, data.nx + 2), Range(-2, data.ny + 2), Range(-1, data.nz + 2));

    portableWrapper::assign(remap_data.dm, 0.0);
    portableWrapper::assign(remap_data.rho1, data.rho);

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
            T_indexType ixm = ix - 1;
            T_indexType izm = iz - 1;
            // vx at Bx(i,j,k)
            T_dataType vxb = (data.vx1(ix, iy, iz) + data.vx1(ix, iym, iz) +
                data.vx1(ix, iy, izm) + data.vx1(ix, iym, izm)) *
                0.25;

            // vx at Bx(i-1,j,k)
            T_dataType vxbm = (data.vx1(ixm, iy, iz) + data.vx1(ixm, iym, iz) +
                    data.vx1(ixm, iy, izm) + data.vx1(ixm, iym, izm)) *
                0.25;

            // vy at By(i,j,k)
            T_dataType vyb = (data.vy1(ix, iy, iz) + data.vy1(ixm, iy, iz) +
                data.vy1(ix, iy, izm) + data.vy1(ixm, iy, izm)) *
                0.25;

            // vy at By(i,j-1,k)
            T_dataType vybm = (data.vy1(ix, iym, iz) + data.vy1(ixm, iym, iz) +
                    data.vy1(ix, iym, izm) + data.vy1(ixm, iym, izm)) *
                0.25;

            // vz at Bz(i,j,k)
            T_dataType vzb = (data.vz1(ix, iy, iz) + data.vz1(ixm, iy, iz) +
                data.vz1(ix, iym, iz) + data.vz1(ixm, iym, iz)) *
                0.25;

            // vz at Bz(i,j,k-1)
            T_dataType vzbm = (data.vz1(ix, iy, izm) + data.vz1(ixm, iy, izm) +
                    data.vz1(ix, iym, izm) + data.vz1(ixm, iym, izm)) *
                0.25;

            T_dataType vol = data.cv(ix, iy, iz);
            T_dataType dvxdx = remap_data.xpass * (vxb * data.dxab(ix, iy, iz) - vxbm * data.dxab(ixm, iy, iz)) / vol;
            T_dataType dvydy = (vyb * data.dyab(ix, iy, iz) - vybm * data.dyab(ix, iym, iz)) / vol;
            T_dataType dvzdz = remap_data.zpass * (vzb * data.dzab(ix, iy, iz) - vzbm * data.dzab(ix, iy, izm)) / vol;

            T_dataType dv = (dvxdx + dvzdz) * data.dt;

            // Control volume after remap
            remap_data.cv2(ix, iy, iz) = vol * (1.0 + dv);

            dv = dv + dvydy * data.dt;

            // Control volume before remap
            data.cv1(ix, iy, iz) = vol * (1.0 + dv);

            // dyb before remap
            remap_data.db1(ix, iy, iz) = data.hyc(ix) * data.dyb(iy) + (vyb - vybm) * data.dt;
        }, Range(-1, data.nx + 2), Range(-1, data.ny + 2), Range(-1, data.nz + 2));
    portableWrapper::fence();

    // cvc1 = vertex CV before remap
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixp = ix + 1;
            T_indexType iyp = iy + 1;
            T_indexType izp = iz + 1;
            remap_data.cvc1(ix, iy, iz) = 0.125 *(
                data.cv1(ix, iy, iz) + data.cv1(ixp, iy, iz) +
                data.cv1(ix, iyp, iz) + data.cv1(ixp, iyp, iz) +
                data.cv1(ix, iy, izp) + data.cv1(ixp, iy, izp) +
                data.cv1(ix, iyp, izp) + data.cv1(ixp, iyp, izp));
        }, Range(-1, data.nx + 1), Range(-1, data.ny + 1), Range(-1, data.nz + 1));
    portableWrapper::fence();

    //Evans and Hawley constrained transport remap of magnetic fluxes
    vy_bx_flux(data, remap_data);

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
            data.bx(ix, iy, iz) = data.bx(ix, iy, iz) - remap_data.flux(ix, iy, iz) + remap_data.flux(ix, iym, iz);
        }, Range(0, data.nx), Range(1, data.ny), Range(1, data.nz));

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            data.by(ix, iy, iz) = data.by(ix, iy, iz) + remap_data.flux(ix, iy, iz) - remap_data.flux(ixm, iy, iz);
        }, Range(1, data.nx), Range(0, data.ny), Range(1, data.nz));

    portableWrapper::fence();

    vy_bz_flux(data, remap_data);

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
            data.bz(ix, iy, iz) = data.bz(ix, iy, iz) - remap_data.flux(ix, iy, iz) + remap_data.flux(ix, iym, iz);
        }, Range(1, data.nx), Range(1, data.ny), Range(0, data.nz));

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType izm = iz - 1;
            data.by(ix, iy, iz) = data.by(ix, iy, iz) + remap_data.flux(ix, iy, iz) - remap_data.flux(ix, iy, izm);
        }, Range(1, data.nx), Range(0, data.ny), Range(1, data.nz));

    portableWrapper::fence();

    // Remap of mass + calculation of mass fluxes (dm) needed for later remaps
    y_mass_flux(data, remap_data);
    // Need dm(0:nx+1,-1:ny+1,0:nz+1) for velocity remap
    dm_y_bcs(data, remap_data);

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
            data.rho(ix, iy, iz) = (remap_data.rho1(ix, iy, iz) * data.cv1(ix, iy, iz) +
                                    remap_data.dm(ix, iym, iz) - remap_data.dm(ix, iy, iz)) /
                                    remap_data.cv2(ix, iy, iz);            
        }, Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
    portableWrapper::fence();

    y_energy_flux<&simulationData::energy_electron>(data, remap_data);
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
            data.energy_electron(ix, iy, iz) = 
                (
                    data.energy_electron(ix, iy, iz) * data.cv1(ix, iy, iz) * remap_data.rho1(ix, iy, iz) + 
                    remap_data.flux(ix, iym, iz) - remap_data.flux(ix, iy, iz)
                ) / 
                (remap_data.cv2(ix, iy, iz) * data.rho(ix, iy, iz));
        }, Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));

    portableWrapper::fence();

    y_energy_flux<&simulationData::energy_ion>(data, remap_data);
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
            data.energy_ion(ix, iy, iz) = 
                (
                    data.energy_ion(ix, iy, iz) * data.cv1(ix, iy, iz) * remap_data.rho1(ix, iy, iz) + 
                    remap_data.flux(ix, iym, iz) - remap_data.flux(ix, iy, iz)
                ) /
                (remap_data.cv2(ix, iy, iz) * data.rho(ix, iy, iz));
        }, Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
    portableWrapper::fence();

    // Redefine dyb1, cv1, cv2, dm and vy1 for velocity (vertex) cells.
    // In some of these calculations the flux variable is used as a temporary array

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixp = ix + 1;
            T_indexType iyp = iy + 1;
            T_indexType izp = iz + 1;

            remap_data.rho_v(ix, iy, iz) =
            remap_data.rho1(ix, iy, iz) * data.cv1(ix, iy, iz) +
            remap_data.rho1(ixp, iy, iz) * data.cv1(ixp, iy, iz) +
            remap_data.rho1(ix, iyp, iz) * data.cv1(ix, iyp, iz) +
            remap_data.rho1(ixp, iyp, iz) * data.cv1(ixp, iyp, iz) +
            remap_data.rho1(ix, iy, izp) * data.cv1(ix, iy, izp) +
            remap_data.rho1(ixp, iy, izp) * data.cv1(ixp, iy, izp) +
            remap_data.rho1(ix, iyp, izp) * data.cv1(ix, iyp, izp) +
            remap_data.rho1(ixp, iyp, izp) * data.cv1(ixp, iyp, izp);

            remap_data.rho_v(ix, iy, iz) *= 0.125 / remap_data.cvc1(ix, iy, iz);

        }, Range(0, data.nx), Range(-1, data.ny), Range(0, data.nz));
    portableWrapper::fence();

    //Move cv2 to vertex using flux array as a temporary
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixp = ix + 1;
            T_indexType iyp = iy + 1;
            T_indexType izp = iz + 1;

            remap_data.flux(ix, iy, iz) = 0.125 * (
                remap_data.cv2(ix, iy, iz) + remap_data.cv2(ixp, iy, iz) +
                remap_data.cv2(ix, iyp, iz) + remap_data.cv2(ixp, iyp, iz) +
                remap_data.cv2(ix, iy, izp) + remap_data.cv2(ixp, iy, izp) +
                remap_data.cv2(ix, iyp, izp) + remap_data.cv2(ixp, iyp, izp));

        }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
    portableWrapper::fence();
    //Now copy it back
    portableWrapper::assign(remap_data.cv2(Range(0,data.nx), Range(0,data.ny), Range(0,data.nz)), remap_data.flux(Range(0,data.nx), Range(0,data.ny), Range(0,data.nz)));
    portableWrapper::fence();

    //Move vy1 to y face centred for momentum remap
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iyp = iy + 1;
            remap_data.flux(ix, iy, iz) = (data.vy1(ix, iy, iz) + data.vy1(ix, iyp, iz)) * 0.5;
        }, Range(0, data.nx), Range(-2, data.ny+1), Range(0, data.nz));
    portableWrapper::fence();
    //And copy it back
    portableWrapper::assign(data.vy1(Range(0,data.nx), Range(-2,data.ny+1), Range(0,data.nz)), remap_data.flux(Range(0,data.nx), Range(-2,data.ny+1), Range(0,data.nz)));
    portableWrapper::fence();

    //Vertex control volume mass change
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixp = ix + 1;
            T_indexType iyp = iy + 1;
            T_indexType izp = iz + 1;

            remap_data.flux(ix, iy, iz) = 0.125 * (
                remap_data.dm(ix, iy, iz) + remap_data.dm(ixp, iy, iz) +
                remap_data.dm(ix, iyp, iz) + remap_data.dm(ixp, iyp, iz) +
                remap_data.dm(ix, iy, izp) + remap_data.dm(ixp, iy, izp) +
                remap_data.dm(ix, iyp, izp) + remap_data.dm(ixp, iyp, izp));
        }, Range(0, data.nx), Range(-1, data.ny), Range(0, data.nz));
    portableWrapper::fence();

    portableWrapper::assign(remap_data.dm(Range(0,data.nx), Range(-1,data.ny), Range(0,data.nz)), remap_data.flux(Range(0,data.nx), Range(-1,data.ny), Range(0,data.nz)));
    portableWrapper::fence();

    //Update vertex mass
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
            remap_data.rho_v1(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.cv1(ix,iy,iz) + remap_data.dm(ix,iym,iz) - remap_data.dm(ix,iy,iz)) /
                remap_data.cv2(ix, iy, iz);
        }, Range(0, data.nx), Range(-1, data.ny), Range(0, data.nz));
    portableWrapper::fence();

    y_mom_flux<&simulationData::vx>(data, remap_data);
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
        data.vx(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.vx(ix, iy, iz) * data.cv1(ix, iy, iz) +
                          remap_data.flux(ix, iym, iz) - remap_data.flux(ix, iy, iz)) /
                         (remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz));
        }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));

    y_mom_flux<&simulationData::vy>(data, remap_data);
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
        data.vy(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.vy(ix, iy, iz) * data.cv1(ix, iy, iz) +
                          remap_data.flux(ix, iym, iz) - remap_data.flux(ix, iy, iz)) /
                         (remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz));
        }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));

    y_mom_flux<&simulationData::vz>(data, remap_data);
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
        data.vz(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.vz(ix, iy, iz) * data.cv1(ix, iy, iz) +
                          remap_data.flux(ix, iym, iz) - remap_data.flux(ix, iy, iz)) /
                         (remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz));
        }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));

    this->boundary_conditions(data);
    remap_data.ypass = 0.0;

} //END simulation::remap_y

// Evans & Hawley constrained transport remap of vy*Bx fluxes
void vy_bx_flux(simulationData &data, remapData &remap_data) {
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType ixp = ix + 1;
        T_indexType iyp = iy + 1;
        T_indexType izm = iz - 1;
        T_indexType iym = iy - 1;
        T_indexType iyp2 = iy + 2;

        T_dataType v_advect = (data.vy1(ix, iy, iz) + data.vy1(ix, iy, izm)) * 0.5;
        T_dataType o_v = v_advect * data.dt;

        T_dataType dby = (remap_data.db1(ix, iy, iz) + remap_data.db1(ix, iyp, iz)) * 0.5;
        T_dataType dbyp = (remap_data.db1(ix, iyp, iz) + remap_data.db1(ixp, iyp, iz)) * 0.5;
        T_dataType dbyp2 = (remap_data.db1(ix, iyp2, iz) + remap_data.db1(ixp, iyp2, iz)) * 0.5;
        T_dataType dbym = (remap_data.db1(ix, iym, iz) + remap_data.db1(ixp, iym, iz)) * 0.5;

        T_dataType fm = data.bx(ix, iym, iz) / dbym;
        T_dataType fi = data.bx(ix, iy, iz) / dby;
        T_dataType fp = data.bx(ixp, iyp, iz) / dbyp;
        T_dataType fp2 = data.bx(ix, iyp2, iz) / dbyp2;

        T_dataType dfm = fi - fm;
        T_dataType dfi = fp - fi;
        T_dataType dfp = fp2 - fp;

        T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
        T_dataType vad_p = (sign_v + 1.0) * 0.5;
        T_dataType vad_m = 1.0 - vad_p;

        T_dataType fu = fi * vad_p + fp * vad_m;
        T_dataType dfu = dfm * vad_p + dfp * vad_m;
        T_dataType dyci = remap_data.cvc1(ix, iy, iz);
        T_dataType dycu = remap_data.cvc1(ix, iym, iz) * vad_p + remap_data.cvc1(ix, iyp, iz) * vad_m;
        T_dataType dybu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ix, iyp, iz) * vad_m;

        T_dataType dyu = dby * vad_p + dbyp * vad_m;
        T_dataType phi = std::abs(o_v) / dyu;

        T_dataType Da = (2.0 - phi) * std::abs(dfi) / dyci + (1.0 + phi) * std::abs(dfu) / dycu;
        Da = Da * sixth;

        T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

        T_dataType Di = sign_v * ss * portableWrapper::min({std::abs(Da) * dybu, std::abs(dfi), std::abs(dfu)});

        remap_data.flux(ix, iy, iz) = (fu + Di * (1.0 - phi)) * o_v;
      }
    , Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
    portableWrapper::fence();
}


  // Evans & Hawley constrained transport remap of vy*Bz fluxes
    void vy_bz_flux(simulationData &data, remapData &remap_data) {
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType ixm = ix - 1;
        T_indexType iym = iy - 1;
        T_indexType iyp = iy + 1;
        T_indexType iyp2 = iy + 2;
        T_indexType izp = iz + 1;

        T_dataType v_advect = (data.vy1(ix, iy, iz) + data.vy1(ixm, iy, iz)) * 0.5;
        T_dataType o_v = v_advect * data.dt;

        T_dataType dby = (remap_data.db1(ix, iy, iz) + remap_data.db1(ix, iy, izp)) * 0.5;
        T_dataType dbyp = (remap_data.db1(ix, iyp, iz) + remap_data.db1(ix, iyp, izp)) * 0.5;
        T_dataType dbyp2 = (remap_data.db1(ix, iyp2, iz) + remap_data.db1(ix, iyp2, izp)) * 0.5;
        T_dataType dbym = (remap_data.db1(ix, iym, iz) + remap_data.db1(ix, iym, izp)) * 0.5;

        T_dataType fm = data.bz(ix, iym, iz) / dbym;
        T_dataType fi = data.bz(ix, iy, iz) / dby;
        T_dataType fp = data.bz(ix, iyp, iz) / dbyp;
        T_dataType fp2 = data.bz(ix, iyp2, iz) / dbyp2;

        T_dataType dfm = fi - fm;
        T_dataType dfi = fp - fi;
        T_dataType dfp = fp2 - fp;

        T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
        T_dataType vad_p = (sign_v + 1.0) * 0.5;
        T_dataType vad_m = 1.0 - vad_p;

        T_dataType fu = fi * vad_p + fp * vad_m;
        T_dataType dfu = dfm * vad_p + dfp * vad_m;
        T_dataType dyci = remap_data.cvc1(ix, iy, iz);
        T_dataType dycu = remap_data.cvc1(ix, iym, iz) * vad_p + remap_data.cvc1(ix, iyp, iz) * vad_m;
        T_dataType dybu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ix, iyp, iz) * vad_m;

        T_dataType dyu = dby * vad_p + dbyp * vad_m;
        T_dataType phi = std::abs(o_v) / dyu;

        T_dataType Da = (2.0 - phi) * std::abs(dfi) / dyci + (1.0 + phi) * std::abs(dfu) / dycu;
        Da = Da * sixth;

        T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

        T_dataType Di = sign_v * ss * portableWrapper::min({std::abs(Da) * dybu, std::abs(dfi), std::abs(dfu)});

        remap_data.flux(ix, iy, iz) = (fu + Di * (1.0 - phi)) * o_v;
    }
    , Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
    portableWrapper::fence();
}

void y_mass_flux(simulationData &data, remapData &remap_data) {
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType izm = iz - 1;
            T_indexType iym = iy - 1;
            T_indexType iyp = iy + 1;
            T_indexType iyp2 = iy + 2;
            T_indexType ixm = ix - 1;
            T_dataType area = data.dyab(ix, iy, iz);

            T_dataType v_advect = (data.vy1(ixm, iy, iz) + data.vy1(ix, iy, iz) +
                            data.vy1(ixm, iy, izm) + data.vy1(ix, iy, izm)) *
                            0.25;
            T_dataType o_v = v_advect * data.dt * area;

            T_dataType fm = data.rho(ix, iym, iz);
            T_dataType fi = data.rho(ix, iy, iz);
            T_dataType fp = data.rho(ix, iyp, iz);
            T_dataType fp2 = data.rho(ix, iyp2, iz);

            T_dataType dfm = fi - fm;
            T_dataType dfi = fp - fi;
            T_dataType dfp = fp2 - fp;

            T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
            T_dataType vad_p = (sign_v + 1.0) * 0.5;
            T_dataType vad_m = 1.0 - vad_p;

            T_dataType fu = fi * vad_p + fp * vad_m;
            T_dataType dfu = dfm * vad_p + dfp * vad_m;
            T_dataType dyci = remap_data.cvc1(ix, iy, iz);
            T_dataType dycu = remap_data.cvc1(ix, iym, iz) * vad_p + remap_data.cvc1(ix, iyp, iz) * vad_m;
            T_dataType dybu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ix, iyp, iz) * vad_m;

            T_dataType phi = std::abs(o_v) / dybu;

            T_dataType Da = (2.0 - phi) * std::abs(dfi) / dyci + (1.0 + phi) * std::abs(dfu) / dycu;
            Da = Da * sixth;

            T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

            T_dataType Di = sign_v * ss * portableWrapper::min({std::abs(Da) * dybu, std::abs(dfi), std::abs(dfu)});

            remap_data.dm(ix, iy, iz) = (fu + Di * (1.0 - phi)) * o_v;
        }, Range(0, data.nx+1), Range(0, data.ny), Range(0, data.nz+1)); 

    portableWrapper::fence();
}

/**
 * This has been designed to be both y_energy_electron_flux and y_energy_ion_flux.
 * The template parameter mPtr allows us to pass in the member function pointer
 * for the appropriate energy type.
 */
template<auto mPtr>
void y_energy_flux(simulationData &data, remapData &remap_data) {
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            T_indexType iym = iy - 1;
            T_indexType iyp = iy + 1;
            T_indexType iyp2 = iy + 2;
            T_indexType izm = iz - 1;

            T_dataType area = data.dyab(ix, iy, iz);

            T_dataType v_advect = (data.vy1(ixm, iy, iz) + data.vy1(ix, iy, iz) +
                                data.vy1(ixm, iy, izm) + data.vy1(ix, iy, izm)) *
                                0.25;
            T_dataType o_v = v_advect * data.dt * area;

            T_dataType fm = (data.*mPtr)(ix, iym, iz);
            T_dataType fi = (data.*mPtr)(ix, iy, iz);
            T_dataType fp = (data.*mPtr)(ix, iyp, iz);
            T_dataType fp2 = (data.*mPtr)(ix, iyp2, iz);

            T_dataType dfm = fi - fm;
            T_dataType dfi = fp - fi;
            T_dataType dfp = fp2 - fp;

            T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
            T_dataType vad_p = (sign_v + 1.0) * 0.5;
            T_dataType vad_m = 1.0 - vad_p;

            T_dataType fu = fi * vad_p + fp * vad_m;
            T_dataType dfu = dfm * vad_p + dfp * vad_m;
            T_dataType dyci = remap_data.cvc1(ix, iy, iz);
            T_dataType dycu = remap_data.cvc1(ix, iym, iz) * vad_p + remap_data.cvc1(ix, iyp, iz) * vad_m;
            T_dataType dybu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ix, iyp, iz) * vad_m;

            T_dataType phi = std::abs(o_v) / dybu;

            T_dataType Da = (2.0 - phi) * std::abs(dfi) / dyci + (1.0 + phi) * std::abs(dfu) / dycu;
            Da = Da * sixth;

            T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

            T_dataType Di = sign_v * ss * portableWrapper::min({std::abs(Da) * dybu, std::abs(dfi), std::abs(dfu)});

            T_dataType rhou = remap_data.rho1(ix, iy, iz) * vad_p + remap_data.rho1(ix, iyp, iz) * vad_m;
            T_dataType dmu = std::abs(remap_data.dm(ix, iy, iz)) / dybu / rhou;

            remap_data.flux(ix, iy, iz) = (fu + Di * (1.0 - dmu)) * remap_data.dm(ix, iy, iz);
        }, Range(0, data.nx), Range(-1, data.ny), Range(0, data.nz));
    portableWrapper::fence();
}

template<auto mPtr>
void y_mom_flux(simulationData &data, remapData &remap_data)
{
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz){
    // Main flux calculation loop
        T_indexType iym = iy - 1;
        T_indexType iyp = iy + 1;
        T_indexType iyp2 = iy + 2;
        T_dataType area = data.dyac(ix, iy, iz);

        T_dataType v_advect = data.vy1(ix, iy, iz);
        T_dataType o_v = v_advect * data.dt * area;

        T_dataType fm  = (data.*mPtr)(ix, iym, iz);
        T_dataType fi  = (data.*mPtr)(ix, iy, iz);
        T_dataType fp  = (data.*mPtr)(ix, iyp, iz);
        T_dataType fp2 = (data.*mPtr)(ix, iyp2, iz);

        T_dataType dfm = fi - fm;
        T_dataType dfi = fp - fi;
        T_dataType dfp = fp2 - fp;

        T_dataType sign_v = (o_v >= 0.0) ? 1.0 : -1.0;
        T_dataType vad_p = (sign_v + 1.0) * 0.5;
        T_dataType vad_m = 1.0 - vad_p;

        T_dataType fu = fi * vad_p + fp * vad_m;
        T_dataType dfu = dfm * vad_p + dfp * vad_m;
        T_dataType dyci = data.cv1(ix, iyp, iz);
        T_dataType dycu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ix, iyp2, iz) * vad_m;
        T_dataType dybu = remap_data.cvc1(ix, iy, iz) * vad_p + remap_data.cvc1(ix, iyp, iz) * vad_m;

        T_dataType phi = std::abs(o_v) / dybu;

        T_dataType Da = (2.0 - phi) * std::abs(dfi) / dyci + (1.0 + phi) * std::abs(dfu) / dycu;
        Da = Da * sixth;

        T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

        T_dataType Di = sign_v * ss * portableWrapper::min({std::abs(Da) * dybu, std::abs(dfi), std::abs(dfu)});

        T_dataType rhou = remap_data.rho_v(ix, iy, iz) * vad_p + remap_data.rho_v(ix, iyp, iz) * vad_m;
        T_dataType dmu = std::abs(remap_data.dm(ix, iy, iz)) / dybu / rhou;

        remap_data.flux(ix, iy, iz) = fu + Di * (1.0 - dmu);
      }, Range(0, data.nx), Range(-1, data.ny), Range(0, data.nz));

        // Kinetic energy correction if rke is enabled
        if (data.rke)
        {
            portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixp = ix + 1;
                T_indexType iym = iy - 1;
                T_indexType iyp = iy + 1;
                T_indexType izp = iz + 1;

                T_dataType m = remap_data.rho_v1(ix, iy, iz) * remap_data.cv2(ix, iy, iz);
                T_dataType mp = remap_data.rho_v1(ix, iyp, iz) * remap_data.cv2(ix, iyp, iz);

                T_dataType ai = ((data.*mPtr)(ix, iy, iz) - remap_data.flux(ix, iym, iz)) * remap_data.dm(ix, iym, iz) / m - ((data.*mPtr)(ix, iy, iz) - remap_data.flux(ix, iy, iz)) * remap_data.dm(ix, iy, iz) / m;

                T_dataType aip = ((data.*mPtr)(ix, iyp, iz) - remap_data.flux(ix, iy, iz)) * remap_data.dm(ix, iy, iz) / mp - ((data.*mPtr)(ix, iyp, iz) - remap_data.flux(ix, iyp, iz)) * remap_data.dm(ix, iyp, iz) / mp;

                T_dataType dk = ((data.*mPtr)(ix, iyp, iz) - (data.*mPtr)(ix, iy, iz)) * (remap_data.flux(ix, iy, iz) - 0.5 * ((data.*mPtr)(ix, iyp, iz) + (data.*mPtr)(ix, iy, iz))) - 0.5 * ai * ((data.*mPtr)(ix, iy, iz) - remap_data.flux(ix, iy, iz)) + 0.5 * aip * ((data.*mPtr)(ix, iyp, iz) - remap_data.flux(ix, iy, iz));

                dk = dk * remap_data.dm(ix, iy, iz) * 0.5;
                data.delta_ke(ix, iyp, iz) += dk;
                data.delta_ke(ixp, iyp, iz) += dk;
                data.delta_ke(ix, iyp, izp) += dk;
                data.delta_ke(ixp, iyp, izp) += dk;
            }, Range(0, data.nx), Range(0, data.ny-1), Range(0, data.nz));
            portableWrapper::fence();
        }

    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        remap_data.flux(ix,iy,iz) *= remap_data.dm(ix,iy,iz);
        }, Range(0, data.nx), Range(-1, data.ny), Range(0, data.nz));
    portableWrapper::fence();
}
