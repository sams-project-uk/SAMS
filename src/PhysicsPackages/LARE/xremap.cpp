#include "shared_data.h"
#include "remapData.h"

void vx_by_flux(simulationData &data, remapData &remap_data);
void vx_bz_flux(simulationData &data, remapData &remap_data);
void x_mass_flux(simulationData &data, remapData &remap_data);
template<auto mPtr>
void x_energy_flux(simulationData &data, remapData &remap_data);
template<auto mPtr>
void x_mom_flux(simulationData &data, remapData &remap_data);

void simulation::remap_x(simulationData &data, remapData &remap_data) {
    using Range = portableWrapper::Range;
    portableWrapper::portableArrayManager xRemapManager;
    remap_data.flux.nullify();

    xRemapManager.allocate(remap_data.flux, Range(-2, data.nx + 2), Range(-1, data.ny + 2), Range(-1, data.nz + 2));

    portableWrapper::assign(remap_data.dm,0.0);
    portableWrapper::assign(remap_data.rho1, data.rho);

    // Main remap loop
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz){
            T_indexType ixm = ix - 1;
            T_indexType iym = iy - 1;
            T_indexType izm = iz - 1;

            // vx at Bx(i,j,k)
            T_dataType vxb = (data.vx1(ix, iy, iz) + data.vx1(ix, iym, iz) + data.vx1(ix, iy, izm) + data.vx1(ix, iym, izm)) * 0.25;

            // vx at Bx(i-1,j,k)
            T_dataType vxbm = (data.vx1(ixm, iy, iz) + data.vx1(ixm, iym, iz) + data.vx1(ixm, iy, izm) + data.vx1(ixm, iym, izm)) * 0.25;

            // vy at By(i,j,k)
            T_dataType vyb = (data.vy1(ix, iy, iz) + data.vy1(ixm, iy, iz) + data.vy1(ix, iy, izm) + data.vy1(ixm, iy, izm)) * 0.25;

            // vy at By(i,j-1,k)
            T_dataType vybm = (data.vy1(ix, iym, iz) + data.vy1(ixm, iym, iz) + data.vy1(ix, iym, izm) + data.vy1(ixm, iym, izm)) * 0.25;

            // vz at Bz(i,j,k)
            T_dataType vzb = (data.vz1(ix, iy, iz) + data.vz1(ixm, iy, iz) + data.vz1(ix, iym, iz) + data.vz1(ixm, iym, iz)) * 0.25;

            // vz at Bz(i,j,k-1)
            T_dataType vzbm = (data.vz1(ix, iy, izm) + data.vz1(ixm, iy, izm) + data.vz1(ix, iym, izm) + data.vz1(ixm, iym, izm)) * 0.25;

            T_dataType vol = data.cv(ix, iy, iz);
            T_dataType dvxdx = (vxb * data.dxab(ix, iy, iz) - vxbm * data.dxab(ixm, iy, iz)) / vol;
            T_dataType dvydy = remap_data.ypass * (vyb * data.dyab(ix, iy, iz) - vybm * data.dyab(ix, iym, iz)) / vol;
            T_dataType dvzdz = remap_data.zpass * (vzb * data.dzab(ix, iy, iz) - vzbm * data.dzab(ix, iy, izm)) / vol;

            T_dataType dv = (dvydy + dvzdz) * data.dt;

            // Control volume after remap
            remap_data.cv2(ix, iy, iz) = vol * (1.0 + dv);

            dv = dv + dvxdx * data.dt;

            // Control volume before remap
            data.cv1(ix, iy, iz) = vol * (1.0 + dv);

            // dxb before remap
            remap_data.db1(ix, iy, iz) = data.dxb(ix) + (vxb - vxbm) * data.dt;
        }, 
        Range(-1, data.nx+2), Range(-1, data.ny+2), Range(-1, data.nz+2));
    portableWrapper::fence();

    // cvc1 = vertex CV before remap
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz){
            T_indexType ixp = ix + 1;
            T_indexType iyp = iy + 1;
            T_indexType izp = iz + 1;
            remap_data.cvc1(ix, iy, iz) = 0.125 * (data.cv1(ix, iy, iz) + data.cv1(ixp, iy, iz) + data.cv1(ix, iyp, iz) + data.cv1(ixp, iyp, iz) + data.cv1(ix, iy, izp) + data.cv1(ixp, iy, izp) + data.cv1(ix, iyp, izp) + data.cv1(ixp, iyp, izp));
        }, 
    Range(-1, data.nx + 1), Range(-1, data.ny + 1), Range(-1, data.nz + 1));

    portableWrapper::fence();


    vx_by_flux(data, remap_data);   // Evans and Hawley constrained transport remap of magnetic fluxes

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            data.by(ix, iy, iz) = data.by(ix, iy, iz) - remap_data.flux(ix, iy, iz) + remap_data.flux(ixm, iy, iz);
        }
    , Range(1, data.nx), Range(0, data.ny), Range(1, data.nz));

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType iym = iy - 1;
            data.bx(ix, iy, iz) = data.bx(ix, iy, iz) + remap_data.flux(ix, iy, iz) - remap_data.flux(ix, iym, iz);
        }
    , Range(0, data.nx), Range(1, data.ny), Range(1, data.nz));
    portableWrapper::fence();

    vx_bz_flux(data, remap_data);   // Flux of bz due to vx
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            data.bz(ix, iy, iz) = data.bz(ix, iy, iz) - remap_data.flux(ix, iy, iz) + remap_data.flux(ixm, iy, iz);
        }
    , Range(0, data.nx), Range(1, data.ny), Range(0, data.nz));

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType izm = iz - 1;
            data.bx(ix, iy, iz) = data.bx(ix, iy, iz) + remap_data.flux(ix, iy, iz) - remap_data.flux(ix, iy, izm);
        }
    , Range(1, data.nx), Range(1, data.ny), Range(0, data.nz));
    portableWrapper::fence();


    x_mass_flux(data, remap_data); // Mass flux in x-direction
    dm_x_bcs(data, remap_data); // Apply boundary conditions to mass flux

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            data.rho(ix,iy,iz) = (remap_data.rho1(ix,iy,iz) * data.cv1(ix,iy,iz) + remap_data.dm(ixm,iy,iz) - remap_data.dm(ix,iy,iz)) / remap_data.cv2(ix,iy,iz);
        },
    Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));



    x_energy_flux<&simulationData::energy_electron>(data, remap_data);

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            data.energy_electron(ix, iy, iz) = 
                (
                    data.energy_electron(ix, iy, iz) * data.cv1(ix, iy, iz) * remap_data.rho1(ix, iy, iz) + 
                    remap_data.flux(ixm, iy, iz) - remap_data.flux(ix, iy, iz)
                ) / 
                (remap_data.cv2(ix, iy, iz) * data.rho(ix, iy, iz));
        },
    Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));

    portableWrapper::fence();

    x_energy_flux<&simulationData::energy_ion>(data, remap_data);

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            data.energy_ion(ix, iy, iz) = 
                (
                    data.energy_ion(ix, iy, iz) * data.cv1(ix, iy, iz) * remap_data.rho1(ix, iy, iz) + 
                    remap_data.flux(ixm, iy, iz) - remap_data.flux(ix, iy, iz)
                ) / 
                (remap_data.cv2(ix, iy, iz) * data.rho(ix, iy, iz));
        },
    Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));

    portableWrapper::fence();

  // Redefine db1, cv1, cv2, dm and vx1 for velocity (vertex) cells.
  // In some of these calculations the flux variable is used as a temporary array

    //Calculate vertex density
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixp = ix + 1;
            T_indexType iyp = iy + 1;
            T_indexType izp = iz + 1;

                remap_data.rho_v(ix, iy, iz) = (
                    remap_data.rho1(ix, iy, iz) * data.cv1(ix, iy, iz) + 
                    remap_data.rho1(ixp, iy, iz) * data.cv1(ixp, iy, iz) + 
                    remap_data.rho1(ix, iyp, iz) * data.cv1(ix, iyp, iz) + 
                    remap_data.rho1(ixp, iyp, iz) * data.cv1(ixp, iyp, iz) + 
                    remap_data.rho1(ix, iy, izp) * data.cv1(ix, iy, izp) + 
                    remap_data.rho1(ixp, iy, izp) * data.cv1(ixp, iy, izp) + 
                    remap_data.rho1(ix, iyp, izp) * data.cv1(ix, iyp, izp) + 
                    remap_data.rho1(ixp, iyp, izp) * data.cv1(ixp, iyp, izp)) * 
                    0.125 / remap_data.cvc1(ix, iy, iz);
        },
    Range(-1, data.nx+1), Range(0, data.ny), Range(0, data.nz));

    //Use flux as a temporary to store the new cv2
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixp = ix + 1;
            T_indexType iyp = iy + 1;
            T_indexType izp = iz + 1;

            remap_data.flux(ix, iy, iz) = (remap_data.cv2(ix, iy, iz) + remap_data.cv2(ixp, iy, iz) + remap_data.cv2(ix, iyp, iz) + remap_data.cv2(ixp, iyp, iz) + remap_data.cv2(ix, iy, izp) + remap_data.cv2(ixp, iy, izp) + remap_data.cv2(ix, iyp, izp) + remap_data.cv2(ixp, iyp, izp))*0.125;

        }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));

    portableWrapper::fence();

    //Now copy it back to cv2
    portableWrapper::assign(remap_data.cv2(Range(0, data.nx), Range(0, data.ny), Range(0, data.nz)), remap_data.flux(Range(0,data.nx), Range(0, data.ny), Range(0, data.nz)));

    //Now shift vx
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixp = ix + 1;

            remap_data.flux(ix,iy,iz) = (data.vx1(ix, iy, iz) + data.vx1(ixp, iy, iz)) * 0.5;
        }, Range(-2, data.nx+1), Range(0, data.ny), Range(0, data.nz));

    portableWrapper::fence();

    portableWrapper::assign(data.vx1(Range(-2, data.nx+1), Range(0, data.ny), Range(0, data.nz)), remap_data.flux(Range(-2, data.nx+1), Range(0, data.ny), Range(0, data.nz)));

    //Now shift mass flux to temporary
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixp = ix + 1;
            T_indexType iyp = iy + 1;
            T_indexType izp = iz + 1;

            remap_data.flux(ix, iy, iz) = (remap_data.dm(ix, iy, iz) + remap_data.dm(ixp, iy, iz) + remap_data.dm(ix, iyp, iz) + remap_data.dm(ixp, iyp, iz) + remap_data.dm(ix, iy, izp) + remap_data.dm(ixp, iy, izp) + remap_data.dm(ix, iyp, izp) + remap_data.dm(ixp, iyp, izp))*0.125;
        }, Range(-1, data.nx), Range(0, data.ny), Range(0, data.nz));

    portableWrapper::fence();

    //And copy back to dm
    portableWrapper::assign(remap_data.dm(Range(-1, data.nx), Range(0, data.ny), Range(0, data.nz)), remap_data.flux(Range(-1, data.nx), Range(0, data.ny), Range(0, data.nz)));

    portableWrapper::fence();

    //Calculate vertex density after remap
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            // Vertex density after remap
            remap_data.rho_v1(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.cv1(ix, iy, iz) + remap_data.dm(ixm, iy, iz) - remap_data.dm(ix, iy, iz)) / remap_data.cv2(ix, iy, iz);
        }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
    portableWrapper::fence();


    x_mom_flux<&simulationData::vx>(data, remap_data); // Momentum flux in x-direction

    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            data.vx(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.vx(ix, iy, iz) * data.cv1(ix, iy, iz) + remap_data.flux(ixm, iy, iz) - remap_data.flux(ix, iy, iz)) / (remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz));
        }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));

    portableWrapper::fence();

    x_mom_flux<&simulationData::vy>(data, remap_data); // Momentum flux in y-direction
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            data.vy(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.vy(ix, iy, iz) * data.cv1(ix, iy, iz) + remap_data.flux(ixm, iy, iz) - remap_data.flux(ix, iy, iz)) / (remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz));
        }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
    portableWrapper::fence();

    x_mom_flux<&simulationData::vz>(data, remap_data); // Momentum flux in z-direction
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            data.vz(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.vz(ix, iy, iz) * data.cv1(ix, iy, iz) + remap_data.flux(ixm, iy, iz) - remap_data.flux(ix, iy, iz)) / (remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz));
        }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
    portableWrapper::fence();
    remap_data.xpass = 0;
    
    this->boundary_conditions(data);
} //END simulation::remap_x

//Flux of by due to vx
void vx_by_flux(simulationData &data, remapData &remap_data) {
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType ixp = ix + 1;
        T_indexType iyp = iy + 1;
        T_indexType izm = iz - 1;
        T_indexType ixm = ix - 1;
        T_indexType ixp2 = ix + 2;

        T_dataType v_advect = (data.vx1(ix, iy, iz) + data.vx1(ix, iy, izm)) * 0.5;
        T_dataType o_v = v_advect * data.dt;

        T_dataType dbx = (remap_data.db1(ix, iy, iz) + remap_data.db1(ix, iyp, iz)) * 0.5;
        T_dataType dbxp = (remap_data.db1(ixp, iy, iz) + remap_data.db1(ixp, iyp, iz)) * 0.5;
        T_dataType dbxp2 = (remap_data.db1(ixp2, iy, iz) + remap_data.db1(ixp2, iyp, iz)) * 0.5;
        T_dataType dbxm = (remap_data.db1(ixm, iy, iz) + remap_data.db1(ixm, iyp, iz)) * 0.5;

        T_dataType fm = data.by(ixm, iy, iz) / dbxm;
        T_dataType fi = data.by(ix, iy, iz) / dbx;
        T_dataType fp = data.by(ixp, iy, iz) / dbxp;
        T_dataType fp2 = data.by(ixp2, iy, iz) / dbxp2;

        T_dataType dfm = fi - fm;
        T_dataType dfi = fp - fi;
        T_dataType dfp = fp2 - fp;

        T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
        T_dataType vad_p = (sign_v + 1.0) * 0.5;
        T_dataType vad_m = 1.0 - vad_p;

        T_dataType fu = fi * vad_p + fp * vad_m;
        T_dataType dfu = dfm * vad_p + dfp * vad_m;
        T_dataType dxci = remap_data.cvc1(ix, iy, iz);
        T_dataType dxcu = remap_data.cvc1(ixm, iy, iz) * vad_p + remap_data.cvc1(ixp, iy, iz) * vad_m;
        T_dataType dxbu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ixp, iy, iz) * vad_m;

        T_dataType dxu = dbx * vad_p + dbxp * vad_m;
        T_dataType phi = std::abs(o_v) / dxu;

        T_dataType Da = (2.0 - phi) * std::abs(dfi) / dxci + (1.0 + phi) * std::abs(dfu) / dxcu;
        Da = Da * sixth;

        T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

        T_dataType Di = sign_v * ss * portableWrapper::min({std::abs(Da) * dxbu, std::abs(dfi), std::abs(dfu)});

        remap_data.flux(ix, iy, iz) = (fu + Di * (1.0 - phi)) * o_v;
      }
    , Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));

    portableWrapper::fence();
  }

  //Flux of bz due to vx
  void vx_bz_flux(simulationData &data, remapData &remap_data) {
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType ixm = ix - 1;
        T_indexType ixp = ix + 1;
        T_indexType ixp2 = ix + 2;
        T_indexType iym = iy - 1;
        T_indexType izp = iz + 1;

        T_dataType v_advect = (data.vx1(ix, iy, iz) + data.vx1(ix, iym, iz)) * 0.5;
        T_dataType o_v = v_advect * data.dt;

        T_dataType dbx = (remap_data.db1(ix, iy, iz) + remap_data.db1(ix, iy, izp)) * 0.5;
        T_dataType dbxp = (remap_data.db1(ixp, iy, iz) + remap_data.db1(ixp, iy, izp)) * 0.5;
        T_dataType dbxp2 = (remap_data.db1(ixp2, iy, iz) + remap_data.db1(ixp2, iy, izp)) * 0.5;
        T_dataType dbxm = (remap_data.db1(ixm, iy, iz) + remap_data.db1(ixm, iy, izp)) * 0.5;

        T_dataType fm = data.bz(ixm, iy, iz) / dbxm;
        T_dataType fi = data.bz(ix, iy, iz) / dbx;
        T_dataType fp = data.bz(ixp, iy, iz) / dbxp;
        T_dataType fp2 = data.bz(ixp2, iy, iz) / dbxp2;

        T_dataType dfm = fi - fm;
        T_dataType dfi = fp - fi;
        T_dataType dfp = fp2 - fp;

        T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
        T_dataType vad_p = (sign_v + 1.0) * 0.5;
        T_dataType vad_m = 1.0 - vad_p;

        T_dataType fu = fi * vad_p + fp * vad_m;
        T_dataType dfu = dfm * vad_p + dfp * vad_m;
        T_dataType dxci = remap_data.cvc1(ix, iy, iz);
        T_dataType dxcu = remap_data.cvc1(ixm, iy, iz) * vad_p + remap_data.cvc1(ixp, iy, iz) * vad_m;
        T_dataType dxbu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ixp, iy, iz) * vad_m;

        T_dataType dxu = dbx * vad_p + dbxp * vad_m;
        T_dataType phi = std::abs(o_v) / dxu;

        T_dataType Da = (2.0 - phi) * std::abs(dfi) / dxci + (1.0 + phi) * std::abs(dfu) / dxcu;
        Da = Da * sixth;

        T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

        T_dataType Di = sign_v * ss * portableWrapper::min({std::abs(Da) * dxbu, std::abs(dfi), std::abs(dfu)});

        remap_data.flux(ix, iy, iz) = (fu + Di * (1.0 - phi)) * o_v;
    }
    , Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
    portableWrapper::fence();
}

void x_mass_flux(simulationData &data, remapData &remap_data)
{
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType ixm = ix - 1;
        T_indexType ixp = ix + 1;
        T_indexType ixp2 = ix + 2;
        T_indexType iym = iy - 1;
        T_indexType izm = iz - 1;
        T_dataType area = data.dxab(ix, iy, iz);

        T_dataType v_advect = (data.vx1(ix, iy, iz) + data.vx1(ix, iym, iz) +
                           data.vx1(ix, iy, izm) + data.vx1(ix, iym, izm)) *
                          0.25;
        T_dataType o_v = v_advect * data.dt * area;

        T_dataType fm = data.rho(ixm, iy, iz);
        T_dataType fi = data.rho(ix, iy, iz);
        T_dataType fp = data.rho(ixp, iy, iz);
        T_dataType fp2 = data.rho(ixp2, iy, iz);

        T_dataType dfm = fi - fm;
        T_dataType dfi = fp - fi;
        T_dataType dfp = fp2 - fp;

        T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
        T_dataType vad_p = (sign_v + 1.0) * 0.5;
        T_dataType vad_m = 1.0 - vad_p;

        T_dataType fu = fi * vad_p + fp * vad_m;
        T_dataType dfu = dfm * vad_p + dfp * vad_m;
        T_dataType dxci = remap_data.cvc1(ix, iy, iz);
        T_dataType dxcu = remap_data.cvc1(ixm, iy, iz) * vad_p + remap_data.cvc1(ixp, iy, iz) * vad_m;
        T_dataType dxbu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ixp, iy, iz) * vad_m;

        T_dataType phi = std::abs(o_v) / dxbu;

        T_dataType Da = (2.0 - phi) * std::abs(dfi) / dxci + (1.0 + phi) * std::abs(dfu) / dxcu;
        Da = Da * sixth;

        T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

        T_dataType Di = sign_v * ss * portableWrapper::min({std::abs(Da) * dxbu, std::abs(dfi), std::abs(dfu)});

        remap_data.dm(ix, iy, iz) = (fu + Di * (1.0 - phi)) * o_v;
    }, Range(0, data.nx), Range(0, data.ny+1), Range(0, data.nz+1));
    portableWrapper::fence();
}

/**
 * This has been designed to be both x_energy_electron_flux and x_energy_ion_flux.
 * The template parameter mPtr allows us to pass in the member function pointer
 * for the appropriate energy type.
 */
template<auto mPtr>
void x_energy_flux(simulationData &data, remapData &remap_data) {
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm = ix - 1;
            T_indexType ixp = ix + 1;
            T_indexType ixp2 = ix + 2;
            T_indexType iym = iy - 1;
            T_indexType izm = iz - 1;

            T_dataType area = data.dxab(ix, iy, iz);

            T_dataType v_advect = (data.vx1(ix, iy, iz) + data.vx1(ix, iym, iz) +
                                data.vx1(ix, iy, izm) + data.vx1(ix, iym, izm)) *
                                0.25;
            T_dataType o_v = v_advect * data.dt * area;

            T_dataType fm = (data.*mPtr)(ixm, iy, iz);
            T_dataType fi = (data.*mPtr)(ix, iy, iz);
            T_dataType fp = (data.*mPtr)(ixp, iy, iz);
            T_dataType fp2 = (data.*mPtr)(ixp2, iy, iz);

            T_dataType dfm = fi - fm;
            T_dataType dfi = fp - fi;
            T_dataType dfp = fp2 - fp;

            T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
            T_dataType vad_p = (sign_v + 1.0) * 0.5;
            T_dataType vad_m = 1.0 - vad_p;

            T_dataType fu = fi * vad_p + fp * vad_m;
            T_dataType dfu = dfm * vad_p + dfp * vad_m;
            T_dataType dxci = remap_data.cvc1(ix, iy, iz);
            T_dataType dxcu = remap_data.cvc1(ixm, iy, iz) * vad_p + remap_data.cvc1(ixp, iy, iz) * vad_m;
            T_dataType dxbu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ixp, iy, iz) * vad_m;

            T_dataType phi = std::abs(o_v) / dxbu;

            T_dataType Da = (2.0 - phi) * std::abs(dfi) / dxci + (1.0 + phi) * std::abs(dfu) / dxcu;
            Da = Da * sixth;

            T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

            T_dataType Di = sign_v * ss * portableWrapper::min({std::abs(Da) * dxbu, std::abs(dfi), std::abs(dfu)});

            T_dataType rhou = remap_data.rho1(ix, iy, iz) * vad_p + remap_data.rho1(ixp, iy, iz) * vad_m;
            T_dataType dmu = std::abs(remap_data.dm(ix, iy, iz)) / dxbu / rhou;

            remap_data.flux(ix, iy, iz) = (fu + Di * (1.0 - dmu)) * remap_data.dm(ix, iy, iz);
        }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
    portableWrapper::fence();
}

template<auto mPtr>
void x_mom_flux(simulationData &data, remapData &remap_data) {
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_indexType ixm  = ix - 1;
            T_indexType ixp  = ix + 1;
            T_indexType ixp2 = ix + 2;
            T_dataType area = data.dxac(ix, iy, iz);

            T_dataType v_advect = data.vx1(ix, iy, iz);
            T_dataType o_v = v_advect * data.dt * area;

            T_dataType fm = (data.*mPtr)(ixm, iy, iz);
            T_dataType fi = (data.*mPtr)(ix, iy, iz);
            T_dataType fp = (data.*mPtr)(ixp, iy, iz);
            T_dataType fp2 = (data.*mPtr)(ixp2, iy, iz);

            T_dataType dfm = fi - fm;
            T_dataType dfi = fp - fi;
            T_dataType dfp = fp2 - fp;

            T_dataType sign_v = (o_v >= 0.0) ? 1.0 : -1.0;
            T_dataType vad_p = (sign_v + 1.0) * 0.5;
            T_dataType vad_m = 1.0 - vad_p;

            T_dataType fu = fi * vad_p + fp * vad_m;
            T_dataType dfu = dfm * vad_p + dfp * vad_m;
            T_dataType dxci = data.cv1(ixp, iy, iz);
            T_dataType dxcu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ixp2, iy, iz) * vad_m;
            T_dataType dxbu = remap_data.cvc1(ix, iy, iz) * vad_p + remap_data.cvc1(ixp, iy, iz) * vad_m;

            T_dataType phi = std::abs(o_v) / dxbu;

            T_dataType Da = (2.0 - phi) * std::abs(dfi) / dxci + (1.0 + phi) * std::abs(dfu) / dxcu;
            Da = Da * sixth;

            T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

            T_dataType Di = sign_v * ss * portableWrapper::min({std::abs(Da) * dxbu, std::abs(dfi), std::abs(dfu)});

            T_dataType rhou = remap_data.rho_v(ix, iy, iz) * vad_p + remap_data.rho_v(ixp, iy, iz) * vad_m;
            T_dataType dmu = std::abs(remap_data.dm(ix, iy, iz)) / dxbu / rhou;

            remap_data.flux(ix, iy, iz) = fu + Di * (1.0 - dmu);
        }, Range(-1, data.nx), Range(0, data.ny), Range(0, data.nz));
    portableWrapper::fence();
    if (data.rke)
    {
        portableWrapper::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixm = ix - 1;
                T_indexType ixp = ix + 1;
                T_indexType iyp = iy + 1;
                T_indexType izp = iz + 1;
                T_dataType m = remap_data.rho_v1(ix, iy, iz) * remap_data.cv2(ix, iy, iz);
                T_dataType mp = remap_data.rho_v1(ixp, iy, iz) * remap_data.cv2(ixp, iy, iz);

                T_dataType ai = ((data.*mPtr)(ix, iy, iz) - remap_data.flux(ixm, iy, iz)) * remap_data.dm(ixm, iy, iz) / m - ((data.*mPtr)(ix, iy, iz) - remap_data.flux(ix, iy, iz)) * remap_data.dm(ix, iy, iz) / m;

                T_dataType aip = ((data.*mPtr)(ixp, iy, iz) - remap_data.flux(ix, iy, iz)) * remap_data.dm(ix, iy, iz) / mp - ((data.*mPtr)(ixp, iy, iz) - remap_data.flux(ixp, iy, iz)) * remap_data.dm(ixp, iy, iz) / mp;

                T_dataType dk = ((data.*mPtr)(ixp, iy, iz) - (data.*mPtr)(ix, iy, iz)) * (remap_data.flux(ix, iy, iz) - 0.5 * ((data.*mPtr)(ixp, iy, iz) + (data.*mPtr)(ix, iy, iz))) - 0.5 * ai * ((data.*mPtr)(ix, iy, iz) - remap_data.flux(ix, iy, iz)) + 0.5 * aip * ((data.*mPtr)(ixp, iy, iz) - remap_data.flux(ix, iy, iz));

                dk = dk * remap_data.dm(ix, iy, iz) * 0.5;
                portableWrapper::atomic::accelerated::Add(data.delta_ke(ixp, iy, iz), dk);
                portableWrapper::atomic::accelerated::Add(data.delta_ke(ixp, iyp, iz), dk);
                portableWrapper::atomic::accelerated::Add(data.delta_ke(ixp, iyp, izp), dk);
                portableWrapper::atomic::accelerated::Add(data.delta_ke(ixp, iy, izp), dk);
            }, Range(0, data.nx-1), Range(0, data.ny), Range(0, data.nz));
    }
    portableWrapper::fence();
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            remap_data.flux(ix,iy,iz) *= remap_data.dm(ix,iy,iz);
        }, Range(-1, data.nx), Range(0, data.ny), Range(0, data.nz));
    portableWrapper::fence();
}
