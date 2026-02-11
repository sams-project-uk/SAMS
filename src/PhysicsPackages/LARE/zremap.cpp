#include "shared_data.h"
#include "remapData.h"

namespace LARE
{

    namespace pw = portableWrapper;

    void vz_bx_flux(simulationData &data, remapData &remap_data);
    void vz_by_flux(simulationData &data, remapData &remap_data);
    void z_mass_flux(simulationData &data, remapData &remap_data);
    template <auto mPtr>
    void z_energy_flux(simulationData &data, remapData &remap_data);
    template <auto mPtr>
    void z_mom_flux(simulationData &data, remapData &remap_data);

    void LARE3D::remap_z(simulationData &data, remapData &remap_data)
    {
        using Range = pw::Range;
        pw::portableArrayManager zRemapManager;
        remap_data.flux.nullify();

        zRemapManager.allocate(remap_data.flux, Range(-1, data.nx + 2), Range(-1, data.ny + 2), Range(-2, data.nz + 2));

        pw::assign(data.dm, 0.0);
        pw::assign(remap_data.rho1, data.rho);

        pw::applyKernel(
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
                T_dataType dvydy = remap_data.ypass * (vyb * data.dyab(ix, iy, iz) - vybm * data.dyab(ix, iym, iz)) / vol;
                T_dataType dvzdz = (vzb * data.dzab(ix, iy, iz) - vzbm * data.dzab(ix, iy, izm)) / vol;

                T_dataType dv = (dvxdx + dvydy) * data.dt;

                // Control volume after remap
                remap_data.cv2(ix, iy, iz) = vol * (1.0 + dv);

                dv = dv + dvzdz * data.dt;

                // Control volume before remap
                data.cv1(ix, iy, iz) = vol * (1.0 + dv);

                // dyb before remap
                remap_data.db1(ix, iy, iz) = data.hzc(ix, iy) * data.dzb(iz) + (vzb - vzbm) * data.dt;
            },
            Range(-1, data.nx + 2), Range(-1, data.ny + 2), Range(-1, data.nz + 2));
        pw::fence();

        // cvc1 = vertex CV before remap
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixp = ix + 1;
                T_indexType iyp = iy + 1;
                T_indexType izp = iz + 1;
                remap_data.cvc1(ix, iy, iz) =
                    0.125 * (data.cv1(ix, iy, iz) + data.cv1(ixp, iy, iz) +
                             data.cv1(ix, iyp, iz) + data.cv1(ixp, iyp, iz) +
                             data.cv1(ix, iy, izp) + data.cv1(ixp, iy, izp) +
                             data.cv1(ix, iyp, izp) + data.cv1(ixp, iyp, izp));
            },
            Range(-1, data.nx + 1), Range(-1, data.ny + 1), Range(-1, data.nz + 1));
        pw::fence();

        // Evans and Hawley (ApJ, vol 332, p650, (1988))
        // constrained transport remap of magnetic fluxes
        vz_bx_flux(data, remap_data);

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        data.bx(ix, iy, iz) = data.bx(ix, iy, iz) - remap_data.flux(ix,iy,iz) + remap_data.flux(ix,iy,izm); }, Range(0, data.nx), Range(1, data.ny), Range(1, data.nz));

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType ixm = ix - 1;
        data.bz(ix, iy, iz) = data.bz(ix, iy, iz) + remap_data.flux(ix,iy,iz) - remap_data.flux(ixm,iy,iz); }, Range(1, data.nx), Range(1, data.ny), Range(0, data.nz));

        pw::fence();

        vz_by_flux(data, remap_data);

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        data.by(ix, iy, iz) = data.by(ix, iy, iz) - remap_data.flux(ix,iy,iz) + remap_data.flux(ix,iy,izm); }, Range(1, data.nx), Range(0, data.ny), Range(1, data.nz));

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType iym = iy - 1;
        data.bz(ix, iy, iz) = data.bz(ix, iy, iz) + remap_data.flux(ix,iy,iz) - remap_data.flux(ix,iym,iz); }, Range(1, data.nx), Range(1, data.ny), Range(0, data.nz));
        pw::fence();

        z_mass_flux(data, remap_data);
        dm_z_bcs();

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        data.rho(ix,iy,iz) = (remap_data.rho1(ix,iy,iz) * data.cv1(ix,iy,iz) + data.dm(ix,iy,izm) - data.dm(ix,iy,iz)) / remap_data.cv2(ix,iy,iz); }, Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
        pw::fence();

        z_energy_flux<&simulationData::energy_electron>(data, remap_data);
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        data.energy_electron(ix, iy, iz) = (data.energy_electron(ix, iy, iz) * data.cv1(ix, iy, iz) * remap_data.rho1(ix, iy, iz) + remap_data.flux(ix, iy, izm) - remap_data.flux(ix, iy, iz)) / (remap_data.cv2(ix, iy, iz) * data.rho(ix, iy, iz)); }, Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
        pw::fence();

        z_energy_flux<&simulationData::energy_ion>(data, remap_data);
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        data.energy_ion(ix, iy, iz) = (data.energy_ion(ix, iy, iz) * data.cv1(ix, iy, iz) * remap_data.rho1(ix, iy, iz) + remap_data.flux(ix, iy, izm) - remap_data.flux(ix, iy, iz)) / (remap_data.cv2(ix, iy, iz) * data.rho(ix, iy, iz)); }, Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
        pw::fence();

        // Redefine dyb1, cv1, cv2, dm and vz1 for velocity (vertex) cells.
        // In some of these calculations the flux variable is used as a temporary array

        pw::applyKernel(
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
            },
            Range(0, data.nx), Range(0, data.ny), Range(-1, data.nz + 1));
        pw::fence();

        // Move cv2 to vertex using flux array as a temporary
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixp = ix + 1;
                T_indexType iyp = iy + 1;
                T_indexType izp = iz + 1;

                remap_data.flux(ix, iy, iz) = 0.125 * (remap_data.cv2(ix, iy, iz) + remap_data.cv2(ixp, iy, iz) +
                                                       remap_data.cv2(ix, iyp, iz) + remap_data.cv2(ixp, iyp, iz) +
                                                       remap_data.cv2(ix, iy, izp) + remap_data.cv2(ixp, iy, izp) +
                                                       remap_data.cv2(ix, iyp, izp) + remap_data.cv2(ixp, iyp, izp));
            },
            Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();
        // Now copy it back
        pw::assign(remap_data.cv2(Range(0, data.nx), Range(0, data.ny), Range(0, data.nz)), remap_data.flux(Range(0, data.nx), Range(0, data.ny), Range(0, data.nz)));
        pw::fence();

        // Move vz1 to z face centred for momentum remap
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType izp = iz + 1;
                remap_data.flux(ix, iy, iz) = (data.vz1(ix, iy, iz) + data.vz1(ix, iy, izp)) * 0.5;
            },
            Range(0, data.nx), Range(0, data.ny), Range(-2, data.nz + 1));
        pw::fence();
        // And copy it back
        pw::assign(data.vz1(Range(0, data.nx), Range(0, data.ny), Range(-2, data.nz + 1)), remap_data.flux(Range(0, data.nx), Range(0, data.ny), Range(-2, data.nz + 1)));
        pw::fence();

        // Vertex control volume mass change
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixp = ix + 1;
                T_indexType iyp = iy + 1;
                T_indexType izp = iz + 1;

                remap_data.flux(ix, iy, iz) = 0.125 * (data.dm(ix, iy, iz) + data.dm(ixp, iy, iz) +
                                                       data.dm(ix, iyp, iz) + data.dm(ixp, iyp, iz) +
                                                       data.dm(ix, iy, izp) + data.dm(ixp, iy, izp) +
                                                       data.dm(ix, iyp, izp) + data.dm(ixp, iyp, izp));
            },
            Range(0, data.nx), Range(0, data.ny), Range(-1, data.nz));
        pw::fence();

        pw::assign(data.dm(Range(0, data.nx), Range(0, data.ny), Range(-1, data.nz)), remap_data.flux(Range(0, data.nx), Range(0, data.ny), Range(-1, data.nz)));
        pw::fence();

        // Update vertex density after remap
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType izm = iz - 1;
                remap_data.rho_v1(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * remap_data.cvc1(ix, iy, iz) + data.dm(ix, iy, izm) - data.dm(ix, iy, iz)) /remap_data.cv2(ix, iy, iz);
            },
            Range(0, data.nx), Range(0, data.ny), Range(-1, data.nz+1));
        pw::fence();

        z_mom_flux<&simulationData::vx>(data, remap_data);
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        data.vx(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.vx(ix, iy, iz) * remap_data.cvc1(ix, iy, iz) + remap_data.flux(ix, iy, izm) - remap_data.flux(ix, iy, iz)) / (remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz)); }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();

        z_mom_flux<&simulationData::vy>(data, remap_data);
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
        data.vy(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.vy(ix, iy, iz) * remap_data.cvc1(ix, iy, iz) + remap_data.flux(ix, iy, izm) - remap_data.flux(ix, iy, iz)) / (remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz)); }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();

        z_mom_flux<&simulationData::vz>(data, remap_data);
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz -1;
        data.vz(ix, iy, iz) = (remap_data.rho_v(ix, iy, iz) * data.vz(ix, iy, iz) * remap_data.cvc1(ix, iy, iz) + remap_data.flux(ix, iy, izm) - remap_data.flux(ix, iy, iz)) / (remap_data.cv2(ix, iy, iz) * remap_data.rho_v1(ix, iy, iz)); }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();

        this->boundary_conditions();
        pw::fence();
        remap_data.zpass = 0.0;

    } // END LARE3D::remap_z

    // Evans & Hawley constrained transport remap of vz*Bx fluxes
    void vz_bx_flux(simulationData &data, remapData &remap_data)
    {
        using Range = pw::Range;
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType ixp = ix + 1;
        T_indexType iym = iy - 1;
        T_indexType izm = iz - 1;
        T_indexType izp = iz + 1;
        T_indexType izp2 = iz + 2;

        T_dataType v_advect = (data.vz1(ix, iy, iz) + data.vz1(ix, iym, iz)) * 0.5;
        T_dataType o_v = v_advect * data.dt;

        T_dataType dbz = (remap_data.db1(ix, iy, iz) + remap_data.db1(ixp, iy, iz)) * 0.5;
        T_dataType dbzp = (remap_data.db1(ix, iy, izp) + remap_data.db1(ixp, iy, izp)) * 0.5;
        T_dataType dbzp2 = (remap_data.db1(ix, iy, izp2) + remap_data.db1(ixp, iy, izp2)) * 0.5;
        T_dataType dbzm = (remap_data.db1(ix, iy, izm) + remap_data.db1(ixp, iy, izm)) * 0.5;

        T_dataType fm = data.bx(ix, iy, izm) / dbzm;
        T_dataType fi = data.bx(ix, iy, iz) / dbz;
        T_dataType fp = data.bx(ix, iy, izp) / dbzp;
        T_dataType fp2 = data.bx(ix, iy, izp2) / dbzp2;

        T_dataType dfm = fi - fm;
        T_dataType dfi = fp - fi;
        T_dataType dfp = fp2 - fp;

        T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
        T_dataType vad_p = (sign_v + 1.0) * 0.5;
        T_dataType vad_m = 1.0 - vad_p;

        T_dataType fu = fi * vad_p + fp * vad_m;
        T_dataType dfu = dfm * vad_p + dfp * vad_m;
        T_dataType dzci = remap_data.cvc1(ix, iy, iz);
        T_dataType dzcu = remap_data.cvc1(ix, iy, izm) * vad_p + remap_data.cvc1(ix, iy, izp) * vad_m;
        T_dataType dzbu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ix, iy, izp) * vad_m;

        T_dataType dyu = dbz * vad_p + dbzp * vad_m;
        T_dataType phi = std::abs(o_v) / dyu;

        T_dataType Da = (2.0 - phi) * std::abs(dfi) / dzci + (1.0 + phi) * std::abs(dfu) / dzcu;
        Da = Da * sixth;

        T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

        T_dataType Di = sign_v * ss * pw::min({std::abs(Da) * dzbu, std::abs(dfi), std::abs(dfu)});

        remap_data.flux(ix, iy, iz) = (fu + Di * (1.0 - phi)) * o_v; }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();
    }

    // Evans & Hawley constrained transport remap of vz*Bx fluxes
    void vz_by_flux(simulationData &data, remapData &remap_data)
    {
        using Range = pw::Range;
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType ixm = ix - 1;
        T_indexType iyp = iy + 1;
        T_indexType izm = iz - 1;
        T_indexType izp = iz + 1;
        T_indexType izp2 = iz + 2;

        T_dataType v_advect = (data.vz1(ix, iy, iz) + data.vz1(ixm, iy, iz)) * 0.5;
        T_dataType o_v = v_advect * data.dt;

        T_dataType dbz = (remap_data.db1(ix, iy, iz) + remap_data.db1(ix, iyp, iz)) * 0.5;
        T_dataType dbzp = (remap_data.db1(ix, iy, izp) + remap_data.db1(ix, iyp, izp)) * 0.5;
        T_dataType dbzp2 = (remap_data.db1(ix, iy, izp2) + remap_data.db1(ix, iyp, izp2)) * 0.5;
        T_dataType dbzm = (remap_data.db1(ix, iy, izm) + remap_data.db1(ix, iyp, izm)) * 0.5;

        T_dataType fm = data.by(ix, iy, izm) / dbzm;
        T_dataType fi = data.by(ix, iy, iz) / dbz;
        T_dataType fp = data.by(ix, iy, izp) / dbzp;
        T_dataType fp2 = data.by(ix, iy, izp2) / dbzp2;

        T_dataType dfm = fi - fm;
        T_dataType dfi = fp - fi;
        T_dataType dfp = fp2 - fp;

        T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
        T_dataType vad_p = (sign_v + 1.0) * 0.5;
        T_dataType vad_m = 1.0 - vad_p;

        T_dataType fu = fi * vad_p + fp * vad_m;
        T_dataType dfu = dfm * vad_p + dfp * vad_m;
        T_dataType dzci = remap_data.cvc1(ix, iy, iz);
        T_dataType dzcu = remap_data.cvc1(ix, iy, izm) * vad_p + remap_data.cvc1(ix, iy, izp) * vad_m;
        T_dataType dzbu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ix, iy, izp) * vad_m;

        T_dataType dyu = dbz * vad_p + dbzp * vad_m;
        T_dataType phi = std::abs(o_v) / dyu;

        T_dataType Da = (2.0 - phi) * std::abs(dfi) / dzci + (1.0 + phi) * std::abs(dfu) / dzcu;
        Da = Da * sixth;

        T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

        T_dataType Di = sign_v * ss * pw::min({std::abs(Da) * dzbu, std::abs(dfi), std::abs(dfu)});

        remap_data.flux(ix, iy, iz) = (fu + Di * (1.0 - phi)) * o_v; }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();
    }

    void z_mass_flux(simulationData &data, remapData &remap_data)
    {
        using Range = pw::Range;
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixm = ix - 1;
                T_indexType iym = iy - 1;
                T_indexType izm = iz - 1;
                T_indexType izp = iz + 1;
                T_indexType izp2 = iz + 2;
                T_dataType area = data.dzab(ix, iy, iz);

                T_dataType v_advect = (data.vz1(ixm, iy, iz) + data.vz1(ix, iy, iz) +
                                       data.vz1(ixm, iym, iz) + data.vz1(ix, iym, iz)) *
                                      0.25;
                T_dataType o_v = v_advect * data.dt * area;

                T_dataType fm = data.rho(ix, iy, izm);
                T_dataType fi = data.rho(ix, iy, iz);
                T_dataType fp = data.rho(ix, iy, izp);
                T_dataType fp2 = data.rho(ix, iy, izp2);

                T_dataType dfm = fi - fm;
                T_dataType dfi = fp - fi;
                T_dataType dfp = fp2 - fp;

                T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
                T_dataType vad_p = (sign_v + 1.0) * 0.5;
                T_dataType vad_m = 1.0 - vad_p;

                T_dataType fu = fi * vad_p + fp * vad_m;
                T_dataType dfu = dfm * vad_p + dfp * vad_m;
                T_dataType dzci = remap_data.cvc1(ix, iy, iz);
                T_dataType dzcu = remap_data.cvc1(ix, iy, izm) * vad_p + remap_data.cvc1(ix, iy, izp) * vad_m;
                T_dataType dzbu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ix, iy, izp) * vad_m;

                T_dataType phi = std::abs(o_v) / dzbu;

                T_dataType Da = (2.0 - phi) * std::abs(dfi) / dzci + (1.0 + phi) * std::abs(dfu) / dzcu;
                Da = Da * sixth;

                T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

                T_dataType Di = sign_v * ss * pw::min({std::abs(Da) * dzbu, std::abs(dfi), std::abs(dfu)});

                data.dm(ix, iy, iz) = (fu + Di * (1.0 - phi)) * o_v;
            },
            Range(0, data.nx + 1), Range(0, data.ny + 1), Range(0, data.nz));

        pw::fence();
    }

    /**
     * This has been designed to be both y_energy_electron_flux and y_energy_ion_flux.
     * The template parameter mPtr allows us to pass in the member function pointer
     * for the appropriate energy type.
     */
    template <auto mPtr>
    void z_energy_flux(simulationData &data, remapData &remap_data)
    {
        using Range = pw::Range;
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixm = ix - 1;
                T_indexType iym = iy - 1;
                T_indexType izm = iz - 1;
                T_indexType izp = iz + 1;
                T_indexType izp2 = iz + 2;

                T_dataType area = data.dzab(ix, iy, iz);

                T_dataType v_advect = (data.vz1(ixm, iy, iz) + data.vz1(ix, iy, iz) +
                                       data.vz1(ixm, iym, iz) + data.vz1(ix, iym, iz)) *
                                      0.25;
                T_dataType o_v = v_advect * data.dt * area;

                T_dataType fm = (data.*mPtr)(ix, iy, izm);
                T_dataType fi = (data.*mPtr)(ix, iy, iz);
                T_dataType fp = (data.*mPtr)(ix, iy, izp);
                T_dataType fp2 = (data.*mPtr)(ix, iy, izp2);

                T_dataType dfm = fi - fm;
                T_dataType dfi = fp - fi;
                T_dataType dfp = fp2 - fp;

                T_dataType sign_v = (v_advect >= 0.0) ? 1.0 : -1.0;
                T_dataType vad_p = (sign_v + 1.0) * 0.5;
                T_dataType vad_m = 1.0 - vad_p;

                T_dataType fu = fi * vad_p + fp * vad_m;
                T_dataType dfu = dfm * vad_p + dfp * vad_m;
                T_dataType dzci = remap_data.cvc1(ix, iy, iz);
                T_dataType dzcu = remap_data.cvc1(ix, iy, izm) * vad_p + remap_data.cvc1(ix, iy, izp) * vad_m;
                T_dataType dzbu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ix, iy, izp) * vad_m;

                T_dataType phi = std::abs(o_v) / dzbu;

                T_dataType Da = (2.0 - phi) * std::abs(dfi) / dzci + (1.0 + phi) * std::abs(dfu) / dzcu;
                Da = Da * sixth;

                T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

                T_dataType Di = sign_v * ss * pw::min({std::abs(Da) * dzbu, std::abs(dfi), std::abs(dfu)});

                T_dataType rhou = remap_data.rho1(ix, iy, iz) * vad_p + remap_data.rho1(ix, iy, izp) * vad_m;
                T_dataType dmu = std::abs(data.dm(ix, iy, iz)) / dzbu / rhou;

                remap_data.flux(ix, iy, iz) = (fu + Di * (1.0 - dmu)) * data.dm(ix, iy, iz);
            },
            Range(0, data.nx), Range(0, data.ny), Range(0, data.nz));
        pw::fence();
    }

    template <auto mPtr>
    void z_mom_flux(simulationData &data, remapData &remap_data)
    {
        using Range = pw::Range;
        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
    // Main flux calculation loop
        T_indexType izm = iz - 1;
        T_indexType izp = iz + 1;
        T_indexType izp2 = iz + 2;
        T_dataType area = data.dyac(ix, iy, iz);

        T_dataType v_advect = data.vz1(ix, iy, iz);
        T_dataType o_v = v_advect * data.dt * area;

        T_dataType fm  = (data.*mPtr)(ix, iy, izm);
        T_dataType fi  = (data.*mPtr)(ix, iy, iz);
        T_dataType fp  = (data.*mPtr)(ix, iy, izp);
        T_dataType fp2 = (data.*mPtr)(ix, iy, izp2);

        T_dataType dfm = fi - fm;
        T_dataType dfi = fp - fi;
        T_dataType dfp = fp2 - fp;

        T_dataType sign_v = (o_v >= 0.0) ? 1.0 : -1.0;
        T_dataType vad_p = (sign_v + 1.0) * 0.5;
        T_dataType vad_m = 1.0 - vad_p;

        T_dataType fu = fi * vad_p + fp * vad_m;
        T_dataType dfu = dfm * vad_p + dfp * vad_m;
        T_dataType dzci = data.cv1(ix, iy, izp);
        T_dataType dzcu = data.cv1(ix, iy, iz) * vad_p + data.cv1(ix, iy, izp2) * vad_m;
        T_dataType dzbu = remap_data.cvc1(ix, iy, iz) * vad_p + remap_data.cvc1(ix, iy, izp) * vad_m;

        T_dataType phi = std::abs(o_v) / dzbu;

        T_dataType Da = (2.0 - phi) * std::abs(dfi) / dzci + (1.0 + phi) * std::abs(dfu) / dzcu;
        Da = Da * sixth;

        T_dataType ss = 0.5 * ((dfi >= 0.0 ? 1.0 : -1.0) + (dfu >= 0.0 ? 1.0 : -1.0));

        T_dataType Di = sign_v * ss * pw::min({std::abs(Da) * dzbu, std::abs(dfi), std::abs(dfu)});

        T_dataType rhou = remap_data.rho_v(ix, iy, iz) * vad_p + remap_data.rho_v(ix, iy, izp) * vad_m;
        T_dataType dmu = std::abs(data.dm(ix, iy, iz))/ (dzbu * rhou);

        remap_data.flux(ix, iy, iz) = fu + Di * (1.0 - dmu); }, Range(0, data.nx), Range(0, data.ny), Range(-1, data.nz));

        // Kinetic energy correction if rke is enabled
        if (data.rke)
        {
            pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixp = ix + 1;
                T_indexType iyp = iy + 1;
                T_indexType izm = iz - 1;
                T_indexType izp = iz + 1;

                T_dataType m = remap_data.rho_v1(ix, iy, iz) * remap_data.cv2(ix, iy, iz);
                T_dataType mp = remap_data.rho_v1(ix, iy, izp) * remap_data.cv2(ix, iy, izp);

                T_dataType ai = ((data.*mPtr)(ix, iy, iz) - remap_data.flux(ix, iy, izm)) * data.dm(ix, iy, izm) / m - ((data.*mPtr)(ix, iy, iz) - remap_data.flux(ix, iy, iz)) * data.dm(ix, iy, iz) / m;

                T_dataType aip = ((data.*mPtr)(ix, iy, izp) - remap_data.flux(ix, iy, iz)) * data.dm(ix, iy, iz) / mp - ((data.*mPtr)(ix, iy, izp) - remap_data.flux(ix, iy, izp)) * data.dm(ix, iy, izp) / mp;

                T_dataType dk = ((data.*mPtr)(ix, iy, izp) - (data.*mPtr)(ix, iy, iz)) * (remap_data.flux(ix, iy, iz) - 0.5 * ((data.*mPtr)(ix, iy, izp) + (data.*mPtr)(ix, iy, iz))) - 0.5 * ai * ((data.*mPtr)(ix, iy, iz) - remap_data.flux(ix, iy, iz)) + 0.5 * aip * ((data.*mPtr)(ix, iy, izp) - remap_data.flux(ix, iy, iz));

                dk = dk * data.dm(ix, iy, iz) * 0.5;
                pw::atomic::accelerated::Add(data.delta_ke(ix, iy, izp), dk);
                pw::atomic::accelerated::Add(data.delta_ke(ixp, iy, izp), dk);
                pw::atomic::accelerated::Add(data.delta_ke(ix, iyp, izp), dk);
                pw::atomic::accelerated::Add(data.delta_ke(ixp, iyp, izp), dk); }, Range(0, data.nx), Range(0, data.ny), Range(0, data.nz - 1));
            pw::fence();
        }

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) { remap_data.flux(ix, iy, iz) *= data.dm(ix, iy, iz); }, Range(0, data.nx), Range(0, data.ny), Range(-1, data.nz));
        pw::fence();
    }
}