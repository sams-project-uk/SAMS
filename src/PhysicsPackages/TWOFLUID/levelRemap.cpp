/*
Remap code for the level populations
*/
#include "LARE3DSingleTemperature/shared_data.h"
#include "LARE3DNeutralFluid/shared_data.h"
#include "twofluid.h"

namespace PIP
{

    namespace pw = portableWrapper;

    template<typename T_EOS>
    void PIP<T_EOS>::eulerian_remap(LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source, LARE::LARE3DNF<T_EOS>::remapData &remap_data, const domainData & core_data, const SAMS::timeState &ts)
    {
        using Range = pw::Range;
        int case_test;
        pw::portableArrayManager remapManager;

        // We can allocate everything other than flux here
        remapManager.allocate(remap_data.rho1, Range(-1, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-1, core_data.nz + 2));
        remapManager.allocate(remap_data.cv2, Range(-1, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-1, core_data.nz + 2));
        remapManager.allocate(remap_data.cvc1, Range(-1, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-1, core_data.nz + 2));
        remapManager.allocate(remap_data.db1, Range(-1, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-1, core_data.nz + 2));
        remapManager.allocate(remap_data.rho_v, Range(-1, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-1, core_data.nz + 2));
        remapManager.allocate(remap_data.rho_v1, Range(-1, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-1, core_data.nz + 2));
        // Flux is one element larger in the direction of remap, so it is allocated in each remap function

        if (data.rke)
        {
            pw::assign(data.delta_ke, 0.0);
        }
        remap_data.xpass = 1.0;
        remap_data.ypass = 1.0;
        remap_data.zpass = 1.0;


        //case_test = data.step % 6;
        case_test = ts.step % 6;

        // Strang ordering
        switch (case_test)
        {
        case 0:
            remap_x(dataNeutral, remap_data, plasma_source, core_data);
            remap_y(dataNeutral, remap_data, plasma_source, core_data);
            remap_z(dataNeutral, remap_data, plasma_source, core_data);
            break;
        case 1:
            remap_y(dataNeutral, remap_data, plasma_source, core_data);
            remap_z(dataNeutral, remap_data, plasma_source, core_data);
            remap_x(dataNeutral, remap_data, plasma_source, core_data);
            break;
        case 2:
            remap_z(dataNeutral, remap_data, plasma_source, core_data);
            remap_x(dataNeutral, remap_data, plasma_source, core_data);
            remap_y(dataNeutral, remap_data, plasma_source, core_data);
            break;
        case 3:
            remap_x(dataNeutral, remap_data, plasma_source, core_data);
            remap_z(dataNeutral, remap_data, plasma_source, core_data);
            remap_y(dataNeutral, remap_data, plasma_source, core_data);
            break;
        case 4:
            remap_z(dataNeutral, remap_data, plasma_source, core_data);
            remap_y(dataNeutral, remap_data, plasma_source, core_data);
            remap_x(dataNeutral, remap_data, plasma_source, core_data);
            break;
        case 5:
            remap_y(dataNeutral, remap_data, plasma_source, core_data);
            remap_x(dataNeutral, remap_data, plasma_source, core_data);
            remap_z(dataNeutral, remap_data, plasma_source, core_data);
            break;
        }


        remapManager.deallocate(remap_data.rho1);
        remapManager.deallocate(remap_data.cv2);
        remapManager.deallocate(remap_data.cvc1);
        remapManager.deallocate(remap_data.db1);
        remapManager.deallocate(remap_data.rho_v);
        remapManager.deallocate(remap_data.rho_v1);
        
        //Check the mass conservation per cell
        printf("MASS CONSERVATION NOT CHECKED");
    }
 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// x-remap
    template<typename T_EOS>
    void LARE3DNF<T_EOS>::remap_x(simulationData &data, remapData &remap_data, data_two_fluid_source &plasma_source, const domainData & core_data)
    {
        using Range = pw::Range;
        pw::portableArrayManager xRemapManager;
        remap_data.flux.nullify();

        xRemapManager.allocate(remap_data.flux, Range(-2, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-1, core_data.nz + 2));

        pw::assign(data.dm, 0.0);
        pw::assign(remap_data.rho1, data.rho);

        // Main remap loop
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
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

                T_dataType vol = core_data.cv(ix, iy, iz);
                T_dataType dvxdx = (vxb * core_data.dxab(ix, iy, iz) - vxbm * core_data.dxab(ixm, iy, iz)) / vol;
                T_dataType dvydy = remap_data.ypass * (vyb * core_data.dyab(ix, iy, iz) - vybm * core_data.dyab(ix, iym, iz)) / vol;
                T_dataType dvzdz = remap_data.zpass * (vzb * core_data.dzab(ix, iy, iz) - vzbm * core_data.dzab(ix, iy, izm)) / vol;

                T_dataType dv = (dvydy + dvzdz) * data.dt;

                // Control volume after remap
                remap_data.cv2(ix, iy, iz) = vol * (1.0 + dv);

                dv = dv + dvxdx * data.dt;

                // Control volume before remap
                data.cv1(ix, iy, iz) = vol * (1.0 + dv);

                // dxb before remap
                remap_data.db1(ix, iy, iz) = core_data.dxb(ix) + (vxb - vxbm) * data.dt;
            },
            Range(-1, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-1, core_data.nz + 2));
        pw::fence();

        // cvc1 = vertex CV before remap
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixp = ix + 1;
                T_indexType iyp = iy + 1;
                T_indexType izp = iz + 1;
                remap_data.cvc1(ix, iy, iz) = 0.125 * (data.cv1(ix, iy, iz) + data.cv1(ixp, iy, iz) + data.cv1(ix, iyp, iz) + data.cv1(ixp, iyp, iz) + data.cv1(ix, iy, izp) + data.cv1(ixp, iy, izp) + data.cv1(ix, iyp, izp) + data.cv1(ixp, iyp, izp));
            },
            Range(-1, core_data.nx + 1), Range(-1, core_data.ny + 1), Range(-1, core_data.nz + 1));

        pw::fence();

        x_mass_flux(data, remap_data, core_data); // Mass flux in x-direction
        dm_x_bcs();    // Apply boundary conditions to mass flux

        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixm = ix - 1;
                for (LARE::T_indexType lower_level = 1; lower_level < nLevels; ++lower_level) {
                    plasma_source.level_populations(ix,iy,iz,lower_level) = (remap_data.level_populations(ix,iy,iz,lower_level) * data.cv1(ix, iy, iz) + data.dm(ixm, iy, iz) - data.dm(ix, iy, iz)) / remap_data.cv2(ix, iy, iz);
                }
            },
            Range(1, core_data.nx), Range(1, core_data.ny), Range(1, core_data.nz));
    }
    
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// y-remap
template<typename T_EOS>
    void LARE3DNF<T_EOS>::remap_y(simulationData &data, remapData &remap_data, data_two_fluid_source &plasma_source, const domainData & core_data)
    {
    return;
        using Range = pw::Range;
        pw::portableArrayManager yRemapManager;
        remap_data.flux.nullify();

        yRemapManager.allocate(remap_data.flux, Range(-1, core_data.nx + 2), Range(-2, core_data.ny + 2), Range(-1, core_data.nz + 2));

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

                T_dataType vol = core_data.cv(ix, iy, iz);
                T_dataType dvxdx = remap_data.xpass * (vxb * core_data.dxab(ix, iy, iz) - vxbm * core_data.dxab(ixm, iy, iz)) / vol;
                T_dataType dvydy = (vyb * core_data.dyab(ix, iy, iz) - vybm * core_data.dyab(ix, iym, iz)) / vol;
                T_dataType dvzdz = remap_data.zpass * (vzb * core_data.dzab(ix, iy, iz) - vzbm * core_data.dzab(ix, iy, izm)) / vol;

                T_dataType dv = (dvxdx + dvzdz) * data.dt;

                // Control volume after remap
                remap_data.cv2(ix, iy, iz) = vol * (1.0 + dv);

                dv = dv + dvydy * data.dt;

                // Control volume before remap
                data.cv1(ix, iy, iz) = vol * (1.0 + dv);

                // dyb before remap
                remap_data.db1(ix, iy, iz) = core_data.hyc(ix) * core_data.dyb(iy) + (vyb - vybm) * data.dt;
            },
            Range(-1, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-1, core_data.nz + 2));
        pw::fence();

        // cvc1 = vertex CV before remap
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType ixp = ix + 1;
                T_indexType iyp = iy + 1;
                T_indexType izp = iz + 1;
                remap_data.cvc1(ix, iy, iz) = 0.125 * (data.cv1(ix, iy, iz) + data.cv1(ixp, iy, iz) + data.cv1(ix, iyp, iz) + data.cv1(ixp, iyp, iz) + data.cv1(ix, iy, izp) + data.cv1(ixp, iy, izp) + data.cv1(ix, iyp, izp) + data.cv1(ixp, iyp, izp));
            },
            Range(-1, core_data.nx + 1), Range(-1, core_data.ny + 1), Range(-1, core_data.nz + 1));
        pw::fence();


        // Remap of mass + calculation of mass fluxes (dm) needed for later remaps
        y_mass_flux(data, remap_data, core_data);
        // Need dm(0:nx+1,-1:ny+1,0:nz+1) for velocity remap
        dm_y_bcs();

        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_indexType iym = iy - 1;
                for (LARE::T_indexType lower_level = 1; lower_level < nLevels; ++lower_level) {
                    plasma_source.level_populations(ix,iy,iz,lower_level) = (remap_data.level_populations(ix,iy,iz,lower_level) * data.cv1(ix, iy, iz) +
                                            data.dm(ix, iym, iz) - data.dm(ix, iy, iz)) /
                                           remap_data.cv2(ix, iy, iz);
               }
            },
            Range(1, core_data.nx), Range(1, core_data.ny), Range(1, core_data.nz));
        pw::fence();
    }


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// z-remap

template<typename T_EOS>
    void LARE3DNF<T_EOS>::remap_z(simulationData &data, remapData &remap_data, data_two_fluid_source &plasma_source, const domainData & core_data)
    {
        using Range = pw::Range;
        pw::portableArrayManager zRemapManager;
        remap_data.flux.nullify();

        zRemapManager.allocate(remap_data.flux, Range(-1, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-2, core_data.nz + 2));

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

                T_dataType vol = core_data.cv(ix, iy, iz);
                T_dataType dvxdx = remap_data.xpass * (vxb * core_data.dxab(ix, iy, iz) - vxbm * core_data.dxab(ixm, iy, iz)) / vol;
                T_dataType dvydy = remap_data.ypass * (vyb * core_data.dyab(ix, iy, iz) - vybm * core_data.dyab(ix, iym, iz)) / vol;
                T_dataType dvzdz = (vzb * core_data.dzab(ix, iy, iz) - vzbm * core_data.dzab(ix, iy, izm)) / vol;

                T_dataType dv = (dvxdx + dvydy) * data.dt;

                // Control volume after remap
                remap_data.cv2(ix, iy, iz) = vol * (1.0 + dv);

                dv = dv + dvzdz * data.dt;

                // Control volume before remap
                data.cv1(ix, iy, iz) = vol * (1.0 + dv);

                // dyb before remap
                remap_data.db1(ix, iy, iz) = core_data.hzc(ix, iy) * core_data.dzb(iz) + (vzb - vzbm) * data.dt;
            },
            Range(-1, core_data.nx + 2), Range(-1, core_data.ny + 2), Range(-1, core_data.nz + 2));
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
            Range(-1, core_data.nx + 1), Range(-1, core_data.ny + 1), Range(-1, core_data.nz + 1));
        pw::fence();

        z_mass_flux(data, remap_data, core_data);
        dm_z_bcs();

        pw::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        T_indexType izm = iz - 1;
            for (LARE::T_indexType lower_level = 1; lower_level < nLevels; ++lower_level) {
                plasma_source.level_populations(ix,iy,iz,lower_level) = (remap_data.level_populations(ix,iy,iz,lower_level) * data.cv1(ix,iy,iz) + data.dm(ix,iy,izm) - data.dm(ix,iy,iz)) / remap_data.cv2(ix,iy,iz); 
            }    
        }, Range(1, core_data.nx), Range(1, core_data.ny), Range(1, core_data.nz));
        pw::fence();
    }
}
