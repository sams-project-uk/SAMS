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
#include "variableRegistry.h"
// #include "utils.h"

/**
 * Register variables with the portable array manager.
 */

 void simulation::registerVars(){

    auto& varRegistry = SAMS::getvariableRegistry();
    auto& typeRegistry = SAMS::gettypeRegistry();
    SAMS::typeID type = typeRegistry.getTypeID<T_dataType>();
    varRegistry.registerVariable("energy_electron", type, SAMS::memorySpace::DEVICE, SAMS::dimension("X",2), SAMS::dimension("Y",2), SAMS::dimension("Z",2));
    varRegistry.registerVariable("energy_ion", type, SAMS::memorySpace::DEVICE, SAMS::dimension("X",2), SAMS::dimension("Y",2), SAMS::dimension("Z",2));
    varRegistry.registerVariable("rho", type, SAMS::memorySpace::DEVICE, SAMS::dimension("X",2), SAMS::dimension("Y",2), SAMS::dimension("Z",2));
    varRegistry.registerVariable("vx", type, SAMS::memorySpace::DEVICE, SAMS::dimension("X",2, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y",2, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z",2, SAMS::staggerType::HALF_CELL));
    varRegistry.registerVariable("vy", type, SAMS::memorySpace::DEVICE, SAMS::dimension("X",2, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y",2, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z",2, SAMS::staggerType::HALF_CELL));
    varRegistry.registerVariable("vz", type, SAMS::memorySpace::DEVICE, SAMS::dimension("X",2, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y",2, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z",2, SAMS::staggerType::HALF_CELL));
    varRegistry.registerVariable("bx", type, SAMS::memorySpace::DEVICE, SAMS::dimension("X",2, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y",2), SAMS::dimension("Z",2));
    varRegistry.registerVariable("by", type, SAMS::memorySpace::DEVICE, SAMS::dimension("X",2), SAMS::dimension("Y",2, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z",2));
    varRegistry.registerVariable("bz", type, SAMS::memorySpace::DEVICE, SAMS::dimension("X",2), SAMS::dimension("Y",2), SAMS::dimension("Z",2, SAMS::staggerType::HALF_CELL));
 }

/**
 * Allocate the data arrays for the simulation.
 * This allocates the permanent state arrays that are used throughout the simulation.
 */
void simulation::allocate(simulationData &data)
{
    T_sizeType nx, ny, nz;

    const auto& axRegistry = SAMS::getaxisRegistry();
    nx = axRegistry.getElements("X");
    ny = axRegistry.getElements("Y");
    nz = axRegistry.getElements("Z");

    data.nx = nx;
    data.ny = ny;
    data.nz = nz;

    data.mu0_si = mu0_si;
    data.time = 0.0;

    manager.clear(); // Delete any allocated data

    auto& varRegistry = SAMS::getvariableRegistry();

    using Range = portableWrapper::Range;
    //Grab the final variable sizes from the registry and wrap the arrays
    {    
        const auto& vardef = varRegistry.getVariable("energy_electron");
        const auto& dims = vardef.getDimensions();
        manager.wrap(data.energy_electron, static_cast<T_dataType*>(vardef.getDataPtr()), Range(1-dims[0].lowerGhosts, nx + dims[0].upperGhosts), 
                                                Range(1-dims[1].lowerGhosts, ny + dims[1].upperGhosts), 
                                                Range(1-dims[2].lowerGhosts, nz + dims[2].upperGhosts));
    }
    {
        const auto& vardef = varRegistry.getVariable("energy_ion");
        const auto& dims = vardef.getDimensions();
        manager.wrap(data.energy_ion, static_cast<T_dataType*>(vardef.getDataPtr()), Range(1-dims[0].lowerGhosts, nx + dims[0].upperGhosts), 
                                                Range(1-dims[1].lowerGhosts, ny + dims[1].upperGhosts), 
                                                Range(1-dims[2].lowerGhosts, nz + dims[2].upperGhosts));
    }
    {
        const auto& vardef = varRegistry.getVariable("rho");
        const auto& dims = vardef.getDimensions();
        manager.wrap(data.rho, static_cast<T_dataType*>(vardef.getDataPtr()), Range(1-dims[0].lowerGhosts, nx + dims[0].upperGhosts), 
                                                Range(1-dims[1].lowerGhosts, ny + dims[1].upperGhosts), 
                                                Range(1-dims[2].lowerGhosts, nz + dims[2].upperGhosts));
    }
    {
        const auto& vardef = varRegistry.getVariable("vx");
        const auto& dims = vardef.getDimensions();
        manager.wrap(data.vx, static_cast<T_dataType*>(vardef.getDataPtr()), Range(0-dims[0].lowerGhosts, nx + dims[0].upperGhosts), 
                                                Range(0-dims[1].lowerGhosts, ny + dims[1].upperGhosts), 
                                                Range(0-dims[2].lowerGhosts, nz + dims[2].upperGhosts));
    }
    {
        const auto& vardef = varRegistry.getVariable("vy");
        const auto& dims = vardef.getDimensions();
        manager.wrap(data.vy, static_cast<T_dataType*>(vardef.getDataPtr()), Range(0-dims[0].lowerGhosts, nx + dims[0].upperGhosts), 
                                                Range(0-dims[1].lowerGhosts, ny + dims[1].upperGhosts), 
                                                Range(0-dims[2].lowerGhosts, nz + dims[2].upperGhosts));
    }
    {
        const auto& vardef = varRegistry.getVariable("vz");
        const auto& dims = vardef.getDimensions();
        manager.wrap(data.vz, static_cast<T_dataType*>(vardef.getDataPtr()), Range(0-dims[0].lowerGhosts, nx + dims[0].upperGhosts), 
                                                Range(0-dims[1].lowerGhosts, ny + dims[1].upperGhosts), 
                                                Range(0-dims[2].lowerGhosts, nz + dims[2].upperGhosts));
    }
    {
        const auto& vardef = varRegistry.getVariable("bx");
        const auto& dims = vardef.getDimensions();
        manager.wrap(data.bx, static_cast<T_dataType*>(vardef.getDataPtr()), Range(0-dims[0].lowerGhosts, nx + dims[0].upperGhosts), 
                                                Range(1-dims[1].lowerGhosts, ny + dims[1].upperGhosts), 
                                                Range(1-dims[2].lowerGhosts, nz + dims[2].upperGhosts));
    }
    {
        const auto& vardef = varRegistry.getVariable("by");
        const auto& dims = vardef.getDimensions();
        manager.wrap(data.by, static_cast<T_dataType*>(vardef.getDataPtr()), Range(1-dims[0].lowerGhosts, nx + dims[0].upperGhosts),
                                                Range(0-dims[1].lowerGhosts, ny + dims[1].upperGhosts), 
                                                Range(1-dims[2].lowerGhosts, nz + dims[2].upperGhosts));
    }
    {
        const auto& vardef = varRegistry.getVariable("bz");
        const auto& dims = vardef.getDimensions();
        manager.wrap(data.bz, static_cast<T_dataType*>(vardef.getDataPtr()), Range(1-dims[0].lowerGhosts, nx + dims[0].upperGhosts), 
                                                Range(1-dims[1].lowerGhosts, ny + dims[1].upperGhosts), 
                                                Range(0-dims[2].lowerGhosts, nz + dims[2].upperGhosts));
    }

    manager.allocate(data.p_visc, Range(-1, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
    manager.allocate(data.vx1, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
    manager.allocate(data.vy1, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
    manager.allocate(data.vz1, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
    manager.allocate(data.eta, Range(-1, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
    manager.allocate(data.dxab, Range(-2, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
    manager.allocate(data.dyab, Range(-1, nx + 2), Range(-2, ny + 2), Range(-1, nz + 2));
    manager.allocate(data.dzab, Range(-1, nx + 2), Range(-1, ny + 2), Range(-2, nz + 2));
    manager.allocate(data.dxac, Range(-1, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
    manager.allocate(data.dyac, Range(-1, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
    manager.allocate(data.dzac, Range(-1, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
    manager.allocate(data.cv, Range(-1, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
    manager.allocate(data.cv1, Range(-1, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
    manager.allocate(data.cvc, Range(-1, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
    manager.allocateManaged(data.xc, Range(-1, nx + 2));
    manager.allocateManaged(data.yc, Range(-1, ny + 2));
    manager.allocateManaged(data.zc, Range(-1, nz + 2));
    manager.allocate(data.xb, Range(-2, nx + 2));
    manager.allocate(data.yb, Range(-2, ny + 2));
    manager.allocate(data.zb, Range(-2, nz + 2));
    manager.allocate(data.xb_global, Range(-2, nx + 2));
    manager.allocate(data.yb_global, Range(-2, ny + 2));
    manager.allocate(data.zb_global, Range(-2, nz + 2));
    manager.allocate(data.dxc, Range(-1, nx + 2));
    manager.allocate(data.dyc, Range(-1, ny + 2));
    manager.allocate(data.dzc, Range(-1, nz + 2));
    manager.allocate(data.dxb, Range(-2, nx + 2));
    manager.allocate(data.dyb, Range(-2, ny + 2));
    manager.allocate(data.dzb, Range(-2, nz + 2));
    manager.allocate(data.hy, Range(-2, nx + 2));
    manager.allocate(data.hz, Range(-2, nx + 2), Range(-2, ny + 2));
    manager.allocate(data.hyc, Range(-1, nx + 2));
    manager.allocate(data.hzc, Range(-1, nx + 2), Range(-1, ny + 2));
    manager.allocate(data.hz1, Range(-2, nx + 2), Range(-2, ny + 2));
    manager.allocate(data.hz2, Range(-2, nx + 2), Range(-2, ny + 2));
    manager.allocate(data.x, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
    manager.allocate(data.y, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
    manager.allocate(data.z, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
    manager.allocate(data.xp, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
    manager.allocate(data.yp, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
    manager.allocate(data.zp, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
    if (data.rke)
    {
        manager.allocate(data.delta_ke, Range(-1, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
    }
}

/**
 * Setup the basic simulation parameters like grid points etc.
 */
void simulation::grid(simulationData &data)
{

    using Range = portableWrapper::Range;
    portableWrapper::portableArrayManager localManager;

    auto hyv = localManager.create<double>(portableWrapper::Range(-2, data.nx + 2));
    auto hzv = localManager.create<double>(portableWrapper::Range(-2, data.nx + 2), portableWrapper::Range(-2, data.ny + 2));

    data.length_x = data.x_max - data.x_min;
    data.length_y = data.y_max - data.y_min;
    data.length_z = data.z_max - data.z_min;

    data.dx = data.length_x / static_cast<T_dataType>(data.nx);
    data.dy = data.length_y / static_cast<T_dataType>(data.ny);
    data.dz = data.length_z / static_cast<T_dataType>(data.nz);

    // Setup global arrays
    {
        // Should this be done on the host and copied?
        // Set xb
        auto l1 = LAMBDA(T_indexType ix) { data.xb_global(ix) = data.x_min + ix * data.dx; };
        portableWrapper::applyKernel(l1, portableWrapper::Range(-2, data.nx + 2));
        // Set yb
        auto l2 = LAMBDA(T_indexType iy) { data.yb_global(iy) = data.y_min + iy * data.dy; };
        portableWrapper::applyKernel(l2, portableWrapper::Range(-2, data.ny + 2));
        // Set zb
        auto l3 = LAMBDA(T_indexType iz) { data.zb_global(iz) = data.z_min + iz * data.dz; };
        portableWrapper::applyKernel(l3, portableWrapper::Range(-2, data.nz + 2));
        portableWrapper::fence();
    }

    // Stretch the arrays if requested (todo)
    {
    }

    // Apply periodic boundary conditions (todo)
    {
        auto l1 = LAMBDA(T_indexType ix)
        {
            if (data.xbc_min == BCType::BC_PERIODIC)
            {
                data.xb_global(data.nx + 1) = data.xb_global(data.nx) + (data.xb_global(1) - data.xb_global(0));
                data.xb_global(data.nx + 2) = data.xb_global(data.nx) + (data.xb_global(2) - data.xb_global(0));
                data.xb_global(-1) = data.xb_global(0) - (data.xb_global(data.nx) - data.xb_global(data.nx - 1));
                data.xb_global(-2) = data.xb_global(0) - (data.xb_global(data.nx) - data.xb_global(data.nx - 2));
            }
            else
            {
                data.xb_global(data.nx + 1) = 2.0 * data.xb_global(data.nx) - data.xb_global(data.nx - 1);
                data.xb_global(data.nx + 2) = 2.0 * data.xb_global(data.nx) - data.xb_global(data.nx - 2);
                data.xb_global(-1) = 2.0 * data.xb_global(0) - data.xb_global(1);
                data.xb_global(-2) = 2.0 * data.xb_global(0) - data.xb_global(2);
            }
            if (data.ybc_min == BCType::BC_PERIODIC)
            {
                data.yb_global(data.ny + 1) = data.yb_global(data.ny) + (data.yb_global(1) - data.yb_global(0));
                data.yb_global(data.ny + 2) = data.yb_global(data.ny) + (data.yb_global(2) - data.yb_global(0));
                data.yb_global(-1) = data.yb_global(0) - (data.yb_global(data.ny) - data.yb_global(data.ny - 1));
                data.yb_global(-2) = data.yb_global(0) - (data.yb_global(data.ny) - data.yb_global(data.ny - 2));
            }
            else
            {
                data.yb_global(data.ny + 1) = 2.0 * data.yb_global(data.ny) - data.yb_global(data.ny - 1);
                data.yb_global(data.ny + 2) = 2.0 * data.yb_global(data.ny) - data.yb_global(data.ny - 2);
                data.yb_global(-1) = 2.0 * data.yb_global(0) - data.yb_global(1);
                data.yb_global(-2) = 2.0 * data.yb_global(0) - data.yb_global(2);
            }
            if (data.zbc_min == BCType::BC_PERIODIC)
            {
                data.zb_global(data.nz + 1) = data.zb_global(data.nz) + (data.zb_global(1) - data.zb_global(0));
                data.zb_global(data.nz + 2) = data.zb_global(data.nz) + (data.zb_global(2) - data.zb_global(0));
                data.zb_global(-1) = data.zb_global(0) - (data.zb_global(data.nz) - data.zb_global(data.nz - 1));
                data.zb_global(-2) = data.zb_global(0) - (data.zb_global(data.nz) - data.zb_global(data.nz - 2));
            }
            else
            {
                data.zb_global(data.nz + 1) = 2.0 * data.zb_global(data.nz) - data.zb_global(data.nz - 1);
                data.zb_global(data.nz + 2) = 2.0 * data.zb_global(data.nz) - data.zb_global(data.nz - 2);
                data.zb_global(-1) = 2.0 * data.zb_global(0) - data.zb_global(1);
                data.zb_global(-2) = 2.0 * data.zb_global(0) - data.zb_global(2);
            }
        };

        // Range here is entirely fake, this is just to get device code
        portableWrapper::applyKernel(l1, portableWrapper::Range(0, 0));
    }

    // Copy over xb,yb and zb from global
    {
        auto l1 = LAMBDA(T_indexType ix) { data.xb(ix) = data.xb_global(ix); };
        portableWrapper::applyKernel(l1, portableWrapper::Range(-2, data.nx + 2));
        auto l2 = LAMBDA(T_indexType iy) { data.yb(iy) = data.yb_global(iy); };
        portableWrapper::applyKernel(l2, portableWrapper::Range(-2, data.ny + 2));
        auto l3 = LAMBDA(T_indexType iz) { data.zb(iz) = data.zb_global(iz); };
        portableWrapper::applyKernel(l3, portableWrapper::Range(-2, data.nz + 2));
        portableWrapper::fence();
    }

    // Now calculate xc, yc and zc
    {
        auto l1 = LAMBDA(T_indexType ix) { data.xc(ix) = 0.5 * (data.xb_global(ix - 1) + data.xb_global(ix)); };
        portableWrapper::applyKernel(l1, portableWrapper::Range(-1, data.nx + 2));
        auto l2 = LAMBDA(T_indexType iy) { data.yc(iy) = 0.5 * (data.yb_global(iy - 1) + data.yb_global(iy)); };
        portableWrapper::applyKernel(l2, portableWrapper::Range(-1, data.ny + 2));
        auto l3 = LAMBDA(T_indexType iz) { data.zc(iz) = 0.5 * (data.zb_global(iz - 1) + data.zb_global(iz)); };
        portableWrapper::applyKernel(l3, portableWrapper::Range(-1, data.nz + 2));
        portableWrapper::fence();
    }

    // Calculate the cell edge distances
    {
        auto l1 = LAMBDA(T_indexType ix)
        {
            T_indexType ixm = ix - 1;
            data.dxb(ix) = (data.xb(ix) - data.xb(ixm));
        };
        portableWrapper::applyKernel(l1, portableWrapper::Range(-1, data.nx + 2));
        auto l2 = LAMBDA(T_indexType iy)
        {
            T_indexType iym = iy - 1;
            data.dyb(iy) = (data.yb(iy) - data.yb(iym));
        };
        portableWrapper::applyKernel(l2, portableWrapper::Range(-1, data.ny + 2));
        auto l3 = LAMBDA(T_indexType iz)
        {
            T_indexType izm = iz - 1;
            data.dzb(iz) = (data.zb(iz) - data.zb(izm));
        };
        portableWrapper::applyKernel(l3, portableWrapper::Range(-1, data.nz + 2));
        portableWrapper::fence();
    }

    // Calculate the cell centre distances
    {
        auto l1 = LAMBDA(T_indexType ix)
        {
            T_indexType ixm = ix - 1;
            data.dxc(ixm) = (data.xc(ix) - data.xc(ixm));
        };
        portableWrapper::applyKernel(l1, portableWrapper::Range(0, data.nx + 2));
        auto l2 = LAMBDA(T_indexType iy)
        {
            T_indexType iym = iy - 1;
            data.dyc(iym) = (data.yc(iy) - data.yc(iym));
        };
        portableWrapper::applyKernel(l2, portableWrapper::Range(0, data.ny + 2));
        auto l3 = LAMBDA(T_indexType iz)
        {
            T_indexType izm = iz - 1;
            data.dzc(izm) = (data.zc(iz) - data.zc(izm));
        };
        portableWrapper::applyKernel(l3, portableWrapper::Range(0, data.nz + 2));
        portableWrapper::fence();
    }

    if (data.geometry == geometryType::Cartesian)
    {
        portableWrapper::assign(data.hy, 1.0);
        portableWrapper::assign(data.hyc, 1.0);
        portableWrapper::assign(hyv, 1.0);
        portableWrapper::fence();
    }
    else if (data.geometry == geometryType::Cylindrical || data.geometry == geometryType::Spherical)
    {
        auto l1 = LAMBDA(T_indexType ix)
        {
            data.hy(ix) = std::abs(data.xb(ix));
            data.hyc(ix) = std::abs(data.xc(ix));
            hyv(ix) = (std::abs(data.xc(0)) < none_zero && ix == 1) ? std::abs(data.xc(ix)) : 0.25 * std::abs(data.xc(ix));
        };
        portableWrapper::applyKernel(l1, portableWrapper::Range(-2, data.nx + 2));
    }
    portableWrapper::fence();

    if (data.geometry == geometryType::Cartesian || data.geometry == geometryType::Cylindrical)
    {
        portableWrapper::assign(data.hz, 1.0);
        portableWrapper::assign(data.hzc, 1.0);
        portableWrapper::assign(hzv, 1.0);
        portableWrapper::assign(data.hz1, 1.0);
        portableWrapper::assign(data.hz2, 1.0);
        portableWrapper::fence();
    }
    else if (data.geometry == geometryType::Spherical)
    {
        auto l1 = LAMBDA(T_indexType iy)
        {
            T_dataType s = std::abs(std::sin(data.yb(iy)));
            T_indexType ix = -2;
            data.hz(ix, iy) = data.hy(ix) * s;
            hzv(ix + 2, iy + 2) = hyv(ix) * s;
            for (ix = -1; ix <= data.nx + 2; ++ix)
            {
                data.hz(ix, iy) = data.hy(ix) * s;
                hzv(ix, iy) = hyv(ix) * s;
                data.hz2(ix, iy) = data.hyc(ix) * s;
            }
        };
        portableWrapper::applyKernel(l1, portableWrapper::Range(-2, data.ny + 2));

        auto l2 = LAMBDA(T_indexType iy)
        {
            T_dataType sc = std::abs(std::sin(data.yc(iy)));
            T_indexType ix = -2;
            data.hz1(ix, iy) = data.hy(ix) * sc;
            for (ix = -1; ix <= data.nx + 2; ++ix)
            {
                data.hz1(ix, iy) = data.hy(ix) * sc;
                data.hzc(ix, iy) = data.hyc(ix) * sc;
            }
        };
        portableWrapper::applyKernel(l2, portableWrapper::Range(-1, data.ny + 2));
        portableWrapper::fence();
    }

    {
        // Cell centred areas and volumes
        auto l = LAMBDA(T_indexType iy, T_indexType iz)
        {
            T_dataType dy = data.dyb(iy);
            T_dataType dz = data.dzb(iz);
            T_dataType dydz = dy * dz;
            data.dxab(-2, iy, iz) = dydz * data.hy(-2) * data.hz1(-2, iy);
            for (T_dataType ix = -1; ix <= data.nx + 2; ++ix)
            {
                T_dataType dx = data.dxb(ix);
                data.dxab(ix, iy, iz) = dydz * data.hy(ix) * data.hz1(ix, iy);
                data.dyab(ix, iy, iz) = dx * dz * data.hz2(ix, iy);
                data.dzab(ix, iy, iz) = dx * dy * data.hy(ix);
                data.cv(ix, iy, iz) = dx * dydz * data.hyc(ix) * data.hzc(ix, iy);
            }
        };
        portableWrapper::applyKernel(l, portableWrapper::Range(-1, data.ny + 2), portableWrapper::Range(-1, data.nz + 2));
        portableWrapper::fence();
    }
    {
        // Fix negative y boundaries
        auto l = LAMBDA(T_indexType ix, T_indexType iz)
        {
            T_dataType dx = data.dxb(ix);
            T_dataType dz = data.dzc(iz);
            data.dyab(ix, -2, iz) = dx * dz * data.hz2(ix, -2);
        };
        portableWrapper::applyKernel(l, portableWrapper::Range(-1, data.nx + 2), portableWrapper::Range(-1, data.nz + 2));
        portableWrapper::fence();
    }
    {
        // Fix negative z boundaries
        auto l = LAMBDA(T_indexType ix, T_indexType iy)
        {
            T_dataType dx = data.dxb(ix);
            T_dataType dy = data.dyc(iy);
            data.dzab(ix, iy, -2) = dx * dy * data.hyc(ix);
        };
        portableWrapper::applyKernel(l, portableWrapper::Range(-1, data.nx + 2), portableWrapper::Range(-1, data.ny + 2));
        portableWrapper::fence();
    }
    {
        // Node centred areas and volumes
        auto l = LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz)
        {
            T_dataType dx = data.dxb(ix);
            T_dataType dy = data.dyb(iy);
            T_dataType dz = data.dzc(iz);
            T_dataType dydz = dy * dz;
            data.dxac(ix, iy, iz) = dydz * data.hyc(ix) * data.hz2(ix, iy);
            data.dyac(ix, iy, iz) = dx * dz * data.hz1(ix, iy);
            data.dzac(ix, iy, iz) = dx * dy * data.hy(ix);
            data.cvc(ix, iy, iz) = dx * dydz * hyv(ix) * data.hz(ix, iy);
        };
        portableWrapper::applyKernel(l, portableWrapper::Range(-1, data.nx + 2), portableWrapper::Range(-1, data.ny + 2), portableWrapper::Range(-1, data.nz + 2));
        portableWrapper::fence();
    }
    // Set up the cartesian coordinates array
    if (data.geometry == geometryType::Cartesian)
    {
        auto l = LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz)
        {
            data.x(ix, iy, iz) = data.xc(ix);
            data.y(ix, iy, iz) = data.yc(iy);
            data.z(ix, iy, iz) = data.zc(iz);
        };
        portableWrapper::applyKernel(l, portableWrapper::Range(-1, data.nx + 2), portableWrapper::Range(-1, data.ny + 2), portableWrapper::Range(-1, data.nz + 2));
        portableWrapper::fence();
    }
    else if (data.geometry == geometryType::Cylindrical)
    {
        auto l = LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz)
        {
            T_dataType r = data.xb(ix);
            T_dataType theta = data.yb(iy);
            T_dataType zz = data.zb(iz);
            data.x(ix, iy, iz) = r;
            data.y(ix, iy, iz) = theta;
            data.z(ix, iy, iz) = zz;
            data.xp(ix, iy, iz) = r * std::cos(theta);
            data.yp(ix, iy, iz) = r * std::sin(theta);
            data.zp(ix, iy, iz) = zz;
        };
        portableWrapper::applyKernel(l, portableWrapper::Range(-2, data.nx + 2), portableWrapper::Range(-2, data.ny + 2), portableWrapper::Range(-2, data.nz + 2));
        portableWrapper::fence();
    }
    else if (data.geometry == geometryType::Spherical)
    {
        auto l = LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz)
        {
            T_dataType r = data.xb(ix);
            T_dataType theta = data.yb(iy);
            T_dataType phi = data.zb(iz);
            data.x(ix, iy, iz) = r;
            data.y(ix, iy, iz) = theta;
            data.z(ix, iy, iz) = phi;
            data.xp(ix, iy, iz) = r * std::sin(theta) * std::cos(phi);
            data.yp(ix, iy, iz) = r * std::sin(theta) * std::sin(phi);
            data.zp(ix, iy, iz) = r * std::cos(theta);
        };
        portableWrapper::applyKernel(l, portableWrapper::Range(-2, data.nx + 2), portableWrapper::Range(-2, data.ny + 2), portableWrapper::Range(-2, data.nz + 2));
        portableWrapper::fence();
    }
}
