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
#include "variableRegistry.h"
#include "axisRegistry.h"

namespace LARE
{

    namespace pw = portableWrapper;

    /**
     * Register LARE's axes with the axis registry.
     */
    template<typename T_EOS>
    void LARE3DNF<T_EOS>::registerAxes(SAMS::harness &harness)
    {
        auto &axisReg = harness.axisRegistry;
        axisReg.registerAxis("X", SAMS::MPIAxis(0));
        axisReg.registerAxis("Y", SAMS::MPIAxis(1));
        axisReg.registerAxis("Z", SAMS::MPIAxis(2));
    }

    /**
     * Register variables with the portable array manager.
     */
    template<typename T_EOS>
    void LARE3DNF<T_EOS>::registerVariables(SAMS::harness &harness)
    {

        auto &varRegistry = harness.variableRegistry;

        const int ghosts = 2; // 2 Ghost cells at top and bottom of each dimension

        varRegistry.registerVariable<T_dataType>("energy_ion", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

        varRegistry.registerVariable<T_dataType>("rho", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

        varRegistry.registerVariable<T_dataType>("vx", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<T_dataType>("vy", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<T_dataType>("vz", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<T_dataType>("LARE/vx1", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<T_dataType>("LARE/vy1", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<T_dataType>("LARE/vz1", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<T_dataType>("LARE/dm", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::CENTRED), SAMS::dimension("Y", ghosts, SAMS::staggerType::CENTRED), SAMS::dimension("Z", ghosts, SAMS::staggerType::CENTRED));
    }

    /**
     * Allocate the data arrays for the LARE3D.
     * This allocates the permanent state arrays that are used throughout the LARE3D.
     */
    template<typename T_EOS>
    void LARE3DNF<T_EOS>::allocate(SAMS::harness &harness, simulationData &data)
    {
        T_sizeType nx, ny, nz;

        auto &axRegistry = harness.axisRegistry;
        auto &varRegistry = harness.variableRegistry;
        // Centred since LARE thinks in terms of cell centres for nx, ny, nz
        nx = axRegistry.getLocalDomainElements("X", SAMS::staggerType::CENTRED);
        ny = axRegistry.getLocalDomainElements("Y", SAMS::staggerType::CENTRED);
        nz = axRegistry.getLocalDomainElements("Z", SAMS::staggerType::CENTRED);

        // Get the ranges for the whole local domain
        data.xcLocalRange = axRegistry.getLocalRange("X", SAMS::staggerType::CENTRED);
        data.ycLocalRange = axRegistry.getLocalRange("Y", SAMS::staggerType::CENTRED);
        data.zcLocalRange = axRegistry.getLocalRange("Z", SAMS::staggerType::CENTRED);
        data.xbLocalRange = axRegistry.getLocalRange("X", SAMS::staggerType::HALF_CELL);
        data.ybLocalRange = axRegistry.getLocalRange("Y", SAMS::staggerType::HALF_CELL);
        data.zbLocalRange = axRegistry.getLocalRange("Z", SAMS::staggerType::HALF_CELL);

        // Get the ranges for the ghost cells for centred axes
        data.xcminBCRange = axRegistry.getLocalNonDomainRange("X", SAMS::staggerType::CENTRED, SAMS::domain::edges::lower );
        data.xcmaxBCRange = axRegistry.getLocalNonDomainRange("X", SAMS::staggerType::CENTRED, SAMS::domain::edges::upper);
        data.ycminBCRange = axRegistry.getLocalNonDomainRange("Y", SAMS::staggerType::CENTRED, SAMS::domain::edges::lower);
        data.ycmaxBCRange = axRegistry.getLocalNonDomainRange("Y", SAMS::staggerType::CENTRED, SAMS::domain::edges::upper);
        data.zcminBCRange = axRegistry.getLocalNonDomainRange("Z", SAMS::staggerType::CENTRED, SAMS::domain::edges::lower);
        data.zcmaxBCRange = axRegistry.getLocalNonDomainRange("Z", SAMS::staggerType::CENTRED, SAMS::domain::edges::upper);

        // Get the ranges for the ghost cells for half cell staggered axes
        data.xbminBCRange = axRegistry.getLocalNonDomainRange("X", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::lower);
        data.xbmaxBCRange = axRegistry.getLocalNonDomainRange("X", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::upper);
        data.ybminBCRange = axRegistry.getLocalNonDomainRange("Y", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::lower);
        data.ybmaxBCRange = axRegistry.getLocalNonDomainRange("Y", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::upper);
        data.zbminBCRange = axRegistry.getLocalNonDomainRange("Z", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::lower);
        data.zbmaxBCRange = axRegistry.getLocalNonDomainRange("Z", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::upper);


        // Get the ranges for the actual domain (no ghost cells)
        data.xcLocalDomainRange = axRegistry.getLocalDomainRange("X", SAMS::staggerType::CENTRED);
        data.ycLocalDomainRange = axRegistry.getLocalDomainRange("Y", SAMS::staggerType::CENTRED);
        data.zcLocalDomainRange = axRegistry.getLocalDomainRange("Z", SAMS::staggerType::CENTRED);
        data.xbLocalDomainRange = axRegistry.getLocalDomainRange("X", SAMS::staggerType::HALF_CELL);
        data.ybLocalDomainRange = axRegistry.getLocalDomainRange("Y", SAMS::staggerType::HALF_CELL);
        data.zbLocalDomainRange = axRegistry.getLocalDomainRange("Z", SAMS::staggerType::HALF_CELL);

        data.nx = nx;
        data.ny = ny;
        data.nz = nz;

        data.nx_global = axRegistry.getDimension("X").getGlobalDomainCount(SAMS::staggerType::CENTRED);
        data.ny_global = axRegistry.getDimension("Y").getGlobalDomainCount(SAMS::staggerType::CENTRED);
        data.nz_global = axRegistry.getDimension("Z").getGlobalDomainCount(SAMS::staggerType::CENTRED);

        using Range = pw::Range;
        // Grab the final variable sizes from the registry and wrap the arrays
        varRegistry.fillPPArray("energy_ion", data.energy_ion);
        pw::assign(data.energy_ion, 0.0);
        varRegistry.fillPPArray("rho", data.rho);
        pw::assign(data.rho, 0.0);
        varRegistry.fillPPArray("vx", data.vx);
        pw::assign(data.vx, 0.0);
        varRegistry.fillPPArray("vy", data.vy);
        pw::assign(data.vy, 0.0);
        varRegistry.fillPPArray("vz", data.vz);
        pw::assign(data.vz, 0.0);
        varRegistry.fillPPArray("LARE/vx1", data.vx1);
        pw::assign(data.vx1, 0.0);
        varRegistry.fillPPArray("LARE/vy1", data.vy1);
        pw::assign(data.vy1, 0.0);
        varRegistry.fillPPArray("LARE/vz1", data.vz1);
        pw::assign(data.vz1, 0.0);
        varRegistry.fillPPArray("LARE/dm", data.dm);
        pw::assign(data.dm, 0.0);

        data.isxLB = harness.MPIManager.isEdge(0, SAMS::domain::edges::lower);
        data.isxUB = harness.MPIManager.isEdge(0, SAMS::domain::edges::upper);
        data.isyLB = harness.MPIManager.isEdge(1, SAMS::domain::edges::lower);
        data.isyUB = harness.MPIManager.isEdge(1, SAMS::domain::edges::upper);
        data.iszLB = harness.MPIManager.isEdge(2, SAMS::domain::edges::lower);
        data.iszUB = harness.MPIManager.isEdge(2, SAMS::domain::edges::upper);

        SAMS::debugAll3 << "Edge detection: "
                        << " XLB: " << data.isxLB << " XUB: " << data.isxUB
                        << " YLB: " << data.isyLB << " YUB: " << data.isyUB
                        << " ZLB: " << data.iszLB << " ZUB: " << data.iszUB
                        << std::endl;

        manager.allocate(data.p_visc, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.cv, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.cv1, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.cvc, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);

        Range xcp = pw::Range(0, data.nx + 1);
        Range ycp = pw::Range(0, data.ny + 1);
        Range zcp = pw::Range(0, data.nz + 1);
        Range ycpp = pw::Range(0, data.ny + 2);
        Range zcpp = pw::Range(0, data.nz + 2);
        Range xbp = pw::Range(-1, data.nx + 1);
        Range ybp = pw::Range(-1, data.ny + 1);
        Range zbp = pw::Range(-1, data.nz + 1);
        // Allocate arrays using the portableArrayManager
        manager.allocate(data.alpha1, xcp, ycpp, zcpp);
        manager.allocate(data.alpha2, xbp, ycp, zcpp);
        manager.allocate(data.alpha3, data.xcLocalRange, data.ycLocalRange, zcp);
        manager.allocate(data.visc_heat, xcp, ycp, zcp);
        manager.allocate(data.pressure, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.rho_v, xbp, ybp, zbp);
        manager.allocate(data.cv_v, xbp, ybp, zbp);
        manager.allocate(data.fx, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.fy, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.fz, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.fx_visc, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.fy_visc, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.fz_visc, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.flux_x, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.flux_y, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.flux_z, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);

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

        data.mpiType = SAMS::gettypeRegistry().getMPIType<T_dataType>();

        data.eos.setGamma(data.gas_gamma);
    }

    /**
     * Setup the basic LARE3D parameters like grid points etc.
     */
    template<typename T_EOS>
    void LARE3DNF<T_EOS>::grid(simulationData &data, const LARE3DST<T_EOS>::simulationData & core_data)
    {
    };
}