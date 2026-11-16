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
#ifndef CONSERVED_SOLVER_H
#define CONSERVED_SOLVER_H

#include "LARECommon/types.h"
#include "pp/parallelWrapper.h"
#include "harness.h"

namespace conservedSolver
{
    namespace pw = portableWrapper;

    using T_dataType = SAMS::T_dataType;
    using T_indexType = SAMS::T_indexType;
    using volumeArray = portableWrapper::acceleratedArray<T_dataType, 3>;
    using lineArray = portableWrapper::acceleratedArray<T_dataType, 1>;

    /**
     * The conserved variables of ideal MHD, or their fluxes, at cell centres or faces
     */
    struct variables
    {
        volumeArray rho;    // Mass density
        volumeArray mx;     // X-momentum density
        volumeArray my;     // Y-momentum density
        volumeArray mz;     // Z-momentum density
        volumeArray energy; // Total energy density
        volumeArray bx;     // X-magnetic field
        volumeArray by;     // Y-magnetic field
        volumeArray bz;     // Z-magnetic field
    };

    /**
     * The data pack of the solvers of the conserved variables, such as HLLE3D and KT3D
     */
    struct simulationData
    {
        T_dataType gas_gamma; // Ratio of specific heats
        T_dataType cfl;       // Courant number
        T_dataType dt;        // Timestep

        T_indexType nx, ny, nz; // Number of local cells

        pw::Range xcLocalRange, ycLocalRange, zcLocalRange; // Local cell ranges including ghost cells

        variables U;  // Conserved variables
        variables U0; // Conserved variables at the start of the step
        variables fx; // Fluxes on X faces. Face i + 1/2 is at index i
        variables fy; // Fluxes on Y faces
        variables fz; // Fluxes on Z faces

        volumeArray ex, ey, ez; // Electric field at cell edges for flux-CT. Edge (j + 1/2, k + 1/2) of ex is at index (i, j, k), and the same for the others

        lineArray dxb, dyb, dzb; // Cell widths
    };
}

#endif // CONSERVED_SOLVER_H
