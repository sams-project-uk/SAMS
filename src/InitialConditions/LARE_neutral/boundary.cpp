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
#include "shared_data_neutral.h"
#include "variableRegistry.h"


namespace LARE_neutral
{

    //Lare style boundary conditions are now provided
    //via the LARE3D_neutralInitialConditions class in the
    //InitialConditions/LARE package.
    namespace pw = portableWrapper;

    void LARE3D_neutral::boundary_conditions()
    {
        bfield_bcs();
        energy_bcs();
        density_bcs();
        velocity_bcs();
    }

    void LARE3D_neutral::bfield_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("bx_n");
        varRegistry.applyBoundaryConditions("by_n");
        varRegistry.applyBoundaryConditions("bz_n");
        pw::fence();
    }

    void LARE3D_neutral::energy_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("energy_neutral");
        pw::fence();
    }

    void LARE3D_neutral::density_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("rho_n");
        pw::fence();
    }

    void LARE3D_neutral::velocity_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("vx_n");
        varRegistry.applyBoundaryConditions("vy_n");
        varRegistry.applyBoundaryConditions("vz_n");
        pw::fence();
    }

    void LARE3D_neutral::remap_v_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARE/vx1_n");
        varRegistry.applyBoundaryConditions("LARE/vy1_n");
        varRegistry.applyBoundaryConditions("LARE/vz1_n");
        pw::fence();
    }

    void LARE3D_neutral::dm_x_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARE/dm_n", 0); //Apply on X dimension
        pw::fence();
    }

    void LARE3D_neutral::dm_y_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARE/dm_n", 1); //Apply on Y dimension
        pw::fence();
    }

    void LARE3D_neutral::dm_z_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARE/dm_n", 2); //Apply on Z dimension
        pw::fence();
    }
}
