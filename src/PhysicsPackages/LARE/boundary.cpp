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


namespace LARE
{

    //Lare style boundary conditions are now provided
    //via the LARE3DKokkosInitialConditions class in the
    //InitialConditions/LARE package.
    namespace pw = portableWrapper;

    template<typename T_EOS>
    void LARE3D<T_EOS>::boundary_conditions()
    {
        bfield_bcs();
        energy_bcs();
        density_bcs();
        velocity_bcs();
    }

    template<typename T_EOS>
    void LARE3D<T_EOS>::bfield_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("bx");
        varRegistry.applyBoundaryConditions("by");
        varRegistry.applyBoundaryConditions("bz");
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3D<T_EOS>::energy_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("energy_electron");
        varRegistry.applyBoundaryConditions("energy_ion");
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3D<T_EOS>::density_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("rho");
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3D<T_EOS>::velocity_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("vx");
        varRegistry.applyBoundaryConditions("vy");
        varRegistry.applyBoundaryConditions("vz");
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3D<T_EOS>::remap_v_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARE/vx1");
        varRegistry.applyBoundaryConditions("LARE/vy1");
        varRegistry.applyBoundaryConditions("LARE/vz1");
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3D<T_EOS>::dm_x_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARE/dm", 0); //Apply on X dimension
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3D<T_EOS>::dm_y_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARE/dm", 1); //Apply on Y dimension
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3D<T_EOS>::dm_z_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARE/dm", 2); //Apply on Z dimension
        pw::fence();
    }

    #include "./instantiate.h"
}
