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


namespace LARE
{

    //Lare style boundary conditions are now provided
    //via the LARE3DKokkosInitialConditions class in the
    //InitialConditions/LARE package.
    namespace pw = portableWrapper;

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::boundary_conditions()
    {
        energy_bcs();
        density_bcs();
        velocity_bcs();
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::energy_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARENF/energy");
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::density_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARENF/rho");
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::velocity_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARENF/vx");
        varRegistry.applyBoundaryConditions("LARENF/vy");
        varRegistry.applyBoundaryConditions("LARENF/vz");
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::remap_v_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARENF/vx1");
        varRegistry.applyBoundaryConditions("LARENF/vy1");
        varRegistry.applyBoundaryConditions("LARENF/vz1");
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::dm_x_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARENF/dm", 0); //Apply on X dimension
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::dm_y_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARENF/dm", 1); //Apply on Y dimension
        pw::fence();
    }

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::dm_z_bcs()
    {
        auto &varRegistry = harness.variableRegistry;
        varRegistry.applyBoundaryConditions("LARENF/dm", 2); //Apply on Z dimension
        pw::fence();
    }
}
