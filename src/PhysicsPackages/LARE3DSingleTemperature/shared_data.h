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
#ifndef SHARED_DATA_ST_H
#define SHARED_DATA_ST_H

#include <cstdint>
#include <cassert>
#include <string>
#include "LARECommon/types.h"
#include "LARECommon/eos.h"
#include "pp/parallelWrapper.h"
#include "mpiManager.h"
#include "variableDef.h"
#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"

#include "LARE/shared_data.h"

namespace LARE
{
    namespace pw = portableWrapper;

    template<typename T_EOS=idealGas>
    class LARE3DST
    {

    private:

        using T_dataType = SAMS::T_dataType;
        using T_sizeType = SAMS::T_sizeType;
        using T_indexType = SAMS::T_indexType;

        using volumeArray = portableWrapper::acceleratedArray<T_dataType, 3>;
        using hostVolumeArray = portableWrapper::hostArray<T_dataType, 3>;
        using planeArray = portableWrapper::acceleratedArray<T_dataType, 2>;
        using hostPlaneArray = portableWrapper::hostArray<T_dataType, 2>;
        using lineArray = portableWrapper::acceleratedArray<T_dataType, 1>;
        using hostLineArray = portableWrapper::hostArray<T_dataType, 1>;

    public:

    using remapData = LARE::LARE3D<T_EOS>::remapData;
    using lagranData = LARE::LARE3D<T_EOS>::lagranData;
    using simulationData = LARE::LARE3D<T_EOS>::simulationData;

    private:

        SAMS::harness &harness;

         inline void getHostVersion(simulationData &data, pw::portableArrayManager &manager, volumeArray &device, hostVolumeArray &host);

    public:


        /**
         * Portable array manager for handling memory allocation and deallocation
         */
        pw::portableArrayManager& manager;

        void set_dt(simulationData &data);

        /**
         * Lagrangian predictor step
         */
        void predictor_step(simulationData &data);
        /**
         * Lagrangian corrector step
         */
        void corrector_step(simulationData &data);

        /**
         * Boundary conditions for magnetic field
         */
        void bfield_bcs();
        /**
         * Boundary conditions for specific internal energy
         */
        void energy_bcs();
        /**
         * Boundary conditions for density
         */
        void density_bcs();
        /**
         * Boundary conditions for velocity
         */
        void velocity_bcs();
        /**
         * Boundary conditions for remap velocity
         * Normally the same as velocity_bcs, but can be different for some LARE3Ds
         */
        void remap_v_bcs();

        /**
         * Boundary conditions for remap phase X mass flux
         */
        void dm_x_bcs();

        /**
         * Boundary conditions for remap phase Y mass flux
         */
        void dm_y_bcs();

        /**
         * Boundary conditions for remap phase Z mass flux
         */
        void dm_z_bcs();

        /**
         * Function to perform the X sweep remap
         */
        void remap_x(simulationData &data, remapData &remapData);

        /**
         * Function to perform the Y sweep remap
         */
        void remap_y(simulationData &data, remapData &remapData);

        /**
         * Function to perform the Z sweep remap
         */
        void remap_z(simulationData &data, remapData &remapData);

    public:

        /**
         * Constructor for LARE3D simulation data
         */
        LARE3DST(SAMS::harness &harnessRef) : harness(harnessRef), manager(harnessRef.memoryRegistry.getArrayManager()) {}

        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        inline static constexpr auto nameType = SAMS::constexprName("LARE3DST") + T_EOS::name;
        constexpr static std::string_view name = nameType;

        /**
         * Lare is a core simulation
         */
        constexpr static bool coreSimulation = true;

        /**
         * Lare should be timed
         */
        constexpr static bool timeSimulation = true;

        /**
         * Lare's dataPack is simulationData and remapData
         */
        using dataPack = SAMS::dataPacks::multiPack<simulationData, remapData>;
        //using dataPack = std::tuple<simulationData, remapData>;


        /**
         * Initialize the LARE3D simulation. Called by the runner at the start of the simulation.
         */
        void initialize(){
        }

        /**
         * Register axes with the harness's axis registry.
         * Called by the runner before registering variables.
         * @param harnessRef SAMS harness
         */
        void registerAxes(SAMS::harness &harnessRef);

        /**
         * Register variables with the harness
         * Called by the runner after registering axes.
         * @param harnessRef SAMS harness
         */
        void registerVariables(SAMS::harness &harnessRef);


        /**
         * Set default values for control parameters
         */
        void defaultValues(simulationData &data);

        /**
         * Set default values for control parameters
         */
        void defaultVariables(simulationData &data);

        /**
         * Get the variables needed for LARE3D
         * i.e. convert the raw memory from the variable registry into
         * the volumeArray/lineArray types used by LARE3D
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        void getVariables(SAMS::harness &harnessRef, simulationData &data){
            allocate(harnessRef, data);
            grid(data);
        }

        void prepareTimestepCalculation(simulationData &data){
            lagrangian_prepare(data);
        }
        void startOfTimestep(simulationData &data){
            lagrangian_step(data);
        }

        /**
         * This is the corrector step of the LARE3D timestep
         * @param data LARE3D simulation data
         */
        void halfTimestep(simulationData &data){
            corrector_step(data);
        }

        /**
         * This is called at the end of the LARE3D timestep
         * @param data LARE3D simulation data
         */
        void endOfTimestep(simulationData &data, remapData &remap_data){
            eulerian_remap(data, remap_data);
            if (data.rke){
                energy_correction(data);
            }
            eta_calc(data);
        }

        /**
         * Set the timestep based on LARE3D data
         * @note This function is called in response to the control function setTimestep being called
         * by a package. The runner will NOT call this function directly.
         * @param timeData SAMS timeState data
         * @param data LARE3D simulation data
         */
        void calculateTimestep(SAMS::timeState &timeData, simulationData &data){
            set_dt(data);
            timeData.dt = data.dt<timeData.dt ? data.dt : timeData.dt;
        }

        /**
         * Gather the timestep back after all packages have calculated it
         * @param timeData SAMS timeState data
         * @param data LARE3D simulation data
         */
        void getTimestep(SAMS::timeState &timeData, simulationData &data){
            data.dt = timeData.dt;
        }

        template<typename T>
        void registerOutputMeshes(writer<T> &writer, simulationData &data);

        template<typename T>
        void registerOutputVariables(writer<T> &writer, simulationData &data);

        template<typename T>
        void writeOutputMeshes(writer<T> &writer, simulationData &data);

        template<typename T>
        void writeOutputVariables(writer<T> &writer, simulationData &data);

        /**
         * Allocate the LARE3D data arrays
         * @param harness SAMS harness
         * @param data Simulation data struct
         * This function allocates the arrays in the simulationData struct.
         * It uses the portableArrayManager to handle the memory allocation and deallocation.
         */
        void allocate(SAMS::harness &harness, simulationData &data);

        /**
         * Setup the LARE3D data
         * @param data Simulation data struct
         * This function sets up the LARE3D data, including the geometry, boundary conditions,
         * and other parameters. It also calculates the cell sizes and initializes the arrays.
         */
        void controlvariables(simulationData &data);
        /**
         * Setup the grid for the LARE3D
         * @param harness SAMS harness
         * @param data Simulation data struct
         * This function sets up the grid for the LARE3D, including the cell sizes and coordinates. Automatically creates for the specified geometry.
         */
        void grid(simulationData &data);

        /**
         * Call all the boundary condition functions
         * @param data Simulation data struct
         * This function calls all the boundary condition functions for the LARE3D.
         * It is called at the end of each time step to apply the boundary conditions.
         */
        void boundary_conditions();

        void shock_viscosity(simulationData &data);
        void resistive_effects(simulationData &data);
        void rkstep(simulationData &data);
        void bstep(simulationData &data);
        void b_field_and_cv1_update(simulationData &data);
        void shock_heating(simulationData &data);

        void vx_by_flux(simulationData &data, remapData &remap_data);
        void vx_bz_flux(simulationData &data, remapData &remap_data);
        void x_mass_flux(simulationData &data, remapData &remap_data);
        template <auto mPtr>
        void x_energy_flux(simulationData &data, remapData &remap_data);
        template <auto mPtr>
        void x_mom_flux(simulationData &data, remapData &remap_data);

        void vy_bx_flux(simulationData &data, remapData &remap_data);
        void vy_bz_flux(simulationData &data, remapData &remap_data);
        void y_mass_flux(simulationData &data, remapData &remap_data);
        template <auto mPtr>
        void y_energy_flux(simulationData &data, remapData &remap_data);
        template <auto mPtr>
        void y_mom_flux(simulationData &data, remapData &remap_data);

        void vz_bx_flux(simulationData &data, remapData &remap_data);
        void vz_by_flux(simulationData &data, remapData &remap_data);
        void z_mass_flux(simulationData &data, remapData &remap_data);
        template <auto mPtr>
        void z_energy_flux(simulationData &data, remapData &remap_data);
        template <auto mPtr>
        void z_mom_flux(simulationData &data, remapData &remap_data);

        void lagrangian_prepare(simulationData &data);
        void lagrangian_step(simulationData &data);

        /**
         * Calculate the resistivity eta based on current density
         * @param data Simulation data struct
         */
        void eta_calc(simulationData &data);

        /**
         * Core remap control function
         */
        void eulerian_remap(simulationData &data, remapData &remap_data);

        /**
         * Function to add back the kinetic energy correction
         */
        void energy_correction(simulationData &data);
    };
}

//Use an X macro to instantiate the LARE3DST template for all EOS types that use density and energy as inputs. This allows us to easily add new EOS types in the future by simply adding them to the EOS_DENSITY_ENERGY macro in eos.h without having to modify this file.
#define EOS_DEF(value) template class LARE::LARE3DST<value>;
EOS_DENSITY_ENERGY
#undef EOS_DEF

#endif // SHARED_DATA_KOKKOS_H
