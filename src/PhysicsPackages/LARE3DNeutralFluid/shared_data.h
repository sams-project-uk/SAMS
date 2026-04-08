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
#ifndef SHARED_DATA_NF_H
#define SHARED_DATA_NF_H

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

#include "LARE3DSingleTemperature/shared_data.h"

/*
Quantities refer to _ion or _i to remain as similar to core LARE as possible.
*/

namespace LARE
{
    namespace pw = portableWrapper;

    template<typename T_EOS=idealGas>
    class LARE3DNF
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

    struct remapData
    {
        T_dataType xpass, ypass, zpass;
        volumeArray rho1;
        volumeArray cv2;
        volumeArray cvc1;
        volumeArray db1;
        volumeArray rho_v;
        volumeArray rho_v1;
        volumeArray flux;
    };

   /**
     * Class representing data only needed during the lagrangian step
     */
    struct lagranData
    {
        volumeArray alpha1;    // Alpha1 coefficient for magnetic field update
        volumeArray alpha2;    // Alpha2 coefficient for magnetic field update
        volumeArray alpha3;    // Alpha3 coefficient for magnetic field update
        volumeArray visc_heat; // Viscous heating
        volumeArray pressure;  // Pressure array
        volumeArray rho_v;     // Density at half timestep
        volumeArray cv_v;      // Control volume at half timestep
        volumeArray fx;        // X-force
        volumeArray fy;        // Y-force
        volumeArray fz;        // Z-force
        volumeArray fx_visc;   // X-viscous force
        volumeArray fy_visc;   // Y-viscous force
        volumeArray fz_visc;   // Z-viscous force
        volumeArray flux_x;    // X-flux
        volumeArray flux_y;    // Y-flux
        volumeArray flux_z;    // Z-flux
    };

       struct simulationData
    {

        bool configured = false; // Indicates if the data has been configured

        pw::Range xcLocalRange;
        pw::Range xcLocalDomainRange;
        pw::Range ycLocalRange;
        pw::Range ycLocalDomainRange;
        pw::Range zcLocalRange;
        pw::Range zcLocalDomainRange;
        pw::Range xbLocalRange;
        pw::Range xbLocalDomainRange;
        pw::Range ybLocalRange;
        pw::Range ybLocalDomainRange;
        pw::Range zbLocalRange;
        pw::Range zbLocalDomainRange;

        pw::Range xcminBCRange;
        pw::Range xcmaxBCRange;
        pw::Range ycminBCRange;
        pw::Range ycmaxBCRange;
        pw::Range zcminBCRange;
        pw::Range zcmaxBCRange;

        pw::Range xbminBCRange;
        pw::Range xbmaxBCRange;
        pw::Range ybminBCRange;
        pw::Range ybmaxBCRange;
        pw::Range zbminBCRange;
        pw::Range zbmaxBCRange;

        using eosType = T_EOS;
        eosType eos; // Equation of state object

        T_dataType none_zero = std::numeric_limits<T_dataType>::epsilon();  // Smallest non-zero value for T_dataTyp
        T_dataType largest_number = std::numeric_limits<T_dataType>::max(); // Largest number for T_dataType

        T_dataType dt_1; // dt indicated by Lagrangian step

        // Boundary conditions
        BCType xbc_min, xbc_max;
        BCType ybc_min, ybc_max;
        BCType zbc_min, zbc_max;

        // Physics selectors
        bool rke;          // Remap phase kinetic energy correction

        // Shock viscosity coefficients
        T_dataType visc1;      // Linear shock viscosity coefficient
        T_dataType visc2;      // Quadratic shock viscosity coefficient
        T_dataType visc2_norm; // Normalized quadratic shock viscosity coefficient

        // Physical constants
        T_dataType gas_gamma;      // Ratio of specific heats

        // Physical arrays
        volumeArray energy;

        volumeArray p_visc;          // Viscous pressure
        volumeArray rho;             // Density
        volumeArray vx;              // X-velocity
        volumeArray vy;              // Y-velocity
        volumeArray vz;              // Z-velocity
        volumeArray vx1;             // Half timestep X-velocity
        volumeArray vy1;             // Half timestep Y-velocity
        volumeArray vz1;             // Half timestep Z-velocity
        volumeArray cv1; // Half timestep control volume
        volumeArray cvc;

        lineArray grav_r;
        lineArray grav_z;
        volumeArray delta_ke; // Remap kinetic energy correction
        volumeArray x, y, z;
        volumeArray xp, yp, zp;

        volumeArray alpha1;    // Alpha1 coefficient for magnetic field update
        volumeArray alpha2;    // Alpha2 coefficient for magnetic field update
        volumeArray alpha3;    // Alpha3 coefficient for magnetic field update
        volumeArray visc_heat; // Viscous heating
        volumeArray pressure;  // Pressure array
        volumeArray rho_v;     // Density at half timestep
        volumeArray cv_v;      // Control volume at half timestep
        volumeArray fx;        // X-force
        volumeArray fy;        // Y-force
        volumeArray fz;        // Z-force
        volumeArray fx_visc;   // X-viscous force
        volumeArray fy_visc;   // Y-viscous force
        volumeArray fz_visc;   // Z-viscous force
        volumeArray flux_x;    // X-flux
        volumeArray flux_y;    // Y-flux
        volumeArray flux_z;    // Z-flux
        volumeArray dm;     // Mass flux for remap

        bool isxLB = false; // Is this processor on the x-min boundary
        bool isxUB = false; // Is this processor on the x-max boundary
        bool isyLB = false; // Is this processor on the y-min boundary
        bool isyUB = false; // Is this processor on the y-max boundary
        bool iszLB = false; // Is this processor on the z-min boundary
        bool iszUB = false; // Is this processor on the z-max boundary

        MPI_Datatype mpiType = MPI_DATATYPE_NULL; // MPI datatype for T_dataType
    
        T_dataType time, dt;  // Current LARE3D time
        int64_t step;      // Current LARE3D step
 
    };

    //To contain copies of things like sizes, geometry factors etc
    struct domainData{
        T_dataType nx;
        T_dataType ny;
        T_dataType nz;
        T_dataType dt_multiplier;
        int64_t nsteps;
        
        lineArray xb;
        lineArray yb;
        lineArray zb;
 
        volumeArray dxab;
        volumeArray dyab;
        volumeArray dzab;
        volumeArray dxac;
        volumeArray dyac;
        volumeArray dzac;
        
        lineArray dxc;
        lineArray dyc;
        lineArray dzc;
        
        lineArray dxb;
        lineArray dyb;
        lineArray dzb;
        
        geometryType geometry;
        lineArray hy;
        planeArray hz;
        lineArray hyc;
        planeArray hzc;
        //planeArray hz1;
        planeArray hz2;
        volumeArray cv;  // Control volume
 
    };

    private:

        SAMS::harness &harness;

         inline void getHostVersion(simulationData &data, const domainData & core_data, pw::portableArrayManager &manager, volumeArray &device, hostVolumeArray &host);

    public:


        /**
         * Portable array manager for handling memory allocation and deallocation
         */
        pw::portableArrayManager& manager;

        void set_dt(simulationData &data, const domainData & core_data);

        /**
         * Lagrangian predictor step
         */
        void predictor_step(simulationData &data, const domainData & core_data);
        /**
         * Lagrangian corrector step
         */
        void corrector_step(simulationData &data, const domainData & core_data);

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
        void remap_x(simulationData &data, remapData &remapData, const domainData & core_data);

        /**
         * Function to perform the Y sweep remap
         */
        void remap_y(simulationData &data, remapData &remapData, const domainData & core_data);

        /**
         * Function to perform the Z sweep remap
         */
        void remap_z(simulationData &data, remapData &remapData, const domainData & core_data);

        void copy_domain(domainData & core_data, const LARE3DST<T_EOS>::simulationData & lareData){
            
            core_data.nx = lareData.nx;
            core_data.ny = lareData.ny;
            core_data.nz = lareData.nz;
            
            core_data.dt_multiplier = lareData.dt_multiplier;
            core_data.nsteps = lareData.nsteps;

            core_data.xb = lareData.xb;
            core_data.yb = lareData.yb;
            core_data.zb = lareData.zb;
 
            core_data.dxab = lareData.dxab;
            core_data.dyab = lareData.dyab;
            core_data.dzab = lareData.dzab;
            core_data.dxac = lareData.dxac;
            core_data.dyac = lareData.dyac;
            core_data.dzac = lareData.dzac;
   
            core_data.dxc = lareData.dxc;
            core_data.dyc = lareData.dyc;
            core_data.dzc = lareData.dzc;
        
            core_data.dxb = lareData.dxb;
            core_data.dyb = lareData.dyb;
            core_data.dzb = lareData.dzb;

            core_data.hy = lareData.hy;
            core_data.hz = lareData.hz;
            core_data.hyc = lareData.hyc;
            core_data.hzc = lareData.hzc;
            core_data.hz2 = lareData.hz2;
            core_data.geometry = lareData.geometry;
            core_data.cv = lareData.cv;

        }

    public:

        /**
         * Constructor for LARE3D simulation data
         */
        LARE3DNF(SAMS::harness &harnessRef) : harness(harnessRef), manager(harnessRef.memoryRegistry.getArrayManager()) {}

        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        inline static constexpr auto nameType = SAMS::constexprName("LARE3DNF") + T_EOS::name;
        constexpr static std::string_view name = nameType;

        /**
         * Lare neutral Fluid is not a core simulation
         */
        constexpr static bool coreSimulation = false;

        /**
         * Lare should be timed
         */
        constexpr static bool timeSimulation = true;

        /**
         * Lare's dataPack is simulationData and remapData
         */
        using dataPack = SAMS::dataPacks::multiPack<simulationData, remapData, domainData>;


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
        void getVariables(SAMS::harness &harnessRef, simulationData &data, domainData & core_data, const LARE3DST<T_EOS>::simulationData & lareData){
            allocate(harnessRef, data, core_data);
            grid(data, core_data);
            copy_domain(core_data, lareData);
        }

        void beforeStartOfTimestep(simulationData &data, const domainData & core_data){
            lagrangian_prepare(data, core_data);
        }
 
        void startOfTimestep(simulationData &data, const domainData & core_data){
            lagrangian_step(data, core_data);
        }

        /**
         * This is the corrector step of the LARE3D timestep
         * @param data LARE3D simulation data
         */
        void halfTimestep(simulationData &data, const domainData & core_data){
            corrector_step(data, core_data);
        }

        /**
         * This is called at the end of the LARE3D timestep
         * @param data LARE3D simulation data
         */
        void endOfTimestep(simulationData &data, remapData &remap_data, const domainData & core_data){
            eulerian_remap(data, remap_data, core_data);
            if (data.rke){
                energy_correction(data, core_data);
            }
        }

        /**
         * Set the timestep based on LARE3D data
         * @note This function is called in response to the control function setTimestep being called
         * by a package. The runner will NOT call this function directly.
         * @param timeData SAMS timeState data
         * @param data LARE3D simulation data
         */
        void calculateTimestep(SAMS::timeState &timeData, simulationData &data, const domainData & core_data){
            set_dt(data, core_data);
            timeData.dt = data.dt<timeData.dt ? data.dt : timeData.dt;
        }

        /**
         * Gather the timestep back after all packages have calculated it
         * @param timeData SAMS timeState data
         * @param data LARE3D simulation data
         */
        void getTimestep(SAMS::timeState &timeData, simulationData &data, const domainData & core_data){
            data.dt = timeData.dt;
            data.step = timeData.step;
            data.time = timeData.time;
        };

        template<typename T>
        void registerOutputMeshes(writer<T> &writer, simulationData &data, const domainData & core_data);

        template<typename T>
        void registerOutputVariables(writer<T> &writer, simulationData &data, const domainData & core_data);

        template<typename T>
        void writeOutputMeshes(writer<T> &writer, simulationData &data, const domainData & core_data);

        template<typename T>
        void writeOutputVariables(writer<T> &writer, simulationData &data, const domainData & core_data);

        /**
         * Allocate the LARE3D data arrays
         * @param harness SAMS harness
         * @param data Simulation data struct
         * This function allocates the arrays in the simulationData struct.
         * It uses the portableArrayManager to handle the memory allocation and deallocation.
         */
        void allocate(SAMS::harness &harness, simulationData &data, domainData & core_data);

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
        void grid(simulationData &data, const domainData & core_data);

        /**
         * Call all the boundary condition functions
         * @param data Simulation data struct
         * This function calls all the boundary condition functions for the LARE3D.
         * It is called at the end of each time step to apply the boundary conditions.
         */
        void boundary_conditions();
        void cv1_update(simulationData &data, const domainData & core_data);
 
        void shock_viscosity(simulationData &data, const domainData & core_data );
        void rkstep(simulationData &data);
        void shock_heating(simulationData &data, const domainData & core_data);

        void x_mass_flux(simulationData &data, remapData &remap_data, const domainData & core_data);
        template <auto mPtr>
        void x_energy_flux(simulationData &data, remapData &remap_data, const domainData & core_data);
        template <auto mPtr>
        void x_mom_flux(simulationData &data, remapData &remap_data, const domainData & core_data);

        void y_mass_flux(simulationData &data, remapData &remap_data, const domainData & core_data);
        template <auto mPtr>
        void y_energy_flux(simulationData &data, remapData &remap_data, const domainData & core_data);
        template <auto mPtr>
        void y_mom_flux(simulationData &data, remapData &remap_data, const domainData & core_data);

        void z_mass_flux(simulationData &data, remapData &remap_data, const domainData & core_data);
        template <auto mPtr>
        void z_energy_flux(simulationData &data, remapData &remap_data, const domainData & core_data);
        template <auto mPtr>
        void z_mom_flux(simulationData &data, remapData &remap_data, const domainData & core_data);

        void lagrangian_prepare(simulationData &data, const domainData & core_data);
        void lagrangian_step(simulationData &data, const domainData & core_data);

        /**
         * Core remap control function
         */
        void eulerian_remap(simulationData &data, remapData &remap_data, const domainData & core_data);

        /**
         * Function to add back the kinetic energy correction
         */
        void energy_correction(simulationData &data, const domainData & core_data);
    };
}

//Use an X macro to instantiate the LARE3DNF template for all EOS types that use density and energy as inputs. This allows us to easily add new EOS types in the future by simply adding them to the EOS_DENSITY_ENERGY macro in eos.h without having to modify this file.
#define EOS_DEF(value) template class LARE::LARE3DNF<value>;
EOS_DENSITY_ENERGY
#undef EOS_DEF

#endif // SHARED_DATA_KOKKOS_H
