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
#ifndef SHARED_DATA_H
#define SHARED_DATA_H

#include <cstdint>
#include <cassert>
#include <string>
#include "constants.h"
#include "pp/parallelWrapper.h"
#include "remapData.h"
#include "typedefs.h"
#include "mpiManager.h"
#include "variableDef.h"
#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"

namespace LARE
{



    namespace pw = portableWrapper;
    // Possible geometry types
    enum class geometryType
    {
        Cartesian = 0,
        Cylindrical = 1,
        Spherical = 2,
    };

    // Boundary condition types
    enum class BCType
    {
        BC_OTHER = 0,      // Other boundary condition
        BC_PERIODIC = 1,   // Periodic boundary condition
        BC_REFLECTIVE = 2, // Reflective boundary condition
        BC_OUTFLOW = 3,    // Outflow boundary condition
        BC_INFLOW = 4,     // Inflow boundary condition
        BC_EXTERNAL = 5   // External boundary condition (do not apply Lare Style BCs)
    };

    /**
     * This is a struct that hold all LARE3D data
     * Putting this is a struct allows separate LARE3Ds in the same code
     */
    struct simulationData
    {

        bool configured = false; // Indicates if the LARE3D data has been configured

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

        T_dataType none_zero = std::numeric_limits<T_dataType>::epsilon();  // Smallest non-zero value for T_dataTyp
        T_dataType largest_number = std::numeric_limits<T_dataType>::max(); // Largest number for T_dataType

        // Simulation parameters
        T_dataType dt, dtr, dt_multiplier;
        T_dataType time;  // Current LARE3D time
        int64_t step;      // Current LARE3D step
        int64_t nsteps;   // Maximum number of steps, if < 0 run until t_end
        T_dataType t_end; // End time of the LARE3D

        // Domain parameters
        T_indexType nx, ny, nz; // Could be unsigned but when comparing signed and unsigned, unsigned wins
        T_indexType nx_global, ny_global, nz_global;
        T_dataType x_min, x_max, length_x, dx;
        T_dataType y_min, y_max, length_y, dy;
        T_dataType z_min, z_max, length_z, dz;
        bool x_stretch, y_stretch, z_stretch;
        geometryType geometry;

        // Boundary conditions
        BCType xbc_min, xbc_max;
        BCType ybc_min, ybc_max;
        BCType zbc_min, zbc_max;

        // Physics selectors
        bool resistiveMHD; // Resistive MHD
        bool rke;          // Remap phase kinetic energy correction

        // Shock viscosity coefficients
        T_dataType visc1;      // Linear shock viscosity coefficient
        T_dataType visc2;      // Quadratic shock viscosity coefficient
        T_dataType visc2_norm; // Normalized quadratic shock viscosity coefficient

        // Physical constants
        T_dataType gas_gamma;      // Ratio of specific heats
        T_dataType j_max;          // Maximum current
        T_dataType eta0;           // Limited resisivity (applied J>j_max)
        T_dataType eta_background; // Background resisivity
        T_dataType mf;             // Average mass of an ion in proton masses
        T_dataType mu0;             // Vacuum permeability

        // IO control
        T_dataType dt_snapshots; // Time between snapshots
        T_dataType lastOutputTime=0.0; // Time of last output

        // Physical arrays
        volumeArray energy_electron; // Electron specific internal energy
        volumeArray energy_ion;      // Ion specific
        volumeArray p_visc;          // Viscous pressure
        volumeArray rho;             // Density
        volumeArray vx;              // X-velocity
        volumeArray vy;              // Y-velocity
        volumeArray vz;              // Z-velocity
        volumeArray vx1;             // Half timestep X-velocity
        volumeArray vy1;             // Half timestep Y-velocity
        volumeArray vz1;             // Half timestep Z-velocity
        volumeArray bx;              // X-magnetic field
        volumeArray by;              // Y-magnetic field
        volumeArray bz;              // Z-magnetic field
        volumeArray eta;             // Resisivity
        volumeArray dxab;
        volumeArray dyab;
        volumeArray dzab;
        volumeArray dxac;
        volumeArray dyac;
        volumeArray dzac;
        volumeArray cv;  // Control volume
        volumeArray cv1; // Half timestep control volume
        volumeArray cvc;
        lineArray xc;          // Cell center X-coordinates
        lineArray yc;          // Cell center Y-coordinates
        lineArray zc;          // Cell center Z-coordinates
        lineArray xb;          // Cell boundary X-coordinates
        lineArray yb;          // Cell boundary Y-coordinates
        lineArray zb;          // Cell boundary Z-coordinates
        hostLineArray xc_host; // Host copy of cell center X-coordinates
        hostLineArray yc_host; // Host copy of cell center Y-coordinates
        hostLineArray zc_host; // Host copy of cell center Z-coordinates
        hostLineArray xb_host; // Host copy of cell boundary X-coordinates
        hostLineArray yb_host; // Host copy of cell boundary Y-coordinates
        hostLineArray zb_host; // Host copy of cell boundary Z-coordinates
        lineArray xb_global;   // Global cell boundary X-coordinates
        lineArray yb_global;   // Global cell boundary Y-coordinates
        lineArray zb_global;   // Global cell boundary Z-coordinates
        lineArray dxc;         // Cell center X-dx
        lineArray dyc;         // Cell center Y-dy
        lineArray dzc;         // Cell center Z-dz
        lineArray dxb;         // Cell boundary X-dx
        lineArray dyb;         // Cell boundary Y-dy
        lineArray dzb;         // Cell boundary Z-dz
        lineArray hy;
        planeArray hz;
        lineArray hyc;
        planeArray hzc;
        planeArray hz1;
        planeArray hz2;
        lineArray grav_r;
        lineArray grav_z;
        volumeArray delta_ke; // Remap kinetic energy correction
        volumeArray x, y, z;
        volumeArray xp, yp, zp;

        volumeArray bx1;       // X-magnetic field at half timestep
        volumeArray by1;       // Y-magnetic field at half timestep
        volumeArray bz1;       // Z-magnetic field at half timestep
        volumeArray alpha1;    // Alpha1 coefficient for magnetic field update
        volumeArray alpha2;    // Alpha2 coefficient for magnetic field update
        volumeArray alpha3;    // Alpha3 coefficient for magnetic field update
        volumeArray visc_heat; // Viscous heating
        volumeArray pressure;  // Pressure array
        volumeArray p_e;       // Electron pressure
        volumeArray p_i;       // Ion pressure
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
        volumeArray curlb;     // Curl of the magnetic field
        volumeArray dm;     // Mass flux for remap

        bool isxLB = false; // Is this processor on the x-min boundary
        bool isxUB = false; // Is this processor on the x-max boundary
        bool isyLB = false; // Is this processor on the y-min boundary
        bool isyUB = false; // Is this processor on the y-max boundary
        bool iszLB = false; // Is this processor on the z-min boundary
        bool iszUB = false; // Is this processor on the z-max boundary

        MPI_Datatype mpiType = MPI_DATATYPE_NULL; // MPI datatype for T_dataType
    };


    class LARE3D
    {
    private:
        SAMS::harness &harness;



        /*SAMS::variableDef *rho=nullptr;
        SAMS::variableDef *energy_electron=nullptr;
        SAMS::variableDef *energy_ion=nullptr;
        SAMS::variableDef *vx=nullptr;
        SAMS::variableDef *vy=nullptr;
        SAMS::variableDef *vz=nullptr;
        SAMS::variableDef *vx1=nullptr;
        SAMS::variableDef *vy1=nullptr;
        SAMS::variableDef *vz1=nullptr;
        SAMS::variableDef *bx=nullptr;
        SAMS::variableDef *by=nullptr;
        SAMS::variableDef *bz=nullptr;
        SAMS::variableDef *dm=nullptr;*/

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
        LARE3D(SAMS::harness &harnessRef) : harness(harnessRef), manager(harnessRef.memoryRegistry.getArrayManager()) {}

        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        constexpr static std::string_view name = "LARE3D";

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

        /**
         * Physics timestep functions
         * This is the predictor step of the LARE3D timestep
         * @param data LARE3D simulation data
         */
        void startOfTimestep(simulationData &data, SAMS::controlFunctions &controlFns){
            lagrangian_step(data, controlFns);
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

        /**
         * Lagrangian step for the LARE3D
         * @param data Simulation data struct
         * This function performs a Lagrangian step for the LARE3D
         */
        void lagrangian_step(simulationData &data, SAMS::controlFunctions &controlFns);

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

#endif // SHARED_DATA_H
