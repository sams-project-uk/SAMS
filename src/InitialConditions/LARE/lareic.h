#ifndef LAREIC_SHARED_DATA_H
#define LAREIC_SHARED_DATA_H

#include "shared_data.h"
#include "harness.h"
#include "lareBoundaryClass.h"
#include "runner.h"

namespace LARE{

    class LARE3DInitialConditions{

        using bcFuncPtr = void(*)(simulationData &, SAMS::timeState &time, int, SAMS::domain::edges);

        //Definitions for boundary condition functions
        static void bx_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void by_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void bz_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void energy_ion_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void energy_electron_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void density_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void vx_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void vy_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void vz_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void remap_vx_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void remap_vy_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void remap_vz_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void dm_x_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void dm_y_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        static void dm_z_bcs(simulationData &data, SAMS::timeState &time, int dimension, SAMS::domain::edges edge);
        public:

        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        constexpr static std::string_view name = "LARE3DInitialConditions";

        /**
         * Initial conditions should not be timed
         */
        constexpr static bool timeSimulation = false;

        /**
         * Set up control variables for LARE3D
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        void control_variables(SAMS::harness &harnessRef, simulationData &data);

        /**
         * Set the initial conditions for LARE3D
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        void initial_conditions(SAMS::harness &harnessRef, simulationData &data);

        /**
         * Shim to match the SAMS runner control function style
         * Simply calls control_variables
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        void controlVariables(SAMS::harness &harnessRef, simulationData &data){
            control_variables(harnessRef, data);
        }

        /**
         * Set the initial conditions for LARE3D
         * @param data LARE3D simulation data
         */
        void initialConditions(SAMS::harness &harnessRef, simulationData &data){
            initial_conditions(harnessRef, data);
        }

        /**
         * Attach a boundary condition function to a variable based on LARE3D BC types
         * @param varName Name of the variable
         * @param bcFunc Pointer to the boundary condition function
         * @param harnessRef Reference to the SAMS harness
         * @param data LARE3D simulation data
         */
        void setBoundary(const std::string& varName, bcFuncPtr bcFunc, SAMS::harness &harnessRef, simulationData &data, SAMS::timeState &timeStateRef){
            auto &varRegistry = harnessRef.variableRegistry;
            std::shared_ptr<SAMS::boundaryConditions> bcPtr = std::make_shared<LAREBoundaryConditions>(LAREBoundaryConditions(bcFunc, data, timeStateRef));

            if (data.xbc_min != BCType::BC_EXTERNAL){
                varRegistry.addBoundaryCondition(varName, 0, SAMS::domain::edges::lower, bcPtr);
            }
            if (data.xbc_max != BCType::BC_EXTERNAL){
                varRegistry.addBoundaryCondition(varName, 0, SAMS::domain::edges::upper, bcPtr);
            }
            if (data.ybc_min != BCType::BC_EXTERNAL){
                varRegistry.addBoundaryCondition(varName, 1, SAMS::domain::edges::lower, bcPtr);
            }
            if (data.ybc_max != BCType::BC_EXTERNAL){
                varRegistry.addBoundaryCondition(varName, 1, SAMS::domain::edges::upper, bcPtr);
            }
            if (data.zbc_min != BCType::BC_EXTERNAL){
                varRegistry.addBoundaryCondition(varName, 2, SAMS::domain::edges::lower, bcPtr);
            }
            if (data.zbc_max != BCType::BC_EXTERNAL){
                varRegistry.addBoundaryCondition(varName, 2, SAMS::domain::edges::upper, bcPtr);
            }
        }

        /**
         * Set the domain and boundary conditions for LARE3D
         * @details
         * Sets up the number of grid points and physical dimensions in each direction.
         * @param harness Reference to the SAMS harness
         * @param data LARE3D simulation data
         */
        void setDomain(SAMS::harness &harness, simulationData &data)
        {
            auto &axisReg = harness.axisRegistry;
            axisReg.setDomain("X", data.nx, data.x_min, data.x_max);
            axisReg.setDomain("Y", data.ny, data.y_min, data.y_max);
            axisReg.setDomain("Z", data.nz, data.z_min, data.z_max);
        }

        /**
         * Set the boundary conditions for LARE3D
         * @details
         * Attaches boundary condition functions to each variable based on LARE3D BC types.
         * @param harness Reference to the SAMS harness
         * @param data LARE3D simulation data
         * @param timeStateRef Reference to the SAMS time state (used for time-dependent BCs)
         */
        void setBoundaryConditions(SAMS::harness &harness, simulationData &data, SAMS::timeState &timeStateRef)
        {
            //Grab variables and set boundary condition functions
            setBoundary("bx", bx_bcs, harness, data, timeStateRef);
            setBoundary("by", by_bcs, harness, data, timeStateRef);
            setBoundary("bz", bz_bcs, harness, data, timeStateRef);
            setBoundary("energy_ion", energy_ion_bcs, harness, data, timeStateRef);
            setBoundary("energy_electron", energy_electron_bcs, harness, data, timeStateRef);
            setBoundary("rho", density_bcs, harness, data, timeStateRef);
            setBoundary("vx", vx_bcs, harness, data, timeStateRef);
            setBoundary("vy", vy_bcs, harness, data, timeStateRef);
            setBoundary("vz", vz_bcs, harness, data, timeStateRef);
            setBoundary("LARE/vx1", remap_vx_bcs, harness, data, timeStateRef);
            setBoundary("LARE/vy1", remap_vy_bcs, harness, data, timeStateRef);
            setBoundary("LARE/vz1", remap_vz_bcs, harness, data, timeStateRef);
            setBoundary("LARE/dm", dm_x_bcs, harness, data, timeStateRef);

        }

       /**
         * Check whether to terminate the simulation
         * @param terminate Boolean flag to set to true to terminate the simulation
         * @param data LARE3D simulation data
         * This function checks whether the simulation should terminate based on LARE3D data.
         * It sets the terminate flag to true if the simulation should end.
         */
        void queryTerminate(bool &terminate, simulationData &data, SAMS::timeState &tData){
            if ((tData.step >= data.nsteps && data.nsteps >= 0) || (tData.time >= data.t_end)){
                terminate |= true;
            }
        }

        /**
         * Check whether to output data to disk
         * @param shouldOutput Boolean flag to set to true to output data
         * @param data LARE3D simulation data
         * This function checks whether data should be output to disk based on LARE3D data.
         * It returns true if data should be output.
         */
        void queryOutput(bool &shouldOutput, simulationData &data, SAMS::timeState &tData){
            return;
            if (data.dt_snapshots > 0){
                if (tData.time >= (data.lastOutputTime + data.dt_snapshots)){
                    shouldOutput |= true;
                }
            }
        }

    };

}//namespace LARE

#endif