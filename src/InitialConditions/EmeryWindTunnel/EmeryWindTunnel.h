#ifndef EMERYWINDTUNNEL_H
#define EMERYWINDTUNNEL_H

#include "harness.h"
#include "runner.h"
#include "shared_data.h"
#include "builtInBoundaryConditions.h"

namespace examples
{

    namespace pw = portableWrapper;

    

    /**
     * Initial conditions, boundary conditions and domain setup for the Emery Wind Tunnel problem
     */
    class EmeryWindTunnel
    {
    public:
        struct emeryParams{
            SAMS::T_dataType ambPressure;
            SAMS::T_dataType density;
            SAMS::T_dataType flowVx;

            SAMS:: T_dataType step_x = 0.6; // Step starts at x=0.6
            SAMS:: T_dataType step_height = 0.2; // Step is 0.2 high
        };

    private:
        void setZbcs( SAMS::variableDef &varDef);
        void setInflowXbcs(SAMS::variableDef &varDef, SAMS::T_dataType clampValue);
        void setZeroGradientXbcs(SAMS::variableDef &varDef);
        void setMirrorYbcs(SAMS::variableDef &varDef);
        void setZeroGradientYbcs(SAMS::variableDef &varDef);
        void setScalarBcs(std::string varName, SAMS::harness &harness, SAMS::T_dataType clampValue);
        void setScalarBcs(std::string varName, SAMS::harness &harness);
        void setYParallelVectorBcs(std::string varName, SAMS::harness &harness, SAMS::T_dataType clampValue);
        void setYParallelVectorBcs(std::string varName, SAMS::harness &harness);


    public:
        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        constexpr static std::string_view name = "EmeryWindTunnel";

        /**
         * Initial conditions should not be timed
         */
        constexpr static bool timeSimulation = false;

        using dataPack = emeryParams;

        /**
         * Set up the LARE style simulation control variables
         * @note This should later be moved to a more general approach
         * when we have multiple core solvers.
         * @param data LARE3D simulation data
         */
        void controlVariables(LARE::simulationData &data, emeryParams &problemParams);

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param data LARE3D simulation data
         */
        void setDomain(SAMS::harness &harness, LARE::simulationData &data);

        /**
         * Set boundary conditions for the simulation
         * @param harness SAMS harness
         */
        void setBoundaryConditions(SAMS::harness &harness, LARE::simulationData &data, emeryParams &problemParams);

        /**
         * Initialize the Sod Shock Tube initial conditions
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        void initialConditions(SAMS::harness &harnessRef, LARE::simulationData &data, emeryParams &problemParams);

       /**
         * Check whether to terminate the simulation
         * @param terminate Boolean flag to set to true to terminate the simulation
         * @param data LARE3D simulation data
         * This function checks whether the simulation should terminate based on LARE3D data.
         * It sets the terminate flag to true if the simulation should end.
         */
        void queryTerminate(bool &terminate, LARE::simulationData &data, SAMS::timeState &tData);

        /**
         * Check whether to output data to disk
         * @param shouldOutput Boolean flag to set to true to output data
         * @param data LARE3D simulation data
         * This function checks whether data should be output to disk based on LARE3D data.
         * It returns true if data should be output.
         */
        void queryOutput(bool &shouldOutput, LARE::simulationData &data, SAMS::timeState &tData);

        /**
         * Execute the first half of the timestep for the Emery Wind Tunnel problem
         * This is needed since emery has to clamp the velocity in the step to zero
         * @param data LARE3D simulation data
         */
        void startOfTimestep(LARE::simulationData &data, emeryParams &problemParams);

        /**
         * Execute the second half of the timestep for the Emery Wind Tunnel problem
         * This is needed since emery has to clamp the velocity in the step to zero
         * @param data LARE3D simulation data
         */
        void halfTimestep(LARE::simulationData &data, emeryParams &problemParams);

        /**
         * Execute the end of the timestep for the Emery Wind Tunnel problem
         * This is needed since emery has to clamp the velocity in the step to zero
         * @param data LARE3D simulation data
         */
        void endOfTimestep(LARE::simulationData &data, emeryParams &problemParams);
    };
}
#endif // EMERYWINDTUNNEL_H
