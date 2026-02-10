#ifndef MHDROTOR_H
#define MHDROTOR_H

#include "harness.h"
#include "runner.h"
#include "shared_data.h"
#include "builtInBoundaryConditions.h"

namespace examples
{

    namespace pw = portableWrapper;

    /**
     * Initial conditions, boundary conditions and domain setup for the Balsara and Spicer MHD Rotor problem
     */
    class MHDRotor
    {
    private:

        /**
         * Internal function to attach boundary conditions to a variable
         * @param varName Name of the variable to attach boundary conditions to
         * @param harness SAMS harness
         */
        void attachBoundaryConditions(const std::string& varName, SAMS::harness &harness);

    public:
        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        constexpr static std::string_view name = "MHDRotor";

        /**
         * Initial conditions should not be timed
         */
        constexpr static bool timeSimulation = false;

        /**
         * Set up the LARE style simulation control variables
         * @note This should later be moved to a more general approach
         * when we have multiple core solvers.
         * @param data LARE3D simulation data
         */
        void controlVariables(LARE::simulationData &data);

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
        void setBoundaryConditions(SAMS::harness &harness);

        /**
         * Initialize the Sod Shock Tube initial conditions
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        void initialConditions(SAMS::harness &harnessRef, LARE::simulationData &data);

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
    };
}
#endif // MHDROTOR_H
