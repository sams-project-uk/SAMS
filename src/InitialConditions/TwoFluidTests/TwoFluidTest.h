#ifndef TWOFLUIDTEST_H
#define TWOFLUIDTEST_H

#include "harness.h"
#include "runner.h"
#include "builtInBoundaryConditions.h"
#include "TWOFLUID/twofluid.h"
//#include "../PhysicsPackages/TWOFLUID/twofluid.h"


namespace examples
{

    namespace pw = portableWrapper;

    /**
     * Initial conditions, boundary conditions and domain setup for the Sod Shock Tube problem
     */
    class TwoFluidTest
    {
    private:

        /**
         * Internal function to attach boundary conditions to a variable
         * @param varName Name of the variable to attach boundary conditions to
         * @param harness SAMS harness
         */
        void attachBoundaryConditions(const std::string& varName, SAMS::harness &harness);

    public:

        using T_EOS = LARE::idealGas;
        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        constexpr static std::string_view name = "TwoFluidTest";

        const std::string problem = "hillier"; // Which problem/ TODO - this should be a data item somewhere...
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
        void controlVariables(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral);

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param data LARE3D simulation data
         */
        void setDomain(SAMS::harness &harness, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral);

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
        void initialConditions(SAMS::harness &harnessRef, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral);

       /**
         * Check whether to terminate the simulation
         * @param terminate Boolean flag to set to true to terminate the simulation
         * @param data LARE3D simulation data
         * @param tData SAMS time state data
         * This function checks whether the simulation should terminate based on LARE3D data.
         * It sets the terminate flag to true if the simulation should end.
         */
        void queryTerminate(bool &terminate, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, SAMS::timeState &tData);

        /**
         * Check whether to output data to disk
         * @param shouldOutput Boolean flag to set to true to output data
         * @param data LARE3D simulation data
        * @param tData SAMS time state data
         * This function checks whether data should be output to disk based on LARE3D data.
         * It returns true if data should be output.
         */
        void queryOutput(bool &shouldOutput, LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, SAMS::timeState &tData);

    };
}
#endif // SODSHOCKTUBE_H
