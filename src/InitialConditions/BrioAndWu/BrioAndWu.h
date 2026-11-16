#ifndef BRIOANDWU_H
#define BRIOANDWU_H

#include "harness.h"
#include "runner.h"
#include "shared_data.h"
#include "builtInBoundaryConditions.h"

namespace examples
{

    namespace pw = portableWrapper;

    /**
     * Initial conditions, boundary conditions and domain setup for the Brio and Wu problem
     */
    template<typename T_EOS = LARE::idealGas>
    class BrioAndWu
    {
    private:

        /**
         * Internal function to attach boundary conditions to a variable
         * @param varName Name of the variable to attach boundary conditions to
         * @param harness SAMS harness
         */
        void attachBoundaryConditions(const std::string& varName, SAMS::harness &harness);

        inline static constexpr auto nameType = SAMS::makeName("BrioAndWu") + T_EOS::name;

    public:
        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        constexpr static std::string_view name = nameType;

        /**
         * Initial conditions should not be timed
         */
        constexpr static bool timeSimulation = false;

        /**
         * Set the deck keys that the deck does not set
         * @param kvs The key value store
         */
        void defaultDeck(SAMS::KVS &kvs);

        /**
         * Set up the LARE style simulation control variables
         * @note This should later be moved to a more general approach
         * when we have multiple core solvers.
         * @param data LARE3D simulation data
         */
        void controlVariables(LARE::LARE3D<T_EOS>::simulationData &data, SAMS::KVS &kvs);

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param kvs The key value store
         */
        void setDomain(SAMS::harness &harness, SAMS::KVS &kvs);
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
        void initialConditions(SAMS::harness &harnessRef, LARE::LARE3D<T_EOS>::simulationData &data);

    };

    #define EOS_DEF(value) extern template class BrioAndWu<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}
#endif // BRIOANDWU_H