#ifndef DECKINITIALCONDITIONS_H
#define DECKINITIALCONDITIONS_H

#include "harness.h"
#include "runner.h"
#include "shared_data.h"
#include "builtInBoundaryConditions.h"
#include "deckHelpers/deckBoundaries.h"

namespace LARE
{

    namespace pw = portableWrapper;

    /**
     * Initial and boundary conditions from the deck
     * @details Each key of the initial_conditions block is the name of a variable in the variable registry and its value is an expression for it.
     * The keys are set in the order of the deck, so an expression can use the values of variables set by earlier keys.
     * Boundary keys are x_min to z_max or x to z, in a variable block or at the top level as a default. Values: periodic, symmetric, zero_gradient, clamp(v), mirror(v), expression(e).
     * Remap velocities follow velocities; LARE/dm follows rho with zero clamp/mirror values unless given
     */
    template<typename T_EOS = LARE::idealGas>
    class LAREDeckInitialConditions
    {
    private:

        inline static constexpr auto nameType = SAMS::makeName("LAREDeckInitialConditions") + T_EOS::name;

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
         * Run before LARE3D, so that the parser variables are registered before LARE3D sets its parameters
         */
        constexpr static uint8_t runnerOrder = 64;

        /**
         * Activate the deck services, which hold the host data of the deck
         * @param runnerRef The runner
         */
        template <typename T_runner>
        void runnerInteraction(T_runner &runnerRef)
        {
            runnerRef.activatePackage("deckServices");
        }

        /**
         * Default deck keys, and LARE3D's parameters as parser variables
         * @param data LARE3D simulation data
         * @param kvs The key value store
         * @param hostData The host data of the deck, which holds the parser
         */
        void defaultDeck(typename LARE::LARE3D<T_EOS>::simulationData &data, SAMS::KVS &kvs, SAMS::deckHostData<3> &hostData);

        /**
         * Read the end time
         * @param data LARE3D simulation data
         * @param kvs The key value store
         */
        void controlVariables(typename LARE::LARE3D<T_EOS>::simulationData &data, SAMS::KVS &kvs);

        /**
         * Set up the simulation domain
         * @param harness SAMS harness
         * @param kvs The key value store
         */
        void setDomain(SAMS::harness &harness, SAMS::KVS &kvs);

        /**
         * Set boundary conditions for the simulation from the boundaries block
         * @param harness SAMS harness
         * @param hostData The host parameters of deck expressions
         * @param kvs The key value store
         * @param tData SAMS time state data
         */
        void setBoundaryConditions(SAMS::harness &harness, SAMS::deckHostData<3> &hostData, SAMS::KVS &kvs, SAMS::timeState &tData);

        /**
         * Set the variables from the expressions in the initial_conditions block of the deck
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         * @param kvs The key value store
         * @param files The locations of the files of the run
         * @param hostData The host parameters of deck expressions, which hold the parser
         */
        void initialConditions(SAMS::harness &harnessRef, typename LARE::LARE3D<T_EOS>::simulationData &data, SAMS::KVS &kvs, SAMS::fileInformation &files, SAMS::deckHostData<3> &hostData);
    };

    #define EOS_DEF(value) extern template class LAREDeckInitialConditions<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}
#endif //DECKINITIALCONDITIONS_H
