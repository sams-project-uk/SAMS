#ifndef HLLEDECKINITIALCONDITIONS_H
#define HLLEDECKINITIALCONDITIONS_H

#include "harness.h"
#include "runner.h"
#include "ConservedSolver/conservedSolver.h"
#include "deckHelpers/deckBoundaries.h"

namespace conservedVariable
{

    namespace pw = portableWrapper;

    /**
     * Initial and boundary conditions from the deck for a solver of the conserved variables, such as HLLE3D or KT3D
     * @details As LAREDeckInitialConditions, for rho, px, py, pz, energy, bx, by and bz. Variables that the deck does not set are 0
     */
    class deckInitialConditions
    {
    private:

        inline static constexpr auto nameType = SAMS::makeName("conservedVariableDeckInitialConditions");

        SAMS::T_dataType t_end;

    public:
        constexpr static std::string_view name = nameType;

        constexpr static bool timeSimulation = false;

        /**
         * Run first, so that the parser variables are registered before other packages read the deck
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
         * Default deck keys, and the parameters as parser variables
         */
        void defaultDeck(conservedSolver::simulationData &data, SAMS::KVS &kvs, SAMS::deckHostData<3> &hostData);

        /**
         * Read the end time
         */
        void controlVariables(SAMS::KVS &kvs);

        void setDomain(SAMS::harness &harness, SAMS::KVS &kvs);

        void setBoundaryConditions(SAMS::harness &harness, SAMS::deckHostData<3> &hostData, SAMS::KVS &kvs, SAMS::timeState &tData);

        void initialConditions(SAMS::harness &harnessRef, SAMS::KVS &kvs, SAMS::fileInformation &files, SAMS::deckHostData<3> &hostData);
    };

}
#endif //HLLEDECKINITIALCONDITIONS_H
