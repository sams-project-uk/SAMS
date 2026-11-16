#ifndef HLLEORSZAGTANG_H
#define HLLEORSZAGTANG_H

#include "harness.h"
#include "runner.h"
#include "HLLE/hlle.h"
#include "builtInBoundaryConditions.h"

namespace examples
{

    namespace pw = portableWrapper;

    /**
     * Initial conditions, boundary conditions and domain setup for the Orszag-Tang Vortex problem with the HLLE solver
     */
    template<typename T_EOS = LARE::idealGas>
    class HLLEOrszagTang
    {
    private:

        /**
         * Internal function to attach boundary conditions to a variable
         * @param varName Name of the variable to attach boundary conditions to
         * @param harness SAMS harness
         */
        void attachBoundaryConditions(const std::string& varName, SAMS::harness &harness);

        inline static constexpr auto nameType = SAMS::makeName("HLLEOrszagTang") + T_EOS::name;

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
         * Initialize the Orszag-Tang initial conditions
         * @param harnessRef SAMS harness
         * @param data HLLE3D simulation data
         */
        void initialConditions(SAMS::harness &harnessRef, typename HLLE::HLLE3D<T_EOS>::simulationData &data);
    };

    #define EOS_DEF(value) extern template class HLLEOrszagTang<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}
#endif //HLLEORSZAGTANG_H
