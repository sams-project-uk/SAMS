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
    template<typename T_EOS = LARE::idealGas>
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
        void setScalarBcs(std::string varName, SAMS::harness &harness, SAMS::T_dataType clampValue);
        void setScalarBcs(std::string varName, SAMS::harness &harness);
        void setYParallelVectorBcs(std::string varName, SAMS::harness &harness, SAMS::T_dataType clampValue);
        void setYParallelVectorBcs(std::string varName, SAMS::harness &harness);

        inline static constexpr auto nameType = SAMS::makeName("EmeryWindTunnel") + T_EOS::name;        

    public:
        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        constexpr static std::string_view name = nameType;

        /**
         * Emery is not just an initial condition, so time it
         */
        constexpr static bool timeSimulation = true;

        using dataPack = emeryParams;

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
        void controlVariables(LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams, SAMS::KVS &kvs);

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
        void setBoundaryConditions(SAMS::harness &harness, LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams);

        /**
         * Initialize the Sod Shock Tube initial conditions
         * @param harnessRef SAMS harness
         * @param data LARE3D simulation data
         */
        void initialConditions(SAMS::harness &harnessRef, LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams);

        /**
         * Execute the first half of the timestep for the Emery Wind Tunnel problem
         * This is needed since emery has to clamp the velocity in the step to zero
         * @param data LARE3D simulation data
         */
        void startOfTimestep(LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams);

        /**
         * Execute the second half of the timestep for the Emery Wind Tunnel problem
         * This is needed since emery has to clamp the velocity in the step to zero
         * @param data LARE3D simulation data
         */
        void halfTimestep(LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams);

        /**
         * Execute the end of the timestep for the Emery Wind Tunnel problem
         * This is needed since emery has to clamp the velocity in the step to zero
         * @param data LARE3D simulation data
         */
        void endOfTimestep(LARE::LARE3D<T_EOS>::simulationData &data, emeryParams &problemParams);
    };

    #define EOS_DEF(value) extern template class EmeryWindTunnel<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}
#endif // EMERYWINDTUNNEL_H
