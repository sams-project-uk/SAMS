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
#ifndef SHARED_DATA_HLLE_H
#define SHARED_DATA_HLLE_H

#include "LARECommon/types.h"
#include "LARECommon/eos.h"
#include "pp/parallelWrapper.h"
#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"
#include "io/readerProto.h"
#include "ConservedSolver/conservedSolver.h"

namespace HLLE
{
    namespace pw = portableWrapper;

    using T_dataType = SAMS::T_dataType;
    using T_indexType = SAMS::T_indexType;
    using volumeArray = portableWrapper::acceleratedArray<T_dataType, 3>;
    using lineArray = portableWrapper::acceleratedArray<T_dataType, 1>;

    /**
     * The conserved variables of ideal MHD, or their fluxes, at cell centres or faces
     */
    using conservedVariables = conservedSolver::variables;

    /**
     * Ideal MHD with an HLLE Riemann solver, MUSCL reconstruction, Toth's flux-CT for div B and a two stage Runge-Kutta step.
     * All variables are cell centred. Units are normalised, with mu0 = 1
     */
    template<typename T_EOS = LARE::idealGas>
    class HLLE3D
    {
    public:

        using simulationData = conservedSolver::simulationData;

    private:

        SAMS::harness &harness;

        /**
         * Portable array manager for handling memory allocation and deallocation
         */
        pw::portableArrayManager &manager;

        /**
         * Handles of the outputs that HLLE3D registers
         */
        SAMS::outputHandle rhoOutput, mxOutput, myOutput, mzOutput, energyOutput, bxOutput, byOutput, bzOutput;

        /**
         * Apply the boundary conditions to the conserved variables
         */
        void boundary_conditions();

        /**
         * Compute the fluxes and update the conserved variables for one stage of the Runge-Kutta step
         * @param data Simulation data
         * @param weightOld Weight of the variables at the start of the step
         * @param weightNew Weight of the updated variables
         */
        void stage(simulationData &data, T_dataType weightOld, T_dataType weightNew);

    public:

        // These launch device lambdas, which nvcc does not allow in private member functions

        /**
         * Compute the HLLE fluxes on the faces normal to one direction
         * @param data Simulation data
         * @param direction The direction normal to the faces
         * @param flux The fluxes to fill
         */
        void compute_fluxes(simulationData &data, int direction, conservedVariables &flux);

        /**
         * Replace the fluxes of the magnetic field with those of Toth's flux-CT
         * @param data Simulation data
         */
        void constrained_transport(simulationData &data);

        /**
         * Update the conserved variables as weightOld * U0 + weightNew * (U - dt * div F)
         * @param data Simulation data
         * @param weightOld Weight of the variables at the start of the step
         * @param weightNew Weight of the updated variables
         */
        void update(simulationData &data, T_dataType weightOld, T_dataType weightNew);

        /**
         * Constructor
         */
        HLLE3D(SAMS::harness &harnessRef) : harness(harnessRef), manager(harnessRef.memoryRegistry.getArrayManager()) {}

        /**
         * Name of the simulation. Must be unique across all simulations in the executable.
         */
        inline static constexpr auto nameType = SAMS::makeName("HLLE3D") + T_EOS::name;
        constexpr static std::string_view name = nameType;

        /**
         * HLLE3D is a core simulation
         */
        constexpr static bool coreSimulation = true;

        /**
         * HLLE3D should be timed
         */
        constexpr static bool timeSimulation = true;

        /**
         * HLLE3D's dataPack is simulationData
         */
        using dataPack = simulationData;

        /**
         * Register axes with the harness's axis registry
         * @param harnessRef SAMS harness
         */
        void registerAxes(SAMS::harness &harnessRef);

        /**
         * Register variables with the harness
         * @param harnessRef SAMS harness
         */
        void registerVariables(SAMS::harness &harnessRef);

        /**
         * Set default values for control parameters
         */
        void defaultValues(simulationData &data, SAMS::KVS &kvs);

        /**
         * Put the control parameters back into the deck
         */
        void repopulateDeck(simulationData &data, SAMS::KVS &kvs);

        /**
         * Get the variables from the variable registry and allocate the work arrays
         * @param harnessRef SAMS harness
         * @param data Simulation data
         */
        void getVariables(SAMS::harness &harnessRef, simulationData &data);

        /**
         * Apply the boundary conditions before the first step
         */
        void initializeBoundaryConditions() {boundary_conditions();}

        /**
         * First stage of the Runge-Kutta step
         * @param data Simulation data
         */
        void startOfTimestep(simulationData &data);

        /**
         * Second stage of the Runge-Kutta step
         * @param data Simulation data
         */
        void halfTimestep(simulationData &data);

        /**
         * Set the timestep from the fast magnetosonic speed
         * @param timeData SAMS timeState data
         * @param data Simulation data
         */
        void calculateTimestep(SAMS::timeState &timeData, simulationData &data);

        /**
         * Gather the timestep back after all packages have calculated it
         * @param timeData SAMS timeState data
         * @param data Simulation data
         */
        void getTimestep(SAMS::timeState &timeData, simulationData &data){
            data.dt = timeData.dt;
        }

        /**
         * Register the outputs that HLLE3D can write
         * @param outputs The registry of outputs
         */
        void registerOutputVariables(SAMS::outputRegistry &outputs);

        template<typename T>
        void registerDumpMeshes(SAMS::IO::writer<T> &writer, SAMS::IO::registryWriter<T> &regWriter, const SAMS::outputLevel &level);

        template<typename T>
        void registerDumpVariables(SAMS::IO::writer<T> &writer, SAMS::IO::registryWriter<T> &regWriter, const SAMS::outputLevel &level, simulationData &data);

        /**
         * Read the variables of HLLE3D from a restart dump
         * @param reader The reader of the dump
         * @param regReader The reader for variables in the registries
         */
        template<typename T>
        void restart(SAMS::IO::reader<T> &reader, SAMS::IO::registryReader<T> &regReader);
    };
}

#define EOS_DEF(value) extern template class HLLE::HLLE3D<value>;
EOS_DENSITY_ENERGY
#undef EOS_DEF

#endif // SHARED_DATA_HLLE_H
