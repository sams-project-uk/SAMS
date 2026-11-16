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
#include "kt.h"
#include "io/writers.h"
#include "io/readers.h"
#include "variableRegistry.h"
#include "axisRegistry.h"

namespace KT
{

    /**
     * Register KT3D's axes with the axis registry.
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::registerAxes(SAMS::harness &harnessRef)
    {
        auto &axisReg = harnessRef.axisRegistry;
        axisReg.registerAxis("X", SAMS::MPIAxis(0));
        axisReg.registerAxis("Y", SAMS::MPIAxis(1));
        axisReg.registerAxis("Z", SAMS::MPIAxis(2));
    }

    /**
     * Register the conserved variables, which are all cell centred
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::registerVariables(SAMS::harness &harnessRef)
    {
        auto &varRegistry = harnessRef.variableRegistry;

        const int ghosts = 2; // 2 ghost cells for MUSCL reconstruction and flux-CT

        varRegistry.registerVariable<T_dataType>("rho", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        varRegistry.registerVariable<T_dataType>("mx", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        varRegistry.registerVariable<T_dataType>("my", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        varRegistry.registerVariable<T_dataType>("mz", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        varRegistry.registerVariable<T_dataType>("energy", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        varRegistry.registerVariable<T_dataType>("bx", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        varRegistry.registerVariable<T_dataType>("by", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        varRegistry.registerVariable<T_dataType>("bz", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
    }

    /**
     * Set default values for control parameters
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::defaultValues(simulationData &data, SAMS::KVS &kvs)
    {
        // Ratio of specific heats
        data.gas_gamma = kvs.getReal("/KT/gas_gamma", 5.0 / 3.0);
        // Courant number
        data.cfl = kvs.getReal("/KT/cfl", 0.8);
        data.dt = 0.0;
    }

    /**
     * Put the control parameters back into the deck
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::repopulateDeck(simulationData &data, SAMS::KVS &kvs)
    {
        kvs.setContentPath("/KT/gas_gamma", data.gas_gamma);
        kvs.setContentPath("/KT/cfl", data.cfl);
    }

    /**
     * Get the variables from the variable registry and allocate the work arrays
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::getVariables(SAMS::harness &harnessRef, simulationData &data)
    {
        auto &axRegistry = harnessRef.axisRegistry;
        auto &varRegistry = harnessRef.variableRegistry;

        data.nx = axRegistry.getLocalDomainElements("X", SAMS::staggerType::CENTRED);
        data.ny = axRegistry.getLocalDomainElements("Y", SAMS::staggerType::CENTRED);
        data.nz = axRegistry.getLocalDomainElements("Z", SAMS::staggerType::CENTRED);

        data.xcLocalRange = axRegistry.getLocalRange("X", SAMS::staggerType::CENTRED);
        data.ycLocalRange = axRegistry.getLocalRange("Y", SAMS::staggerType::CENTRED);
        data.zcLocalRange = axRegistry.getLocalRange("Z", SAMS::staggerType::CENTRED);

        varRegistry.fillPPArray("rho", data.U.rho);
        varRegistry.fillPPArray("mx", data.U.mx);
        varRegistry.fillPPArray("my", data.U.my);
        varRegistry.fillPPArray("mz", data.U.mz);
        varRegistry.fillPPArray("energy", data.U.energy);
        varRegistry.fillPPArray("bx", data.U.bx);
        varRegistry.fillPPArray("by", data.U.by);
        varRegistry.fillPPArray("bz", data.U.bz);

        // The fluxes and the variables at the start of the step have the same shape as the variables
        for (conservedVariables *work : {&data.U0, &data.fx, &data.fy, &data.fz}){
            manager.allocate(work->rho, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
            manager.allocate(work->mx, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
            manager.allocate(work->my, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
            manager.allocate(work->mz, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
            manager.allocate(work->energy, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
            manager.allocate(work->bx, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
            manager.allocate(work->by, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
            manager.allocate(work->bz, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        }
        manager.allocate(data.ex, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.ey, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.ez, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);

        // The width of cell i is the distance between edges i - 1 and i
        axRegistry.fillPPLocalDelta("X", data.dxb, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalDelta("Y", data.dyb, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalDelta("Z", data.dzb, SAMS::staggerType::HALF_CELL);
    }

    /**
     * Apply the boundary conditions to the conserved variables
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::boundary_conditions()
    {
        auto &varRegistry = harness.variableRegistry;
        pw::fence();
        varRegistry.applyBoundaryConditions("rho");
        varRegistry.applyBoundaryConditions("mx");
        varRegistry.applyBoundaryConditions("my");
        varRegistry.applyBoundaryConditions("mz");
        varRegistry.applyBoundaryConditions("energy");
        varRegistry.applyBoundaryConditions("bx");
        varRegistry.applyBoundaryConditions("by");
        varRegistry.applyBoundaryConditions("bz");
    }

    /**
     * Register the outputs that KT3D can write. They are all written by default and needed for a restart
     * @param outputs The registry of outputs
     */
    template<typename T_EOS>
    void KT3D<T_EOS>::registerOutputVariables(SAMS::outputRegistry &outputs)
    {
        rhoOutput = outputs.registerOutput("rho", "Mass density");
        mxOutput = outputs.registerOutput("px", "Momentum density in the x direction");
        myOutput = outputs.registerOutput("py", "Momentum density in the y direction");
        mzOutput = outputs.registerOutput("pz", "Momentum density in the z direction");
        energyOutput = outputs.registerOutput("energy", "Total energy density");
        bxOutput = outputs.registerOutput("bx", "Magnetic field in the x direction");
        byOutput = outputs.registerOutput("by", "Magnetic field in the y direction");
        bzOutput = outputs.registerOutput("bz", "Magnetic field in the z direction");
    }

    /**
     * Register the output mesh. Every variable is cell centred
     * @tparam T_writer The writer type
     * @param regWriter The writer for variables in the registries
     */
    template<typename T_EOS>
    template<typename T_writer>
    void KT3D<T_EOS>::registerDumpMeshes(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &regWriter, const SAMS::outputLevel &)
    {
        using SAMS::staggerType;
        regWriter.registerRectilinearMesh("MeshCC", {{"X", staggerType::CENTRED}, {"Y", staggerType::CENTRED}, {"Z", staggerType::CENTRED}});
    }

    /**
     * Register the output variables that the dump writes
     * @tparam T_writer The writer type
     * @param regWriter The writer for variables in the registries
     * @param level The outputs that the dump writes
     */
    template<typename T_EOS>
    template<typename T_writer>
    void KT3D<T_EOS>::registerDumpVariables(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &regWriter, const SAMS::outputLevel &level, simulationData &)
    {
        // Restart dumps write the variables with their ghost cells
        const bool restart = level.isActive("restart");
        auto registerVariable = [&](const SAMS::outputHandle &handle, const char* varName, const char* outputName){
            if (!level.includes(handle)) return;
            if (restart) regWriter.template registerDataWithGhosts<T_dataType, 3>(outputName, "MeshCC", varName);
            else regWriter.template registerData<T_dataType, 3>(outputName, "MeshCC", varName);
        };
        registerVariable(rhoOutput, "rho", "rho");
        registerVariable(mxOutput, "mx", "px");
        registerVariable(myOutput, "my", "py");
        registerVariable(mzOutput, "mz", "pz");
        registerVariable(energyOutput, "energy", "energy");
        registerVariable(bxOutput, "bx", "bx");
        registerVariable(byOutput, "by", "by");
        registerVariable(bzOutput, "bz", "bz");
    }

    /**
     * Read the variables of KT3D from a restart dump, which holds them with their ghost cells
     * @tparam T_reader The reader type
     * @param regReader The reader for variables in the registries
     */
    template<typename T_EOS>
    template<typename T_reader>
    void KT3D<T_EOS>::restart(SAMS::IO::reader<T_reader> &, SAMS::IO::registryReader<T_reader> &regReader)
    {
        regReader.template readDataWithGhosts<T_dataType, 3>("rho", "rho");
        regReader.template readDataWithGhosts<T_dataType, 3>("px", "mx");
        regReader.template readDataWithGhosts<T_dataType, 3>("py", "my");
        regReader.template readDataWithGhosts<T_dataType, 3>("pz", "mz");
        regReader.template readDataWithGhosts<T_dataType, 3>("energy", "energy");
        regReader.template readDataWithGhosts<T_dataType, 3>("bx", "bx");
        regReader.template readDataWithGhosts<T_dataType, 3>("by", "by");
        regReader.template readDataWithGhosts<T_dataType, 3>("bz", "bz");
    }

    #include "./instantiateWriters.h"
    #include "./instantiateReaders.h"
    #include "./instantiate.h"
}
