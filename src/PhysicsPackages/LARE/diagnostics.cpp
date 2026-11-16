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
#if defined(USE_HDF5)
#if !__has_include(<hdf5.h>)
#pragma error "HDF5 support is enabled but the HDF5 library is not available. Please install HDF5 or disable HDF5 support."
#endif
#endif

#include "io/writers.h"
#include "io/readers.h"
#include "shared_data.h"
#include "mpiManager.h"
#include "shared_data.h"

namespace LARE
{

    namespace pw = portableWrapper;

    /**
     * Register the outputs that LARE3D can write. They are all written by default and needed for a restart
     * @param outputs The registry of outputs
     */
    template<typename T_EOS>
    void LARE3D<T_EOS>::registerOutputVariables(SAMS::outputRegistry &outputs)
    {
        rhoOutput = outputs.registerOutput("rho", "Mass density");
        energyElectronOutput = outputs.registerOutput("energy_electron", "Specific internal energy of the electrons");
        energyIonOutput = outputs.registerOutput("energy_ion", "Specific internal energy of the ions");
        vxOutput = outputs.registerOutput("vx", "Velocity in the x direction, at the cell vertices");
        vyOutput = outputs.registerOutput("vy", "Velocity in the y direction, at the cell vertices");
        vzOutput = outputs.registerOutput("vz", "Velocity in the z direction, at the cell vertices");
        bxOutput = outputs.registerOutput("bx", "Magnetic field in the x direction, at the cell faces normal to x");
        byOutput = outputs.registerOutput("by", "Magnetic field in the y direction, at the cell faces normal to y");
        bzOutput = outputs.registerOutput("bz", "Magnetic field in the z direction, at the cell faces normal to z");
    }

    /**
     * Register the output meshes
     * @tparam T_writer The writer type
     * @param regWriter The writer for variables in the registries
     * @param level The outputs that the dump writes
     */
    template<typename T_EOS>
    template<typename T_writer>
    void LARE3D<T_EOS>::registerDumpMeshes(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &regWriter, [[maybe_unused]] const SAMS::outputLevel &level)
    {
        using SAMS::staggerType;
        // One mesh for each combination of staggers of the output variables
        regWriter.registerRectilinearMesh("MeshCC", {{"X", staggerType::CENTRED}, {"Y", staggerType::CENTRED}, {"Z", staggerType::CENTRED}});
        regWriter.registerRectilinearMesh("MeshNode", {{"X", staggerType::HALF_CELL}, {"Y", staggerType::HALF_CELL}, {"Z", staggerType::HALF_CELL}});
        regWriter.registerRectilinearMesh("MeshFaceX", {{"X", staggerType::HALF_CELL}, {"Y", staggerType::CENTRED}, {"Z", staggerType::CENTRED}});
        regWriter.registerRectilinearMesh("MeshFaceY", {{"X", staggerType::CENTRED}, {"Y", staggerType::HALF_CELL}, {"Z", staggerType::CENTRED}});
        regWriter.registerRectilinearMesh("MeshFaceZ", {{"X", staggerType::CENTRED}, {"Y", staggerType::CENTRED}, {"Z", staggerType::HALF_CELL}});
    }

    /**
     * Register the output variables that the dump writes
     * @tparam T_writer The writer type
     * @param regWriter The writer for variables in the registries
     * @param level The outputs that the dump writes
     */
    template<typename T_EOS>
    template<typename T_writer>
    void LARE3D<T_EOS>::registerDumpVariables(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &regWriter, const SAMS::outputLevel &level, simulationData &)
    {
        // Restart dumps write the variables with their ghost cells
        const bool restart = level.isActive("restart");
        auto registerVariable = [&](const SAMS::outputHandle &handle, const char* name, const char* meshName){
            if (!level.includes(handle)) return;
            if (restart) regWriter.template registerDataWithGhosts<T_dataType, 3>(name, meshName, name);
            else regWriter.template registerData<T_dataType, 3>(name, meshName, name);
        };
        registerVariable(rhoOutput, "rho", "MeshCC");
        registerVariable(energyElectronOutput, "energy_electron", "MeshCC");
        registerVariable(energyIonOutput, "energy_ion", "MeshCC");
        registerVariable(vxOutput, "vx", "MeshNode");
        registerVariable(vyOutput, "vy", "MeshNode");
        registerVariable(vzOutput, "vz", "MeshNode");
        registerVariable(bxOutput, "bx", "MeshFaceX");
        registerVariable(byOutput, "by", "MeshFaceY");
        registerVariable(bzOutput, "bz", "MeshFaceZ");
    }

    /**
     * Write the output meshes that are not written through the registry writer
     * @tparam T_writer The writer type
     */
    template<typename T_EOS>
    template<typename T_writer>
    void LARE3D<T_EOS>::writeDumpMeshes(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &)
    {
    }

    /**
     * Write the output variables that are not written through the registry writer
     * @tparam T_writer The writer type
     */
    template<typename T_EOS>
    template<typename T_writer>
    void LARE3D<T_EOS>::writeDumpVariables(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &)
    {
    }

    /**
     * Read the variables of LARE3D from a restart dump, which holds them with their ghost cells
     * @tparam T_reader The reader type
     * @param regReader The reader for variables in the registries
     */
    template<typename T_EOS>
    template<typename T_reader>
    void LARE3D<T_EOS>::restart(SAMS::IO::reader<T_reader> &, SAMS::IO::registryReader<T_reader> &regReader)
    {
        regReader.template readDataWithGhosts<T_dataType, 3>("rho", "rho");
        regReader.template readDataWithGhosts<T_dataType, 3>("energy_electron", "energy_electron");
        regReader.template readDataWithGhosts<T_dataType, 3>("energy_ion", "energy_ion");
        regReader.template readDataWithGhosts<T_dataType, 3>("vx", "vx");
        regReader.template readDataWithGhosts<T_dataType, 3>("vy", "vy");
        regReader.template readDataWithGhosts<T_dataType, 3>("vz", "vz");
        regReader.template readDataWithGhosts<T_dataType, 3>("bx", "bx");
        regReader.template readDataWithGhosts<T_dataType, 3>("by", "by");
        regReader.template readDataWithGhosts<T_dataType, 3>("bz", "bz");
    }

    #include "./instantiateWriters.h"
    #include "./instantiateReaders.h"

    template<typename T_EOS>
    void LARE3D<T_EOS>::energy_correction(simulationData &data)
    {
        using Range = pw::Range;
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_dataType dke = pw::max(-data.delta_ke(ix, iy, iz), 0.0) / (data.rho(ix, iy, iz) * data.cv(ix, iy, iz));
                data.energy_electron(ix, iy, iz) += 0.5 * dke;
                data.energy_ion(ix, iy, iz) += 0.5 * dke;
            },
            Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
				energy_bcs(true);
    }

    #include "./instantiate.h"
}
