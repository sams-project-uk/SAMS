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

#if defined(USE_HDF5)
#include "io/writerHDF5.h"
#else
#include "io/writerSimple.h"
#endif
#include "shared_data.h"
#include "mpiManager.h"
#include "shared_data.h"

namespace LARE
{

    namespace pw = portableWrapper;

    template<typename T_EOS>
    template<typename T_writer>
    void LARE3D<T_EOS>::registerOutputMeshes(writer<T_writer> &writer)
    {
        using SAMS::staggerType;
        // One mesh for each combination of staggers of the output variables
        writer.registerRectilinearMesh("MeshCC", {"X", staggerType::CENTRED}, {"Y", staggerType::CENTRED}, {"Z", staggerType::CENTRED});
        writer.registerRectilinearMesh("MeshNode", {"X", staggerType::HALF_CELL}, {"Y", staggerType::HALF_CELL}, {"Z", staggerType::HALF_CELL});
        writer.registerRectilinearMesh("MeshFaceX", {"X", staggerType::HALF_CELL}, {"Y", staggerType::CENTRED}, {"Z", staggerType::CENTRED});
        writer.registerRectilinearMesh("MeshFaceY", {"X", staggerType::CENTRED}, {"Y", staggerType::HALF_CELL}, {"Z", staggerType::CENTRED});
        writer.registerRectilinearMesh("MeshFaceZ", {"X", staggerType::CENTRED}, {"Y", staggerType::CENTRED}, {"Z", staggerType::HALF_CELL});
    }

    template<typename T_EOS>
    template<typename T_writer>
    void LARE3D<T_EOS>::registerOutputVariables(writer<T_writer> &writer, simulationData &)
    {
        writer.registerData("rho", "MeshCC", "rho");
        writer.registerData("energy_electron", "MeshCC", "energy_electron");
        writer.registerData("energy_ion", "MeshCC", "energy_ion");
        writer.registerData("vx", "MeshNode", "vx");
        writer.registerData("vy", "MeshNode", "vy");
        writer.registerData("vz", "MeshNode", "vz");
        writer.registerData("bx", "MeshFaceX", "bx");
        writer.registerData("by", "MeshFaceY", "by");
        writer.registerData("bz", "MeshFaceZ", "bz");
    }

    template<typename T_EOS>
    template<typename T_writer>
    void LARE3D<T_EOS>::writeOutputMeshes(writer<T_writer> &writer){
        writer.writeRectilinearMesh("MeshCC");
        writer.writeRectilinearMesh("MeshNode");
        writer.writeRectilinearMesh("MeshFaceX");
        writer.writeRectilinearMesh("MeshFaceY");
        writer.writeRectilinearMesh("MeshFaceZ");
    }

    template<typename T_EOS>
    template<typename T_writer>
    void LARE3D<T_EOS>::writeOutputVariables(writer<T_writer> &writer)
    {
        writer.writeData("rho");
        writer.writeData("energy_electron");
        writer.writeData("energy_ion");
        writer.writeData("vx");
        writer.writeData("vy");
        writer.writeData("vz");
        writer.writeData("bx");
        writer.writeData("by");
        writer.writeData("bz");
    }

//Need a better solution than this against future additions of writers
//Perhaps another X macro?
#if defined(USE_HDF5)
//Instantiate the templates for HDF5 writer
    template void LARE3D<LARE::idealGas>::registerOutputMeshes<HDF5File>(writer<HDF5File> &writer);
    template void LARE3D<LARE::idealGas>::registerOutputVariables<HDF5File>(writer<HDF5File> &writer, simulationData &data);
    template void LARE3D<LARE::idealGas>::writeOutputMeshes<HDF5File>(writer<HDF5File> &writer);
    template void LARE3D<LARE::idealGas>::writeOutputVariables<HDF5File>(writer<HDF5File> &writer);
#else
//Instantiate the templates for simple writer
    template void LARE3D<LARE::idealGas>::registerOutputMeshes<simpleFile>(writer<simpleFile> &writer);
    template void LARE3D<LARE::idealGas>::registerOutputVariables<simpleFile>(writer<simpleFile> &writer, simulationData &data);
    template void LARE3D<LARE::idealGas>::writeOutputMeshes<simpleFile>(writer<simpleFile> &writer);
    template void LARE3D<LARE::idealGas>::writeOutputVariables<simpleFile>(writer<simpleFile> &writer);
#endif

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
    }

    #include "./instantiate.h"
}