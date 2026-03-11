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
#include "shared_data_neutral.h"
#include "mpiManager.h"

namespace LARE_neutral
{

    namespace pw = portableWrapper;
    /**
     * Get the part of the data that should be written to disk
     */
    void getHostVersion(simulationData &data, pw::portableArrayManager &manager, volumeArray &device, hostVolumeArray &host)
    {
        using Range = pw::Range;
        // Copy the data if needed
        auto fullHost = manager.makeHostAvailable(device);

        // Now allocate memory for just the part we want to write
        manager.allocate(host, Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
        //Have to copy data since the HDF writer expects contiguous data
        manager.copyDataHost(host, fullHost(Range(1,data.nx), Range(1,data.ny), Range(1,data.nz)));
        manager.deallocate(fullHost);
    }

    template<typename T_writer>
    void LARE3D_neutral::registerOutputMeshes(writer<T_writer> &writer, simulationData &data)
    {
        writer.template registerRectilinearMesh<T_dataType>("MeshCC_n", data.nx, data.ny, data.nz);

    }

    template<typename T_writer>
    void LARE3D_neutral::registerOutputVariables(writer<T_writer> &writer, simulationData &)
    {
        writer.template registerData<T_dataType>("rho_n", "MeshCC_n");
        writer.template registerData<T_dataType>("energy_neutral", "MeshCC_n");
        writer.template registerData<T_dataType>("vx_n", "MeshCC_n");
        writer.template registerData<T_dataType>("vy_n", "MeshCC_n");
        writer.template registerData<T_dataType>("vz_n", "MeshCC_n");
        writer.template registerData<T_dataType>("bx_n", "MeshCC_n");
        writer.template registerData<T_dataType>("by_n", "MeshCC_n");
        writer.template registerData<T_dataType>("bz_n", "MeshCC_n");
    }

    template<typename T_writer>
    void LARE3D_neutral::writeOutputMeshes(writer<T_writer> &writer, simulationData &data){
        writer.writeRectilinearMesh("MeshCC_n", &data.xc_host(1), &data.yc_host(1), &data.zc_host(1));
    }

    template <typename T_writer>
    void LARE3D_neutral::writeOutputVariables(writer<T_writer> &writer, simulationData &data)
    {
        pw::portableArrayManager manager;
        hostVolumeArray host;

        getHostVersion(data, manager, data.rho, host);
        writer.writeData("rho_n", host.data());

        getHostVersion(data, manager, data.energy_neutral, host);
        writer.writeData("energy_neutral", host.data());

        getHostVersion(data, manager, data.vx, host);
        writer.writeData("vx_n", host.data());

        getHostVersion(data, manager, data.vy, host);
        writer.writeData("vy_n", host.data());

        getHostVersion(data, manager, data.vz, host);
        writer.writeData("vz_n", host.data());

        getHostVersion(data, manager, data.bx, host);
        writer.writeData("bx_n", host.data());

        getHostVersion(data, manager, data.by, host);
        writer.writeData("by_n", host.data());

        getHostVersion(data, manager, data.bz, host);
        writer.writeData("bz_n", host.data());
    }

//Need a better solution than this against future additions of writers
//Perhaps another X macro?
#if defined(USE_HDF5)
//Instantiate the templates for HDF5 writer
    template void LARE3D_neutral::registerOutputMeshes<HDF5File>(writer<HDF5File> &writer, simulationData &data);
    template void LARE3D_neutral::registerOutputVariables<HDF5File>(writer<HDF5File> &writer, simulationData &data);
    template void LARE3D_neutral::writeOutputMeshes<HDF5File>(writer<HDF5File> &writer, simulationData &data);
    template void LARE3D_neutral::writeOutputVariables<HDF5File>(writer<HDF5File> &writer, simulationData &data);
#else
//Instantiate the templates for simple writer
    template void LARE3D_neutral::registerOutputMeshes<simpleFile>(writer<simpleFile> &writer, simulationData &data);
    template void LARE3D_neutral::registerOutputVariables<simpleFile>(writer<simpleFile> &writer, simulationData &data);
    template void LARE3D_neutral::writeOutputMeshes<simpleFile>(writer<simpleFile> &writer, simulationData &data);
    template void LARE3D_neutral::writeOutputVariables<simpleFile>(writer<simpleFile> &writer, simulationData &data);
#endif

    void LARE3D_neutral::energy_correction(simulationData &data)
    {
        using Range = pw::Range;
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_dataType dke = pw::max(-data.delta_ke(ix, iy, iz), 0.0) / (data.rho(ix, iy, iz) * data.cv(ix, iy, iz));
                data.energy_neutral(ix, iy, iz) += 0.5 * dke;
            },
            Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
    }
}
