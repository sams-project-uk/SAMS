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
#include "LARE3DNeutralFluid/shared_data.h"
#include "mpiManager.h"

namespace LARE
{

    namespace pw = portableWrapper;
    /**
     * Get the part of the data that should be written to disk
     */
    template<typename T_EOS>
    inline void LARE3DNF<T_EOS>::getHostVersion(simulationData &data, const domainData & core_data, pw::portableArrayManager &manager, volumeArray &device, hostVolumeArray &host)
    {
        using Range = pw::Range;
        // Copy the data if needed
        auto fullHost = manager.makeHostAvailable(device);

        // Now allocate memory for just the part we want to write
        manager.allocate(host, Range(1, core_data.nx), Range(1, core_data.ny), Range(1, core_data.nz));
        //Have to copy data since the HDF writer expects contiguous data
        manager.copyDataHost(host, fullHost(Range(1,core_data.nx), Range(1,core_data.ny), Range(1,core_data.nz)));
        manager.deallocate(fullHost);
    }

    
    template<typename T_EOS>
    template<typename T_writer>
    void LARE3DNF<T_EOS>::registerOutputMeshes([[maybe_unused]] writer<T_writer> &writer, [[maybe_unused]] simulationData &data, [[maybe_unused]] const domainData & core_data){;}//No additional meshes

    template<typename T_EOS>
    template<typename T_writer>
    void LARE3DNF<T_EOS>::registerOutputVariables(writer<T_writer> &writer, simulationData &, const domainData & core_data)
    {
        writer.template registerData<T_dataType>("rho_n", "MeshCC");
        writer.template registerData<T_dataType>("energy_n", "MeshCC");
        writer.template registerData<T_dataType>("vx_n", "MeshCC");
        writer.template registerData<T_dataType>("vy_n", "MeshCC");
        writer.template registerData<T_dataType>("vz_n", "MeshCC");
    }

    template<typename T_EOS>
    template<typename T_writer>
    void LARE3DNF<T_EOS>::writeOutputMeshes( [[maybe_unused]] writer<T_writer> &writer, [[maybe_unused]] simulationData &data, const domainData & core_data){;}

    template<typename T_EOS>
    template<typename T_writer>
    void LARE3DNF<T_EOS>::writeOutputVariables(writer<T_writer> &writer, simulationData &data, const domainData & core_data)
    {
        pw::portableArrayManager manager;
        hostVolumeArray host;

        getHostVersion(data, core_data, manager, data.rho, host);
        writer.writeData("rho_n", host.data());

        getHostVersion(data, core_data, manager, data.energy, host);
        writer.writeData("energy_n", host.data());

        getHostVersion(data, core_data, manager, data.vx, host);
        writer.writeData("vx_n", host.data());

        getHostVersion(data, core_data, manager, data.vy, host);
        writer.writeData("vy_n", host.data());

        getHostVersion(data, core_data, manager, data.vz, host);
        writer.writeData("vz_n", host.data());

    }

//Need a better solution than this against future additions of writers
//Perhaps another X macro?
#if defined(USE_HDF5)
//Instantiate the templates for HDF5 writer
    template void LARE3DNF<LARE::idealGas>::registerOutputMeshes<HDF5File>(writer<HDF5File> &writer, simulationData &data, const domainData & core_data);
    template void LARE3DNF<LARE::idealGas>::registerOutputVariables<HDF5File>(writer<HDF5File> &writer, simulationData &data, const domainData & core_data);
    template void LARE3DNF<LARE::idealGas>::writeOutputMeshes<HDF5File>(writer<HDF5File> &writer, simulationData &data, const domainData & core_data);
    template void LARE3DNF<LARE::idealGas>::writeOutputVariables<HDF5File>(writer<HDF5File> &writer, simulationData &data, const domainData & core_data);
#else
//Instantiate the templates for simple writer
    template void LARE3DNF<LARE::idealGas>::registerOutputMeshes<simpleFile>(writer<simpleFile> &writer, simulationData &data, const domainData & core_data);
    template void LARE3DNF<LARE::idealGas>::registerOutputVariables<simpleFile>(writer<simpleFile> &writer, simulationData &data, const domainData & core_data);
    template void LARE3DNF<LARE::idealGas>::writeOutputMeshes<simpleFile>(writer<simpleFile> &writer, simulationData &data, const domainData & core_data);
    template void LARE3DNF<LARE::idealGas>::writeOutputVariables<simpleFile>(writer<simpleFile> &writer, simulationData &data, const domainData & core_data);
#endif

    template<typename T_EOS>
    void LARE3DNF<T_EOS>::energy_correction(simulationData &data, const domainData & core_data)
    {
        using Range = pw::Range;
        pw::applyKernel(
            LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                T_dataType dke = pw::max(-data.delta_ke(ix, iy, iz), 0.0) / (data.rho(ix, iy, iz) * core_data.cv(ix, iy, iz));
                data.energy(ix, iy, iz) += dke;
            },
            Range(1, core_data.nx), Range(1, core_data.ny), Range(1, core_data.nz));
    }
}