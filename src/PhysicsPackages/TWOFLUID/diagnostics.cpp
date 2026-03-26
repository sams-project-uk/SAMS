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
#include "twofluid.h"
#include "mpiManager.h"

namespace TWOFLUID
{

    namespace pw = portableWrapper;
    /**
     * Get the part of the data that should be written to disk
     */
    void getHostVersion(data_two_fluid_source &data, pw::portableArrayManager &manager, LARE::volumeArray &device, LARE::hostVolumeArray &host)
    {
        using Range = pw::Range;
        // Includes ghost cells, so no need to copy only part
        host = manager.makeHostAvailable(device);
    }

    template<typename T_EOS>
    template<typename T_writer>
        void PIP<T_EOS>::writeOutputMeshes(writer<T_writer> &writer, LARE::LARE3DST<T_EOS>::simulationData &data){
            writer.writeRectilinearMesh("MeshCC_source", data.xc_host.data(), data.yc_host.data(), data.zc_host.data());
            writer.writeRectilinearMesh("MeshBBB_source", data.xb_host.data(), data.yb_host.data(), data.zb_host.data());
    }
    
     template<typename T_EOS>
    template<typename T_writer>
    void PIP<T_EOS>::registerOutputMeshes(writer<T_writer> &writer, LARE::LARE3DST<T_EOS>::simulationData &data)
    {
        T_indexType nx = data.xc_host.getSize(0);
        T_indexType ny = data.yc_host.getSize(0);
        T_indexType nz = data.zc_host.getSize(0);
        writer.template registerRectilinearMesh<LARE::T_dataType>("MeshCC_source", nx, ny, nz);
        T_indexType nxb = data.xb_host.getSize(0);
        T_indexType nyb = data.yb_host.getSize(0);
        T_indexType nzb = data.zb_host.getSize(0);
        writer.template registerRectilinearMesh<LARE::T_dataType>("MeshBBB_source", nxb, nyb, nzb);
    }

    template<typename T_EOS>
    template<typename T_writer>
    void PIP<T_EOS>::registerOutputVariables(writer<T_writer> &writer, data_two_fluid_source &data)
    {
        writer.template registerData<LARE::T_dataType>("source_mass", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_mass_n", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_energy", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_energy_n", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_v_x", "MeshBBB_source");
        writer.template registerData<LARE::T_dataType>("source_v_x_n", "MeshBBB_source");
        writer.template registerData<LARE::T_dataType>("source_v_y", "MeshBBB_source");
        writer.template registerData<LARE::T_dataType>("source_v_y_n", "MeshBBB_source");
        writer.template registerData<LARE::T_dataType>("source_v_z", "MeshBBB_source");
        writer.template registerData<LARE::T_dataType>("source_v_z_n", "MeshBBB_source");
        writer.template registerData<LARE::T_dataType>("ac", "MeshCC_source");
    }

    template<typename T_EOS>
    template <typename T_writer>
    void PIP<T_EOS>::writeOutputVariables(writer<T_writer> &writer, data_two_fluid_source &data)
    {
        pw::portableArrayManager manager;
        LARE::hostVolumeArray host;

        getHostVersion(data, manager, data.ac, host);
        writer.writeData("ac", host.data());
    
        getHostVersion(data, manager, data.source_mass, host);
        writer.writeData("source_mass", host.data());
        
        getHostVersion(data, manager, data.source_mass_n, host);
        writer.writeData("source_mass_n", host.data());
        
        getHostVersion(data, manager, data.source_energy, host);
        writer.writeData("source_energy", host.data());
        
        getHostVersion(data, manager, data.source_energy_n, host);
        writer.writeData("source_energy_n", host.data());
        
        getHostVersion(data, manager, data.source_v_x, host);
        writer.writeData("source_v_x", host.data());
        
        getHostVersion(data, manager, data.source_v_x_n, host);
        writer.writeData("source_v_x_n", host.data());
        
        getHostVersion(data, manager, data.source_v_y, host);
        writer.writeData("source_v_y", host.data());
        
        getHostVersion(data, manager, data.source_v_y_n, host);
        writer.writeData("source_v_y_n", host.data());
        
        getHostVersion(data, manager, data.source_v_z, host);
        writer.writeData("source_v_z", host.data());
        
        getHostVersion(data, manager, data.source_v_z_n, host);
        writer.writeData("source_v_z_n", host.data());

    }

//Need a better solution than this against future additions of writers
//Perhaps another X macro?
#if defined(USE_HDF5)
//Instantiate the templates for HDF5 writer
    template void PIP<idealGas>::registerOutputVariables<HDF5File>(writer<HDF5File> &writer, data_two_fluid_source &data);
    template void PIP<idealGas>::writeOutputVariables<HDF5File>(writer<HDF5File> &writer, data_two_fluid_source &data);
    template void PIP<idealGas>::registerOutputMeshes(writer<HDF5File> &writer, LARE::LARE3DST<idealGas>::simulationData &data);
    template void PIP<idealGas>::writeOutputMeshes(writer<HDF5File> &writer, LARE::LARE3DST<idealGas>::simulationData &data);
#else
//Instantiate the templates for simple writer
    //template void LARE3D::registerOutputVariables<simpleFile>(writer<simpleFile> &writer, simulationData &data);
    //template void LARE3D::writeOutputVariables<simpleFile>(writer<simpleFile> &writer, simulationData &data);
#endif
}
