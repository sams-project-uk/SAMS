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
        // Copy the data if needed
        auto fullHost = manager.makeHostAvailable(device);

        LARE::T_indexType nx=data.source_mass.getSize(0);
        LARE::T_indexType ny=data.source_mass.getSize(1);
        LARE::T_indexType nz=data.source_mass.getSize(2);
        //printf("%li \n",nx);
        // Now allocate memory for just the part we want to write
        manager.allocate(host, Range(1, nx), Range(1, ny), Range(1, nz));
        //Have to copy data since the HDF writer expects contiguous data
        manager.copyDataHost(host, fullHost(Range(1,nx), Range(1,ny), Range(1,nz)));
        manager.deallocate(fullHost);
    }

    template<typename T_writer>
        void PIP::writeOutputMeshes(writer<T_writer> &writer, LARE::simulationData &data){
            writer.writeRectilinearMesh("MeshCC_source", &data.xc_host(1), &data.yc_host(1), &data.zc_host(1));
    }
    
    template<typename T_writer>
    void PIP::registerOutputMeshes(writer<T_writer> &writer, data_two_fluid_source &data)
    {
        LARE::T_indexType nx=data.source_mass.getSize(0);
        LARE::T_indexType ny=data.source_mass.getSize(1);
        LARE::T_indexType nz=data.source_mass.getSize(2);
        writer.template registerRectilinearMesh<LARE::T_dataType>("MeshCC_source", nx, ny, nz);

    }

    template<typename T_writer>
    void PIP::registerOutputVariables(writer<T_writer> &writer, data_two_fluid_source &data)
    {
        writer.template registerData<LARE::T_dataType>("source_mass", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_mass_n", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_energy", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_energy_n", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_v_x", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_v_x_n", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_v_y", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_v_y_n", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_v_z", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("source_v_z_n", "MeshCC_source");
        writer.template registerData<LARE::T_dataType>("ac", "MeshCC_source");
    }

    template <typename T_writer>
    void PIP::writeOutputVariables(writer<T_writer> &writer, data_two_fluid_source &data)
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
    template void PIP::registerOutputVariables<HDF5File>(writer<HDF5File> &writer, data_two_fluid_source &data);
    template void PIP::writeOutputVariables<HDF5File>(writer<HDF5File> &writer, data_two_fluid_source &data);
    template void PIP::registerOutputMeshes(writer<HDF5File> &writer, data_two_fluid_source &data);
    template void PIP::writeOutputMeshes(writer<HDF5File> &writer, LARE::simulationData &data);
#else
//Instantiate the templates for simple writer
    //template void LARE3D::registerOutputVariables<simpleFile>(writer<simpleFile> &writer, simulationData &data);
    //template void LARE3D::writeOutputVariables<simpleFile>(writer<simpleFile> &writer, simulationData &data);
#endif
}
