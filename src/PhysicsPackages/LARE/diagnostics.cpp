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

/**
 * Get the part of the data that should be written to disk
 */
void getHostVersion(simulationData &data, portableWrapper::portableArrayManager &manager, volumeArray &device, hostVolumeArray &host)
{
    using Range = portableWrapper::Range;
    // Copy the data if needed
    auto fullHost = manager.makeHostAvailable(device);
    // Now allocate memory for just the part we want to write
    manager.allocate(host, Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
    // This should DEFINITELY be in the backend somewhere, but for now just copy
    portableWrapper::applyKernelHost(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) { host(ix, iy, iz) = fullHost(ix, iy, iz); }, Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
    // manager.copyData(host, fullHost(Range(1,data.nx), Range(1,data.ny), Range(1,data.nz)));
    manager.deallocate(fullHost);
}

template <typename T_writer>
void writeDiagnosticsCore(std::string Name, simulationData &data, writer<T_writer> &writer)
{
    portableWrapper::portableArrayManager manager;
    hostVolumeArray host;

    writer.openFile(Name.c_str());
    writer.template registerRectilinearMesh<T_dataType>("MeshCC", data.nx, data.ny, data.nz);

    writer.template registerData<T_dataType>("rho", "MeshCC");
    writer.template registerData<T_dataType>("energy_electron", "MeshCC");
    writer.template registerData<T_dataType>("energy_ion", "MeshCC");
    writer.template registerData<T_dataType>("vx", "MeshCC");
    writer.template registerData<T_dataType>("vy", "MeshCC");
    writer.template registerData<T_dataType>("vz", "MeshCC");
    writer.template registerData<T_dataType>("bx", "MeshCC");
    writer.template registerData<T_dataType>("by", "MeshCC");
    writer.template registerData<T_dataType>("bz", "MeshCC");

    writer.writeRectilinearMesh("MeshCC", &data.xc(1), &data.yc(1), &data.zc(1));

    getHostVersion(data, manager, data.rho, host);
    writer.writeData("rho", host.data());

    getHostVersion(data, manager, data.energy_electron, host);
    writer.writeData("energy_electron", host.data());

    getHostVersion(data, manager, data.energy_ion, host);
    writer.writeData("energy_ion", host.data());

    getHostVersion(data, manager, data.vx, host);
    writer.writeData("vx", host.data());

    getHostVersion(data, manager, data.vy, host);
    writer.writeData("vy", host.data());

    getHostVersion(data, manager, data.vz, host);
    writer.writeData("vz", host.data());

    getHostVersion(data, manager, data.bx, host);
    writer.writeData("bx", host.data());

    getHostVersion(data, manager, data.by, host);
    writer.writeData("by", host.data());

    getHostVersion(data, manager, data.bz, host);
    writer.writeData("bz", host.data());

    writer.closeFile();
}

void simulation::output(simulationData &data)
{
#if defined(USE_HDF5)
    HDF5File writer;
#else
    simpleFile writer;
#endif
    std::string Name = "diagnostics_step_" + std::to_string(data.step);
    writeDiagnosticsCore(Name, data, writer);
}

void simulation::energy_correction(simulationData &data)
{
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            T_dataType dke = portableWrapper::max(-data.delta_ke(ix, iy, iz), 0.0) / (data.rho(ix, iy, iz) * data.cv(ix, iy, iz));
            data.energy_electron(ix, iy, iz) += 0.5 * dke;
            data.energy_ion(ix, iy, iz) += 0.5 * dke;
        },
        Range(1, data.nx), Range(1, data.ny), Range(1, data.nz));
}