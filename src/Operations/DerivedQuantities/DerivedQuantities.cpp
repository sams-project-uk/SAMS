
#include "DerivedQuantities.h"
#include "io/writers.h"
#include "io/derivedQuantities.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace SAMS
{

    /**
     * Register the derived quantities as outputs. They are only written when a schedule asks for them, and are not needed for a restart
     * @param outputs The registry of outputs
     */
    template<typename T_EOS>
    void derivedQuantities<T_EOS>::registerOutputVariables(outputRegistry &outputs)
    {
        pressureOutput = outputs.registerOutput("pressure", "Gas pressure, at the cell centres", false, false);
        temperatureOutput = outputs.registerOutput("temperature", "Temperature, at the cell centres", false, false);
        magneticFieldOutput = outputs.registerOutput("magnetic_field_magnitude", "Magnitude of the magnetic field, at the cell centres", false, false);
        columnDensityOutput = outputs.registerOutput("column_density", "Integral of the density along z, at the cell centres of x and y", false, false);
    }

    /**
     * Register the meshes of the derived quantities that the dump writes
     * @param regWriter The writer for variables in the registries
     * @param level The outputs that the dump writes
     */
    template<typename T_EOS>
    template<typename T_writer>
    void derivedQuantities<T_EOS>::registerDumpMeshes(IO::writer<T_writer> &, IO::registryWriter<T_writer> &regWriter, const outputLevel &level)
    {
        //A mesh that LARE3D has already registered with the same axes is not registered again
        if (level.includes(pressureOutput) || level.includes(temperatureOutput) || level.includes(magneticFieldOutput)){
            regWriter.registerRectilinearMesh("MeshCC", {{"X", staggerType::CENTRED}, {"Y", staggerType::CENTRED}, {"Z", staggerType::CENTRED}});
        }
        if (level.includes(columnDensityOutput)){
            regWriter.registerRectilinearMesh("MeshCC_XY", {{"X", staggerType::CENTRED}, {"Y", staggerType::CENTRED}});
        }
    }

    /**
     * Register the derived quantities that the dump writes with the writer
     * @param writer The writer of the dump
     * @param level The outputs that the dump writes
     */
    template<typename T_EOS>
    template<typename T_writer>
    void derivedQuantities<T_EOS>::registerDumpVariables(IO::writer<T_writer> &writer, IO::registryWriter<T_writer> &, const outputLevel &level)
    {
        if (level.includes(pressureOutput)) writer.template registerData<T_dataType>("pressure", "MeshCC");
        if (level.includes(temperatureOutput)) writer.template registerData<T_dataType>("temperature", "MeshCC");
        if (level.includes(magneticFieldOutput)) writer.template registerData<T_dataType>("magnetic_field_magnitude", "MeshCC");
        if (level.includes(columnDensityOutput)) writer.template registerData<T_dataType>("column_density", "MeshCC_XY");
    }

    /**
     * Calculate and write the derived quantities that the dump writes
     * @param writer The writer of the dump
     * @param level The outputs that the dump writes
     * @param h SAMS harness
     * @param data LARE3D simulation data
     */
    template<typename T_EOS>
    template<typename T_writer>
    void derivedQuantities<T_EOS>::writeDumpVariables(IO::writer<T_writer> &writer, IO::registryWriter<T_writer> &, const outputLevel &level, harness &h, typename LARE::LARE3D<T_EOS>::simulationData &data)
    {
        const IO::simulationAxis x{"X", staggerType::CENTRED}, y{"Y", staggerType::CENTRED}, z{"Z", staggerType::CENTRED};
        //The arrays that the calculations use, so that the calculators capture them and not the whole of the simulation data
        const T_dataType gamma = data.gas_gamma;
        auto rho = data.rho;
        auto energyElectron = data.energy_electron;
        auto energyIon = data.energy_ion;
        auto bx = data.bx;
        auto by = data.by;
        auto bz = data.bz;

        if (level.includes(pressureOutput) || level.includes(temperatureOutput) || level.includes(magneticFieldOutput)){
            IO::derivedQuantity<T_dataType, 3> quantity(h);
            const IO::decomposition decomp = writer.makeDecomposition(x, y, z);
            //Pressure and temperature are hard coded for an ideal gas in normalised units until they come from the EOS
            if (level.includes(pressureOutput)){
                quantity({x, y, z}, LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz){
                    return (gamma - 1.0) * rho(ix, iy, iz) * (energyElectron(ix, iy, iz) + energyIon(ix, iy, iz));
                });
                writer.writeData("pressure", "MeshCC", quantity.data(), decomp);
            }
            if (level.includes(temperatureOutput)){
                quantity({x, y, z}, LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz){
                    return (gamma - 1.0) * (energyElectron(ix, iy, iz) + energyIon(ix, iy, iz));
                });
                writer.writeData("temperature", "MeshCC", quantity.data(), decomp);
            }
            //The field is on the faces, so each component is averaged to the cell centre
            if (level.includes(magneticFieldOutput)){
                quantity({x, y, z}, LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz){
                    const T_dataType bxc = 0.5 * (bx(ix - 1, iy, iz) + bx(ix, iy, iz));
                    const T_dataType byc = 0.5 * (by(ix, iy - 1, iz) + by(ix, iy, iz));
                    const T_dataType bzc = 0.5 * (bz(ix, iy, iz - 1) + bz(ix, iy, iz));
                    return std::sqrt(bxc * bxc + byc * byc + bzc * bzc);
                });
                writer.writeData("magnetic_field_magnitude", "MeshCC", quantity.data(), decomp);
            }
        }

        if (level.includes(columnDensityOutput)){
            //The integral along the part of each column of cells that this rank holds
            IO::derivedQuantity<T_dataType, 2> column(h);
            auto dz = data.dzb;
            const portableWrapper::Range zRange = h.axisRegistry.getLocalDomainRange("Z", staggerType::CENTRED);
            const T_indexType zLower = zRange.lower_bound, zUpper = zRange.upper_bound;
            column({x, y}, LAMBDA(T_indexType ix, T_indexType iy){
                T_dataType sum = 0.0;
                for (T_indexType iz = zLower; iz <= zUpper; ++iz) sum += rho(ix, iy, iz) * dz(iz);
                return sum;
            });
            //Only the ranks that are first in z write, because the output has no z axis
            const IO::decomposition decomp = writer.makeDecomposition(x, y);
            std::vector<T_dataType> sums(column.data(), column.data() + decomp.counts[0] * decomp.counts[1]);
#ifdef USE_MPI
            //The parts of each column are summed onto the rank that is first in z, which is rank 0 of the communicator along z
            SAMSMPI::comm zComm = h.MPIManager.getSubComm({false, false, true});
            int zRank;
            if (MPI_Comm_rank(zComm, &zRank) != MPI_SUCCESS) {
                throw std::runtime_error("Could not get the rank along z for the column density");
            }
            if (MPI_Reduce(zRank == 0 ? MPI_IN_PLACE : sums.data(), sums.data(), static_cast<int>(sums.size()), SAMS::gettypeRegistry().getMPIType<T_dataType>(), MPI_SUM, 0, zComm) != MPI_SUCCESS) {
                throw std::runtime_error("Could not sum the column density along z");
            }
#endif
            writer.writeData("column_density", "MeshCC_XY", sums.data(), decomp);
        }
    }

    #define EOS_DEF(value) template class derivedQuantities<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

    #define WRITER_DEF(eos, T_writer) \
        template void derivedQuantities<eos>::registerDumpMeshes<T_writer>(IO::writer<T_writer> &, IO::registryWriter<T_writer> &, const outputLevel &); \
        template void derivedQuantities<eos>::registerDumpVariables<T_writer>(IO::writer<T_writer> &, IO::registryWriter<T_writer> &, const outputLevel &); \
        template void derivedQuantities<eos>::writeDumpVariables<T_writer>(IO::writer<T_writer> &, IO::registryWriter<T_writer> &, const outputLevel &, harness &, typename LARE::LARE3D<eos>::simulationData &);
    #define EOS_DEF(eos) WRITER_TYPES(eos)
    EOS_DENSITY_ENERGY
    #undef EOS_DEF
    #undef WRITER_DEF

}
