#ifndef DERIVEDQUANTITIES_H
#define DERIVEDQUANTITIES_H

#include "harness.h"
#include "runner.h"
#include "shared_data.h"
#include "io/writerProto.h"

namespace SAMS
{

    /**
     * Example of writing quantities that are derived from the variables of LARE3D but are not variables themselves. They are written through the writer, not the registry writer
     * @details Pressure, temperature and the magnitude of the magnetic field are on the cell centres. The column density, the integral of the density along z, is on the cell centres of x and y
     */
    template<typename T_EOS = LARE::idealGas>
    class derivedQuantities
    {
    private:

        inline static constexpr auto nameType = SAMS::makeName("DerivedQuantities") + T_EOS::name;

        outputHandle pressureOutput, temperatureOutput, magneticFieldOutput, columnDensityOutput;

    public:
        /**
         * Name of the package. Must be unique across all packages in the executable.
         */
        constexpr static std::string_view name = nameType;

        /**
         * Derived quantities should not be timed
         */
        constexpr static bool timeSimulation = false;

        /**
         * Register the derived quantities as outputs. They are only written when a schedule asks for them, and are not needed for a restart
         * @param outputs The registry of outputs
         */
        void registerOutputVariables(outputRegistry &outputs);

        /**
         * Register the meshes of the derived quantities that the dump writes
         * @param regWriter The writer for variables in the registries
         * @param level The outputs that the dump writes
         */
        template<typename T_writer>
        void registerDumpMeshes(IO::writer<T_writer> &writer, IO::registryWriter<T_writer> &regWriter, const outputLevel &level);

        /**
         * Register the derived quantities that the dump writes with the writer
         * @param writer The writer of the dump
         * @param level The outputs that the dump writes
         */
        template<typename T_writer>
        void registerDumpVariables(IO::writer<T_writer> &writer, IO::registryWriter<T_writer> &regWriter, const outputLevel &level);

        /**
         * Calculate and write the derived quantities that the dump writes
         * @param writer The writer of the dump
         * @param level The outputs that the dump writes
         * @param h SAMS harness
         * @param data LARE3D simulation data
         */
        template<typename T_writer>
        void writeDumpVariables(IO::writer<T_writer> &writer, IO::registryWriter<T_writer> &regWriter, const outputLevel &level, harness &h, typename LARE::LARE3D<T_EOS>::simulationData &data);
    };

    #define EOS_DEF(value) extern template class derivedQuantities<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}
#endif //DERIVEDQUANTITIES_H
