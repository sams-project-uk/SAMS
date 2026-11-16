#ifndef OUTPUTCONTROL_H
#define OUTPUTCONTROL_H

#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"
#include "io/readerProto.h"
#include <string>
#include <vector>

namespace SAMS
{

    /**
     * Decides when collections of outputs are written, from the schedule blocks in the diagnostics block of the deck
     * @details Each schedule block has a name, the collections that it activates and the time between its outputs, dt_snapshots.
     * A schedule first activates its collections at time 0
     */
    class outputControl
    {
    private:

        /**
         * One schedule block of the deck
         */
        struct schedule
        {
            //The name of the schedule, which is also the name of its dumps before the dump number
            std::string name;
            //Whether another schedule has the same name, in which case disambiguationID tells them apart
            bool shared = false;
            //Tells apart schedules with the same name. 0 if it is not set
            int64_t disambiguationID = 0;
            std::vector<std::string> collections;
            T_dataType dt_snapshots;
            //The number of digits of the dump number
            int ndigits;
            //The output format of the dumps. Empty for the format of the diagnostics block
            std::string format;
            //The time of the next output
            T_dataType nextOutputTime = 0.0;
        };

        std::vector<schedule> schedules;

        /**
         * The key of the time of the next output of a schedule in restart dumps
         * @param current The schedule
         */
        static std::string restartKey(const schedule &current);

    public:
        /**
         * Name of the package. Must be unique across all packages in the executable.
         */
        constexpr static std::string_view name = packages::outputControl;

        /**
         * Output control should not be timed
         */
        constexpr static bool timeSimulation = false;

        /**
         * Active unless the deck deactivates it with ~OutputControl
         */
        constexpr static bool defaultActivate = true;

        /**
         * Read the schedule blocks from the diagnostics block of the deck
         * @param kvs The key value store
         */
        void controlVariables(KVS &kvs);

        /**
         * Activate the collections of each schedule whose next output time has been reached
         * @param request The collections of outputs
         * @param tData SAMS time state data
         */
        void queryOutput(outputRequest &request, timeState &tData);

        /**
         * Write the time of the next output of each schedule to restart dumps
         * @param writer The writer of the dump
         * @param regWriter The writer for variables in the registries
         * @param level The outputs that the dump writes
         */
        template<typename T>
        void writeDumpVariables(IO::writer<T> &writer, IO::registryWriter<T> &regWriter, const outputLevel &level);

        /**
         * Read the time of the next output of each schedule from a restart dump. A schedule that is not in the dump keeps its next output time
         * @param reader The reader of the dump
         * @param regReader The reader for variables in the registries
         */
        template<typename T>
        void restart(IO::reader<T> &reader, IO::registryReader<T> &regReader);
    };

}
#endif //OUTPUTCONTROL_H
