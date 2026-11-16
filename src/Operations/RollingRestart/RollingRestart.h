#ifndef ROLLINGRESTART_H
#define ROLLINGRESTART_H

#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"
#include "io/readerProto.h"
#include <cstdint>

namespace SAMS
{

    /**
     * Writes restart dumps every dt_snapshots, cycling through a fixed number of files
     */
    class rollingRestart
    {
    private:
        T_dataType dt_snapshots;
        T_dataType nextOutputTime;
        int64_t files;
        //File written next
        int64_t index = 0;

    public:
        constexpr static std::string_view name = "RollingRestart";

        constexpr static bool timeSimulation = false;

        void controlVariables(KVS &kvs);

        void queryOutput(outputRequest &request, timeState &tData);

        template<typename T>
        void writeDumpVariables(IO::writer<T> &writer, IO::registryWriter<T> &regWriter, const outputLevel &level);

        template<typename T>
        void restart(IO::reader<T> &reader, IO::registryReader<T> &regReader, timeState &tData);
    };

}
#endif //ROLLINGRESTART_H
