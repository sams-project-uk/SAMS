
#include "RollingRestart.h"
#include "io/writers.h"
#include "io/readers.h"
#include <string>

namespace SAMS
{

    void rollingRestart::controlVariables(KVS &kvs)
    {
        dt_snapshots = kvs.getReal("/RollingRestart/dt_snapshots");
        files = kvs.getInt("/RollingRestart/files", 2);
        //BookendRestart writes a restart at the start
        nextOutputTime = dt_snapshots;
    }

    void rollingRestart::queryOutput(outputRequest &request, timeState &tData)
    {
        if (tData.time < nextOutputTime) return;
        const std::string name = "restart_rolling_" + std::to_string(index);
        request.activate("restart", name);
        request.disableTimeCoalescence(name);
        index = (index + 1) % files;
        nextOutputTime += dt_snapshots;
    }

    template<typename T_writer>
    void rollingRestart::writeDumpVariables(IO::writer<T_writer> &writer, IO::registryWriter<T_writer> &, const outputLevel &level)
    {
        if (!level.isActive("restart")) return;
        writer.template setKeyValue<SAMS::T_dataType>("RollingRestart/next_output", nextOutputTime);
        writer.template setKeyValue<int64_t>("RollingRestart/index", index);
    }

    template<typename T_reader>
    void rollingRestart::restart(IO::reader<T_reader> &reader, IO::registryReader<T_reader> &, timeState &tData)
    {
        //A dump written without RollingRestart is treated as its last write
        if (reader.hasKeyValue("RollingRestart/next_output")) nextOutputTime = reader.template readKeyValue<SAMS::T_dataType>("RollingRestart/next_output");
        else nextOutputTime = tData.time + dt_snapshots;
        if (reader.hasKeyValue("RollingRestart/index")) index = reader.template readKeyValue<int64_t>("RollingRestart/index");
    }

    #define WRITER_DEF(param, T_writer) \
        template void rollingRestart::writeDumpVariables<T_writer>(IO::writer<T_writer> &, IO::registryWriter<T_writer> &, const outputLevel &);
    WRITER_TYPES(0)
    #undef WRITER_DEF

    #define READER_DEF(param, T_reader) \
        template void rollingRestart::restart<T_reader>(IO::reader<T_reader> &, IO::registryReader<T_reader> &, timeState &);
    READER_TYPES(0)
    #undef READER_DEF

}
