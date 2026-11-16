
#include "OutputControl.h"
#include "io/writers.h"
#include "io/readers.h"

namespace SAMS
{

    /**
     * Read the schedule blocks from the diagnostics block of the deck
     * @param kvs The key value store
     */
    void outputControl::controlVariables(KVS &kvs)
    {
        KVS::Element diagnostics = kvs.getPath("/diagnostics", KVS::Element());
        if (!diagnostics.isType(KVS::Type::Ordered)) return;
        std::vector<std::reference_wrapper<KVS::Element>> blocks = diagnostics.get<KVS::Ordered&>().getAllByKey("schedule");
        int unnamed = 0;
        for (KVS::Element &block : blocks){
            schedule current;
            if (!block.hasElement("name")) ++unnamed;
            current.name = block.getPath("name", std::string("diagnostics")).asString();
            KVS::Element collections = block.getPath("collections");
            for (size_t i = 0; i < collections.size(); i++){
                current.collections.push_back(collections[i].asString());
            }
            current.dt_snapshots = block.getPath("dt_snapshots").asReal();
            current.ndigits = static_cast<int>(block.getPath("ndigits", 5).asInt());
            current.format = block.getPath("format", std::string()).asString();
            if (block.hasElement("disambiguationID")) current.disambiguationID = block.getPath("disambiguationID").asInt();
            schedules.push_back(current);
        }
        if (unnamed > 1) kvs.reportFatalError("Only one schedule block can leave out name");

        //Schedules with the same name are told apart by their disambiguationID
        for (size_t i = 0; i < schedules.size(); i++){
            for (size_t j = 0; j < schedules.size(); j++){
                if (i != j && schedules[i].name == schedules[j].name) schedules[i].shared = true;
            }
        }
        for (size_t i = 0; i < schedules.size(); i++){
            if (!schedules[i].shared || schedules[i].disambiguationID != 0) continue;
            //The smallest ID that no other schedule with the same name has
            int64_t id = 1;
            for (bool used = true; used; ){
                used = false;
                for (const schedule &other : schedules){
                    if (other.name == schedules[i].name && other.disambiguationID == id){
                        used = true;
                        ++id;
                    }
                }
            }
            schedules[i].disambiguationID = id;
            blocks[i].get().setContent("disambiguationID", id);
        }
        for (size_t i = 0; i < schedules.size(); i++){
            for (size_t j = i + 1; j < schedules.size(); j++){
                if (schedules[i].name == schedules[j].name && schedules[i].disambiguationID == schedules[j].disambiguationID){
                    kvs.reportFatalError("Schedule blocks with the name " + schedules[i].name + " have the same disambiguationID " + std::to_string(schedules[i].disambiguationID));
                }
            }
        }
    }

    std::string outputControl::restartKey(const schedule &current)
    {
        if (!current.shared) return "OutputControl/" + current.name + "/next_output";
        return "OutputControl/" + current.name + "/" + std::to_string(current.disambiguationID) + "/next_output";
    }

    /**
     * Activate the collections of each schedule whose next output time has been reached
     * @param request The collections of outputs
     * @param tData SAMS time state data
     */
    void outputControl::queryOutput(outputRequest &request, timeState &tData)
    {
        for (schedule &current : schedules){
            if (tData.time < current.nextOutputTime) continue;
            const std::string dump = current.name + "{_" + std::string(current.ndigits, 'N') + "}";
            for (const std::string &collection : current.collections){
                request.activate(collection, dump);
            }
            if (!current.format.empty()) request.setWriter(dump, current.format);
            current.nextOutputTime += current.dt_snapshots;
        }
    }

    /**
     * Write the time of the next output of each schedule to restart dumps
     * @param writer The writer of the dump
     * @param level The outputs that the dump writes
     */
    template<typename T_writer>
    void outputControl::writeDumpVariables(IO::writer<T_writer> &writer, IO::registryWriter<T_writer> &, const outputLevel &level)
    {
        if (!level.isActive("restart")) return;
        for (const schedule &current : schedules){
            writer.template setKeyValue<SAMS::T_dataType>(restartKey(current).c_str(), current.nextOutputTime);
        }
    }

    /**
     * Read the time of the next output of each schedule from a restart dump. A schedule that is not in the dump keeps its next output time
     * @param reader The reader of the dump
     */
    template<typename T_reader>
    void outputControl::restart(IO::reader<T_reader> &reader, IO::registryReader<T_reader> &)
    {
        for (schedule &current : schedules){
            const std::string key = restartKey(current);
            if (reader.hasKeyValue(key.c_str())) current.nextOutputTime = reader.template readKeyValue<SAMS::T_dataType>(key.c_str());
        }
    }

    #define WRITER_DEF(param, T_writer) \
        template void outputControl::writeDumpVariables<T_writer>(IO::writer<T_writer> &, IO::registryWriter<T_writer> &, const outputLevel &);
    WRITER_TYPES(0)
    #undef WRITER_DEF

    #define READER_DEF(param, T_reader) \
        template void outputControl::restart<T_reader>(IO::reader<T_reader> &, IO::registryReader<T_reader> &);
    READER_TYPES(0)
    #undef READER_DEF

}
