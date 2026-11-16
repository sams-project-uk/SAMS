
#include "BookendRestart.h"
#include "io/readers.h"

namespace SAMS
{

    void bookendRestart::queryOutput(outputRequest &request)
    {
        if (first || request.isFinal()){
            const std::string dumpName = first ? "restart_initial" : "restart_final";
            request.activate("restart", dumpName);
            request.disableTimeCoalescence(dumpName);
        }
        first = false;
    }

    template<typename T_reader>
    void bookendRestart::restart(IO::reader<T_reader> &, IO::registryReader<T_reader> &)
    {
        //The dump that the run restarts from is its restart at the start
        first = false;
    }

    #define READER_DEF(param, T_reader) \
        template void bookendRestart::restart<T_reader>(IO::reader<T_reader> &, IO::registryReader<T_reader> &);
    READER_TYPES(0)
    #undef READER_DEF

}
