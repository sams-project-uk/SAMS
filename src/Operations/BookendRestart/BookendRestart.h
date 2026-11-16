#ifndef BOOKENDRESTART_H
#define BOOKENDRESTART_H

#include "harness.h"
#include "runner.h"
#include "io/readerProto.h"

namespace SAMS
{

    /**
     * Writes a restart dump at the start and the end of the run. A restarted run does not write one at the start
     */
    class bookendRestart
    {
    private:
        //Whether the next request is the one at the start of the run
        bool first = true;

    public:
        constexpr static std::string_view name = "GenerateBookendRestart";

        inline static constexpr auto description = SAMS::makeName("Writes a restart dump at the start and the end of the run. A restarted run does not write one at the start");

        constexpr static bool timeSimulation = false;

        constexpr static bool defaultActivate = true;

        void queryOutput(outputRequest &request);

        template<typename T>
        void restart(IO::reader<T> &reader, IO::registryReader<T> &regReader);
    };

}
#endif //BOOKENDRESTART_H
