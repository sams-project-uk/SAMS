#ifndef WALLTIMETERMINATE_H
#define WALLTIMETERMINATE_H

#include "harness.h"
#include "runner.h"
#include <chrono>

namespace SAMS
{

    /**
     * Ends the run once the wall time since the package was initialised exceeds the time given by /time/walltime_end, in HH:MM:SS
     */
    class walltimeTerminate
    {
    private:
        //The time when the package was initialised
        std::chrono::steady_clock::time_point start;
        //The wall time after which the run ends
        std::chrono::seconds limit{0};
        //The wall time as it is given in the deck
        std::string limitText;

    public:
        constexpr static std::string_view name = "WalltimeTerminate";

        inline static constexpr auto description = SAMS::makeName("Ends the run once the wall time given by /time/walltime_end, in HH:MM:SS, has passed");

        constexpr static bool timeSimulation = false;

        void initialize();

        void controlVariables(KVS &kvs, harness &h);

        void queryTerminate(terminationRequest &request, harness &h);
    };

}
#endif //WALLTIMETERMINATE_H
