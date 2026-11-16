
#include "WalltimeTerminate.h"
#include <string>

namespace SAMS
{

    void walltimeTerminate::initialize()
    {
        start = std::chrono::steady_clock::now();
    }

    void walltimeTerminate::controlVariables(KVS &kvs, harness &h)
    {
        KVS::Element walltime = kvs.getPath("/time/walltime_end", KVS::Element());
        if (walltime.isType(KVS::Type::None)) h.abort("WalltimeTerminate needs /time/walltime_end", false);
        //HH:MM:SS, with as many digits of hours as wanted
        const std::string text = walltime.asString();
        const size_t first = text.find(':');
        const size_t second = text.find(':', first + 1);
        if (first == std::string::npos || first == 0 || second == std::string::npos || second - first != 3 || text.size() - second != 3 || text.find_first_not_of("0123456789:") != std::string::npos){
            h.abort("Invalid /time/walltime_end: " + text + " is not in the format HH:MM:SS", false);
        }
        limit = std::chrono::hours(std::stoll(text.substr(0, first))) + std::chrono::minutes(std::stoll(text.substr(first + 1, 2))) + std::chrono::seconds(std::stoll(text.substr(second + 1, 2)));
        limitText = text;
    }

    void walltimeTerminate::queryTerminate(terminationRequest &request, [[maybe_unused]] harness &h)
    {
        int ended = std::chrono::steady_clock::now() - start > limit ? 1 : 0;
#ifdef USE_MPI
        //Rank 0 decides, so that every rank ends on the same step
        MPI_Bcast(&ended, 1, MPI_INT, 0, h.MPIManager.getComm());
#endif
        if (ended) request.terminate("The wall time of " + limitText + " has passed");
    }

}
