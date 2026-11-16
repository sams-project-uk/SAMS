
#include "SimulationTimeTerminate.h"
#include <sstream>

namespace SAMS
{

    void simulationTimeTerminate::controlVariables(KVS &kvs, controlFunctions &control)
    {
        KVS::Element end = kvs.getPath(deckKeys::endTime, KVS::Element());
        stepLimit = static_cast<SIGNED_INDEX_TYPE>(kvs.getInt(deckKeys::stepLimit, -1));
        //Deactivating frees the package, so nothing is done after it
        if (end.isType(KVS::Type::None) && stepLimit < 0){
            control.deactivatePackage(std::string(name));
            return;
        }
        if (!end.isType(KVS::Type::None)) t_end = end.asReal();
    }

    void simulationTimeTerminate::queryTerminate(terminationRequest &request, timeState &tData)
    {
        if (stepLimit >= 0 && static_cast<SIGNED_INDEX_TYPE>(tData.step) >= stepLimit){
            std::ostringstream reason;
            reason << "The step limit of " << stepLimit << " has been reached";
            request.terminate(reason.str());
        }
        if (tData.time >= t_end){
            std::ostringstream reason;
            reason << "The end time of " << t_end << " has been reached";
            request.terminate(reason.str());
        }
    }

}
