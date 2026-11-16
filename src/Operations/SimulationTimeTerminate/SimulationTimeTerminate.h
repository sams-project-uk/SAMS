#ifndef SIMULATIONTIMETERMINATE_H
#define SIMULATIONTIMETERMINATE_H

#include "harness.h"
#include "runner.h"
#include <limits>

namespace SAMS
{

    /**
     * Ends the run once the simulation time reaches /time/t_end or the step count reaches /time/nsteps. It deactivates
     * itself if neither is set
     */
    class simulationTimeTerminate
    {
    private:
        //The simulation time at which the run ends
        T_dataType t_end = std::numeric_limits<T_dataType>::max();
        //The step at which the run ends. Negative for no limit
        SIGNED_INDEX_TYPE stepLimit = -1;

    public:
        constexpr static std::string_view name = packages::simulationTimeTerminate;

        inline static constexpr auto description = SAMS::makeName("Ends the run once the simulation time reaches /time/t_end or the step count reaches /time/nsteps");

        constexpr static bool timeSimulation = false;

        constexpr static bool defaultActivate = true;

        void controlVariables(KVS &kvs, controlFunctions &control);

        void queryTerminate(terminationRequest &request, timeState &tData);
    };

}
#endif //SIMULATIONTIMETERMINATE_H
