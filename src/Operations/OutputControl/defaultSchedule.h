#ifndef DEFAULTSCHEDULE_H
#define DEFAULTSCHEDULE_H

#include "harness.h"
#include "KVS/kvs.h"

namespace SAMS
{

    /**
     * Add a schedule block called diagnostics, which writes the default collection every dt_snapshots, if the diagnostics block of the deck has no schedule blocks
     * @details The runner sets /diagnostics/format before defaultDeck, so the diagnostics block exists
     * @param kvs The key value store
     * @param dt_snapshots The time between outputs
     */
    inline void addDefaultSchedule(KVS &kvs, T_dataType dt_snapshots)
    {
        KVS::Ordered &diagnostics = kvs.getPath("/diagnostics").get<KVS::Ordered&>();
        if (!diagnostics.getAllByKey("schedule").empty()) return;
        diagnostics.push_back("schedule", KVS::Ordered{{"name", "diagnostics"},{"collections", KVS::Array{"default"}}, {"dt_snapshots", dt_snapshots}}).setMetadata("autoset", "true");
    }

}
#endif //DEFAULTSCHEDULE_H
