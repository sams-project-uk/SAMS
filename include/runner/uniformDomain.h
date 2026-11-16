#ifndef UNIFORMDOMAIN_H
#define UNIFORMDOMAIN_H

#include "harness.h"
#include "runner.h"

namespace SAMS
{

    /**
     * Set a uniform domain in X, Y and Z, with the numbers of cells and the limits from the deck
     * @param harness SAMS harness
     * @param kvs The key value store
     */
    inline void setUniformDomain(SAMS::harness &harness, SAMS::KVS &kvs)
    {
        auto &axisReg = harness.axisRegistry;
        axisReg.setDomain("X", kvs.getInt(deckKeys::nx), kvs.getReal(deckKeys::xMin), kvs.getReal(deckKeys::xMax));
        axisReg.setDomain("Y", kvs.getInt(deckKeys::ny), kvs.getReal(deckKeys::yMin), kvs.getReal(deckKeys::yMax));
        axisReg.setDomain("Z", kvs.getInt(deckKeys::nz), kvs.getReal(deckKeys::zMin), kvs.getReal(deckKeys::zMax));
    }

}
#endif //UNIFORMDOMAIN_H
