#ifndef LARE_BOUNDARY_CLASS_NEUTRAL_H
#define LARE_BOUNDARY_CLASS_NEUTRAL_H

#include "shared_data_neutral.h"
#include "runner.h"

/**
 * Simple class to implement the old LARE3D style boundary conditions
 */
class LAREBoundaryConditions_neutral : public SAMS::boundaryConditions
{
    public:

    void(*fn)(LARE_neutral::simulationData &, SAMS::timeState &, int, SAMS::domain::edges) = nullptr;
    LARE_neutral::simulationData &data;
    SAMS::timeState &time;
    
    LAREBoundaryConditions_neutral(void(*function)(LARE_neutral::simulationData &, SAMS::timeState &, int , SAMS::domain::edges),
                           LARE_neutral::simulationData &simData, SAMS::timeState &timeStateRef)
        : fn(function), data(simData), time(timeStateRef)
    {
    }

    /**
     * Apply boundary conditions to LARE3D variables
     * @param data LARE3D simulation data
     * @param dimension Dimension to apply BCs in
     * @param edge Edge of the domain to apply BCs on
     */
    void apply(int dimension, SAMS::domain::edges edge) override
    {
        fn(data, time, dimension, edge);
    }
};

#endif
