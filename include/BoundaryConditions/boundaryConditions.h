#ifndef SAMS_BOUNDARYCONDITIONS_H
#define SAMS_BOUNDARYCONDITIONS_H

#include "harnessDef.h"

namespace SAMS
{

    class boundaryConditions
    {
    public:
        boundaryConditions() = default;
        virtual ~boundaryConditions() = default;
        virtual void apply(int dimension, SAMS::domain::edges edge) = 0;
    };

}

#endif // SAMS_BOUNDARYCONDITIONS_H