//This file contains transparent handles for SAMS objects
#ifndef SAMS_HANDLES_H
#define SAMS_HANDLES_H

namespace SAMS{
    struct MPIAxis{
        int direction=-1;
        MPIAxis()=default;
        explicit MPIAxis(int dir):direction(dir){}
        operator int() const { return direction; }
    };
}

#endif // SAMS_HANDLES_H