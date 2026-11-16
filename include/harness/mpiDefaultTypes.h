#ifndef SAMS_MPI_DEFAULTTYPES_H
#define SAMS_MPI_DEFAULTTYPES_H

#include <cstddef>
#ifdef USE_MPI
#include <mpi.h>
#endif

//SAMS names for the MPI types and constants that are used with or without MPI. With MPI they are only other names for MPI_Comm etc.
namespace SAMSMPI
{
#ifdef USE_MPI
    using comm = MPI_Comm;
    using datatype = MPI_Datatype;
    using aint = MPI_Aint;

    inline const comm commWorld = MPI_COMM_WORLD;
    inline const comm commNull = MPI_COMM_NULL;
    inline const int procNull = MPI_PROC_NULL;
    inline const datatype datatypeNull = MPI_DATATYPE_NULL;
#else
    using comm = int;
    using datatype = int;
    using aint = size_t;

    inline constexpr comm commWorld = 0;
    //Different from commWorld, so that a communicator can be compared with it
    inline constexpr comm commNull = -1;
    inline constexpr int procNull = 0;
    inline constexpr datatype datatypeNull = 0;
#endif
}
#endif //SAMS_MPI_DEFAULTTYPES_H