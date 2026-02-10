#ifndef SAMS_MPI_DEFAULTTYPES_H
#define SAMS_MPI_DEFAULTTYPES_H

#ifndef USE_MPI
    typedef int MPI_Comm;
    inline int MPI_COMM_WORLD = 0;
    inline int MPI_PROC_NULL = 0;

    typedef int MPI_Datatype;
    inline int MPI_DATATYPE_NULL = 0;

    typedef size_t MPI_Aint;
    typedef int MPI_Fint;

    inline int MPI_ORDER_C = 0;
#endif
#endif //SAMS_MPI_DEFAULTTYPES_H