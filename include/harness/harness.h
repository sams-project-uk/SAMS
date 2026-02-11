#ifndef HARNESS_H
#define HARNESS_H

#include "memoryRegistry.h"
#include "mpiManager.h"
#include "axisRegistry.h"
#include "variableRegistry.h"

namespace SAMS {
    struct harness{
        SAMS::memoryRegistry memoryRegistry;
        SAMS::axisRegistry axisRegistry{memoryRegistry};
        SAMS::MPIManager<SAMS::MPI_DECOMPOSITION_RANK> MPIManager{axisRegistry};
        SAMS::variableRegistry variableRegistry{MPIManager, axisRegistry, memoryRegistry};

        void initialize([[maybe_unused]] int &argc, [[maybe_unused]] char** &argv){
        }

        void abort(const std::string& message = "", bool localError = false){
            MPIManager.abort(message, localError);
        }

        void finalize(){
            memoryRegistry.finalize();
            MPIManager.finalize();
        }
    };
};

#endif //HARNESS_H