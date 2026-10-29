#ifndef HARNESS_H
#define HARNESS_H

#include "memoryRegistry.h"
#include "mpiManager.h"
#include "axisRegistry.h"
#include "variableRegistry.h"
#ifdef USE_PETSC
#include <petscsys.h>
#endif

namespace SAMS {
    struct harness{
        SAMS::memoryRegistry memoryRegistry;
        SAMS::MPIManager<SAMS::MPI_DECOMPOSITION_RANK> MPIManager;
        SAMS::axisRegistry axisRegistry{memoryRegistry, MPIManager};
        SAMS::variableRegistry variableRegistry{MPIManager, axisRegistry, memoryRegistry};

        void initialize([[maybe_unused]] int &argc, [[maybe_unused]] char** &argv){
        }

        void abort(const std::string& message = "", bool localError = false){
            MPIManager.abort(message, localError);
        }

        /** Whether PETSc has been initialised through initialisePETSc */
        bool petscInitialised = false;

        /**
         * Initialise PETSc if it has not been initialised already. Does nothing if SAMS is built without PETSc
         */
        void initialisePETSc(){
#ifdef USE_PETSC
            if (!petscInitialised){
                PetscErrorCode ierr = PetscInitializeNoArguments();
                if (ierr){
                    abort("PetscInitializeNoArguments failed with error " + std::to_string(ierr));
                }
                petscInitialised = true;
            }
#endif
        }

        /**
         * Finalise PETSc if it was initialised through initialisePETSc
         * The packages must have destroyed their PETSc objects, and the MPI manager must have freed its communicator, before this is called
         */
        void finalizePETSc(){
#ifdef USE_PETSC
            if (petscInitialised){
                PetscFinalize();
                petscInitialised = false;
            }
#endif
        }

        void finalize(){
            memoryRegistry.finalize();
            MPIManager.finalize();
            finalizePETSc();
        }
    };
};

#endif //HARNESS_H
