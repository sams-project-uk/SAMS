#ifndef SAMS_CHECKPOINT_H
#define SAMS_CHECKPOINT_H

/**
 * Restart (checkpoint) module. Implementation in checkpoint.cpp; the runner only calls the
 * checkpointManager at three points (start-up, after each step, and to restore on restart).
 *
 * A checkpoint is one binary file per MPI rank, <dir>/ckpt_rank<NNNNN>.bin, holding
 *   - the run state: simulation time, time step, step number, number of snapshots written;
 *   - every allocated registered variable, byte for byte, including its ghost cells;
 *   - a key/value block (checkpointState) for state that packages keep outside the registered
 *     variables. A package provides it with two optional methods, which the runner calls:
 *         void saveCheckpoint(SAMS::checkpointState &cs);
 *         void loadCheckpoint(const SAMS::checkpointState &cs);
 * A restart continues bit-identically, provided the packages keep no other state between steps.
 * It must use the same executable (grid and set-up are compiled in), the same number of ranks
 * and the same decomposition; the sizes of all variables are checked on reading.
 *
 * Writing is two-phase: every rank writes <file>.tmp, all ranks synchronise, then each rank
 * moves its previous file to <file>.prev and the new one into place. Rank 0 then rewrites
 * <dir>/checkpoint_info.txt (time, step, ranks, status: running | stopped_walltime | complete).
 *
 * Host memory only (variables are copied with memcpy): GPU builds refuse to checkpoint.
 *
 * Control (environment variables, read at start-up):
 *   SAMS_CHECKPOINT_DIR       directory for checkpoints (default "checkpoint", relative to the run directory)
 *   SAMS_CHECKPOINT_DT        write a checkpoint every this much simulation time
 *   SAMS_CHECKPOINT_STEPS     write a checkpoint every this many steps
 *   SAMS_CHECKPOINT_WALLTIME  after this many seconds of wall time, write a checkpoint and stop cleanly
 *                             (set it a safe margin below the batch job's time limit)
 *   SAMS_RESTART              restart from the checkpoint in this directory ("1" or "yes": SAMS_CHECKPOINT_DIR)
 */

#include <map>
#include <string>
#include <stdexcept>
#ifdef USE_MPI
#include <mpi.h>
#else
#include "mpiDefaultTypes.h"
#endif

namespace SAMS
{
    class variableRegistry;

    /**
     * Package state saved in a checkpoint besides the registered variables.
     * Keys should be prefixed with the package name.
     */
    struct checkpointState
    {
        std::map<std::string, double> reals;
        std::map<std::string, long long> ints;

        void setReal(const std::string &key, double v) { reals[key] = v; }
        void setInt(const std::string &key, long long v) { ints[key] = v; }
        bool hasReal(const std::string &key) const { return reals.count(key) > 0; }
        bool hasInt(const std::string &key) const { return ints.count(key) > 0; }
        double getReal(const std::string &key) const {
            auto it = reals.find(key);
            if (it == reals.end()) throw std::runtime_error("checkpoint has no real value '" + key + "'");
            return it->second;
        }
        long long getInt(const std::string &key) const {
            auto it = ints.find(key);
            if (it == ints.end()) throw std::runtime_error("checkpoint has no integer value '" + key + "'");
            return it->second;
        }
    };

    /**
     * Run-level values stored in the checkpoint header
     */
    struct checkpointHeader
    {
        double time = 0.0;
        double dt = 0.0;
        long long step = 0;
        long long outputCount = 0;
        int nRanks = 1;
        int rank = 0;
    };

    /**
     * Owns the checkpoint settings and does all reading and writing. The runner supplies the run
     * state (checkpointHeader), the package state (checkpointState) and the variable registry.
     */
    class checkpointManager
    {
    public:
        /** What to do after a step */
        enum class action { none, write, writeAndStop };

        /** Read the SAMS_CHECKPOINT_* and SAMS_RESTART environment variables */
        void configureFromEnvironment();

        bool enabled() const { return dtSim > 0.0 || everySteps > 0 || wallLimit > 0.0; }
        bool restartRequested() const { return !restartDir.empty(); }
        const std::string &directory() const { return dir; }
        const std::string &restartDirectory() const { return restartDir; }
        double wallTimeLimit() const { return wallLimit; }

        /**
         * Decide after a step (collective: every rank gets the same answer; rank 0's clock decides
         * the wall-time limit). wallSeconds is the wall time since the start of the program.
         */
        action afterStep(double time, long long step, double wallSeconds, MPI_Comm comm);

        /** Write a checkpoint (collective). status goes into checkpoint_info.txt */
        void write(variableRegistry &varReg, const checkpointHeader &h, const checkpointState &state,
                   MPI_Comm comm, const std::string &status);

        /**
         * Read this rank's checkpoint from restartDirectory() into the registered variables and
         * return the run and package state. Checks the rank count and every variable's size.
         */
        void read(variableRegistry &varReg, int rank, int nRanks, checkpointHeader &h, checkpointState &state);

        /** Mark the run complete in checkpoint_info.txt (call on rank 0 at a normal end) */
        void markComplete(int rank);

    private:
        std::string dir = "checkpoint";
        std::string restartDir;          // empty: no restart
        double dtSim = -1.0;             // <= 0: off
        long long everySteps = -1;       // <= 0: off
        double wallLimit = -1.0;         // <= 0: off
        double nextSim = -1.0;           // next simulation time for a checkpoint
    };
} // namespace SAMS

#endif // SAMS_CHECKPOINT_H
