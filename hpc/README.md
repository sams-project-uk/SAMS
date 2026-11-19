# Running SAMS on DIaL3 (DiRAC)

This guide covers building SAMS on DIaL3 and running the collisions-only Snow & Hillier (2021) current sheet there. The target is the long run to t = 10 000.

The scripts in this directory are:

| File | What it does |
| --- | --- |
| `env_dial3.sh` | Loads the compiler, MPI and libraries, and sets the run-time environment. **Edit its module names once.** |
| `build_sams.sh` | Builds an executable for one set-up. The grid and parameters are compiled in. |
| `sams_job.slurm` | Slurm job that checkpoints before its time limit and resubmits itself until the run is complete. |
| `compare_snapshots.py` | Compares two runs' snapshot files, bit for bit or within a tolerance. |

Things to check on DIaL3 itself are marked **(check)**. They come from memory, not from the DIaL3 documentation.

## 1. What SAMS needs

- **Compiler:** C++20, so GCC 12 or newer.
- **MPI:** any MPI. Run one rank per core with `OMP_NUM_THREADS=1`; OpenMP threads give no speed-up.
- **Libraries:** HDF5 (serial is fine, because every rank writes its own file), NetCDF-C, CMake ≥ 3.20, pkg-config and Python 3 (used by the build script).
- **PETSc:** needed only for the implicit two-fluid source solver (`implicit=true`). The long run uses the explicit update, so build with `USE_PETSC=OFF` (the default in `build_sams.sh`) and skip PETSc. If you do want it, PETSc ≥ 3.18 has to be built with **the same MPI** as SAMS; mixing them fails at compile time.

## 2. One-time set-up

```bash
cd $HOME
git clone https://github.com/AstroSnow/SAMS_PI.git SAMS     # branch TwoFluidWIP
cd SAMS && git checkout TwoFluidWIP
module avail gcc openmpi hdf5 netcdf cmake                  # (check) the module names
vi hpc/env_dial3.sh                                         # put those names in
source hpc/env_dial3.sh                                     # warns about anything missing
```

DIaL3 nodes have 128 cores, 2 × 64-core AMD EPYC **(check)**. For AMD, `CMAKE_EXTRA="-DCMAKE_CXX_FLAGS=-march=znver2"` in step 3 may help. Compare its speed with a plain build in the scaling test (step 5) before relying on it.

Also set `--account` and `--partition` in `sams_job.slurm` to your DiRAC project and the right partition **(check)**. Check the maximum job time too; the script asks for 24 h.

## 3. Build an executable

Every set-up is its own executable. The settings are environment variables; the full list is at the top of `build_sams.sh`. The grid is `NX` cells across the **full** domain [−X_HALF, X_HALF], so dx = 2·X_HALF/NX, and the half domain (the default) has NX/2 cells.

```bash
source hpc/env_dial3.sh
mkdir -p $HOME/sams_bin

# Validation: the alpha0 = 1 baseline to t = 1 (same as the workstation runs)
T_END=1 DTS=0.1 hpc/build_sams.sh $HOME/sams_bin/sams_a1_t1

# The long run: L = 10 000, dx = 0.2, t = 10 000, a snapshot every 100
X_HALF=10000 NX=100000 T_END=10000 DTS=100 hpc/build_sams.sh $HOME/sams_bin/sams_L10000_dx02
```

Each build writes a `.config` file next to the executable. It records the settings and the git commit. The build edits the two `TwoFluidTest` source files and restores them afterwards. A lock stops two builds from running at the same time.

## 4. Validate against the workstation

Run the t = 1 baseline on 3 ranks, the same number as the workstation reference. An interactive job or a short batch job is fine:

```bash
mkdir -p $SCRATCH/sams/val_a1_t1 && cd $SCRATCH/sams/val_a1_t1     # (check) the scratch path
cp $HOME/sams_bin/sams_a1_t1 ./sams && ln -s $HOME/SAMS/data data && mkdir SAMS_testdata
srun -n 3 ./sams LARE3DST LARE3DNF PIP TwoFluidTest > run.log 2>&1
```

Copy the workstation reference across (`runs/outfix_1/SAMS_testdata` on the workstation: same set-up, 3 ranks, t = 1, snapshots every 0.1), then compare:

```bash
python3 $HOME/SAMS/hpc/compare_snapshots.py <workstation SAMS_testdata> SAMS_testdata --rtol 1e-8
```

A different compiler and CPU change the round-off, so the runs won't be bit-identical. Differences around 10⁻¹⁰ relative mean the build is good; anything near 10⁻³ or more means something is wrong.

## 5. Scaling test (before the long run)

Measure the time per step at the long-run grid on 1 and 2 nodes. Use a short end time, e.g. `T_END=20` in a scratch build, and compare the step rate in `run.log`. The rest of the run can be estimated from that.

The workstation measured about 2×10⁻⁴ s per cell per step per rank. At 128 ranks, the long run (50 000 cells, about 1.2 million steps) would take about 30 h on one node, if DIaL3 cores are about as fast and the code scales that far. That has not been tested beyond 8 ranks, which is why this step matters.

Keep at least a few hundred cells per rank.

## 6. The long run

```bash
mkdir -p $SCRATCH/sams/long_a1_L10000_dx02 && cd $SCRATCH/sams/long_a1_L10000_dx02
sbatch --export=ALL,SAMS_ROOT=$HOME/SAMS,SAMS_BIN=$HOME/sams_bin/sams_L10000_dx02,CKPT_DT=100 \
       $HOME/SAMS/hpc/sams_job.slurm
```

What each job does:
- **First job:** copies the executable into the run directory as `./sams`, with its `.config`. Later jobs use that copy, so rebuilding `SAMS_BIN` can't change a run halfway.
- **Checkpoints:** writes a checkpoint every `CKPT_DT` of simulation time (`checkpoint/`). It also writes one and stops cleanly 15 minutes (`WALL_MARGIN`) before the job's time limit.
- **Continuing:** if it stopped at the time limit, it submits the next job, which restarts from the checkpoint. A restarted run is bit-identical to one that never stopped.
- **Stopping:** the chain stops when `checkpoint/checkpoint_info.txt` says `status complete`, after `MAX_CHAIN` jobs (30), or if SAMS fails (no resubmission; see `run.log`).
- **Rank count:** keep it the same throughout. A restart refuses a different number of ranks.

To follow it:

```bash
squeue -u $USER
tail run.log                       # "Starting timestep ... at time ..." every 10 steps
cat checkpoint/checkpoint_info.txt # time, step and status of the last checkpoint
```

To continue by hand (after `MAX_CHAIN`, or after fixing a problem), submit the same command again in the same directory. It restarts from the last checkpoint.

### Disk space

Rough figures for the long run:
- **Snapshots:** about 50 MB each, so about 5 GB for 100.
- **Checkpoint:** about 0.7 GB. The previous one is kept as `.prev`.

Use scratch for runs and copy the snapshots home when done.

## 7. Restart files: details

These are controlled by environment variables, which `sams_job.slurm` sets. They can also be used directly:

| Variable | Meaning |
| --- | --- |
| `SAMS_CHECKPOINT_DT` | checkpoint every this much simulation time |
| `SAMS_CHECKPOINT_STEPS` | checkpoint every this many steps |
| `SAMS_CHECKPOINT_WALLTIME` | after this many seconds, checkpoint and stop |
| `SAMS_CHECKPOINT_DIR` | where to write them (default `checkpoint`) |
| `SAMS_RESTART` | continue from the checkpoint in this directory |

A checkpoint is one file per rank holding every registered array, including ghost cells. A restart must use the same executable and the same number of ranks; it checks both. The implementation is in `src/Checkpoint/`.

## 8. Troubleshooting

- **`PETSc was configured with one Open MPI mpi.h version but now appears to be compiling using an older ...`:** PETSc and SAMS see different MPIs. Use `USE_PETSC=OFF`, or rebuild PETSc with the MPI from `env_dial3.sh`.
- **`implicit=true needs the PETSc source solver, but this build has no PETSc`:** the executable was built with `USE_PETSC=OFF`.
- **`restart files were written by N ranks, but this run has M`:** the job's rank count changed. Use the original `--nodes` and `--ntasks-per-node`.
- **`variable ... has ... bytes here but ... in the restart file`:** the executable doesn't match the one that wrote the checkpoint (different grid). Use the `./sams` copy in the run directory.
- **`cannot open restart file ...`:** the checkpoint directory is missing or incomplete. The previous checkpoint is kept as `*.prev`. Rename those files to `ckpt_rank*.bin` to fall back to it.
