# Running SAMS on DIaL3 (DiRAC)

This guide covers building SAMS on DIaL3 and running the collisions-only Snow & Hillier (2021) current sheet there. The target is the long run to t = 10 000.

The scripts in this directory are:

| File | What it does |
| --- | --- |
| `env_dial3.sh` | Compiler, MPI, HDF5 and run-time settings. Same stack as PIP on DIaL3. |
| `build_sams.sh` | Builds an executable for one set-up. The grid and parameters are compiled in. |
| `sams_job.slurm` | Slurm job that checkpoints before its time limit and resubmits itself until the run is complete. |
| `compare_snapshots.py` | Compares two runs' snapshot files, bit for bit or within a tolerance. |

Things not yet tried on DIaL3 are marked **(untested)**.

## 1. The environment

`env_dial3.sh` uses the stack that works for PIP on DIaL3:

- **Compiler and MPI:** `intel-oneapi/2022.1.2` and `openmpi4/intel/4.0.5`, with the PMIx 3 library put on `LD_LIBRARY_PATH`. Without that, `mpirun` fails with "unexpected version of PMIx".
- **HDF5:** your parallel HDF5 in `$HOME/sw/hdf5-1.14.6-ompi4-intel`. SAMS uses only HDF5's C API, so a build without the C++ library is fine.
- **malloc:** `MALLOC_MMAP_MAX_=0` and `MALLOC_TRIM_THRESHOLD_`, as for PIP.
- **Scratch:** `SAMS_SCRATCH=/scratch/dp446/$USER` for run directories.

On top of that, SAMS needs a **C++20** compiler. The environment points the OpenMPI wrappers at Intel's LLVM-based `icpx`/`icx` (`OMPI_CXX=icpx`), because the classic `icpc` doesn't fully support C++20. `icpx` uses the C++ library of a GCC installation, and the system GCC is too old for C++20 (no `<concepts>`). `env_dial3.sh` therefore loads the newest GCC 10–12 module it finds and passes it to `icpx` with `--gcc-toolchain` (via `OMPI_CXXFLAGS`). To choose a different one, set `SAMS_GCC_MODULE=gcc/<version>` before sourcing. **(untested)** `build_sams.sh` checks that C++20 compiles before the long build.

**NetCDF and PETSc aren't needed.** NetCDF is used only by the n-level ionisation model, PETSc only by the implicit two-fluid solver. `build_sams.sh` builds without both (`USE_NETCDF=OFF`, `USE_PETSC=OFF`), and a run that asks for either feature stops at start-up with a message.

## 2. One-time set-up

```bash
cd $HOME
git clone https://github.com/AstroSnow/SAMS_PI.git SAMS
cd SAMS && git checkout TwoFluidWIP
source hpc/env_dial3.sh                  # warns about anything missing
```

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

The DIaL3 nodes are AMD EPYC. `CMAKE_EXTRA="-DCMAKE_CXX_FLAGS=-march=core-avx2"` (Intel compilers) may be faster than the default; compare it in the scaling test (step 5) before using it.

## 4. Validate against the workstation

Run the t = 1 baseline on 3 ranks, the same number as the workstation reference. A short interactive job is enough:

```bash
srun -A dp446 -p slurm -N 1 -n 3 --time=00:30:00 --pty bash      # or a small batch job
source $HOME/SAMS/hpc/env_dial3.sh
mkdir -p $SAMS_SCRATCH/val_a1_t1 && cd $SAMS_SCRATCH/val_a1_t1
cp $HOME/sams_bin/sams_a1_t1 ./sams && ln -s $HOME/SAMS/data data && mkdir -p SAMS_testdata
mpirun -np 3 --bind-to core ./sams LARE3DST LARE3DNF PIP TwoFluidTest > run.log 2>&1
```

Copy the workstation reference across (`runs/outfix_1/SAMS_testdata` on the workstation: same set-up, 3 ranks, t = 1, snapshots every 0.1), then compare:

```bash
python3 $HOME/SAMS/hpc/compare_snapshots.py <workstation SAMS_testdata> SAMS_testdata --rtol 1e-8
```

A different compiler and CPU change the round-off, so the runs won't be bit-identical. Differences around 10⁻¹⁰ relative mean the build is good; anything near 10⁻³ or more means something is wrong.

## 5. Scaling test (before the long run)

Measure the time per step at the long-run grid on 1 and 2 nodes, with and without the `-march` flag from step 3. Use a short end time, e.g. a build with `T_END=20`, and compare the step rate in `run.log`.

The workstation measured about 2×10⁻⁴ s per cell per step per rank. At 128 ranks, the long run (50 000 cells, about 1.2 million steps) would take about 30 h on one node, if DIaL3 cores are about as fast and the code scales that far. That has not been tested beyond 8 ranks, which is why this step matters.

Keep at least a few hundred cells per rank.

## 6. The long run

```bash
source $HOME/SAMS/hpc/env_dial3.sh
mkdir -p $SAMS_SCRATCH/long_a1_L10000_dx02 && cd $SAMS_SCRATCH/long_a1_L10000_dx02
sbatch --mail-user=<your address> \
       --export=ALL,SAMS_ROOT=$HOME/SAMS,SAMS_BIN=$HOME/sams_bin/sams_L10000_dx02,CKPT_DT=100 \
       $HOME/SAMS/hpc/sams_job.slurm
```

The job header asks for 1 node (128 ranks), `--exclusive`, 72 h on partition `slurm`, account `dp446`. Change `--nodes` there if the scaling test says so, but keep it the same for the whole run.

What each job does:
- **First job:** copies the executable into the run directory as `./sams`, with its `.config`. Later jobs use that copy, so rebuilding `SAMS_BIN` can't change a run halfway.
- **Checkpoints:** writes a checkpoint every `CKPT_DT` of simulation time (`checkpoint/`). It also writes one and stops cleanly 15 minutes (`WALL_MARGIN`) before the job's time limit.
- **Continuing:** if it stopped at the time limit, it submits the next job, which restarts from the checkpoint. A restarted run is bit-identical to one that never stopped.
- **Stopping:** the chain stops when `checkpoint/checkpoint_info.txt` says `status complete`, after `MAX_CHAIN` jobs (30), or if SAMS fails (no resubmission; see `run.log`).
- **Launcher:** runs with `mpirun -np $SLURM_NTASKS --bind-to core` (set `LAUNCH` to change it).
- **Rank count:** a restart refuses a different number of ranks.

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

Keep runs on `$SAMS_SCRATCH` and copy the snapshots home when done.

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

- **`build_sams.sh: ... cannot compile C++20`:** load a newer GCC (10 or later) in `env_dial3.sh`, since `icpx` uses its C++ library. Check that `mpicxx --version` shows `icpx`.
- **`unexpected version of PMIx`:** `env_dial3.sh` didn't find the PMIx 3 library (it prints a warning). Check `/cm/shared/apps/cm-pmix3/`.
- **CMake can't find HDF5:** check `HDF5_DIR` in `env_dial3.sh`, and that `h5pcc` is on `PATH`.
- **`implicit=true needs the PETSc source solver` / `ion_rec_nlevel needs the atomic rate tables`:** this build has no PETSc / NetCDF (the defaults). Neither is used by the collisions-only runs.
- **`restart files were written by N ranks, but this run has M`:** the job's rank count changed. Use the original `--nodes` and `--ntasks-per-node`.
- **`variable ... has ... bytes here but ... in the restart file`:** the executable doesn't match the one that wrote the checkpoint (different grid). Use the `./sams` copy in the run directory.
- **`cannot open restart file ...`:** the checkpoint directory is missing or incomplete. The previous checkpoint is kept as `*.prev`. Rename those files to `ckpt_rank*.bin` to fall back to it.
