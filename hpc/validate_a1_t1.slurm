#!/bin/bash
# Validation run: the alpha0 = 1 baseline to t = 1 on 3 ranks (same as the workstation reference).
# Build the executable first (hpc/build_sams.sh, README section 3), then submit from anywhere:
#     sbatch $HOME/TwoFluidWIP/hpc/validate_a1_t1.slurm
# The run directory is $SAMS_SCRATCH/val_a1_t1 (emptied first if it exists). If REF is set to the
# workstation's SAMS_testdata (copied to DIaL3), the snapshots are compared at the end:
#     sbatch --export=ALL,REF=$HOME/ref/outfix_1/SAMS_testdata $HOME/TwoFluidWIP/hpc/validate_a1_t1.slurm
#
# Variables (pass with --export): SAMS_ROOT [$HOME/TwoFluidWIP]  SAMS_BIN [$SAMS_ROOT/sams_bin/sams_a1_t1]
#   RUN_DIR [$SAMS_SCRATCH/val_a1_t1]  REF [none]
#
#SBATCH --job-name=sams_val
#SBATCH -A dp446
#SBATCH -p slurm
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=sams_val-%j.out

set -euo pipefail
SAMS_ROOT=${SAMS_ROOT:-$HOME/TwoFluidWIP}
SAMS_BIN=${SAMS_BIN:-$SAMS_ROOT/sams_bin/sams_a1_t1}
source "$SAMS_ROOT/hpc/env_dial3.sh"
RUN_DIR=${RUN_DIR:-$SAMS_SCRATCH/val_a1_t1}

[ -x "$SAMS_BIN" ] || { echo "No executable $SAMS_BIN (build it with hpc/build_sams.sh)"; exit 1; }
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR/SAMS_testdata"; cd "$RUN_DIR"
cp "$SAMS_BIN" ./sams
[ -f "$SAMS_BIN.config" ] && cp "$SAMS_BIN.config" ./sams.config
ln -s "$SAMS_ROOT/data" data

echo "Running $SAMS_BIN on $SLURM_NTASKS ranks in $RUN_DIR ($(hostname))"
start=$(date +%s)
mpirun -np "$SLURM_NTASKS" --bind-to core ./sams LARE3DST LARE3DNF PIP TwoFluidTest > run.log 2>&1 \
    || { echo "SAMS failed (exit $?); last lines of $RUN_DIR/run.log:"; tail -30 run.log; exit 1; }
echo "SAMS finished in $(( $(date +%s) - start )) s; $(ls SAMS_testdata | wc -l) files in SAMS_testdata"

if [ -n "${REF:-}" ]; then
    python3 "$SAMS_ROOT/hpc/compare_snapshots.py" "$REF" SAMS_testdata --rtol 1e-8
fi
