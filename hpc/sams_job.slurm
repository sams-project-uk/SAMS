#!/bin/bash
# Slurm job for a SAMS run that continues itself from checkpoints until it is complete.
#
# Submit from (or with --chdir to) the run directory:
#     mkdir -p $RUNS/long_a1 && cd $RUNS/long_a1
#     sbatch --export=ALL,SAMS_ROOT=$HOME/SAMS,SAMS_BIN=$HOME/sams_bin/sams_L10000_dx02,CKPT_DT=50 $HOME/SAMS/hpc/sams_job.slurm
#
# Each job:
#   - copies the executable into the run directory on the first job (later jobs reuse that copy, so a
#     rebuilt SAMS_BIN cannot change a run half way) and links the data directory;
#   - restarts from ./checkpoint if a checkpoint exists and the run is not complete;
#   - sets SAMS_CHECKPOINT_WALLTIME to the job's remaining time minus WALL_MARGIN, so SAMS writes a
#     checkpoint and stops cleanly before Slurm kills it;
#   - resubmits itself (same resources) if SAMS stopped at the wall-time limit, up to MAX_CHAIN jobs.
# A restart needs the same number of ranks, so keep --nodes/--ntasks the same for the whole run.
#
# Variables (pass with --export):
#   SAMS_ROOT   repository (for hpc/env_dial3.sh and data/)              required
#   SAMS_BIN    executable from hpc/build_sams.sh                         required for the first job
#   PACKAGES    ["LARE3DST LARE3DNF PIP TwoFluidTest"]
#   CKPT_DT     also write a checkpoint every this much simulation time   [none]
#   WALL_MARGIN seconds kept free at the end of each job                  [900]
#   MAX_CHAIN   maximum number of jobs in the chain                       [30]
#   CHAIN       job number in the chain (set by the script)               [1]
#   ENV_FILE    environment to source                                     [$SAMS_ROOT/hpc/env_dial3.sh]
#
#SBATCH --job-name=sams
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
: "${SAMS_ROOT:?set SAMS_ROOT (sbatch --export=ALL,SAMS_ROOT=...)}"
PACKAGES=${PACKAGES:-"LARE3DST LARE3DNF PIP TwoFluidTest"}
WALL_MARGIN=${WALL_MARGIN:-900}
MAX_CHAIN=${MAX_CHAIN:-30}
CHAIN=${CHAIN:-1}
INFO=checkpoint/checkpoint_info.txt

source "${ENV_FILE:-$SAMS_ROOT/hpc/env_dial3.sh}"
# Settings inherited from the previous job of a chain are recomputed below
unset SAMS_RESTART SAMS_CHECKPOINT_WALLTIME SAMS_CHECKPOINT_DT
cd "${SLURM_SUBMIT_DIR:-.}"
status() { [ -f $INFO ] && awk '$1 == "status" {print $2}' $INFO || echo none; }

if [ "$(status)" = complete ]; then
    echo "Run in $PWD is already complete"; exit 0
fi

# First job: take a private copy of the executable and its settings; link the atomic data
if [ ! -x ./sams ]; then
    : "${SAMS_BIN:?set SAMS_BIN for the first job of a run}"
    cp "$SAMS_BIN" ./sams
    [ -f "$SAMS_BIN.config" ] && cp "$SAMS_BIN.config" ./sams.config
fi
[ -e data ] || ln -s "$SAMS_ROOT/data" data
mkdir -p SAMS_testdata

# Remaining wall time of this job in seconds (squeue %L: [D-]HH:MM:SS, MM:SS or UNLIMITED)
left=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
secs=$(echo "$left" | awk -F'[-:]' '{ if ($0 ~ /UNLIMITED|NOT_SET|INVALID/) {print 0; exit}
    n = NF; s = $n + 60 * $(n-1); if (n >= 3) s += 3600 * $(n-2); if (n == 4) s += 86400 * $1; print s }')
if [ "$secs" -gt $((WALL_MARGIN + 60)) ]; then   # at least a minute of running, or no limit
    export SAMS_CHECKPOINT_WALLTIME=$((secs - WALL_MARGIN))
fi
export SAMS_CHECKPOINT_DIR=checkpoint
[ -n "${CKPT_DT:-}" ] && export SAMS_CHECKPOINT_DT=$CKPT_DT
if [ "$(status)" = running ] || [ "$(status)" = stopped_walltime ]; then
    export SAMS_RESTART=checkpoint
fi

echo "Job $SLURM_JOB_ID (chain $CHAIN/$MAX_CHAIN) on $SLURM_NTASKS ranks in $PWD; time left $left;" \
     "SAMS_CHECKPOINT_WALLTIME=${SAMS_CHECKPOINT_WALLTIME:-unset} restart=${SAMS_RESTART:-no}"
{ echo "===== job $SLURM_JOB_ID, chain $CHAIN, $(date -u +%Y-%m-%dT%H:%M:%SZ), restart=${SAMS_RESTART:-no} ====="; } >> run.log
set +e
srun --cpu-bind=cores ./sams $PACKAGES >> run.log 2>&1
rc=$?
set -e
echo "SAMS exited with status $rc; run status: $(status)"

case "$(status)" in
    complete) echo "Run complete" ;;
    stopped_walltime)
        if [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
            sbatch --export=ALL,CHAIN=$((CHAIN + 1)) "$SAMS_ROOT/hpc/sams_job.slurm"
        else
            echo "MAX_CHAIN ($MAX_CHAIN) reached: resubmit by hand to continue"
        fi ;;
    *)  # no checkpoint status change: SAMS failed or the run finished without checkpointing enabled
        if [ $rc -ne 0 ]; then echo "SAMS failed (exit $rc): not resubmitting; see run.log"; exit $rc; fi ;;
esac
