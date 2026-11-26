#!/bin/bash
# The three long SAMS runs on DIaL3: the alpha0 = 1 current sheet to t = 10 000 on the half domain
# [0, 10 000] with dx = 0.2 (50 000 cells), a snapshot every 100, for
#   coll   thermal collisions only
#   ir     collisions + ionisation/recombination, no ionisation potential (PIP flag_IR = 3, flag_IR_type = 1)
#   irip   collisions + ionisation/recombination + ionisation potential loss (PIP flag_IR = 3, flag_IR_type = 0)
#
#   hpc/long_runs.sh build              build the three executables (login node, ~5 min each, one after another)
#   hpc/long_runs.sh submit [case ...]  create the run directories and submit (default: all three)
#   hpc/long_runs.sh status             step, time and status of each run
#
# Each run is one node (128 ranks), 72 h jobs through hpc/sams_job.slurm: a checkpoint every CKPT_DT of
# simulation time and before each job's time limit, and automatic resubmission until the run is complete.
# Keep the node count the same for a whole run (a restart needs the same number of ranks).
#
# Settings (environment): SAMS_ROOT [the repository this script is in]  CKPT_DT [100]  MAIL_TO [none] (address for END/FAIL mail)
#   NODES [1]  X_HALF [10000]  NX [100000] (full-domain cells; the half domain has NX/2)  T_END [10000]  DTS [100]
set -euo pipefail
SAMS_ROOT=${SAMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BIN=$SAMS_ROOT/sams_bin
CKPT_DT=${CKPT_DT:-100}
NODES=${NODES:-1}
X_HALF=${X_HALF:-10000}; NX=${NX:-100000}; T_END=${T_END:-10000}; DTS=${DTS:-100}
TAG=a1_L${X_HALF%.*}_dx$(python3 -c "print(f'{2*$X_HALF/$NX:g}'.replace('.',''))")
CASES="coll ir irip"

source "${ENV_FILE:-$SAMS_ROOT/hpc/env_dial3.sh}" > /dev/null
case_env() {   # build switches of each case
    case $1 in
        coll) echo "IR=off" ;;
        ir)   echo "IR=on IPOT=off" ;;
        irip) echo "IR=on IPOT=on" ;;
        *) echo "unknown case '$1' (coll, ir, irip)" >&2; exit 1 ;;
    esac
}
exe() { echo "$BIN/sams_long_$1_$TAG"; }
rundir() { echo "$SAMS_SCRATCH/long_$1_$TAG"; }

cmd=${1:-}; shift || true
case $cmd in
build)
    mkdir -p "$BIN"
    for c in ${*:-$CASES}; do
        echo "== building $c: $(case_env $c) X_HALF=$X_HALF NX=$NX T_END=$T_END DTS=$DTS"
        env $(case_env $c) X_HALF=$X_HALF NX=$NX T_END=$T_END DTS=$DTS "$SAMS_ROOT/hpc/build_sams.sh" "$(exe $c)"
    done ;;
submit)
    for c in ${*:-$CASES}; do
        [ -x "$(exe $c)" ] || { echo "no executable $(exe $c): run 'hpc/long_runs.sh build' first"; exit 1; }
        d=$(rundir $c)
        if [ -f "$d/checkpoint/checkpoint_info.txt" ] || [ -x "$d/sams" ]; then
            echo "$d already holds a run: submitting continues it (delete the directory to start again)"
        fi
        mkdir -p "$d"; cd "$d"
        sbatch -J "sams_$c" --nodes="$NODES" ${MAIL_TO:+--mail-user=$MAIL_TO} \
               --export=ALL,SAMS_ROOT=$SAMS_ROOT,SAMS_BIN=$(exe $c),CKPT_DT=$CKPT_DT \
               "$SAMS_ROOT/hpc/sams_job.slurm"
        echo "   $c: $d"
    done ;;
status)
    for c in $CASES; do
        d=$(rundir $c); echo "== $c: $d"
        [ -d "$d" ] || { echo "   not started"; continue; }
        [ -f "$d/checkpoint/checkpoint_info.txt" ] && sed 's/^/   /' "$d/checkpoint/checkpoint_info.txt" | grep -E "time|step|status"
        [ -f "$d/run.log" ] && echo "   last: $(grep 'Starting timestep' "$d/run.log" | tail -1 | cut -c1-90)"
        echo "   snapshots: $(ls "$d/SAMS_testdata" 2>/dev/null | sed 's/_rank.*//' | sort -u | wc -l)"
    done
    squeue -u "$USER" -o "%.10i %.14j %.8T %.10M %.10l %R" 2>/dev/null || true ;;
*)
    sed -n '2,18p' "$0"; exit 1 ;;
esac
