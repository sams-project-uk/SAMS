#!/bin/bash
# PIP reference run for the collisions-only Snow & Hillier (2021) current sheet, half domain [0, x_half].
# See "SAMS PIP collisions-only comparison handoff.md".
#
# usage: scripts/pip_collonly_run.sh <patched PIP source dir> <x_half> <ix> <name> [s_order=1] [mpi ranks=1]
#   <patched PIP source dir>: PIP khinvel at 0474b7a with pip_snow_hillier_2021.patch and
#                             pip_half_domain.patch applied (left unchanged; a copy is built)
#   <ix>: total number of cells on [0, x_half] (PIP splits them over the MPI ranks)
# Creates ./pip_<name>/ with the build, run.log and the snapshots in ./pip_<name>/out/ (t0000..t0010.h5).
# Environment overrides: TEND (default 100), DTOUT (default 10), COL (alpha0, default 1.0).
set -euo pipefail
SRC=$(cd "$1" && pwd); XH=$2; IX=$3; NAME=$4; ORDER=${5:-1}; NP=${6:-1}
TEND=${TEND:-100.0}; DTOUT=${DTOUT:-10.0}; COL=${COL:-1.0}
D=$(pwd)/pip_$NAME
rm -rf "$D"; mkdir -p "$D"
cp "$SRC"/*.f90 "$SRC"/Makefile "$SRC"/setting.txt "$D"/
cp "$SRC"/*.txt "$SRC"/*.hdf5 "$D"/ 2>/dev/null || true
cd "$D"
# --- shock_tube_ion.f90: half domain [0, x_half], sharp sheet
sed -i -E 's/^( *logical,parameter *:: *half_domain *= *)\.(true|false)\./\1.true./' shock_tube_ion.f90
sed -i -E 's/^( *logical,parameter *:: *sharp_interface *= *)\.(true|false)\./\1.true./' shock_tube_ion.f90
sed -i -E "s/^( *start\(1\)=0\.0d0 *; *end\(1\)=)[0-9.]+d0.*/\1${XH}d0/" shock_tube_ion.f90
grep -q "half_domain=.true." shock_tube_ion.f90 || { echo "half_domain edit failed"; exit 1; }
grep -qE "start\(1\)=0\.0d0 *; *end\(1\)=${XH}d0" shock_tube_ion.f90 || { echo "end(1) edit failed"; exit 1; }
# --- setting.txt
sed -i -E "1s|^[^:]*:outdir|out/:outdir|" setting.txt
sed -i -E "s/^[0-9]+:ix;.*/${IX}:ix; total cells on [0, ${XH}]/" setting.txt
sed -i -E "s/^[0-9.eEd+-]+:dtout;.*/${DTOUT}:dtout; output period/" setting.txt
sed -i -E "s/^[0-9.eEd+-]+:tend;.*/${TEND}:tend; end time/" setting.txt
sed -i -E "s/^[0-9]+:flag_IR;.*/0:flag_IR; no ionisation\/recombination (collisions only)/" setting.txt
sed -i -E "s/^[0-9]+:flag_col;.*/2:flag_col; alpha0*sqrt((Tn+Tp)\/2\/Tinit), Alfven units/" setting.txt
sed -i -E "s/^[0-9.eEd+-]+ *:col;.*/${COL}:col; alpha0/" setting.txt
sed -i -E "s/^[0-9]+:s_order;.*/${ORDER}:s_order;order of space for HLL branch (1, 2)/" setting.txt
sed -i -E "s/^[0-9]+ +:mpi_x;.*/${NP}     :mpi_x;node number for x-direction/" setting.txt
for k in ix dtout tend flag_IR flag_col s_order mpi_x; do grep -E "^[^;]*:${k};" setting.txt; done
grep -E ":col;" setting.txt
# --- build (serial make: the Fortran modules must be compiled in order) and run
make FC=h5pfc > build.log 2>&1 || { tail -20 build.log; exit 1; }
mkdir -p out
/usr/bin/time -f "%e s wall" mpirun -np "$NP" ./a.out > run.log 2>&1
grep -E "Neutral fraction|END of simulation" run.log; tail -1 run.log
