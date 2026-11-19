#!/bin/bash
# Build a SAMS executable for the collisions-only Snow & Hillier (2021) current sheet (TwoFluidTest,
# problem snow_hillier_2021). The grid and the run parameters are compiled in, so every set-up is its
# own executable.
#
#   hpc/build_sams.sh <output executable>
#
# Run from anywhere; the repository is found from this script's location. The two TwoFluidTest files
# are edited for the build and restored from a backup afterwards (uncommitted changes in them are kept).
# A lock stops two builds editing them at once. A <output>.config file records the settings used.
#
# Settings (environment variables, defaults in brackets):
#   Problem   ALPHA0 [1.0]  X_HALF [250]  NX [12500] (cells across the FULL domain [-X_HALF, X_HALF]; the
#             half domain uses NX/2, so dx = 2 X_HALF / NX)  HALF [on] (half domain, symmetry plane at x = 0)
#   Time      T_END [10]  DTS [10] (snapshot interval)
#   Numerics  RKE_P, RKE_N [on] (remap kinetic-energy correction per fluid)  RME [on]  VFH [on]
#             (energy-conserving frictional heating)  VISC1 [0.1]  VISC2 [1.0]  IMPLICIT [off]
#   Build     BUILD_DIR [<repo>/build-hpc]  USE_PETSC [OFF]  JOBS [8]  CMAKE_EXTRA [] (e.g. "-DCMAKE_CXX_FLAGS=-march=znver2")
#
# Example (the long run: L = 10000, dx = 0.2, t = 10000, a snapshot every 100):
#   X_HALF=10000 NX=100000 T_END=10000 DTS=100 hpc/build_sams.sh $HOME/sams_bin/sams_L10000_dx02
set -euo pipefail
out=$(realpath -m "${1:?usage: build_sams.sh <output executable>}")
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-$ROOT/build-hpc}
H=$ROOT/src/InitialConditions/TwoFluidTests/TwoFluidTest.h
C=$ROOT/src/InitialConditions/TwoFluidTests/TwoFluidTest.cpp

exec 9> "$ROOT/.build_sams.lock"
flock 9
# Code version, recorded before the TwoFluidTest files are edited for this set-up
VERSION="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$ROOT" diff --quiet HEAD -- src include CMakeLists.txt 2>/dev/null || echo ' plus uncommitted changes')"
cp "$H" "$H.buildbak"; cp "$C" "$C.buildbak"
trap 'mv "$H.buildbak" "$H"; mv "$C.buildbak" "$C"' EXIT

python3 - "$H" "$C" <<'PY'
import os, sys
H, C = sys.argv[1:]; e = os.environ.get
on = lambda k, d="on": e(k, d).lower() in ("on", "1", "yes", "true")
tf = lambda b: "true" if b else "false"
def edit(p, pairs):
    s = open(p).read()
    for a, b in pairs:
        if s.count(a) != 1: sys.exit(f"build_sams.sh: {p}: pattern found {s.count(a)} times: {a!r}")
        s = s.replace(a, b)
    open(p, "w").write(s)
edit(H, [("SAMS::T_dataType t_end = 100.0;", f"SAMS::T_dataType t_end = {e('T_END', '10.0')};"),
         ("SAMS::T_dataType x_half = 2000.0;", f"SAMS::T_dataType x_half = {e('X_HALF', '250.0')};"),
         ("SAMS::T_indexType nx = 200000; // dx = 0.02", f"SAMS::T_indexType nx = {e('NX', '12500')};"),
         ("SAMS::T_dataType alpha0 = 1.0;   // collisional", f"SAMS::T_dataType alpha0 = {e('ALPHA0', '1.0')};   // collisional"),
         ("bool half_domain = true;", f"bool half_domain = {tf(on('HALF'))};")])
edit(C, [("data.visc1 = 0.1;  // was 0.8", f"data.visc1 = {e('VISC1', '0.1')};  // was 0.8"),
         ("            data.visc2 = 1.0;\n", f"            data.visc2 = {e('VISC2', '1.0')};\n"),
         ("data.dt_snapshots = data.t_end / 10;", f"data.dt_snapshots = {e('DTS', '10.0')};"),
         ("            data.rke = false;\n", f"            data.rke = {tf(on('RKE_P'))};\n"),
         ("            dataNeutral.rke = data.rke;\n", f"            dataNeutral.rke = {tf(on('RKE_N'))};\n"),
         ("            data.rke_3d_fix = false;\n", f"            data.rke_3d_fix = {tf(on('RKE_P'))};\n"),
         ("            dataNeutral.rke_3d_fix = data.rke_3d_fix;\n", f"            dataNeutral.rke_3d_fix = {tf(on('RKE_N'))};\n"),
         ("            data.energy_budget = false;\n", f"            data.energy_budget = false;\n            data.rme = {tf(on('RME'))};\n"),
         ("                plasma_source.alpha0 = sh21.alpha0;\n",
          f"                plasma_source.alpha0 = sh21.alpha0;\n"
          f"                plasma_source.vertex_frictional_heating = {tf(on('VFH'))};\n"
          f"                plasma_source.implicit = {tf(on('IMPLICIT', 'off'))};\n")])
PY

# Configure every time (quick once cached), so a changed USE_PETSC or CMAKE_EXTRA takes effect
{
    cmake -S "$ROOT" -B "$BUILD_DIR" -DMPI=ON -DCMAKE_BUILD_TYPE=Release -DUSE_PETSC="${USE_PETSC:-OFF}" \
          -DMPI_CXX_COMPILER="$(command -v mpicxx)" -DMPI_C_COMPILER="$(command -v mpicc)" ${CMAKE_EXTRA:-} \
          > "$BUILD_DIR.cmake.log" 2>&1 || { tail -30 "$BUILD_DIR.cmake.log"; echo "CMake failed (log: $BUILD_DIR.cmake.log)"; exit 1; }
}
touch "$ROOT/src/main.cpp"
if ! make -C "$BUILD_DIR" -j"${JOBS:-8}" > "$BUILD_DIR.make.log" 2>&1; then
    grep -E "error|Error" -A3 "$BUILD_DIR.make.log" | head -40
    echo "Build failed (log: $BUILD_DIR.make.log)"; exit 1
fi
mkdir -p "$(dirname "$out")"
cp "$BUILD_DIR/sams" "$out"
{
    echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ) from $VERSION"
    for k in ALPHA0 X_HALF NX HALF T_END DTS RKE_P RKE_N RME VFH VISC1 VISC2 IMPLICIT USE_PETSC CMAKE_EXTRA; do
        echo "$k=${!k-default}"
    done
} > "$out.config"
echo "built $out (settings in $out.config)"
