# SAMS vs PIP: collisions-only current sheet to t = 100

Nov 6, 2026 · @Ben · run on a larger machine

## Goal

Compare SAMS and PIP on the Snow & Hillier (2021) current sheet with **thermal collisions only**, up to **t = 100**. Make two figures: the full comparison, and a close-up of the slow shock showing the ion–neutral separation. Two requirements:

- **Nothing reaches the boundary.** The outer boundary must stay undisturbed until t = 100.
- **At least 50 cells across the finite width of the shock.**

"Thermal collisions only" here means the following, in both codes:
- **On:** collisional coupling, with momentum and thermal exchange, and α\_c = α0·√((T\_p + T\_n)/2T0), α0 = 1.
- **Off:** ionisation, recombination and the ionisation potential.

The ionisation fraction stays at its equilibrium value at T0 = 10⁴ K (ξ\_n = 0.99734); both codes compute the same number. The rest of the set-up is the usual one: β = 0.1, B\_x = 0.1, B\_y = ∓1, sharp sheet, Alfvén units. Use the half domain [0, L] with a symmetry plane at x = 0.

Everything in this note is on branch `TwoFluidWIP` (after `44d2990`; the collisions-only switch and the scripts are in the two commits that follow). It also builds on the first handoff, "SAMS two-fluid handoff energy conservation and symmetry.md", which covers `visc1 = 0.1`, the energy budget and the PIP set-up.

## What the pilots showed

PIP pilots: half domain, collisions only, first order, run on the workstation.

**Structure at t = 100**, from the sheet outwards (pilot: [0, 150], 2048 cells, dx = 0.073):
- **Outflow region, x ≲ 20.** Hot and evacuated: ρ\_n ≈ 0.02–0.1, v\_y ≈ −5.
- **Slow-mode shock, x ≈ 19–24.** This is the structure to resolve. Neutral density jumps from about 0.1 to 1. The neutrals change over several units, and the plasma has a sharp sub-shock near x ≈ 23, with a plasma density spike just ahead of it.
- **A purely neutral discontinuity at x ≈ 64.** Neutral density drops from 2.1 to 1 there.

**Shock width** (10–90%, measured by `scripts/collonly_compare.py` in a window of ±6 around the steepest rise of ρ\_n):

| Field | Width (x units) | Cells at dx = 0.073 | dx = 0.04 | dx = 0.02 |
| --- | --- | --- | --- | --- |
| B\_y | 2.3 | 31 | 57 | 114 |
| ρ\_n | 2.0–3.0 | 28–41 | 50–75 | 100–150 |
| v\_y, neutral | 3.4–4.3 | 47–58 | 85–107 | 170–210 |
| T\_n | 3.6–4.4 | 49–60 | 90–110 | 180–220 |
| v\_y, plasma | 1.3–1.4 | 18–19 | 33–35 | 65–70 |
| v\_x, plasma (sub-shock) | 0.5 | 7 | 13 | 25 |

The ranges come from two ways of picking the upstream and downstream values. These widths were measured at dx = 0.073 with first-order PIP and may shrink with resolution; the first handoff found PIP's sheet width more than halves with 4× cells. **Re-measure them in the production runs.**

**Recommendation: dx = 0.02.** That gives about 100 cells across the narrowest measure (ρ\_n) and margin if the widths shrink. dx = 0.04 meets the minimum only just.

**The domain has to be very large.** The plasma alone carries a fast wave at its Alfvén speed (about 19.4). Collisions damp it but don't stop it, so a small precursor spreads far ahead of everything else. With L = 150, the outer boundary is already disturbed at t = 10. By t = 100 there is a real flow there (v\_x,n ≈ −0.02; changes up to 0.24 in the outermost 1%). Coarse PIP pilot on [0, 2500] at t = 100 (largest relative change since t = 0, worst field):

| Level | 10⁻³ | 10⁻⁶ | 10⁻⁹ | 10⁻¹² |
| --- | --- | --- | --- | --- |
| Reaches x ≈ | 850 | 1260 | 1520 | 1710 |

**Recommendation: L = 2000.** The boundary then stays below about 10⁻¹². L = 1300 is the cheaper option, with about 10⁻⁶ at the boundary.

## Cost and how to run it fast

Measured on the workstation (single core):

| Code | Cost | Effective signal speed (sets dt = 0.8 dx / c) |
| --- | --- | --- |
| PIP | 1.8×10⁶ cell-steps/s | c ≈ 19 |
| SAMS | 5×10⁻⁵ s per x-cell per step (explicit sources) | c ≈ 25 early, falling towards 19 |

A "cell" for SAMS means one x-position. Its arrays are 3D with ny = nz = 2 plus ghost layers, which is why it costs about 25× more per cell than PIP.

| dx | L | Cells | PIP, 1 core | SAMS, 1 core | SAMS, 64 ranks (if it scales at 75%) |
| --- | --- | --- | --- | --- | --- |
| 0.02 | 2000 | 100,000 | 1.8 h | **7–8 days** | ~4 h |
| 0.02 | 1300 | 65,000 | 1.2 h | 5 days | ~2.5 h |
| 0.04 | 2000 | 50,000 | 25 min | 2 days | ~1 h |
| 0.04 | 1300 | 32,500 | 15 min | 1.3 days | ~40 min |

**SAMS needs MPI.** OpenMP threads don't help:
- **No speed-up.** 4 threads ran no faster than 1, probably because the OpenMP backend (`include/pp/OpenMPUnrolledBackend.h`) uses `schedule(dynamic)` on very small loop bodies.
- **Crashes at 12 threads.** At 5000 cells it crashes with the default thread stack, and once crashed even with `OMP_STACKSIZE=256M`.

Use **`OMP_NUM_THREADS=1` and MPI ranks**. Checked on the workstation (collisions-only, L = 150, 2048 cells, to t = 1):
- **SAMS, 2 ranks:** bit-identical to serial, 1.75× faster.
- **PIP, 2 ranks:** bit-identical to serial. PIP splits `ix` (the total) over `mpi_x` ranks.

How SAMS scales beyond 2 ranks is unknown; test it first (step 4 below).

## Step by step

### 1. Build

**SAMS:**

```
git pull                      # branch TwoFluidWIP
cmake . -DMPI=ON -DCMAKE_BUILD_TYPE=Release    # re-run cmake: new source files since 24e7861
make -j
```

- MPI needs a PETSc built with MPI. A serial PETSc (MPIUNI) forces `-DMPI=OFF`.
- With explicit sources (below) PETSc is not used at run time, but it is still needed to build.

**PIP:** get a patched source tree. Take PIP `khinvel` at `0474b7a` and apply `pip_snow_hillier_2021.patch` and then `pip_half_domain.patch`; the two patch files are in the SAMS repo root. Don't edit it: `scripts/pip_collonly_run.sh` copies the tree and makes all the edits. It needs `h5pfc` and `mpirun`.

### 2. Smoke test (a few minutes; checks both builds against the workstation)

PIP, L = 150, 2048 cells, to t = 1, 2 ranks:

```
TEND=1.0 DTOUT=0.1 scripts/pip_collonly_run.sh <patched PIP dir> 150.0 2048 t1 1 2
```

SAMS: set up the run as in step 5, with `t_end = 1.0`, `x_half = 150.0` and `nx = 4096`. Run it on 2 ranks, then compare:

```
python3 scripts/collonly_compare.py --pip pip_t1/out --sams <run dir>/SAMS_testdata --snap 10 --out smoke
```

Expected values from the workstation:

| | PIP | SAMS |
| --- | --- | --- |
| Time | 1.0013 | 1.0018 |
| Steepest rise of ρ\_n | x = 1.6479 | x = 1.2817 |
| Plateau just outside it: v\_x,p | −0.63967 | −0.67866 |
| Same plateau: T\_p | 0.163532 | 0.162646 |

SAMS should take 1072 steps. Any difference means a different build or set-up.

### 3. PIP reference runs (cheap: run these first)

```
scripts/pip_collonly_run.sh <patched PIP dir> 2000.0 100000 L2000_dx002 1 16   # production, first order
scripts/pip_collonly_run.sh <patched PIP dir> 2000.0 100000 L2000_dx002_o2 2 16 # second order
scripts/pip_collonly_run.sh <patched PIP dir> 2000.0 50000  L2000_dx004 1 16    # resolution check
```

Arguments: patched source, L, total cells, name, spatial order, MPI ranks. The script prints the settings it applied; check them. Snapshots go to `pip_<name>/out/t0000.h5` … `t0010.h5` (every 10 time units).

The script sets:
- **`shock_tube_ion.f90`:** `half_domain = .true.`, `sharp_interface = .true.`, `end(1) = L`.
- **`setting.txt`:** `ix`, `tend = 100`, `dtout = 10`, `flag_IR = 0`, `flag_col = 2`, `col = 1.0`, `s_order`, `mpi_x`.

Before spending time on SAMS, check each PIP run with the comparison script's PIP-only mode:

```
python3 scripts/collonly_compare.py --pip pip_L2000_dx002/out --snap 10
```

It prints the shock position, the 10–90% widths in x and in cells, and the boundary disturbance. Check that:
- the boundary disturbance is tiny;
- the shock width is at least 50 cells at dx = 0.02;
- dx = 0.04 and dx = 0.02 agree. If they don't, the shock isn't converged and the cell count matters more.

On the workstation's L = 150 pilot it correctly flags the boundary as disturbed (changes up to 0.24 in the outermost 1%).

### 4. SAMS scaling test

Before the production run, measure SAMS's speed on 8, 32 and 64 ranks:
- Use the production nx and L with a short `t_end` (e.g. 0.5).
- Compare wall time per step (`Starting timestep N at time t, dt = …` lines, one every 10 steps).
- Check one case against a serial run of a small grid for bit-identity (as the workstation test did).
- Each rank must get the same number of cells, so make nx/2 divisible by the rank count.

### 5. SAMS production run

Edit **`src/InitialConditions/TwoFluidTests/TwoFluidTest.h`** (the `sh21` struct). **`nx` is the full-domain count; the half domain uses nx/2 cells.**

```
const std::string problem = "snow_hillier_2021";
bool coupling = true;
bool ionisation_recombination = false;   // collisions only
bool half_domain = true;
SAMS::T_dataType sheet_width_cells = 0.0; // sharp sheet
SAMS::T_dataType t_end = 100.0;
SAMS::T_dataType x_half = 2000.0;         // L
SAMS::T_indexType nx = 200000;            // = 2 L / dx  (dx = 0.02)
```

Edit **`src/PhysicsPackages/TWOFLUID/twofluid.h`**:

```
bool implicit=false;   // collisions are not stiff at this dt (rate ~ alpha rho << 1/dt)
```

The explicit path applies the collisional time-step limit itself.

Leave **`TwoFluidTest.cpp`** as on the branch: `visc1 = 0.1`, `rke = false`, `rke_3d_fix = false`, `energy_budget = false`.
- **Don't use `energy_budget` with MPI:** it sums only the rank's own cells and runs serial host loops.
- **`noise_check`** (on by default) is cheap and useful: it reports any odd–even velocity mode.

Rebuild (`make -j`), then run in a new directory:

```
mkdir run_L2000_dx002 && cd run_L2000_dx002
ln -s <repo>/data data && mkdir SAMS_testdata
OMP_NUM_THREADS=1 mpirun -np 64 <repo>/sams LARE3DST LARE3DNF PIP TwoFluidTest > run.log 2>&1
```

**Check the start-up block:**
- `HALF DOMAIN x in [0, 2000], 100000 cells`
- `COLLISIONS ONLY`
- `collisions on, empirical rates OFF, ... collisions-only model`
- `method : explicit`
- `shock viscosity : visc1 = 0.1`

**Known harmless problem:** after `Simulation complete…`, rank 0 exits through `MPI_Abort`, so `mpirun` reports an error and a non-zero status. The outputs are complete: `diagnostics_00000_rankR.h5` … `diagnostics_00010_rankR.h5` (plus a duplicate final snapshot, 00011). Each rank writes its own x range; the comparison script joins them.

### 6. Comparison and figures

```
python3 scripts/collonly_compare.py --pip pip_L2000_dx002/out --sams run_L2000_dx002/SAMS_testdata --snap 10 --out collonly_t100
```

- **`collonly_t100_full.png`:** density and temperature (log), v\_x, v\_y, gas pressure, B\_y. Neutral is blue and plasma orange; PIP is solid and SAMS dashed. The slow shock is marked. The right edge defaults to where the disturbance falls below 10⁻³; use `--xmax 120` to show only the main structure.
- **`collonly_t100_shock.png`:** a window around the slow shock, from 2.5 to 1.5 neutral-v\_y widths either side (or set it with `--window x0 x1`). It shows both fluids' density, temperature, v\_x and v\_y; the ion–neutral drift v\_p − v\_n (x and y); and B\_y.
- **Printed numbers:**
  - shock position in both codes and their difference in cells;
  - 10–90% widths in x and in cells for ρ\_n, neutral v\_y, T\_n, B\_y, plasma v\_y and ρ\_p;
  - the largest relative change in the outermost 1% of the domain for each code (the "nothing reaches the boundary" check);
  - plateau values between the sheet and the shock, and just outside it.

The script plots SAMS velocities at their true vertex positions without averaging, so any grid-scale noise stays visible. Run it for other snapshots with `--snap 1` … `--snap 9` (t = 10 … 90).

Suggested acceptance, following the first handoff:
- **Boundary:** the change in the outermost 1% stays below about 10⁻⁶ (L = 1300) or 10⁻¹⁰ (L = 2000).
- **Resolution:** at least 50 cells across the ρ\_n width in both codes.
- **Agreement:** shock positions within a few cells, and plateaus within a few percent. Compare the shock *structure* against both PIP orders, since PIP's own sheet structure depends on order and resolution.

## Things to know

- **Early near-sheet differences.** At t = 1 on the coarse smoke-test grid, SAMS and PIP differ by up to 30–40% near the sheet (v\_x,n, T\_n). The shock is only 1–4 cells wide there, so this is not a verdict. It is why the resolution requirement matters.
- **Time step.** It starts small (start-up ramp, then the hot sheet) and settles near the plasma CFL value after t ≈ 1.
- **PIP builds** need a serial `make` (Fortran module order). The script does this.
- **SAMS snapshots never write vertex 0** (x = 0). The script adds v\_x = 0 there by symmetry; v\_y at x = 0 is missing.
- **Snapshot timing.** Both codes write at the first step at or after each output time, so their times differ by up to one step (printed).
- **Stack.** If OpenMP threads are used anyway, set `OMP_STACKSIZE=256M` or larger: large nx overflows the default thread stack. Even then a 12-thread run crashed once on the workstation (not investigated).
- **Pilot outputs on the workstation:**
  - `runs/pip_col_src/build_pilot` (L = 150), `runs/pip_col_src/build_big` (L = 2500, dx = 0.31);
  - `runs/collonly/sams_t1`, `runs/collonly/sams_t1_mpi2` (SAMS smoke test, serial and 2 ranks).

## Status update, 6 Nov 2026 (14:30): MPI fixed, production runs started

This section records where things stand on the workstation (16 physical cores, 2 × 8) and what is left to do.

### Objectives (unchanged)

- Compare SAMS and PIP on the collisions-only Snow & Hillier (2021) current sheet to **t = 100**, half domain **L = 2000**, at **dx = 0.02** (main) and **dx = 0.04** (resolution check).
- Requirements: nothing reaches the outer boundary, and **≥ 50 cells across the shock** (ρ\_n width).
- Deliverables: the full comparison figure and the slow-shock close-up (`scripts/collonly_compare.py`).

### What changed today

**SAMS MPI bug found and fixed: commit `7740c79`** ("Fix off-by-one in MPI halo exchange for staggered axes", `include/harness/mpiManager.h`).

- **Cause:** `assignVariableMPITypes` sent the shared boundary vertex of a staggered (HALF\_CELL) axis as the neighbour's first ghost. Every vertex-centred array (v\_x, v\_y, v\_z and their `*1` copies) therefore had ghosts shifted one vertex towards the rank boundary. Nothing showed until flow reached a rank boundary, then the errors were order 1.
- **Fix:** the send window moves inward by `getGhostAdjust(stagger)` on the exchanged axis. Cell-centred arrays are unchanged.
- **Scope:** this is a harness bug, so it affected **every SAMS package with staggered variables run on more than one rank**, not only TWOFLUID. Earlier multi-rank SAMS results should be treated as suspect.
- **Checks:**
  - Smoke test (L = 150, 16 ranks, t = 1): bit-identical to serial in all snapshots and all 33 datasets, with the same 1072 steps. Before the fix it was off by 0.3.
  - Hard case (16 ranks, a rank boundary every 1.17, so the shock crosses many of them, t = 0.6): bit-identical.
  - Control: a serial run with B\_x perturbed by 10⁻¹⁵ stays within 10⁻¹². The old MPI divergence was a genuine bug, not chaos.
  - The earlier "2 ranks are bit-identical" check in the handoff presumably passed only because the flow had not reached the rank boundary (x = 75) by t = 1.

**SAMS scaling** (production dx = 0.02 grid, 100,000 cells, `OMP_NUM_THREADS=1`, split in x only, measured from step 10 to step 50 while PIP occupied 8 cores):

| Ranks | Wall time per step | Speed-up vs 4 ranks |
| --- | --- | --- |
| 4 | 3.33 s | 1 |
| 8 | 1.85 s | 1.80 (90 % efficiency) |
| 16 (L = 150 smoke test) | 9.4× faster than serial | |

The serial production run had averaged about 12.8 s/step, so the earlier "7–8 days on 1 core" estimate was optimistic: serial would have taken about 2.5 weeks.

**Serial SAMS production runs stopped.** `runs/sams_L2000_dx002` and `runs/sams_L2000_dx004` were killed at t ≈ 0.024 and t ≈ 0.05. Their directories are kept but are superseded.

### Runs

**PIP reference runs: all finished** (`runs/pip_refs.sh`, 8 ranks each, log `runs/pip_refs.log`):

| Run | Order | Wall time | Output |
| --- | --- | --- | --- |
| `pip_L2000_dx002` | 1 | 40 min | `runs/pip_L2000_dx002/out` |
| `pip_L2000_dx002_o2` | 2 | 93 min | `runs/pip_L2000_dx002_o2/out` |
| `pip_L2000_dx004` | 1 | 8 min | `runs/pip_L2000_dx004/out` |

**SAMS production runs: running**, started 14:26 with the fixed MPI build:

| Run | Cells | Ranks | Early rate | Expected finish |
| --- | --- | --- | --- | --- |
| `runs/sams_L2000_dx002_mpi10` | 100,000 | 10 | ~1.5–1.7 s/step | ~2–2.5 days (≈ 120k steps) |
| `runs/sams_L2000_dx004_mpi5` | 50,000 | 5 | ~1.5 s/step | ~1 day (≈ 60k steps) |

The expected finishes are estimates from the first ~100 steps, assuming dt settles near 0.8 dx / 19. Re-estimate once t > 1.

- **Launch:** `runs/launch_sams_mpi.sh <name> <binary> <np>`. It runs `mpirun` under `nohup` with `SAMS_DECOMPOSITION=0,1,1`, `OMP_NUM_THREADS=1` and `--bind-to none`, copies the binary into the run directory and refuses to overwrite an existing one.
- **Logs:** `<run>/run.log`, with every line timestamped. `/usr/bin/time` appends the total wall time at the end.
- **Output:** `<run>/SAMS_testdata/diagnostics_000NN_rankR.h5`, joined by the comparison script.
- **Progress:** `grep "Starting timestep" runs/sams_L2000_dx002_mpi10/run.log | tail -1`
- **Binaries:** `runs/scaling/sams_dx002`, `runs/scaling/sams_dx004`. The scaling-test logs are in `runs/scaling/np4`, `np8` and `results.txt`.
- **Expected noise in the logs:** "noise GREW" lines come from the TWOFLUID checkerboard diagnostic (`noise_check`). The serial runs print the same values.
- **Expected exit error:** rank 0 exits through `MPI_Abort`, as in the handoff.

### Uncommitted changes (branch `TwoFluidWIP`)

| File | Change | Suggested action |
| --- | --- | --- |
| `include/runner/runner.h` | `SAMS_DECOMPOSITION="nx,ny,nz"` environment override for the MPI decomposition (0 = automatic). Unset keeps the old behaviour. | Commit on its own |
| `scripts/collonly_compare.py` | Falls back to `diagnostics_NNNNN.h5` for serial runs | Commit |
| `src/InitialConditions/TwoFluidTests/TwoFluidTest.h` | Production set-up: `ionisation_recombination = false`, `t_end = 100`, `x_half = 2000`, `nx = 200000` | Run configuration; commit or keep local |
| `src/PhysicsPackages/TWOFLUID/twofluid.h` | `implicit = false` (explicit sources) | Run configuration |

The grid is compile-time. The dx = 0.04 binary was built with `nx = 100000`, and the source was then set back to `nx = 200000`.

### Next steps

1. **Monitor the SAMS runs.** Check that dt settles near the CFL value and refresh the finish estimate after t ≈ 1.
2. **Check intermediate snapshots** against PIP as they appear, starting with dx = 0.04 at t = 10 (`--snap 1`). This catches problems early instead of after two days.
3. **Run the comparison** when the runs finish (handoff step 6):
   ```
   python3 scripts/collonly_compare.py --pip runs/pip_L2000_dx002/out --sams runs/sams_L2000_dx002_mpi10/SAMS_testdata --snap 10 --out collonly_t100
   ```
   Repeat against `pip_L2000_dx002_o2` and against the dx = 0.04 pair.
4. **Re-measure the shock widths** at dx = 0.02 and 0.04 in both codes, and confirm ≥ 50 cells across ρ\_n and a boundary change below 10⁻¹⁰.
5. **Commit** `SAMS_DECOMPOSITION` and the compare-script fix. Decide whether the production set-up in `TwoFluidTest.h` and `twofluid.h` is committed.
6. **Flag the halo fix to other SAMS users.** Any multi-rank run with staggered variables before `7740c79` may be wrong.
