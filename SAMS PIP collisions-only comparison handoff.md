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

## Status update, 9 Nov 2026 (18:30): production runs compared; `rke` test running

### Production runs: finished

| Run | Finished | Steps | Wall time |
| --- | --- | --- | --- |
| `runs/sams_L2000_dx004_mpi5` | 7 Nov 17:57 | 61,660 | 27.5 h |
| `runs/sams_L2000_dx002_mpi10` | 8 Nov 17:33 | 122,810 | 51 h |

- Both logs end with "Simulation complete". The non-zero exit status is the known `MPI_Abort`.
- Snapshots `00000` to `00011` were written for every rank.
- Commit `74eeb57` added the uncommitted changes listed in the 6 Nov update, production set-up included. The source tree on the branch is the t = 100, dx = 0.02 production configuration.

### Comparison at t = 100: SAMS and PIP disagree

The outputs are in `runs/compare_t100/`:
- `dx002_o1_*`: dx = 0.02 against first-order PIP.
- `dx002_o2_*`: dx = 0.02 against second-order PIP.
- `dx004_o1_*`: dx = 0.04 against first-order PIP.
- `evo/s1` … `evo/s9`: dx = 0.02 against first-order PIP at t = 10 … 90.

Each comparison has a `.txt` file with the printed numbers, `_full.png` and `_shock.png`.

| Check | PIP | SAMS | Verdict |
| --- | --- | --- | --- |
| Boundary, outermost 1% (largest relative change) | ≤ 3.5×10⁻¹⁰ | ≤ 1.3×10⁻¹², except B\_y 6.8×10⁻⁸ | Passes 10⁻⁶; misses the 10⁻¹⁰ target (PIP T\_p slightly, SAMS B\_y) |
| Disturbance of 10⁻³ reaches | x = 856 | x = 858 | Agree |
| ρ\_n shock width, dx = 0.02 | 1.72 (86 cells) | 0.38 (19 cells) | SAMS < 50 cells: its neutral jump is much sharper |
| Slow-shock position | 22.13 | 17.97 | **Fails: −4.2, or −208 cells** |
| Plateau between sheet and shock | | T −33%, ρ\_n +45%, B\_y +26% | **Fails** |
| Plateau outside the shock | | ρ within 3%; T −11%; neutral v\_x −17%; v\_y up to −14% | Partly fails |

**What is robust:**
- **Not resolution or PIP order.** SAMS's shock sits at 17.82 (dx = 0.04) and 17.97 (dx = 0.02). PIP's sits at 22.06–22.15 in first order, second order and both resolutions.
- **From the first snapshot on.** SAMS's shock position is 0.81–0.83 of PIP's at every snapshot from t = 10 to 100:

  | t | PIP shock | SAMS shock | Inner-plateau T\_p, SAMS vs PIP |
  | --- | --- | --- | --- |
  | 10 | 3.91 | 3.23 | −19% |
  | 50 | 13.53 | 11.15 | −29% |
  | 100 | 22.13 | 17.97 | −33% |

- **Same structure, shifted.** The close-up shows the same features in both codes, moved towards the sheet: the plasma sub-shock, the plasma temperature spike just ahead of it, and an ion–neutral v\_y drift of about −3.4 (SAMS) vs −3.2 (PIP).

**Working hypothesis:** SAMS puts too little heat into the hot outflow region. Its pressure is then lower, so the slow shock advances more slowly.

**Suspects:**
1. The production runs have `rke = false`. The kinetic energy lost in the remap is never returned as heat, and the remap also dissipates magnetic energy. The first handoff measured about 4% of the released energy at t = 0.01; that may not be enough on its own.
2. Collisional (frictional) heating differs between the codes. Check the source-term energy exchange against PIP.

### Test running now: `rke` on (started 9 Nov 18:24)

**Run:** `runs/sams_L2000_dx004_rke_t20`, 10 ranks, launched with `runs/launch_sams_mpi.sh`.

**Settings:**
- Production dx = 0.04 set-up: L = 2000, `nx = 100000`, collisions only, explicit sources, `visc1 = 0.1`.
- `rke = true` and `rke_3d_fix = true`, for both plasma and neutrals.
- `t_end = 20`, with snapshots every 10, so snapshots 1 and 2 are t = 10 and 20, as in PIP and the rke-off baseline `runs/sams_L2000_dx004_mpi5`.
- The start-up block confirms the `rke` flags, the explicit method and `visc1 = 0.1`.

**Expected finish:** about 21:00 on 9 Nov. That is about 12,900 steps at about 0.7 s/step.

**Binary:** `runs/scaling/sams_dx004_rke_t20`. It was built from temporary edits to `TwoFluidTest.h` and `TwoFluidTest.cpp` (`t_end`, `nx`, `rke`, `rke_3d_fix`, `dt_snapshots = 10`).
- **Source:** the edits were reverted with `git checkout` straight after the build, so the source tree is clean.
- **`build-mpi/sams`:** still holds this test configuration. Rebuild before using it for anything else.

**Monitor:**

```
grep "Starting timestep" runs/sams_L2000_dx004_rke_t20/run.log | tail -1
tail -3 runs/sams_L2000_dx004_rke_t20/run.log        # "... s wall" means it has finished
```

**When it has finished:**

```
runs/rke_test_compare.sh
```

This writes `runs/compare_rke/`, comparing rke on and rke off with PIP at t = 10 and 20 (figures up to x = 120). It prints the shock position and the inner-plateau ρ\_n, T and B\_y for each run.

**How to read it:**
- **`rke` closes most of the gap** (shock position and inner T near PIP's): rerun the dx = 0.02 production run with `rke` on. That needs the LARE developers to agree to `rke_3d_fix`, which is still experimental.
- **Little change:** `rke` is ruled out. Next, compare the collisional heating and momentum-exchange terms with PIP's directly.
- **New `noise GREW` lines or broken energy identities** that the baseline log doesn't have: `rke_3d_fix` misbehaves under MPI. The ghost refresh after the correction has not been tested on more than one rank.

### Next steps

1. Evaluate the `rke` test (above).
2. Depending on the result, either a dx = 0.02 production run with `rke`, or a check of the source-term energy exchange against PIP.
3. Still open from 6 Nov: flag the MPI halo fix (`7740c79`) to other SAMS users.

## Status update, 9 Nov 2026 (18:40): runs queued for 10–15 Nov (machine unattended)

Three things run under nohup while the machine is unattended. Everything is scripted, so nothing needs a terminal.

| Job | What | Ranks | Started | Expected |
| --- | --- | --- | --- | --- |
| `rke` test | `runs/sams_L2000_dx004_rke_t20` (previous section) | 10 | 9 Nov 18:24 | done ~21:00 on 9 Nov |
| dx = 0.02 production with `rke` | `runs/sams_L2000_dx002_rke_mpi10`, started automatically by `runs/chain_dx002_rke.sh` when the `rke` test's log shows `s wall` | 10 | ~21:00 on 9 Nov | ~51 h, so ~12 Nov |
| Coupling scan | `runs/scan/`, SAMS and PIP at five values of α0 | SAMS 5, PIP 4 | 9 Nov 18:30 | PIP within an hour; SAMS one run after another, probably 1–2 days |

At most 15 SAMS ranks run at once on the 16 physical cores.

### Why: the dx = 0.02 production run with `rke`

This run was started before the `rke` test's result was known. It only matters if `rke` closes most of the 19% shock lag. If it does, this is already the production run to compare (handoff step 6, with `--sams runs/sams_L2000_dx002_rke_mpi10/SAMS_testdata`). If it doesn't, discard it.
- **Settings:** binary `runs/scaling/sams_dx002_rke`, the committed production set-up plus `rke = true` and `rke_3d_fix = true`, t = 100, snapshots every 10.
- **Monitor:** `cat runs/chain_dx002_rke.log`, then `grep "Starting timestep" runs/sams_L2000_dx002_rke_mpi10/run.log | tail -1`.

### Why: the coupling scan

**Question:** is the 19% shock lag (SAMS's slow shock at 0.81–0.83 of PIP's distance from the sheet) caused by the MHD/hydro solver, or by the two-fluid coupling?

**How the α0 values answer it:**
- **α0 = 0 (uncoupled):** each fluid is pure (M)HD and the neutrals stay uniform. Any difference is LARE vs PIP's HLLD on an MHD Riemann problem.
- **α0 = 100:** the two fluids move almost as one, so the system is close to single-fluid MHD with the total density.
- **Self-similar extremes:** at both limits the solution becomes self-similar, so an exact MHD Riemann solver for the same initial state is an independent reference. It hasn't been written yet.

**How to read the outcome:**
- **The lag is already there at α0 = 0:** look at LARE: the remap, the shock viscosity, the energy loss found in the first handoff.
- **The lag is absent at α0 = 0 and at 100 but present at intermediate α0:** look at the collision source terms, especially frictional heating and thermal exchange, and compare them term by term with PIP's.

**Set-up (both codes):**
- dx = 0.04, half domain L = 500 (12,500 cells), t = 20, snapshots at t = 10 and 20.
- Collisions only; α0 ∈ {0, 0.1, 1, 10, 100}.
- α0 = 1 repeats the production case on this grid as a control. It should match `runs/sams_L2000_dx004_mpi5` and `runs/pip_L2000_dx004` at t = 10 and 20.

**SAMS:**
- **Binaries:** `runs/scaling/sams_scan_a{0,0.1,1,10,100}`, built by `runs/scan/build_variant.sh`. That script edits the `sh21` settings temporarily, builds in `build-mpi`, then restores the source from git.
- **α0 = 0** uses `sh21.coupling = false`.
- **Run directories:** `runs/scan/sams_a<α0>/`, run in the order 1, 0.1, 10, 100, then 0.
- **Queue:** `runs/scan/sams_scan.sh`, with its log in `runs/scan/sams_scan.log`. It prints `SAMS_SCAN_DONE` at the end.
- **Uncoupled case moved to the end:** its time step kept falling (5×10⁻⁵ by t = 0.04, about 30× below the coupled run's settled value), so it was stopped after 400 steps and re-queued last.
  - It now runs from `runs/scan/sams_a0_last.sh` (log `sams_a0_last.log`, ends with `SAMS_A0_DONE`) once `SAMS_SCAN_DONE` appears.
  - The stopped attempt is in `runs/scan/sams_a0_aborted_reordered`; ignore it. Its `sams_scan.log` entry says "a0 done", but that run was stopped, not finished.
  - If the time step stays that small, this run alone takes a day or more. It may not finish by Monday.
- **Time step, α0 = 100:** check the explicit collisional limit doesn't make it impractically small.

**PIP:**
- **Run directories:** `runs/scan/pip_a<α0>/`, run by `runs/scan/pip_scan.sh` (log `pip_scan.log`).
- **Uncoupled case:** PIP crashes with `col = 0` (floating-point exception in the prologue, `Initial_rot.f90:22`). It uses **`col = 1e-8`** instead, run by `runs/scan/pip_a0_retry.sh` (directory `pip_a1e-8`). α·ρ·t ≈ 2×10⁻⁷ by t = 20, so that is effectively uncoupled.
- **Script change:** `scripts/pip_collonly_run.sh` now takes `COL=<α0>` from the environment, defaulting to 1.0.

**Evaluate:**

```
runs/scan/scan_compare.sh      # writes runs/scan/compare/a<α0>_s{1,2}{.txt,_full.png,_shock.png}
```

It prints the slow-shock positions and the inner-plateau ρ\_p, T\_p and B\_y for each α0. The table to build from it is SAMS shock position / PIP shock position against α0.
- **α0 = 0:** the script finds "the slow shock" from the steepest rise of ρ\_n, but the neutrals are uniform when uncoupled. Ignore the printed position and widths; use the plasma profiles in the figures (`--window` to zoom).
- **α0 = 0.1:** the neutral signal may be weak; check the figure as well.

### Checks on Monday

1. `tail -3 runs/sams_L2000_dx004_rke_t20/run.log`, then `runs/rke_test_compare.sh`. The result decides whether `runs/sams_L2000_dx002_rke_mpi10` is the new production run.
2. `cat runs/scan/pip_scan.log runs/scan/pip_a0_retry.log runs/scan/sams_scan.log runs/scan/sams_a0_last.log`. Each run should end with `exit=0` (PIP) or a `s wall` line (SAMS). A SAMS exit status of 1 is the known `MPI_Abort`; check for "Simulation complete".
3. `runs/scan/scan_compare.sh`, then decide whether the problem is in LARE or in the sources (above).
4. If useful, write the exact MHD Riemann solver for the α0 = 0 and α0 → ∞ limits.

## Status update, 16 Nov 2026: results of the unattended runs

### The disk filled on 11 Nov

`/` reached 100% on 11 Nov at 13:35. The runs alone don't account for it (`runs/` was 17 GB of 228 GB); the disk was probably close to full already. Two runs were hit:
- **`runs/sams_L2000_dx002_rke_mpi10`** crashed at t ≈ 70 (step 86,160) with HDF5 `No space left on device` while writing snapshot `00007`. Snapshots 0–6 (t = 0–60) are good. It won't be restarted (see `rke` below).
- **`runs/scan/sams_a0`** reached t = 20 on 12 Nov, after 52 h and 596k steps with dt ≈ 3.4×10⁻⁵. Its t = 20 snapshot (`00002`) is truncated and unreadable. Only t = 10 can be used.

Everything else finished cleanly: the `rke` test, SAMS α0 = 0.1, 1, 10 and 100, and every PIP scan run.

### Data moved to the data drive

To free space, `runs/` and `hillier_shock_beta_0.5/` were moved to `/media/snow/datadrive2/SAMS_TwoFluidWIP/` (NTFS). Both are now symlinks in the repo.
- **Paths:** full paths in the scripts still resolve. `runs/rke_test_compare.sh` had to be changed, because it found the repo through `runs/..`, which now points at the data drive; it now uses the full path to `scripts/collonly_compare.py`.
- **Repo-root `SAMS_testdata`:** its symlink pointed at `/media/snow/store21`, which no longer exists. It now points at `/media/snow/datadrive2/SAMS_TwoFluidWIP/SAMS_testdata`.
- **New runs should go under `runs/`,** so they land on the data drive.

### `rke` test: ruled out

dx = 0.04, distance between the slow shocks (SAMS − PIP, in SAMS cells), and plateau temperatures relative to PIP. Output in `runs/compare_rke/`.

| t | Shock, `rke` off | Shock, `rke` on | T\_p off / on | ρ\_n off / on |
| --- | --- | --- | --- | --- |
| 10 | −19 | −18 | −16.7% / −15.2% | +15.7% / +14.9% |
| 20 | −25 | −24 | −22.1% / −20.7% | +26.1% / +24.7% |

`rke` closes only about 1.5 percentage points of the temperature deficit and one cell of the lag. **It isn't the cause.** The dx = 0.02 `rke` run is dropped.

### Coupling scan: the lag is there with no coupling and grows with coupling

Ratio of the SAMS slow-shock position to PIP's, and the temperature of the inner plateau, from `runs/scan/scan_compare.sh` (output in `runs/scan/compare/`):

| α0 | Shock ratio, t = 10 | Shock ratio, t = 20 | T\_p plateau vs PIP, t = 10 / 20 |
| --- | --- | --- | --- |
| 0 (PIP: col = 10⁻⁸) | **0.90** | – (snapshot truncated) | −10% / – |
| 0.1 | 0.86 | 0.85 | −9% / −12% |
| 1 (production) | 0.81 | 0.85 | −17% / −22% |
| 10 | 0.78 | 0.73 | −38% / −48% |
| 100 | 1.00 (?) | 0.63 | −74% / −63% |

**α0 = 0 in detail:** the script's ρ\_n shock finder is meaningless here, so these numbers were measured on the plasma, at the steepest drop of ρ\_p.

| t = 10 | PIP | SAMS | SAMS vs PIP |
| --- | --- | --- | --- |
| Slow-shock position | 8.06 | 7.26 | 0.90 |
| Plateau ρ\_p | 6.258×10⁻³ | 6.881×10⁻³ | +10.0% |
| Plateau T\_p | 59.49 | 53.71 | −9.7% |
| Plateau pressure (∝ ρT) | | | −0.7% |
| Plateau v\_y | −18.91 | −18.88 | same |
| Upstream v\_x | −1.20 | −1.27 | SAMS inflow 6% faster (with ripples) |
| T\_p at the sheet (first cell) | 106 | 66 | |

**What this shows:**
- **SAMS's pure-MHD solution is already wrong.** With no coupling, the problem is a single-fluid MHD Riemann problem, and SAMS's slow shock still trails PIP's by 10%. The plateau pressure matches to under 1%, since it is set by pressure balance with the upstream field. SAMS's plateau, though, is denser and cooler: **its slow shock produces too little entropy.** A higher compression ratio for the same mass flux means a slower shock.
- **Coupling adds a second error.** The deficit grows steadily with α0. At α0 = 100 the SAMS plasma barely heats (plateau T ≈ 1.1 against PIP's 3–4.5). Look at frictional heating and thermal exchange once the MHD part is understood.
- **SAMS also differs elsewhere.** It has v\_x and B\_y ripples behind the shock that PIP doesn't, and PIP's sheet peak is much hotter (106 against 66 in the first cell).
- **Anomaly at α0 = 100:** SAMS's shock sits at x = 1.02 at both t = 10 and t = 20, as if stalled, while PIP's moves from 1.02 to 1.62. Check this separately.

**Caveat:** PIP is the reference here by assumption. The exact Riemann solution (next) decides which code is right.

### Exact Riemann solution: PIP is right, SAMS loses 4.7% of the energy at the slow shock

**Solver:** `scripts/mhd_riemann_exact.py` computes the exact self-similar solution of the symmetric sheet in ideal 1D MHD and compares it with SAMS and PIP. Output is in `runs/exact_riemann/`.

**Structure, for x > 0:**
- From the sheet outwards: inner state (B\_y = 0, v\_x = 0), then a switch-off slow shock, then state 1, then a fast rarefaction, then the initial state.
- By symmetry no rotational discontinuity is needed, because B\_y passes through zero.
- The one unknown, ρ1, is fixed by v\_x = 0 behind the shock.

**Limits:**
- `--limit uncoupled`: the plasma alone, ρ = ξ\_i, P = P\_p.
- `--limit coupled`: one fluid with ρ = 1 and P = P\_t.

```
cd runs/exact_riemann
python3 /home/snow/Documents/SAMS/TwoFluidWIP/scripts/mhd_riemann_exact.py --limit uncoupled \
    --pip ../scan/pip_a1e-8/out --sams ../scan/sams_a0/SAMS_testdata --snap 1 --out exact_a0_s1 --xmax 12
```

Use absolute paths to scripts from inside `runs/`, because `runs/..` resolves to the data drive.

**α0 = 0 at t = 10:**

| | Exact | PIP | SAMS |
| --- | --- | --- | --- |
| Slow shock x | 7.977 | 8.06 (1.010) | 7.26 (**0.910**) |
| Inner ρ\_p | 6.266×10⁻³ | −0.13% | **+9.8%** |
| Inner T\_p | 59.41 | +0.13% | **−9.6%** |
| Inner P\_p | 0.4467 | 0.00% | −0.74% |
| State 1: ρ, v\_x, B\_y | 2.498×10⁻³, −1.203, −0.9386 | matches | ρ and B\_y match; v\_x ≈ −1.27 with ±0.1 grid-scale oscillations |
| Compression ρ2/ρ1 | 2.508 | 2.507 | 2.768 |
| Fast rarefaction x = 177–195 | | matches (smoothed) | matches |

**Rankine–Hugoniot check of each code's own states across its slow shock:** shock speed from mass conservation; relative flux mismatch, out minus in.

| | Mass | x-momentum | y-momentum | Induction | **Energy** |
| --- | --- | --- | --- | --- | --- |
| PIP | 0 | −2×10⁻⁶ | 2×10⁻⁴ | −2×10⁻⁴ | −1.6×10⁻⁴ |
| SAMS | 0 | −1×10⁻⁴ | −1.1×10⁻³ | 1.1×10⁻³ | **+4.7×10⁻²** |

**Findings:**
- **SAMS's slow shock is a correct switch-off shock in everything except energy.** It moves at the Alfvén speed on B\_x relative to its upstream (u1 = −2.004 against Bx/√ρ1 = 2.006) and conserves mass, momentum and flux. But 4.7% of the energy flux into the shock does not come out.
- **That missing energy accounts for the whole discrepancy.** The missing energy is missing heat: the post-shock pressure is pinned by momentum balance, so the gas ends up denser and cooler, the compression ratio is higher, and the shock is slower. This is the 10% lag at α0 = 0. Coupling makes it worse (19% at α0 = 1).
- **Where the energy goes:** most of the energy through this shock is the conversion of B\_y² into tangential kinetic energy (v\_y: 0 → −18.9) and heat. `rke` (returning the kinetic energy lost in the remap) barely changed the result, so the loss is probably not in the normal-velocity remap. Suspects: the remap of v\_y and B\_y, and how LARE's shock viscosity turns the tangential velocity jump into heat.
- **The coupled limit can't be tested at α0 = 100 by t = 20.** The exact fully coupled shock is at x = 0.47 (t = 10) and 0.93 (t = 20), with ρ2 = 2.09 and T = 0.40. Neither code is near this yet: at t = 20, PIP's neutral plateau is 90% less dense and 10× hotter. The α0 = 100 run is still far from the self-similar coupled state. Testing it needs α0 ≳ 10⁴ or t ≫ 20.

### Next steps (superseded: see the next section)

1. **Find the 4.7% energy loss in LARE.** Use a cheap 1D SAMS MHD-only run (`sh21.coupling = false`, small L, short t) and log the energy budget per step: Lagrangian step, viscous heating, remap, with v\_y and B\_y kinetic and magnetic energy separately. The exact solution and `runs/scan/sams_a0` are the reference. Small L also gets round the tiny dt of the uncoupled run (52 h for L = 500).
2. Then the coupling terms, including the stall at α0 = 100.
3. Still open: flag the MPI halo fix (`7740c79`) to other SAMS users.

## Status update, 16 Nov 2026 (11:40): the lag is in LARE's MHD scheme itself

### Why the uncoupled runs had a tiny dt (bug, not fixed)

- **Symptom:** PIP-mode runs with `sh21.coupling = false` stepped at dt ≈ 3×10⁻⁵ (dx = 0.02 or 0.04). LARE's own CFL step is 8.2×10⁻⁴; this was confirmed with a temporary print in `LARE3DST::set_dt`, since reverted.
- **Cause:**
  - `get_collisional_source_terms` runs on every step, whatever the coupling setting (`TwoFluid.cpp:296`).
  - `apply_two_fluid_source` correctly does nothing when `collisions` is off (`TwoFluid.cpp:329`).
  - But `calculateTimestep` (`twofluid.h:434`) still calls `set_dt_collisional`, which limits dt to 0.1 × 0.8 / max|source|. In the α0 = 0 snapshot max|`source_energy`| = 2.4×10³, which gives 3.3×10⁻⁵.
- **Consequences:** uncoupled runs take about 25× more steps than needed (`runs/scan/sams_a0` took 52 h). The physics is unaffected, apart from the extra remaps.
- **Not fixed: test pure MHD in MHD mode instead** (below). A fix would be one condition: skip `set_dt_collisional` when `!plasma_source.collisions`.

### Pure-MHD run: same error at the correct time step

**Run:** `sams LARE3D Hillier`, with `Hillier.cpp` temporarily set to the plasma-only problem: ρ = ξ\_i = 0.00266, P = P\_p, B\_x = 0.1, B\_y = ±1, γ = 5/3, `visc1 = 0.1`, dt multiplier 0.8, full domain [−25, 25], dx = 0.02, t = 1. It steps at dt = 8.2×10⁻⁴ and takes 146 s on one core.

**Tools** (in `runs/`, which is on the data drive and not in git):
- `runs/mhd_sheet_build.sh <out> <x_half> <nx> <t_end> <dt_snap>`: edits `Hillier.cpp`, builds in `build-mpi`, copies the binary, then restores the source.
- `runs/launch_sams_mpi.sh <name> <binary> <np> "LARE3D Hillier"`: the 4th argument (the package list) is new; it defaults to the two-fluid set.
- `runs/mhd_sheet_compare.py <run>/SAMS_testdata <snap>`: compares with the exact solution, checks Rankine–Hugoniot and writes a figure.

**Things to know about base `LARE3D`:**
- It writes `energy_electron` as well as `energy_ion`. With the Hillier IC the electron energy stays at 0.
- It has no per-operator energy budget; that exists only in `LARE3DST`.

| | Pure MHD, CFL dt (`runs/mhdsheet_L25_dx002_t1`, t = 1) | PIP mode α0 = 0, small dt (`runs/scan/sams_a0`, t = 10) |
| --- | --- | --- |
| Slow shock / exact | **0.915** (0.730 vs 0.798) | 0.910 |
| Inner ρ / T / P vs exact | +9.8% / −10.1% / −1.3% | +9.8% / −9.6% / −0.7% |
| State 1, v\_x (exact −1.203) | −1.29 with oscillations | −1.27 with oscillations |
| Energy flux missing across SAMS's own shock | **4.8%** | 4.7% |
| y-z mode (B\_y deviation, max \|v\_z\|) | 3×10⁻¹⁰, 5×10⁻¹⁰ | 0.077, 0.11 |

**Ruled out:**
- **dt:** the error is the same at the correct step.
- **The PIP package:** pure `LARE3D` gives the same numbers.
- **The y-z mode:** it's absent here, yet the error is unchanged. The strong y-z mode on the shock in PIP-mode α0 = 0 is a separate issue; the coupled runs have it too, but at 10⁻⁵–10⁻⁷.
- **Resolution:** 37 cells between sheet and shock (this run) and 180 cells (`sams_a0`) give the same 0.91.

**Conclusion:** SAMS/LARE gets the energy jump condition of this MHD slow shock wrong by about 5%, at every resolution. The shock's mass, momentum and flux are right, so the post-shock state is too dense and too cool, and the shock moves too slowly. A scheme that conserves energy would get the shock speed right whatever its dissipation, so the fix has to put the dissipated energy back as heat.

### Per-operator budget (`LARE3DST`, PIP mode, uncoupled; small dt)

**Run:** `runs/ebudget_L25_dx002_t1`: the same problem on the half domain [0, 25], dx = 0.02, to t = 1, `data.energy_budget = true`. It ran 11:07–12:06 on 16 Nov (serial; dt ≈ 3.3×10⁻⁵, the bug above).
- **Binary:** `runs/scaling/sams_ebudget_L25_dx002_t1`, from `runs/ebudget_build.sh`.
- **Read the time series:** `python3 /home/snow/Documents/SAMS/TwoFluidWIP/runs/ebudget_table.py runs/ebudget_L25_dx002_t1/run.log`.

Percentages are of the magnetic energy released so far:

| t | E − E0 | Lagrangian step | x-remap KE | x-remap magnetic | y-remap | x-remap KE: vertices 3–10 | 11–50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | −5.86% | +0.08% | −3.16% | −2.69% | −0.09% | −2.09% | −0.08% |
| 0.2 | −5.71% | +0.06% | −3.07% | −2.64% | −0.07% | −2.61% | −0.04% |
| 0.3 | −5.72% | +0.05% | −3.05% | −2.66% | −0.06% | −1.78% | −0.77% |
| 0.4 | −5.63% | +0.05% | −3.00% | −2.63% | −0.06% | −1.37% | −1.24% |
| 0.6 | −5.67% | +0.05% | −2.98% | −2.67% | −0.06% | −0.89% | −1.76% |
| 0.8 | −5.65% | +0.05% | −2.97% | −2.67% | −0.06% | −0.73% | −1.97% |
| 1.0 | −5.65% | +0.05% | −2.96% | −2.68% | −0.05% | −0.64% | −2.10% |

The run finished at 12:06 on 16 Nov ("Simulation complete", 3531 s). At t = 1 the shock is at cell ≈ 37, and most of the kinetic loss is now in vertices 11–50.

**Reading:**
- **The Lagrangian step is exact.** All of the loss is in the **x-pass of the remap**: about 3.0% kinetic energy and 2.6% magnetic energy, constant in time as expected for a self-similar solution. This is the same order as the 4.8% missing at the shock.
- **The kinetic loss follows the shock.** It moves from vertices 3–10 to 11–50 as the shock moves out (shock at cell ≈ 4 at t = 0.1 and ≈ 16 at t = 0.4). So it is the remap smoothing the v\_y jump of the switch-off shock, not something at the sheet.
- **The magnetic loss is the remap smoothing the B\_y jump.** Its split by cell is meaningless, because the pass also moves field between bins; only the total counts.
- **In short,** the remap dissipates kinetic and magnetic energy at the shock without turning it into heat. `rke` returns only the kinetic part, and only about 1.5 points of the temperature deficit came back at α0 = 1, so `rke` may not be returning the v\_y part properly. That is worth checking in pure MHD.

### Next steps

1. **Cheap pure-MHD tests (2 min each),** by extending `runs/mhd_sheet_build.sh`:
   - `rke` on, with and without `rke_3d_fix` in base `LARE3D`, which doesn't have the fix. Does the shock reach 1.0 of exact?
   - Resistivity on.
   - `visc1` 0.8 against 0.1.
   - A magnetic-energy correction analogous to `rke`, behind a flag. Agree it with the LARE developers.
2. Report to the LARE developers: the energy defect at the MHD slow shock (5%, independent of resolution), with this Riemann problem and `scripts/mhd_riemann_exact.py` as the test.
3. Optional: fix the uncoupled dt bug in `twofluid.h` for PIP-mode uncoupled runs.
4. Then return to the coupling terms and the stall at α0 = 100.

## Status update, 16 Nov 2026 (14:30): pure-MHD tests: the remaining error is magnetic dissipation in the remap

**Runs:** the pure-MHD sheet at t = 1 (`sams LARE3D Hillier`; L = 25, dx = 0.02 unless stated; one rank each, about 3 min).
- **Built and launched by** `runs/mhd_tests.sh`.
- **`runs/mhd_sheet_build.sh`** now takes `VISC1=`, `ETA=` and `RKE=upstream|fix` from the environment. `RKE=fix` temporarily ports LARE3DST's `rke_3d_fix` to base LARE: a 0.25 share in the x, y and z remaps, and `energy_bcs()` after `energy_correction`. Sources are restored after every build.
- **Summary:** `python3 runs/mhd_tests_summary.py runs/<run> ...` gives one line per run: shock against exact, plateau errors, the energy flux missing at the shock, and global E − E0 as a percentage of the magnetic energy released. On the baseline its snapshot energy (−5.78%) agrees with the in-code `LARE3DST` budget (−5.65%).

| Run | Shock / exact | Inner ρ | Inner T | Energy flux missing at shock | E − E0 (% of released) | y-z mode |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (`mhdsheet_L25_dx002_t1`) | 0.915 | +9.8% | −10.1% | 4.8% | −5.78% | 3×10⁻¹⁰ |
| dx = 0.01 (`mhd_dx001`) | 0.921 | +10.1% | −10.2% | 4.9% | −5.78% | 1×10⁻⁶ |
| `visc1` = 0.8 (`mhd_visc08`) | 0.965 | +6.7% | −7.0% | 3.4% | −4.07% | 4×10⁻¹² |
| `rke` as upstream (`mhd_rke_up`) | 1.015 | −0.7% | +0.4% | 0.4% | −0.26% | **5×10⁻³** |
| `rke` + `rke_3d_fix` (`mhd_rke_fix`) | **0.965** | +4.2% | −4.7% | 2.1% | **−2.74%** | 1×10⁻¹⁰ |
| resistive, η = 10⁻³ or 4×10⁻³ | crashed on step 0 | | | | | |

**Findings:**
- **Resolution doesn't matter.** Halving dx changes neither the energy loss (−5.78% in both) nor the shock position.
- **With `rke` fixed, the kinetic part is returned exactly.** Half the error goes: the loss drops from 5.8% to 2.7%, and the shock moves from 0.915 to 0.965. What remains, 2.7%, is the **magnetic energy dissipated by the remap**. That is the same 2.6–2.7% the in-code budget measured for the x-pass, and the loss that the first handoff called "the only loss `rke` cannot return".
- **Upstream `rke` looks perfect for the wrong reason.** It hits the shock speed (1.015) only because it returns twice the kinetic loss (about 6%), which happens to cancel the magnetic loss (about 2.7%). It also breaks y-z uniformity (5×10⁻³): this is the ghost-cell bug.
- **More shock viscosity helps** (`visc1 = 0.8`: 0.965). More of the jump is dissipated in the Lagrangian step, where the energy correctly becomes heat, and less is left for the remap to lose.
- **Resistivity in base LARE is broken.** `resistive_effects` (`LARE/lagran.cpp:484`) does `pw::assign(data.bx1, data.bx)`, but `bx` is face-staggered and `bx1` is allocated cell-centred (`LARE/setup.cpp:194`), so it throws "Source and destination arrays must have the same number of elements". The sub-step count also divides by `dtr`, which is 0 on step 0.
- **The coupled runs behave differently.** At α0 = 1 (two-fluid, dx = 0.04), `rke` + `rke_3d_fix` only recovered 1.5 points of the 17% temperature deficit, against 5 of 10 points here. The coupled lag (19%) therefore has a large part that does not come from the plasma remap.

### Next steps

1. **Magnetic-energy correction in the remap,** analogous to `rke`, behind a flag that defaults off. Return the magnetic energy each remap pass dissipates to the internal energy of the cells involved. Test it in pure MHD: with `rke` + `rke_3d_fix` the target is shock/exact → 1.00 and E − E0 → 0. Agree it with the LARE developers.
2. Optional: fix base LARE's resistive path (face/cell array mismatch, and the dt on step 0) to test resistive smoothing of the B\_y jump.
3. Report to the LARE developers: the energy defect at the MHD slow shock (−5.8% of released, 4.8% of the flux through the shock; resolution-independent), the `rke` 3D bugs, and the resistive crash.
4. Then the two-fluid part of the lag: at α0 = 1 most of it does not come from the plasma remap. Look at the neutral fluid's remap (`LARE3DNF`) and the collisional heating.

## Status update, 16 Nov 2026 (15:00): prototype magnetic-energy correction (`rme`) closes the gap in pure MHD

### What it does

**`data.rme`** is a PROTOTYPE in base LARE (`src/PhysicsPackages/LARE`), off by default. It is the magnetic analogue of `rke`: it returns the magnetic energy that the remap's x-pass dissipates as heat.

**How the loss is measured.** In `remap_x`, before B\_y and B\_z are updated, each face element gets

Q = E\_after − E\_before + g\_i − g\_{i−1}

where:
- **E\_before** uses the Lagrangian width (B = Φ / (area·w1/dx));
- **E\_after** uses the Eulerian width;
- **g** is the energy of the slab swept across each x-face, g = F²L / (2 o\_v T): field F / (o\_v T), volume |o\_v|·L·T, where L is the element's own extent and T the transverse one.

**Properties of Q:**
- ΣQ is exactly the pass's change of magnetic energy.
- Locally, Q is the cell's mixing loss, −½ V₁V₂ (b₁ − b₂)² / (V₁ + V₂), plus the limiter's reconstruction term. Both are second order in the jumps.

**Where it goes.** Q is stored in `delta_me`, half to each of the two cells either side of the face. After the remap, `magnetic_energy_correction` adds −`delta_me` / (ρ·cv) to the internal energy, half to electrons and half to ions, as `energy_correction` does. Then `energy_bcs()` refreshes the ghosts.

**Code** (commit after this note's): `LARE/shared_data.h` (flag, array, hook in `endOfTimestep`), `setup.cpp` (allocation), `remap.cpp` (zeroing), `xremap.cpp` (measurement for B\_y and B\_z), `diagnostics.cpp` (return).

**Limitations:**
- **x-pass only.** The y- and z-passes are not done; their loss is 0.06% here. It also assumes B\_x is not changed by the x-pass, which holds in 1D.
- **Base `LARE3D` only.** `LARE3DST` and `LARE3DNF`, which the two-fluid runs use, don't have it yet.
- **CPU only:** it uses atomics but is untested on GPU.
- **Not yet agreed with the LARE developers.**

**A first attempt that failed.** The first version took g = F·(B\_l + B\_r)/2·L. That has a first-order reversible error wherever B has a gradient. At the fast rarefaction front it produced Q of ±2×10⁻⁷ per column, while the real net loss was about 1% of that. With plasma β\_p ≈ 5×10⁻⁴ that is 100× a cell's thermal energy: the internal energy went negative upstream and the shock ran away (1.38× exact). With the swept-slab g, the positive parts of Q are 10⁻⁴ of the negative parts, and Q sits only at the slow shock.

### Results (pure MHD, L = 25, dx = 0.02, t = 1; `runs/mhd_tests_summary.py`)

| Run | Shock / exact | Inner ρ | Inner T | Energy flux missing at shock | E − E0 (% of released) |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.915 | +9.8% | −10.1% | 4.8% | −5.78% |
| `rme` built in, flag off (`mhd_rme_off`) | 0.915 (bit-identical to baseline) | | | | |
| `rke` + `rke_3d_fix` (`mhd_rke_fix`) | 0.965 | +4.2% | −4.7% | 2.1% | −2.74% |
| `rme` (`mhd_rme`) | 0.965 | +4.7% | −5.4% | 2.4% | −3.08% |
| **`rme` + `rke` + `rke_3d_fix`** (`mhd_rme_rkefix`) | **1.015** | **−0.1%** | **−0.3%** | **0.3%** | **−0.17%** |
| the same at dx = 0.01 (`mhd_rme_rkefix_dx001`) | **1.009** | +0.01% | −0.44% | 0.2% | −0.18% |
| for reference: baseline at dx = 0.01 (`mhd_dx001`) | 0.921 | +10.1% | −10.2% | 4.9% | −5.78% |

**What the combination fixes.** With both corrections the plateau ρ, T, P and v\_y match the exact solution, and the shock is within one cell. PIP gave 1.010. Figure: `runs/mhd_rme_rkefix/mhd_rme_rkefix_vs_exact.png`.

**What remains.** The v\_x oscillations (±0.1) behind and ahead of the shock and some B\_y ringing are still there. They were there before too; they're a separate issue and don't affect the jump conditions.

**The two corrections are about equal,** about 3% each of the released energy, matching the in-code budget's split of the x-pass loss (kinetic 3.0%, magnetic 2.7%).

**Build options:** `runs/mhd_sheet_build.sh` now takes `RME=on`. It also backs up and restores only the files it edits, rather than using `git checkout`, so uncommitted work in them survives a build.

### Next steps (superseded: see the next section)

1. ~~Check the dx = 0.01 result~~: done, see the table (1.009 of exact).
2. Port `rme` (and `rke_3d_fix`) to `LARE3DST` and `LARE3DNF`, and add the y- and z-passes. Then rerun the coupling scan, α0 = 0.1–100 at L = 500 (each run a few hours), to see how much of the two-fluid lag is left.
3. Take `rme`, `rke_3d_fix`, this Riemann problem and the resistive crash to the LARE developers.
4. Optional: the v\_x ringing at the slow shock.

## Status update, 16 Nov 2026 (15:10): `rme` ported to LARE3DST; α0 = 1 two-fluid run with all corrections started

### Port (`f10b80b`, pushed)

- **What's in it:** `rme` is now in `LARE3DST`, the plasma package of the two-fluid runs. The measurement in `xremap.cpp` is identical to base LARE's, and `magnetic_energy_correction` returns the heat to the single internal energy (`energy_ion`). It has an allocation, zeroing and a hook in `endOfTimestep` after the `rke` correction, followed by `energy_bcs()`.
- **Not ported to `LARE3DNF`:** the neutral fluid has no magnetic field, so there is nothing to correct.
- **Start-up print:** now shows `rme (remap mag. energy): plasma on/off`.
- **Energy budget:** its last line is now "rke + rme corrections", since it includes both.

### Check in the two-fluid code (uncoupled, half domain, L = 25, dx = 0.02, in-code budget)

**Runs:** `runs/ebudget_st_rmeoff` and `runs/ebudget_st_rme_rke`, to t = 0.3, serial, built with `runs/ebudget_build.sh` (now takes `RKE=on`, which is `rke` + `rke_3d_fix`, and `RME=on`).

| t = 0.1 | E − E0 | Lagrangian | x-remap KE | x-remap magnetic | y-remap | Correction returned |
| --- | --- | --- | --- | --- | --- | --- |
| `rme` off | −5.86% | +0.08% | −3.16% | −2.69% | −0.09% | 0 |
| `rme` + `rke` + `rke_3d_fix` | **+0.01%** | +0.08% | −3.04% | −2.74% | −0.09% | +5.80% (8.52×10⁻³) |

- **Flag off changes nothing:** the snapshot at t = 0.1 is **bit-identical** to the morning's run (`runs/ebudget_L25_dx002_t1`).
- **Flag on, energy is conserved:** the remap still dissipates as before, and the corrections return all of it. The corrector identities hold to round-off (10⁻¹⁶), and the budget sums to 10⁻¹⁴.
- **Read the t = 0.2 and 0.3 rows,** once the runs finish, with `python3 runs/ebudget_table.py runs/<run>/run.log`.

### Running: α0 = 1 with all corrections

**The question:** with the plasma remap now conserving energy, how much of the two-fluid lag is left? That lag was 19% (shock 0.81–0.85 of PIP's) at α0 = 1.

**Run:** `runs/scan_a1_rme_rke`, started 16 Nov ~15:00 and expected to take about 80 min.
- **Set-up:** the coupling scan's α0 = 1 case: L = 500, dx = 0.04, t = 20, 5 ranks.
- **Corrections:** `rke` and `rke_3d_fix` (plasma and neutrals), and `rme` (plasma).
- **Binary:** `runs/scaling/sams_scan_a1_rme_rke`, from `runs/scan/build_variant.sh <out> scan 1 on on`. The 5th argument, new, switches `rme` on.

**Comparison:** a background job writes `runs/scan/compare_rme/` when the run finishes, against PIP (`runs/scan/pip_a1.0`) at t = 10 and 20. It compares three runs: the original (`base`, `runs/scan/sams_a1`), `rke` only (`rke`, `runs/sams_L2000_dx004_rke_t20`, L = 2000 but the same dx) and the new one (`rme_rke`). To redo it by hand:

```
cd runs/scan/compare_rme
python3 /home/snow/Documents/SAMS/TwoFluidWIP/scripts/collonly_compare.py --pip ../pip_a1.0/out \
    --sams ../../scan_a1_rme_rke/SAMS_testdata --snap 2 --xmax 120 --out a1_rme_rke_s2
```

**How to read it:**
- **Shock ratio near 1:** the two-fluid lag was the plasma remap's energy loss, made worse by coupling, and the fix carries over.
- **A lag well above the pure-MHD ~1%:** the rest is in the coupling. Suspects: the neutral fluid's remap kinetic-energy loss (returned by `rke` in `LARE3DNF`, so check the budget), collisional frictional heating and thermal exchange, or the α0 = 100 stall.

### Next steps (superseded: see the next section)

1. Evaluate the α0 = 1 run (above).
2. Depending on the result: rerun the other α0 values (0.1, 10, 100) with all corrections, or look at the coupling terms.
3. Add the y- and z-passes to `rme`, and take `rme`, `rke_3d_fix` and the resistive crash in base LARE to the LARE developers.
4. Production: once settled, rerun dx = 0.02, L = 2000, t = 100 with all corrections (about 2 days on 10 ranks).

## Status update, 16 Nov 2026 (16:15): the two-fluid lag is not the plasma remap

### α0 = 1 with all corrections: almost no change

`runs/scan_a1_rme_rke` finished 16:06 ("Simulation complete", 3905 s). Comparison with PIP is in `runs/scan/compare_rme/` (`a1_{base,rke,rme_rke}_s{1,2}`):

| α0 = 1 | Shock, t = 10 (PIP 3.98) | Shock, t = 20 (PIP 6.58) | ρ\_n plateau, t = 10 | T\_n | T\_p | B\_y |
| --- | --- | --- | --- | --- | --- | --- |
| no corrections (`runs/scan/sams_a1`) | 3.22 (0.81) | 5.58 (0.85) | +15.7% | −24.4% | −16.7% | +7.0% |
| `rke` + `rke_3d_fix` (`runs/sams_L2000_dx004_rke_t20`) | 3.26 | 5.62 | +14.9% | −23.7% | −15.2% | +3.2% |
| `rme` + `rke` + `rke_3d_fix` (`runs/scan_a1_rme_rke`) | 3.26 (0.82) | 5.62 (0.85) | +14.9% | −23.0% | −14.5% | +1.8% |

**Reading:**
- **The plasma remap is not the cause.** The corrections that fix the pure-MHD shock (0.915 → 1.009) move the α0 = 1 shock by one cell. At α0 = 1 the neutrals carry 99.7% of the mass and the plasma is dragged with them. The plasma remap's loss is therefore a small share of the total energy.
- **The deficit is in the neutrals:** T\_n is −23% and ρ\_n +15%, the same too-dense, too-cool signature of energy missing at the shock.
- **The neutrals' remap is not it either.** Their remap kinetic-energy loss is returned by `rke` in `LARE3DNF`, which was on in both corrected runs. The neutral fluid has no field, so there is nothing for `rme` to do.

### The uncoupled budget runs finished

With `rme` + `rke` + `rke_3d_fix`, E − E0 is +0.01%, +0.01% and +0.02% of the released energy at t = 0.1, 0.2 and 0.3 (`runs/ebudget_st_rme_rke`). Without them it is −5.86%, −5.71% and −5.72% (`runs/ebudget_st_rmeoff`).

### Main suspect: frictional heating in the collision step

In `get_collisional_source_terms` (`TwoFluid.cpp` ~line 819):
- **The heating:** frictional heating is ½ α ρ\_n (v\_n − v\_p)² per unit plasma mass, and the same per unit neutral mass with ρ\_p. It is computed at cell centres from velocities averaged over the 8 vertices around the cell.
- **The momentum exchange:** it acts on the vertex velocities.
- **Why that loses energy:** averaging reduces |v\_n − v\_p|². So where the drift varies from cell to cell, less heat is added than the kinetic energy the momentum exchange removes. The drift spikes in a few cells at the slow shock, so this would make the shock too cool.
- **Two more things to check:** the explicit update's own energy error, and the thermal-exchange factor (the code comment says "the 3/2 here needs fixing").

**Check:** see the next section.

## Status update, 16 Nov 2026 (16:30): the collision step loses energy; energy-conserving frictional heating

### The existing checks can't see it

- **`checkConservation`** saves the state before source half-step 1 and compares after half-step 2. The whole LARE step lies in between, so it doesn't isolate the sources. It also forms kinetic energy as cell ρ × vertex v at the same index.
- **`checkSourceConservation`** re-checks the source formulas cell by cell with the same averaged velocities. It cannot see an averaging error.

### New diagnostic: `check_source_energy` (`twofluid.h`, off by default)

**What it measures:** the total energy of both fluids just before and after every explicit source half-step:
- internal energy ρeV over cells;
- kinetic energy ½m\_v|v|² over vertices, with m\_v = ⅛Σρ·cv of the 8 cells around each vertex and half weight on the x boundaries.

Collisions change neither ρ nor B, so any change in this total is energy the source step creates or destroys.

**Output:** every `check_source_energy_every` steps, the cumulative heat added, the kinetic change, their sum (the error), the 6 x indices with the largest error, and the error in 8 x ranges. It runs serial only (no MPI reduction). Build with `runs/srcenergy_build.sh <out> <x_half> <nx> <t_end> <alpha0> [every]`.

### Result: α0 = 1, explicit sources, L = 25, dx = 0.04, serial (`runs/srcenergy_a1`)

| t | Kinetic energy removed by friction | Heat added | Lost | Lost / KE removed |
| --- | --- | --- | --- | --- |
| 0.10 | 3.92×10⁻⁴ | 3.57×10⁻⁴ | 3.5×10⁻⁵ | 8.9% |
| 2.11 | 7.65×10⁻³ | 7.37×10⁻³ | 2.8×10⁻⁴ | 3.7% |
| 4.67 | 1.49×10⁻² | 1.43×10⁻² | 5.7×10⁻⁴ | 3.8% |

**Where:**
- **The x = 0 vertex:** friction there removes 6.9×10⁻⁴ of kinetic energy. That is the symmetry plane, where the outflow jet along the sheet makes the drift largest.
- **Cells 1–5:** they get slightly more heat than the kinetic energy of their own vertices (+2–8×10⁻⁵ each), which offsets part of the loss.
- **Beyond x ≈ 3,** including the slow shock: below 10⁻⁷.

**Cause:** the momentum exchange acts on vertex velocities. The frictional heating ½αρ\_nρ\_p|v\_n − v\_p|² was computed at cell centres from vertex-averaged velocities, and averaging reduces |v\_n − v\_p|² where the drift varies between vertices.

**Size:** about 2.4% of the magnetic energy released by t = 4.7. This is estimated from the growth of internal plus kinetic energy, since the diagnostic does not measure magnetic energy. For comparison, in pure MHD a 5.8% loss gave a 9% shock lag. So this is likely a few percent of the 19% lag at α0 = 1, not all of it.

### Fix: `vertex_frictional_heating` (`twofluid.h`, `TwoFluid.cpp`; off by default)

**What it does:**
- **No friction in the cell heating:** with the flag on, `get_collisional_source_terms` drops the ½v\_d² friction term from the cell heating and keeps the thermal exchange.
- **Heat from the vertices:** in `apply_two_fluid_source` (explicit), before the velocity update, each vertex's exact kinetic-energy change in the momentum exchange is computed from its old and new velocities, for both fluids, with m\_v = ⅛Σρ·cv.
- **Distribution:** an eighth goes as heat to each of the 8 surrounding cells, half to each fluid (as the old ½v\_d² term split it).
- **Boundaries:** the ghost cells' share on a domain boundary matches the half weight of the boundary vertices.

**Not covered:** the implicit (PETSc) path is unchanged.

**Check** (`runs/srcenergy_a1_vfh`, same set-up): the error is −9×10⁻¹⁹ at t = 0.05, round-off (it was −8.9%).

**Side effect: smaller time step.** With the fix, dt settles near 6×10⁻⁵ early on, where without it dt reached 4×10⁻⁴ by t = 0.1.
- **Likely cause** (to confirm): the collisional time-step limit, `set_dt_collisional`, is 0.08 / max|source|. It uses the *absolute* rate of change of the specific energy, not the rate relative to the energy. The plasma at the sheet now gets its full frictional heat (the averaging under-heated it), so it runs hotter, the thermal-exchange rate is larger and the limit drops.
- **Possible remedy:** a relative limit, for example |de/dt| / e.

### Thermal exchange factor (the "3/2" comment): the formula is already right

SAMS's thermal exchange is α ρ\_n (T\_n − T\_p) / (γ(γ−1)) per unit mass, with T\_p = γP/(2ρ) and T\_n = γP/ρ, the same convention as PIP (`te = pr/de*gm*mu_p`). In terms of γ:
- the exchange is k(T\_n − T\_p)/(γ−1) per particle, which is (3/2)kΔT for γ = 5/3;
- with the code temperature T = γkT/m, that becomes ΔT/(γ(γ−1)) per unit mass;
- that equals PIP's 3/(2γ)·ΔT (`PIP_rot.f90`), exactly at γ = 5/3.

So the formula is unchanged. The comment "the 3/2 here needs fixing" is replaced by this derivation.

**Not to be confused with γ/(γ−1),** which is 5/2 at γ = 5/3 (the enthalpy factor). Used here it would make the thermal exchange about 2.8× too strong.

### Next steps (superseded: see the next section)

1. Confirm the dt limiter (which source term and where), and consider a relative limit.
2. Rerun the α0 = 1 scan case with `vertex_frictional_heating` + `rme` + `rke` + `rke_3d_fix`, to see how much of the lag goes.
3. If a large lag remains, test the neutral fluid's own shock in pure hydrodynamics, as was done for MHD.

## Status update, 16 Nov 2026 (16:50): time-step limiter fixed, per-fluid γ, α0 = 1 run with every correction

### Commits (all pushed)

| Commit | What |
| --- | --- |
| `e64b038` | `vertex_frictional_heating` (energy-conserving friction, off by default) and the `check_source_energy` diagnostic |
| `72f9074` | Collisional time-step limit from relative rates |
| `f185972` | Per-fluid γ in the temperatures and the thermal exchange |

### Time-step limiter (`set_dt_collisional`)

**What was wrong:**
- **Units:** dt was limited by 0.08 / max over every source array of |rate|. Those are absolute rates of quantities in different units: plasma specific energy around 100 against velocities around 1.
- **A ghost cell:** the loop included ghost cell 0 (x < 0).
- **Effect:** with `vertex_frictional_heating` the binding value was `source_energy` = 228 in that ghost cell, so dt = 3.5×10⁻⁴, against LARE's CFL step of 1.6×10⁻³. Even without the new heating the limit was binding (1.15×10⁻³ at t = 0.5), from the plasma heating rate near the sheet.

**New limit:** dt ≤ 0.1 × `dt_multiplier` / r, where r is the fastest *relative* rate over interior cells:
- α(ρ\_n + ρ\_p): momentum and thermal relaxation;
- |dm/dt|/ρ for each fluid: ionisation and recombination;
- |de/dt|/e for each fluid: heating.

With collisions off nothing is applied, so there is no limit. That also removes the 25× slowdown of the uncoupled runs (dt ≈ 3×10⁻⁵).

**Check** (α0 = 1, L = 25, dx = 0.04, t = 5, `runs/srcenergy_a1_vfh_dt`):

| | Old heating and limiter (`runs/srcenergy_a1`) | Energy-conserving heating, new limiter |
| --- | --- | --- |
| dt | 3.9×10⁻⁴ at t = 0.1, 1.7×10⁻³ by t = 2 | 1.64×10⁻³ (CFL) from t ≈ 0.7 |
| Steps to t = 5 | about 3700 | about 3000 |
| Source-step energy error | −3.8% of the frictional KE | 10⁻¹⁷ (round-off) |
| Slow shock (ρ\_n) at t = 5 | 1.82 | 1.86 (+1 cell) |
| Neutral plateau T | 15.53 | 15.57 (+0.3%) |

**Reading:** the frictional-heating error was real but sits at the sheet. It accounts for only a small part of the α0 = 1 lag.

### Per-fluid γ (`TwoFluid.cpp`)

**What changed:**
- **Neutral temperatures** now use `dataNeutral.gas_gamma`. There were 19 places that used the plasma's γ: the collision source terms, the vertex-rate α and the start-up normalisation check.
- **Thermal exchange:** it is now written as k(T\_n − T\_p)/m, with each fluid's code temperature divided by its own γ (k T/m = T\_s/γ\_s), times the exchange heat capacity per particle 1/(γ\_ex − 1). That is 3/2 for γ = 5/3.
- **The shared heat capacity:** it must be the same for both fluids, so the energy one gains is exactly what the other loses. It is taken from the plasma γ. This matters only if the two γ differ; change it in one place (`gamma_ex`) if another choice is wanted.

**Check:** with both γ = 5/3 the results match the previous build to 5×10⁻¹⁰ after 300 steps (round-off from the reordered arithmetic). The source step still conserves energy to 10⁻¹⁸.

### Running: α0 = 1 with every correction (`runs/scan_a1_allfix`)

**Set-up:**
- **Physics:** the coupling scan's α0 = 1 case: L = 500, dx = 0.04, t = 20, 5 ranks, started about 16:30. It was at t = 2.1 at 16:44, at dt = 1.64×10⁻³, the CFL step.
- **Corrections:** `rke` and `rke_3d_fix` (plasma and neutrals), `rme` (plasma), `vertex_frictional_heating`, and the new time-step limiter.
- **Binary:** `runs/scaling/sams_scan_a1_allfix`, from `runs/scan/build_variant.sh <out> scan 1 on on on` (the 6th argument, new, switches `vertex_frictional_heating` on). It was built before `f185972`, which is identical while both γ = 5/3.

**Comparison:** a background job writes `runs/scan/compare_rme/a1_allfix_s{1,2}*` when it finishes, next to `a1_base_*` and `a1_rme_rke_*`. To redo by hand:

```
cd runs/scan/compare_rme
python3 /home/snow/Documents/SAMS/TwoFluidWIP/scripts/collonly_compare.py --pip ../pip_a1.0/out \
    --sams ../../scan_a1_allfix/SAMS_testdata --snap 2 --xmax 120 --out a1_allfix_s2
```

**Expectation:** from the small-domain check, about one more cell than `rme_rke` (shock 3.26 at t = 10; PIP 3.98).

### Next steps

1. Evaluate `runs/scan_a1_allfix`.
2. **If most of the lag remains, test the neutral fluid's slow shock in pure hydrodynamics,** as was done for MHD:
   - write an exact hydrodynamic Riemann solver for the corresponding problem;
   - run LARE3DNF alone, or base LARE3D with B = 0;
   - measure the energy flux across its shock.

   The neutral plateau at α0 = 1 shows the same too-dense, too-cool signature (ρ\_n +15%, T\_n −23%).
3. Then: the other α0 with every correction, the LARE developers (`rme`, `rke_3d_fix`, the resistive crash), and production (dx = 0.02, L = 2000, t = 100).

## Status update, 16 Nov 2026 (17:00): collisional time-step limit made explicit (`58139db`)

### `set_dt_collisional` now

dt ≤ `dt_multiplier` / (fastest of the following rates):

| Rate | Where | What |
| --- | --- | --- |
| **Momentum exchange**, α(ρ\_n + ρ\_p) | the vertices the exchange updates (0..n) | dv\_p/dt = αρ\_n(v\_n − v\_p) and dv\_n/dt = −αρ\_p(v\_n − v\_p) leave the centre-of-mass velocity fixed and relax the drift at α(ρ\_n + ρ\_p): the stiffness of the exchange. It uses the same vertex values as the exchange: `rho_n_ac_vertex + rho_p_ac_vertex` with `vertex_rates`, otherwise the 8-cell averages of α and ρ. |
| \|dm/dt\|/ρ, each fluid | interior cells | ionisation and recombination |
| \|de/dt\|/e, each fluid | interior cells | heating: thermal exchange, friction, ionisation potential. This also covers the thermal relaxation, ~α(ρ\_p + ρ\_n/2). |

**Why not |dv/dt|/|v|:** v can be zero (the symmetry plane, gas at rest), and the ratio is not Galilean invariant.

**The 0.1 factor is gone:** it was left over from debugging the rates. The explicit momentum exchange is stable for dt·α(ρ\_n + ρ\_p) < 2.

**Checks:**
- **At α0 = 1** results are bit-identical to before (`runs/srcenergy_a1_dtv`, `runs/srcenergy_a1_dtm`): LARE's CFL step binds there.
- **At α0 = 100** the momentum rate is about 340, so the limit rises from 2.4×10⁻⁴ (with the 0.1) to 2.4×10⁻³, about the CFL step. Those runs become much faster, with the exchange taken in fairly large explicit steps (dt × rate up to 0.8). Compare with the earlier α0 = 100 runs (dt about 9.5×10⁻⁴ under the old absolute-rate limit) before relying on it.

### Existing bug found: `vertex_rates` can't be switched on (fixed: see the next section)

`plasma_source.vertex_rates = true` (set in `TwoFluidTest`) crashes at start-up, before any time step, with "Error: variable PIPSource/rho_p_ac_vertex not found in registry" (`TwoFluid.cpp:230`, `fillPPArray`).
- **Likely cause:** the registration (line 144) and the fetch are both conditional on `vertex_rates`, and the flag from `TwoFluidTest` probably arrives after PIP registers its variables.
- **Not caused by the limiter change, and not fixed.** Reproduce with `VR=on VFH=on runs/srcenergy_build.sh ...` (`runs/srcenergy_a1_dtv_vr`).

### Still running

`runs/scan_a1_allfix` (α0 = 1, every correction). Its comparison will go into `runs/scan/compare_rme/a1_allfix_*`.

## Status update, 16 Nov 2026 (17:15): PIP flag-dependent registrations fixed

**The common cause:** the runner sets packages up in this order (`include/runner/runner.h:634-640`, stage list at ~150):

registerAxes → registerVariables → defaultValues → controlVariables → setDomain → getVariables

A flag set in a problem's `controlVariables` (as `TwoFluidTest` sets all of `plasma_source`) is therefore not yet known when PIP registers its variables. Every registration that depended on such a flag failed in `getVariables`.

| Flag | Problem | Fix | Commit |
| --- | --- | --- | --- |
| `vertex_rates` | `PIPSource/{rho_p,rho_n}_ac_vertex` and `{gm_ion,gm_rec}_vertex` were registered only if the flag was on; the run stopped with "variable PIPSource/rho_p_ac_vertex not found in registry" | always registered; fetched and used only with the flag | `e7b1db0` |
| `ion_rec_nlevel` | the `species` axis (`registerAxes`) and `level_populations`/`level_rates` were registered only if the flag was on | always registered. A new PIP `setDomain` hook sizes `species` to 7 with `ion_rec_nlevel` (levels 0..nLevels, nLevels = 6 in the rate functions), otherwise to 1. An unsized axis is not allowed ("dimension zones not set"). With the flag off the level arrays cost about one grid-sized array each. | this commit |
| `check_conservation` | its `oldData` copies (`PIPconserve/*`) were registered under `check_source` but fetched under `check_conservation`, and `oldData.vz_n` was fetched from `PIPconserve/vz`, so the old v\_z and v\_z,n were the same array | not registered at all. `copyState` allocates them on first use, shaped like the live arrays (`conservedManager`). | this commit |

**Checks** (α0 = 1, L = 25, dx = 0.04, t = 1, `runs/srcenergy_a1_{reg2,cc,vr2}`):
- **Bit-identical:** the default mode, `check_conservation` on and `vertex_rates` on all match their references, with no new snapshot fields.
- **`vertex_rates`** now runs. With `vertex_frictional_heating` the source step conserves energy to 10⁻¹⁸. It differs from the cell-average rates by under 1% in ρ\_n and e\_n and by 3–4% in v\_y (`runs/srcenergy_a1_vr`).
- **`check_conservation`** now runs. It prints its diagnostic every step, and `copyState` still prints two pointers every step (an old debug line, left as it is).
- **`ion_rec_nlevel`** itself was not run: it needs the n-level rate data. Only its registration and sizing are now independent of when the flag is set.

**Build options:** `runs/srcenergy_build.sh` now also takes `VR=on` (`vertex_rates`) and `CC=on` (`check_conservation`).
