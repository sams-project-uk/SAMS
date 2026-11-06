# SAMS two-fluid handoff: energy conservation and symmetry

Nov 5, 2026 · @Ben · updated the same day after working through the plan (steps 1, 2, 3 and 6, and the problem B checks of step 4)

## Status in brief

Branch `TwoFluidWIP` of `github.com/AstroSnow/SAMS_PI` is now at `44d2990` (pushed). The investigation started from `24e7861`; four commits were added on top:

| Commit | Content |
| --- | --- |
| `ca26e28` | `visc1` 0.8 → 0.1 in `TwoFluidTest`; new `sh21.coupling` switch |
| `5c71bc3` | Build fixes for GCC 12 and PETSc 3.17 (no change to generated code) |
| `e19c500` | In-code per-operator energy budget, `data.energy_budget` (off by default) |
| `44d2990` | Experimental `rke_3d_fix` (off by default): two bugs in `rke` in 3D |

**B (checkerboard) is solved.** It was caused by the linear shock viscosity `visc1 = 0.8`, which commit `c6d4638` (26 Aug) raised from 0.1. At the full CFL step that term is unstable for an odd–even velocity mode. With `visc1 = 0.1` the coupled v\_x checkerboard at t = 0.1 falls from 0.32 to 7.6×10⁻⁴, below the 10⁻³ target.

**A (energy) was mostly a diagnostic artifact.** Snapshots never write the vertex at x = 0, on the symmetry plane, where the outflow jet along the sheet is fastest. `energy_budget.py` therefore missed most of the kinetic energy. The code's own budget gives a true loss of 4.0% at t = 0.01 (not 19%) and 2.7% at t = 0.0021 (not 54%). What is left is the remap's kinetic-energy loss (what `rke` returns) and magnetic dissipation in the remap: 1.65% at t = 0.01 with `visc1 = 0.8`, 2.05% with `visc1 = 0.1`. The magnetic part is the only loss `rke` cannot return, and it is still above the 1% target.

**`rke` has two bugs in 3D,** and they are probably upstream too. It returns exactly twice the kinetic-energy loss as heat. It also does not refresh the ghost cells afterwards, which drives a growing odd–even mode across y and z. Both fixes are behind `rke_3d_fix`.

**SAMS vs PIP (step 6) passes the large-scale criteria:**
- The fast front at t = 0.1 is within 2 cells of PIP's, and the plateau values agree to within 0.8%.
- Near the sheet the structure is not converged even in PIP.
- SAMS's dip at the fast front is deeper than PIP's, which is unexplained.

Start here: read "What is established", then the open items in "Remaining work". The pitfalls section has new entries; the first two caused the largest misreadings so far.

## Code and build context

SAMS is a C++20 framework; physics lives in packages under `src/PhysicsPackages`. The run is `./sams LARE3DST LARE3DNF PIP TwoFluidTest` (packages are called in that order at every runner stage).

| Package | Path | Role |
| --- | --- | --- |
| `LARE3DST` | `src/PhysicsPackages/LARE3DSingleTemperature` | Plasma: Lagrangian-remap MHD (a fork copy of LARE; its data struct is the base `LARE` package's) |
| `LARE3DNF` | `src/PhysicsPackages/LARE3DNeutralFluid` | Neutral fluid: Lagrangian-remap hydrodynamics |
| `PIP` | `src/PhysicsPackages/TWOFLUID` (`twofluid.h`, `TwoFluid.cpp`) | Two-fluid sources (collisions, ionisation, recombination, ionisation potential), implicit PETSc solver, diagnostics |
| `TwoFluidTest` | `src/InitialConditions/TwoFluidTests` | Test problems, run settings, boundary conditions |
| Harness | `include/` | Runner (`include/runner/runner.h`), boundary conditions (`include/boundaryConditions/builtInBoundaryConditions.h`), portable OpenMP backend (`include/pp`) |

**Build.** CMake with OpenMP and PETSc. `make` regenerates a git stamp, printed at start-up. Recompile single objects with `make src/<path>.cpp.o` when iterating, and touch `src/main.cpp` after header changes.
- **Server:** PETSc 3.26, serial (MPIUNI).
- **Workstation:** the only PETSc is a 3.17.1 debug build with OpenMPI. Configure with `PKG_CONFIG_PATH=~/Documents/PETSc/UTF-8multi_ion/multi_ion/petsc-with-docs-3.17.1/petsc-3.17.1/arch-linux-c-debug/lib/pkgconfig cmake . -DMPI=ON -DCMAKE_BUILD_TYPE=Release`. The full build takes about 20 s on 12 cores. GCC 12 needed the fixes in `5c71bc3`.
- **New source files:** sources are collected with `GLOB`, so re-run `cmake .` after adding a `.cpp`, or the link fails.

**Step order** (runner, per step):
1. `beforeStartOfTimestep`: LARE's `lagrangian_prepare` (pressure, shock viscosity, vertex density).
2. `calculateTimestep`.
3. `applySourceTermsStart`: first Strang source half-step (PIP).
4. `afterSourceTermsStart`: LARE re-prepares from the post-source state.
5. `startOfTimestep`: predictor.
6. `halfTimestep`: corrector.
7. `endOfTimestep`: remap, then the `rke` energy correction if on.
8. `afterEndOfTimestep`: second source half-step (PIP).

**Test problems** (`problem` in `TwoFluidTest.h`; settings in its `sh21` and `pulse` structs):

- `snow_hillier_2021`: uniform partially ionised plasma (T0 = 10⁴ K, ξ\_n = 0.99734, β = 0.1, B\_x = 0.1, B\_y = ∓1 across x = 0, τ\_IR = 10⁻³, α0 = 1), Alfvén units. The problem sets T0, t\_ir, alpha0, P\_ref, rho\_ref, collisions, empirical rates and the ionisation potential itself in `TwoFluidTest::controlVariables`.
- `symmetric_pulse`: same plasma, uniform field, a Gaussian hot spot at x = 0 (`pulse.amplitude`, `pulse.width`). Mirror-symmetric with no current sheet. Always full domain (x ∈ \[−10, 10\], 4096 cells). With `pulse.amplitude = 1` and `visc1 = 0.8` it reproduces the problem-B instability in its simplest form.

**Current settings on the branch** (`44d2990`):
- `problem = "snow_hillier_2021"`, `sh21.coupling = true`.
- `half_domain = true`: x ∈ \[0, 10\], 2048 cells, dx = 0.00488, symmetry plane at x = 0.
- `sheet_width_cells = 0` (sharp sheet), `t_end = 0.1`, `pulse.amplitude = 0.02`.
- `implicit = true`, `noise_check = true`, `symmetry_check = false`.
- `startup_dt_fraction = 1e-3`, `dt_max_growth = 1.1`.
- **`visc1 = 0.1`**, `visc2 = 1`, `dt_multiplier = 0.8`.
- `data.rke = false`, `rke_3d_fix = false`, `energy_budget = false`.

**Uncoupled runs.** Set `sh21.coupling = false` in `TwoFluidTest.h`. That turns off collisions, ionisation/recombination and the ionisation potential together, so the plasma is pure ideal MHD. The start-up block reports it. With `half_domain` this is a closed system until fast waves reach x = 10 at t ≈ 0.5.

## What is established

**Mirror symmetry.** Except for the shock-viscosity limiter, every operator in the step has passed an exact mirror-symmetry test. `symmetric_pulse` with `sh21.shock_viscosity = false` stays symmetric to round-off (about 10⁻¹⁴ in ρ, energies and B\_y) to t = 0.1 with all physics on.

**Energy conservation of LARE's operators** (from the in-code budget, uncoupled):
- **Lagrangian step: exact.** Pressure work equals pdV and viscous work equals viscous heating, to round-off. The kinetic and internal energy identities of the corrector hold to about 10⁻¹⁷. The only exception is when the viscous instability below is active.
- **Remap, internal energy: conserved to round-off.**
- **Remap, kinetic energy: lost** (what `rke` returns), about 2.4% (`visc1` 0.8) or 3.3% (`visc1` 0.1) of the released magnetic energy at t = 0.01. Most of the loss is at vertices 0–2.
- **Remap, magnetic energy: lost.** This is numerical reconnection at the sheet: 0.1% at t = 0.0021, growing to 1.65–2.05% at t = 0.01. It is the only loss that `rke` cannot return.

**Bugs found and fixed on this branch** (all in its history):

| Area | Bug | Fix |
| --- | --- | --- |
| Ionisation/recombination energy | Thermal exchange had the same sign in both fluids and lacked the dilution terms, so energy was not conserved | Transfer form with dilution, derived from Snow & Hillier eqs. 3 and 7 |
| Normalisation | T\_f, φ̂ and α\_c used inconsistent normalisations | One `T_norm` from `P_ref`/`rho_ref`, used everywhere |
| PETSc Rosenbrock-W | Stiff sources were passed as an explicit RHS, so the solve was effectively explicit | Sources passed as `IFunction`/`IJacobian` |
| PETSc robustness | A NaN stage aborted the run; after a rejected stage the Jacobian lag was never restored | Stage check rejects non-finite stages; lag reset; sub-stepping fallback |
| Step order | LARE's pressure, shock viscosity and vertex density were prepared before the first source half-step and then used after it | New runner hook `afterSourceTermsStart` re-prepares |
| Ghost cells | Sources were applied to low-side ghost layers only, breaking y/z uniformity | Interior-only updates, ghosts refreshed from boundary conditions (explicit and implicit) |
| Finite-difference Jacobian | One-sided differences were not mirror-symmetric | Central differences (no effect on the symmetry growth, but more accurate) |
| Start-up | Fixed 100 steps at dt = 10⁻⁸, then a jump to the CFL step | One step at 10⁻³ × the allowed step, then growth ≤ ×1.1 per step |
| **Shock viscosity** (`ca26e28`) | `visc1 = 0.8` (since `c6d4638`) is unstable at the full CFL step: odd–even velocity mode, viscosity doing positive work | `visc1 = 0.1`, as every other test problem and LARE3DST's default |
| **Build** (`5c71bc3`) | GCC 12 rejected non-dependent `static_assert(alwaysFalse<T_dataType>)`; the PETSc < 3.18 `PetscCall` shim recursed on 3.17 | `alwaysFalse<decltype(ix)>`; shim only if PETSc lacks `PetscCall` |
| **`rke` in 3D** (`44d2990`, EXPERIMENTAL, off by default) | Loss shared 0.5 to each of 4 cells (returns 2×); no ghost refresh after `energy_correction` (y-z odd–even mode) | `rke_3d_fix`: share 0.25, `energy_bcs()` after the correction. Plasma and neutral packages; base `LARE` unchanged |

**Conclusions about the symmetry break** (unchanged; still valid):

- It starts at round-off and is amplified at the moment shocks form. LARE's Caramana edge viscosity limiter divides by velocity jumps passing through zero there.
- The viscosity code itself was checked term by term and is mirror-symmetric.
- Raising the limiter's "negligible jump" threshold (1e-14 to 1e-10) reduced the asymmetry about tenfold. At 1e-8 it removed dissipation from weak waves and made things worse. The experiment was reverted, so the threshold is the original 1e-14.
- It is not a threading problem: serial and 128-thread runs are bit-identical.
- The `rke` implementation in `LARE3DST` matches upstream `main`'s maintained LARE line for line. That means the two `rke` bugs above are very likely upstream too; report them to the LARE developers.

## Problem A: energy conservation (largely resolved)

**The earlier numbers were wrong.** `energy_budget.py` reads snapshots, and snapshots write vertex fields as entries 1..nx. Vertex 0 is never written. In the half domain, vertex 0 sits at x = 0 on the symmetry plane, where the outflow jet along the sheet is fastest (v\_y ≈ −20). Its kinetic energy (half mass) is 1.1×10⁻³ at t = 0.0021 and 2.24×10⁻³ at t = 0.01, almost all of what the script called "missing". The script is still valid on the full domain, where vertex 0 is the static outer boundary.

The run is the closed, uncoupled half-domain `snow_hillier_2021` (E0 = 5.8000). The in-code budget below is for the plasma, per unit y-z area. Percentages are of the magnetic energy released.

| Uncoupled run | t | Released | Script ("missing") | In-code total error | Remap KE loss | Magnetic error | `rke` heat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `visc1` 0.8, `rke` off | 0.0021 | 2.157e-3 | 54% | −2.67% | −2.56% | −0.11% | 0 |
| `visc1` 0.8, `rke` off | 0.0101 | 1.483e-2 | 19% | −4.02% | −2.38% | −1.65% | 0 |
| `visc1` 0.8, `rke` on (as upstream) | 0.0101 | 1.460e-2 | — | +0.40% | −2.29% | −1.69% | **+4.59% (2×)** |
| `visc1` 0.8, `rke` + `rke_3d_fix` | 0.0101 | 1.474e-2 | — | −1.65% | −2.33% | −1.66% | +2.34% |
| `visc1` 0.1, `rke` off | 0.0100 | 1.496e-2 | — | −5.39% | −3.34% | −2.05% | 0 |

- **The `rke`-as-upstream row does not add up exactly.** Its y-z instability breaks the corrector identities (−0.27%), so that row's split is approximate.
- **Where the magnetic error sits.** It is spread over the Lagrangian step's x-compression (+0.9 to +1.1%), the x-pass (−1.5 to −1.85%) and the y-pass (−1.1 to −1.3%). Only their sum is robust. The split depends on how B is attributed to the Lagrangian cells.
- **`visc1 = 0.1` loses a little more.** There is less viscous dissipation, so the remap dissipates more. `visc1 = 0.1` with `rke` + `rke_3d_fix` has not been run; it should leave about 2.0%, all magnetic.
- **The annihilation hypothesis is real but minor.** Magnetic energy is lost in the remap, but it was never the bulk of the "missing" energy. It grows with time (0.1% → 1.65–2.05% between t = 0.002 and 0.01), so it may matter at t = 0.1.

**Step 3 (`symmetric_pulse`, amplitude 1, uncoupled, t = 0.1)** did not test `rke` on shocks as intended. With `visc1 = 0.8`, the plasma's "shocks" were the odd–even instability, and the viscosity created energy: the plasma gained +2.2×10⁻⁴, several times all its kinetic energy. With `rke_3d_fix` on, `rke` again returned the remap loss 1:1 (1.834e-4 against 1.835e-4). The plain-`rke` run re-triggered the y-z instability: the identity check fired on 561 steps.

Tasks:

- [x] In-code energy budget per operator (plasma). It also gives the exact work and heating split of the corrector.
- [x] Confirm or refute annihilation: it happens, but it is 1.65–2.05% at t = 0.01, not 19–54%.
- [x] Check `rke`: found and fixed (experimentally) a 2× over-return and a missing ghost refresh.
- [ ] Re-run `symmetric_pulse` amplitude 1 with `visc1 = 0.1`, with `rke` off, on, and on + `rke_3d_fix`. This is the real test of `rke` on shocks now that the instability is gone. Target: loss below 1% of the kinetic energy generated.
- [ ] Neutral fluid in the pulse run (`visc1 = 0.8`): it loses about 3.1×10⁻⁵, roughly equal to the kinetic energy it gains, and this doesn't change with `rke`. Unexplained. The budget covers the plasma only, so extend it to `LARE3DNF`.
- [ ] Magnetic dissipation in the remap. Prototype explicit resistivity (`resistiveMHD`, η large enough that the sheet spans several cells) or a magnetic-energy remap correction analogous to `rke`, behind a flag that defaults off. Agree the second with the LARE developers.
- [ ] Agree `rke_3d_fix` with the LARE developers. If accepted, apply it to the base `LARE` package as well and make it the default.

## Problem B: velocity checkerboard (solved)

**Cause.** The linear term of LARE's Caramana edge viscosity, at `visc1 = 0.8` with `dt_multiplier = 0.8`, is unstable for an odd–even vertex velocity mode. The CFL limit includes viscosity only through `c_visc2 = p_visc/ρ`, which scales with the velocity jump. It therefore gives no protection against a small-amplitude mode.
- **Onset.** The mode starts from round-off once the start-up ramp brings dt to its cap, and grows about ×3.5 per step in the corrector.
- **Energy.** The viscous force then does positive work, creating energy.
- **History.** `visc1` was 0.1 until commit `c6d4638` (26 Aug, "tests for substepping"). Every other test problem uses 0.1, and so does LARE3DST's own default.

**Parameter scan.** `symmetric_pulse`, amplitude 1, uncoupled, t = 0.02:

| Variant | Peak odd–even amplitude (v\_x) | Plasma E − E0 |
| --- | --- | --- |
| `visc1` 0.8, `dt_multiplier` 0.8 (old) | 0.13, growing ×3.5/step | +7.6e-5 |
| `visc2` 2 | 0.13 | +5.7e-5 |
| `dt_multiplier` 0.4 | 1e-8 | −5e-8 |
| `visc1` 0.1 | 1e-9 | −3e-8 |
| No shock viscosity | 1e-9 | −3e-8 |

**Effect on the current-sheet problem:**

| Run (half domain) | t | v\_x checkerboard, `visc1` 0.8 | v\_x checkerboard, `visc1` 0.1 |
| --- | --- | --- | --- |
| Uncoupled | 0.01 | 0.20 local (vertices 5–17, −1.06/−1.47 alternation) | gone (smooth −1.23 to −1.37) |
| Coupled (IRIP, implicit) | 0.10 | 0.32 (handoff's measure; reproduced locally) | **7.6×10⁻⁴** (target < 10⁻³) |
| Coupled, v\_y / neutral v\_x | 0.10 | 0.030 / 1.0×10⁻³ | 2.6×10⁻³ / 7.9×10⁻⁴ |

The remaining v\_y reading is at vertex 2, on the edge of the outflow jet profile (see pitfalls); it is not a checkerboard. With `visc1 = 0.1` the coupled inflow plateau is smooth at −0.117, compared with PIP's −0.118.

Hypotheses 1–3 from the original handoff (dissipation deficit, time stepping, fork-specific) are superseded. The base-`LARE3D` comparison and the extended `noiseCheck` were not needed.

## SAMS vs PIP (step 6)

**Setup.** Coupled IRIP, half domain, sharp sheet, identical cells (2048 on \[0, 10\]), t = 0.1. The PIP clone is in `PIP/`, branch `sams_sh21` from `0474b7a`, with both patches applied and these settings:
- `half_domain = .true.` and `end(1) = 10` in `shock_tube_ion.f90`.
- `ix = 2048`, `tend = 0.1`, `dtout = 0.01` in `setting.txt`.

PIP runs in about 1 s. Its outputs are in `PIP/snow_hillier_2021/`; a second-order run is in `_order2` and a 4× finer first-order run in `_fine4x`.

| Quantity | PIP 1st, 2048 | PIP 2nd, 2048 | PIP 1st, 8192 | SAMS `visc1` 0.1 | SAMS `visc1` 0.8 |
| --- | --- | --- | --- | --- | --- |
| Fast front position (v\_x = −0.06) | 1.9545 | 1.9531 | 1.9489 | **1.9440** (−2 cells) | 1.8588 (−20 cells) |
| v\_x plateau (x = 0.3–1) | −0.1183 | −0.1200 | −0.1261 | **−0.1192** (0.8%) | −0.1135 (4%) |
| Other plateaus (ρ\_p, ρ\_n, T\_p, T\_n, p\_p, B\_y) | ref. | | | within 0.2% (v\_y 0.8%) | within 2% |
| Sheet width (v\_y down to 10%) | 0.051 | 0.032 | 0.023 | 0.017 (4 cells) | 0.100 |
| Deepest v\_x at the front | −0.167 | −0.197 | −0.150 | −0.242 | −0.115 |

Plots: `runs/step6_wide.png` (x from 0 to 3) and `runs/step6_zoom.png` (first 30 cells).

- **The large scale agrees.** The acceptance criteria for front position and plateaus are met with `visc1 = 0.1`. With 0.8 they were not.
- **The sheet structure is not converged, even in PIP.** PIP's sheet more than halves in width with 4× resolution or with second order. SAMS's is narrower than all three, which fits a less diffusive scheme. Cell-by-cell differences at the sheet (cell 1: ρ\_p −11%, v\_y −14% against PIP first-order) are not a fair test at this resolution.
- **The inflow plateau depends on resolution too.** It is set by reconnection at the under-resolved sheet; PIP's moves by 6.6% with 4× cells. The 0.8% agreement holds only at matched resolution.
- **Unexplained:** SAMS's dip at the fast front (x ≈ 1.9) is deeper than any PIP run's, and PIP doesn't move towards it when refined.

## Diagnostics and tools available

The in-code checks in `data_two_fluid_source` (`twofluid.h`) run after each of seven stages: initial condition, start of step, after source half-step 1 and re-prepare, after predictor, after corrector, after remap, and after source half-step 2. Each prints one line per stage inside a time window, and otherwise only new powers of ten of growth.

| Tool | Where | What it gives | Notes |
| --- | --- | --- | --- |
| **`energy_budget`** | `data.energy_budget` (set in `TwoFluidTest.cpp`); `LARE3DSingleTemperature/energyBudget.cpp` | Plasma internal, kinetic and magnetic energy after every operator. Changes charged per operator, including each remap pass separately; exact work/heating split of the corrector; where (by x index) the remap changes energy; KE at vertex 0 | Printed at each snapshot time (`energy_budget_every` > 0 for every N steps). Prints `ENERGY BUDGET identity:` if the corrector's energy identity fails in any cell, which is how the `rke` y-z bug was found. Assumes flow varying in x only. Plasma only |
| `symmetry_check` | `PIP::symmetryCheck`, `TwoFluid.cpp` | Mirror asymmetry of every field per stage; `SYMMETRY FIRST BROKEN` and `symmetry GREW` lines | Full domain only (switched off automatically with `half_domain`); parity set by `symmetry_by_odd`, chosen by the problem |
| `noise_check` | `PIP::noiseCheck`, `TwoFluid.cpp` | Checkerboard amplitude of v\_x and v\_y (both fluids), signed, per stage; `noise GREW` lines from 10⁻¹² | Window set by `noise_check_from`/`noise_check_to`; the sign flips if the worst 16-vertex window shifts by one vertex |
| `watch_state` | `PIP::watchState` | NaN/Inf scan and a trend line every `watch_interval` steps | — |
| `check_conservation` | `PIP::checkSourceConservation` | Energy budget of the source terms only | — |
| Start-up block | `printConfiguration`, `setNormalisation` | Git commit, every physics and solver setting (now including `rke`, `rke_3d_fix`, `energy_budget`, coupling), normalisation checks | Read it to confirm what a run actually used |
| `sams_symmetry_check.py` | repo root (untracked) | Per-snapshot y/z spread, mirror asymmetry per field, grid symmetry | `--by-even` for `symmetric_pulse`; `-v` for the per-field table. `SAMS_symmetry_check.py` is an identical copy |
| `compare_sams_pip.py` | repo root (untracked) | Field-by-field SAMS vs PIP difference on the shared grid | Puts SAMS vertex velocities at their true positions. Expects `diagnostics_NNNNN.h5`, but SAMS now writes `diagnostics_NNNNN_rank0.h5`: import `load_pip`/`load_sams` and pass paths |
| `energy_budget.py` | repo root (untracked) | Internal, kinetic (vertex) and magnetic energy per snapshot, and (E − E0)/E0 | **Misses vertex 0. Wrong for half-domain runs** (see problem A); fine for full-domain runs. Prefer the in-code budget |
| Solver tests | `tests/` (untracked) | Standalone PETSc tests: stiff relaxation, NaN recovery, mirror symmetry of the Jacobian | Build against the compiled `petscStiffSolver.cpp.o` |

**PIP reference.**
- **Setup:** apply `pip_snow_hillier_2021.patch` then `pip_half_domain.patch` to PIP `khinvel` at `0474b7a`. This is already done in `PIP/`; note that `~/Documents/KHItube` is at `f314db8` and lacks `0474b7a`.
- **Settings:** set `half_domain = .true.` and `end(1) = 10` in `shock_tube_ion.f90`, and `ix = 2048` in `setting.txt`, for identical cells.
- **Build:** run a plain **serial** `make FC=h5pfc`; `make -j` fails because the Fortran modules must be compiled in order.
- **Run:** `mpirun -np 1 ./a.out`.

PIP's half-domain run reproduces its full-domain run to 10⁻¹².

## Plan status and acceptance criteria

1. ~~Reproduce the baseline~~: done; the snapshot numbers were reproduced exactly.
2. ~~Locate the energy loss~~: done. Most of it was the unwritten vertex 0. The real losses are the remap's kinetic energy and magnetic dissipation.
3. ~~Check `rke` on shocks~~: done, but invalidated by the viscous instability. The `rke` bugs were found and fixed experimentally. **Re-run with `visc1 = 0.1`.**
4. ~~Characterise the checkerboard~~: done; the cause is `visc1 = 0.8`.
5. **Prototype a fix for the magnetic dissipation**: open (now the main energy error).
6. ~~Compare with PIP~~: done; the large-scale criteria are met.

| Check | Target | Status |
| --- | --- | --- |
| Uncoupled closed run, total energy at t = 0.01 | Loss below 1% of the magnetic energy released | 1.65% with `rke` + `rke_3d_fix` (`visc1` 0.8); about 2.0% expected with `visc1` 0.1. **Not met:** magnetic dissipation in the remap |
| `symmetric_pulse`, amplitude 1, `rke` on, t = 0.1 | Loss below 1% of the kinetic energy generated | Not yet measured with `visc1 = 0.1` |
| Coupled half domain, v\_x checkerboard at t = 0.1 | Below 10⁻³ | **Met:** 7.6×10⁻⁴ |
| SAMS vs PIP at t = 0.1, half domain, sharp sheet | Shock positions within a few cells; plateaus within a few percent | **Met:** front −2 cells; plateaus within 0.8% (sheet structure not converged in either code) |

## Remaining work

1. Re-run `symmetric_pulse` amplitude 1 with `visc1 = 0.1` (`rke` off, on, on + `rke_3d_fix`) to measure `rke` on real shocks.
2. Reduce magnetic dissipation in the remap below 1%: resistivity, or a magnetic-energy correction analogous to `rke` (experimental flag).
3. Extend `energy_budget` to the neutral fluid and explain the neutral loss in the pulse run.
4. Explain SAMS's deeper dip at the fast front (x ≈ 1.9, t = 0.1).
5. Resolution study of the sheet (SAMS and PIP at 4× cells) before comparing near-sheet structure.
6. Take `rke_3d_fix` to the LARE developers, together with the `shock_heating` remark in the pitfalls.
7. Consider a viscous stability term in `set_dt`, independent of the velocity jump (amplitude), so that large `visc1` cannot go unstable again. This also changes LARE's numerics.
8. Optionally, write vertex 0 in snapshots (or document it), so that snapshot-based budgets are correct on the half domain.

Working rules (unchanged):

- Build every change against the current branch.
- Keep diagnostic and experimental switches off by default.
- Label any change to LARE's shared numerics as experimental until the LARE developers agree.
- Never change a default silently: the start-up block should report every new setting.

## Pitfalls and conventions

Most wrong turns in this investigation came from a diagnostic, not the code. Check these before trusting a number.

- **Snapshots do not write vertex 0** (new). Vertex fields are written as vertices 1..nx. In the half domain, vertex 0 is on the symmetry plane at x = 0 and carries the outflow jet, so any snapshot-based kinetic-energy sum misses most of the jet's energy. This produced the "19% / 54% missing" of the original handoff. Use the in-code budget.
- **LARE's Lagrangian step does not change the face-centred B** (new). All of the induction equation, both compression by v\_x and shear by v\_y acting on B\_x, is applied in the remap passes. A magnetic-energy drop in the remap is therefore expected even in ideal MHD. Compare it with the Lorentz work done in the corrector, as `energy_budget` does, before calling it a loss.
- **The CFL limit does not protect the linear shock viscosity** (new). `c_visc2` scales with the velocity jump. A large `visc1` can be unstable for small odd–even modes at the full step, with no warning. Symptom: a checkerboard growing in the corrector, and positive viscous work in `energy_budget`.
- **1D runs are 3D with ny = nz = 2** (new). A bug that only breaks y/z uniformity (such as the `rke` ghost cells) shows up as a y-z odd–even mode and as broken energy identities, not in the x profiles. The budget's per-cell identity check catches it.
- **`shock_heating` uses `dy*dy`** in the z-edge terms `a9` and `a11` (all three LARE packages), where `dx*dy` looks intended. It has no effect in 1D runs with dx = dy, but report it upstream.
- **Staggering and indices.** Cell-centred quantities (ρ, energies, B\_y, B\_z) live on cells 1..nx; velocities (all components, both fluids) on vertices 0..nx; B\_x on x-faces. LARE's remap mass flux `dm` holds x-face fluxes during the x-remap, although it is registered cell-centred.
- **Output positions.** Snapshots write vertex fields (velocities, B\_x) as entries 1..nx under the cell-centre coordinates `MeshCC/x`. Entry i really sits at x\_c(i) + dx/2, and its mirror is entry nx − 2 − i, not nx − 1 − i. The Python scripts handle this. Snapshot files are now named `diagnostics_NNNNN_rank0.h5`.
- **Source-field output is mis-shaped.** The `MeshCC_source` arrays (`ac`, `gm_ion`, `source_*`) do not match their declared shape, and their coordinates end in uninitialised memory. Upstream `TwoFluidWIP` fixed this in commits `0fd64db` and `20ec9fc`; they are not ported here. Do not diagnose from those fields until they are.
- **Mirror parity depends on the problem** (B is an axial vector). For the current sheet (plain reflection): v\_x odd, v\_y even, B\_y and B\_z odd, everything else even. For a uniform oblique field (reflection plus a half-turn about x): v\_x and v\_y odd, B\_y even. `symmetry_by_odd` selects this; using the wrong one gives asymmetries of exactly 2.
- **Relative measures on near-zero fields mislead.** An early noise metric divided by the field maximum and reported a single sharp vertex as "100% noise". The current checkerboard measure is absolute. An isolated jump still registers at about 1/32 of its size, so readings within a few cells of the sheet can be the jet profile, not a checkerboard. The 16-vertex window also under-reads a mode whose amplitude varies across the window: 0.11 against a local 0.20 in the uncoupled run.
- **Checkerboard sign.** The signed amplitude flips when the worst 16-vertex window shifts by one vertex; compare signs only at the same window start.
- **Temperatures from conserved variables** must subtract the kinetic energy of all velocity components. The outflow along the sheet (v\_y) is large, and leaving it out once inflated T by more than ten times.
- **Boundary conditions.** The existing `simpleMirrorBC` is a linear extrapolation (ghost = 2·A(boundary) − A(interior)), not a mirror, and its comment mismatches its code. The branch uses `simpleZeroGradientBC` for open boundaries and the new `reflectionBC` for the symmetry plane.
- **Snapshot timing.** Both codes write at the first step at or after each output time (up to one step late), and SAMS writes the final snapshot twice.
- **Do not reintroduce** the shock-limiter threshold parameter (tested and reverted; see what is established). Keep the PETSc Rosenbrock-W workaround in `CheckStage`, which restores the Jacobian lag after a rejected stage; without it every retry reuses a stale Jacobian.
- **Patches.** Earlier patches failed to apply whenever they were built from a copy that differed from the branch in a single setting. Build directly on the branch, and leave run settings (`sh21`, flags) as you find them.
