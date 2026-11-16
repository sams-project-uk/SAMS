# Example Alternate Solvers

SAMS includes two small MHD solvers, HLLE3D and KT3D, alongside LARE3D. They exist to show how more than one core solver can be built into SAMS, and are not intended for production use.

## The solvers

Both solvers evolve the conserved variables of ideal MHD on a cell centred grid, in normalised units with mu0 = 1. Both use MUSCL reconstruction, Toth's flux-CT to keep the divergence of the magnetic field zero, and a two stage Runge-Kutta step. They differ only in their fluxes:

* `HLLE3D` - Uses the HLLE approximate Riemann solver
* `KT3D` - Uses the Kurganov-Tadmor central scheme

The variables are `rho`, the mass density, `mx`, `my` and `mz`, the momentum density, `energy`, the total energy density, and `bx`, `by` and `bz`, the magnetic field. The momentum density is written to output as `px`, `py` and `pz`.

The ratio of specific heats and the Courant number are set by `gas_gamma` and `cfl` in the `HLLE` block of the deck for HLLE3D, and in the `KT` block for KT3D.

The two solvers share a single data pack, `conservedSolver::simulationData`, so only one of them can be active in a run.

## Initial conditions

* `HLLEOrszagTang` - The Orszag-Tang vortex
* `conservedVariableDeckInitialConditions` - Initial and boundary conditions read from the deck, for either solver. The `initial_conditions` block sets `rho`, `px`, `py`, `pz`, `energy`, `bx`, `by` and `bz`, and variables that it does not set are 0. Boundary conditions are set in the `boundaries` block in the same way as for `LAREDeckInitialConditions`

For example, to run the Orszag-Tang vortex with HLLE3D:

```bash
./sams {output_directory} --packages=HLLE3D,HLLEOrszagTang
```

## Example deck

This is the SoundWaveHLLE input deck, which drives a sound wave pulse in from the lower x boundary using HLLE3D. The boundary expressions use the values of the variables on the last domain plane at the boundary, `rhoBoundary`, `pxBoundary` and `energyBoundary`.

```
begin:SAMS
  packages = (HLLE3D, conservedVariableDeckInitialConditions)
end:SAMS

begin:HLLE
  gas_gamma = 5 / 3
end:HLLE

# Sound speed is sqrt(gamma * (gamma - 1) * e) with specific internal energy e = 1
# v is the driven velocity, a Mach 0.05 pulse, Gaussian in y about the middle of the face and in time about 0.1
begin:constants
  c_s = sqrt(gas_gamma * (gas_gamma - 1))
  v = 0.05 * c_s * gauss(y, 0.5, 0.05) * gauss(time, 0.1, 0.03)
end:constants

# The pulse peaks at time 0.1 and then crosses the domain
begin:time
  t_end = 0.1 + 1 / c_s
end:time

begin:diagnostics
  begin:schedule
    name = default
    collections = (default)
    dt_snapshots = t_end / 20
  end:schedule
end:diagnostics

begin:domain
  nx = 256
  ny = 256
  x_min = 0
  x_max = 1
  y_max = 1
end:domain

# energy is the total energy density
begin:initial_conditions
  rho = 1
  energy = 1
end:initial_conditions

begin:boundaries
  x = zero_gradient
  y = periodic
  z = periodic
  begin:px
    x_min = expression(rhoBoundary * v)
  end:px
  # Keeps the internal and magnetic energy at the boundary and replaces the kinetic energy with that of v
  begin:energy
    x_min = expression(energyBoundary - 0.5 * pxBoundary^2 / rhoBoundary + 0.5 * rhoBoundary * v^2)
  end:energy
end:boundaries
```
