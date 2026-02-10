# For existing LARE Users

The existing Fortran code LARE3D has been in use for many years, and the same solver is available in SAMS. If you are an existing LARE user then there is a compatability layer built into SAMS to provide the same general interface as the Fortran code.

## SAMS warning

Note that SAMS is not currently a working, released code. It is currently at version 0.0.3 so should be considered to be alpha or even pre-alpha software. We believe that SAMS-LARE is giving correct results, but we cannot be sure that there are not edge cases or bugs that we have not yet found. Please do not use SAMS for production at this time.

## Basic idea

In order to fit the SAMS framework, the LARE code has been split into two parts. The LARE solver itself, which is a C++ implementation of the LARE3D algorithm, and a set of functions that are used to set up the initial conditions, control variables and boundary conditions for the solver. All of these are set up in much the same way as the Fortran code.

As a LARE user using SAMS, you will only need to edit the files in `src/InitialConditions/LARE` directory to set up your problem, and then rebuild the code. 

It is worth pointing out that this is considered a "fallback" option for existing LARE users. A more modern approach to running simulations with SAMS does exist.

## Control variables

The `control_variables` function from LARE is in the `src/InitialConditions/LARE/control.cpp` file. This will look very familiar to existing LARE users, except for the fact that all of the variables are now members of a `LARE::LARE3D::simulationData` class. An instance of this class called `data` is passed to the `control_variables` function. In there, you simply need to set the variables as you would in the Fortran code, but with `data.` in front of each variable name. For example, to set the number of grid points in the x direction, you would write `data.nx = 100;` instead of `nx = 100`.

## Initial conditions

Initial conditions are also set up in the `src/InitialConditions/LARE/control.cpp` file in the `initial_conditions` function. Again, you will see that the variables are members of the `data` class, but you can no longer set them up as easily as you can in Fortran. This is because SAMS uses performance portability tools, and so your data may be on your GPU and cannot be accessed using normal C++ approaches.

While you should **NOT** write initial conditions using normal loops like you would in Fortran, if you did it would look something like this:

```cpp
for (int ix = 1; ix < data.nx; ++ix)
{    for (int iy = 1; iy < data.ny; ++iy)
    {        for (int iz = 1; iz < data.nz; ++iz)
        {
            data.rho(ix,iy,iz) = 1.0;
            data.energy_electron(ix,iy,iz) = 1.0;
            data.energy_ion(ix,iy,iz) = 1.0;
        }
    }
}
```
You will notice that the loop indices are the same as in the Fortran code, so density and specific internal energy run (1->nx, 1->ny, 1->nz), velocities run (0->nx, 0->ny, 0->nz) and bx runs (0->nx, 1->ny, 1->nz), by runs (1->nx, 0->ny, 1->nz) and bz runs (1->nx, 1->ny, 0->nz). The xc, yc, zc and xb, yb, zb axis arrays are still present as in Fortran. The same indexing is used in SAMS as in LARE3D, although you will notice that my loops are ordered in reverse compared to the Fortran code. This is because SAMS uses a different memory layout to the Fortran code, and so the loops need to be ordered differently to achieve good performance. While you should not write your code like this, it will work so long as you never want to run anywhere other than on your system's CPU.

The correct way to write initial conditions in SAMS is to use the `pw::applyKernel` function. This takes a function kernel describing the operations to be performed on a single grid point, and applies it to the entire grid in a performance portable way. The kernel would normally be written as a lambda function, and the grid is specified using `pw::Range` objects. For example, the above initial conditions would be written as:

```cpp
pw::applyKernel(
    LAMBDA (T_indexType ix, T_indexType iy, T_indexType iz)
    {
        data.rho(ix,iy,iz) = 1.0;
        data.energy_electron(ix,iy,iz) = std::sqrt(xc(ix)*xc(ix) + yc(iy)*yc(iy) + zc(iz)*zc(iz));
        data.energy_ion(ix,iy,iz) = 0.0;
    },
    data.xcLocalDomainRange, data.ycLocalDomainRange, data.zcLocalDomainRange);
```

As you can see the kernel lambda itself looks very similar to the loop body of the previous example, but it is now wrapped in a call to `pw::applyKernel` and the loop indices become the arguments of the lambda function. The grid is specified using the `data.xcLocalDomainRange`, `data.ycLocalDomainRange` and `data.zcLocalDomainRange` objects, which specify the range of indices for each direction. The "LocalDomainRange" objects represent the range of indices for the computational domain local to the current MPI rank not including ghost cells, so they are the correct ranges to use for setting up initial conditions. The xc, yc and zc prefixes mean that these are for variables defined at the cell centres, so they run from 1 to nx, 1 to ny and 1 to nz. For variables defined at the cell faces, you would use the `data.xbLocalDomainRange`, `data.ybLocalDomainRange` and `data.zbLocalDomainRange` objects instead, which run from 0 to nx, 0 to ny and 0 to nz.

For simple initial conditions like rho and energy\_ion above, there is a function to make things simpler. For example, to set the entire of density to 1, use the assign function like this:

```cpp
pw::assign(data.rho, 1.0);
```
This will set the entire `data.rho` array to 1.0 in a performance portable way. You can assign values to arrays, and you can also assign arrays to other arrays using the same function. Just as in Fortran you can also slice arrays, so if you wanted to set the density to 1.0 in the left half of the domain and 2.0 in the right half of the domain, you could write:

```cpp
pw::assign(data.rho, 1.0, pw::Range(0,data.nx/2), pw::Range(0,data.ny), pw::Range(0,data.nz));
pw::assign(data.rho, 2.0, pw::Range(data.nx/2+1,data.nx), pw::Range(0,data.ny), pw::Range(0,data.nz));
```
Note that the last 3 parameters are a Range object - here we specify start and end directly, but we could also use the built-in LocalDomainRange for y and z that we just saw.

### What about initial conditions that require integration along a direction?

Yeah, that is trickier. We should probably write something here about how to do this. We haven't though.

## Boundary conditions

### Built in boundary conditions

You can select from built in boundary conditions in the control_variables function just as you can in the Fortran code, albeit with `BCType::` in front of the boundary condition type. For example, to set the left x boundary to be a fixed boundary, you would write `data.bc_x_left = BCType::OTHER;` instead of `bc_x_left = BC_OTHER`. The same applies for the other boundaries and other boundary condition types.

### Custom boundary conditions

The boundary conditions are defined in the `src/InitialConditions/LARE/boundary.cpp` file. The functions are not quite the same as in LARE because SAMS fundamentally requires separate boundary conditions for separate variables, so there are separate functions for setting boundary conditions for density, energy_ion, energy_electron, each component of velocity and remap velocity and each component of the magnetic field. You can set up your own custom boundary conditions by editing these functions, and you can set different boundary conditions for different variables if you wish. The name of the function indicates which variable and which boundary it is for. For example, the function `density_bcs` is for setting the boundary conditions for density. Inside the function you can see the systems for checking boundary types and domain edges. Simply copy this approach and write your own code to set up the boundary conditions for each variable and each boundary as you wish.

 Set up your boundary conditions using the same approach as for initial conditions, but only applying the kernel to the ghost cells instead of the entire grid. Once again, you should use the performance portability layer to set the boundary conditions, and not normal loops. Just as there are `xcLocalDomainRange` range objects describing the domain, there are the following range objects describing the ghost cells for each boundary:

* `data.xcminBCRange` - range of indices for the ghost cells on the minimum x boundary for cell centred variables
* `data.xcmaxBCRange` - range of indices for the ghost cells on the maximum x boundary for cell centred variables
* `data.ycminBCRange` - range of indices for the ghost cells on the minimum y boundary for cell centred variables
* `data.ycmaxBCRange` - range of indices for the ghost cells on the maximum y boundary for cell centred variables
* `data.zcminBCRange` - range of indices for the ghost cells on the minimum z boundary for cell centred variables
* `data.zcmaxBCRange` - range of indices for the ghost cells on the maximum z boundary for cell centred variables
* `data.xbminBCRange` - range of indices for the ghost cells on the minimum x boundary for cell face variables
* `data.xbmaxBCRange` - range of indices for the ghost cells on the maximum x boundary for cell face variables
* `data.ybminBCRange` - range of indices for the ghost cells on the minimum y boundary for cell face variables
* `data.ybmaxBCRange` - range of indices for the ghost cells on the maximum y boundary for cell face variables
* `data.zbminBCRange` - range of indices for the ghost cells on the minimum z boundary for cell face variables
* `data.zbmaxBCRange` - range of indices for the ghost cells on the maximum z boundary for cell face variables

So to set density to 1.0 in the ghost cells on the minimum x boundary, you would write:

```cpp
pw::assign(data.rho(data.xcminBCRange, data.ycLocalRange, data.zcLocalRange), 1.0);
```

Note the here I am using `data.ycLocalRange` and `data.zcLocalRange` rather than `data.ycLocalDomainRange` and `data.zcLocalDomainRange`. This is because the boundary conditions should set all values along the other edges, not just the values in the computation domain.

More complicated boundary conditions can be set up using the same kernels as the initial conditions, but only applying them to the ghost cell ranges instead of the entire domain. For example, to set a more complicated boundary condition on the minimum x boundary, you could write:

```cpp
pw::applyKernel(
    LAMBDA (T_indexType ix, T_indexType iy, T_indexType iz)
    {
        data.rho(ix,iy,iz) = ix;
    }, data.xcminBCRange, data.ycLocalRange, data.zcLocalRange);
```

### Time in boundary conditions

Each boundary condition function has an argument called `time` which is a `SAMS::timeState` object. To get the current simulation time, use the `time.time` variable. Remember that the `vx1`, `vy1` and ,`vz1` variables have to be set at half of the timestep (`time.dt`) ahead of the other variables, just as in classic LARE.

## Parameterising your problem

Currently SAMS does not have an input deck parser. When one is added in the future it will be possible to parameterise your problem using an input deck. The documentation for that will go here.


## Running your simulation

Once you have set up those elements of the Lare initial conditions, and rebuilt that part of the code, there is nothing else to do. You simply run SAMS activating the LARE3D and LAREInitialConditions packages, i.e.:
```bash
./lare3d LARE3D LAREInitialConditions
```
