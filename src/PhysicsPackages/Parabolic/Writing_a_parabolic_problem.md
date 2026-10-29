# Writing a problem package for the parabolic solver

## Purpose

This document gives a brief overview of how to write a problem package for the parabolic solver. The parabolic solver is designed to solve problems of the form

$$
\beta \frac{\partial u}{\partial t} = \mathcal{L} u
$$

where $u$ is the field being evolved, $\beta$ is the prefactor of the time derivative, and $\mathcal{L}$ is a linear operator whose stencil reaches at most one cell from the centre in each direction, which is up to 27 points. Both $\beta$ and $\mathcal{L}$ can vary in space and time. The solver can cope with any such operator provided that the matrix representing $\mathcal{L}$ is diagonally dominant, with a negative diagonal. The included problems are for diffusion, where

$$
\mathcal{L} u = \nabla \cdot \left( \mathbf{A} \cdot \nabla u \right)
$$

and $\mathbf{A}$ is a diffusion tensor. For example, the Braginskii diffusion problem has a tensor that depends on local temperature, density, and magnetic field. Isotropic thermal conduction uses a 7-point stencil, and Braginskii thermal conduction uses a 19-point stencil.

## Levels

A solver may solve the equation on a series of grids of different resolution, as the multigrid solver does. These grids are called levels. The finest level is the grid of the simulation, and each coarser level has about half as many cells in each direction. A solver that uses only the grid of the simulation, such as the PETSc solvers, has one level.

The level type is defined by the solver, and its full layout is in the Doxygen documentation of the solver. These are the members that a problem uses:

- `data`: the `levelData` of the problem for that level.
- `rhs`: on the finest level, the old field.
- `gridLevel`: the index of the level, counted from 0 for the finest level.
- `dxc`, `dyc`, `dzc`: the widths of the cells in each direction.
- `nx`, `ny`, `nz`: the number of cells in each direction.
- `xcLocalDomainRange`, `ycLocalDomainRange`, `zcLocalDomainRange`: the ranges of the cells.
- `xbLocalDomainRange`, `ybLocalDomainRange`, `zbLocalDomainRange`: the ranges of the edges between the cells.

## The problem specification

A problem package is a class template, for example `braginskii<T_species, T_eos>` or `isotropic<T_species, T_eos>`, that tells the solver which equation to solve. The solvers, `mg`, `petscMatrix` and `petscNoMatrix` in `SAMS::parabolic`, are templates on the problem class and use the members described here. The problem class must be trivially copyable, because it is copied into kernels.

### Mandatory fields

A hook is called with the arguments that its entry lists, followed by any data packs that it names as parameters. In the list below, `{packs}` stands for those packs: it can be zero or more items from the `dataPack` entries of the solver, and they are found by type. A hook that needs none does not name any.

- `name`: a `SAMS::constexprName` that identifies the problem. Each solver appends its own suffix to make the name of its package, so `isotropic_electron` gives `isotropic_electron_mg`.
- `memoryName`: a `SAMS::constexprName` that the arrays and axes of the problem are registered under. It leaves out what is not needed to tell the memory apart, such as the species, so that problems for different species share it.
- `stencilSize`: a `static constexpr int` giving the number of entries of the stencil, which is at most 27.
- `dataPack`: the type of the data pack that the problem needs from the runner, or a `SAMS::dataPacks::multiPack` of several. If you need no data pack, use void.
- `levelData`: a default constructible type holding the arrays that the problem needs for one grid. The multigrid solver keeps one for each level.
- `stencil`: the type of the operator object that the problem returns for a grid. It provides `coefficients(i, j, k, a)` and `beta(i, j, k)`.
- `fieldName()`: a `static constexpr` function returning the name of the variable that is evolved.
- `stencilCellOffset(s, di, dj, dk)`: a `static` device function giving the offset of entry `s` of the stencil, which includes the {0, 0, 0} centre element, in any order. When using the helper functions to provide conservative flux based updates, it must be `constexpr`.
- `registerVariables(harness, {packs})`: registers the variables of the problem. It is called at the same time as `registerVariables` in the runner.
- `registerSecondaryVariables(harness, axisLevels, {packs})`: registers the arrays of `levelData` on every level.
- `getVariables(harness, {packs})`: gets the variables of the problem from the variable registry. It is called at the same time as `getVariables` in the runner.
- `getSecondaryVariables(harness, levels, {packs})`: gets the arrays of `levelData` for every level.
- `prepareSolve(levels, {packs})`: called at the start of each solve. It populates the arrays of `levelData` that are needed, on every level.
- `initialConditions(harness, {packs})`: called at the same time as `initialConditions` in the runner. An initial conditions package normally sets the constants of the problem, so this can be empty.
- `setBoundaryConditions(harness, {packs})`: sets the boundary conditions of the variables of the problem. It is called at the same time as `setBoundaryConditions` in the runner.
- `beforeSolve(harness, {packs})` and `afterSolve(harness, {packs})`: called by the solver before and after the solve of each timestep, not between iterations.
- `makeStencil(level)`: returns the operator of a level.

### `name`

`name` is a `static constexpr` `SAMS::constexprName` that identifies the problem:

```cpp
inline static constexpr auto name = SAMS::constexprName("diffusion");
```

The solver appends its own suffix to make the name of its package.

### `memoryName`

`memoryName` is a `static constexpr` `SAMS::constexprName` that the arrays and axes of the problem are registered under. Each species has its own `name`, but the solves of different problems do not overlap, so the species can share the arrays by leaving the species out of `memoryName`:

```cpp
inline static constexpr auto memoryName = SAMS::constexprName("diffusion");
```

### `stencilSize`

`stencilSize` is a `static constexpr int` giving the number of entries of the stencil, which are the cell and its neighbours. It must not be more than 27, because the stencil reaches at most one cell in each direction. For the 7-point stencil of isotropic conduction:

```cpp
static constexpr int stencilSize = 7;
```

### `dataPack`

`dataPack` is the type of the data pack that the problem needs from the runner. The runner holds one instance of each pack type, shared by every package that names it. If the problem needs more than one pack, `dataPack` is a `SAMS::dataPacks::multiPack` of them:

```cpp
using dataPack = SAMS::dataPacks::multiPack<diffusionData, materialData>;
```

Just as in normal SAMS packages, this can be custom types for this problem or a registered data pack of another package.

The constants of the problem belong in a class called `parameters` inside the problem class, which is one of the packs. An initial conditions package can then set them.

If the problem needs the timestep, add `SAMS::timeState` to the pack. A hook that names it as a parameter gets the timestep of the runner, which has `time`, `dt` and `step`. For example, `beforeSolve(SAMS::harness &harness, const SAMS::timeState &ts)` can use `ts.dt`.

```cpp
using dataPack = SAMS::dataPacks::multiPack<parameters, lareData, SAMS::timeState>;
```

### `levelData`

`levelData` is a default constructible type holding the arrays that the problem needs for one level that are not the primary variable ($u$). The solver keeps one instance for each level, which is why it is called level data.

It should hold what the stencil of a level is built from: the coefficients of the operator on that grid, including $\beta$. The arrays are registered by `registerSecondaryVariables`, found again by `getSecondaryVariables` and filled by `prepareSolve`. For example:

```cpp
using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;

struct levelData
{
	volumeArray conductivity;
	volumeArray beta;
};
```

### `stencil`

`stencil` is the type of the operator object that `makeStencil` returns for one level. It is copied into kernels, so it holds only arrays, and its functions are device functions. It provides two functions that the solvers call, and one optional function, each for the cell $(i, j, k)$:

- `coefficients(i, j, k, a)` fills `a` with the `stencilSize` coefficients of the discretised operator $\mathcal{L}$ in the cell. `a[s]` multiplies `u` in the cell at `stencilCellOffset(s, ...)`, and the entry whose offset is {0, 0, 0} is the cell itself. `parabolic::stencilView<problem>` wraps `a`, sets it to zero and finds an entry from its position, so `view[parabolic::at<1, 0, 0>]` is the coefficient of the cell at $i + 1$.
- `beta(i, j, k)` returns $\beta$ in the cell.
- `fluxWeights(i, j, k)` is optional, and the solvers do not call it. A problem written in the flux form provides it, so that `coefficients` can be built from it by a helper that guarantees conservation. It returns the weights of the fluxes out of the cell, as described in the flux form section of the advanced example.

```cpp
struct stencil
{
	volumeArray conductivity;
	volumeArray betaArray;

	DEVICEPREFIX void coefficients(size_t i, size_t j, size_t k, double *a) const;
	DEVICEPREFIX double beta(size_t i, size_t j, size_t k) const;
};
```

### `fieldName`

`fieldName` is a `static constexpr` function returning the name of the variable that is evolved, which is $u$. The solver gets the array and the grid associated with the named variable from the variable registry. The problem registers the variable in `registerVariables`, with one guard cell on each side:

```cpp
static constexpr const char *fieldName()
{
	return "temperature";
}
```

### `stencilCellOffset`

`stencilCellOffset(s, di, dj, dk)` is a `static` device function that sets `di`, `dj` and `dk` to the position, in cells, of entry `s` of the stencil relative to the cell. One entry must be {0, 0, 0}, which is the cell itself, and the order of the entries does not matter. So `s` runs from 0 to `stencilSize - 1`, and `a[s]` from `coefficients` multiplies `u(i + di, j + dj, k + dk)`. Every offset must be within one cell of the centre in each direction. For the 7-point stencil:

```cpp
DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
{
	const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
	di = table[s][0];
	dj = table[s][1];
	dk = table[s][2];
}
```

### `registerVariables`

`registerVariables(harness, {packs})` registers the variables that the problem owns with the variable registry of the harness. It is called at the same time as `registerVariables` in the runner. It must register the variable named by `fieldName`, with one guard cell on each side in each direction, so that the stencil can read its neighbours. For example:

```cpp
void registerVariables(SAMS::harness &harness)
{
	const int ghosts = 1;
	harness.variableRegistry.registerVariable<SAMS::T_dataType>(fieldName(), pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
}
```

### `registerSecondaryVariables`

`registerSecondaryVariables(harness, axisLevels, {packs})` registers the arrays of `levelData` on every level. `axisLevels` is the number of derived axes that the solver uses in each direction, which is `{0, 0, 0}` for a solver with one level. `parabolic::registerArrayLevels` registers an array under `memoryName`, on the axes of each level, and takes the stagger in each direction. An array that is only needed on the finest level is registered with `parabolic::registerArray`.

An array on the edges between cells is staggered by half a cell in the direction of the edges. Its index $i$ is the edge between cells $i$ and $i + 1$, so index 0 is the lower edge of the domain. For example:

```cpp
void registerSecondaryVariables(SAMS::harness &harness, const std::array<int, 3> &axisLevels)
{
	const std::string memory = memoryName.c_str();
	parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cx", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
	parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "beta", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
}
```

### `getSecondaryVariables`

`getSecondaryVariables(harness, levels, {packs})` gets the arrays that `registerSecondaryVariables` registered and stores them in the `levelData` of each level. `parabolic::fillArrayLevels` does this for one array on every level. It takes a function that returns the array of a level, because only the problem knows where in `levelData` the array is. `parabolic::fillArray` does it for one level, which is the finest by default.

```cpp
void getSecondaryVariables(SAMS::harness &harness, std::vector<level> &levels)
{
	const std::string memory = memoryName.c_str();
	parabolic::fillArrayLevels(harness, memory, "cx", levels, [](level &l) -> volumeArray & { return l.data.conductance.cx; });
	parabolic::fillArrayLevels(harness, memory, "beta", levels, [](level &l) -> volumeArray & { return l.data.beta; });
}
```

Here and in the later examples, `level` is the level type of the solver.

### `prepareSolve`

`prepareSolve(levels, {packs})` is called at the start of each solve, after `beforeSolve` and after the solver has applied the boundary conditions of the evolved variable. Its purpose is to populate the arrays of `levelData` that are needed, on every level, from the old field. The stencil of each level then computes the coefficients of the discretised operator $\mathcal{L}$ and $\beta$ from those arrays.

### `initialConditions`

`initialConditions(harness, {packs})` is called at the same time as `initialConditions` in the runner. If the `parameters` of the problem are to be filled from the KVS, they should be set here.

### `setBoundaryConditions`

`setBoundaryConditions(harness, {packs})` sets the boundary conditions of the variables that the problem registered. It is called at the same time as `setBoundaryConditions` in the runner. The solver applies the boundary conditions of the evolved variable before each solve.

### `beforeSolve`

`beforeSolve(harness, {packs})` is called by the solver once at the start of each timestep, and not between iterations. It runs before the solver applies the boundary conditions of the evolved variable and before `prepareSolve`. Its purpose is to set the evolved variable $u$ from the variables of the rest of the simulation. For example, the thermal conduction problems evolve the temperature, so it converts the energy to the temperature:

```cpp
void beforeSolve(SAMS::harness &harness, const lareData &lare)
{
	auto &varRegistry = harness.variableRegistry;
	// The energy needs its boundary conditions before it is converted
	varRegistry.applyBoundaryConditions(species::fieldName());
	volumeArray energy, temperature;
	varRegistry.fillPPArray(species::fieldName(), energy);
	varRegistry.fillPPArray(fieldName(), temperature);
	const SAMS::variableDef &def = varRegistry.getVariable(fieldName());
	const auto &eos = lare.eos;
	// Convert the energy to the temperature, including the guard cells
	auto convert = LAMBDA(size_t i, size_t j, size_t k)
	{
		temperature(i, j, k) = eos.getTemperature(LARE::eosEnergy(energy(i, j, k)));
	};
	pw::applyKernel(convert, def.getDimension(0).getLocalRange(SAMS::staggerType::CENTRED), def.getDimension(1).getLocalRange(SAMS::staggerType::CENTRED), def.getDimension(2).getLocalRange(SAMS::staggerType::CENTRED));
	pw::fence();
}
```

### `afterSolve`

`afterSolve(harness, {packs})` is called by the solver once after the solve of each timestep, and not between iterations. Its purpose is to set the variables of the rest of the simulation from the evolved variable $u$, which is the opposite of `beforeSolve`. For example, the thermal conduction problems convert the temperature back to the energy:

```cpp
void afterSolve(SAMS::harness &harness, const lareData &lare)
{
	auto &varRegistry = harness.variableRegistry;
	// The energy is set from the temperature that was evolved
	volumeArray energy, temperature;
	varRegistry.fillPPArray(species::fieldName(), energy);
	varRegistry.fillPPArray(fieldName(), temperature);
	const SAMS::variableDef &def = varRegistry.getVariable(fieldName());
	const auto &eos = lare.eos;
	// Convert the temperature back to the energy in the cells of the domain, not the guard cells
	auto convert = LAMBDA(size_t i, size_t j, size_t k)
	{
		energy(i, j, k) = eos.getEnergy(LARE::eosTemperature(temperature(i, j, k)));
	};
	pw::applyKernel(convert, def.getDimension(0).getLocalDomainRange(SAMS::staggerType::CENTRED), def.getDimension(1).getLocalDomainRange(SAMS::staggerType::CENTRED), def.getDimension(2).getLocalDomainRange(SAMS::staggerType::CENTRED));
	pw::fence();
}
```

### `makeStencil`

`makeStencil(level)` returns the `stencil` of a level. The solver calls it when it needs the operator of that level, after `prepareSolve` has populated the arrays of `levelData`. It copies those arrays from the level into the stencil, together with anything else that `coefficients` needs, such as the cell widths of the level. The stencil is copied into kernels, so it holds the arrays and not the level. For example:

```cpp
stencil makeStencil(const level &l) const
{
	stencil s;
	s.conductivity = l.data.conductivity;
	s.betaArray = l.data.beta;
	return s;
}
```

## Worked example: isotropic diffusion

This example writes a complete problem for isotropic diffusion with a constant conductivity $\kappa$ and a constant $\beta$, and no flux limiter. The class is called `diffusion`.

### The equation

For a constant $\kappa$ the equation is

$$
\beta \frac{\partial u}{\partial t} = \kappa \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right)
$$

Let the grid have constant cell widths $\Delta x$, $\Delta y$ and $\Delta z$. The centred finite difference for the second derivative is

$$
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1,j,k} - 2 u_{i,j,k} + u_{i-1,j,k}}{\Delta x^2}
$$

and the same in $y$ and $z$. The solver wants the right hand side multiplied by the cell volume $V = \Delta x \Delta y \Delta z$, so that the coefficients are integrated over the cell. The coefficients $a_s$ of $u$ in the cell and its six neighbours are then

$$
a_1 = a_2 = \frac{\kappa V}{\Delta x^2} \, , \qquad a_3 = a_4 = \frac{\kappa V}{\Delta y^2} \, , \qquad a_5 = a_6 = \frac{\kappa V}{\Delta z^2}
$$

for $u_{i-1,j,k}$ and $u_{i+1,j,k}$, $u_{i,j-1,k}$ and $u_{i,j+1,k}$, and $u_{i,j,k-1}$ and $u_{i,j,k+1}$, and for the cell itself

$$
a_0 = -\left( a_1 + a_2 + a_3 + a_4 + a_5 + a_6 \right)
$$

These are the coefficients that `coefficients` returns, in the order of `stencilCellOffset`. The diagonal $a_0$ is negative and is minus the sum of the neighbours, so the matrix that the solver builds from them is diagonally dominant.

### Step 1: the identity of the problem

The problem needs a `name` and a `memoryName`, a class of parameters that is its data pack, the name of the field, the number of entries of the stencil, which is 7 for the cell and its six face neighbours, and `stencilCellOffset`, which gives the position of each entry, including the cell. The order does not matter, because the coefficients are set by position:

```cpp
struct diffusion
{
	using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

	inline static constexpr auto name = SAMS::constexprName("diffusion");
	inline static constexpr auto memoryName = SAMS::constexprName("diffusion");

	struct parameters
	{
		double kappa;
		double beta;
	};
	using dataPack = parameters;

	static constexpr const char *fieldName()
	{
		return "u";
	}

	static constexpr int stencilSize = 7;

	DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
	{
		const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
		di = table[s][0];
		dj = table[s][1];
		dk = table[s][2];
	}
};
```

The stencil is the cell and the six cardinal points in each dimension, which are the points that the discretised second derivative uses. `stencilCellOffset` lists the cell and the six cardinal points, which are $i - 1$, $i + 1$, $j - 1$, $j + 1$, $k - 1$ and $k + 1$. Their order does not matter, because `coefficients` sets each coefficient by the position of its cell.

### Step 2: the stencil

The stencil can compute the coefficients directly from the cell widths, so no arrays of coefficients are needed in this simple case. `makeStencil` only has the level, so `levelData` holds the constants, and `prepareSolve` copies them from the parameters into every level. `makeStencil` copies the constants and the cell widths of the level into the stencil. `coefficients` is the equation above, and `beta` is constant. There is no `fluxWeights`, because the solvers only call those two, and `fluxWeights` is only for the flux form described in the advanced example:

```cpp
struct diffusion
{
	using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

	inline static constexpr auto name = SAMS::constexprName("diffusion");
	inline static constexpr auto memoryName = SAMS::constexprName("diffusion");

	struct parameters
	{
		double kappa;
		double beta;
	};
	using dataPack = parameters;

	static constexpr const char *fieldName()
	{
		return "u";
	}

	static constexpr int stencilSize = 7;

	DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
	{
		const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
		di = table[s][0];
		dj = table[s][1];
		dk = table[s][2];
	}

	// The constants are copied to every level, because makeStencil only has the level
	struct levelData
	{
		double kappa;
		double beta;
	};

	struct stencil
	{
		lineArray dx, dy, dz;
		double kappa;
		double betaValue;

		DEVICEPREFIX void coefficients(size_t i, size_t j, size_t k, double *a) const
		{
			// The cell volume, and the distances to the centres of the neighbours in each direction
			const double v = dx(i) * dy(j) * dz(k);
			const double hxm = 0.5 * (dx(i - 1) + dx(i)), hxp = 0.5 * (dx(i) + dx(i + 1));
			const double hym = 0.5 * (dy(j - 1) + dy(j)), hyp = 0.5 * (dy(j) + dy(j + 1));
			const double hzm = 0.5 * (dz(k - 1) + dz(k)), hzp = 0.5 * (dz(k) + dz(k + 1));
			using parabolic::at;
			// The view finds each coefficient from the position of its cell relative to the cell, and starts them at zero
			parabolic::stencilView<diffusion> coefficient(a);
			coefficient[at<-1, 0, 0>] = kappa * v / (dx(i) * hxm);
			coefficient[at<1, 0, 0>] = kappa * v / (dx(i) * hxp);
			coefficient[at<0, -1, 0>] = kappa * v / (dy(j) * hym);
			coefficient[at<0, 1, 0>] = kappa * v / (dy(j) * hyp);
			coefficient[at<0, 0, -1>] = kappa * v / (dz(k) * hzm);
			coefficient[at<0, 0, 1>] = kappa * v / (dz(k) * hzp);
			coefficient[at<0, 0, 0>] = -(coefficient[at<-1, 0, 0>] + coefficient[at<1, 0, 0>] + coefficient[at<0, -1, 0>] + coefficient[at<0, 1, 0>] + coefficient[at<0, 0, -1>] + coefficient[at<0, 0, 1>]);
		}

		DEVICEPREFIX double beta(size_t i, size_t j, size_t k) const
		{
			return betaValue;
		}
	};

	stencil makeStencil(const level &l) const
	{
		stencil s;
		s.dx = l.dxc;
		s.dy = l.dyc;
		s.dz = l.dzc;
		s.kappa = l.data.kappa;
		s.betaValue = l.data.beta;
		return s;
	}

	void prepareSolve(std::vector<level> &levels, const parameters &params) const
	{
		for (level &l : levels)
		{
			l.data.kappa = params.kappa;
			l.data.beta = params.beta;
		}
	}
};
```

### Step 3: the variables

The problem registers the field, with one guard cell on each side. This registration is only needed because we want this problem to be able to function without a "core simulation" to provide the basic variables. It has no arrays of `levelData`, so the hooks that register and get them are empty, and so is `getVariables`:

```cpp
struct diffusion
{
	using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

	inline static constexpr auto name = SAMS::constexprName("diffusion");
	inline static constexpr auto memoryName = SAMS::constexprName("diffusion");

	struct parameters
	{
		double kappa;
		double beta;
	};
	using dataPack = parameters;

	static constexpr const char *fieldName()
	{
		return "u";
	}

	static constexpr int stencilSize = 7;

	DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
	{
		const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
		di = table[s][0];
		dj = table[s][1];
		dk = table[s][2];
	}

	// The constants are copied to every level, because makeStencil only has the level
	struct levelData
	{
		double kappa;
		double beta;
	};

	struct stencil
	{
		lineArray dx, dy, dz;
		double kappa;
		double betaValue;

		DEVICEPREFIX void coefficients(size_t i, size_t j, size_t k, double *a) const
		{
			// The cell volume, and the distances to the centres of the neighbours in each direction
			const double v = dx(i) * dy(j) * dz(k);
			const double hxm = 0.5 * (dx(i - 1) + dx(i)), hxp = 0.5 * (dx(i) + dx(i + 1));
			const double hym = 0.5 * (dy(j - 1) + dy(j)), hyp = 0.5 * (dy(j) + dy(j + 1));
			const double hzm = 0.5 * (dz(k - 1) + dz(k)), hzp = 0.5 * (dz(k) + dz(k + 1));
			using parabolic::at;
			// The view finds each coefficient from the position of its cell relative to the cell, and starts them at zero
			parabolic::stencilView<diffusion> coefficient(a);
			coefficient[at<-1, 0, 0>] = kappa * v / (dx(i) * hxm);
			coefficient[at<1, 0, 0>] = kappa * v / (dx(i) * hxp);
			coefficient[at<0, -1, 0>] = kappa * v / (dy(j) * hym);
			coefficient[at<0, 1, 0>] = kappa * v / (dy(j) * hyp);
			coefficient[at<0, 0, -1>] = kappa * v / (dz(k) * hzm);
			coefficient[at<0, 0, 1>] = kappa * v / (dz(k) * hzp);
			coefficient[at<0, 0, 0>] = -(coefficient[at<-1, 0, 0>] + coefficient[at<1, 0, 0>] + coefficient[at<0, -1, 0>] + coefficient[at<0, 1, 0>] + coefficient[at<0, 0, -1>] + coefficient[at<0, 0, 1>]);
		}

		DEVICEPREFIX double beta(size_t i, size_t j, size_t k) const
		{
			return betaValue;
		}
	};

	stencil makeStencil(const level &l) const
	{
		stencil s;
		s.dx = l.dxc;
		s.dy = l.dyc;
		s.dz = l.dzc;
		s.kappa = l.data.kappa;
		s.betaValue = l.data.beta;
		return s;
	}

	void prepareSolve(std::vector<level> &levels, const parameters &params) const
	{
		for (level &l : levels)
		{
			l.data.kappa = params.kappa;
			l.data.beta = params.beta;
		}
	}

	void registerVariables(SAMS::harness &harness)
	{
		const int ghosts = 1;
		harness.variableRegistry.registerVariable<SAMS::T_dataType>(fieldName(), pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
	}

	void registerSecondaryVariables(SAMS::harness &harness, const std::array<int, 3> &axisLevels)
	{
	}

	void getVariables(SAMS::harness &harness)
	{
	}

	void getSecondaryVariables(SAMS::harness &harness, std::vector<level> &levels)
	{
	}
};
```

### Step 4: the remaining hooks

The parameters are set by an initial conditions package, and the problem has no boundary conditions of its own. The field is $u$ itself, so there is nothing to convert before or after the solve. These hooks are still required, and are empty. This is the complete class:

```cpp
struct diffusion
{
	using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

	inline static constexpr auto name = SAMS::constexprName("diffusion");
	inline static constexpr auto memoryName = SAMS::constexprName("diffusion");

	struct parameters
	{
		double kappa;
		double beta;
	};
	using dataPack = parameters;

	static constexpr const char *fieldName()
	{
		return "u";
	}

	static constexpr int stencilSize = 7;

	DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
	{
		const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
		di = table[s][0];
		dj = table[s][1];
		dk = table[s][2];
	}

	// The constants are copied to every level, because makeStencil only has the level
	struct levelData
	{
		double kappa;
		double beta;
	};

	struct stencil
	{
		lineArray dx, dy, dz;
		double kappa;
		double betaValue;

		DEVICEPREFIX void coefficients(size_t i, size_t j, size_t k, double *a) const
		{
			// The cell volume, and the distances to the centres of the neighbours in each direction
			const double v = dx(i) * dy(j) * dz(k);
			const double hxm = 0.5 * (dx(i - 1) + dx(i)), hxp = 0.5 * (dx(i) + dx(i + 1));
			const double hym = 0.5 * (dy(j - 1) + dy(j)), hyp = 0.5 * (dy(j) + dy(j + 1));
			const double hzm = 0.5 * (dz(k - 1) + dz(k)), hzp = 0.5 * (dz(k) + dz(k + 1));
			using parabolic::at;
			// The view finds each coefficient from the position of its cell relative to the cell, and starts them at zero
			parabolic::stencilView<diffusion> coefficient(a);
			coefficient[at<-1, 0, 0>] = kappa * v / (dx(i) * hxm);
			coefficient[at<1, 0, 0>] = kappa * v / (dx(i) * hxp);
			coefficient[at<0, -1, 0>] = kappa * v / (dy(j) * hym);
			coefficient[at<0, 1, 0>] = kappa * v / (dy(j) * hyp);
			coefficient[at<0, 0, -1>] = kappa * v / (dz(k) * hzm);
			coefficient[at<0, 0, 1>] = kappa * v / (dz(k) * hzp);
			coefficient[at<0, 0, 0>] = -(coefficient[at<-1, 0, 0>] + coefficient[at<1, 0, 0>] + coefficient[at<0, -1, 0>] + coefficient[at<0, 1, 0>] + coefficient[at<0, 0, -1>] + coefficient[at<0, 0, 1>]);
		}

		DEVICEPREFIX double beta(size_t i, size_t j, size_t k) const
		{
			return betaValue;
		}
	};

	stencil makeStencil(const level &l) const
	{
		stencil s;
		s.dx = l.dxc;
		s.dy = l.dyc;
		s.dz = l.dzc;
		s.kappa = l.data.kappa;
		s.betaValue = l.data.beta;
		return s;
	}

	void prepareSolve(std::vector<level> &levels, const parameters &params) const
	{
		for (level &l : levels)
		{
			l.data.kappa = params.kappa;
			l.data.beta = params.beta;
		}
	}

	void registerVariables(SAMS::harness &harness)
	{
		const int ghosts = 1;
		harness.variableRegistry.registerVariable<SAMS::T_dataType>(fieldName(), pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
	}

	void registerSecondaryVariables(SAMS::harness &harness, const std::array<int, 3> &axisLevels)
	{
	}

	void getVariables(SAMS::harness &harness)
	{
	}

	void getSecondaryVariables(SAMS::harness &harness, std::vector<level> &levels)
	{
	}

	void initialConditions(SAMS::harness &harness)
	{
	}

	void setBoundaryConditions(SAMS::harness &harness)
	{
	}

	void beforeSolve(SAMS::harness &harness)
	{
	}

	void afterSolve(SAMS::harness &harness)
	{
	}
};
```

## Advanced example: coefficients stored in arrays

The class above computes the coefficients in the stencil from the cell widths. If the coefficients are expensive to compute, or depend on other quantities as the Braginskii conductivity does, a problem can compute them once per solve and store them in arrays of `levelData`, which the stencil then reads. This example does that for the same equation, with the conductances $c$ stored on every level.

### The flux form

The coefficients of a diffusion problem are the net flux into a cell through its six faces. A problem that is written in that form has its coefficients built by a helper, and is guaranteed to conserve $\sum V \beta u$. The stencil of such a problem supplies `fluxWeights(i, j, k)`, which the solvers do not call and which is only used by the helper below. It returns a `parabolic::fluxDivergence::faceWeights<problem>` for the cell. Its three members, `fluxWeightsX`, `fluxWeightsY` and `fluxWeightsZ`, hold the weights of the flux out of the cell in the positive direction, through its face on the high side in $x$, $y$ and $z$. A flux is the sum of its weights times the values of the cells.

A weight is found from the position of its cell relative to the cell, as in `w.fluxWeightsX[parabolic::at<1, 0, 0>]`, where `at<1, 0, 0>` is the cell at $i + 1$ and `at<0, 0, 0>` is the cell itself. The position is a template argument, so a position that is not in the stencil is a compile error. The weights start at zero, so only the cells that a flux uses need to be set.

`w.makeSymmetric()` sets the weights with a position of 1 in the direction of a face from those with a position of 0, so only the weights with a position of 0 need to be written. The weight of the cell itself and of the cells in line with it becomes minus the weight at 0, and the weight of a cell to the side becomes the same as at 0. It replaces whatever was in the weights on the high side.

In `faceWeights<problem>` and `coefficientsFromFluxes<problem>`, `problem` must be the class that encloses the stencil, which is the problem class itself, because the helpers read `stencilSize` and `stencilCellOffset` from it. Using another class gives the wrong stencil. `parabolic::fluxDivergence::coefficientsFromFluxes<problem>(stencil, i, j, k, a)` fills `a` from them. The net flux into a cell is the flux through the face on its low side, which is the flux through the high side of the cell before it, minus the flux through the face on its high side. So the helper calls `fluxWeights` for the cell and for the three cells before it. It adds the weights of the faces on the high side to `a` with a minus sign, and adds the weights of the faces on the low side, moved to the indexing of this cell, with a plus sign. Two cells that share a face use the same weights with opposite signs, so the coefficients of all of the cells add up to zero for every column.

Every cell that a flux uses must be an entry of the stencil, and `stencilCellOffset` must be `constexpr`, because the positions and the helper search it at compile time to find the entries. A problem that is not in this form needs no `fluxWeights`. It can write `coefficients` directly, but then it is not guaranteed to conserve.

### Step 1: the name, the parameters and the field

The problem needs a `name` and a `memoryName`, a class of parameters that is its data pack, the name of the field, and the number of entries of the stencil, which is 7 for the cell and its six face neighbours.

```cpp
struct diffusion
{
	using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
	using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

	inline static constexpr auto name = SAMS::constexprName("diffusion");
	inline static constexpr auto memoryName = SAMS::constexprName("diffusion");

	struct parameters
	{
		double kappa;
		double beta;
	};
	using dataPack = parameters;

	static constexpr const char *fieldName()
	{
		return "u";
	}

	static constexpr int stencilSize = 7;
};
```

### Step 2: the stencil

The solver builds its matrix from the stencil. `levelData` holds the arrays that the stencil is made from: the conductance of the faces in each direction, and $\beta$. `stencilCellOffset` gives the position of each entry, including the cell. The order does not matter, because the entries are found by position. Here we introduce conductance arrays `cx`, `cy` and `cz` that are staggered by half a cell in the direction of the edges, so that `cx(i,:,:)` is the conductance of the face between cells $i$ and $i + 1$. The stencil reads those arrays to compute the coefficients. `fluxWeights` gives the weights of the fluxes through the faces on the high side of the cell, `coefficients` is made from them by a helper, and `beta` gives $\beta$:

```cpp
struct diffusion
{
	using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
	using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

	inline static constexpr auto name = SAMS::constexprName("diffusion");
	inline static constexpr auto memoryName = SAMS::constexprName("diffusion");

	struct parameters
	{
		double kappa;
		double beta;
	};
	using dataPack = parameters;

	static constexpr const char *fieldName()
	{
		return "u";
	}

	static constexpr int stencilSize = 7;

	// The conductance of the faces is stored on the edges between cells. cx(i) is between cells i and i + 1
	struct levelData
	{
		volumeArray cx, cy, cz;
		volumeArray beta;
	};

	DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
	{
		const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
		di = table[s][0];
		dj = table[s][1];
		dk = table[s][2];
	}

	struct stencil
	{
		volumeArray cx, cy, cz;
		volumeArray betaArray;

		DEVICEPREFIX parabolic::fluxDivergence::faceWeights<diffusion> fluxWeights(size_t i, size_t j, size_t k) const
		{
			using parabolic::at;
			// All of the weights start at zero, and only the cells that a flux uses are set, found from their position relative to the cell
			parabolic::fluxDivergence::faceWeights<diffusion> w{};
			// The flux out of the cell through its face on the high side in x is the conductance of that face, which is cx(i), times
			// the difference between the cell and the next cell in x, which is at<1, 0, 0>. Only the cell itself is set here
			w.fluxWeightsX[at<0, 0, 0>] = cx(i, j, k);
			// The same in y and z, where the next cells are at<0, 1, 0> and at<0, 0, 1>
			w.fluxWeightsY[at<0, 0, 0>] = cy(i, j, k);
			w.fluxWeightsZ[at<0, 0, 0>] = cz(i, j, k);
			// The weights of the next cells are minus those of the cell, which makeSymmetric sets
			w.makeSymmetric();
			return w;
		}

		DEVICEPREFIX void coefficients(size_t i, size_t j, size_t k, double *a) const
		{
			// The helper makes the coefficients from the weights of the fluxes through the faces of this cell. It uses the fluxes
			// through the faces on the low side from the cells before it, so that both cells that share a face use the same flux
			parabolic::fluxDivergence::coefficientsFromFluxes<diffusion>(*this, i, j, k, a);
		}

		DEVICEPREFIX double beta(size_t i, size_t j, size_t k) const
		{
			return betaArray(i, j, k);
		}
	};
};
```

### Step 3: the variables

The problem registers the field, with one guard cell, and the arrays of `levelData` on every level. The arrays of the edges are staggered by half a cell in the direction of the edges. `getSecondaryVariables` gets the arrays back into the `levelData` of each level. There are no other variables to get, so `getVariables` does nothing:

```cpp
struct diffusion
{
	using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
	using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

	inline static constexpr auto name = SAMS::constexprName("diffusion");
	inline static constexpr auto memoryName = SAMS::constexprName("diffusion");

	struct parameters
	{
		double kappa;
		double beta;
	};
	using dataPack = parameters;

	static constexpr const char *fieldName()
	{
		return "u";
	}

	static constexpr int stencilSize = 7;

	// The conductance of the faces is stored on the edges between cells. cx(i) is between cells i and i + 1
	struct levelData
	{
		volumeArray cx, cy, cz;
		volumeArray beta;
	};

	DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
	{
		const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
		di = table[s][0];
		dj = table[s][1];
		dk = table[s][2];
	}

	struct stencil
	{
		volumeArray cx, cy, cz;
		volumeArray betaArray;

		DEVICEPREFIX parabolic::fluxDivergence::faceWeights<diffusion> fluxWeights(size_t i, size_t j, size_t k) const
		{
			using parabolic::at;
			// All of the weights start at zero, and only the cells that a flux uses are set, found from their position relative to the cell
			parabolic::fluxDivergence::faceWeights<diffusion> w{};
			// The flux out of the cell through its face on the high side in x is the conductance of that face, which is cx(i), times
			// the difference between the cell and the next cell in x, which is at<1, 0, 0>. Only the cell itself is set here
			w.fluxWeightsX[at<0, 0, 0>] = cx(i, j, k);
			// The same in y and z, where the next cells are at<0, 1, 0> and at<0, 0, 1>
			w.fluxWeightsY[at<0, 0, 0>] = cy(i, j, k);
			w.fluxWeightsZ[at<0, 0, 0>] = cz(i, j, k);
			// The weights of the next cells are minus those of the cell, which makeSymmetric sets
			w.makeSymmetric();
			return w;
		}

		DEVICEPREFIX void coefficients(size_t i, size_t j, size_t k, double *a) const
		{
			// The helper makes the coefficients from the weights of the fluxes through the faces of this cell. It uses the fluxes
			// through the faces on the low side from the cells before it, so that both cells that share a face use the same flux
			parabolic::fluxDivergence::coefficientsFromFluxes<diffusion>(*this, i, j, k, a);
		}

		DEVICEPREFIX double beta(size_t i, size_t j, size_t k) const
		{
			return betaArray(i, j, k);
		}
	};

	void registerVariables(SAMS::harness &harness)
	{
		const int ghosts = 1;
		harness.variableRegistry.registerVariable<SAMS::T_dataType>(fieldName(), pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
	}

	void registerSecondaryVariables(SAMS::harness &harness, const std::array<int, 3> &axisLevels)
	{
		const std::string memory = memoryName.c_str();
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cx", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cy", SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED);
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cz", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL);
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "beta", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
	}

	void getVariables(SAMS::harness &harness)
	{
	}

	void getSecondaryVariables(SAMS::harness &harness, std::vector<level> &levels)
	{
		const std::string memory = memoryName.c_str();
		parabolic::fillArrayLevels(harness, memory, "cx", levels, [](level &l) -> volumeArray & { return l.data.cx; });
		parabolic::fillArrayLevels(harness, memory, "cy", levels, [](level &l) -> volumeArray & { return l.data.cy; });
		parabolic::fillArrayLevels(harness, memory, "cz", levels, [](level &l) -> volumeArray & { return l.data.cz; });
		parabolic::fillArrayLevels(harness, memory, "beta", levels, [](level &l) -> volumeArray & { return l.data.beta; });
	}
};
```

### Step 4: populating the level data

`prepareSolve` fills the arrays of `levelData` on every level, using the cell widths of that level. The conductance of the edge $i$ is $\kappa \, \Delta y_j \Delta z_k / h_i$, with $h_i = \tfrac{1}{2} (\Delta x_i + \Delta x_{i+1})$, and it is set on the range of the edges in its direction. $\beta$ is a constant, so it is set on every level without restricting it. `makeStencil` copies the arrays of a level into a `stencil`. The kernels capture copies of the arrays and the constants, and not the problem or the parameters:

```cpp
struct diffusion
{
	using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
	using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

	inline static constexpr auto name = SAMS::constexprName("diffusion");
	inline static constexpr auto memoryName = SAMS::constexprName("diffusion");

	struct parameters
	{
		double kappa;
		double beta;
	};
	using dataPack = parameters;

	static constexpr const char *fieldName()
	{
		return "u";
	}

	static constexpr int stencilSize = 7;

	// The conductance of the faces is stored on the edges between cells. cx(i) is between cells i and i + 1
	struct levelData
	{
		volumeArray cx, cy, cz;
		volumeArray beta;
	};

	DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
	{
		const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
		di = table[s][0];
		dj = table[s][1];
		dk = table[s][2];
	}

	struct stencil
	{
		volumeArray cx, cy, cz;
		volumeArray betaArray;

		DEVICEPREFIX parabolic::fluxDivergence::faceWeights<diffusion> fluxWeights(size_t i, size_t j, size_t k) const
		{
			using parabolic::at;
			// All of the weights start at zero, and only the cells that a flux uses are set, found from their position relative to the cell
			parabolic::fluxDivergence::faceWeights<diffusion> w{};
			// The flux out of the cell through its face on the high side in x is the conductance of that face, which is cx(i), times
			// the difference between the cell and the next cell in x, which is at<1, 0, 0>. Only the cell itself is set here
			w.fluxWeightsX[at<0, 0, 0>] = cx(i, j, k);
			// The same in y and z, where the next cells are at<0, 1, 0> and at<0, 0, 1>
			w.fluxWeightsY[at<0, 0, 0>] = cy(i, j, k);
			w.fluxWeightsZ[at<0, 0, 0>] = cz(i, j, k);
			// The weights of the next cells are minus those of the cell, which makeSymmetric sets
			w.makeSymmetric();
			return w;
		}

		DEVICEPREFIX void coefficients(size_t i, size_t j, size_t k, double *a) const
		{
			// The helper makes the coefficients from the weights of the fluxes through the faces of this cell. It uses the fluxes
			// through the faces on the low side from the cells before it, so that both cells that share a face use the same flux
			parabolic::fluxDivergence::coefficientsFromFluxes<diffusion>(*this, i, j, k, a);
		}

		DEVICEPREFIX double beta(size_t i, size_t j, size_t k) const
		{
			return betaArray(i, j, k);
		}
	};

	void registerVariables(SAMS::harness &harness)
	{
		const int ghosts = 1;
		harness.variableRegistry.registerVariable<SAMS::T_dataType>(fieldName(), pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
	}

	void registerSecondaryVariables(SAMS::harness &harness, const std::array<int, 3> &axisLevels)
	{
		const std::string memory = memoryName.c_str();
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cx", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cy", SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED);
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cz", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL);
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "beta", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
	}

	void getVariables(SAMS::harness &harness)
	{
	}

	void getSecondaryVariables(SAMS::harness &harness, std::vector<level> &levels)
	{
		const std::string memory = memoryName.c_str();
		parabolic::fillArrayLevels(harness, memory, "cx", levels, [](level &l) -> volumeArray & { return l.data.cx; });
		parabolic::fillArrayLevels(harness, memory, "cy", levels, [](level &l) -> volumeArray & { return l.data.cy; });
		parabolic::fillArrayLevels(harness, memory, "cz", levels, [](level &l) -> volumeArray & { return l.data.cz; });
		parabolic::fillArrayLevels(harness, memory, "beta", levels, [](level &l) -> volumeArray & { return l.data.beta; });
	}

	void prepareSolve(std::vector<level> &levels, const parameters &params) const
	{
		const double kappa = params.kappa, betaValue = params.beta;
		for (level &l : levels)
		{
			const volumeArray cx = l.data.cx, cy = l.data.cy, cz = l.data.cz, beta = l.data.beta;
			const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;

			auto conductanceX = LAMBDA(size_t i, size_t j, size_t k)
			{
				cx(i, j, k) = kappa * dy(j) * dz(k) / (0.5 * (dx(i) + dx(i + 1)));
			};
			pw::applyKernel(conductanceX, l.xbLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);

			auto conductanceY = LAMBDA(size_t i, size_t j, size_t k)
			{
				cy(i, j, k) = kappa * dx(i) * dz(k) / (0.5 * (dy(j) + dy(j + 1)));
			};
			pw::applyKernel(conductanceY, l.xcLocalDomainRange, l.ybLocalDomainRange, l.zcLocalDomainRange);

			auto conductanceZ = LAMBDA(size_t i, size_t j, size_t k)
			{
				cz(i, j, k) = kappa * dx(i) * dy(j) / (0.5 * (dz(k) + dz(k + 1)));
			};
			pw::applyKernel(conductanceZ, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zbLocalDomainRange);

			auto setBeta = LAMBDA(size_t i, size_t j, size_t k)
			{
				beta(i, j, k) = betaValue;
			};
			pw::applyKernel(setBeta, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
		}
	}

	stencil makeStencil(const level &l) const
	{
		stencil s;
		s.cx = l.data.cx;
		s.cy = l.data.cy;
		s.cz = l.data.cz;
		s.betaArray = l.data.beta;
		return s;
	}
};
```

If $\beta$ varied in space, it would be set on the finest level and averaged onto the coarser levels with `parabolic::restrictCells`.

### Step 5: the remaining hooks

The problem has no initial conditions to set, because an initial conditions package sets the parameters, and no boundary conditions of its own. The field is $u$ itself, so there is nothing to convert before or after the solve. The hooks are still required, and are empty. This is the complete class:

```cpp
struct diffusion
{
	using volumeArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 3>;
	using lineArray = portableWrapper::acceleratedArray<SAMS::T_dataType, 1>;

	inline static constexpr auto name = SAMS::constexprName("diffusion");
	inline static constexpr auto memoryName = SAMS::constexprName("diffusion");

	struct parameters
	{
		double kappa;
		double beta;
	};
	using dataPack = parameters;

	static constexpr const char *fieldName()
	{
		return "u";
	}

	static constexpr int stencilSize = 7;

	// The conductance of the faces is stored on the edges between cells. cx(i) is between cells i and i + 1
	struct levelData
	{
		volumeArray cx, cy, cz;
		volumeArray beta;
	};

	DEVICEPREFIX static constexpr void stencilCellOffset(int s, int &di, int &dj, int &dk)
	{
		const int table[stencilSize][3] = {{0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
		di = table[s][0];
		dj = table[s][1];
		dk = table[s][2];
	}

	struct stencil
	{
		volumeArray cx, cy, cz;
		volumeArray betaArray;

		DEVICEPREFIX parabolic::fluxDivergence::faceWeights<diffusion> fluxWeights(size_t i, size_t j, size_t k) const
		{
			using parabolic::at;
			// All of the weights start at zero, and only the cells that a flux uses are set, found from their position relative to the cell
			parabolic::fluxDivergence::faceWeights<diffusion> w{};
			// The flux out of the cell through its face on the high side in x is the conductance of that face, which is cx(i), times
			// the difference between the cell and the next cell in x, which is at<1, 0, 0>. Only the cell itself is set here
			w.fluxWeightsX[at<0, 0, 0>] = cx(i, j, k);
			// The same in y and z, where the next cells are at<0, 1, 0> and at<0, 0, 1>
			w.fluxWeightsY[at<0, 0, 0>] = cy(i, j, k);
			w.fluxWeightsZ[at<0, 0, 0>] = cz(i, j, k);
			// The weights of the next cells are minus those of the cell, which makeSymmetric sets
			w.makeSymmetric();
			return w;
		}

		DEVICEPREFIX void coefficients(size_t i, size_t j, size_t k, double *a) const
		{
			// The helper makes the coefficients from the weights of the fluxes through the faces of this cell. It uses the fluxes
			// through the faces on the low side from the cells before it, so that both cells that share a face use the same flux
			parabolic::fluxDivergence::coefficientsFromFluxes<diffusion>(*this, i, j, k, a);
		}

		DEVICEPREFIX double beta(size_t i, size_t j, size_t k) const
		{
			return betaArray(i, j, k);
		}
	};

	void registerVariables(SAMS::harness &harness)
	{
		const int ghosts = 1;
		harness.variableRegistry.registerVariable<SAMS::T_dataType>(fieldName(), pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
	}

	void registerSecondaryVariables(SAMS::harness &harness, const std::array<int, 3> &axisLevels)
	{
		const std::string memory = memoryName.c_str();
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cx", SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cy", SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL, SAMS::staggerType::CENTRED);
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "cz", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::HALF_CELL);
		parabolic::registerArrayLevels(harness, memory, fieldName(), axisLevels, "beta", SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED, SAMS::staggerType::CENTRED);
	}

	void getVariables(SAMS::harness &harness)
	{
	}

	void getSecondaryVariables(SAMS::harness &harness, std::vector<level> &levels)
	{
		const std::string memory = memoryName.c_str();
		parabolic::fillArrayLevels(harness, memory, "cx", levels, [](level &l) -> volumeArray & { return l.data.cx; });
		parabolic::fillArrayLevels(harness, memory, "cy", levels, [](level &l) -> volumeArray & { return l.data.cy; });
		parabolic::fillArrayLevels(harness, memory, "cz", levels, [](level &l) -> volumeArray & { return l.data.cz; });
		parabolic::fillArrayLevels(harness, memory, "beta", levels, [](level &l) -> volumeArray & { return l.data.beta; });
	}

	void initialConditions(SAMS::harness &harness)
	{
	}

	void setBoundaryConditions(SAMS::harness &harness)
	{
	}

	void beforeSolve(SAMS::harness &harness)
	{
	}

	void afterSolve(SAMS::harness &harness)
	{
	}

	void prepareSolve(std::vector<level> &levels, const parameters &params) const
	{
		const double kappa = params.kappa, betaValue = params.beta;
		for (level &l : levels)
		{
			const volumeArray cx = l.data.cx, cy = l.data.cy, cz = l.data.cz, beta = l.data.beta;
			const lineArray dx = l.dxc, dy = l.dyc, dz = l.dzc;

			auto conductanceX = LAMBDA(size_t i, size_t j, size_t k)
			{
				cx(i, j, k) = kappa * dy(j) * dz(k) / (0.5 * (dx(i) + dx(i + 1)));
			};
			pw::applyKernel(conductanceX, l.xbLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);

			auto conductanceY = LAMBDA(size_t i, size_t j, size_t k)
			{
				cy(i, j, k) = kappa * dx(i) * dz(k) / (0.5 * (dy(j) + dy(j + 1)));
			};
			pw::applyKernel(conductanceY, l.xcLocalDomainRange, l.ybLocalDomainRange, l.zcLocalDomainRange);

			auto conductanceZ = LAMBDA(size_t i, size_t j, size_t k)
			{
				cz(i, j, k) = kappa * dx(i) * dy(j) / (0.5 * (dz(k) + dz(k + 1)));
			};
			pw::applyKernel(conductanceZ, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zbLocalDomainRange);

			auto setBeta = LAMBDA(size_t i, size_t j, size_t k)
			{
				beta(i, j, k) = betaValue;
			};
			pw::applyKernel(setBeta, l.xcLocalDomainRange, l.ycLocalDomainRange, l.zcLocalDomainRange);
		}
	}

	stencil makeStencil(const level &l) const
	{
		stencil s;
		s.cx = l.data.cx;
		s.cy = l.data.cy;
		s.cz = l.data.cz;
		s.betaArray = l.data.beta;
		return s;
	}
};
```