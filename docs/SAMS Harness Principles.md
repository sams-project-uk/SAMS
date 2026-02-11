# The SAMS Harness

The SAMS harness is the part of the code that provides the infrastructure for handling things like memory, parallelism and the synchronisation of things like array sizes and axis decompositions across the various packages. The harness is split up into various components, only some of which a developer will normally have to interact with. A given package will always interact with a specific harness instance that it is handed by the runner that is running the package.

It is important to note that the harness is not itself intended to be high performance. Core elements are highly optimised, but it is generally expected that a physics package will obtain things like arrays before they start running and will hold onto them for the duration of the run. Only things like the application of boundary conditions should generally be requested directly from the harness while running.

## The variable definitions

The SAMS harness defines four basic types that should be used to ensure that types are consistent across the various packages. These are:

1) `SAMS::T_dataType` - The floating point type used for the main data arrays. Default `double`

2) `SAMS::T_complexType` - The floating point type used for complex data arrays. Default `std::complex<SAMS::T_dataType>`

3) `SAMS::T_sizeType` - The integer type used for array sizes and indices. Default `std::size_t`

4) `SAMS::T_indexType` - The integer type used for array indices. Default `std::make_signed<SAMS::T_sizeType>::type`

There is no restriction on the types that are permitted, but these types should be used to ensure simply interoperability between the various packages. For example, if you want to define a new array type in your package, you should use `SAMS::T_dataType` as the floating point type, and `SAMS::T_sizeType` as the integer type for the array sizes.

## The `SAMS::memoryRegister` class

The `SAMS::memoryRegister` class is a simple class that provides a registry for memory allocations. Given a size in bytes, optionally an alignment and a portableWrapper array tag to indicate memory location (host or device). With the returned pointer to the memory it is also possible to deallocate the memory. This class should never be used directly by a package developer, but rather through the array handling facilities provided by the SAMS harness or by a package using whatever approach they wish for unmanaged memory.

## The `SAMS::typeRegistry` class

A large part of the harness is essentially a form of soft type erasure around memory handling. This means that there has to be some way in which the types can be checked at runtime. The `SAMS::typeRegistry` class provides a registry for types that can be used in the SAMS code. Unlike the other SAMS classes there is only one type registry, not one per harness instance. Mostly package writers will not have to interact with the typeRegistry because it is preconfigured with `double`, `float`, `int32_t`, `int64_t`, `uint32_t`, `uint64_t`, `std::complex<double>` and `std::complex<float>`. However, if a package writer wanted to use a different type, for example `std::complex<long double>` or a custom class they would have to register that type.

```cpp

struct MyCustomType{
    double a;
    double b;
};

SAMS::typeRegistry &typeReg = SAMS::gettypeRegistry();//Get the type registry

typeReg.registerType<MyCustomType>("MyCustomType"); //Register the type with the name "MyCustomType"
```

This type will now word for non MPI decomposed variables, but if you wanted to use it for an MPI decomposed variable you would also have to provide an MPI datatype describing the type. This is done by providing a second argument to the `registerType` method, which is a `MPI_Datatype`. For example:

```cpp

struct MyCustomType{
    double a;
    double b;
};

SAMS::typeRegistry &typeReg = SAMS::gettypeRegistry();//Get the type registry
typeReg.registerType<MyCustomType>("MyCustomType", myCustomTypeMPI); //Register the type with the name "MyCustomType" and the MPI datatype myCustomTypeMPI
```

The MPI data type can be obtained from the MPI registry, but that is described in the `SAMS::MPIRegistry` section of this document

## The `SAMS::axisRegistry` class

This is the first class that a developer should interact with. The `SAMS::axisRegistry` class needs to be constructed with a reference to a `SAMS::memoryRegistry` class. The `SAMS::axisRegistry` class provides a registry for the various axes that are used in the SAMS code. Axes are both a description of a physical or logical axis, but also the discretisation of that axis, the values at and between the discretised points and where relevant the decomposition of that axis over the MPI ranks. From a package developer perspective the main point of the axis registry is to register axes that can be used to defined variables and then to get the actual grid and grid delta associated with the axis when needed. Registering an axis is very simple

```cpp
SAMS::axisRegistry &axReg = harness.axisRegistry;//Get the axis registry for this harness

//Register an axis with the name "X"
axReg.registerAxis("X"); //This creates an axis named "X" with no MPI decomposition
//Register an axis with the name "Y" and a decomposition over the second MPI dimension
axReg.registerAxis("Y", SAMS::MPIAxis(1)); //This creates an axis named "Y" with a decomposition over the second MPI dimension
```

Note that registering an axis says nothing about the number of grid points in the axis. That is specified later. Registering an axis is done purely to allow the axis to be used to define variables. Registering the same axis twice is valid so long as the MPI mapping is equivalent. For example, the following is valid:

```cpp
SAMS::axisRegistry &axReg = harness.axisRegistry;//Get the axis registry for this harness

//Register an axis with the name "X"
axReg.registerAxis("X"); //This creates an axis named "X" with no MPI decomposition
//Register an axis with the name "X" (again)
axReg.registerAxis("X");

//axReg.registerAxis("X", SAMS::MPIAxis(1)); //This would be invalid as the MPI mapping is not equivalent
```

There are also different axes called logical axes. Logical axes are axes where it is not meaningful to talk about values between the grid points, for example a species ID is not meaningful except at the defined points. This only matters when creating grid points and only when using specific types that do not implement operators such as addition and multiplication by doubles.  Logical axes are registered in the same way as normal axes, but with the `registerLogicalAxis` method of the `SAMS::axisRegistry` class. For example:

```cpp
SAMS::axisRegistry &axReg = harness.axisRegistry;//Get the axis registry for this harness
//Register a logical axis with the name "species"
axReg.registerLogicalAxis("species");
```

Logical axes can also be MPI decomposed, but this would generally be less common.

Normally one would then move on to registering variables that use the axes, but that is covered in the `Developing SAMS Packages` document in order. Here it will be covered later in the section on the `SAMS::variableRegistry` class.

The next phase for the `SAMS::axisRegistry` is to specify the number of grid points in the axis and how to generate the grid points. At the moment the only fully supported function is the `setDomain` method, which specifies the number of grid points and the position of the bottom and top of the axis. For example:

 ```cpp
SAMS::axisRegistry &axReg = harness.axisRegistry;//Get the axis registry for this harness
//Register an axis with the name "X"
axReg.registerAxis("X"); //This creates an axis named "X" with no MPI decomposition
/*
Variable registration would go here
*/
//Set the domain of the axis "X" to have 100 grid points, a lower bound of 0.0 and an upper bound of 1.0
axReg.setDomain("X", 100, 0.0, 1.0);
```

When `setDomain` is called this sets the number of grid points in the domain of the axis, but extra points are allocated at the top and bottom of the domain corresponding to the number of ghost points specified for the axis. The grid points are then generated using a simple linear spacing between the lower and upper bounds. In the future it is planned to add support for more complex grid generation functions, but at the moment only linear spacing is supported.

The type associated with the axis is determined by the type of the lower and upper bounds. For example, if the lower and upper bounds are `double` then the grid points will be generated as `double`. If the lower and upper bounds are `float` then the grid points will be generated as `float`. If the lower and upper bounds are of a custom type then the grid points will be generated as that custom type.The custom type must implement mathematical operators both with itself and with doubles.

For logical axes where it is not meaningful to talk about such mathematical operators the equivalent function is `setDomainValues`, which takes a vector of values to use as the grid points. For example:

```cpp
SAMS::axisRegistry &axReg = harness.axisRegistry;//Get the axis registry for this harness
//Register a logical axis with the name "species"
axReg.registerLogicalAxis("species");
/*
Variable registration would go here
*/
//Set the domain of the logical axis "species" to have the values 0, 1, 2 and 3
axReg.setDomainValues("species", {0, 1, 2, 3});
```

You can query axes for a variety of information, but mostly you do not want to. Normally a package developer is interested in the properties of axes as they apply to specific variables, and that information is available through the variable registry, which is described in the `SAMS::variableRegistry` section of this document.

The other thing that you want from the axis registry is the actual grid points and the grid deltas. Mostly you would do this with `fillPPLocalAxis` and `fillPPLocalDelta` methods

```cpp
        auto &axRegistry = harness.axisRegistry;

        portableWrapper::portableArray<T_dataType, 1, portableWrapper::arrayTags::accelerated> xc,xb,dxc,dxb;

        axRegistry.fillPPLocalAxis("X", data.xc, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalAxis("X", data.xb, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalDelta("X", data.dxc, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalDelta("X", data.dxb, SAMS::staggerType::HALF_CELL);

```

These functions give you portableWrapper::portableArrays that wrap the actual data for the grid points and grid deltas. The arrays are allocated and owned by the harness, so you don't have to worry about memory management for these arrays. The arrays are also automatically updated if the domain of the axis changes, so you don't have to worry about that either. Both host and accelerated versions of all arrays are available and the correct one is returned depending on the portable array type that you pass to the function. 

Future expansions will include functions that return a `kokkos::view` or `kokkos::offset_view` instead of a `portableWrapper::portableArray`, but the `portableWrapper::portableArray` is the main way to get the grid points and grid deltas at the moment.

## The `SAMS::MPIManager` class

Mostly package developers don't need to interact directly with the `SAMS::MPIManager` class. It is created, being passed a reference to the `SAMS::axisRegistry` class, and then used by the harness to manage the MPI decomposition of axes and variables. Variables that you want to have MPI decomposed should just be registered with the `SAMS::variableRegistry` class, so you don't have to write any MPI code for handling decomposition or communication, so mostly you will only want to query the `SAMS::MPIManager` class for information things like the communicator, the current rank, the size of the communicator etc. So, to show the "normal" features of the `SAMS::MPIManager` class, here is how you would get the usual properties

```cpp
SAMS::MPIManager &mpiManager = harness.MPIManager;//Get the MPI manager for this harness
MPI_Comm comm = mpiManager.getComm(); //Get the MPI communicator
int nprocs = mpiManager.getSize(); //Get the size of the communicator
int rank = mpiManager.getRank(); //Get the rank of the current process
```

Slightly more advanced is the decomposition. In this case you can get the number of MPI decomposition dimensions, the size of the decomposition in each dimension and the coordinates of the current rank in the decomposition. For example:

```cpp
SAMS::MPIManager &mpiManager = harness.MPIManager;//Get the MPI manager for this harness
std::array<int, SAMS::MPI_DECOMPOSITION_RANK> sizes = mpiManager.getDims(); //Get the size of the decomposition in each dimension
std::array<int, SAMS::MPI_DECOMPOSITION_RANK> coords = mpiManager.getCoords(); //Get the coordinates of the current rank in the decomposition
```

You can also get your immediate neighbours in the cartesian communicators. This is an array of `2*SAMS::MPI_DECOMPOSITION_RANK` integers, where index `2*i` is the rank of the neighbour in the negative direction of dimension `i` and index `2*i+1` is the rank of the neighbour in the positive direction of dimension `i`. For example: 

```cpp
SAMS::MPIManager &mpiManager = harness.MPIManager;//Get the MPI manager for this harness
std::array<int, 2*SAMS::MPI_DECOMPOSITION_RANK> neighbours = mpiManager.getNeighbours(); //Get the ranks of the neighbours in the cartesian communicator
```

Finally you can also query for whether a given MPI decomposition direction is periodic or not, and also whether the current rank is at the edge of the domain, or would be at the edge of the domain if the decomposition was not periodic. For example:

```cpp
SAMS::MPIManager &mpiManager = harness.MPIManager;//Get the MPI manager for this harness
bool isPeriodic = mpiManager.isPeriodic(0); //Check if the first MPI decomposition dimension is periodic
bool isLowerEdge = mpiManager.isEdge(0, SAMS::domain::edges::lower);
bool isUpperEdge = mpiManager.isEdge(0, SAMS::domain::edges::upper);
bool isNonPeriodicLowerEdge = mpiManager.isEdgeNP(0, SAMS::domain::edges::lower);
```

The other reason to use the `SAMS::MPIManager` class is to create MPI datatypes for custom MPI types that you wish to use. You can, of course, use the normal MPI type creation routines, but the main advantage of using the `SAMS::MPIManager` class is that it both handles the lifetimes of the MPI datatypes and also dedeuplicates MPI datatypes. For types representing structs this is a limited benefit, but for things like vectors and subarrays that can minimise the proliferation of datatypes. There are several options for creating MPI types. There are actually several different functions added, but we will just show two - creating a struct type and using the `SAMS::MPIManager` to handle and deduplicate a type that you have manually created. For example, to create a struct type:

```cpp
struct MyCustomType{
    double a;
    double b;
};
SAMS::MPIManager &mpiManager = harness.MPIManager;//Get the MPI manager for this harness
MPI_Datatype myCustomTypeMPI = mpiManager.createMPIStructType<MyCustomType, &MyCustomType::a, &MyCustomType::b>(); //Create an MPI datatype for MyCustomType
```

As you can see this is quite different to the built in MPI approach to type creation. There is also an overload to `createMPIStructType` that has the same signature as the normal MPI type creation function, but this one is easier for C++ programmers to use. The types inside the struct must be registered with the `SAMS::typeRegistry` class for this to work, since it uses the type registry to get the MPI datatypes for the individual members of the struct.

While there are direct methods on the `SAMS::MPIManager` class for creating quite a few MPI types, you can also just create the MPI datatype yourself using the normal MPI type creation functions and then use the `cacheMPIType` method of the `SAMS::MPIManager` class to have the `SAMS::MPIManager` handle the lifetime and deduplication of the type. For example:

```cpp
MPI_Datatype myCustomTypeMPI;
//Code to create the MPI datatype using normal MPI functions goes here
SAMS::MPIManager &mpiManager = harness.MPIManager;//Get the MPI manager for this harness
mpiManager.cacheMPIType(myCustomTypeMPI); //Cache the MPI datatype with the MPI manager
```

It is worth noting that the value of myCustomTypeMPI may chance when cacheMPIType is called. If the type is not already in the cache then it will be added to the cache and the value of myCustomTypeMPI will be unchanged. However, if the type is already in the cache then the value of myCustomTypeMPI will be changed to the value of the cached type. The type that you created will be freed if the type is already in the cache, which is why `cacheMPIType` modifies it's parameter rather than returning it.

## The `SAMS::variableRegistry` class

The most important part of the harness for a package developer is the `SAMS::variableRegistry` class. Most variables and all variables that are intended to be used between packages should be registered with the `SAMS::variableRegistry` class. Using the variable registry means that you hand off management of memory allocation and lifetime, domain decomposition and most other housekeeping tasks to the harness.

The first job of the variable registry is to register variables. This is done by calling the `registerVariable` method of the `SAMS::variableRegistry` class. This method is templated on the type of the variable (the type must be registered with the `SAMS::typeRegistry` class) and takes as arguments the name of the variable, a portable wrapper array tag saying whether the variable should be on the host or on a device and a set of comma separated set of `SAMS::dimension` object

```cpp
    auto &varRegistry = harness.variableRegistry;
    const int ghosts = 2; // 2 Ghost cells at top and bottom of each dimension
    //Register densty (cell centred on all axes)
    varRegistry.registerVariable<T_dataType>("rho", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

    //Register x velocity (face centred on all axes)
    varRegistry.registerVariable<T_dataType>("vx", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

    //Register By (face centred on Y, cell centred on X and Z)
    varRegistry.registerVariable<T_dataType>("by", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts));
```

The `SAMS::dimension` objects map the axis of the variable to the already registered axes by name, also specifying the number of ghost points that this variable requires in this dimension and whether the variable is staggered in this dimension or not.

The second phase of interaction with the variable registry is to get the actual arrays. This is the part that makes this a type erasure system, albeit a soft one that is reversed before any actual calculations are done. To get the arrays you call the `getVariable` method of the `SAMS::variableRegistry` class, which is templated on the type of the variable and takes as an argument the name of the variable. For example:

```cpp
auto &varRegistry = harness.variableRegistry;
//Get the array for density
auto rho = varRegistry.getPPArray<T_dataType, 3, portableWrapper::arrayTags::accelerated>("rho");
```

This returns a `portableWrapper::portableArray` that wraps the actual data. It is a trivially copyable object that can be passed around by value and captured by performance portability systems without issue. There is also a related function called `fillPPArray` that fills an already existing `portableWrapper::portableArray` with the data for the variable. For example:

```cpp
auto &varRegistry = harness.variableRegistry;
//Get the array for density
portableWrapper::portableArray<T_dataType, 3, portableWrapper::arrayTags::accelerated> rho;
varRegistry.fillPPArray("rho", rho);
```

Note that while the term is "fill" this is not copying the data, it is just filling the `portableArray` with the correct pointer and size information to point to the data for the variable.


The second thing that is requested from the variable registry is the application of boundary conditions. You can do this either by calling the `applyBoundaryConditions` method of the `SAMS::variableRegistry` class, which takes as an argument the name of the variable, or by requesting a `SAMS::variableDef` object for the variable, which has an `applyBoundaryConditions` method. For example, using the `applyBoundaryConditions` method of the `SAMS::variableRegistry` class:

```cpp
auto &varRegistry = harness.variableRegistry;
//Apply boundary conditions to density
varRegistry.applyBoundaryConditions("rho");
```
This will apply boundary conditions (both actual domain boundarys and MPI interprocessor boundaries) to all edges of the variable "rho". The same thing using a variableDef object would be

```cpp
SAMS::variableDef rhoDef = varRegistry.getVariableDef("rho"); //Get the variable definition for density
rhoDef.applyBoundaryConditions(); //Apply boundary conditions to density
```

As you can see, `variableDef` objects are copyable objects and are also default constructible, so it is possible to cache a variable definitition in a physics package to avoid the overhead of looking up the variable definition by name every time you want to apply boundary conditions. Normally this overhead is minimal and ignorable, but this does provide an option for performance critical code.

It is also possible to apply boundary conditions either only to a specific axis, or to only a specific edge of a specific axis. For example, to apply boundary conditions only to the lower edge of the X axis you could do either of the following:

```cpp
auto &varRegistry = harness.variableRegistry;
//Apply boundary conditions to the lower edge of the X axis for density
varRegistry.applyBoundaryConditions("rho", 0, SAMS::domain::edges::lower);
//Apply boundary to both edges of the Y axis for density
varRegistry.applyBoundaryConditions("rho", 1);
```

You will note that axes are specified by their index. This index is the same as the order in which the axes were specified when the variable was registered. So, for example, if you registered a variable with `SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts)` then the X axis would be index 0, the Y axis would be index 1 and the Z axis would be index 2.

The final thing that you may want to do with the variable registry is to query it for information about the variables and in particular what indexes are part of the domain or the ghost cells and the MPI decomposition. The best way of doing this is by getting a dimension object back from a variable definition object and then querying the dimension object for the information. For example, to get the range of the domain in the X axis for the variable "rho" you could do the following:

```cpp
auto &dim = varRegistry.getDimension("rho", 0); //Get the dimension object for the X axis of density
std::cout << "The domain in the X axis for density is " << dim.getDomainRange() << std::endl;
```

Before going further we must make some terms clear as we use them

* Global array - The array that would exist if there was no MPI decomposition, including ghost cells that would exist on that global array. If the code is in MPI mode then the global array is a purely conceptual object that does not actually exist anywhere, but it is still useful to talk about it.

* Global domain - The part of the global array that is not ghost cells. This is the part of the array that is actually evolved by the physics code. The global domain is decomposed over the MPI ranks, but the ghost cells are not decomposed, they are just replicated on each rank.

* Local array - The actual array that exists on each MPI rank. This includes the local portion of the global domain and the ghost cells that are associated with that local portion of the global domain.

* Local domain - The part of the local array that is not ghost cells. This is the part of the array that is actually evolved by the physics code on that rank

* Non-domain cells - The part of the local array that is ghost cells. This is the part of the array that is not evolved by the physics code, but is used for things like applying boundary conditions and for MPI communication.

The possible queries to the dimension object are

1) `getLB` - Get the lower bound of the global array

2) `getLBZeroBased` - Get the lower bound of the global array, but with the global domain starting at index 0.

3) `GetUB` - Get the upper bound of the global array

4) `GetUBZeroBased` - Get the upper bound of the global array, but with the global domain starting at index 0.

5) `GetRange` A `portableWrapper::Range` object representing the range of the global array, including ghost cells is [GetLB, GetUB]

6) `GetRangeZeroBased` A `portableWrapper::Range` object representing the range of the global array, including ghost cells but with the global domain starting at index 0 is [GetLBZeroBased, GetUBZeroBased]

7) `getCount` - Get the total number of points in the global array, including ghost cells

8) `getDomainLB` - Get the lower bound of the global domain

9) `getDomainLBZeroBased` - Get the lower bound of the global domain, but with the global **array** starting at index 0. 

10) `getDomainUB` - Get the upper bound of the global domain

11) `getDomainUBZeroBased` - Get the upper bound of the global domain, but with the global **array** starting at index 0.

12) `getDomainRange` - A `portableWrapper::Range` object representing the range of the global domain is [getDomainLB, getDomainUB]

13) `getDomainRangeZeroBased` - A `portableWrapper::Range` object representing the range of the global domain, but with the global **array** starting at index 0 is [getDomainLBZeroBased, getDomainUBZeroBased]

14) `getDomainCount` - Get the total number of points in the global domain

15) `getLocalLB` - Get the lower bound of the local array

16) `getLocalLBZeroBased` - Get the lower bound of the local array, but with the local array starting at index 0.

17) `getLocalUB` - Get the upper bound of the local array

18) `getLocalUBZeroBased` - Get the upper bound of the local array, but with the local array starting at index 0.

19) `getLocalRange` - A `portableWrapper::Range` object representing the range of the local array, including ghost cells is [getLocalLB, getLocalUB]

20) `getLocalRangeZeroBased` - A `portableWrapper::Range` object representing the range of the local array, including ghost cells but with the local array starting at index 0 is [getLocalLBZeroBased, getLocalUBZeroBased]

21) `getLocalCount` - Get the total number of points in the local array, including ghost cells

22) `getLocalDomainLB` - Get the lower bound of the local domain

23) `getLocalDomainLBZeroBased` - Get the lower bound of the local domain, but with the local **array** starting at index 0.

24) `getLocalDomainUB` - Get the upper bound of the local domain

25) `getLocalDomainUBZeroBased` - Get the upper bound of the local domain, but with the local **array** starting at index 0.

26) `getLocalDomainRange` - A `portableWrapper::Range` object representing the range of the local domain is [getLocalDomainLB, getLocalDomainUB]

27) `getLocalDomainRangeZeroBased` - A `portableWrapper::Range` object representing the range of the local domain, but with the local **array** starting at index 0 is [getLocalDomainLBZeroBased, getLocalDomainUBZeroBased]

28) `getLocalDomainCount` - Get the total number of points in the local domain

29) `getGlobalLB` - Get the lower bound of the global array on this rank. This is the same as `getLB` for non MPI decomposed axes, but for MPI decomposed axes this is the lower bound of the local portion of the global array on this rank.

30) `getGlobalLBZeroBased` - Get the lower bound of the global array on this rank, but with the global domain starting at index 0. This is the same as `getLBZeroBased` for non MPI decomposed axes, but for MPI decomposed axes this is the lower bound of the local portion of the global array on this rank, but with the global domain starting at index 0.

31) `getGlobalUB` - Get the upper bound of the global array on this rank. This is the same as `getUB` for non MPI decomposed axes, but for MPI decomposed axes this is the upper bound of the local portion of the global array on this rank.

32) `getGlobalUBZeroBased` - Get the upper bound of the global array on this rank, but with the global domain starting at index 0. This is the same as `getUBZeroBased` for non MPI decomposed axes, but for MPI decomposed axes this is the upper bound of the local portion of the global array on this rank, but with the global domain starting at index 0.

33) `getGlobalRange` - A `portableWrapper::Range` object representing the range of the global array on this rank is [getGlobalLB, getGlobalUB]

34) `getGlobalRangeZeroBased` - A `portableWrapper::Range` object representing the range of the global array on this rank, but with the global domain starting at index 0 is [getGlobalLBZeroBased, getGlobalUBZeroBased]

35) `getGlobalCount` - Get the total number of points in the global array on this rank

36) `getLocalNonDomainLB(SAMS::domain::edges edge)` - Get the lower bound of the non-domain cells in the local array for the specified edge

37) `getLocalNonDomainLBZeroBased(SAMS::domain::edges edge)` - Get the lower bound of the non-domain cells in the local array for the specified edge, but with the local array starting at index 0.

38) `getLocalNonDomainUB(SAMS::domain::edges edge)` - Get the upper bound of the non-domain cells in the local array for the specified edge

39) `getLocalNonDomainUBZeroBased(SAMS::domain::edges edge)` - Get the upper bound of the non-domain cells in the local array for the specified edge, but with the local array starting at index 0.

40) `getLocalNonDomainRange(SAMS::domain::edges edge)` - A `portableWrapper::Range` object representing the range of the non-domain cells in the local array for the specified edge is [getLocalNonDomainLB(edge), getLocalNonDomainUB(edge)]

41) `getLocalNonDomainRangeZeroBased(SAMS::domain::edges edge)` - A `portableWrapper::Range` object representing the range of the non-domain cells in the local array for the specified edge, but with the local array starting at index 0 is [getLocalNonDomainLBZeroBased(edge), getLocalNonDomainUBZeroBased(edge)]


