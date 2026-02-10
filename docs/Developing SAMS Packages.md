# The core SAMS design

SAMS is designed to be a modular framework for solar physics simulations. The core design is based on the idea of "packages", which are self-contained units of code that implement specific events. This allows for a high degree of flexibility and extensibility, as new packages can be added without modifying the core codebase.

The philosophy of SAMS is that each unit should be free to design itself as far as possible without being constrained to use any core interfaces for any purposes other than gathering information about things like the domain and the time step. This should increase the ease with which new packages can be developed.

## The current state of SAMS development

SAMS is currently at v 0.0.3, so should be considered alpha or even pre-alpha software. The interfaces described in this document will almost certainly be extended. This should not break existing packages. It may also remove, change or modify existing interfaces which will break existing packages. In particular, IO is very much a placeholder, and one with some obvious limitations, so it is likely to be changed.

## The Input Deck Parser

We don't have one yet. The idea is that any part of the code can access the input deck parser when one is put in place to allow users to select parameterised values for their simulations. In some places in this document we talk about "reading user input" but this is just a placeholder for "reading from the input deck parser when it is implemented".

## The SAMS runner

The packages in SAMS are marshalled by a class called the runner, which is responsible for calling the appropriate functions in the packages at the appropriate times. The runner itself does little other than call the functions on the packages, and with one exception that is mentioned later packages do not interact directly with the runner. The runner does the following operations

1. Determines which packages are activated by the user at run time and initialises them. 

2. Sets up the MPI domain decomposition (This will likely change in future versions of SAMS because it requires recompilation to change the domain decomposition)

3. Calls the packages to ask what axes they need, and how they want those axes mapped to the MPI domain decomposition

4. Calls the packages to ask what variables they need, how they are mapped to the axes and how many ghost cells they need in each direction

5. Allocate the data arrays and allow the packages to grab them and set up their arrays

6. Run the packages in order, querying them for output and termination conditions

7. Running output when needed, and terminating when needed

8. Clean up and exit

The runner does this using the Automatic Signature Modular Framework, a set of functions that a package can choose to implement to implement phases of a simulation. The runner determines which functions a package implements at compile time, and calls them at the appropriate times during the simulation. This allows for a high degree of flexibility in how packages are implemented, as they can choose to implement only the functions that they need to implement, and the runner will call them at the appropriate times.


# The ASMF (Automatic Signature Modular Framework)

The core of the SAMS runner system is the ASMF, or Automatic Signature Modular Framework. This framework allows packages to be added to the runner, and then the interfaces that the package implements are determined automatically at compile time, including the signatures of the functions that the package implements. First we will describe the phases that the ASMF supports, what a package should do in each phase and what a package needs to implement to be a valid package.

## Class

A SAMS package is a class that implements the SAMS runner interfaces. There are no restrictions on the class itself, just on the member variables and functions that it implements. No restrictions are placed on the class itself - it may be templated, it can inherit from other classes, it can have any member variables and functions that it wants, as long as it implements the required interfaces. The only requirement is that the class must either be default constructible(i.e. can be constructed with no arguments) or that it has a constructor that takes a single argument of type `const SAMS::harness&`. If both constructors are present then the default constructor is preferred.


## Name

The only required element of a package is the name. This should be a static constexpr member of type `std::string_view` called `name`. This is used by the runner to identify the package, to allow the user to specify which packages to run, and in various parts of the IO system. The name should be unique among all packages, and should be descriptive of what the package does. The name should not contain any spaces, but otherwise there are no restrictions on the name. The name is not used for anything other than identification, so it can be whatever the developer wants it to be, as long as it is unique and descriptive.

```cpp
class MyPackage {
public:
    static constexpr std::string_view name = "MyPackage";
    // other member variables and functions
};
```

This is a correct, but minimal package. It has a name, but it does not implement any of the Runner interfaces, so it will not do anything when run.

## Core simulation status

Optionally you can flag your package as being a core simulation by setting a constexpr static bool member of your class called `coreSimulation` to true. Core simulations status doesn't mean anything in terms of how the package is run, but the runner will refuse to run if no core simulations are activated, and will give a warning if more than one core simulation is activated. This is to prevent the user from accidentally running a simulation without activating any core simulations, or from accidentally activating multiple core simulations. If a package does not have a `coreSimulation` member, or if it is set to false, then the package is not considered a core simulation.

## Data packs

Since we are using performance portability, most of the code in SAMS is heavily dependent on lambdas. Unfortunately, there is quite a strong restriction on lambdas in connection with CUDA and HIP, in that they do not allow more sophisticated lambda capture groups. This means that you cannot easily capture a class member variable. Normally one would capture the entire class, but that is both inefficient (since the class would be copied to the device) and also not possible if the class is not trivially copyable. The easiest solution would be for every function in your class to copy the member variables that it needs into local variables which are correctly captured, but this is labour intensive and error prone. The alternative that SAMS has chosen to use is the data pack system. A package can define a data pack, which is a struct that contains all the member variables that the package needs to capture in its lambdas. This structure is specified to the runner which creates a copy of it, and when a package needs it, it uses the ASMF to ask the runner for a reference to the data pack, which it can then capture in its lambdas. This allows the package to easily capture the data pack in its lambdas without having to worry about copying the entire class or about the restrictions on lambda capture groups. The data pack is also used to allow the runner to determine which variables a package needs, and to allocate the appropriate amount of memory for those variables. While the idea is quite complex, the implementation is quite simple - you just put a `using dataPack = ...;` in your class, and then the runner will take care of the rest. So, for example a simple datapack might look like this:

```cpp
namespace mysimulation {
    struct myDataPack {
        portableWrapper::portableArray<double,2> myVariable;
    };

    class MyPackage {
    public:
        static constexpr std::string_view name = "MyPackage";
        using dataPack = myDataPack;
        // other member variables and functions
    };
}
```

It is easiest not to put the definition of your datapack class in your package class, since there are restrictions placed on the visibility of such subclasses in CUDA programming. If the dataPack declaration is not found then no specific member is created by the runner. NOTE - the data pack should not be a simple type (meaning a string or a double etc). The types are deduplicated by the runner, so if two packages want a data pack of the same type then they will share the same instance of it. This is intended since it allows very close coupling between packages if they want to share data, but it means that if you do not want this behaviour, you must force the types to be distinct by wrapping them in a struct. If your code wants more than one subset of variables, then you can define several structs and use the `dataPacks::multiPack` struct to register multiple data packs with the runner. For example:

```cpp
namespace mysimulation {
    struct myDataPack {
        portableWrapper::portableArray<double,2> myVariable;
    };
    struct myOtherDataPack {
        portableWrapper::portableArray<double,2> myOtherVariable;
    };

    class MyPackage {
    public:
        static constexpr std::string_view name = "MyPackage";
        using dataPack = SAMS::dataPacks::multiPack<myDataPack, myOtherDataPack>;
        // other member variables and functions
    };
}
```

The contents of the multiPack will be unpacked by the runner, so the package can ask for a reference to either `myDataPack` or `myOtherDataPack` and it will get the correct one. The data pack system is quite flexible, and allows for a wide range of different data pack designs, so you can design your data pack(s) in whatever way makes the most sense for your package.

## Runner events

To actually get your package to do useful work you have to implement runner events. These are functions that the runner will call at specific times during the simulation, and they allow your package to do work at those times. The runner events are listed here in the order that they are called during the simulation, and for each event we will describe what it is for, when it is called, and what a package should do in that event. NOTE! The entire point of the ASMF is that you only implement events that you are interested in and there is no fixed signature for the events. Some event have some fixed parameters which must appear at the start of the signature, but after that you can request a reference to any registered dataPack(or any of the members of a multiPack), any package (although this is fragile since it will fail to compile if the specified package is not compiled in), and a set of built in objects for performing certain tasks. Other than those initial parameters there are no restrictions on the order of the other parameters and the ASMF will automatically determine which parameters you want and will pass them to your function. Generally for performance requirements you will want to request references to data packs rather than passing them by value, but the ASMF will allow you to pass them by value if you want to. Const references are also allowed, and the ASMF will automatically determine whether you want a const reference or a non-const reference and will pass the correct one.

NOTE : Currently the result of requesting a dataPack that is not registered is a rather unreadable template error message. This is being worked on

### `runnerInteraction` event

Mostly it is not possible for a package to interact directly with the runner, since the type of the runner is not known to the packages. There is a type erased interface to allow some interaction with the runner, but this is quite limited, so this event is called once before the packages are initialised to allow an activated package to interact with the runner. This allows a package to do things like activate other packages. This is mostly intended for automated testing, where one specifies the test package only and the test package activates the other packages that are needed for the test.

While we mention this first since it is the first to be called as the runner starts, it is not a typical package event. It has to be templated with a single typename parameter which will be filled with the type of the runner. So, to activate another package by name, you would do something like this:

```cpp
template<typename RunnerType>
void runnerInteraction(RunnerType& runner){
    runner.activatePackage("MyOtherPackage");
}
```

The specified function just needs to be implemented by the chosen package for it to be called by the runner. 

NOTE: runnerInteraction functions should be unusual. If there is something in your package which seems to need this function, please discuss this with the core dev team. 

### `initialize` event

The `initialize` event is more typical structurally to a "normal" package event. Other than the unusual runner interaction event it is called first as a runner starts up. It is only called on active packages, including packages activated by the `runnerInteraction` event. This is not for things like setting up control variables or initial conditions (there are events later for those), but for things that have to be initialised at the very start of a simulation. This might include things like loading data from a file.

This function can be used to demonstrate how the ASMF works. It is possible to implement `initialize` with an empty signature, so

```cpp
void initialize(){
    //Just set up internal state
}
```
is a valid implementation of `initialize`. In fact, it is a common one since initialize is for setting up internal state before any arrays are allocated, so it is common for a package to use this to initialise state that is stored in the package class itself rather than in a data pack. However, if you want to have access to the data pack in `initialize` then you can just request a reference to it in the signature, and the ASMF will automatically pass it to you. So, for example, if your package has a data pack called `myDataPack` then you can implement `initialize` like this:

```cpp
using dataPack = myDataPack;

void initialize(myDataPack& dataPack){
    //Set up internal state using the data pack
}
```

This demonstrates the ASMF in action - the package just requests a reference to the data pack in the signature of `initialize`, and the ASMF automatically determines that it needs to pass a reference to the data pack to `initialize`, and it does so when it calls `initialize`. There are no restrictions on which datapacks can be requested through the ASMF, so you can request any data pack that is registered with the runner, including data packs from other packages. You can also request specific members of a multiPack, so if your package has a multiPack that contains `myDataPack` and `myOtherDataPack`, then you can request a reference to just `myDataPack` in `initialize` like this:

```cpp
using dataPack = SAMS::dataPacks::multiPack<myDataPack, myOtherDataPack>;

void initialize(myDataPack& dataPack){
    //Set up internal state using the data pack
}
```

NOTE: the order in which each stage is called on each package is fixed, but not trivial to manipulate. If you are implementing a package which requires a specific ordering with respect to another, please contact the core dev team.

## `registerAxes` event

The register axes event is called after `initialize`, and is for registering the axes that the package needs. This event is the first that will make use of the `SAMS::harness` object. The harness handles all common features of the simulation, and is used to specify spatial domains, variables, ghost cells etc. The harness is always available through the ASMF, so you can request a reference to it in the signature of any event. This only makes sense to do for some specific events, and `registerAxes` is the first of those events. In `registerAxes` you should use the harness to register the axes that your package needs, and to specify how those axes are mapped to the MPI domain decomposition. The runner will then use this information to set up the MPI domain decomposition and to allocate the appropriate amount of memory for the variables that your package needs. The harness has a member variable called `axisRegistry` which is used to register axes. See the full developer documentation for the `axisRegistry` class for full details on how to use it, but an example from LARE3D is shown here:

```cpp
void registerAxes(SAMS::harness& harness){
    //Register x, y, z axes, mapped to corresponding MPI directions
        harness.axisRegistry.registerAxis("X", SAMS::MPIAxis(0));
        harness.axisRegistry.registerAxis("Y", SAMS::MPIAxis(1));
        harness.axisRegistry.registerAxis("Z", SAMS::MPIAxis(2));
}
```

this registers three spatial axes (i.e. axes where grid points have a position in a continuous spatial domain) called "X", "Y" and "Z". The second parameter is optional and maps the axis to a specific MPI domain decomposition direction. Generally spatial axes for primary simulations should be mapped to the MPI domain decomposition, but multiple axes can be mapped to the same MPI direction if needed. Note that no guarantees about how a given axis is subdivided over the MPI domain decomposition are made by the runner, so if you have multiple axes mapped to the same MPI direction then you should not make any assumptions about the same grid points being on the same MPI rank for those axes.

It is also possible to register logical, or non-spatial axes. These are used for things like species in a multi-species simulation. They mostly behave like spatial axes, but when the time comes to actually get the axis values there are fewer restrictions on logical axes than on spatial axes. These restrictions will be described later but to register a logical axis you just need to specify that it is a logical axis when you register it, like this:

```cpp
void registerAxes(SAMS::harness& harness){
    //Register a logical axis called "Species"
    harness.axisRegistry.registerLogicalAxis("Species");
}
```

At this stage all that registration involves is setting up the platonic ideal of an axis. "There exists an axis called "X" that is mapped to the first MPI direction". All of the details are registered later.

## `registerVariables` event

The register variables event is called after `registerAxes`, and is for registering the variables that the package needs. In this event you should use the harness to register the variables that your package needs, and to specify how those variables are mapped to the axes that you registered in `registerAxes`. Note that in general a package should only register the variables that it needs for its own calculations, so a package that only sets up an initial condition for a variable should NOT register the variable that it is setting up. This is not because you shouldn't register a variable twice (see next paragraph), but to help avoid a silent failure mode where an initial condition registers a variable but no simulation then makes use of it.

On the other hand, if two packages both make use of a variable for their calculations then they should both register it. This is especially important because part of the registration process is to say how many ghost cells a package needs for its calculations. Each package should register the number of ghost cells that it needs for its own algorithms, and the harness system will select the largest number of ghost cells that any package needs for a given variable and will allocate that many ghost cells for that variable. This is one reason why we use LARE style indexing with negative indices for ghost cells. Regardless of the number of ghost cells, the domain always starts at either 1 if the variable is cell centred, or 0 if the variable is face centred.

When you register a variable, you specify a name and a list of axes that the variable is defined on. The name must be unique among all variables, and should be descriptive of what the variable represents. You also specify the staggering of a variable on each axis, whether the array should reside on the CPU always or on the GPU (if GPU mode is being used) and the number of ghost cells that you need for that variable on each axis. So, taking LARE3D as an example again, we have the following

```cpp

void registerVariables(SAMS::harness& harness){
        auto &varRegistry = harness.variableRegistry;   // Take care to remember the reference marker & in this idiom!
        const int ghosts = 2; // 2 Ghost cells at top and bottom of each dimension
        //Register density (cell centred on all axes)
        varRegistry.registerVariable<T_dataType>("rho", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

        //Register x velocity (face centred on all axes)
        varRegistry.registerVariable<T_dataType>("vx", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        //Register By (face centred on Y, cell centred on X and Z)
        varRegistry.registerVariable<T_dataType>("by", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts));
}
```

This shows a fragment of the variable registration for LARE3D. The first variable is density, which is cell centred on all axes, so it is registered with the default staggering of `SAMS::staggerType::CENTRED` (NOTE the UK spelling!). The second variable is the x velocity, which is face centred on all axes, so it is registered with the staggering set to `SAMS::staggerType::HALF_CELL` on all axes. The third variable is the y component of the magnetic field, which is face centred on the Y axis and cell centred on the X and Z axes, so it is registered with the staggering set to `SAMS::staggerType::HALF_CELL` on the Y axis and left as the default on the X and Z axes. Note the that order of the axes in the variable registration maps the axes to the dimensions of the variable, so the first axis in the variable registration is the X axis, the second axis is the Y axis and the third axis is the Z axis. The MPI decomposition of the variable matches the specified MPI decomposition of the axes.

## `defaultValues` event

The purpose of this event is to set up default values for non array variables. The idea of this is that a package may want to set up default values that will be overriden by another package that is setting up an initial condition. For example, in LARE3D the `mu0` parameter of the `simulationData` data pack that represents the magnetic susceptibility is set to `mu0_si` the physically correct SI value. Most simulations will not want to change this value, but some simulations may work in normalised units and thus require LARE to run with `mu0` set to 1. 
Packages which use the default value set by LARE3D do not need to do anything - packages which require another value can override the default value in the `controlValues` event. 

NOTE: as mentioned above, the order in which packages apply their default and control values is fixed, so there are no 'race'-like conditions if multiple packages set the same value. However, two packages setting inconsistent values indicates that they are not compatible. So if your package sets values on the core solver, or values on other packages, please make sure to document this carefully. 

The philosophical difference between `defaultValues` and `controlValues` is that `defaultValues` is for setting up default values that are intended to be overridden by other packages, while `controlValues` is for setting up control variables that are intended to be used by a particular simulation. So in general a package implementing a problem would not set `defaultValues`, but would set `controlValues` to set up the control variables for that problem, while a package implementing a physics module would set `defaultValues` to set up default values for parameters that are used in the physics module but not `controlValues` because a physics module does not know what problem is going to be run and so cannot set up the control values. As a consequence of this, it would be unusual for `defaultValues` to read user input, but it is very common for `controlValues` to read user input, since the control values are intended to be set by the user for a particular simulation.

Mechanically this is a very simple event. It uses the ASMF and usually a package will request it's own datapack objects and then just set the values appropriately. For example, in LARE3D we have the following `defaultValues` function:

```cpp
    void LARE3D::defaultValues(simulationData &data){

        data.dt_multiplier = 0.8; // Default multiplier for time step

        // Geometry options: cartesian, cylindrical, spherical
        data.geometry = geometryType::Cartesian;

        // Shock viscosity coefficients
        data.visc1 = 0.1;
        data.visc2 = 1.0;

        // Ratio of specific heat capacities
        data.gas_gamma = 1.4;

        // Average mass of an ion in proton masses
        data.mf = 1.2;

        data.rke = false; // Remap kinetic energy correction off by default

        // Physical constants
        data.mu0 = mu0_si;
    }
```

This sets up sensible default values for values that LARE3D needs to have set but that a user may wish to override for a particular simulation.

## `controlValues` event

The `controlValues` event is for setting up control variables for a particular simulation. These are variables that are intended to be set by the user for a particular simulation, and that are intended to be used by a particular simulation. They generally mirror `defaultValues` since the purpose of this is to have a place for the user to set values that will override the default values set in `defaultValues`. There isn't much to add to the description of this event that wasn't said in the description of `defaultValues`. 

Generally `controlValues` will be implemented in a package that is setting up a particular problem. It may or may not parameterise the problem on values from the input deck, but it is also valid to hard code values.

## `setDomain` event

The `setDomain` event is mostly fairly simple, but quite crucial. There must be one and only one package that is responsible for actually giving the axes their final number of domain gridpoints and actual grid values. At the moment the tools that this even will use are in a fairly primitive state, but the general principle is in place.

Once again, one will almost always want the `SAMS::harness` object in this event since one once again uses the axis registry to set up the axes. This time for each axis one has to specify the number of grid points in the global domain and somehow the grid values. There are partial implementations for general stretched grids, but at present the only fully implemented option is for a uniform grid. For a uniform grid you specify the number of grid points and the position of the lower and upper boundaries and the grid will be generated. To give a simple example

```cpp
void setDomain(SAMS::harness& harness){
    auto& axRegistry = harness.axisRegistry;
    const int nx = 100;
    const int ny = 100;
    const int nz = 100;
    axRegistry.setDomain("X", nx, 0.0, 1.0);
    axRegistry.setDomain("Y", ny, 0.0, 1.0);
    axRegistry.setDomain("Z", nz, 0.0, 1.0);
}
```

The parameters here specify several things. The first parameter is just the name of the axis, the same name that you passed to `registerAxis`. The second parameter is the number of cells in the global domain (across all MPI ranks). Note that this is the number of **cells**, so if you set it to 1 then you will have one grid cell. The next two parameters are the position of the lower and upper boundaries of the domain. Note that these are the positions of the boundaries, so the entire domain will go from 0 to 1 in this case. That means that the position of the first cell centre will be at 0.5*dx, where dx is the grid spacing, and the position of the last cell centre will be at 1 - 0.5*dx. The harness generates both cell centred and face centred grid values for each axis, but you specify the domain in terms of the edges of the domain in all cases.

Future expansion of the axis registry will add the ability to specify the positions of grid sizes or grid edges, or the widths of cells and a lower bound to allow for more general stretched grids.

Generally, whatever package is responsible for setting up the problem will also set up the domain. This is one of the parts where SAMS works by contract as much as by code - most packages that do the actual work expect that their domains have been set up by other packages that set up their initial conditions. If the package setting up the initial conditions fails to do so then the physics package will fail when it tries to get its arrays from the registry.

### `getVariables` event

The `getVariables` event is the first event called after the memory for the actual axes and variables has been allocated. At this point the package can ask the harness for the variables that will hold simulation quantities, and for the axes that those variables are defined on. Once again, the `SAMS::harness` object is needed for this event, since one asks the `axisRegistry` for axes and the `variableRegistry` for variables. Usually a package will also want its data packs in this event so that the values in the data packs can be populated. For example, in LARE3D the following is a partial example of the `getVariables` event:

```cpp
void LARE3D::getVariables(SAMS::harness& harness, simulationData& data){
        auto &axRegistry = harness.axisRegistry;
        auto &varRegistry = harness.variableRegistry;

        axRegistry.fillPPLocalAxis("X", data.xc, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalAxis("X", data.xb, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalAxis("X", data.xc_host, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalAxis("X", data.xb_host, SAMS::staggerType::HALF_CELL);

        varRegistry.fillPPArray("rho", data.rho);

}
```

This needs some explanation. The first part is getting the axes that the package needs. The `fillPPLocalAxis` function is used to fill an array with the values of the axis on the local MPI rank. The first parameter is the name of the axis, the second parameter is the array to fill, and the third parameter is the staggering of the axis. So the first two lines are quite simple - it is getting the X axis values for the cell centred and face centred versions of the X axis and filling the `xc` and `xb` arrays in the data pack with those values. The next two lines look redundant, but they are actually different because of the definition of those members of the data pack.

```cpp
struct simulationData {
    //...
    portableWrapper::portableArray<SAMS::T_dataType, 1, SAMS::arrayTags::accelerated> xc; // Cell centred X axis values on the device
    portableWrapper::portableArray<SAMS::T_dataType, 1, SAMS::arrayTags::accelerated> xb; // Face centred X axis values on the device
    portableWrapper::portableArray<SAMS::T_dataType, 1, SAMS::arrayTags::host> xc_host; // Cell centred X axis values on the host
    portableWrapper::portableArray<SAMS::T_dataType, 1, SAMS::arrayTags::host> xb_host; // Face centred X axis values on the host
    //...
};
```

as you can see the difference is that the first two arrays are allocated on the device (if GPU mode is being used) while the second two arrays are allocated on the host. The `fillPPLocalAxis` function will allocate memory in the appropriate place and fill it with the appropriate values.

The variable grabbing function is quite similar but simpler. You simply specify the name of the variable and a suitable portable array to associate with the memory. Note that while the term used is "fill" the function is actually just associating the array with the memory that has already been allocated for that variable. Unlike with the axes a variable is defined as being either on the device or on the host when it is registered and passing an array with the wrong tags will cause a runtime error.


### `setBoundaryConditions` event

The purpose of this event is to provide a correctly positioned time in the sequence of events for setting up boundary conditions. It occurs just before the main simulation loop starts, so that all parts of the initial conditions are in place, so if a boundary condition wants to record the initial state of an edge for some reason then that data is available for it to do so. 

Setting up boundary conditions is detailed in the separate `Using SAMS Boundary Conditions` and `Developing SAMS Boundary Conditions` documents, but the general principle is that you use the ASMF to request the harness, and then you get the variable definitions from the variable registry and you attach boundary condition objects to the variable definitions. There are several built in boundary conditions that provide conventional boundary conditions, so as an example if you want to clamp the density at the lower X boundary to a value of 1.0 then you can do this:

```cpp
void setBoundaryConditions(SAMS::harness& harness){
    auto& varRegistry = harness.variableRegistry;
    auto& rhoDef = varRegistry.getVariable("rho"); // Take care not to miss the reference specifier, &
    rhoDef.addBoundaryCondition(0, SAMS::domain::edges::lower, SAMS::simpleClamp<SAMS::T_dataType, 3, SAMS::arrayTags::accelerated>(rhoDef, 1.0));
}
```

This is detailed better in the other documents, but you can see the basic principle. You attach a boundary condition to an edge by index (0 - first index, 1 - second index etc.) and by which edge (lower or upper). The boundary condition is an object that is called by the runner to apply the boundary condition at each time step. The `simpleClamp` boundary condition is a built in boundary condition that just clamps the value of the variable to a specified value at the boundary. In this case it clamps the density to 1.0 at the lower X boundary. The built in boundary conditions are templated on the type, rank and array tag just like the portable arrays, and each have their own constructor parameters that specify the details of the boundary condition.

### `registerOutputMesh` event

The `registerOutputMesh` event is one of the four events that is involved in outputting data from the simulation. It is another templated event, but this time it is templated on the inner type of the IO writer, so the minimal signature would look like

```cpp
template<typename T>
void registerOutputMesh(writer<T>& writer){
    //Register meshse output with the writer
}
```

You can still use the ASMF to request data packs etc. in this event, but the first parameter must be a reference to the writer that you want to register output with, and it must be templated on the inner type of the writer.

The purpose of this is to register the axes the package wants to output with the writer. This phase is to allow writing writer packages that need the variables and axes to be registered before the data is written. This often does nothing, because many writers do not need axes to be preregistered, but you should always implement it so that your package can be used with any writer, including writers that do need axes to be preregistered.

The details on writing these is documented in the `Writing data from SAMS packages` document, but the general principle is that you use the ASMF to request the writer that you want to register output with, and then you call the appropriate functions on the writer to register the variables and axes that you want to output. For example, if you want to register the density variable and the X axis with a writer then you would do something like this:

```cpp
    template<typename T_writer>
    void LARE3D::registerOutputMesh(writer<T_writer> &writer, simulationData &data)
    {
        writer.template registerRectilinearMesh<T_dataType>("MeshCC", data.nx, data.ny, data.nz);

        writer.template registerData<T_dataType>("rho", "MeshCC");
    }
```

### `registerOutputVariable` event

The `registerOutputVariable` event is one of the four events that is involved in outputting data from the simulation. It is another templated event, but this time it is templated on the inner type of the IO writer, so the minimal signature would look like

```cpp
template<typename T>
void registerOutputVariable(writer<T>& writer){
    //Register variables to output with the writer
}
```

You can still use the ASMF to request data packs etc. in this event, but the first parameter must be a reference to the writer that you want to register output with, and it must be templated on the inner type of the writer.

The purpose of this is to register the variables the package wants to output with the writer. This phase is to allow writing writer packages that need all variables and axes to be pre-specified before any data is written. This often does nothing, because many writers can write axes and variables one by one, but you should always implement it so that your package can be used with any writer, including writers that do need variables to be preregistered.

Note that there is a separate write mesh and write variable phase because some writers need the meshes to be registered before the variables, while others do not.

The details on writing these is documented in the `Writing data from SAMS packages` document, but the general principle is that you use the ASMF to request the writer that you want to register output with, and then you call the appropriate functions on the writer to register the variables and axes that you want to output. For example, if you want to register the density variable and the X axis with a writer then you would do something like this:

```cpp
    template<typename T_writer>
    void LARE3D::registerOutputVariable(writer<T_writer> &writer, simulationData &data)
    {

        writer.template registerData<T_dataType>("rho", "MeshCC");
    }
```


### `writeOutputMesh` event

This is the counterpart of `registerOutputMesh`, and is called immediately after the two registration events. The purpose of this event is to actually write the data that was registered in `registerOutput` to the output files. Again, this is detailed in the `Writing data from SAMS packages` document, but this shows a basic example of IO from LARE3D.

```cpp
    template <typename T_writer>
    void LARE3D::writeOutputMesh(writer<T_writer> &writer, simulationData &data)
    {
        pw::portableArrayManager manager;
        pw::portableArray<SAMS::T_dataType, 3, SAMS::arrayTags::host> host;

        writer.writeRectilinearMesh("MeshCC", &data.xc_host(1), &data.yc_host(1), &data.zc_host(1));

        getHostVersion(data, manager, data.rho, host);
        writer.writeData("rho", host.data());

    }

```

This shows the basic approach, and also shows the key point - the data has to be on the host to be written. More support code for this will likely be added to the SAMS framework in the future, but at the moment, the portable wrapper array manager method `makeHostAvailable` is likely the easiest way to get data onto the host for writing. This method allocates a new array if needed and copies data to the host. If the data is already on the host then the new array is just a view on the same data.


### `writeOutputMesh` event

This is the counterpart of `registerOutputMesh`, and is called immediately after the two registration events. The purpose of this event is to actually write the data that was registered in `registerOutputMesh` to the output files. Again, this is detailed in the `Writing data from SAMS packages` document, but this shows a basic example of IO from LARE3D.

```cpp
    template <typename T_writer>
    void LARE3D::writeOutputMesh(writer<T_writer> &writer, simulationData &data)
    {
        writer.writeRectilinearMesh("MeshCC", &data.xc_host(1), &data.yc_host(1), &data.zc_host(1));
    }

```

This shows the basic approach.

### `writeOutputVariable` event

This is the counterpart of `registerOutputVariable`, and is called immediately after the two registration events. The purpose of this event is to actually write the data that was registered in `registerOutputVariable` to the output files. Again, this is detailed in the `Writing data from SAMS packages` document, but this shows a basic example of IO from LARE3D.

```cpp
    template <typename T_writer>
    void LARE3D::writeOutputVariable(writer<T_writer> &writer, simulationData &data)
    {
        pw::portableArrayManager manager;
        pw::portableArray<SAMS::T_dataType, 3, SAMS::arrayTags::host> host;

        getHostVersion(data, manager, data.rho, host);
        writer.writeData("rho", host.data());

    }

```

This shows the basic approach, and also shows the key point - the data has to be on the host to be written. More support code for this will likely be added to the SAMS framework in the future, but at the moment, the portable wrapper array manager method `makeHostAvailable` is likely the easiest way to get data onto the host for writing. `getHostVersion` wraps this function away as well as removing the ghost cells from around the edge of the domain. This method allocates a new array if needed and copies data to the host. If the data is already on the host then the new array is just a view on the same data.

### `startOfTimestep` event

This event occurs at the start of each timestep and maps to the predictor step in a predictor corrector timestepping scheme. Physics packages should implement this event. There are no restrictions on the signature of this event, but generally a package will want to request the data packs that it needs in this event so that it can perform the predictor step of its calculations.

You will in general want to know the timestep and possibly the absolute time. Both of these are available from the `timeState` object. This is a built in object to the runner that is available through the ASMF, so simply add the `timeState` object to the signature of the `startOfTimestep` event and the ASMF will automatically pass it to you. The `timeState` object has a member variable called `dt` which is the time delta, and a member variable called `time` which is the current absolute time. Note that these are the values at the start of the timestep.

In LARE3D this stage is the predictor step of the predictor-corrector time stepping scheme.

Physics packages that want to implement an integrated predictor-corrector step should use this phase for their predictor step as well.

### `halfTimestep` event

This event occurs at the half timestep and maps to the corrector step in a predictor corrector timestepping scheme. Physics packages should implement this event. There are no restrictions on the signature of this event, but generally a package will want to request the data packs that it needs in this event so that it can perform the corrector step of its calculations. You may well also want the `timeState` object here, but it should be noted that the time is not updated at the half timestep, so you should add dt/2.0 to the time if you want the correct time at this stage. (NOTE THIS MAY CHANGE IN THE FUTURE, BUT THIS IS THE CURRENT BEHAVIOUR)

In LARE3D this stage is the corrector step of the predictor-corrector time stepping scheme.

Physics packages that want to implement an integrated predictor-corrector step should use this phase for their corrector step as well.

### `endOfTimestep` event

This event occurs at the end of each timestep and is for any calculations that need to be done at the end of the timestep. Physics packages may implement this event if they have any calculations that need to be done at the end of the timestep, but many packages will not need to implement this event. There are no restrictions on the signature of this event, but generally a package will want to request the data packs that it needs in this event so that it can perform whatever calculations it needs to do at the end of the timestep. You may well also want the `timeState` object here, and in this case it will have the updated time and timestep values for the next timestep.

In LARE3D this stage is used for the remap step of the LARE3D time stepping scheme, and also for the kinetic energy correction if that is turned on.

Other physics packages should only use this step if they want to fully operator split their calculations from the main predictor-corrector steps, since this is the last stage of the timestep and so it is not possible to do any further calculations on the updated variables until the next timestep. The advantage of this is that the grid in LARE is unambiguously back to the Eulerian state.

### `queryOutput` event

This event is triggered after the update sequence completes. This event has a single bool reference parameter as the first parameter. If a given physics package wants to trigger output at the end of a given timestep, then it should set this parameter to true. You should NEVER set this parameter to false (as that would cancel an output request from any prior packages), so the correct way to use this is just

```cpp
void queryOutput(bool& outputThisStep){
    if (/*some condition for output*/){
        outputThisStep = true;
    }
}
```

This event uses the ASMF, so you can request data packs etc. in this event as well, but the first parameter must be a reference to a bool that is used to trigger output.

### `queryTerminate` event

This event is triggered after the update sequence completes and after the `queryOutput` event. This event has a single bool reference parameter as the first parameter. If a given physics package wants to trigger termination of the simulation at the end of a given timestep, then it should set this parameter to true. You should NEVER set this parameter to false, (as that would cancel a terminate request from any prior packages), so the correct way to use this is just

```cpp
void queryTerminate(bool& terminateThisStep){
    if (/*some condition for termination*/){
        terminateThisStep = true;
    }
}
```

This event uses the ASMF, so you can request data packs etc. in this event as well, but the first parameter must be a reference to a bool that is used to trigger termination of the simulation.


### `finalize` event

This event is called at the end of the simulation and is intended to allow a simulation to tear down all of its resources. While generally this event will be called just before the code terminates, it is good practice to manually release all memory etc. in this event, and to do any other clean up that is needed. Variables allocated through the variable registry will be automatically deallocated by the runner, but if you have allocated any other resources then you should release them in this event. For example, if you have allocated any memory that is not managed by the variable registry, or if you have opened any files, then you should release that memory and close those files in this event.

### `calculateTimestep` event

The only thing that hasn't been described is setting the timestep. The runner cannot itself calculate the timestep, but it does find the minimum timestep across all packages and across all MPI ranks. This is done by calling the `calculateTimestep` event on each package at the end of each timestep. So if a package needs to set a specific timestep for any reason, then it should implement the `calculateTimestep` event and set the desired timestep from that event. If a package does not implement the `calculateTimestep` event then it is assumed that that package has no constraints on the timestep. This is done by the function requesting the `timeState` object and setting the dt member variable to the **minimum of the current value of dt and the desired timestep**. For example, in LARE3D we have the following `calculateTimestep` function:

```cpp
        void calculateTimestep(SAMS::timeState &timeData, simulationData &data){
            set_dt(data);
            timeData.dt = data.dt<timeData.dt ? data.dt : timeData.dt;
        }
```

### `getTimestep` event

A package must not assume that the timestep that it has specified is the smallest of all packages, or the minimum across all ranks, so there is a separate `getTimestep` event that is called after the required timestep has been calculated. This allows a package to get the actual timestep that will be used for the next timestep, and to update any internal data structures that need to be updated with the new timestep. For example, in LARE3D we have the following `getTimestep` function:

```cpp
        void getTimestep(SAMS::timeState &timeData, simulationData &data){
            data.dt = timeData.dt;
        }
```

### When the timestep is set

Different core solver packages may need timestep to be set a different times, so there isn't a specific timing for the `calculateTimestep` or `getTimestep` events, although they do immediately follow each other. They are triggered in response to a request from a core solver package. That request is done using an object called `controlFunctions`. This object can be requested through the ASMF in any function and is essentially a type erased wrapper around the runner class. As SAMS evolves it will have a range of functions, but at the moment it has two

1) `calculateTimestep()` - this function triggers the `calculateTimestep` event on all packages to obtain the minimum timestep across all packages and all ranks. This is called by a core solver package when it needs to know the timestep for the next timestep.

2) `isPackageActive(name)` - this function returns true if a package with the given name is active in the current simulation. This is for use in packages that need to know whether another package is active or not, for example because they need to know whether they can use a variable that is only registered by that package.
