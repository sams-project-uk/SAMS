# Using SAMS Boundary Conditions

SAMS boundary conditions are designed to be flexible and powerful, allowing you to specify boundary conditions by allowing you to add boundary condition objects to each edge of each variable while abstracting most of the details of how the boundary conditions are applied to the simulation. This means that you can specify a wide variety of boundary conditions, including ones that are not commonly used, without having to write any code that directly interacts with the main simulation loop.

It is assumed that you have already read the `Developing SAMS Packages` document since there is no reason to use boundary conditions without developing a package. Since there are powerful built in boundary conditions, this document does not assume that you have read the `Developing SAMS Boundary Conditions` document, but it is recommended that you read that document if you want to understand how to develop your own boundary conditions or if you want to understand how the built in boundary conditions work.

## Theory of boundary conditions

This is covered in more depth in the document on `Developing SAMS Boundary Conditions`, but the general principle is that boundary condition objects are descended from a base class called `boundaryCondition` and they have to implement a single virtual method called `apply`. There is nothing else specific to a boundary condition and a developer of a boundary condition has a lot of freedom in how to implement them. 

From the perspective of a non-boundary condition developer most of this doesn't matter. There are only two elements that you need to know

1) How to add a boundary condition to a variable definition

2) How to call a boundary condition so that it applies the boundary condition to the variable

## How to add boundary conditions

You add boundary conditions to variable definitions, which you get from the variable registry in the harness. You should add boundary conditions in the `setBoundaryConditions` event, which is a special event that is called once at the start of the simulation.

To actually add a boundary condition you have three options:

### Create a boundary condition object and copy it to the variable definition

```cpp
void setBoundaryConditions(SAMS::harness& harness){
    {
        auto& varDef = harness.getVariableRegistry().getVariableDefinition("rho");
        auto bc = SAMS::simpleClamp<double,3,SAMS::arrayTags::accelerated>(varDef, 1.0);
        //Attach to the lower X edge (0 is the X direction, lower is the lower edge)
        varDef.addBoundaryCondition(0, SAMS::boundaryConditions::edge::lower, bc);
    }

    {
        auto& varDef = harness.getVariableRegistry().getVariableDefinition("vx");
        auto bc = SAMS::simpleClamp<double,3,SAMS::arrayTags::accelerated>(varDef, 0.0);
        //Attach to both edges in the Y direction
        varDef.addBoundaryCondition(1, bc);
    }

    {
        auto& varDef = harness.getVariableRegistry().getVariableDefinition("vz");
        auto bc = SAMS::simpleClamp<double,3,SAMS::arrayTags::accelerated>(varDef, 0.0);
        //Attach to all edges
        varDef.addBoundaryCondition(bc);
    }

}
```

Now the above example is very poor since it doesn't actually add enough boundary conditions to any variable other then vz, but it is just meant to show how to add boundary conditions. The first example shows how to add a boundary condition to a single edge, the second example shows how to add a boundary condition to both edges in a single direction, and the third example shows how to add a boundary condition to all edges. 

This method of adding boundary conditions is the most straightforward, but it does involve copying the boundary condition object to the variable definition, which can be expensive if the boundary condition object is large. None of the built in boundary conditions are more than a few hundred bytes, but this can matter. It is also worth noting that it only copies the boundary condition **once**. The versions that add to multiple edges only copy the boundary condition once and then they add the same boundary condition object to each edge, this means that

```cpp
varDef.addBoundaryCondition(1, bc);
```

is not exactly the same as

```cpp
varDef.addBoundaryCondition(1, SAMS::boundaryConditions::edge::lower, bc);
varDef.addBoundaryCondition(1, SAMS::boundaryConditions::edge::upper, bc);
```

The first version adds the same boundary condition object to both edges, while the second version creates two copies of the boundary condition object and addes them to each edge. Again, for the built in boundary conditions this doesn't matter in the least, but it can be important for large boundary conditions or boundary conditions that rely on tracking state since the first version will have a single state that is shared between both edges, while the second version will have two separate states.

If you wish to deliberately add the same boundary condition object on multiple edges then you should capture the return value from the `addBoundaryCondition` method. The result will be a shared_ptr to a base `boundaryCondition` object that is the boundary condition that was added to the variable definition. You can then use this shared pointer to add the same boundary condition to multiple edges, and you can also use it to modify the boundary condition after it has been added to the variable definition if you need to.

```cpp
void setBoundaryConditions(SAMS::harness& harness){
    auto& varDef = harness.getVariableRegistry().getVariableDefinition("rho");
    auto bc = SAMS::simpleClamp<double,3,SAMS::arrayTags::accelerated>(varDef, 1.0);
    std::shared_ptr<boundaryCondition> bcPtr = varDef.addBoundaryCondition(0, SAMS::boundaryConditions::edge::lower, bc);
    varDef.addBoundaryCondition(0, SAMS::boundaryConditions::edge::upper, bcPtr);
}
```

This also shows the second way in which you can add boundary conditions

### Create a std::shared_ptr to a boundary condition and add that to the variable definition

```cpp
void setBoundaryConditions(SAMS::harness& harness){
    auto& varDef = harness.getVariableRegistry().getVariableDefinition("rho");
    std::shared_ptr<boundaryCondition> bcPtr = std::make_shared<SAMS::simpleClamp<double,3,SAMS::arrayTags::accelerated>>(varDef, 1.0);
    varDef.addBoundaryCondition(0, SAMS::boundaryConditions::edge::lower, bcPtr);
    varDef.addBoundaryCondition(0, SAMS::boundaryConditions::edge::upper, bcPtr);
}
```

This is little different to the first method really. You just create a shared pointer to the boundary condition and then you add that to the variable definition. This is more efficient if you want to add the same boundary condition to multiple edges since it only creates a single boundary condition object and then it adds that same object to each edge, while the first method would create a separate boundary condition object for each edge. The example given above uses std::make_shared to create the boundary condition object from the constructor parameters, but you could also create the boundary condition object in some other way and then create a shared pointer to it if you wanted to.

While it isn't really necessary, this method also returns a shared_ptr to the boundary condition that you have just added.

As well as avoiding unnecessary copying of boundary condition objects, this method also allows you to create boundary conditions that cannot be copied. The third method of adding boundary conditions is also to deal with this case but without the shared_ptr boilerplate.

### emplaceBoundaryCondition

```cpp
void setBoundaryConditions(SAMS::harness& harness){
    auto& varDef = harness.getVariableRegistry().getVariableDefinition("rho");
    varDef.emplaceBoundaryCondition<SAMS::simpleClamp<double,3,SAMS::arrayTags::accelerated>>(0, SAMS::boundaryConditions::edge::lower, varDef, 1.0);
    varDef.emplaceBoundaryCondition<SAMS::simpleClamp<double,3,SAMS::arrayTags::accelerated>>(0, SAMS::boundaryConditions::edge::upper, varDef, 1.0);
}
```

`emplaceBoundaryCondition` is a method that allows you to construct the boundary condition object directly in the variable definition without having to create a separate boundary condition object and then add it to the variable definition. It is pretty much pure syntactic sugar. As with the previous methods it returns a shared_ptr to the boundary condition that was just added, so you can use that to add the same boundary condition to multiple edges if you want to.

## Built in boundary conditions

There are four built in boundary conditions currently provided with SAMS. All of them work for any type, rank and memory space having any grid staggering so long as they are appropriately templated.

### simpleClamp

Clamp all boundary values to a specified value. The type is

```cpp
SAMS::simpleClamp<valueType, rank, memorySpace> (variableDefinition, clampValue);
```

### simpleMirrorBC

Calculate the value on the boundary edge and mirror ghost cells across the boundary edge. The type is

```cpp
SAMS::simpleMirrorBC<valueType, rank, memorySpace> (variableDefinition);
```

### simpleZeroGradientBC

Set the values on the boundary so that the gradient across the boundary is zero. The type is

```cpp
SAMS::simpleZeroGradientBC<valueType, rank, memorySpace> (variableDefinition);
```

### simplePeriodicBC

Set the values on the boundary so that they are periodic with the opposite edge. The type is

```cpp
SAMS::simplePeriodicBC<valueType, rank, memorySpace> (variableDefinition);
```


### How to apply a boundary condition

Within your package it is very easy to apply a boundary condition. If you have the variable definition for the variable that you want to apply the boundary condition to then you can just call the `applyBoundaryConditions` method on the variable definition and it will apply all of the boundary conditions that are attached to that variable definition.

```cpp
auto& varDef = harness.variableRegistry.getVariableDefinition("rho"); // Take care not to forget the reference specifier '&' here
varDef.applyBoundaryConditions();
```

as a convenience there is also an `applyBoundaryConditions` method on the variable registry that will apply the boundary conditions to a named variable, so you can also do this

```cpp
harness.variableRegistry.applyBoundaryConditions("rho");
```

If you wish to apply the boundary conditions to only specific dimensions or even only specific edges then you can also specify that as arguments to the `applyBoundaryConditions` method. For example if you only wanted to apply the boundary conditions on the lower X edge then you could do this

```cpp
varDef.applyBoundaryConditions(0);
varDef.applyBoundaryConditions(0, SAMS::boundaryConditions::edge::lower);
```

Applying the boundary conditions automatically calls the MPI communication routines as needed to deal with MPI interprocessor boundaries while applying boundary conditions to real edges. This means that if MPI periodic dimensions are set up no boundary conditions will be applied on any MPI periodic edge.
