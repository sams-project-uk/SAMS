#Functions to read the variables and meshes from a SAMS ADIOS2 output file
#Uses the adios2 Python bindings from version 2.10, which have FileReader
import numpy as np
from adios2 import FileReader


def isADIOS2File(path):
    #SAMS ADIOS2 files have the extension .bp
    return path.rstrip("/").endswith(".bp")


def openFile(path):
    #Open a SAMS ADIOS2 output file for reading
    return FileReader(path.rstrip("/"))


def read(dump, name, step):
    #Read a variable at a step. A file that dumps share has a step for each dump. Everything in a step, including the metadata, is a variable
    #A single number is read as an array of one element, or of no dimensions if the file has one step, so the element is returned. A single string is read as itself
    value = dump.read(name, step_selection=[step, 1])
    if isinstance(value, str) or dump.available_variables()[name]["Shape"].strip():
        return value
    return np.asarray(value).item()


def stepCount(dump):
    #The number of dumps in the file
    return dump.num_steps()


def readTime(dump, step):
    #The simulation time of a dump, which is the variable time, or None if the file does not have it
    return float(read(dump, "time", step)) if "time" in dump.available_variables() else None


def listVariables(dump):
    #The variables are the ADIOS2 variables with a mesh name, which is the variable variable/meshName. Axes and blobs have none
    available = dump.available_variables()
    return sorted(name for name in available if name + "/meshName" in available)


def meshName(dump, varName, step):
    #Name of the mesh that a variable is on
    return read(dump, varName + "/meshName", step)


def readShapes(dump, variables):
    #Shapes of the variables, from the sizes separated by commas that ADIOS2 lists for each
    available = dump.available_variables()
    shapes = {}
    for varName in variables:
        shape = available[varName]["Shape"]
        shapes[varName] = tuple(int(size) for size in shape.split(",")) if shape.strip() else ()
    return shapes


def readVariable(dump, varName, step):
    #Read the values of a variable into memory
    return np.asarray(read(dump, varName, step))


def readAxes(dump, varName, shape, step):
    #Read the coordinates of each element of a variable along each axis of its mesh
    #A mesh names an axis variable for each dimension in meshes/<mesh>/axis<dimension>
    #All positions are core indices. An axis gives the index of its first element in axis/start,
    #and the variable gives the index of its first element in each dimension in variable/starts
    #Each entry is (axis name, coordinates)
    mesh = "meshes/" + meshName(dump, varName, step)
    dataStarts = read(dump, varName + "/starts", step)
    axes = []
    for dim in range(len(shape)):
        axisName = read(dump, mesh + "/axis" + str(dim), step)
        first = int(dataStarts[dim]) - int(read(dump, axisName + "/start", step))
        coords = np.asarray(read(dump, axisName, step))[first:first + shape[dim]]
        #The axis is labelled by the last part of its name
        axes.append((axisName.split("/")[-1], coords))
    return axes


def readPlottedSlices(dump, varName, shape, step):
    #Slices of a variable that leave only the area that its mesh plots
    #The mesh gives the core index of its first cell and its number of cells in each dimension in meshes/<mesh>/starts and sizes,
    #and the variable gives the core index of its first element in each dimension in variable/starts
    mesh = "meshes/" + meshName(dump, varName, step)
    meshStarts = read(dump, mesh + "/starts", step)
    meshSizes = read(dump, mesh + "/sizes", step)
    dataStarts = read(dump, varName + "/starts", step)
    slices = []
    for dim in range(len(shape)):
        lower = int(meshStarts[dim]) - int(dataStarts[dim])
        slices.append(slice(lower, lower + int(meshSizes[dim])))
    return tuple(slices)
