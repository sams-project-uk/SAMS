#Functions to read the variables and meshes from a SAMS simple output directory
#Each item is a file with a text header, the number of header lines then the lines, followed by its values in native byte order with the last dimension fastest
import os

import numpy as np

#The numpy type of each category and size in bytes that a header gives
numpyTypes = {
    ("float", 4): np.float32, ("float", 8): np.float64,
    ("int", 1): np.int8, ("int", 2): np.int16, ("int", 4): np.int32, ("int", 8): np.int64,
    ("uint", 1): np.uint8, ("uint", 2): np.uint16, ("uint", 4): np.uint32, ("uint", 8): np.uint64,
    ("complex", 8): np.complex64, ("complex", 16): np.complex128,
}


class simpleDump:
    #An open simple output directory. There is nothing to close
    def __init__(self, path):
        self.path = path

    def close(self):
        pass


def isSimpleFile(path):
    #A simple output is a directory. ADIOS2 files are also directories, with the extension .bp
    return os.path.isdir(path) and not path.rstrip("/").endswith(".bp")


def openFile(path):
    #Open a SAMS simple output directory for reading
    return simpleDump(path.rstrip("/"))


def readHeader(fileName):
    #The header lines of an item and the offset of its values
    with open(fileName, "rb") as file:
        count = int(file.readline())
        lines = [file.readline().decode().rstrip("\n") for line in range(count)]
        return lines, file.tell()


def numpyType(line):
    #The numpy type of a header line of the category and size in bytes
    category, size = line.split()
    return numpyTypes[(category, int(size))]


def listVariables(dump):
    #The variables are the .data files, named by their path in the directory without the extension
    names = []
    for directory, subdirectories, files in os.walk(dump.path):
        for fileName in files:
            if fileName.endswith(".data"):
                relative = os.path.relpath(os.path.join(directory, fileName[:-len(".data")]), dump.path)
                names.append(relative.replace(os.sep, "/"))
    return sorted(names)


def dataHeader(dump, varName):
    #The header of a variable is the mesh name, the type, the sizes, then the core index of its first element in each dimension
    return readHeader(dump.path + "/" + varName + ".data")


def readShapes(dump, variables):
    #Shapes of the variables, from their headers
    return {varName: tuple(int(size) for size in dataHeader(dump, varName)[0][2].split()) for varName in variables}


def stepCount(dump):
    #A simple output directory holds one dump
    return 1


def readTime(dump, step):
    #The simulation time of the dump, which is the key time in the keyValues file of the directory
    for line in readHeader(dump.path + "/keyValues")[0]:
        key, value = line.split(" ", 1)
        if key == "time":
            return float(value)


def readVariable(dump, varName, step):
    #Read the values of a variable into memory
    lines, offset = dataHeader(dump, varName)
    shape = tuple(int(size) for size in lines[2].split())
    return np.fromfile(dump.path + "/" + varName + ".data", dtype=numpyType(lines[1]), count=int(np.prod(shape)), offset=offset).reshape(shape)


def meshHeader(dump, varName):
    #The header of the mesh of a variable is the rank, then for each dimension the axis, the first cell and the number of cells
    return readHeader(dump.path + "/" + dataHeader(dump, varName)[0][0] + ".mesh")[0]


def readAxes(dump, varName, shape, step):
    #Read the coordinates of each element of a variable along each axis of its mesh
    #All positions are core indices. The header of an axis gives the index of its first element on its fourth line,
    #and the variable gives the index of its first element in each dimension on the fourth line of its header
    #Each entry is (axis name, coordinates)
    dataStarts = dataHeader(dump, varName)[0][3].split()
    meshLines = meshHeader(dump, varName)
    axes = []
    for dim in range(len(shape)):
        axisName = meshLines[1 + dim].split()[0]
        axisFile = dump.path + "/" + axisName + ".axis"
        axisLines, offset = readHeader(axisFile)
        values = np.fromfile(axisFile, dtype=numpyType(axisLines[0]), count=int(axisLines[1]), offset=offset)
        first = int(dataStarts[dim]) - int(axisLines[2])
        #The axis is labelled by the last part of its name
        axes.append((axisName.split("/")[-1], values[first:first + shape[dim]]))
    return axes


def readPlottedSlices(dump, varName, shape, step):
    #Slices of a variable that leave only the area that its mesh plots
    #The mesh gives the core index of its first cell and its number of cells in each dimension,
    #and the variable gives the core index of its first element in each dimension
    dataStarts = dataHeader(dump, varName)[0][3].split()
    meshLines = meshHeader(dump, varName)
    slices = []
    for dim in range(len(shape)):
        axisName, start, size = meshLines[1 + dim].split()
        lower = int(start) - int(dataStarts[dim])
        slices.append(slice(lower, lower + int(size)))
    return tuple(slices)
