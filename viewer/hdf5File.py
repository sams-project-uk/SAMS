#Functions to read the variables and meshes from a SAMS HDF5 output file
import h5py
import numpy as np

def isHDF5File(path):
    #Only HDF5 files with the extension used by SAMS can be opened
    return path.endswith(".h5") and h5py.is_hdf5(path)


def openFile(path):
    #Open a SAMS HDF5 output file for reading
    return h5py.File(path, "r")


def listVariables(h5file):
    #The variables are the datasets in the root group. The axes are in the mesh groups, so they are not listed
    return sorted(name for name, item in h5file.items() if isinstance(item, h5py.Dataset))


def meshName(h5file, varName):
    #Name of the mesh that a variable is on, or None if it has no mesh
    name = h5file[varName].attrs.get("meshName")
    if name is None:
        return None
    if isinstance(name, bytes):
        name = name.decode()
    return str(name)


def readShapes(h5file, variables):
    #Shapes of the variables, read once so that they can be looked up without touching the file
    #Only the metadata of each variable is read, not its values
    return {varName: h5file[varName].shape for varName in variables}


def stepCount(h5file):
    #An HDF5 file holds one dump
    return 1


def readVariable(h5file, varName, step):
    #Read the values of a variable into memory
    return np.asarray(h5file[varName][...])


def attributeString(attrs, key):
    #A string attribute, or None if there is no attribute with that key
    value = attrs.get(key)
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def readTime(h5file, step):
    #The simulation time of the dump, which is the attribute time of the root group, or None if the dump does not have it
    time = attributeString(h5file.attrs, "time")
    return float(time) if time is not None else None


def readAxes(h5file, varName, shape, step):
    #Read the coordinates of each element of a variable along each axis of its mesh
    #A mesh is a group in the meshes group. For each dimension it names an axis dataset in the attribute axis<dimension>
    #All positions are core indices. An axis gives the index of its first element in the attribute start,
    #and the variable gives the index of its first element in each dimension in the attribute starts
    #Each entry is (axis name, coordinates), with coordinates of None if the mesh does not give them
    mesh = meshName(h5file, varName)
    group = h5file.get("meshes/" + mesh) if mesh is not None else None
    dataStarts = h5file[varName].attrs["starts"]
    axes = []
    for dim in range(len(shape)):
        label = "axis" + str(dim)
        coords = None
        axisName = attributeString(group.attrs, "axis" + str(dim)) if isinstance(group, h5py.Group) else None
        if axisName is not None:
            #The axis is labelled by the last part of its name
            label = axisName.split("/")[-1]
            first = int(dataStarts[dim]) - int(h5file[axisName].attrs["start"][0])
            coords = np.asarray(h5file[axisName][first:first + shape[dim]])
        axes.append((label, coords))
    return axes


def readPlottedSlices(h5file, varName, shape, step):
    #Slices of a variable that leave only the area that its mesh plots
    #The mesh gives the core index of its first cell and its number of cells in each dimension in the attributes starts and sizes,
    #and the variable gives the core index of its first element in each dimension in the attribute starts
    mesh = meshName(h5file, varName)
    group = h5file["meshes/" + mesh]
    dataStarts = h5file[varName].attrs["starts"]
    slices = []
    for dim in range(len(shape)):
        lower = int(group.attrs["starts"][dim]) - int(dataStarts[dim])
        slices.append(slice(lower, lower + int(group.attrs["sizes"][dim])))
    return tuple(slices)
