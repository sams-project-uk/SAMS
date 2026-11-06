#Functions to read the variables and meshes from a SAMS HDF5 output file
import h5py
import numpy as np

#Names of the axis datasets in a mesh group, in the order of the dimensions of the data
axisDatasetNames = ["x", "y", "z", "a", "b", "c", "d", "e"]


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


def readVariable(h5file, varName):
    #Read the values of a variable into memory
    return np.asarray(h5file[varName][...])


def readAxes(h5file, varName, shape):
    #Read the coordinates of each axis of a variable from its mesh
    #Each entry is (axis name, coordinates), with coordinates of None if the mesh does not give them
    mesh = meshName(h5file, varName)
    group = h5file.get(mesh) if mesh is not None else None
    axes = []
    for dim in range(len(shape)):
        name = axisDatasetNames[dim] if dim < len(axisDatasetNames) else "axis" + str(dim)
        label = name
        coords = None
        if isinstance(group, h5py.Group) and name in group:
            coords = np.asarray(group[name][...])
            #The writer stores the name of the axis on its dataset. Older files do not have it, so they keep the name from the position
            stored = group[name].attrs.get("axisName")
            if stored is not None:
                label = stored.decode() if isinstance(stored, bytes) else str(stored)
        axes.append((label, coords))
    return axes
