#3D window for a variable with three axes to plot
import threading

import numpy as np
import pyvista

import plotWindow


def show3D(varName, data, axes):
    #A variable with three axes to plot is shown in a 3D window, with a slice plane along each axis that can be moved
    edges = [coords is not None and len(coords) == cells + 1 for (name, coords), cells in zip(axes, data.shape)]
    if all(edges):
        #The mesh gives the cell edges, so the values are on the cells of the grid
        grid = pyvista.RectilinearGrid(*[np.asarray(coords, dtype=float) for name, coords in axes])
        grid.cell_data[varName] = data.ravel(order="F")
    else:
        #Otherwise the values are on the points of the grid
        grid = pyvista.RectilinearGrid(*[np.asarray(plotWindow.linePositions(coords, cells), dtype=float) for (name, coords), cells in zip(axes, data.shape)])
        grid.point_data[varName] = data.ravel(order="F")
    plotter = pyvista.Plotter(title=varName)
    plotter.add_mesh(grid.outline(), color="black")
    plotter.add_mesh_slice_orthogonal(grid, scalars=varName)
    plotter.show_bounds(xtitle=axes[0][0], ytitle=axes[1][0], ztitle=axes[2][0])
    plotter.show()


def open3DWindow(varName, data, axes):
    #The 3D window runs in its own thread so that the list of variables can still be used while it is open
    threading.Thread(target=show3D, args=(varName, data, axes), daemon=True).start()
