#Window showing a visualisation of one variable from a SAMS output file
import tkinter as tk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import plot1D
import plot2D
#The 3D view needs pyvista. Without it the 3D view is not available
try:
    import plot3D
    has3D = True
except ImportError:
    has3D = False

#Axes with this many cells or fewer are ignorable, and the variable is plotted without them
maxIgnorableCells = 3


def countPlottedAxes(shape):
    #Number of axes of a variable that are plotted. These are the axes that are not ignorable, up to three
    return min(sum(1 for cells in shape if cells > maxIgnorableCells), 3)


def axisPositions(coords, cells):
    #Positions along an axis for plotting. The mesh can give either the cell positions or the cell edges
    if coords is not None and len(coords) in (cells, cells + 1):
        return coords
    return np.arange(cells)


def linePositions(coords, cells):
    #Positions along an axis for a line plot, which needs one position per cell
    positions = axisPositions(coords, cells)
    if len(positions) == cells + 1:
        return 0.5 * (positions[1:] + positions[:-1])
    return positions


def timeText(time):
    #The time of a dump as text, which is N/A for a dump that does not have its time
    return "time = " + (format(time, "g") if time is not None else "N/A")


def plotValue(figure, varName, data):
    #A variable with no axes left to plot is shown as its value
    figure.clear()
    figure.text(0.5, 0.5, varName + " = " + str(data), ha="center", va="center")


def openPlotWindow(root, fileFormat, dump, varName, step, viewDims, clipGhosts):
    #Open a window showing a visualisation of a variable at a step. fileFormat is the module that reads the format of the open file dump
    #The window always shows that step, and gives its time in its title
    data = fileFormat.readVariable(dump, varName, step)
    axes = fileFormat.readAxes(dump, varName, data.shape, step)
    title = varName + " (" + timeText(fileFormat.readTime(dump, step)) + ")"

    #Leave out the ghost cells if they are clipped. The coordinates of each axis are clipped with the data
    if clipGhosts:
        slices = fileFormat.readPlottedSlices(dump, varName, data.shape, step)
        data = data[slices]
        axes = [(name, coords[dimSlice] if coords is not None else None) for (name, coords), dimSlice in zip(axes, slices)]

    #Remove the ignorable axes by taking the middle cell along each of them
    keptAxes = []
    index = []
    for dim, cells in enumerate(data.shape):
        if cells <= maxIgnorableCells:
            index.append(cells // 2)
        else:
            index.append(slice(None))
            keptAxes.append(axes[dim])
    data = data[tuple(index)]

    #Only three axes can be shown, so take the middle cell along any others
    while data.ndim > 3:
        data = data[..., data.shape[-1] // 2]
        keptAxes.pop()

    #viewDims is the number of axes of the view that was chosen. A view with fewer axes than the variable shows a slice or a line through it
    if data.ndim == 3 and viewDims == 3:
        plot3D.open3DWindow(title, data, keptAxes)
        return

    window = tk.Toplevel(root)
    window.title(title)
    figure = Figure(figsize=(7, 6))
    canvas = FigureCanvasTkAgg(figure, master=window)
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    if data.ndim == 0:
        plotValue(figure, varName, data)
    elif data.ndim == 1:
        plot1D.plotLine(figure, varName, data, keptAxes)
    elif viewDims == 1:
        plot1D.addLineControls(window, figure, canvas, varName, data, keptAxes)
    elif data.ndim == 2:
        plot2D.addImageControls(window, figure, canvas, varName, data, keptAxes)
    else:
        plot2D.addSliceControls(window, figure, canvas, varName, data, keptAxes)
    canvas.draw()
