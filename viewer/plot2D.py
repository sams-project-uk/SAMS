#Colour map of a variable with two axes to plot, or of a slice through a variable with three axes to plot
import itertools
import tkinter as tk
from tkinter import ttk

import plotWindow

#Colour tables offered in the drop-down. The first is the default
colourTables = ["viridis", "plasma", "inferno", "magma", "cividis", "gray", "coolwarm", "RdBu_r", "seismic", "jet"]


def addColourChoice(parent, redraw):
    #Add a drop-down to choose the colour table, which redraws the plot when it changes. Returns the drop-down
    ttk.Label(parent, text="Colours").pack(side=tk.LEFT)
    colourChoice = ttk.Combobox(parent, values=colourTables, state="readonly", width=10)
    colourChoice.current(0)
    colourChoice.pack(side=tk.LEFT)
    colourChoice.bind("<<ComboboxSelected>>", redraw)
    return colourChoice


def plotImage(figure, varName, data, axes, title=None, colourTable=None):
    #A variable with two axes to plot is shown as a colour map
    figure.clear()
    plot = figure.add_subplot(111)
    (xName, xCoords), (yName, yCoords) = axes
    #The first index of the data is along the horizontal axis
    mesh = plot.pcolormesh(plotWindow.axisPositions(xCoords, data.shape[0]), plotWindow.axisPositions(yCoords, data.shape[1]), data.T, shading="auto", cmap=colourTable)
    figure.colorbar(mesh, ax=plot, label=varName)
    plot.set_xlabel(xName)
    plot.set_ylabel(yName)
    plot.set_title(title if title else varName)


def addImageControls(window, figure, canvas, varName, data, axes):
    #A variable with two axes to plot is shown as a colour map, with the colour table chosen from a list
    controls = ttk.Frame(window)
    controls.pack(side=tk.BOTTOM, fill=tk.X)

    def redraw(*args):
        plotImage(figure, varName, data, axes, colourTable=colourChoice.get())
        canvas.draw_idle()

    colourChoice = addColourChoice(controls, redraw)
    redraw()


def addSliceControls(window, figure, canvas, varName, data, axes):
    #A variable with three axes to plot is shown as a 2D slice. The pair of axes in the plane of the slice is chosen
    #from a list, and the cell of the slice across the other axis is chosen with the slider or by scrolling over the plot
    pairs = list(itertools.combinations(range(3), 2))
    controls = ttk.Frame(window)
    controls.pack(side=tk.BOTTOM, fill=tk.X)
    ttk.Label(controls, text="Plane").pack(side=tk.LEFT)
    pairChoice = ttk.Combobox(controls, values=[axes[first][0] + "-" + axes[second][0] for first, second in pairs], state="readonly", width=8)
    pairChoice.current(0)
    pairChoice.pack(side=tk.LEFT)
    colourChoice = addColourChoice(controls, lambda *args: redraw())
    ttk.Label(controls, text="Cell").pack(side=tk.LEFT)
    position = tk.Scale(controls, orient=tk.HORIZONTAL, **{"from": 0, "to": 0})
    position.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def acrossAxis():
        #The axis across the slice is the one that is not in the plane
        first, second = pairs[pairChoice.current()]
        return 3 - first - second

    def redraw(*args):
        first, second = pairs[pairChoice.current()]
        across = acrossAxis()
        cells = data.shape[across]
        cell = min(int(position.get()), cells - 1)
        index = [slice(None)] * 3
        index[across] = cell
        name, coords = axes[across]
        where = plotWindow.linePositions(coords, cells)[cell]
        plotImage(figure, varName, data[tuple(index)], [axes[first], axes[second]], varName + " at " + name + " = " + format(where, "g"), colourChoice.get())
        canvas.draw_idle()

    def changePlane(*args):
        #A new plane starts with the slice in the middle of the axis across it
        cells = data.shape[acrossAxis()]
        position.configure(to=cells - 1)
        position.set(cells // 2)
        redraw()

    def scroll(step):
        #The slider keeps the cell inside the axis
        position.set(int(position.get()) + step)

    pairChoice.bind("<<ComboboxSelected>>", changePlane)
    position.configure(command=redraw)
    widget = canvas.get_tk_widget()
    widget.bind("<MouseWheel>", lambda event: scroll(1 if event.delta > 0 else -1))
    #X11 reports the scroll wheel as buttons 4 and 5
    widget.bind("<Button-4>", lambda event: scroll(1))
    widget.bind("<Button-5>", lambda event: scroll(-1))
    changePlane()
