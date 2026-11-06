#Line plot of a variable with one axis to plot, or of a line through a variable with two or three axes to plot
import tkinter as tk
from tkinter import ttk

import plotWindow


def plotLine(figure, varName, data, axes, title=None):
    #A variable with one axis to plot is shown as a line
    figure.clear()
    plot = figure.add_subplot(111)
    name, coords = axes[0]
    plot.plot(plotWindow.linePositions(coords, data.shape[0]), data)
    plot.set_xlabel(name)
    plot.set_ylabel(varName)
    plot.set_title(title if title else varName)


def addLineControls(window, figure, canvas, varName, data, axes):
    #A variable with two or three axes to plot is shown as a line along one axis. The axis of the line is chosen from a
    #list, and the cell of the line along each other axis is chosen with a slider. Scrolling over the plot moves the
    #first slider
    controls = ttk.Frame(window)
    controls.pack(side=tk.BOTTOM, fill=tk.X)
    choiceRow = ttk.Frame(controls)
    choiceRow.pack(side=tk.TOP, fill=tk.X)
    ttk.Label(choiceRow, text="Line along").pack(side=tk.LEFT)
    axisChoice = ttk.Combobox(choiceRow, values=[name for name, coords in axes], state="readonly", width=8)
    axisChoice.current(0)
    axisChoice.pack(side=tk.LEFT)

    #One slider for each axis that the line is not along. They are relabelled when the axis of the line changes
    sliders = []
    for slider in range(data.ndim - 1):
        row = ttk.Frame(controls)
        row.pack(side=tk.TOP, fill=tk.X)
        label = ttk.Label(row, width=8)
        label.pack(side=tk.LEFT)
        position = tk.Scale(row, orient=tk.HORIZONTAL, **{"from": 0, "to": 0})
        position.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sliders.append((label, position))

    def otherAxes():
        #The axes that the line is not along
        along = axisChoice.current()
        return [dim for dim in range(data.ndim) if dim != along]

    def redraw(*args):
        index = [slice(None)] * data.ndim
        where = []
        for dim, (label, position) in zip(otherAxes(), sliders):
            cells = data.shape[dim]
            cell = min(int(position.get()), cells - 1)
            index[dim] = cell
            name, coords = axes[dim]
            where.append(name + " = " + format(plotWindow.linePositions(coords, cells)[cell], "g"))
        plotLine(figure, varName, data[tuple(index)], [axes[axisChoice.current()]], varName + " at " + ", ".join(where))
        canvas.draw_idle()

    def changeAxis(*args):
        #A new line starts through the middle of each other axis
        for dim, (label, position) in zip(otherAxes(), sliders):
            label.configure(text=axes[dim][0])
            position.configure(to=data.shape[dim] - 1)
            position.set(data.shape[dim] // 2)
        redraw()

    def scroll(step):
        #The slider keeps the cell inside the axis
        label, position = sliders[0]
        position.set(int(position.get()) + step)

    axisChoice.bind("<<ComboboxSelected>>", changeAxis)
    for label, position in sliders:
        position.configure(command=redraw)
    widget = canvas.get_tk_widget()
    widget.bind("<MouseWheel>", lambda event: scroll(1 if event.delta > 0 else -1))
    #X11 reports the scroll wheel as buttons 4 and 5
    widget.bind("<Button-4>", lambda event: scroll(1))
    widget.bind("<Button-5>", lambda event: scroll(-1))
    changeAxis()
