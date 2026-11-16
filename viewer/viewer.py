#!/usr/bin/env python3
#Viewer for SAMS output files in the HDF5, ADIOS2 and simple formats
#Shows a list of the variables in a file. Clicking on a variable opens a window showing it
#Usage: ./viewer.py {file.h5, file.bp or directory}
import sys
import tkinter as tk
from tkinter import messagebox

import plotWindow

if len(sys.argv) != 2:
    print("Usage: ./viewer.py {file.h5, file.bp or directory}")
    sys.exit(1)
path = sys.argv[1]

root = tk.Tk()
root.withdraw()

#Tk is slow to show its first image on a remote display, so show a splash while it shows a tiny one
#The splash is hidden rather than destroyed, so that the image stays alive and the plots do not pay that cost again
splash = tk.Toplevel(root)
splash.withdraw()
splash.title("SAMS viewer")
tk.Label(splash, text="Loading " + sys.argv[1] if len(sys.argv) > 1 else "Starting SAMS viewer", padx=20, pady=20).pack()
#Centre the splash on the screen
splash.update_idletasks()
splashX = (splash.winfo_screenwidth() - splash.winfo_reqwidth()) // 2
splashY = (splash.winfo_screenheight() - splash.winfo_reqheight()) // 2
splash.geometry("+" + str(splashX) + "+" + str(splashY))
splash.deiconify()
#Make sure that the splash is on the screen before the slow image is made
splash.wait_visibility()
splash.update()
warmImage = tk.PhotoImage(width=1, height=1)
warmCanvas = tk.Canvas(splash, width=1, height=1, highlightthickness=0)
warmCanvas.pack()
warmCanvas.create_image(0, 0, image=warmImage, anchor=tk.NW)
#The image is only drawn once the canvas is on the screen, so wait for that before drawing it
warmCanvas.wait_visibility()
splash.update()
splash.withdraw()

#The format is chosen from the path. Only the module of that format is imported, so only its library is needed
if path.endswith(".h5"):
    import hdf5File as fileFormat
    valid = fileFormat.isHDF5File(path)
elif path.rstrip("/").endswith(".bp"):
    import adios2File as fileFormat
    valid = fileFormat.isADIOS2File(path)
else:
    import simpleFile as fileFormat
    valid = fileFormat.isSimpleFile(path)
if not valid:
    messagebox.showerror("Cannot open file", path + " is not a SAMS output file")
    sys.exit(1)
dump = fileFormat.openFile(path)
variables = fileFormat.listVariables(dump)
shapes = fileFormat.readShapes(dump, variables)

root.title(path)
listbox = tk.Listbox(root, width=40, height=max(10, min(len(variables), 40)))
for name in variables:
    listbox.insert(tk.END, name)
listbox.pack(fill=tk.BOTH, expand=True)


def openVariable(viewDims):
    #Open a window for the variable that is selected in the list, with a view that has viewDims axes
    selection = listbox.curselection()
    if not selection:
        return
    chosen = selection[0]
    try:
        plotWindow.openPlotWindow(root, fileFormat, dump, variables[chosen], step.get(), viewDims, clipGhosts.get())
    except Exception as error:
        messagebox.showerror("Cannot show " + variables[chosen], str(error))


def updateButtons(event):
    #A view can have no more axes than the variable has to plot. A variable with no axes to plot is shown by the 1D view
    #The 3D view is only available when pyvista is
    selection = listbox.curselection()
    plottedAxes = plotWindow.countPlottedAxes(shapes[variables[selection[0]]]) if selection else -1
    visualise1D.configure(state=tk.NORMAL if plottedAxes >= 0 else tk.DISABLED)
    visualise2D.configure(state=tk.NORMAL if plottedAxes >= 2 else tk.DISABLED)
    visualise3D.configure(state=tk.NORMAL if plottedAxes == 3 and plotWindow.has3D else tk.DISABLED)


visualise3D = tk.Button(root, text="Visualise (3D)", command=lambda: openVariable(3), state=tk.DISABLED)
visualise3D.pack(side=tk.BOTTOM, fill=tk.X)
visualise2D = tk.Button(root, text="Visualise (2D)", command=lambda: openVariable(2), state=tk.DISABLED)
visualise2D.pack(side=tk.BOTTOM, fill=tk.X)
visualise1D = tk.Button(root, text="Visualise (1D)", command=lambda: openVariable(1), state=tk.DISABLED)
visualise1D.pack(side=tk.BOTTOM, fill=tk.X)
#Clip each variable to the area that its mesh plots, leaving out any ghost cells
clipGhosts = tk.BooleanVar(value=False)
tk.Checkbutton(root, text="Clip ghost cells", variable=clipGhosts).pack(side=tk.BOTTOM, anchor=tk.W)
#The step that new windows show. A window keeps the step that it was opened with. The choice is only shown for a file with more than one step
stepCount = fileFormat.stepCount(dump)
step = tk.IntVar(value=stepCount - 1)
if stepCount > 1:
    stepRow = tk.Frame(root)
    stepRow.pack(side=tk.BOTTOM, fill=tk.X)
    tk.Label(stepRow, text="Step").pack(side=tk.LEFT)
    stepTime = tk.Label(stepRow)

    def showTime():
        stepTime.configure(text=plotWindow.timeText(fileFormat.readTime(dump, step.get())))

    tk.Spinbox(stepRow, from_=0, to=stepCount - 1, textvariable=step, width=6, command=showTime).pack(side=tk.LEFT)
    stepTime.pack(side=tk.LEFT)
    showTime()
listbox.bind("<<ListboxSelect>>", updateButtons)
root.deiconify()
root.mainloop()
dump.close()
