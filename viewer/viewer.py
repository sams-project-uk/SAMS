#!/usr/bin/env python3
#Viewer for SAMS HDF5 output files
#Shows a list of the variables in a file. Clicking on a variable opens a window showing it
#Usage: ./viewer.py [file.h5]
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import hdf5File
import plotWindow


root = tk.Tk()
root.withdraw()

#Tk is slow to show its first image on a remote display, so show a splash while it shows a tiny one
#The splash is hidden rather than destroyed, so that the image stays alive and the plots do not pay that cost again
splash = tk.Toplevel(root)
splash.withdraw()
splash.title("SAMS viewer")
tk.Label(splash, text="Starting SAMS viewer", padx=20, pady=20).pack()
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

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = filedialog.askopenfilename(title="Open SAMS HDF5 file", filetypes=[("HDF5 files", "*.h5")])
    if not path:
        sys.exit()
if not hdf5File.isHDF5File(path):
    messagebox.showerror("Cannot open file", path + " is not an HDF5 file")
    sys.exit(1)
h5file = hdf5File.openFile(path)
variables = hdf5File.listVariables(h5file)
shapes = hdf5File.readShapes(h5file, variables)

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
        plotWindow.openPlotWindow(root, h5file, variables[chosen], viewDims)
    except Exception as error:
        messagebox.showerror("Cannot show " + variables[chosen], str(error))


def updateButtons(event):
    #A view can have no more axes than the variable has to plot. A variable with no axes to plot is shown by the 1D view
    selection = listbox.curselection()
    plottedAxes = plotWindow.countPlottedAxes(shapes[variables[selection[0]]]) if selection else -1
    visualise1D.configure(state=tk.NORMAL if plottedAxes >= 0 else tk.DISABLED)
    visualise2D.configure(state=tk.NORMAL if plottedAxes >= 2 else tk.DISABLED)
    visualise3D.configure(state=tk.NORMAL if plottedAxes == 3 else tk.DISABLED)


visualise3D = tk.Button(root, text="Visualise (3D)", command=lambda: openVariable(3), state=tk.DISABLED)
visualise3D.pack(side=tk.BOTTOM, fill=tk.X)
visualise2D = tk.Button(root, text="Visualise (2D)", command=lambda: openVariable(2), state=tk.DISABLED)
visualise2D.pack(side=tk.BOTTOM, fill=tk.X)
visualise1D = tk.Button(root, text="Visualise (1D)", command=lambda: openVariable(1), state=tk.DISABLED)
visualise1D.pack(side=tk.BOTTOM, fill=tk.X)
listbox.bind("<<ListboxSelect>>", updateButtons)
root.deiconify()
root.mainloop()
h5file.close()
