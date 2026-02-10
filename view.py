#!/usr/bin/env python3
"""
Print structure of an HDF5 file and optionally plot a dataset.

Usage:
    python h5_tree.py path/to/file.h5 [dataset_path]

If a dataset_path is provided, it will be read and plotted as a pseudocolor plot
(for 2D) or a line plot (for 1D) using matplotlib.

Requires: h5py (pip install h5py)
If plotting is used: matplotlib (pip install matplotlib)
"""
import sys
import argparse

try:
    import h5py
except Exception:
    sys.exit("Missing dependency: install with 'pip install h5py'")

import numpy as np

def print_node(name, obj, indent=0, show_attrs=False):
    pad = "  " * indent
    if isinstance(obj, h5py.Group):
        print(f"{pad}{name}/ (Group)  attrs={len(obj.attrs)}")
        if show_attrs and len(obj.attrs):
            for k, v in obj.attrs.items():
                print(f"{pad}  @ {k} = {v}")
        for key in obj:
            print_node(key, obj[key], indent + 1, show_attrs)
    else:  # Dataset
        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", None)
        print(f"{pad}{name} (Dataset) shape={shape} dtype={dtype} attrs={len(obj.attrs)}")
        if show_attrs and len(obj.attrs):
            for k, v in obj.attrs.items():
                print(f"{pad}  @ {k} = {v}")

def main():
    parser = argparse.ArgumentParser(description="Print HDF5 file structure and optionally plot a dataset")
    parser.add_argument("file", help="Path to HDF5 file")
    parser.add_argument("dataset", nargs="?", help="Optional: dataset path inside the HDF5 file to plot")
    parser.add_argument("--attrs", action="store_true", help="Show attributes for each node")
    args = parser.parse_args()


    try:
        with h5py.File(args.file, "r") as f:
            print(f"/ (root)  attrs={len(f.attrs)}")
            if args.attrs and len(f.attrs):
                for k, v in f.attrs.items():
                    print(f"  @ {k} = {v}")
            for key in f:
                print_node(key, f[key], 1, args.attrs)

            if args.dataset:
                # Try to open the requested dataset
                try:
                    obj = f[args.dataset]
                except KeyError:
                    sys.exit(f"Dataset not found: {args.dataset}")
                if not isinstance(obj, h5py.Dataset):
                    sys.exit(f"Not a dataset: {args.dataset}")

                # Import matplotlib only when plotting is requested
                try:
                    import matplotlib
                    import matplotlib.pyplot as plt
                except Exception:
                    sys.exit("Missing dependency for plotting: install with 'pip install matplotlib'")

                matplotlib.use('TkAgg')

                # Read data into memory
                try:
                    data = obj[()]
                except Exception as e:
                    sys.exit(f"Error reading dataset: {e}")

                #Print the range of data values
                if np.issubdtype(data.dtype, np.number):
                    print(f"Data range: min={np.min(data)}, max={np.max(data)}")
                else:
                    print("Data is not numeric; skipping range calculation.")

                # Handle dimensionality
                if getattr(data, "ndim", None) is None:
                    sys.exit("Dataset has no ndim (not an array-like).")

                if data.ndim == 1:
                    plt.figure()
                    plt.plot(data)
                    plt.title(args.dataset)
                    plt.xlabel("index")
                    plt.ylabel(str(obj.dtype))
                    plt.grid(True)
                elif data.ndim == 2:
                    plt.figure()
                    plt.pcolormesh(data, cmap="viridis", shading="auto")
                    plt.colorbar()
                    plt.title(args.dataset)
                else:
                    # For >2D, attempt to plot the central 2D slice along the first axis
                    try:
                        if len(data.shape) == 0 or data.shape[0] == 0:
                            sys.exit(f"Cannot take a central slice from dataset with shape {data.shape}")
                        center_idx = data.shape[0] // 2
                        slice2d = data[center_idx]
                        if data.ndim == 3:
                            # Plot central slice along each axis (axis0, axis1, axis2)
                            centers = [s // 2 for s in data.shape]
                            c0, c1, c2 = centers
                            s_ax0 = data[c0, :, :]
                            s_ax1 = data[:, c1, :]
                            s_ax2 = data[:, :, c2]
                            if not (s_ax0.ndim == 2 and s_ax1.ndim == 2 and s_ax2.ndim == 2):
                                sys.exit(f"Unexpected slice ndim for 3D dataset: {[s_ax0.ndim, s_ax1.ndim, s_ax2.ndim]}")
                            print(f"Dataset has 3 dimensions; plotting central slices (axis0 index {c0}, axis1 index {c1}, axis2 index {c2}) and a line along X at y={c1}, z={c2}.")
                            plt.figure(figsize=(16, 4))
                            # Prepare 3 2D slices + 1D line along X at (y/2, z/2)
                            slices = [
                                (s_ax0, f"{args.dataset} [axis0={c0}]"),
                                (s_ax1, f"{args.dataset} [axis1={c1}]"),
                                (s_ax2, f"{args.dataset} [axis2={c2}]"),
                            ]
                            for i, (slc, title) in enumerate(slices, start=1):
                                ax = plt.subplot(1, 4, i)
                                pcm = ax.pcolormesh(slc, cmap="viridis", shading="auto")
                                plt.colorbar(pcm, ax=ax)
                                ax.set_title(title)
                                if (i==1):
                                    ax.set_ylabel("Y Index")
                                    ax.set_xlabel("Z Index")
                                elif (i==2):
                                    ax.set_ylabel("X Index")
                                    ax.set_xlabel("Z Index")
                                else:
                                    ax.set_ylabel("X Index")
                                    ax.set_xlabel("Y Index")
                            # Fourth plot: 1D line along X at y=c1, z=c2
                            ax4 = plt.subplot(1, 4, 4)
                            line = data[:, c1, c2]
                            ax4.plot(line)
                            ax4.set_title(f"{args.dataset} [X at y={c1}, z={c2}]")
                            ax4.set_xlabel("X Index")
                            ax4.set_ylabel(str(obj.dtype))
                            ax4.grid(True)
                        else:
                            if slice2d.ndim == 2:
                                print(f"Dataset has {data.ndim} dimensions; plotting central slice (index {center_idx}) as 2D.")
                                plt.figure()
                                plt.pcolormesh(slice2d, cmap="viridis", shading="auto")
                                plt.colorbar()
                                plt.title(f"{args.dataset} [{center_idx}]")
                            else:
                                sys.exit(f"Cannot plot dataset with ndim={data.ndim}")
                    except Exception as e:
                        sys.exit(f"Cannot prepare 2D slice for plotting: {e}")

                plt.tight_layout()
                plt.show()

    except (OSError, IOError) as e:
        sys.exit(f"Error opening file: {e}")

if __name__ == "__main__":
    main()
