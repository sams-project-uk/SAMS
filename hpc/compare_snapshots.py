#!/usr/bin/env python3
"""Compare the snapshots two SAMS runs have in common (every dataset, every rank file).

    compare_snapshots.py <reference SAMS_testdata> <other SAMS_testdata> [--rtol R] [--skip a,b,...]

Without --rtol the comparison is bit for bit (for restarts and repeated runs on the same machine).
With --rtol it reports datasets whose largest difference exceeds R times the dataset's largest
magnitude (for a different machine or compiler, where round-off differs: use e.g. 1e-8).
--skip leaves out datasets whose name contains any of the given strings.
Both runs must use the same number of ranks (the rank files are compared pairwise)."""
import argparse, glob, os, sys
import h5py, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ref"); ap.add_argument("other")
ap.add_argument("--rtol", type=float, default=None)
ap.add_argument("--skip", default="")
a = ap.parse_args()
skip = [s for s in a.skip.split(",") if s]

common = sorted(set(map(os.path.basename, glob.glob(a.ref + "/*.h5"))) & set(map(os.path.basename, glob.glob(a.other + "/*.h5"))))
if not common: sys.exit("no snapshot files in common")
nsets = 0; bad = 0; worst = 0.0
for name in common:
    fa, fb = h5py.File(os.path.join(a.ref, name), "r"), h5py.File(os.path.join(a.other, name), "r")
    keys = []; fa.visit(lambda k: keys.append(k) if isinstance(fa[k], h5py.Dataset) else None)
    for k in keys:
        if any(s in k for s in skip): continue
        if k not in fb: print(f"MISSING {name} {k}"); bad += 1; continue
        x, y = np.asarray(fa[k]), np.asarray(fb[k]); nsets += 1
        if x.shape != y.shape: print(f"SHAPE {name} {k}: {x.shape} vs {y.shape}"); bad += 1; continue
        if x.dtype.kind not in "fiu":
            if not np.array_equal(x, y): print(f"DIFF {name} {k}"); bad += 1
            continue
        if np.array_equal(x, y, equal_nan=True): continue
        d = np.abs(x.astype(float) - y.astype(float))
        rel = np.nanmax(d) / max(np.nanmax(np.abs(x)), 1e-300)
        if np.isnan(d).any() and not np.isnan(rel): rel = np.inf
        worst = max(worst, rel)
        if a.rtol is None or not rel <= a.rtol:
            print(f"DIFF {name} {k}: max |difference| / max |value| = {rel:.3e}"); bad += 1
print(f"{len(common)} files, {nsets} datasets compared: " +
      ("bit-identical" if worst == 0 and bad == 0 else
       f"{bad} outside tolerance" if bad else f"all within rtol {a.rtol:g} (largest relative difference {worst:.2e})"))
sys.exit(1 if bad else 0)
