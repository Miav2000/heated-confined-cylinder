#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -------------------------------------------------
# Matplotlib style (similar to your residual script)
# -------------------------------------------------
mpl.rcParams.update({
    "figure.figsize"     : (12, 7),
    "savefig.dpi"        : 300,
    "text.usetex"        : False,
    "font.family"        : "DejaVu Serif",
    "mathtext.fontset"   : "stix",
    "font.size"          : 16,
    "axes.titlesize"     : 18,
    "axes.labelsize"     : 16,
    "legend.fontsize"    : 12,
    "xtick.labelsize"    : 12,
    "ytick.labelsize"    : 12,
    "lines.linewidth"    : 2.0,
})

# -------------------------------------------------
# USER SETTINGS – EDIT THESE
# -------------------------------------------------
case_dir    = "."   # run the script from the case root
post_dir    = os.path.join(case_dir, "postProcessing")

# Nu history file (time vs Nu) produced by surfaceFieldValue
nu_file     = os.path.join(post_dir, "NuAvgCylinder", "0", "surfaceFieldValue.dat")

# inletMeanUT file (area-averaged U and T)
inletMean_file = os.path.join(post_dir, "inletMeanUT", "0", "surfaceFieldValue.dat")

# time after which flow is ready for Nu averaging
t_start_avg = 0.25   # <-- change once you know when transients end (look at Nu plot)

# fluid / geometry for Re
rho = (998+997)/2                 # [kg/m3] at film temperature 22.5 C
mu  = (1.002e-3+0.891e-3)/2       # [Pa s] at film temperature 22.5 C
D   = 0.01         # [m] cylinder diameter
D_channel = 0.046  # [m] channel height (plate spacing)

# positions along centerline where we want values, measured from inlet [mm]
centerline_x_rel_mm = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]

# inlet centre is at x = -0.15 m, centerline goes to x = 0.0 m
x_inlet_center = -0.15  # [m]

# position for Re_cyl: 100 mm from inlet -> x = -0.05 m
x_rel_for_Re_cyl_mm = 100.0
x_abs_for_Re_cyl = x_inlet_center + x_rel_for_Re_cyl_mm/1000.0  # = -0.05 m


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def latest_time_dir(base):
    """Return the path and value of the latest time directory under base."""
    times = []
    for name in os.listdir(base):
        try:
            t = float(name)
            times.append((t, name))
        except ValueError:
            continue
    if not times:
        raise RuntimeError(f"No numerical time directories in {base}")
    times.sort()
    t_latest, dirname = times[-1]
    return os.path.join(base, dirname), t_latest


def load_xy(fname):
    """Load a standard OpenFOAM sample .xy file (two columns)."""
    return np.loadtxt(fname, comments="#")


def bulk_velocity_from_profile(y, u):
    """
    Compute bulk velocity for a 2D channel from a vertical profile (y, u(y)).
    Assumes uniform depth in z and that y spans the full height.
    """
    H = y.max() - y.min()
    return np.trapz(u, y) / H

def read_cell_count(log_path):
    """
    Read number of cells from an OpenFOAM log file that contains a line like:
        cells:     40000
    """
    if not os.path.isfile(log_path):
        print(f"[Warning] Log file not found for cell count: {log_path}")
        return None

    with open(log_path, "r") as f:
        for line in f:
            if "cells:" in line:
                parts = line.split()
                # take last integer on the line
                for p in reversed(parts):
                    if p.isdigit():
                        return int(p)
    print(f"[Warning] Could not find 'cells:' line in {log_path}")
    return None



# -------------------------------------------------
# 0) Read inletMeanUT: U stored as a vector "(Ux Uy Uz)"
# -------------------------------------------------
def read_inletMeanUT(fname):
    """
    Read OpenFOAM surfaceFieldValue.dat from inletMeanUT and return
    latest U_bulk (Ux component) and corresponding Re_bulk based on D_channel.
    """
    if not os.path.isfile(fname):
        print(f"\n[Warning] inletMeanUT file not found: {fname}")
        return None, None

    times = []
    Ux_vals = []

    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # time is first entry
            try:
                t = float(parts[0])
            except ValueError:
                continue  # skip weird lines

            # the rest of the line is the vector "(Ux Uy Uz)" possibly split
            vec_str = " ".join(parts[1:])
            vec_str = vec_str.strip("()")  # remove parentheses
            comps = vec_str.split()
            try:
                ux = float(comps[0])
            except (IndexError, ValueError):
                continue

            times.append(t)
            Ux_vals.append(ux)

    if not Ux_vals:
        print("\n[Warning] No usable U data found in inletMeanUT file.")
        return None, None

    U_last = Ux_vals[-1]
    Re_bulk = rho * U_last * D_channel / mu

    print("\nInlet bulk velocity from inletMeanUT:")
    print(f"  U_bulk = {U_last:.6g} m/s")
    print(f"  Re_bulk = {Re_bulk:.6g}")

    return U_last, Re_bulk


# -------------------------------------------------
# 1) Time-averaged Nusselt number from NuAvgCylinder/0
# -------------------------------------------------
def compute_time_averaged_Nu(nu_file, t_start):
    data = np.loadtxt(nu_file, comments="#")
    t  = data[:, 0]
    Nu = data[:, 1]

    mask = t >= t_start
    if mask.sum() == 0:
        raise RuntimeError("No Nu data after t_start_avg; reduce t_start_avg.")

    Nu_sel = Nu[mask]

    Nu_mean = Nu_sel.mean()
    Nu_std  = Nu_sel.std()

    # Plot history
    plt.figure()
    plt.plot(t, Nu, "-", label=r"$Nu_\mathrm{avg}$")
    plt.axvline(t_start, linestyle="--", label=r"$t_\mathrm{start}$ for averaging")
    plt.grid(True, linestyle="--", linewidth=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$Nu_{\mathrm{avg,cylinder}}$ [-]")
    plt.title("Time history of surface-averaged Nusselt number")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Nu_time_history.png", bbox_inches="tight")
    plt.savefig("Nu_time_history.pdf", bbox_inches="tight")

    print(f"\nTime-averaged Nusselt number (t >= {t_start} s):")
    print(f"  Nu_mean = {Nu_mean:.6g}")
    print(f"  Nu_std  = {Nu_std:.6g} (std dev over averaging window)")

    return Nu_mean, Nu_std


# -------------------------------------------------
# 2) Profiles at latest time: inlet / internal locations
#    Centerline *_Ux* profiles:
#      - excluded from plots
#      - used only for printing values and Re_cyl
# -------------------------------------------------
def plot_profiles_and_centerline(profiles_root):
    latest_dir, t_latest = latest_time_dir(profiles_root)
    print(f"\nUsing profile data from latest time t = {t_latest} s")
    print(f"  {latest_dir}")

    # Collect profile files
    U_files = []
    T_files = []
    for name in sorted(os.listdir(latest_dir)):
        if name.endswith(".xy"):
            if "_U" in name:
                U_files.append(name)
            elif "_T" in name:
                T_files.append(name)

    if not U_files and not T_files:
        print("[Warning] No *.xy profile files found in latest time directory.")
        return None, None

    # --- Plot all velocity profiles (excluding centerline) ---
    if U_files:
        plt.figure()
        for fname in U_files:
            # skip centerline profiles in the plot
            if "centerline" in fname.lower():
                continue
            data = load_xy(os.path.join(latest_dir, fname))
            coord = data[:, 0]   # coordinate (y or x)
            u     = data[:, 1]
            label = fname.replace(".xy", "")
           
            plt.plot(u, coord, label=label)
        plt.grid(True, linestyle="--", linewidth=0.7)
        plt.ylabel("y [m]")
        plt.xlabel(r"$U$ [m/s]")
        plt.title("Velocity profiles (latest time)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("profiles_U_latest.png", bbox_inches="tight")
        plt.savefig("profiles_U_latest.pdf", bbox_inches="tight")

    # --- Plot all temperature profiles (excluding centerline) ---
    if T_files:
        plt.figure()
        for fname in T_files:
            if "centerline" in fname.lower():
                continue
            data = load_xy(os.path.join(latest_dir, fname))
            coord = data[:, 0]
            T     = data[:, 1]
            label = fname.replace(".xy", "")
            
            plt.plot(T, coord, label=label)
        plt.grid(True, linestyle="--", linewidth=0.7)
        plt.xlabel(r"$T$ [K]")
        plt.ylabel("y [m]")
        plt.title("Temperature profiles (latest time)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("profiles_T_latest.png", bbox_inches="tight")
        plt.savefig("profiles_T_latest.pdf", bbox_inches="tight")

    # --- Centerline values at various x positions (no plot, just numbers) ---
    center_U = [f for f in U_files if "centerline" in f.lower()]
    center_T = [f for f in T_files if "centerline" in f.lower()]

    if not center_U:
        print("\n[Info] No 'centerline' profile files found – cannot report centerline values.")
        return None, None

    fnameU = center_U[0]
    dataU  = load_xy(os.path.join(latest_dir, fnameU))
    x_center = dataU[:, 0]   # x from inlet to cylinder
    U_centerline = dataU[:, 1]

    # Optional temperature along same centerline
    T_centerline = None
    if center_T:
        fnameT = center_T[0]
        dataT  = load_xy(os.path.join(latest_dir, fnameT))
        x_center_T = dataT[:, 0]
        T_centerline = dataT[:, 1]
        # assuming grids match; if not, could interpolate

    # desired positions along centerline, as absolute x in domain
    x_targets = x_inlet_center + np.array(centerline_x_rel_mm) / 1000.0

    print("\nCenterline values U(x), T(x) at selected positions (from inlet):")
    print("  (x_rel = distance from inlet centre)")
    for x_rel_mm, x_abs in zip(centerline_x_rel_mm, x_targets):
        if (x_abs < x_center.min()) or (x_abs > x_center.max()):
            print(f"  x_rel = {x_rel_mm:5.1f} mm (x = {x_abs: .4f} m): outside sampled range")
            continue

        # interpolate velocity
        U_val = np.interp(x_abs, x_center, U_centerline)

        if T_centerline is not None:
            T_val = np.interp(x_abs, x_center, T_centerline)
            print(f"  x_rel = {x_rel_mm:5.1f} mm (x = {x_abs: .4f} m):  U = {U_val:.6g} m/s,  T = {T_val:.6g} K")
        else:
            print(f"  x_rel = {x_rel_mm:5.1f} mm (x = {x_abs: .4f} m):  U = {U_val:.6g} m/s")

    # --- Re at cylinder position (x = -0.05 m, i.e. 100 mm from inlet) ---
    if (x_abs_for_Re_cyl >= x_center.min()) and (x_abs_for_Re_cyl <= x_center.max()):
        U_cyl = np.interp(x_abs_for_Re_cyl, x_center, U_centerline)
        Re_cyl = rho * U_cyl * D / mu
        print("\nReynolds number at cylinder (x_rel = 100 mm, x = -0.05 m):")
        print(f"  U(x=-0.05) = {U_cyl:.6g} m/s")
        print(f"  Re_cyl     = {Re_cyl:.6g}")
    else:
        print("\n[Warning] x = -0.05 m is outside centerline sampling range.")
        Re_cyl = None

    return U_centerline, Re_cyl


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    # 1) Time-averaged Nu
    if os.path.isfile(nu_file):
        Nu_mean, Nu_std = compute_time_averaged_Nu(nu_file, t_start_avg)
    else:
        print(f"[Warning] Nu file not found: {nu_file}")
        Nu_mean = Nu_std = None

    # 2) Inlet bulk Re from inletMeanUT
    U_bulk, Re_bulk = read_inletMeanUT(inletMean_file)

    # 3) Profiles & Re_cyl from centerline
    profiles_root = os.path.join(post_dir, "profiles")
    if os.path.isdir(profiles_root):
        _, Re_cyl = plot_profiles_and_centerline(profiles_root)
    else:
        print(f"[Warning] profiles directory not found: {profiles_root}")
        Re_cyl = None

    print("\nSummary:")
    if U_bulk is not None:
        print(f"  U_bulk (inletMeanUT) = {U_bulk:.6g} m/s")
        print(f"  Re_bulk              = {Re_bulk:.6g}")
    if Re_cyl is not None:
        print(f"  Re_cyl (x = -0.05 m) = {Re_cyl:.6g}")
    if Nu_mean is not None:
        print(f"  Nu_mean (t >= {t_start_avg} s) = {Nu_mean:.6g} ± {Nu_std:.6g}")

    print("\nFigures written:")
    print("  Nu_time_history.[png/pdf]")
    print("  profiles_U_latest.[png/pdf]")
    print("  profiles_T_latest.[png/pdf]")
    print("\nDone.\n")

    # 4) Cell count from log file 
    log_path = os.path.join(case_dir, "log.checkMesh")  # or log.simpleFoam etc.
    n_cells = read_cell_count(log_path)

    # 5) Write summary file for this case
    summary_file = os.path.join(case_dir, "Nu_summary.txt")  # not .tex :)
    try:
        with open(summary_file, "w") as f:
            f.write(f"Nu_mean = {Nu_mean:.8f}\n")
            f.write(f"Nu_std  = {Nu_std:.8f}\n")
            if n_cells is not None:
                f.write(f"cells   = {n_cells:d}\n")
            if Re_bulk is not None:
                f.write(f"Re_bulk = {Re_bulk:.8f}\n")
            if Re_cyl is not None:
                f.write(f"Re_cyl  = {Re_cyl:.8f}\n")
        print(f"\nSummary written to: {summary_file}")
    except Exception as e:
        print(f"\n[Warning] Could not write summary file: {e}")

    plt.show()

