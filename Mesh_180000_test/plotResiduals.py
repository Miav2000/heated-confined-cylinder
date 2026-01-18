import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- Style setup ----------
mpl.rcParams.update({
    "figure.figsize"     : (16, 9),
    "savefig.dpi"        : 300,
    "text.usetex"        : False,
    "font.family"        : "DejaVu Serif",
    "mathtext.fontset"   : "stix",
    "font.size"          : 18,
    "axes.titlesize"     : 22,
    "axes.labelsize"     : 20,
    "legend.fontsize"    : 14,
    "xtick.labelsize"    : 14,
    "ytick.labelsize"    : 14,
    "lines.linewidth"    : 2.0,
})

# ---------- Helper: load time + residual ----------
def load_tr(fname):
    try:
        data = np.loadtxt(fname)
        return data[:, 0], data[:, 1]
    except Exception:
        print(f"[Warning] Could not read {fname}")
        return None, None

# ---------- Files to plot ----------
files = {
    r"$U_x$ initial"     : "logs/Ux_0",
    r"$U_x$ final"       : "logs/UxFinalRes_0",
    r"$U_y$ initial"     : "logs/Uy_0",
    r"$U_y$ final"       : "logs/UyFinalRes_0",
    r"$T$ initial"       : "logs/T_0",
    r"$T$ final"         : "logs/TFinalRes_0",
    r"$p_{rgh}$ initial" : "logs/p_rgh_0",
    r"$p_{rgh}$ final"   : "logs/p_rghFinalRes_3",
    r"$k$ initial"       : "logs/k_0",
    r"$k$ final"         : "logs/kFinalRes_0",
    r"$omega$ initial"   : "logs/omega_0",
    r"$omega$ final"     : "logs/omegaFinalRes_0",
}

# ---------- Custom colors ----------
custom_colors = {
    r"$U_x$ initial"     : "#1f77b4",
    r"$U_x$ final"       : "#1f77b4",
    r"$U_y$ initial"     : "#2ca02c",
    r"$U_y$ final"       : "#2ca02c",
    r"$T$ initial"       : "#ff7f0e",
    r"$T$ final"         : "#ff7f0e",
    r"$p_{rgh}$ initial" : "#9467bd",
    r"$p_{rgh}$ final"   : "#9467bd",
    r"$k$ initial"       : "#8c564b",
    r"$k$ final"         : "#8c564b",

    # Strongly contrasting ω colors
    r"$omega$ initial"   : "black",
    r"$omega$ final"     : "darkred",
}

# ---------- Line style rule ----------
def linestyle(label):
    return "-" if "initial" in label else "--"  # dashed for final curves


# ---------- Plot ----------
plt.figure()

for label, fname in files.items():
    t, r = load_tr(fname)
    if t is not None:
        plt.semilogy(
            t, r,
            label=label,
            color=custom_colors.get(label),
            linestyle=linestyle(label)
        )

plt.grid(True, which="both", linestyle="--", linewidth=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Residual")
plt.title("OpenFOAM Residual Convergence")

# Legend outside plot
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()

plt.savefig("residuals.png", bbox_inches="tight")
plt.savefig("residuals.pdf", bbox_inches="tight")

print("\nSaved:")
print("  → residuals.png")
print("  → residuals.pdf\n")

plt.show()
