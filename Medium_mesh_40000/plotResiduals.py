import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- Style setup ----------
mpl.rcParams.update({
    "figure.figsize"     : (16, 9),
    "savefig.dpi"        : 300,
    "text.usetex"        : False,
    "font.family"        : "DejaVu Serif",   # exists on Ubuntu
    "mathtext.fontset"   : "stix",
    "font.size"          : 18,
    "axes.titlesize"     : 22,
    "axes.labelsize"     : 20,
    "legend.fontsize"    : 14,
    "xtick.labelsize"    : 14,
    "ytick.labelsize"    : 14,
    "lines.linewidth"    : 2.0,
})

# ---------- Helper: load time + residual from a foamLog file ----------
def load_tr(fname):
    try:
        data = np.loadtxt(fname)
        t = data[:, 0]
        r = data[:, 1]
        return t, r
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
}

# ---------- Plot ----------
plt.figure()

for label, fname in files.items():
    t, r = load_tr(fname)
    if t is not None:
        plt.semilogy(t, r, label=label)

plt.grid(True, which="both", linestyle="--", linewidth=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Residual")
plt.title("OpenFOAM Residual Convergence")
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()

plt.savefig("residuals.png", bbox_inches="tight")
plt.savefig("residuals.pdf", bbox_inches="tight")

print("\nSaved:")
print("  → residuals.png  (bitmap)")
print("  → residuals.pdf  (vector, high quality)\n")

plt.show()
