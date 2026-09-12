"""
Generate fig_scaling_composite.png: 2-panel scaling law
(a) Bit-depth scaling: b_eff vs b
(b) RMSE ratio vs kappa (reuse fig9 data)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from pathlib import Path

FIG_DIR = Path(r"D:\QPDE\sn-article-template\figures")
OUTPUT = FIG_DIR / "fig_scaling_composite.png"
DATA_EXP2 = Path(r"D:\QPDE\模拟退火\outputs\exp2_bitdepth.json")

plt.rcParams.update({
    "font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax1, ax2 = axes

# ===================== Panel (a): Bit-depth scaling =====================
with open(DATA_EXP2) as f:
    data2 = json.load(f)

bits_list = sorted(set(r["bits"] for r in data2))
sac_beff = {}
ls_beff = {}
for r in data2:
    if r["p"] == 1:
        sac_beff[r["bits"]] = r["b_eff"]
    else:
        ls_beff[r["bits"]] = r["b_eff"]

b_vals = np.array(bits_list)
sac_vals = np.array([sac_beff[b] for b in bits_list])
ls_vals = np.array([ls_beff[b] for b in bits_list])
delta_vals = sac_vals - ls_vals

ax1.plot(b_vals, sac_vals, "o-", c="#2196F3", linewidth=1.8, markersize=8,
         label="SAC (p=1)")
ax1.plot(b_vals, ls_vals, "s--", c="#FF5722", linewidth=1.8, markersize=8,
         label="LS (p=2)")
ax1.plot(b_vals, b_vals, "k:", linewidth=1, alpha=0.4, label="Ideal $b_{\\rm eff}=b$")

for i, b in enumerate(b_vals):
    ax1.annotate(f"{delta_vals[i]:.2f}", (b, (sac_vals[i] + ls_vals[i]) / 2),
                 textcoords="offset points", xytext=(18, 0), fontsize=8.5,
                 color="#555555", ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.6))

ax1.set_xlabel("Encoding bit-depth $b$")
ax1.set_ylabel("Effective precision $b_{\\rm eff}$")
ax1.set_title("Bit-depth scaling (float64, $\\kappa$=2.09)")
ax1.legend(loc="upper left", framealpha=0.8)
ax1.set_xticks(b_vals)
ax1.grid(True, alpha=0.25)
ax1.set_xlim(3.5, 7.5)
ax1.set_ylim(2.8, 6.8)

# ===================== Panel (b): RMSE ratio vs kappa =====================
# Load existing fig9 data
with open(r"D:\QPDE\模拟退火\outputs\exp4_scaling_law\scaling_law_results.json") as f:
    data4 = json.load(f)

details = data4["details"]
kappas = [2, 4, 8, 16, 32, 64]
bit_depths = [4, 5, 6]

def get_ratios(quantize_mode):
    result = {}
    for k in kappas:
        ratios = []
        for b in bit_depths:
            sac = next(r for r in details
                       if r["kappa"] == k and r["p"] == 1
                       and r["bits"] == b and r["quantize"] == quantize_mode)
            ls = next(r for r in details
                      if r["kappa"] == k and r["p"] == 2
                      and r["bits"] == b and r["quantize"] == quantize_mode)
            ratio = ls["rmse"] / sac["rmse"] if sac["rmse"] > 1e-16 else float("inf")
            ratios.append(ratio)
        result[k] = ratios
    return result

ratios_float64 = get_ratios("float64")
ratios_int8 = get_ratios("int8")

# Regime bands
ax2.axvspan(1.5, 8.5, alpha=0.06, color="green", zorder=0)
ax2.axvspan(8.5, 21, alpha=0.06, color="orange", zorder=0)
ax2.axvspan(21, 72, alpha=0.06, color="red", zorder=0)

ax2.text(0.18, 0.015, "I   Scaling law", ha="center", va="bottom", fontsize=8,
         fontstyle="italic", color="#2E7D32", transform=ax2.transAxes)
ax2.text(0.47, 0.015, "II   Transition", ha="center", va="bottom", fontsize=8,
         fontstyle="italic", color="#E65100", transform=ax2.transAxes)
ax2.text(0.78, 0.015, "III   Quantization wall", ha="center", va="bottom", fontsize=8,
         fontstyle="italic", color="#B71C1C", transform=ax2.transAxes)

# Theory line
kappa_range = np.array(kappas)
ax2.plot(kappa_range, kappa_range, "k--", linewidth=1.2, alpha=0.5,
         label=r"Theory: ratio $=\kappa$")

for k in kappas:
    x_jitter = k * np.array([0.93, 1.0, 1.07])
    r_f = ratios_float64[k]
    ax2.scatter(x_jitter, r_f, c="#2196F3", s=28, zorder=5,
                edgecolors="white", linewidth=0.4)
    ax2.scatter([k], [np.mean(r_f)], c="#1565C0", s=65, marker="D", zorder=6,
                edgecolors="white", linewidth=0.6)
    r_i = [min(x, 80) for x in ratios_int8[k]]
    ax2.scatter(x_jitter, r_i, c="#FF5722", s=28, zorder=5,
                edgecolors="white", linewidth=0.4)
    ax2.scatter([k], [np.mean(r_i)], c="#BF360C", s=65, marker="s", zorder=6,
                edgecolors="white", linewidth=0.6)

# Legend
legend_elements = [
    Line2D([0], [0], color="k", linestyle="--", linewidth=1.2, alpha=0.5,
           label=r"Theory: ratio $=\kappa$"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
           markersize=5, label="float64"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#1565C0",
           markersize=6.5, label="float64 mean"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722",
           markersize=5, label="int8"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#BF360C",
           markersize=6.5, label="int8 mean"),
]
ax2.legend(handles=legend_elements, loc="upper left", framealpha=0.85,
           ncol=1, borderpad=0.4, labelspacing=0.25, handletextpad=0.4,
           borderaxespad=0.5)

ax2.set_xscale("log", base=2)
ax2.set_yscale("log", base=2)
ax2.set_xlabel(r"Condition number $\kappa(A)$")
ax2.set_ylabel(r"RMSE(LS) / RMSE(SAC)")
ax2.set_title("Condition-number scaling (int8 quant., n=3)")
ax2.set_xticks(kappas)
ax2.set_xticklabels([str(k) for k in kappas])
ax2.set_xlim(1.5, 70)
ax2.set_ylim(0.35, 90)
ax2.grid(True, alpha=0.25, which="major", linestyle="-")
ax2.grid(True, alpha=0.10, which="minor", linestyle=":")

# Panel labels
ax1.text(-0.06, 1.02, "(a)", transform=ax1.transAxes, fontsize=13,
         fontweight="bold", va="bottom", ha="left")
ax2.text(-0.06, 1.02, "(b)", transform=ax2.transAxes, fontsize=13,
         fontweight="bold", va="bottom", ha="left")

plt.tight_layout(pad=0.8, w_pad=1.5)
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUTPUT}")
