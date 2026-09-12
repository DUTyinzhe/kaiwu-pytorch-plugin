"""
Generate fig9_scaling_ratio.png: RMSE(LS)/RMSE(SAC) vs kappa(A)
Clean layout — no overlapping annotations.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

OUTPUT_DIR = Path(r"D:\QPDE\sn-article-template\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(r"D:\QPDE\模拟退火\outputs\exp4_scaling_law\scaling_law_results.json") as f:
    data = json.load(f)

details = data["details"]
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

# ------- Plot -------
plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(7.2, 4.8))

# Regime background bands (very subtle)
ax.axvspan(1.5, 8.5, alpha=0.06, color="green", zorder=0)
ax.axvspan(8.5, 21, alpha=0.06, color="orange", zorder=0)
ax.axvspan(21, 72, alpha=0.06, color="red", zorder=0)

# Regime labels: placed in axes-fraction space below data area
ax.text(0.18, 0.015, "I   Scaling law", ha="center", va="bottom", fontsize=8,
        fontstyle="italic", color="#2E7D32", transform=ax.transAxes)
ax.text(0.47, 0.015, "II   Transition", ha="center", va="bottom", fontsize=8,
        fontstyle="italic", color="#E65100", transform=ax.transAxes)
ax.text(0.78, 0.015, "III   Quantization wall", ha="center", va="bottom", fontsize=8,
        fontstyle="italic", color="#B71C1C", transform=ax.transAxes)

# Theoretical line
kappa_range = np.array(kappas)
ax.plot(kappa_range, kappa_range, "k--", linewidth=1.2, alpha=0.5,
        label=r"Theory: ratio $=\kappa$")

# Data points
for k in kappas:
    x_jitter = k * np.array([0.93, 1.0, 1.07])

    # float64
    r_f = ratios_float64[k]
    ax.scatter(x_jitter, r_f, c="#2196F3", s=32, zorder=5,
               edgecolors="white", linewidth=0.4)
    ax.scatter([k], [np.mean(r_f)], c="#1565C0", s=75, marker="D", zorder=6,
               edgecolors="white", linewidth=0.6)

    # int8 — clip extreme values for display
    r_i = [min(x, 80) for x in ratios_int8[k]]
    ax.scatter(x_jitter, r_i, c="#FF5722", s=32, zorder=5,
               edgecolors="white", linewidth=0.4)
    ax.scatter([k], [np.mean(r_i)], c="#BF360C", s=75, marker="s", zorder=6,
               edgecolors="white", linewidth=0.6)

# Legend
legend_elements = [
    Line2D([0], [0], color="k", linestyle="--", linewidth=1.2, alpha=0.5,
           label=r"Theory: ratio $=\kappa$"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
           markersize=5.5, label="float64"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#1565C0",
           markersize=7, label="float64 mean"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722",
           markersize=5.5, label="int8"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#BF360C",
           markersize=7, label="int8 mean"),
]
leg = ax.legend(handles=legend_elements, loc="upper left", framealpha=0.85,
                ncol=1, borderpad=0.4, labelspacing=0.25, handletextpad=0.4,
                borderaxespad=0.5)

# Axes
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel(r"Condition number $\kappa(A)$")
ax.set_ylabel(r"RMSE(LS) / RMSE(SAC)")
ax.set_title("Precision scaling law verification (exhaustive enumeration, n=3)")

ax.set_xticks(kappas)
ax.set_xticklabels([str(k) for k in kappas])
ax.set_xlim(1.5, 70)
ax.set_ylim(0.35, 90)

ax.grid(True, alpha=0.25, which="major", linestyle="-")
ax.grid(True, alpha=0.10, which="minor", linestyle=":")

plt.tight_layout(pad=0.5)
outpath = OUTPUT_DIR / "fig9_scaling_ratio.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {outpath}")
