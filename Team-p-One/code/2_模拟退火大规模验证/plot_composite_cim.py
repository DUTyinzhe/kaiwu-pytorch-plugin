"""
Generate fig_cim_composite.png: 4-panel CIM hardware results
(a) SPD, (b) Indefinite, (c) Negative definite, (d) Condition number scaling
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

FIG_DIR = Path(r"D:\QPDE\sn-article-template\figures")
OUTPUT = FIG_DIR / "fig_cim_composite.png"

images = {
    "a": mpimg.imread(FIG_DIR / "fig1_pde_single_rmse.png"),
    "b": mpimg.imread(FIG_DIR / "fig2_indef_rmse.png"),
    "c": mpimg.imread(FIG_DIR / "fig3_negdef_rmse.png"),
    "d": mpimg.imread(FIG_DIR / "fig4_cond_scaling.png"),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, (label, img) in zip(axes, images.items()):
    ax.imshow(img)
    ax.axis("off")
    ax.text(-0.01, 1.01, f"({label})", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")

plt.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.1)
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white",
            edgecolor="none")
plt.close(fig)
print(f"Saved: {OUTPUT}")
