"""
Generate paper-quality figures for DD-QUBO paper.
Run: python generate_figures.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import json, os

OUTPUT_DIR = 'D:/QPDE/论文/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'font.family': 'serif',
})

# ===== Load data =====
with open('D:/QPDE/experime2/cond_results.json') as f:
    cond_data = json.load(f)
with open('D:/QPDE/experime2/result_2x2.json') as f:
    res_2x2 = json.load(f)
with open('D:/QPDE/experime2/result_4x4.json') as f:
    res_4x4 = json.load(f)
with open('D:/QPDE/pde_single_outputs/result_ls.json') as f:
    res_single = json.load(f)
with open('D:/QPDE/pde_energy_single_outputs/result_energy.json') as f:
    res_energy_single = json.load(f)

# 3×3 RMSE values manually from pde_ls_qubo.ipynb output
rmse_3x3 = [0.7437273, 0.6386099, 0.3928175, 0.5781268, 0.4659734,
            0.5542109, 0.4183916, 0.6127401, 0.5230972]

# ============================================================
# Figure 1: Condition Number Analysis
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: condition number vs interior nodes
global_data = cond_data['global']
sub_data = cond_data['subdomain']
n_internal = [int(g['n_internal']) for g in global_data]
cond_K = [g['cond_K'] for g in global_data]
cond_K2 = [g['cond_K2'] for g in global_data]
cond_K2_sub = sub_data[0]['cond_K2_mean']

ax1.loglog(n_internal, cond_K, 'bo-', lw=2, ms=8, label=r'$\kappa(K_{ii})$ global')
ax1.loglog(n_internal, cond_K2, 'rs-', lw=2, ms=8, label=r'$\kappa(K_{ii}^2)$ global')
ax1.axhline(y=cond_K2_sub, color='green', ls='--', lw=2,
            label=rf'$\kappa(K_{{ii}}^{{2}})$ subdomain ($\approx {cond_K2_sub:.0f}$)')
ax1.axhline(y=2**16, color='red', ls=':', lw=1.5, alpha=0.6,
            label=r'8-bit dynamic range ($2^{16}$)')
ax1.set_xlabel('Number of interior nodes $n_i$', fontsize=13)
ax1.set_ylabel('Condition number $\kappa$', fontsize=13)
ax1.set_title('(a) Condition number growth', fontsize=14)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.annotate(r'$\kappa \propto N^4$', xy=(200, 7000), fontsize=11, color='red',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.annotate(r'$\kappa_{\rm sub} = 44.7$ (constant)', xy=(200, 40), fontsize=11,
             color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Right: QUBO variable count comparison
configs = ['2×2', '3×3', '4×4']
x = np.arange(len(configs))
global_vars = [int(g['qubo_vars']) for g in global_data]
sub_vars = [int(s['qubo_vars_per_sub']) for s in sub_data]
w = 0.3

bars1 = ax2.bar(x - w/2, global_vars, w, label='Global QUBO variables',
                color='#E53935', alpha=0.85, edgecolor='white')
bars2 = ax2.bar(x + w/2, sub_vars, w, label='Subdomain QUBO variables (fixed)',
                color='#43A047', alpha=0.85, edgecolor='white')
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
             f'{int(bar.get_height())}', ha='center', fontsize=9, weight='bold')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
             f'{int(bar.get_height())}', ha='center', fontsize=9, weight='bold',
             color='#2E7D32')
ax2.set_xticks(x)
ax2.set_xticklabels(configs)
ax2.set_xlabel('Subdomain configuration', fontsize=13)
ax2.set_ylabel('Number of QUBO variables', fontsize=13)
ax2.set_title('(b) QUBO variable count', fontsize=14)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Figure 1: Condition number explosion in global LS-QUBO\nvs. DD-QUBO decoupling',
             fontsize=15, y=1.03)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig_cond_analysis.pdf')
plt.savefig(f'{OUTPUT_DIR}/fig_cond_analysis.png')
plt.close()
print('Saved: fig_cond_analysis.pdf')

# ============================================================
# Figure 2: RMSE Scaling (Boxplot)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: RMSE boxplot
rmse_data = [res_2x2['rmse_per_subdomain'], rmse_3x3, res_4x4['rmse_per_subdomain']]
labels = ['2×2\n(81 interior)', '3×3\n(196 interior)', '4×4\n(361 interior)']

bp = ax1.boxplot(rmse_data, labels=labels, patch_artist=True, widths=0.5,
                  medianprops={'color': 'black', 'linewidth': 2})
colors = ['#64B5F6', '#42A5F5', '#1E88E5']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

# Add mean markers
means = [np.mean(d) for d in rmse_data]
ax1.scatter([1, 2, 3], means, marker='D', color='red', s=80, zorder=5,
            label='Mean RMSE', edgecolors='darkred')
for i, m in enumerate(means):
    ax1.annotate(f'{m:.3f}', (i+1.15, m), fontsize=9, color='darkred', weight='bold')

ax1.set_ylabel('RMSE vs. classical (per subdomain)', fontsize=13)
ax1.set_xlabel('Subdomain configuration', fontsize=13)
ax1.set_title('(a) Per-subdomain RMSE distribution', fontsize=14)
ax1.legend(loc='upper right', framealpha=0.9)
ax1.grid(True, alpha=0.3, axis='y')

# Right: Global L2 error comparison
configs_full = ['Global\nLS-QUBO\n(36 interior)', 'DD 2×2\n(81 interior)',
                'DD 3×3\n(196 interior)', 'DD 4×4\n(361 interior)']
# Compute approximate L2 for 3x3 from RMSE data
# For DD, L2_abs = global RMSE
l2_errors = [res_single['rmse_vs_exact'],
             res_2x2['L2_abs_quantum'],
             np.sqrt(np.mean([r**2 for r in rmse_3x3])),  # Approximate
             res_4x4['L2_abs_quantum']]
colors_bar = ['#E53935', '#64B5F6', '#42A5F5', '#1E88E5']

bars = ax2.bar(range(4), l2_errors, color=colors_bar, edgecolor='white', width=0.6)
for i, (bar, val) in enumerate(zip(bars, l2_errors)):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
             f'{val:.3f}', ha='center', fontsize=11, weight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels(configs_full, fontsize=9)
ax2.set_ylabel('$L_2$ error (RMSE vs. exact)', fontsize=13)
ax2.set_title('(b) Global $L_2$ error comparison', fontsize=14)
ax2.axhline(y=res_single['rmse_vs_exact'], color='#E53935', ls='--', lw=1,
            alpha=0.5, label=f'Single-shot baseline ({res_single["rmse_vs_exact"]:.2f})')
ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Figure 2: DD-QUBO accuracy does not degrade with increasing problem size',
             fontsize=15, y=1.03)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig_scaling.pdf')
plt.savefig(f'{OUTPUT_DIR}/fig_scaling.png')
plt.close()
print('Saved: fig_scaling.pdf')

# ============================================================
# Figure 3: Framework Diagram
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14); ax.set_ylim(0, 7)
ax.axis('off')

C_PDE, C_DD, C_QUBO, C_CIM = '#1565C0', '#EF6C00', '#2E7D32', '#7B1FA2'
C_BOUND, C_SOL = '#C62828', '#00695C'

def draw_box(ax, x, y, w, h, text, color, fs=10, tc='white'):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor='white', linewidth=1.5,
                          alpha=0.92, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, color=tc,
            weight='bold', zorder=4)

def arrow(ax, x1, y1, x2, y2, color='#546E7A', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->,head_length=0.6,head_width=0.5',
                                color=color, lw=lw), zorder=2)

# Row 1: PDE + DD
draw_box(ax, 2.0, 5.8, 3.2, 1.1, r'$-\nabla^2 u = f$'+'\nin Ω,  u=g on ∂Ω', C_PDE, 11)
draw_box(ax, 7.0, 5.8, 3.6, 1.1, 'Domain\nDecomposition', C_DD, 11)
arrow(ax, 3.7, 5.8, 5.1, 5.8)

# Row 2: Split
y2 = 4.2
draw_box(ax, 3.5, y2, 3.4, 0.9, 'Subdomain QUBOs\n'+r'$Q_k = K_k^2\otimes(ss^T)+\cdots$', C_QUBO, 9)
draw_box(ax, 9.5, y2, 3.2, 0.9, 'Global Boundary\n'+r'$S u_b = g$ (classical)', C_BOUND, 9)
arrow(ax, 7.0, 5.2, 3.5, 4.65)
arrow(ax, 8.8, 5.2, 9.5, 4.65)

# Row 3: CIM parallel
y3 = 2.9
draw_box(ax, 2.3, y3, 1.6, 0.7, r'$\mathbf{CIM}_1$', C_CIM, 9)
draw_box(ax, 4.2, y3, 1.6, 0.7, r'$\mathbf{CIM}_2$', C_CIM, 9)
ax.text(5.6, y3, '...', ha='center', va='center', fontsize=18, color=C_CIM, weight='bold')
draw_box(ax, 7.0, y3, 1.6, 0.7, r'$\mathbf{CIM}_k$', C_CIM, 9)
arrow(ax, 3.5, 3.75, 2.3, 3.25)
arrow(ax, 3.5, 3.75, 4.2, 3.25)
arrow(ax, 3.5, 3.75, 7.0, 3.25)
ax.annotate('Parallel\nSubmission', xy=(3.8, 3.4), fontsize=8, color=C_CIM, weight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=C_CIM, alpha=0.8))

# Row 4: Decode
y4 = 1.8
draw_box(ax, 3.0, y4, 2.0, 0.7, r'$x_k^{(int)}$'+'\ndecode', C_QUBO, 9)
draw_box(ax, 9.5, y4, 2.0, 0.7, r'$u_b$'+'\nboundary', C_BOUND, 9)
arrow(ax, 2.3, 2.55, 3.0, 2.15)
arrow(ax, 4.2, 2.55, 3.0, 2.15)
arrow(ax, 7.0, 2.55, 3.0, 2.15)
arrow(ax, 9.5, 3.75, 9.5, 2.15)

# Merge
arrow(ax, 4.0, y4, 6.5, y4)
arrow(ax, 8.5, y4, 6.5, y4)

# Final solution
draw_box(ax, 7.5, 0.8, 4.0, 0.9, 'Global Solution\n'+r'$u(x)$ on full domain Ω', C_SOL, 11)
arrow(ax, 6.5, 1.35, 7.5, 1.2)

# Key insight box
ax.text(12.0, 5.5, 'Key Properties:', fontsize=10, weight='bold', color='#212121')
insights = [
    r'$\kappa(Q_k) \propto m^4$, independent of $N$',
    r'$n_{\rm var}^{\rm(sub)} = 128$, constant',
    'Subdomain QUBOs are independent',
    'Arbitrarily large PDE at fixed\nquantum resource per subproblem',
]
for i, t in enumerate(insights):
    ax.text(12.0, 4.8 - i*0.55, r'$\diamond$ ' + t, fontsize=8.5, color='#37474F',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5F5F5',
                      edgecolor='#BDBDBD', alpha=0.85))

ax.set_title('DD-QUBO: Domain-Decomposed QUBO for Scalable PDE-Constrained Optimization',
             fontsize=13, weight='bold', pad=15)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig_framework.pdf')
plt.savefig(f'{OUTPUT_DIR}/fig_framework.png')
plt.close()
print('Saved: fig_framework.pdf')

# ============================================================
# Summary statistics for paper
# ============================================================
print("\n=== Paper Statistics ===")
print(f"Condition number scaling:")
for g in cond_data['global']:
    print(f"  {g['nblk']}: grid={g['grid']}, n_i={g['n_internal']}, "
          f"κ(K)={g['cond_K']:.1f}, κ(K²)={g['cond_K2']:.1f}")
for s in cond_data['subdomain']:
    print(f"  {s['nblk']}: κ(K)={s['cond_K_mean']:.2f}±{s['cond_K_std']:.1e}, "
          f"κ(K²)={s['cond_K2_mean']:.2f}±{s['cond_K2_std']:.1e}")

print(f"\nSingle-shot LS (global): RMSE={res_single['rmse_vs_exact']:.4f}")
print(f"Single-shot Energy (global): RMSE={res_energy_single['rmse_vs_exact']:.4f}")
print(f"DD 2×2: L2={res_2x2['L2_abs_quantum']:.4f}, "
      f"RMSE mean={res_2x2['rmse_mean']:.4f}±{res_2x2['rmse_std']:.4f}")
print(f"DD 3×3: RMSE mean={np.mean(rmse_3x3):.4f}±{np.std(rmse_3x3):.4f}")
print(f"DD 4×4: L2={res_4x4['L2_abs_quantum']:.4f}, "
      f"RMSE mean={res_4x4['rmse_mean']:.4f}±{res_4x4['rmse_std']:.4f}")

print("\nDone! Figures saved to:", OUTPUT_DIR)
