"""
论文框架图: Domain-Decomposed QUBO (DD-QUBO) 范式
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(0, 16); ax.set_ylim(0, 8)
ax.axis('off')

# 配色
C_PDE     = '#2196F3'   # 蓝
C_DD      = '#FF9800'   # 橙
C_QUBO    = '#4CAF50'   # 绿
C_CIM     = '#9C27B0'   # 紫
C_BOUND   = '#F44336'   # 红
C_SOL     = '#00897B'   # 青
C_TEXT    = '#212121'
C_ARROW   = '#546E7A'

def draw_box(ax, x, y, w, h, text, color, fontsize=11, text_color='white', bold=True):
    """圆角矩形"""
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle="round,pad=0.15", facecolor=color,
                          edgecolor='white', linewidth=2, alpha=0.95, zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, style='simple', lw=2.5):
    """箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=f'{style},head_length=0.8,head_width=0.6',
                                color=color, lw=lw, alpha=0.8), zorder=2)

def draw_label(ax, x, y, text, fontsize=9, color=C_TEXT, ha='center', rotation=0):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize, color=color,
            style='italic', rotation=rotation)

# ===== 第一行: PDE & 域分解 =====
y1 = 6.5
draw_box(ax, 2.0, y1, 3.2, 1.2, r'$\mathbf{-\nabla^2 u = f}$'+'\nin Ω,  u=g on ∂Ω',
         C_PDE, 11)
draw_box(ax, 6.5, y1, 3.6, 1.2, r'$\mathbf{Domain\ Decomposition}$'+"\nΩ = ∪Ωₖ, 静态凝聚",
         C_DD, 11)

draw_arrow(ax, 3.7, y1, 4.6, y1)

# ===== 两路分支 =====
# 左路: 子域 QUBO (并行)
y2 = 4.5
x_sub = 4.2
draw_box(ax, x_sub, y2, 3.6, 1.0, r'$\mathbf{Subdomain\ QUBOs}$'+"\nQₖ = Kₖ²⊗(ssᵀ)+2·diag(…)",
         C_QUBO, 9.5)

# 右路: 全局边界系统
x_bnd = 9.5
draw_box(ax, x_bnd, y2, 3.2, 1.0, r'$\mathbf{Global\ Boundary}$'+"\nS uᵦ = g, 经典求解",
         C_BOUND, 9.5)

draw_arrow(ax, 6.5, y1-0.6, x_sub, y2+0.5, C_ARROW)
draw_arrow(ax, 8.3, y1-0.6, x_bnd, y2+0.5, C_ARROW)

# 标签
draw_label(ax, 6.7, y1-0.3, '子域内部', ha='center', fontsize=8, color=C_QUBO)
draw_label(ax, 8.8, y1-0.3, '子域边界', ha='center', fontsize=8, color=C_BOUND)

# ===== 第三行: CIM & 并行标注 =====
y3 = 3.2
draw_box(ax, 2.5, y3, 2.2, 0.9, r'$\mathbf{CIM}_1$', C_CIM, 10)
draw_box(ax, 4.7, y3, 2.2, 0.9, r'$\mathbf{CIM}_2$', C_CIM, 10)
draw_box(ax, 6.9, y3, 2.2, 0.9, r'$\mathbf{CIM}_k$', C_CIM, 10)
ax.text(5.8, y3-0.07, '...', ha='center', va='center', fontsize=20, color=C_CIM, weight='bold')

draw_arrow(ax, x_sub, y2-0.5, 2.5, y3+0.45, C_ARROW)
draw_arrow(ax, x_sub, y2-0.5, 4.7, y3+0.45, C_ARROW)
draw_arrow(ax, x_sub, y2-0.5, 6.9, y3+0.45, C_ARROW)

# 并行标注
ax.annotate('并行提交', xy=(4.0, 2.5), fontsize=9, color=C_CIM, weight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_CIM, alpha=0.8))

# ===== 第四行: 回填 & 全局解 =====
y4 = 2.0
draw_box(ax, 3.5, y4, 2.2, 0.8, r'$u_k^{(internal)}$'+'\n解码', C_QUBO, 9)
draw_box(ax, 9.5, y4, 2.2, 0.8, r'$u_b$'+'\n边界解', C_BOUND, 9)

draw_arrow(ax, 2.5, y3-0.45, 3.5, y4+0.4, C_ARROW)
draw_arrow(ax, 4.7, y3-0.45, 3.5, y4+0.4, C_ARROW)
draw_arrow(ax, 6.9, y3-0.45, 3.5, y4+0.4, C_ARROW)
draw_arrow(ax, x_bnd, y2-0.5, 9.5, y4+0.4, C_ARROW)

# 汇合
draw_arrow(ax, 4.6, y4, 7.5, y4, C_ARROW)
draw_arrow(ax, 8.4, y4, 7.5, y4, C_ARROW)

# ===== 最终解 =====
y5 = 1.0
draw_box(ax, 8.0, y5, 3.8, 1.0, r'$\mathbf{Global\ Solution}$'+"\nu(x) on full domain Ω",
         C_SOL, 11)

draw_arrow(ax, 7.5, y4-0.4, 8.0, y5+0.5, C_ARROW)

# ===== 左下标注: 核心创新点 =====
innovations = [
    r'$\kappa(Q_k)\propto m^4$, 与全局网格 $N$ 无关',
    r'$n_{\rm var}$ 固定, 网格加密不增加量子资源',
    '子域 QUBO 独立, 可并行提交',
]
for i, text in enumerate(innovations):
    ax.text(11.5, 6.5 - i*0.5, f'✦ {text}', fontsize=9, color=C_TEXT,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.9))

ax.text(11.5, 5.0, '核心贡献:', fontsize=10, weight='bold', color=C_TEXT)

# ===== 底部: 方法名 =====
ax.text(8.0, 0.2, 'DD-QUBO: Domain-Decomposed QUBO for Scalable PDE-Constrained Optimization',
        ha='center', fontsize=12, weight='bold', color=C_TEXT)

plt.tight_layout()
plt.savefig('D:/QPDE/experime2/framework_diagram.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("框架图已保存: D:/QPDE/experime2/framework_diagram.png")
