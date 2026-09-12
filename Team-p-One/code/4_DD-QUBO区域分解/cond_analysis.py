"""
条件数分析：全局 K vs 子域 K
证明 DD 将 QUBO 条件数从网格相关解耦为子域固定
"""
import numpy as np, matplotlib.pyplot as plt, json, os

Lx, Ly = 3.0, 2.0
BIT_WIDTH = 8
OUTPUT_DIR = 'D:/QPDE/experime2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== FEM kernels =====
def gauss_points_2x2():
    sq = np.sqrt(3.0)
    return [(-1/sq,-1/sq,1), (-1/sq,1/sq,1), (1/sq,-1/sq,1), (1/sq,1/sq,1)]

def shape_functions(xi, eta):
    N = np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])/4.0
    dN = np.array([[-(1-eta),-(1-xi)],[(1-eta),-(1+xi)],[(1+eta),(1+xi)],[-(1+eta),(1-xi)]])/4.0
    return N, dN

def elem_stiffness(xe, ye):
    Ke = np.zeros((4,4))
    for xi, eta, w in gauss_points_2x2():
        _, dN = shape_functions(xi, eta)
        J = np.zeros((2,2))
        for i in range(4):
            J[0,0] += dN[i,0]*xe[i]; J[0,1] += dN[i,0]*ye[i]
            J[1,0] += dN[i,1]*xe[i]; J[1,1] += dN[i,1]*ye[i]
        detJ = np.linalg.det(J); invJ = np.linalg.inv(J)
        dN_dxy = np.array([invJ @ dN[i] for i in range(4)])
        Ke += (dN_dxy[:,0:1] @ dN_dxy[:,0:1].T + dN_dxy[:,1:2] @ dN_dxy[:,1:2].T) * detJ * w
    return Ke


def build_global_fem(nblk_x, nblk_y, ne_per_blk_x=5, ne_per_blk_y=5):
    """构建全局 FEM, 提取 K_ii (内部节点刚度矩阵)"""
    nx = nblk_x * ne_per_blk_x; ny = nblk_y * ne_per_blk_y
    Nx, Ny = nx + 1, ny + 1
    n_nodes = Nx * Ny
    x = np.linspace(0, Lx, Nx); y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    # 组装全局刚度矩阵
    K_global = np.zeros((n_nodes, n_nodes))
    for j in range(ny):
        for i in range(nx):
            n0 = j*Nx + i; n1 = n0+1; n2 = n0+Nx+1; n3 = n0+Nx
            conn = [n0, n1, n2, n3]
            xe = nodes[conn,0]; ye = nodes[conn,1]
            Ke = elem_stiffness(xe, ye)
            for a, la in enumerate(conn):
                for b, lb in enumerate(conn):
                    K_global[la, lb] += Ke[a, b]

    # 边界判断
    eps = 1e-12
    boundary = ((abs(X.ravel()) < eps) | (abs(X.ravel() - Lx) < eps) |
                (abs(Y.ravel()) < eps) | (abs(Y.ravel() - Ly) < eps))
    internal = ~boundary
    n_i = internal.sum()

    # 提取 K_ii
    idx_map = -np.ones(n_nodes, dtype=int)
    idx_map[internal] = np.arange(n_i)
    K_ii = np.zeros((n_i, n_i))
    for gi in np.where(internal)[0]:
        ki = idx_map[gi]
        for gj in np.where(internal)[0]:
            kj = idx_map[gj]
            K_ii[ki, kj] = K_global[gi, gj]

    return K_ii, n_i, nodes, internal, Nx, Ny


def build_dd_and_extract_subdomain_Kii(nblk_x, nblk_y, ne_per_blk_x=5, ne_per_blk_y=5):
    """构建 DD, 提取每个子域的 K_ii 矩阵"""
    nx = nblk_x * ne_per_blk_x; ny = nblk_y * ne_per_blk_y
    Nx, Ny = nx + 1, ny + 1
    n_nodes = Nx * Ny
    x = np.linspace(0, Lx, Nx); y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    # 单元 → 子域分配
    blk_w = Lx / nblk_x; blk_h = Ly / nblk_y
    elements = []
    for j in range(ny):
        for i in range(nx):
            elements.append([j*Nx+i, j*Nx+i+1, (j+1)*Nx+i+1, (j+1)*Nx+i])

    def get_sid(cx, cy):
        return min(int(cy/blk_h), nblk_y-1) * nblk_x + min(int(cx/blk_w), nblk_x-1)

    subdomains = [{'elements': [], 'nodes': set()} for _ in range(nblk_x * nblk_y)]
    for idx_e, elem in enumerate(elements):
        coords = nodes[elem]
        cx, cy = np.mean(coords[:,0]), np.mean(coords[:,1])
        sid = get_sid(cx, cy)
        subdomains[sid]['elements'].append(elem)
        for n in elem:
            subdomains[sid]['nodes'].add(n)
    for sd in subdomains:
        sd['nodes'] = list(sd['nodes'])

    def on_subdomain_boundary(xy, blk_x, blk_y):
        x, y = xy; eps=1e-12
        return (abs(x-blk_x*blk_w)<eps or abs(x-(blk_x+1)*blk_w)<eps or
                abs(y-blk_y*blk_h)<eps or abs(y-(blk_y+1)*blk_h)<eps)

    subdomain_Kiis = []
    for sid, sd in enumerate(subdomains):
        blk_ix = sid % nblk_x; blk_iy = sid // nblk_x
        local_nodes = sd['nodes']
        n_local = len(local_nodes)
        g2l = {gid: i for i, gid in enumerate(local_nodes)}

        K_local = np.zeros((n_local, n_local))
        for elem in sd['elements']:
            xe = nodes[elem,0]; ye = nodes[elem,1]
            Ke = elem_stiffness(xe, ye)
            for a, ga in enumerate(elem):
                for b, gb in enumerate(elem):
                    K_local[g2l[ga], g2l[gb]] += Ke[a, b]

        is_b = np.array([on_subdomain_boundary(nodes[gid], blk_ix, blk_iy) for gid in local_nodes])
        idx_i = np.where(~is_b)[0]
        idx_b = np.where(is_b)[0]

        if len(idx_i) > 0 and len(idx_b) > 0:
            K_ii = K_local[np.ix_(idx_i, idx_i)]
            subdomain_Kiis.append(K_ii)

    return subdomain_Kiis, Nx, Ny, n_nodes


# ===== 实验 1: 全局 K_ii 条件数随网格增长 =====
print("=" * 70)
print("实验 1: 全局 K_ii 条件数 vs 网格规模")
print("=" * 70)

configs = [(2,2), (3,3), (4,4)]
results_global = []
for nb_x, nb_y in configs:
    K_ii, n_i, _, _, Nx, Ny = build_global_fem(nb_x, nb_y)
    c_K = np.linalg.cond(K_ii)
    K2 = K_ii @ K_ii
    c_K2 = np.linalg.cond(K2)
    results_global.append({
        'nblk': f'{nb_x}×{nb_y}', 'grid': f'{Nx}×{Ny}',
        'n_internal': n_i, 'cond_K': c_K, 'cond_K2': c_K2,
        'qubo_vars': n_i * BIT_WIDTH
    })
    print(f"  {nb_x}×{nb_y} DD 等价网格 {Nx}×{Ny}: "
          f"内部节点={n_i}, κ(K)={c_K:.2e}, κ(K²)={c_K2:.2e}, "
          f"8位可解={'✓' if c_K2 < 65000 else '✗'}")

# ===== 实验 2: 子域 K_ii 条件数 (DD 场景) =====
print()
print("=" * 70)
print("实验 2: 子域 K_ii 条件数 (固定子域大小, 变化子域数量)")
print("=" * 70)

results_sub = []
for nb_x, nb_y in configs:
    sub_Kiis, Nx, Ny, n_nodes = build_dd_and_extract_subdomain_Kii(nb_x, nb_y)
    conds_K = [np.linalg.cond(Kii) for Kii in sub_Kiis]
    conds_K2 = [np.linalg.cond(Kii@Kii) for Kii in sub_Kiis]
    results_sub.append({
        'nblk': f'{nb_x}×{nb_y}', 'n_subdomains': len(sub_Kiis),
        'n_internal_per_sub': sub_Kiis[0].shape[0],
        'cond_K_mean': np.mean(conds_K), 'cond_K_std': np.std(conds_K),
        'cond_K2_mean': np.mean(conds_K2), 'cond_K2_std': np.std(conds_K2),
        'qubo_vars_per_sub': sub_Kiis[0].shape[0] * BIT_WIDTH
    })
    print(f"  {nb_x}×{nb_y} 子域: κ(K)={np.mean(conds_K):.1f}±{np.std(conds_K):.1f}, "
          f"κ(K²)={np.mean(conds_K2):.1f}±{np.std(conds_K2):.1f}, "
          f"QUBO变量/子域={sub_Kiis[0].shape[0] * BIT_WIDTH}")

# ===== 汇总对比表 =====
print()
print("=" * 70)
print("汇总: 全局 vs DD 条件数对比")
print("=" * 70)
print(f"{'配置':<10} {'全局 κ(K)':<14} {'全局 κ(K²)':<14} {'子域 κ(K)':<14} {'子域 κ(K²)':<14} {'QUBO(全局)':<12} {'QUBO(子域)':<12}")
print("-" * 90)
for rg, rs in zip(results_global, results_sub):
    print(f"{rg['nblk']:<10} {rg['cond_K']:<14.2e} {rg['cond_K2']:<14.2e} "
          f"{rs['cond_K_mean']:<14.1f} {rs['cond_K2_mean']:<14.1f} "
          f"{rg['qubo_vars']:<12} {rs['qubo_vars_per_sub']:<12}")

# ===== 可视化 =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左: 条件数 vs 内部节点数
n_vals = [r['n_internal'] for r in results_global]
ax1.loglog(n_vals, [r['cond_K'] for r in results_global], 'bo-', label=r'$\kappa(K)$ 全局')
ax1.loglog(n_vals, [r['cond_K2'] for r in results_global], 'rs-', label=r'$\kappa(K^2)$ 全局')
ax1.axhline(y=results_sub[0]['cond_K2_mean'], color='green', linestyle='--',
            label=rf'$\kappa(K^2)$ 子域 (~{results_sub[0]["cond_K2_mean"]:.0f})')
ax1.axhline(y=2**16, color='red', linestyle=':', alpha=0.5, label='8位动态范围上限')
ax1.set_xlabel('内部节点数'); ax1.set_ylabel('条件数')
ax1.set_title('条件数增长: 全局 K² 平方爆炸 vs 子域恒定')
ax1.legend(); ax1.grid(True, alpha=0.3)

# 右: QUBO 变量数 vs 子域数
n_subdomains = [r['n_subdomains'] for r in results_sub]
ax2.bar(np.arange(len(configs)) - 0.15,
        [r['qubo_vars'] for r in results_global], 0.3,
        label='全局 QUBO 变量', color='red', alpha=0.7)
ax2.bar(np.arange(len(configs)) + 0.15,
        [r['qubo_vars_per_sub'] for r in results_sub], 0.3,
        label='每子域 QUBO 变量 (固定)', color='green', alpha=0.7)
ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels([f"{nb_x}×{nb_y}" for nb_x, nb_y in configs])
ax2.set_ylabel('QUBO 变量数'); ax2.set_xlabel('子域配置')
ax2.set_title('QUBO 规模: 全局线性增长 vs DD 恒定')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cond_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n图表已保存: {OUTPUT_DIR}/cond_analysis.png")

# 保存数据
with open(f'{OUTPUT_DIR}/cond_results.json', 'w') as f:
    json.dump({'global': results_global, 'subdomain': results_sub}, f, indent=2, default=str)
print(f"数据已保存: {OUTPUT_DIR}/cond_results.json")
