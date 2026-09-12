"""
一键生成 5 个实验 notebook:
  1. ls_indef.ipynb       — LS-QUBO 不定矩阵 (A = K - αI)
  2. energy_indef.ipynb   — SAC-QUBO 不定矩阵 (A = K - αI, |A|, sign(A))
  3. ls_negdef.ipynb      — LS-QUBO 负定矩阵 (A = -K)
  4. energy_negdef.ipynb  — SAC-QUBO 负定矩阵 (A = -K, |A|=K, sign(A)=-I)
  5. cond_measure.ipynb   — 条件数实测 (无需 CIM)

用法: python generate_all.py
"""
import json, os

OUT = os.path.dirname(__file__) or '.'
os.makedirs(f'{OUT}/output_ls_indef', exist_ok=True)
os.makedirs(f'{OUT}/output_energy_indef', exist_ok=True)
os.makedirs(f'{OUT}/output_ls_negdef', exist_ok=True)
os.makedirs(f'{OUT}/output_energy_negdef', exist_ok=True)

# ============================================================
# Cell builders
# ============================================================

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def code(lines):
    """lines: list of strings, each is one line of code"""
    if isinstance(lines, str):
        lines = [lines]
    src = [l + "\n" for l in lines[:-1]] + [lines[-1] + "\n"]
    return {
        "cell_type": "code",
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None,
    }

def make_notebook(cells, kernel="python3"):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": kernel, "language": "python", "name": kernel},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def save_nb(name, cells):
    nb = make_notebook(cells)
    path = os.path.join(OUT, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  OK  {name}  ({len(cells)} cells)")

# ============================================================
# 公共片段
# ============================================================

_COMMON_PREAMBLE = [
    "import numpy as np, matplotlib.pyplot as plt, time, warnings, json, os",
    "warnings.filterwarnings('ignore')",
    "import kaiwu as kw",
    "kw.common.CheckpointManager.save_dir = '/tmp'",
    "",
    "Lx, Ly = 3.0, 2.0",
    "Nx, Ny = 10, 8          # 更大网格: 48 内部节点 (单次求解 8×8 仅 36)",
    "BIT_WIDTH = 8",
]

_COMMON_MESH = [
    "# ===== 网格 =====",
    "x = np.linspace(0, Lx, Nx); y = np.linspace(0, Ly, Ny)",
    "X, Y = np.meshgrid(x, y, indexing='ij')",
    "nodes = np.column_stack([X.ravel(), Y.ravel()])",
    "n_nodes = Nx * Ny",
    "boundary = (X.ravel()==0) | (X.ravel()==Lx) | (Y.ravel()==0) | (Y.ravel()==Ly)",
    "internal = ~boundary",
    "n_i = internal.sum()",
    "print(f'总节点: {n_nodes}, 内部: {n_i}, QUBO变量: {n_i * BIT_WIDTH}')",
]

_COMMON_EXACT = [
    "# ===== 解析解 & 源项 =====",
    "def u_exact(x, y):",
    "    return x * y * np.cos(2*np.pi*x/3.0)",
    "",
    "def source_term(x, y):",
    "    u_xx = -(2*np.pi/3.0)**2 * x * y * np.cos(2*np.pi*x/3.0) - (4*np.pi/3.0) * y * np.sin(2*np.pi*x/3.0)",
    "    return -(u_xx + 0.0)",
    "",
    "u_exact_all = np.array([u_exact(xi, yi) for xi, yi in nodes])",
    "print(f'精确解: [{u_exact_all.min():.4f}, {u_exact_all.max():.4f}]')",
]

_COMMON_FEM = [
    "# ===== FEM 装配 (5-point FD) =====",
    "hx = Lx/(Nx-1); hy = Ly/(Ny-1)",
    "K = np.zeros((n_nodes, n_nodes)); F = np.zeros(n_nodes)",
    "for i in range(1, Nx-1):",
    "    for j in range(1, Ny-1):",
    "        k = i*Ny + j",
    "        K[k,k] = 2/hx**2 + 2/hy**2",
    "        K[k, k-1] = K[k, k+1] = -1/hy**2",
    "        K[k, k-Ny] = K[k, k+Ny] = -1/hx**2",
    "        F[k] = source_term(nodes[k,0], nodes[k,1])",
    "print(f'K: {K.shape}, cond(K)={np.linalg.cond(K):.2e}')",
]

_COMMON_BC_REDUCE = [
    "# ===== Dirichlet BC 归约 =====",
    "idx_map = -np.ones(n_nodes, dtype=int)",
    "idx_map[internal] = np.arange(n_i)",
    "K_ii = np.zeros((n_i, n_i)); rhs = np.zeros(n_i)",
    "for gi in np.where(internal)[0]:",
    "    ki = idx_map[gi]; K_ii[ki, ki] = K[gi, gi]; rhs[ki] = F[gi]",
    "    for gj in [gi-1, gi+1, gi-Ny, gi+Ny]:",
    "        if 0 <= gj < n_nodes:",
    "            if internal[gj]: K_ii[ki, idx_map[gj]] = K[gi, gj]",
    "            else: rhs[ki] -= K[gi, gj] * u_exact(nodes[gj,0], nodes[gj,1])",
    "u_poisson = np.linalg.solve(K_ii, rhs)             # 原 Poisson 解 (仅参考)",
    "print(f'K_ii cond: {np.linalg.cond(K_ii):.2f}')",
]

_LS_QUBO_FUNC = [
    "def build_ls_qubo(A, b, bw, lb, ub):",
    "    '''广义 LS-QUBO: min||A(Mz+c) - b||²",
    "    Q = A²⊗(ssᵀ) + 2·diag(MᵀA(Ac - b))'''",
    "    n = A.shape[0]; nvar = n*bw",
    "    scale = (ub-lb)/(2**bw-1)",
    "    s = scale * np.array([2**k for k in range(bw)]); ssT = np.outer(s,s)",
    "    c = lb * np.ones(n); d = A@c - b; Ad = A@d; A2 = A@A",
    "    Q = np.zeros((nvar, nvar))",
    "    for i in range(n):",
    "        for j in range(i, n):",
    "            if abs(A2[i,j]) < 1e-15: continue",
    "            ri, rj = i*bw, j*bw; blk = A2[i,j]*ssT",
    "            Q[ri:ri+bw, rj:rj+bw] += blk",
    "            if i != j: Q[rj:rj+bw, ri:ri+bw] += blk.T",
    "    for i in range(n): Q[i*bw:(i+1)*bw, i*bw:(i+1)*bw] += np.diag(2*Ad[i]*s)",
    "    return Q.astype(np.float32), nvar, scale",
]

_SAC_QUBO_FUNC = [
    "def build_sac_qubo(A_abs, r_sac, bw, lb, ub):",
    "    '''SAC 能量 QUBO: min ½xᵀ|A|x - xᵀr_sac",
    "    Q = ½|A|⊗(ssᵀ) + diag(Mᵀ(|A|c - r_sac))'''",
    "    n = A_abs.shape[0]; nvar = n*bw",
    "    scale = (ub-lb)/(2**bw-1)",
    "    s = scale * np.array([2**k for k in range(bw)]); ssT = np.outer(s,s)",
    "    c = lb * np.ones(n); w = A_abs@c - r_sac",
    "    Q = np.zeros((nvar, nvar))",
    "    for i in range(n):",
    "        for j in range(i, n):",
    "            aij = A_abs[i,j]",
    "            if abs(aij) < 1e-15: continue",
    "            ri, rj = i*bw, j*bw; blk = 0.5*aij*ssT",
    "            Q[ri:ri+bw, rj:rj+bw] += blk",
    "            if i != j: Q[rj:rj+bw, ri:ri+bw] += blk.T",
    "    for i in range(n): Q[i*bw:(i+1)*bw, i*bw:(i+1)*bw] += np.diag(w[i]*s)",
    "    return Q.astype(np.float32), nvar, scale",
]

_DECODE_FUNC = [
    "def decode(z_best, n, bw, scale, lb):",
    "    M = np.zeros((n, n*bw)); s = scale*np.array([2**k for k in range(bw)])",
    "    for i in range(n): M[i, i*bw:(i+1)*bw] = s",
    "    return M@z_best + lb",
]

# ============================================================
# 1. LS-QUBO 不定矩阵 (A = K - αI)
# ============================================================
def build_ls_indef():
    cells = [
        md("# LS-QUBO — 不定矩阵 $A = K - \\alpha I$\n"
           "## $\\min_x \\|A x - b\\|^2$  →  $Q = A^2 \\otimes (s s^T) + 2\\cdot\\text{diag}(M^T A(Ac - b))$\n"
           "不定矩阵 → LS 平方所有特征值 → 条件数爆炸 $\\kappa(A^2) = \\kappa(A)^2$"),
        code(_COMMON_PREAMBLE + [
            "OUTPUT_DIR = 'D:/QPDE/fuck/output_ls_indef'",
            "os.makedirs(OUTPUT_DIR, exist_ok=True)",
            "TASK_NAME = 'ls_indef'",
            "METHOD = 'ls_indef'",
            "print(f'网格: {Nx}×{Ny}, 位宽: {BIT_WIDTH}')",
        ]),
        code(_COMMON_MESH),
        code(_COMMON_EXACT),
        code(_COMMON_FEM),
        code(_COMMON_BC_REDUCE),
        code([
            "# ===== 构造不定矩阵 A = K - αI =====",
            "eigvals = np.linalg.eigvalsh(K_ii)",
            "lam_min, lam_max = eigvals.min(), eigvals.max()",
            "ALPHA = lam_min + 0.4*(lam_max - lam_min)  # ~40% 特征值变负",
            "A_mat = K_ii - ALPHA * np.eye(n_i)",
            "b_vec = rhs.copy()",
            "eigvals_A = np.linalg.eigvalsh(A_mat)",
            "n_neg = (eigvals_A < 0).sum(); n_pos = (eigvals_A > 0).sum()",
            "print(f'K_ii 特征值: [{lam_min:.3f}, {lam_max:.3f}]')",
            "print(f'α = {ALPHA:.3f}')",
            "print(f'A 特征值: [{eigvals_A.min():.3f}, {eigvals_A.max():.3f}], 正:{n_pos}, 负:{n_neg}')",
            "print(f'κ(K_ii)={lam_max/lam_min:.1f}, κ(A)={np.linalg.cond(A_mat):.1f}')",
            "print(f'κ(A²)={np.linalg.cond(A_mat@A_mat):.1f} = κ(A)²')",
        ]),
        code([
            "# ===== 参考解: A x = b 的精确解 =====",
            "u_ref_A = np.linalg.solve(A_mat, b_vec)",
            "print(f'A⁻¹b: [{u_ref_A.min():.4f}, {u_ref_A.max():.4f}]')",
            "print(f'(原 Poisson 解: [{u_poisson.min():.4f}, {u_poisson.max():.4f}])')",
        ]),
        code([
            "# ===== 编码边界 (基于 u_ref_A) =====",
            "margin = 0.5; rng = max(u_ref_A.max()-u_ref_A.min(), 0.1)",
            "lb = u_ref_A.min() - margin*rng; ub = u_ref_A.max() + margin*rng",
            "print(f'编码界: [{lb:.4f}, {ub:.4f}]')",
        ]),
        code(_LS_QUBO_FUNC),
        code(_DECODE_FUNC),
        code([
            "# ===== LS-QUBO 构建 =====",
            "Q_float, nvar, scale = build_ls_qubo(A_mat, b_vec, BIT_WIDTH, lb, ub)",
            "Q_qubo = kw.qubo.adjust_qubo_matrix_precision(Q_float)",
            "print(f'QUBO: {nvar}x{nvar}, 值: [{Q_qubo.min():.0f}, {Q_qubo.max():.0f}]')",
        ]),
        code([
            "# ===== CIM 提交 =====",
            "ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_qubo)",
            "vars_ = [f'x[{i}]' for i in range(ising_mat.shape[0])]",
            "model = kw.ising.IsingModel(variables=vars_, ising_matrix=ising_mat, bias=ising_bias)",
            "opt = kw.cim.CIMOptimizer(task_name=TASK_NAME, task_mode='quota')",
            "t0 = time.time()",
            "opt.solve(model.get_matrix())",
            "print(f'已提交, Ising: {ising_mat.shape}')",
        ]),
        code([
            "# ===== CIM 取回 + 解码 =====",
            "sol = opt.solve(model.get_matrix())",
            "dt = time.time()-t0; print(f'返回: {sol.shape}, 耗时: {dt:.1f}s')",
            "",
            "sols_bin = (sol[:,:-1]*sol[:,-1:]+1)/2",
            "energies = np.array([z@Q_qubo@z for z in sols_bin])",
            "z_best = sols_bin[np.argmin(energies)]",
            "u_quantum_i = decode(z_best, n_i, BIT_WIDTH, scale, lb)",
            "",
            "# 全节点解",
            "u_quantum = np.zeros(n_nodes); u_quantum[internal] = u_quantum_i",
            "for gi in np.where(boundary)[0]: u_quantum[gi] = u_exact(nodes[gi,0], nodes[gi,1])",
            "",
            "# 参考解 (全节点)",
            "u_ref_A_all = np.zeros(n_nodes); u_ref_A_all[internal] = u_ref_A",
            "for gi in np.where(boundary)[0]: u_ref_A_all[gi] = u_exact(nodes[gi,0], nodes[gi,1])",
            "",
            "rmse_ref = np.sqrt(np.mean((u_quantum_i - u_ref_A)**2))  # vs A⁻¹b",
            "print(f'最优能量: {energies.min():.1f}')",
            "print(f'RMSE vs A⁻¹b (参考): {rmse_ref:.6e}')",
            "print(f'κ(A²) = {np.linalg.cond(A_mat@A_mat):.1f}')",
        ]),
        code([
            "# ===== 保存 =====",
            "np.save(f'{OUTPUT_DIR}/u_quantum.npy', u_quantum)",
            "np.save(f'{OUTPUT_DIR}/u_ref_A.npy', u_ref_A_all)",
            "res = {'method':METHOD,'grid':f'{Nx}x{Ny}','n_internal':int(n_i),",
            "       'qubo_size':int(nvar),'time_sec':dt,",
            "       'rmse_vs_ref':float(rmse_ref),",
            "       'best_energy':float(energies.min()),",
            "       'cond_K_ii':float(np.linalg.cond(K_ii)),",
            "       'cond_A':float(np.linalg.cond(A_mat)),",
            "       'cond_A2':float(np.linalg.cond(A_mat@A_mat)),",
            "       'alpha':float(ALPHA)}",
            "with open(f'{OUTPUT_DIR}/result.json','w') as f: json.dump(res, f, indent=2, default=str)",
            "print('结果已保存')",
        ]),
        code([
            "# ===== 可视化 =====",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
            "for ax, data, title in [",
            "    (axes[0,0], u_quantum, f'LS-QUBO (κ(A²)={np.linalg.cond(A_mat@A_mat):.0f})'),",
            "    (axes[0,1], u_ref_A_all, 'Reference A⁻¹b'),",
            "    (axes[1,0], u_quantum-u_ref_A_all, 'Error (QUBO - Ref)'),",
            "    (axes[1,1], u_ref_A_all-u_exact_all, 'Δ(A⁻¹b - Poisson)')]:",
            "    cmap = 'coolwarm' if 'Error' in title else 'viridis'",
            "    im = ax.tricontourf(nodes[:,0], nodes[:,1], data, levels=20, cmap=cmap)",
            "    ax.set_title(title); ax.axis('equal'); fig.colorbar(im, ax=ax)",
            "fig.suptitle('Indefinite Matrix — LS-QUBO ($A = K - \\\\alpha I$)', fontsize=13, y=1.01)",
            "plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/solution.png', dpi=150, bbox_inches='tight')",
            "plt.show()",
        ]),
    ]
    save_nb("ls_indef.ipynb", cells)


# ============================================================
# 2. SAC-QUBO 不定矩阵 (A = K - αI)
# ============================================================
def build_energy_indef():
    cells = [
        md("# SAC-QUBO — 不定矩阵 $A = K - \\alpha I$\n"
           "## $E(x) = \\frac{1}{2}x^T|A| x - x^T\\text{sign}(A)b$\n"
           "$Q = \\frac{1}{2}|A| \\otimes (s s^T) + \\text{diag}(M^T(|A|c - \\text{sign}(A)b))$\n"
           "SAC 仅翻转负特征值 → $\\kappa(|A|) = \\kappa(A) \\ll \\kappa(A^2)$"),
        code(_COMMON_PREAMBLE + [
            "OUTPUT_DIR = 'D:/QPDE/fuck/output_energy_indef'",
            "os.makedirs(OUTPUT_DIR, exist_ok=True)",
            "TASK_NAME = 'energy_indef'",
            "METHOD = 'energy_indef'",
            "print(f'网格: {Nx}×{Ny}, 位宽: {BIT_WIDTH}')",
        ]),
        code(_COMMON_MESH),
        code(_COMMON_EXACT),
        code(_COMMON_FEM),
        code(_COMMON_BC_REDUCE),
        code([
            "# ===== 构造不定矩阵 A = K - αI =====",
            "eigvals = np.linalg.eigvalsh(K_ii)",
            "lam_min, lam_max = eigvals.min(), eigvals.max()",
            "ALPHA = lam_min + 0.4*(lam_max - lam_min)",
            "A_mat = K_ii - ALPHA * np.eye(n_i)",
            "b_vec = rhs.copy()",
            "eigvals_A = np.linalg.eigvalsh(A_mat)",
            "n_neg = (eigvals_A < 0).sum(); n_pos = (eigvals_A > 0).sum()",
            "print(f'K_ii 特征值: [{lam_min:.3f}, {lam_max:.3f}]')",
            "print(f'α = {ALPHA:.3f}')",
            "print(f'A 特征值: [{eigvals_A.min():.3f}, {eigvals_A.max():.3f}], 正:{n_pos}, 负:{n_neg}')",
        ]),
        code([
            "# ===== SAC: 计算 |A| 和 sign(A) =====",
            "eigvals_full, Q_mat = np.linalg.eigh(A_mat)",
            "absA = Q_mat @ np.diag(np.abs(eigvals_full)) @ Q_mat.T",
            "signA = Q_mat @ np.diag(np.sign(eigvals_full)) @ Q_mat.T",
            "r_sac = signA @ b_vec",
            "print(f'|A| 构造完毕, κ(|A|) = {np.linalg.cond(absA):.1f}')",
            "print(f'(cf. κ(A) = {np.linalg.cond(A_mat):.1f}, κ(A²) = {np.linalg.cond(A_mat@A_mat):.1f})')",
            "print(f'SAC 条件数优势: {np.linalg.cond(A_mat@A_mat)/np.linalg.cond(absA):.1f}×')",
        ]),
        code([
            "# ===== 参考解: A x = b 的精确解 =====",
            "u_ref_A = np.linalg.solve(A_mat, b_vec)",
            "print(f'A⁻¹b: [{u_ref_A.min():.4f}, {u_ref_A.max():.4f}]')",
        ]),
        code([
            "# ===== 编码边界 (基于 u_ref_A) =====",
            "margin = 0.5; rng = max(u_ref_A.max()-u_ref_A.min(), 0.1)",
            "lb = u_ref_A.min() - margin*rng; ub = u_ref_A.max() + margin*rng",
            "print(f'编码界: [{lb:.4f}, {ub:.4f}]')",
        ]),
        code(_SAC_QUBO_FUNC),
        code(_DECODE_FUNC),
        code([
            "# ===== SAC-QUBO 构建 =====",
            "Q_float, nvar, scale = build_sac_qubo(absA, r_sac, BIT_WIDTH, lb, ub)",
            "Q_qubo = kw.qubo.adjust_qubo_matrix_precision(Q_float)",
            "print(f'SAC QUBO: {nvar}x{nvar}, 值: [{Q_qubo.min():.0f}, {Q_qubo.max():.0f}]')",
        ]),
        code([
            "# ===== CIM 提交 =====",
            "ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_qubo)",
            "vars_ = [f'x[{i}]' for i in range(ising_mat.shape[0])]",
            "model = kw.ising.IsingModel(variables=vars_, ising_matrix=ising_mat, bias=ising_bias)",
            "opt = kw.cim.CIMOptimizer(task_name=TASK_NAME, task_mode='quota')",
            "t0 = time.time()",
            "opt.solve(model.get_matrix())",
            "print(f'已提交, Ising: {ising_mat.shape}')",
        ]),
        code([
            "# ===== CIM 取回 + 解码 =====",
            "sol = opt.solve(model.get_matrix())",
            "dt = time.time()-t0; print(f'返回: {sol.shape}, 耗时: {dt:.1f}s')",
            "",
            "sols_bin = (sol[:,:-1]*sol[:,-1:]+1)/2",
            "energies = np.array([z@Q_qubo@z for z in sols_bin])",
            "z_best = sols_bin[np.argmin(energies)]",
            "u_quantum_i = decode(z_best, n_i, BIT_WIDTH, scale, lb)",
            "",
            "u_quantum = np.zeros(n_nodes); u_quantum[internal] = u_quantum_i",
            "for gi in np.where(boundary)[0]: u_quantum[gi] = u_exact(nodes[gi,0], nodes[gi,1])",
            "",
            "u_ref_A_all = np.zeros(n_nodes); u_ref_A_all[internal] = u_ref_A",
            "for gi in np.where(boundary)[0]: u_ref_A_all[gi] = u_exact(nodes[gi,0], nodes[gi,1])",
            "",
            "rmse_ref = np.sqrt(np.mean((u_quantum_i - u_ref_A)**2))",
            "print(f'最优能量: {energies.min():.1f}')",
            "print(f'RMSE vs A⁻¹b (参考): {rmse_ref:.6e}')",
            "print(f'κ(|A|) = {np.linalg.cond(absA):.1f}  (cf. κ(A²) = {np.linalg.cond(A_mat@A_mat):.1f})')",
        ]),
        code([
            "# ===== 保存 =====",
            "np.save(f'{OUTPUT_DIR}/u_quantum.npy', u_quantum)",
            "np.save(f'{OUTPUT_DIR}/u_ref_A.npy', u_ref_A_all)",
            "res = {'method':METHOD,'grid':f'{Nx}x{Ny}','n_internal':int(n_i),",
            "       'qubo_size':int(nvar),'time_sec':dt,",
            "       'rmse_vs_ref':float(rmse_ref),",
            "       'best_energy':float(energies.min()),",
            "       'cond_K_ii':float(np.linalg.cond(K_ii)),",
            "       'cond_A':float(np.linalg.cond(A_mat)),",
            "       'cond_absA':float(np.linalg.cond(absA)),",
            "       'cond_A2':float(np.linalg.cond(A_mat@A_mat)),",
            "       'alpha':float(ALPHA)}",
            "with open(f'{OUTPUT_DIR}/result.json','w') as f: json.dump(res, f, indent=2, default=str)",
            "print('结果已保存')",
        ]),
        code([
            "# ===== 可视化 =====",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
            "for ax, data, title in [",
            "    (axes[0,0], u_quantum, f'SAC-QUBO (κ(|A|)={np.linalg.cond(absA):.0f})'),",
            "    (axes[0,1], u_ref_A_all, 'Reference A⁻¹b'),",
            "    (axes[1,0], u_quantum-u_ref_A_all, 'Error (QUBO - Ref)'),",
            "    (axes[1,1], u_ref_A_all-u_exact_all, 'Δ(A⁻¹b - Poisson)')]:",
            "    cmap = 'coolwarm' if 'Error' in title else 'viridis'",
            "    im = ax.tricontourf(nodes[:,0], nodes[:,1], data, levels=20, cmap=cmap)",
            "    ax.set_title(title); ax.axis('equal'); fig.colorbar(im, ax=ax)",
            "fig.suptitle('Indefinite Matrix — SAC-QUBO ($|A|$, sign$(A)$)', fontsize=13, y=1.01)",
            "plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/solution.png', dpi=150, bbox_inches='tight')",
            "plt.show()",
        ]),
    ]
    save_nb("energy_indef.ipynb", cells)


# ============================================================
# 3. LS-QUBO 负定矩阵 (A = -K)
# ============================================================
def build_ls_negdef():
    cells = [
        md("# LS-QUBO — 负定矩阵 $A = -K$\n"
           "## $\\min_x \\|A x - b\\|^2$  →  $Q = K^2 \\otimes (s s^T) + 2\\cdot\\text{diag}(M^T A(Ac - b))$\n"
           "所有特征值 < 0 → LS 平方后全变正 → $\\kappa(A^2) = \\kappa(K)^2$"),
        code(_COMMON_PREAMBLE + [
            "OUTPUT_DIR = 'D:/QPDE/fuck/output_ls_negdef'",
            "os.makedirs(OUTPUT_DIR, exist_ok=True)",
            "TASK_NAME = 'ls_negdef'",
            "METHOD = 'ls_negdef'",
            "print(f'网格: {Nx}×{Ny}, 位宽: {BIT_WIDTH}')",
        ]),
        code(_COMMON_MESH),
        code(_COMMON_EXACT),
        code(_COMMON_FEM),
        code(_COMMON_BC_REDUCE),
        code([
            "# ===== 构造负定矩阵 A = -K =====",
            "A_mat = -K_ii",
            "b_vec = rhs.copy()",
            "eigvals_A = np.linalg.eigvalsh(A_mat)",
            "print(f'A = -K_ii, 特征值全负: [{eigvals_A.max():.3f}, {eigvals_A.min():.3f}]')",
            "print(f'κ(K_ii) = {np.linalg.cond(K_ii):.1f}')",
            "print(f'κ(A) = {np.linalg.cond(A_mat):.1f}')",
            "print(f'κ(A²) = κ(K²) = {np.linalg.cond(A_mat@A_mat):.1f}')",
        ]),
        code([
            "# ===== 参考解: A x = b 的精确解 =====",
            "u_ref_A = np.linalg.solve(A_mat, b_vec)",
            "print(f'A⁻¹b: [{u_ref_A.min():.4f}, {u_ref_A.max():.4f}]')",
            "print(f'(原 Poisson 解: [{u_poisson.min():.4f}, {u_poisson.max():.4f}])')",
        ]),
        code([
            "# ===== 编码边界 (基于 u_ref_A) =====",
            "margin = 0.5; rng = max(u_ref_A.max()-u_ref_A.min(), 0.1)",
            "lb = u_ref_A.min() - margin*rng; ub = u_ref_A.max() + margin*rng",
            "print(f'编码界: [{lb:.4f}, {ub:.4f}]')",
        ]),
        code(_LS_QUBO_FUNC),
        code(_DECODE_FUNC),
        code([
            "# ===== LS-QUBO 构建 =====",
            "Q_float, nvar, scale = build_ls_qubo(A_mat, b_vec, BIT_WIDTH, lb, ub)",
            "Q_qubo = kw.qubo.adjust_qubo_matrix_precision(Q_float)",
            "print(f'QUBO: {nvar}x{nvar}, 值: [{Q_qubo.min():.0f}, {Q_qubo.max():.0f}]')",
            "print(f'κ(A²) = {np.linalg.cond(A_mat@A_mat):.1f}  (LS 支付了 κ(K)² 的代价)')",
        ]),
        code([
            "# ===== CIM 提交 =====",
            "ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_qubo)",
            "vars_ = [f'x[{i}]' for i in range(ising_mat.shape[0])]",
            "model = kw.ising.IsingModel(variables=vars_, ising_matrix=ising_mat, bias=ising_bias)",
            "opt = kw.cim.CIMOptimizer(task_name=TASK_NAME, task_mode='quota')",
            "t0 = time.time()",
            "opt.solve(model.get_matrix())",
            "print(f'已提交, Ising: {ising_mat.shape}')",
        ]),
        code([
            "# ===== CIM 取回 + 解码 =====",
            "sol = opt.solve(model.get_matrix())",
            "dt = time.time()-t0; print(f'返回: {sol.shape}, 耗时: {dt:.1f}s')",
            "",
            "sols_bin = (sol[:,:-1]*sol[:,-1:]+1)/2",
            "energies = np.array([z@Q_qubo@z for z in sols_bin])",
            "z_best = sols_bin[np.argmin(energies)]",
            "u_quantum_i = decode(z_best, n_i, BIT_WIDTH, scale, lb)",
            "",
            "u_quantum = np.zeros(n_nodes); u_quantum[internal] = u_quantum_i",
            "for gi in np.where(boundary)[0]: u_quantum[gi] = u_exact(nodes[gi,0], nodes[gi,1])",
            "",
            "u_ref_A_all = np.zeros(n_nodes); u_ref_A_all[internal] = u_ref_A",
            "for gi in np.where(boundary)[0]: u_ref_A_all[gi] = u_exact(nodes[gi,0], nodes[gi,1])",
            "",
            "rmse_ref = np.sqrt(np.mean((u_quantum_i - u_ref_A)**2))",
            "print(f'最优能量: {energies.min():.1f}')",
            "print(f'RMSE vs A⁻¹b (参考): {rmse_ref:.6e}')",
            "print(f'κ(A²) = {np.linalg.cond(A_mat@A_mat):.1f}')",
        ]),
        code([
            "# ===== 保存 =====",
            "np.save(f'{OUTPUT_DIR}/u_quantum.npy', u_quantum)",
            "np.save(f'{OUTPUT_DIR}/u_ref_A.npy', u_ref_A_all)",
            "res = {'method':METHOD,'grid':f'{Nx}x{Ny}','n_internal':int(n_i),",
            "       'qubo_size':int(nvar),'time_sec':dt,",
            "       'rmse_vs_ref':float(rmse_ref),",
            "       'best_energy':float(energies.min()),",
            "       'cond_K_ii':float(np.linalg.cond(K_ii)),",
            "       'cond_A':float(np.linalg.cond(A_mat)),",
            "       'cond_A2':float(np.linalg.cond(A_mat@A_mat))}",
            "with open(f'{OUTPUT_DIR}/result.json','w') as f: json.dump(res, f, indent=2, default=str)",
            "print('结果已保存')",
        ]),
        code([
            "# ===== 可视化 =====",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
            "for ax, data, title in [",
            "    (axes[0,0], u_quantum, f'LS-QUBO (κ(A²)={np.linalg.cond(A_mat@A_mat):.0f})'),",
            "    (axes[0,1], u_ref_A_all, 'Reference A⁻¹b'),",
            "    (axes[1,0], u_quantum-u_ref_A_all, 'Error (QUBO - Ref)'),",
            "    (axes[1,1], u_ref_A_all-u_exact_all, 'Δ(A⁻¹b - Poisson)')]:",
            "    cmap = 'coolwarm' if 'Error' in title else 'viridis'",
            "    im = ax.tricontourf(nodes[:,0], nodes[:,1], data, levels=20, cmap=cmap)",
            "    ax.set_title(title); ax.axis('equal'); fig.colorbar(im, ax=ax)",
            "fig.suptitle('Negative Definite Matrix — LS-QUBO ($A = -K$)', fontsize=13, y=1.01)",
            "plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/solution.png', dpi=150, bbox_inches='tight')",
            "plt.show()",
        ]),
    ]
    save_nb("ls_negdef.ipynb", cells)


# ============================================================
# 4. SAC-QUBO 负定矩阵 (A = -K)
# ============================================================
def build_energy_negdef():
    cells = [
        md("# SAC-QUBO — 负定矩阵 $A = -K$\n"
           "## $|A| = K,\\quad \\text{sign}(A) = -I,\\quad r_* = -b$\n"
           "$E(x) = \\frac{1}{2}x^T K x - x^T(-b) = \\frac{1}{2}x^T K x + x^T b$\n"
           "SAC 退化: 翻转符号 → $\\kappa(|A|) = \\kappa(K) \\ll \\kappa(K^2)$"),
        code(_COMMON_PREAMBLE + [
            "OUTPUT_DIR = 'D:/QPDE/fuck/output_energy_negdef'",
            "os.makedirs(OUTPUT_DIR, exist_ok=True)",
            "TASK_NAME = 'energy_negdef'",
            "METHOD = 'energy_negdef'",
            "print(f'网格: {Nx}×{Ny}, 位宽: {BIT_WIDTH}')",
        ]),
        code(_COMMON_MESH),
        code(_COMMON_EXACT),
        code(_COMMON_FEM),
        code(_COMMON_BC_REDUCE),
        code([
            "# ===== 构造负定矩阵 A = -K =====",
            "A_mat = -K_ii",
            "b_vec = rhs.copy()",
            "eigvals_A = np.linalg.eigvalsh(A_mat)",
            "print(f'A = -K_ii, 特征值全负: [{eigvals_A.max():.3f}, {eigvals_A.min():.3f}]')",
            "",
            "# ===== SAC: |A| = K, sign(A) = -I → r_* = -b =====",
            "absA = K_ii.copy()         # |-K| = K  (无需特征分解!)",
            "r_sac = -b_vec               # sign(-K)·b = -b",
            "print(f'|A| = K (直接使用), κ(|A|) = {np.linalg.cond(absA):.1f}')",
            "print(f'SAC 优势: κ(|A|)={np.linalg.cond(absA):.0f} vs κ(A²)={np.linalg.cond(K_ii@K_ii):.0f}')",
            "print(f'条件数降低 {np.linalg.cond(K_ii@K_ii)/np.linalg.cond(K_ii):.0f}×')",
        ]),
        code([
            "# ===== 参考解: A x = b 的精确解 =====",
            "u_ref_A = np.linalg.solve(A_mat, b_vec)",
            "print(f'A⁻¹b: [{u_ref_A.min():.4f}, {u_ref_A.max():.4f}]')",
        ]),
        code([
            "# ===== 编码边界 (基于 u_ref_A) =====",
            "margin = 0.5; rng = max(u_ref_A.max()-u_ref_A.min(), 0.1)",
            "lb = u_ref_A.min() - margin*rng; ub = u_ref_A.max() + margin*rng",
            "print(f'编码界: [{lb:.4f}, {ub:.4f}]')",
        ]),
        code(_SAC_QUBO_FUNC),
        code(_DECODE_FUNC),
        code([
            "# ===== SAC-QUBO 构建 (|A| = K, r_sac = -b) =====",
            "Q_float, nvar, scale = build_sac_qubo(absA, r_sac, BIT_WIDTH, lb, ub)",
            "Q_qubo = kw.qubo.adjust_qubo_matrix_precision(Q_float)",
            "print(f'SAC QUBO: {nvar}x{nvar}, 值: [{Q_qubo.min():.0f}, {Q_qubo.max():.0f}]')",
            "print(f'(同正定 Energy-QUBO, 仅右端项符号翻转)')",
        ]),
        code([
            "# ===== CIM 提交 =====",
            "ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_qubo)",
            "vars_ = [f'x[{i}]' for i in range(ising_mat.shape[0])]",
            "model = kw.ising.IsingModel(variables=vars_, ising_matrix=ising_mat, bias=ising_bias)",
            "opt = kw.cim.CIMOptimizer(task_name=TASK_NAME, task_mode='quota')",
            "t0 = time.time()",
            "opt.solve(model.get_matrix())",
            "print(f'已提交, Ising: {ising_mat.shape}')",
        ]),
        code([
            "# ===== CIM 取回 + 解码 =====",
            "sol = opt.solve(model.get_matrix())",
            "dt = time.time()-t0; print(f'返回: {sol.shape}, 耗时: {dt:.1f}s')",
            "",
            "sols_bin = (sol[:,:-1]*sol[:,-1:]+1)/2",
            "energies = np.array([z@Q_qubo@z for z in sols_bin])",
            "z_best = sols_bin[np.argmin(energies)]",
            "u_quantum_i = decode(z_best, n_i, BIT_WIDTH, scale, lb)",
            "",
            "u_quantum = np.zeros(n_nodes); u_quantum[internal] = u_quantum_i",
            "for gi in np.where(boundary)[0]: u_quantum[gi] = u_exact(nodes[gi,0], nodes[gi,1])",
            "",
            "u_ref_A_all = np.zeros(n_nodes); u_ref_A_all[internal] = u_ref_A",
            "for gi in np.where(boundary)[0]: u_ref_A_all[gi] = u_exact(nodes[gi,0], nodes[gi,1])",
            "",
            "rmse_ref = np.sqrt(np.mean((u_quantum_i - u_ref_A)**2))",
            "print(f'最优能量: {energies.min():.1f}')",
            "print(f'RMSE vs A⁻¹b (参考): {rmse_ref:.6e}')",
            "print(f'κ(|A|) = κ(K) = {np.linalg.cond(K_ii):.1f}  (cf. κ(A²) = {np.linalg.cond(K_ii@K_ii):.1f})')",
        ]),
        code([
            "# ===== 保存 =====",
            "np.save(f'{OUTPUT_DIR}/u_quantum.npy', u_quantum)",
            "np.save(f'{OUTPUT_DIR}/u_ref_A.npy', u_ref_A_all)",
            "res = {'method':METHOD,'grid':f'{Nx}x{Ny}','n_internal':int(n_i),",
            "       'qubo_size':int(nvar),'time_sec':dt,",
            "       'rmse_vs_ref':float(rmse_ref),",
            "       'best_energy':float(energies.min()),",
            "       'cond_K_ii':float(np.linalg.cond(K_ii)),",
            "       'cond_A':float(np.linalg.cond(A_mat)),",
            "       'cond_absA':float(np.linalg.cond(absA)),",
            "       'cond_A2':float(np.linalg.cond(A_mat@A_mat))}",
            "with open(f'{OUTPUT_DIR}/result.json','w') as f: json.dump(res, f, indent=2, default=str)",
            "print('结果已保存')",
        ]),
        code([
            "# ===== 可视化 =====",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
            "for ax, data, title in [",
            "    (axes[0,0], u_quantum, f'SAC-QUBO (κ(|A|)={np.linalg.cond(absA):.0f})'),",
            "    (axes[0,1], u_ref_A_all, 'Reference A⁻¹b'),",
            "    (axes[1,0], u_quantum-u_ref_A_all, 'Error (QUBO - Ref)'),",
            "    (axes[1,1], u_ref_A_all-u_exact_all, 'Δ(A⁻¹b - Poisson)')]:",
            "    cmap = 'coolwarm' if 'Error' in title else 'viridis'",
            "    im = ax.tricontourf(nodes[:,0], nodes[:,1], data, levels=20, cmap=cmap)",
            "    ax.set_title(title); ax.axis('equal'); fig.colorbar(im, ax=ax)",
            "fig.suptitle('Negative Definite Matrix — SAC-QUBO ($|A|=K$, $r_*=-b$)', fontsize=13, y=1.01)",
            "plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/solution.png', dpi=150, bbox_inches='tight')",
            "plt.show()",
        ]),
    ]
    save_nb("energy_negdef.ipynb", cells)


# ============================================================
# 5. 条件数实测 (无需 CIM)
# ============================================================
def build_cond_measure():
    cells = [
        md("# 条件数实测 — 无需量子计算\n"
           "比较多矩阵类型的条件数随网格规模增长:\n"
           "- 正定: $\\kappa(K)$ vs $\\kappa(K^2)$\n"
           "- 不定: $\\kappa(A)$ vs $\\kappa(A^2)$ vs $\\kappa(|A|)$\n"
           "- 负定: $\\kappa(A)$ vs $\\kappa(A^2)$ vs $\\kappa(|A|)$\n\n"
           "验证: SAC $\\kappa(|A|) = \\kappa(A) \\ll \\kappa(A^2)$"),
        code([
            "import numpy as np, matplotlib.pyplot as plt, json, os",
            "",
            "Lx, Ly = 3.0, 2.0",
            "OUTPUT_DIR = 'D:/QPDE/fuck'",
            "os.makedirs(OUTPUT_DIR, exist_ok=True)",
        ]),
        code([
            "# ===== FEM 装配 (5-point FD, 任意网格) =====",
            "def build_fem(Nx, Ny):",
            "    x = np.linspace(0, Lx, Nx); y = np.linspace(0, Ly, Ny)",
            "    X, Y = np.meshgrid(x, y, indexing='ij')",
            "    n_nodes = Nx * Ny",
            "    hx = Lx/(Nx-1); hy = Ly/(Ny-1)",
            "    K = np.zeros((n_nodes, n_nodes))",
            "    for i in range(1, Nx-1):",
            "        for j in range(1, Ny-1):",
            "            k = i*Ny + j",
            "            K[k,k] = 2/hx**2 + 2/hy**2",
            "            K[k, k-1] = K[k, k+1] = -1/hy**2",
            "            K[k, k-Ny] = K[k, k+Ny] = -1/hx**2",
            "",
            "    boundary = ((X.ravel()==0) | (X.ravel()==Lx) |",
            "                (Y.ravel()==0) | (Y.ravel()==Ly))",
            "    internal = ~boundary",
            "    n_i = internal.sum()",
            "    idx_map = -np.ones(n_nodes, dtype=int)",
            "    idx_map[internal] = np.arange(n_i)",
            "    K_ii = np.zeros((n_i, n_i))",
            "    for gi in np.where(internal)[0]:",
            "        ki = idx_map[gi]; K_ii[ki, ki] = K[gi, gi]",
            "        for gj in [gi-1, gi+1, gi-Ny, gi+Ny]:",
            "            if 0 <= gj < n_nodes and internal[gj]:",
            "                K_ii[ki, idx_map[gj]] = K[gi, gj]",
            "    return K_ii, n_i",
        ]),
        code([
            "# ===== 多网格条件数测量 =====",
            "grids = [(6,5), (8,6), (10,8), (12,8), (14,10), (16,12)]",
            "results = []",
            "",
            "for Nx, Ny in grids:",
            "    K_ii, n_i = build_fem(Nx, Ny)",
            "    eigvals = np.linalg.eigvalsh(K_ii)",
            "    lam_min, lam_max = eigvals.min(), eigvals.max()",
            "",
            "    # --- SPD (原始 K) ---",
            "    kappa_K = np.linalg.cond(K_ii)",
            "    kappa_K2 = np.linalg.cond(K_ii @ K_ii)",
            "",
            "    # --- 不定: A = K - αI ---",
            "    alpha = lam_min + 0.4*(lam_max - lam_min)",
            "    A_indef = K_ii - alpha * np.eye(n_i)",
            "    eigvals_A = np.linalg.eigvalsh(A_indef)",
            "    kappa_A_indef = np.linalg.cond(A_indef)",
            "    kappa_A2_indef = np.linalg.cond(A_indef @ A_indef)",
            "    kappa_absA_indef = np.abs(eigvals_A).max() / np.abs(eigvals_A).min()",
            "",
            "    # --- 负定: A = -K ---",
            "    A_negdef = -K_ii",
            "    kappa_A_negdef = np.linalg.cond(A_negdef)",
            "    kappa_A2_negdef = np.linalg.cond(A_negdef @ A_negdef)",
            "    kappa_absA_negdef = np.linalg.cond(K_ii)  # |-K| = K",
            "",
            "    row = {",
            "        'grid': f'{Nx}×{Ny}', 'n_internal': int(n_i),",
            "        'kappa_K': float(kappa_K),",
            "        'kappa_K2': float(kappa_K2),",
            "        'alpha': float(alpha),",
            "        'n_neg_A': int((eigvals_A < 0).sum()),",
            "        'n_pos_A': int((eigvals_A > 0).sum()),",
            "        'kappa_A_indef': float(kappa_A_indef),",
            "        'kappa_A2_indef': float(kappa_A2_indef),",
            "        'kappa_absA_indef': float(kappa_absA_indef),",
            "        'kappa_A_negdef': float(kappa_A_negdef),",
            "        'kappa_A2_negdef': float(kappa_A2_negdef),",
            "        'kappa_absA_negdef': float(kappa_absA_negdef),",
            "    }",
            "    results.append(row)",
            "",
            "    print(f'N={Nx}×{Ny} n_i={n_i:3d}: '",
            "          f'κ(K)={kappa_K:.1f}, κ(K²)={kappa_K2:.1f}, '",
            "          f'不定 α={alpha:.1f}: κ(A)={kappa_A_indef:.1f}, κ(A²)={kappa_A2_indef:.1f}, κ(|A|)={kappa_absA_indef:.1f} | '",
            "          f'SAC优势: {kappa_A2_indef/kappa_absA_indef:.1f}×')",
            "",
            "with open(f'{OUTPUT_DIR}/cond_measure_results.json', 'w') as f:",
            "    json.dump(results, f, indent=2)",
            "print(f'\\n已保存: {OUTPUT_DIR}/cond_measure_results.json')",
        ]),
        code([
            "# ===== 可视化: 三联合图 =====",
            "fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))",
            "",
            "n_vals = [r['n_internal'] for r in results]",
            "",
            "# 左: SPD K vs K²",
            "ax = axes[0]",
            "ax.loglog(n_vals, [r['kappa_K'] for r in results], 'bo-', lw=2, ms=8, label=r'$\kappa(K)$')",
            "ax.loglog(n_vals, [r['kappa_K2'] for r in results], 'rs-', lw=2, ms=8, label=r'$\kappa(K^2)$')",
            "ax.axhline(y=2**16, color='red', ls=':', alpha=0.5, label='8-bit limit ($2^{16}$)')",
            "ax.set_xlabel('Interior nodes $n_i$'); ax.set_ylabel('Condition number $\kappa$')",
            "ax.set_title('(a) SPD: $K$ vs $K^2$'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)",
            "",
            "# 中: 不定 A = K-αI",
            "ax = axes[1]",
            "ax.loglog(n_vals, [r['kappa_A_indef'] for r in results], 'bo-', lw=2, ms=8, alpha=0.7, label=r'$\kappa(A)$ (indef)')",
            "ax.loglog(n_vals, [r['kappa_A2_indef'] for r in results], 'rs-', lw=2, ms=8, alpha=0.7, label=r'$\kappa(A^2)$ (LS)')",
            "ax.loglog(n_vals, [r['kappa_absA_indef'] for r in results], 'g^-', lw=2.5, ms=9, label=r'$\kappa(|A|)$ (SAC)')",
            "ax.set_xlabel('Interior nodes $n_i$'); ax.set_ylabel('Condition number $\kappa$')",
            "ax.set_title(r'(b) Indefinite $A=K-\\alpha I$'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)",
            "",
            "# 右: 负定 A = -K",
            "ax = axes[2]",
            "ax.loglog(n_vals, [r['kappa_A_negdef'] for r in results], 'bo-', lw=2, ms=8, alpha=0.7, label=r'$\kappa(-K)$')",
            "ax.loglog(n_vals, [r['kappa_A2_negdef'] for r in results], 'rs-', lw=2, ms=8, alpha=0.7, label=r'$\kappa(K^2)$ (LS)')",
            "ax.loglog(n_vals, [r['kappa_absA_negdef'] for r in results], 'g^-', lw=2.5, ms=9, label=r'$\kappa(K)$ (SAC)')",
            "ax.set_xlabel('Interior nodes $n_i$'); ax.set_ylabel('Condition number $\kappa$')",
            "ax.set_title('(c) Negative Definite $A=-K$'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)",
            "",
            "fig.suptitle('SAC 条件数优势: $\\kappa(|A|) = \\kappa(A) \\ll \\kappa(A^2) = \\kappa(A)^2$',",
            "             fontsize=14, y=1.02)",
            "plt.tight_layout()",
            "plt.savefig(f'{OUTPUT_DIR}/cond_measure.png', dpi=150, bbox_inches='tight')",
            "plt.show()",
        ]),
        code([
            "# ===== 汇总表 =====",
            "print('\\n' + '='*115)",
            "print(f'{\"Grid\":<10} {\"n_i\":>4}  {\"κ(K)\":>8}  {\"κ(K²)\":>8}  {\"κ(A)indef\":>10}  {\"κ(A²)indef\":>11}  {\"κ(|A|)indef\":>12}  {\"SAC优势\":>8}')",
            "print('-'*115)",
            "for r in results:",
            "    sac_gain = r['kappa_A2_indef'] / r['kappa_absA_indef']",
            "    print(f'{r[\"grid\"]:<10} {r[\"n_internal\"]:>4}  {r[\"kappa_K\"]:>8.1f}  {r[\"kappa_K2\"]:>8.1f}  '",
            "          f'{r[\"kappa_A_indef\"]:>10.1f}  {r[\"kappa_A2_indef\"]:>11.1f}  {r[\"kappa_absA_indef\"]:>12.1f}  {sac_gain:>7.1f}×')",
            "",
            "print()",
            "print('关键结论:')",
            "last = results[-1]",
            "print(f'  正定: κ(K²) / κ(K) = {last[\"kappa_K2\"]/last[\"kappa_K\"]:.1f}×  (= κ(K))')",
            "indef_gain = last['kappa_A2_indef'] / last['kappa_absA_indef']",
            "print(f'  不定: κ(A²) / κ(|A|) = {indef_gain:.1f}×  (SAC 条件数优势)')",
            "negdef_gain = last['kappa_A2_negdef'] / last['kappa_absA_negdef']",
            "print(f'  负定: κ(A²) / κ(|A|) = {negdef_gain:.1f}×  (SAC 条件数优势)')",
            "print(f'  8-bit 上限 ~65000, κ(K²) 在 n_i≈{results[-1][\"n_internal\"]} 时已接近 {last[\"kappa_K2\"]:.0f}')",
        ]),
    ]
    save_nb("cond_measure.ipynb", cells)


# ============================================================
if __name__ == '__main__':
    print("Generating 5 experiment notebooks...")
    print()
    build_ls_indef()
    build_energy_indef()
    build_ls_negdef()
    build_energy_negdef()
    build_cond_measure()
    print()
    print("Done! All notebooks in:", OUT)
    print()
    print("Run order recommendation:")
    print("  1. cond_measure.ipynb       (no CIM, fast)")
    print("  2. ls_indef.ipynb            (CIM)")
    print("  3. energy_indef.ipynb        (CIM)")
    print("  4. ls_negdef.ipynb           (CIM)")
    print("  5. energy_negdef.ipynb       (CIM)")
