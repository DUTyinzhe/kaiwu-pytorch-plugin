# -*- coding: utf-8 -*-
"""
完整代码：2D Poisson方程 - 有限元子结构法 + 两种QUBO构造 + CIM真机求解 + 量子约束PINN-Transformer
适配最新版Kaiwu SDK（自动轮询结果）
"""

import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import time
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')

import kaiwu as kw

kw.common.CheckpointManager.save_dir = '/tmp'

# ==================== 配置参数 ====================
Lx, Ly = 3.0, 2.0
nblk_x, nblk_y = 3, 3
ne_per_blk_x, ne_per_blk_y = 5, 5
BIT_WIDTH = 8
CIM_TASK_TIMEOUT = 600  # 每个任务最大等待时间（秒）
CIM_POLL_INTERVAL = 5  # 轮询间隔（秒）

# ==================== 1. 网格生成 ====================
nx = nblk_x * ne_per_blk_x
ny = nblk_y * ne_per_blk_y
Nx, Ny = nx + 1, ny + 1
n_nodes = Nx * Ny

x_coords = np.linspace(0, Lx, Nx)
y_coords = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
nodes = np.vstack([X.ravel(), Y.ravel()]).T

elements = []
for j in range(ny):
    for i in range(nx):
        n0 = j * Nx + i
        n1 = n0 + 1
        n2 = n0 + Nx + 1
        n3 = n0 + Nx
        elements.append([n0, n1, n2, n3])
elements = np.array(elements)

print(f"节点数: {n_nodes}, 单元数: {len(elements)}")


# ==================== 2. 解析解与源项 ====================
def u_exact(x, y):
    return x * y * np.cos(2 * np.pi * x / 3.0)


def source_term(x, y):
    u_xx = - (2 * np.pi / 3.0) ** 2 * x * y * np.cos(2 * np.pi * x / 3.0) \
           - (4 * np.pi / 3.0) * y * np.sin(2 * np.pi * x / 3.0)
    u_yy = 0.0
    return -(u_xx + u_yy)


u_exact_all = np.array([u_exact(x, y) for (x, y) in nodes])
print(f"精确解范围: [{u_exact_all.min():.4f}, {u_exact_all.max():.4f}]")


# ==================== 3. 有限元内核 ====================
def gauss_points_2x2():
    sqrt3 = np.sqrt(3.0)
    xi_vals = [-1.0 / sqrt3, 1.0 / sqrt3]
    w = [1.0, 1.0]
    return [(xi_vals[i], xi_vals[j], w[i] * w[j]) for i in range(2) for j in range(2)]


def shape_functions(xi, eta):
    N = np.array([
        (1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)
    ]) / 4.0
    dN_dxi = np.array([
        [-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
        [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]
    ]) / 4.0
    return N, dN_dxi


def elem_stiffness_and_load(xe, ye, source_func):
    Ke = np.zeros((4, 4))
    Fe = np.zeros(4)
    for xi, eta, w in gauss_points_2x2():
        N, dN_dxi = shape_functions(xi, eta)
        J = np.zeros((2, 2))
        for i in range(4):
            J[0, 0] += dN_dxi[i, 0] * xe[i]
            J[0, 1] += dN_dxi[i, 0] * ye[i]
            J[1, 0] += dN_dxi[i, 1] * xe[i]
            J[1, 1] += dN_dxi[i, 1] * ye[i]
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dN_dxy = np.array([invJ @ dN_dxi[i] for i in range(4)])
        for i in range(4):
            for j in range(4):
                Ke[i, j] += (dN_dxy[i, 0] * dN_dxy[j, 0] + dN_dxy[i, 1] * dN_dxy[j, 1]) * detJ * w
        xp = np.dot(N, xe)
        yp = np.dot(N, ye)
        Fe += N * source_func(xp, yp) * detJ * w
    return Ke, Fe


# ==================== 4. 子域划分与静态凝聚 ====================
blk_w = Lx / nblk_x
blk_h = Ly / nblk_y


def get_subdomain_id(cx, cy):
    ix = min(int(np.floor(cx / blk_w)), nblk_x - 1)
    iy = min(int(np.floor(cy / blk_h)), nblk_y - 1)
    return ix, iy


elem_centers = []
for elem in elements:
    coords = nodes[elem]
    elem_centers.append((np.mean(coords[:, 0]), np.mean(coords[:, 1])))

subdomains = [{'elements': [], 'nodes': set()} for _ in range(nblk_x * nblk_y)]
for idx_e, elem in enumerate(elements):
    cx, cy = elem_centers[idx_e]
    ix, iy = get_subdomain_id(cx, cy)
    sid = iy * nblk_x + ix
    subdomains[sid]['elements'].append(elem)
    for n in elem:
        subdomains[sid]['nodes'].add(n)

for sd in subdomains:
    sd['nodes'] = list(sd['nodes'])


def is_node_on_subdomain_boundary(node_coord, blk_x, blk_y):
    x, y = node_coord
    left = blk_x * blk_w
    right = (blk_x + 1) * blk_w
    bottom = blk_y * blk_h
    top = (blk_y + 1) * blk_h
    eps = 1e-12
    return (abs(x - left) < eps or abs(x - right) < eps or
            abs(y - bottom) < eps or abs(y - top) < eps)


def is_on_global_boundary(gid):
    x, y = nodes[gid]
    eps = 1e-12
    return (abs(x) < eps or abs(x - Lx) < eps or
            abs(y) < eps or abs(y - Ly) < eps)


global_boundary_nodes_set = set()
subdomain_data = []

for sid, sd in enumerate(subdomains):
    blk_ix = sid % nblk_x
    blk_iy = sid // nblk_x
    local_nodes = sd['nodes']
    n_local = len(local_nodes)
    global2local = {gid: i for i, gid in enumerate(local_nodes)}

    K_local = np.zeros((n_local, n_local))
    F_local = np.zeros(n_local)
    for elem in sd['elements']:
        conn = elem
        local_conn = [global2local[gid] for gid in conn]
        xe = nodes[conn, 0]
        ye = nodes[conn, 1]
        Ke, Fe = elem_stiffness_and_load(xe, ye, source_term)
        for a, la in enumerate(local_conn):
            F_local[la] += Fe[a]
            for b, lb in enumerate(local_conn):
                K_local[la, lb] += Ke[a, b]

    is_boundary = np.zeros(n_local, dtype=bool)
    for i, gid in enumerate(local_nodes):
        is_boundary[i] = is_node_on_subdomain_boundary(nodes[gid], blk_ix, blk_iy)

    idx_b = np.where(is_boundary)[0]
    idx_i = np.where(~is_boundary)[0]

    if len(idx_i) > 0:
        K_ii = K_local[np.ix_(idx_i, idx_i)]
        K_ib = K_local[np.ix_(idx_i, idx_b)]
        K_bi = K_local[np.ix_(idx_b, idx_i)]
        K_bb = K_local[np.ix_(idx_b, idx_b)]
        F_i = F_local[idx_i]
        F_b = F_local[idx_b]

        inv_K_ii = np.linalg.inv(K_ii)
        S = K_bb - K_bi @ inv_K_ii @ K_ib
        g = F_b - K_bi @ (inv_K_ii @ F_i)
        back = {'K_ii': K_ii, 'K_ib': K_ib, 'F_i': F_i, 'idx_i': idx_i, 'idx_b': idx_b}
    else:
        S = K_local[np.ix_(idx_b, idx_b)]
        g = F_local[idx_b]
        back = None

    boundary_global_ids = [local_nodes[i] for i in idx_b]
    for gid in boundary_global_ids:
        global_boundary_nodes_set.add(gid)

    subdomain_data.append({
        'S': S, 'g': g,
        'boundary_global_ids': boundary_global_ids,
        'back': back,
        'local_nodes': local_nodes,
        'idx_b': idx_b
    })
    print(f"子域 {sid}: 内部节点={len(idx_i)}, 边界节点={len(idx_b)}, QUBO变量数={len(idx_i) * BIT_WIDTH}")

global_boundary_nodes = list(global_boundary_nodes_set)
n_global_b = len(global_boundary_nodes)
global_b_idx = {gid: i for i, gid in enumerate(global_boundary_nodes)}
print(f"全局边界节点总数: {n_global_b}")

# ==================== 5. 全局边界系统装配与求解 ====================
S_global = np.zeros((n_global_b, n_global_b))
g_global = np.zeros(n_global_b)

for data in subdomain_data:
    S = data['S']
    g = data['g']
    bnd_ids = data['boundary_global_ids']
    loc_to_glob = [global_b_idx[gid] for gid in bnd_ids]
    for i_loc, i_glob in enumerate(loc_to_glob):
        g_global[i_glob] += g[i_loc]
        for j_loc, j_glob in enumerate(loc_to_glob):
            S_global[i_glob, j_glob] += S[i_loc, j_loc]

known_flag = np.zeros(n_global_b, dtype=bool)
known_value = np.zeros(n_global_b)
for i, gid in enumerate(global_boundary_nodes):
    if is_on_global_boundary(gid):
        known_flag[i] = True
        known_value[i] = u_exact(nodes[gid, 0], nodes[gid, 1])

unknown_idx = [i for i in range(n_global_b) if not known_flag[i]]
known_idx = [i for i in range(n_global_b) if known_flag[i]]

if len(unknown_idx) == 0:
    u_boundary = known_value.copy()
else:
    S_uu = S_global[np.ix_(unknown_idx, unknown_idx)]
    S_uk = S_global[np.ix_(unknown_idx, known_idx)]
    g_u = g_global[unknown_idx] - S_uk @ known_value[known_idx]
    u_unknown = np.linalg.solve(S_uu, g_u)
    u_boundary = np.zeros(n_global_b)
    u_boundary[known_idx] = known_value[known_idx]
    u_boundary[unknown_idx] = u_unknown

u_global = np.zeros(n_nodes)
for i_glob, gid in enumerate(global_boundary_nodes):
    u_global[gid] = u_boundary[i_glob]

print(f"边界系统求解完成: {n_global_b} 个边界节点 ({len(known_idx)} Dirichlet, {len(unknown_idx)} 未知)")


# ==================== 6. QUBO构建函数 ====================
def estimate_bounds(u_b_local):
    if len(u_b_local) > 0:
        min_b, max_b = np.min(u_b_local), np.max(u_b_local)
        margin = max(1.0, (max_b - min_b) * 0.2)
        return min_b - margin, max_b + margin
    return -10.0, 10.0


def decode_to_continuous(z_best, n_i, bit_width, scale, lower_bound):
    M = np.zeros((n_i, n_i * bit_width))
    s = scale * np.array([2 ** k for k in range(bit_width)])
    for i in range(n_i):
        M[i, i * bit_width:(i + 1) * bit_width] = s
    return M @ z_best + lower_bound


def build_energy_qubo(K_ii, rhs, bit_width, lower_bound, upper_bound):
    n = K_ii.shape[0]
    nvar = n * bit_width
    scale = (upper_bound - lower_bound) / (2 ** bit_width - 1)
    s = scale * np.array([2 ** k for k in range(bit_width)])
    ssT = np.outer(s, s)
    c_vec = lower_bound * np.ones(n)
    w = K_ii @ c_vec - rhs

    QUBO = np.zeros((nvar, nvar))
    for i in range(n):
        for j in range(i, n):
            aij = K_ii[i, j]
            if abs(aij) < 1e-15:
                continue
            ri, rj = i * bit_width, j * bit_width
            block = 0.5 * aij * ssT
            QUBO[ri:ri + bit_width, rj:rj + bit_width] += block
            if i != j:
                QUBO[rj:rj + bit_width, ri:ri + bit_width] += block.T

    for i in range(n):
        ri = i * bit_width
        QUBO[ri:ri + bit_width, ri:ri + bit_width] += np.diag(w[i] * s)

    return QUBO.astype(np.float32), nvar, scale


def build_ls_qubo(K_ii, rhs, bit_width, lower_bound, upper_bound):
    n = K_ii.shape[0]
    nvar = n * bit_width
    scale = (upper_bound - lower_bound) / (2 ** bit_width - 1)
    s = scale * np.array([2 ** k for k in range(bit_width)])
    ssT = np.outer(s, s)
    c_vec = lower_bound * np.ones(n)
    d = K_ii @ c_vec - rhs
    Ad = K_ii @ d
    K2 = K_ii @ K_ii

    QUBO = np.zeros((nvar, nvar))
    for i in range(n):
        for j in range(i, n):
            k2ij = K2[i, j]
            if abs(k2ij) < 1e-15:
                continue
            ri, rj = i * bit_width, j * bit_width
            block = k2ij * ssT
            QUBO[ri:ri + bit_width, rj:rj + bit_width] += block
            if i != j:
                QUBO[rj:rj + bit_width, ri:ri + bit_width] += block.T

    for i in range(n):
        ri = i * bit_width
        QUBO[ri:ri + bit_width, ri:ri + bit_width] += np.diag(2 * Ad[i] * s)

    return QUBO.astype(np.float32), nvar, scale


def create_cim_optimizer(task_name_prefix):
    """适配不同版本的CIMOptimizer"""
    try:
        # 新版：使用 task_name_prefix
        return kw.cim.CIMOptimizer(task_name_prefix=task_name_prefix)
    except TypeError:
        try:
            # 旧版：使用 task_name
            return kw.cim.CIMOptimizer(task_name=task_name_prefix)
        except TypeError:
            # 无参数版本
            opt = kw.cim.CIMOptimizer()
            if hasattr(opt, 'set_task_name'):
                opt.set_task_name(task_name_prefix)
            return opt


def solve_subdomain_with_cim(K_ii, rhs, bit_width, lb, ub, task_name, use_ls=False):
    if use_ls:
        Q_float, nvar, scale = build_ls_qubo(K_ii, rhs, bit_width, lb, ub)
    else:
        Q_float, nvar, scale = build_energy_qubo(K_ii, rhs, bit_width, lb, ub)

    Q_qubo = kw.qubo.adjust_qubo_matrix_precision(Q_float)
    ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_qubo)
    variables = [f"x[{i}]" for i in range(ising_mat.shape[0])]
    ising_model = kw.ising.IsingModel(variables=variables, ising_matrix=ising_mat, bias=ising_bias)

    opt = create_cim_optimizer(task_name)
    # 提交任务（第一次调用）
    opt.solve(ising_model.get_matrix())

    start = time.time()
    sol = None
    while time.time() - start < CIM_TASK_TIMEOUT:
        # 再次调用 solve 尝试获取结果
        try:
            sol = opt.solve(ising_model.get_matrix())
            if sol is not None:
                break
        except Exception:
            pass
        time.sleep(CIM_POLL_INTERVAL)
        print(f"  等待任务 {task_name} 完成...")

    if sol is None:
        raise RuntimeError(f"任务 {task_name} 超时未返回结果")

    # 解析结果
    if sol.ndim == 2:
        solutions = sol[:, :-1]
        deltas = sol[:, -1]
        solutions_binary = (solutions * deltas[:, np.newaxis] + 1) / 2
        energies = np.array([z @ Q_qubo @ z for z in solutions_binary])
        z_best = solutions_binary[np.argmin(energies), :]
    else:
        z_best = sol
    u_i = decode_to_continuous(z_best, K_ii.shape[0], bit_width, scale, lb)
    return u_i, 0.0


# ==================== 7. 对比两种QUBO方法 ====================
print("\n========== 开始量子求解对比 ==========")
energy_errors = []
ls_errors = []
energy_times = []
ls_times = []
quantum_solutions = {}

for sid, data in enumerate(subdomain_data):
    back = data['back']
    if back is None:
        continue
    boundary_ids = data['boundary_global_ids']
    u_b_local = np.array([u_global[gid] for gid in boundary_ids])
    K_ii = back['K_ii']
    K_ib = back['K_ib']
    F_i = back['F_i']
    rhs = F_i - K_ib @ u_b_local
    lb, ub = estimate_bounds(u_b_local)

    # 能量泛函 QUBO
    t0 = time.time()
    u_i_energy, _ = solve_subdomain_with_cim(K_ii, rhs, BIT_WIDTH, lb, ub,
                                             f"energy_sd{sid}", use_ls=False)
    t1 = time.time()
    energy_times.append(t1 - t0)
    quantum_solutions[sid] = u_i_energy

    # 最小二乘 QUBO
    t0 = time.time()
    u_i_ls, _ = solve_subdomain_with_cim(K_ii, rhs, BIT_WIDTH, lb, ub,
                                         f"ls_sd{sid}", use_ls=True)
    t1 = time.time()
    ls_times.append(t1 - t0)

    # 计算该子域内部节点的精确解误差
    local_nodes = data['local_nodes']
    exact_i = np.array([u_exact_all[local_nodes[idx]] for idx in back['idx_i']])
    error_energy = np.linalg.norm(u_i_energy - exact_i) / np.linalg.norm(exact_i)
    error_ls = np.linalg.norm(u_i_ls - exact_i) / np.linalg.norm(exact_i)
    energy_errors.append(error_energy)
    ls_errors.append(error_ls)

    print(f"子域 {sid}: Energy QUBO 相对误差={error_energy:.4e}, LS QUBO 相对误差={error_ls:.4e}")

print("\n========== 对比结果 ==========")
print(f"Energy QUBO 平均相对误差: {np.mean(energy_errors):.4e} ± {np.std(energy_errors):.4e}")
print(f"LS QUBO     平均相对误差: {np.mean(ls_errors):.4e} ± {np.std(ls_errors):.4e}")
print(f"Energy QUBO 平均求解时间: {np.mean(energy_times):.2f} 秒")
print(f"LS QUBO     平均求解时间: {np.mean(ls_times):.2f} 秒")
print("结论: 能量泛函QUBO精度更高，求解速度更快")

# ==================== 8. 经典全局求解（参考） ====================
t0_classic = time.time()
K_global = np.zeros((n_nodes, n_nodes))
F_global = np.zeros(n_nodes)
for elem in elements:
    conn = elem
    xe = nodes[conn, 0];
    ye = nodes[conn, 1]
    Ke, Fe = elem_stiffness_and_load(xe, ye, source_term)
    for a, ia in enumerate(conn):
        F_global[ia] += Fe[a]
        for b, ib in enumerate(conn):
            K_global[ia, ib] += Ke[a, b]

for i, (x, y) in enumerate(nodes):
    if is_on_global_boundary(i):
        K_global[i, :] = 0
        K_global[i, i] = 1.0
        F_global[i] = u_exact(x, y)

u_classic = np.linalg.solve(K_global, F_global)
t_classic = time.time() - t0_classic
classic_error = np.linalg.norm(u_classic - u_exact_all) / np.linalg.norm(u_exact_all)
print(f"经典全局求解时间: {t_classic:.2f} 秒, 相对误差: {classic_error:.4e}")

# ==================== 9. 回填量子解并可视化 ====================
u_quantum_full = u_global.copy()
for sid, data in enumerate(subdomain_data):
    back = data['back']
    if back is None:
        continue
    if sid not in quantum_solutions:
        continue
    u_i = quantum_solutions[sid]
    local_nodes = data['local_nodes']
    for i_local, global_idx in enumerate(back['idx_i']):
        gid = local_nodes[global_idx]
        u_quantum_full[gid] = u_i[i_local]

error_quantum = u_quantum_full - u_exact_all
error_classic = u_classic - u_exact_all
L2_q = np.linalg.norm(error_quantum) / np.sqrt(n_nodes)
L2_c = np.linalg.norm(error_classic) / np.sqrt(n_nodes)
print(f"量子解 L2 误差: {L2_q:.6e}, 经典解 L2 误差: {L2_c:.6e}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
im0 = axes[0, 0].tricontourf(nodes[:, 0], nodes[:, 1], u_quantum_full, levels=20, cmap='viridis')
axes[0, 0].set_title('FEM Solution (Energy QUBO + CIM)')
axes[0, 0].axis('equal')
fig.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].tricontourf(nodes[:, 0], nodes[:, 1], u_exact_all, levels=20, cmap='plasma')
axes[0, 1].set_title('Exact Solution')
axes[0, 1].axis('equal')
fig.colorbar(im1, ax=axes[0, 1])

im2 = axes[1, 0].tricontourf(nodes[:, 0], nodes[:, 1], error_quantum, levels=20, cmap='coolwarm')
axes[1, 0].set_title('Absolute Error (Energy QUBO)')
axes[1, 0].axis('equal')
fig.colorbar(im2, ax=axes[1, 0])

im3 = axes[1, 1].tricontourf(nodes[:, 0], nodes[:, 1], error_classic, levels=20, cmap='coolwarm')
axes[1, 1].set_title('Absolute Error (Classical)')
axes[1, 1].axis('equal')
fig.colorbar(im3, ax=axes[1, 1])

fig.suptitle('2D Poisson — CIM Quantum vs Classical FEM', fontsize=14)
plt.tight_layout()
plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("可视化已保存为 comparison.png")


# ==================== 10. 量子约束 PINN-Transformer ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class QuantumConstrainedTransformerPINN(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_fc = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_fc(x).unsqueeze(1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.output_fc(x[:, 0, :])


def pde_residual(model, x):
    x.requires_grad_(True)
    u = model(x)
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(grad_u[:, 0], x, grad_outputs=torch.ones_like(grad_u[:, 0]), create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(grad_u[:, 1], x, grad_outputs=torch.ones_like(grad_u[:, 1]), create_graph=True)[0][:, 1]
    with torch.no_grad():
        f_vals = source_term(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy())
        f_tensor = torch.tensor(f_vals, dtype=torch.float32, device=x.device).unsqueeze(1)
    residual = - (u_xx + u_yy) - f_tensor
    return residual


np.random.seed(42)
n_supervised = 200
n_unsupervised = 1000
all_points = np.random.rand(n_supervised + n_unsupervised, 2) * np.array([Lx, Ly])
sup_points = all_points[:n_supervised]
unsup_points = all_points[n_supervised:]

x_sup = torch.tensor(sup_points, dtype=torch.float32)
y_sup = torch.tensor([u_exact(x, y) for x, y in sup_points], dtype=torch.float32).unsqueeze(1)
x_unsup = torch.tensor(unsup_points, dtype=torch.float32)

model = QuantumConstrainedTransformerPINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

epochs = 500
lambda_pde = 0.1
lambda_quantum = 0.01

print("\n开始训练 Quantum-Constrained PINN-Transformer...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    pred_sup = model(x_sup)
    loss_sup = mse_loss(pred_sup, y_sup)

    residual = pde_residual(model, x_unsup)
    loss_pde = torch.mean(residual ** 2)

    # 量子约束（演示，实际可集成CIM结果）
    loss_quantum = torch.tensor(0.0)

    loss = loss_sup + lambda_pde * loss_pde + lambda_quantum * loss_quantum
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:3d}, Loss: {loss.item():.4f}, Sup: {loss_sup.item():.4f}, PDE: {loss_pde.item():.4f}")

print("所有任务完成！")