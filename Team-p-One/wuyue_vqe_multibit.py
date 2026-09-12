# -*- coding: utf-8 -*-
"""
WuYue SDK VQE — 多比特二进制编码版
修复单比特编码只能取±α的问题：每个连续变量用q个qubit的二进制展开编码
x_i = lb + alpha * sum_k(2^k * z_{i,k}), z ∈ {0,1}
"""
import sys
sys.path.insert(0, r"C:\Users\username\Desktop\琶洲算法大赛\论文\WuYueSDK")

import numpy as np
from scipy.optimize import minimize
import json
import os
import time

from wuyue.utils.ml_backend import set_ml_backend
set_ml_backend("numpy")

from wuyue.programe import QuantumProg
from wuyue.register.quantumregister import QuantumRegister
from wuyue.register.classicalregister import ClassicalRegister
from wuyue.backend.backend import Backend
from wuyue.element.gate import RY, CNOT

OUTPUT_DIR = r"C:\Users\username\Desktop\琶洲算法大赛\队伍2\wuyue_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. FEM矩阵构造（与之前一致）
# ============================================================

def make_fem_poisson(nx, ny, Lx=3.0, Ly=2.0):
    n = nx * ny
    hx = Lx / (nx - 1)
    hy = Ly / (ny - 1)
    K = np.zeros((n, n))
    boundary = np.zeros(n, dtype=bool)

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary[idx] = True
                K[idx, idx] = 1.0
            else:
                K[idx, idx] = 2.0 / hx**2 + 2.0 / hy**2
                K[idx, idx + 1] = -1.0 / hx**2
                K[idx, idx - 1] = -1.0 / hx**2
                K[idx, idx + nx] = -1.0 / hy**2
                K[idx, idx - nx] = -1.0 / hy**2

    interior = ~boundary
    K_ii = K[np.ix_(interior, interior)]
    return K_ii


def make_indefinite_matrix(K_ii, alpha_frac=0.3):
    eigs = np.linalg.eigvalsh(K_ii)
    lam_min, lam_max = eigs[0], eigs[-1]
    alpha = lam_min + alpha_frac * (lam_max - lam_min)
    A = K_ii - alpha * np.eye(len(K_ii))
    return A, alpha


def make_rhs(n_i, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n_i)


# ============================================================
# 2. 多比特QUBO编码 + Ising哈密顿量
# ============================================================

def build_qubo_multibit(K, f, bits=3, lb=-2.0, ub=2.0):
    """
    多比特二进制编码的QUBO矩阵构造
    x_i = lb + alpha * sum_k(2^k * z_{i,k}), z_{i,k} ∈ {0,1}
    E(x) = 1/2 x^T K x - x^T f → E(z) = z^T Q z + const
    """
    n = K.shape[0]
    alpha = (ub - lb) / (2**bits - 1)
    s = alpha * np.array([2**k for k in range(bits)])  # [α, 2α, 4α, ...]

    # M = I_n ⊗ s^T, x = lb·1 + Mz
    # Q_quad = 1/2 M^T K M
    ssT = np.outer(s, s)
    Q_quad = 0.5 * np.kron(K, ssT)

    # Q_lin: diag of linear coefficient
    v = lb * K @ np.ones(n) - f
    Q_lin = np.diag(np.kron(v, s))

    Q = Q_quad + Q_lin

    return Q, alpha, s, n


def qubo_to_ising_hamiltonian(Q):
    """
    将QUBO矩阵Q转换为Ising哈密顿量的Pauli-Z表示
    z_i = (1 - σ_i)/2, σ_i ∈ {+1, -1}
    H = Σ_i h_i Z_i + Σ_{i<j} J_{ij} Z_i Z_j + const
    """
    N = Q.shape[0]
    # QUBO → Ising参数
    # E = z^T Q z, z ∈ {0,1}, z = (1-σ)/2
    # J_ij = Q_ij / 4
    # h_i = -(Q_ii/2 + Σ_{j≠i} Q_ij / 4 + Σ_{j≠i} Q_ji / 4) = -(Σ_j Q_ij + Σ_j Q_ji) / 4
    # For symmetric Q: h_i = -Σ_j Q_ij / 2

    J = Q / 4.0  # J_ij = Q_ij/4
    h = -0.5 * np.sum(Q, axis=1)  # h_i = -Σ_j Q_ij / 2

    return h, J


def build_hamiltonian_diag(h, J):
    """构造哈密顿量对角元（仅用于小规模精确验证）"""
    N = len(h)
    dim = 2 ** N
    diag = np.zeros(dim, dtype=np.float64)

    for i in range(N):
        pattern_i = np.tile(np.repeat([1, -1], 2**i), 2**(N - i - 1))
        diag += h[i] * pattern_i

    for i in range(N):
        for j in range(i + 1, N):
            if abs(J[i, j]) > 1e-15:
                pattern_i = np.tile(np.repeat([1, -1], 2**i), 2**(N - i - 1))
                pattern_j = np.tile(np.repeat([1, -1], 2**j), 2**(N - j - 1))
                diag += J[i, j] * pattern_i * pattern_j

    return diag


def compute_energy_from_state(state, h, J):
    """
    直接从态矢量计算能量，避免构造哈密顿量矩阵
    ⟨ψ|H|ψ⟩ = Σ_i h_i ⟨Z_i⟩ + Σ_{i<j} J_{ij} ⟨Z_i Z_j⟩
    """
    N = len(h)
    n_qubits = int(np.log2(len(state)))
    prob = np.real(state.conj() * state)

    # ⟨Z_i⟩
    zi_exp = np.zeros(N)
    for i in range(N):
        pattern = np.tile(np.repeat([1, -1], 2**i), 2**(n_qubits - i - 1))
        zi_exp[i] = np.dot(prob, pattern)

    energy = np.dot(h, zi_exp)

    # ⟨Z_i Z_j⟩
    for i in range(N):
        for j in range(i + 1, N):
            if abs(J[i, j]) > 1e-15:
                pattern_i = np.tile(np.repeat([1, -1], 2**i), 2**(n_qubits - i - 1))
                pattern_j = np.tile(np.repeat([1, -1], 2**j), 2**(n_qubits - j - 1))
                zij_exp = np.dot(prob, pattern_i * pattern_j)
                energy += J[i, j] * zij_exp

    return energy


def compute_gs_energy_exact(h, J):
    """通过穷举搜索计算精确基态能量（仅适用于≤16 qubits的小规模验证）"""
    N = len(h)
    if N > 16:
        return None  # 太大，无法穷举

    best = float('inf')
    for idx in range(2**N):
        bits = np.array([(idx >> i) & 1 for i in range(N)])
        spins = 1 - 2 * bits  # {0,1} → {+1,-1}
        energy = np.dot(h, spins)
        for i in range(N):
            for j in range(i + 1, N):
                if abs(J[i, j]) > 1e-15:
                    energy += J[i, j] * spins[i] * spins[j]
        if energy < best:
            best = energy
    return best


# ============================================================
# 3. VQE电路与优化
# ============================================================

def build_vqe_circuit(n_qubits, params, n_layers=3):
    """硬件高效变分拟设: RY层 + 环形CNOT + 最终RY层"""
    qubit = QuantumRegister(n_qubits)
    cbit = ClassicalRegister(n_qubits)
    prog = QuantumProg(qubit, cbit)

    idx = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            prog.add(RY, qubit[i], paras=float(params[idx]))
            idx += 1
        for i in range(n_qubits):
            prog.add(CNOT, qubit[i], qubit[(i + 1) % n_qubits])
    for i in range(n_qubits):
        prog.add(RY, qubit[i], paras=float(params[idx]))
        idx += 1

    return prog


def get_state_vector(prog):
    sim = Backend.get_device(device_name="Full amplitude")
    sim.apply(prog)
    state = sim.get_states()
    sim.clear()
    return state


def vqe_expectation(params, n_qubits, n_layers, h, J):
    """计算 ⟨ψ(θ)|H|ψ(θ)⟩ = Σ_i h_i⟨Z_i⟩ + Σ_{i<j} J_{ij}⟨Z_i Z_j⟩"""
    params_list = [float(p) for p in params]

    try:
        prog = build_vqe_circuit(n_qubits, params_list, n_layers)
        state = get_state_vector(prog)
        return float(compute_energy_from_state(state, h, J))
    except Exception as e:
        return 1e10


def decode_multibit(params, n_qubits, n_layers, bits, n_nodes, lb, alpha, s):
    """从VQE最优参数解码连续解向量"""
    params_list = [float(p) for p in params]
    prog = build_vqe_circuit(n_qubits, params_list, n_layers)
    state = get_state_vector(prog)

    # 计算每个qubit的⟨Z_i⟩
    prob = np.abs(state) ** 2
    z_expectations = np.zeros(n_qubits)
    for i in range(n_qubits):
        pattern = np.tile(np.repeat([1, -1], 2**i), 2**(n_qubits - i - 1))
        z_expectations[i] = np.dot(prob, pattern)  # ⟨Z_i⟩ ∈ [-1, 1]

    # z_i = (1 - ⟨Z_i⟩) / 2 → {0,1}概率值
    z_vals = (1.0 - z_expectations) / 2.0  # ∈ [0, 1]
    z_vals = z_vals.reshape(n_nodes, bits)

    # x_i = lb + alpha * Σ_k 2^k z_{i,k}
    x = lb + alpha * np.dot(z_vals, 2.0 ** np.arange(bits))
    return x


def decode_multibit_hard(params, n_qubits, n_layers, bits, n_nodes, lb, alpha, s):
    """硬解码：取概率最大的计算基态"""
    params_list = [float(p) for p in params]
    prog = build_vqe_circuit(n_qubits, params_list, n_layers)
    state = get_state_vector(prog)

    # 找概率最大的计算基态
    prob = np.abs(state) ** 2
    best_idx = np.argmax(prob)
    z_bits = np.array([(best_idx >> i) & 1 for i in range(n_qubits)])
    z_vals = z_bits.reshape(n_nodes, bits)

    x = lb + alpha * np.dot(z_vals, 2.0 ** np.arange(bits))
    return x


def run_vqe(h, J, n_qubits, n_layers=3, n_restarts=5):
    """运行VQE优化（直接使用Ising参数，避免构造H矩阵）"""
    n_params = n_qubits * (n_layers + 1)

    best_result = None
    best_energy = float('inf')
    history = []

    for restart in range(n_restarts):
        rng = np.random.RandomState(42 + restart)
        init_params = rng.uniform(0, 2 * np.pi, n_params)

        t0 = time.time()
        result = minimize(
            lambda p: vqe_expectation(p, n_qubits, n_layers, h, J),
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'disp': False}
        )
        elapsed = time.time() - t0

        history.append({
            'restart': restart,
            'energy': float(result.fun),
            'nfev': int(result.nfev),
            'time': elapsed
        })

        print(f"  Restart {restart+1}: energy={result.fun:.6f}, nfev={result.nfev}, time={elapsed:.2f}s")

        if result.fun < best_energy:
            best_energy = result.fun
            best_result = result

    return best_result, history


# ============================================================
# 4. 主实验
# ============================================================

def run_experiment(name, K, f, bits=3, lb=-2.0, ub=2.0, n_layers=3):
    """运行多比特VQE实验"""
    n_nodes = K.shape[0]
    n_qubits = n_nodes * bits
    dim = 2 ** n_qubits
    kappa = np.linalg.cond(K)

    print(f"\n{'='*60}")
    print(f"Experiment: {name} (MULTI-BIT ENCODING)")
    print(f"  n_nodes={n_nodes}, bits={bits}, n_qubits={n_qubits}, dim={dim}")
    print(f"  κ(A)={kappa:.2f}, n_layers={n_layers}")
    print(f"{'='*60}")

    x_exact = np.linalg.solve(K, f)

    # 构造多比特QUBO
    Q, alpha, s, _ = build_qubo_multibit(K, f, bits=bits, lb=lb, ub=ub)

    # QUBO → Ising 哈密顿量
    h, J = qubo_to_ising_hamiltonian(Q)
    print(f"  QUBO vars={Q.shape[0]}, Ising h range=[{h.min():.2f}, {h.max():.2f}]")

    # 精确基态能量（穷举搜索，≤16 qubits可行）
    exact_gs = compute_gs_energy_exact(h, J)
    if exact_gs is None:
        print(f"  WARNING: {n_qubits} qubits too large for exact GS, using Ising estimate")
        # 用最小Ising能量的启发式估计
        exact_gs = np.sum(h) - np.sum(np.abs(h))  # 粗略下界

    # 运行VQE
    t0 = time.time()
    result, history = run_vqe(h, J, n_qubits, n_layers=n_layers, n_restarts=5)
    total_time = time.time() - t0

    # 解码解向量（两种方式）
    x_vqe_soft = decode_multibit(result.x, n_qubits, n_layers, bits, n_nodes, lb, alpha, s)
    x_vqe_hard = decode_multibit_hard(result.x, n_qubits, n_layers, bits, n_nodes, lb, alpha, s)

    # 评估
    rmse_soft = np.sqrt(np.mean((x_vqe_soft - x_exact)**2))
    rmse_hard = np.sqrt(np.mean((x_vqe_hard - x_exact)**2))
    rel_err = np.linalg.norm(x_vqe_hard - x_exact) / np.linalg.norm(x_exact)

    print(f"\n  Exact GS energy: {exact_gs:.6f}")
    print(f"  VQE found energy: {result.fun:.6f}")
    print(f"  Energy gap: {result.fun - exact_gs:.2e}")
    print(f"  RMSE (soft decode): {rmse_soft:.6f}")
    print(f"  RMSE (hard decode): {rmse_hard:.6f}")
    print(f"  Rel error: {rel_err:.6f}")
    print(f"  Total time: {total_time:.2f}s")

    results = {
        'name': name,
        'encoding': f'multibit_{bits}bits',
        'n_nodes': n_nodes,
        'n_qubits': n_qubits,
        'bits': bits,
        'kappa': float(kappa),
        'n_layers': n_layers,
        'vqe_energy': float(result.fun),
        'exact_gs_energy': float(exact_gs),
        'energy_error': float(result.fun - exact_gs),
        'rmse_soft': float(rmse_soft),
        'rmse_hard': float(rmse_hard),
        'rel_error': float(rel_err),
        'total_time': total_time,
        'nfev_total': sum(h['nfev'] for h in history),
        'history': history,
        'x_vqe': x_vqe_hard.tolist(),
        'x_exact': x_exact.tolist(),
    }

    out_path = os.path.join(OUTPUT_DIR, f'vqe_{name}_multibit.json')
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    return results


def main():
    print("="*60)
    print("WuYue SDK VQE — 多比特二进制编码")
    print("="*60)

    all_results = {}

    # ---- Exp 1: SPD 4×4, 3-bit encoding (12 qubits, dim=4096) ----
    K_ii = make_fem_poisson(4, 4)
    f = make_rhs(K_ii.shape[0])
    print(f"\n[Exp1] SPD 4×4 grid, 3-bit → 12 qubits, dim=4096")
    all_results['spd_4x4_3bit'] = run_experiment(
        'spd_4x4_3bit', K_ii, f, bits=3, lb=-2.0, ub=2.0, n_layers=5)

    # ---- Exp 2: SPD 4×4, 4-bit encoding (16 qubits, dim=65536) ----
    print(f"\n[Exp2] SPD 4×4 grid, 4-bit → 16 qubits, dim=65536")
    all_results['spd_4x4_4bit'] = run_experiment(
        'spd_4x4_4bit', K_ii, f, bits=4, lb=-2.0, ub=2.0, n_layers=6)

    # ---- Exp 3: Indefinite 4×4, 3-bit encoding ----
    K_ii3 = make_fem_poisson(4, 4)
    A_indef, _ = make_indefinite_matrix(K_ii3, alpha_frac=0.3)
    f3 = make_rhs(K_ii3.shape[0], seed=123)
    print(f"\n[Exp3] Indefinite 4×4 grid, 3-bit → 12 qubits")
    all_results['indef_4x4_3bit'] = run_experiment(
        'indef_4x4_3bit', A_indef, f3, bits=3, lb=-3.0, ub=3.0, n_layers=5)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Multi-bit VQE Summary:")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(f"  {name}: κ={res['kappa']:.1f}, qubits={res['n_qubits']}, "
              f"RMSE(soft)={res['rmse_soft']:.4f}, RMSE(hard)={res['rmse_hard']:.4f}, "
              f"energy_err={res['energy_error']:.2e}, time={res['total_time']:.1f}s")

    summary = {name: {'kappa': r['kappa'], 'n_qubits': r['n_qubits'],
                      'rmse_soft': r['rmse_soft'], 'rmse_hard': r['rmse_hard'],
                      'energy_error': r['energy_error'], 'time': r['total_time']}
               for name, r in all_results.items()}
    summary_path = os.path.join(OUTPUT_DIR, 'vqe_multibit_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    return all_results


if __name__ == '__main__':
    main()
