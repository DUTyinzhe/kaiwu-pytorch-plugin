# -*- coding: utf-8 -*-
"""
WuYue SDK VQE实验：求解PDE线性系统的变分量子本征求解器
用于跨技术路线验证 —— Kaiwu CIM vs WuYue VQE vs 经典

对2D泊松方程FEM离散得到的SPD矩阵和不定矩阵，
使用WuYue SDK构建变分量子线路，通过VQE求解基态。

Author: Team 2
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
# 1. FEM矩阵构造
# ============================================================

def make_fem_poisson(nx, ny, Lx=3.0, Ly=2.0):
    """5点差分构造2D泊松方程的刚度矩阵（Dirichlet边界）"""
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
    return K_ii, interior, boundary


def make_indefinite_matrix(K_ii, alpha_frac=0.3):
    """构造不定矩阵 A = K - alpha*I, alpha在特征值范围内"""
    eigs = np.linalg.eigvalsh(K_ii)
    lam_min, lam_max = eigs[0], eigs[-1]
    alpha = lam_min + alpha_frac * (lam_max - lam_min)
    A = K_ii - alpha * np.eye(len(K_ii))
    return A, alpha


def make_rhs(n_i, seed=42):
    """生成随机右端项"""
    rng = np.random.RandomState(seed)
    return rng.randn(n_i)


# ============================================================
# 2. Pauli算符与哈密顿量构造
# ============================================================

def pauli_z_matrix(i, n_qubits):
    """构造 Z_i = I⊗...⊗Z⊗...⊗I 的矩阵表示"""
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    result = np.array([[1]], dtype=complex)
    for j in range(n_qubits):
        result = np.kron(result, Z if j == i else np.eye(2, dtype=complex))
    return result


def build_hamiltonian_matrix(K, f, alpha):
    """
    从能量泛函 E(x)=½xᵀKx-xᵀf 构造VQE哈密顿量
    编码: x_i = alpha * ⟨Z_i⟩
    H = ½alpha² Σ_ij K_ij Z_i Z_j - alpha Σ_i f_i Z_i
    """
    n_qubits = K.shape[0]
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    # 二次项: ½ alpha² Σ_ij K_ij Z_i Z_j
    for i in range(n_qubits):
        Zi = pauli_z_matrix(i, n_qubits)
        for j in range(n_qubits):
            if abs(K[i, j]) > 1e-12:
                Zj = pauli_z_matrix(j, n_qubits)
                H += 0.5 * alpha**2 * K[i, j] * (Zi @ Zj)

    # 线性项: -alpha Σ_i f_i Z_i
    for i in range(n_qubits):
        Zi = pauli_z_matrix(i, n_qubits)
        H -= alpha * f[i] * Zi

    return H


# ============================================================
# 3. VQE电路构建 (WuYue SDK)
# ============================================================

def build_vqe_circuit(n_qubits, params, n_layers=3):
    """
    构建硬件高效变分拟设: RY层 + 环形CNOT纠缠，重复n_layers次
    params: 每层n_qubits个RY角度
    """
    qubit = QuantumRegister(n_qubits)
    cbit = ClassicalRegister(n_qubits)
    prog = QuantumProg(qubit, cbit)

    param_idx = 0
    for layer in range(n_layers):
        # RY旋转层
        for i in range(n_qubits):
            prog.add(RY, qubit[i], paras=params[param_idx])
            param_idx += 1
        # 环形CNOT纠缠
        for i in range(n_qubits):
            prog.add(CNOT, qubit[i], qubit[(i + 1) % n_qubits])

    # 最终RY层
    for i in range(n_qubits):
        prog.add(RY, qubit[i], paras=params[param_idx])
        param_idx += 1

    return prog


def get_state_vector(prog):
    """运行电路并获取态矢量"""
    sim = Backend.get_device(device_name="Full amplitude")
    sim.apply(prog)
    state = sim.get_states()
    sim.clear()
    return state


# ============================================================
# 4. VQE优化
# ============================================================

def vqe_expectation(params, n_qubits, n_layers, H_matrix):
    """计算 ⟨ψ(θ)|H|ψ(θ)⟩ """
    # 将参数转为Python float列表 (WuYue SDK需要)
    params_list = [float(p) for p in params]

    try:
        prog = build_vqe_circuit(n_qubits, params_list, n_layers)
        state = get_state_vector(prog)
        energy = np.real(state.conj().T @ H_matrix @ state)
        return float(energy)
    except Exception as e:
        print(f"  VQE evaluation error: {e}")
        return 1e10


def decode_solution(params, n_qubits, n_layers, alpha):
    """从最优参数解码解向量 x_i = alpha * ⟨Z_i⟩"""
    params_list = [float(p) for p in params]
    prog = build_vqe_circuit(n_qubits, params_list, n_layers)
    state = get_state_vector(prog)

    x = np.zeros(n_qubits)
    for i in range(n_qubits):
        Zi = pauli_z_matrix(i, n_qubits)
        x[i] = alpha * np.real(state.conj().T @ Zi @ state)
    return x


def run_vqe(H_matrix, n_qubits, n_layers=3, alpha=2.0, n_restarts=3):
    """
    运行VQE求解
    n_restarts: 随机重启次数
    """
    n_params = n_qubits * (n_layers + 1)  # 每层n个RY + 最终n个RY

    best_result = None
    best_energy = float('inf')
    history = []

    for restart in range(n_restarts):
        # 随机初始化参数
        rng = np.random.RandomState(42 + restart)
        init_params = rng.uniform(0, 2 * np.pi, n_params)

        t0 = time.time()
        result = minimize(
            lambda p: vqe_expectation(p, n_qubits, n_layers, H_matrix),
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'disp': False}
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
# 5. 主实验
# ============================================================

def run_experiment(name, K_or_A, f, alpha=2.0, n_layers=3):
    """运行单个VQE实验并保存结果"""
    n_qubits = K_or_A.shape[0]
    kappa = np.linalg.cond(K_or_A)
    dim = 2 ** n_qubits

    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"  n_qubits={n_qubits}, dim={dim}, kappa={kappa:.2f}")
    print(f"  n_layers={n_layers}, alpha={alpha}")
    print(f"{'='*60}")

    # 精确解
    x_exact = np.linalg.solve(K_or_A, f)

    # 构造哈密顿量
    H_matrix = build_hamiltonian_matrix(K_or_A, f, alpha)

    # 计算精确基态能量（用于对比）
    exact_eigs = np.linalg.eigvalsh(H_matrix)
    exact_gs = exact_eigs[0]

    # 运行VQE
    t0 = time.time()
    result, history = run_vqe(
        H_matrix, n_qubits, n_layers=n_layers, alpha=alpha, n_restarts=5
    )
    total_time = time.time() - t0

    # 解码解向量
    x_vqe = decode_solution(result.x, n_qubits, n_layers, alpha)

    # 评估
    rmse_vqe = np.sqrt(np.mean((x_vqe - x_exact)**2))
    rel_err = np.linalg.norm(x_vqe - x_exact) / np.linalg.norm(x_exact)

    print(f"\n  Exact GS energy: {exact_gs:.8f}")
    print(f"  VQE found energy: {result.fun:.8f}")
    print(f"  Energy gap: {result.fun - exact_gs:.2e}")
    print(f"  RMSE vs exact: {rmse_vqe:.6f}")
    print(f"  Rel error: {rel_err:.6f}")
    print(f"  Total VQE time: {total_time:.2f}s")

    # 保存结果
    results = {
        'name': name,
        'n_qubits': n_qubits,
        'kappa': float(kappa),
        'n_layers': n_layers,
        'alpha': alpha,
        'vqe_energy': float(result.fun),
        'exact_gs_energy': float(exact_gs),
        'energy_error': float(result.fun - exact_gs),
        'rmse': float(rmse_vqe),
        'rel_error': float(rel_err),
        'total_time': total_time,
        'nfev_total': sum(h['nfev'] for h in history),
        'history': history,
        'x_vqe': x_vqe.tolist(),
        'x_exact': x_exact.tolist(),
    }

    # 写入JSON
    out_path = os.path.join(OUTPUT_DIR, f'vqe_{name}.json')
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    return results


def main():
    print("="*60)
    print("WuYue SDK VQE — PDE线性系统跨平台验证实验")
    print("="*60)

    all_results = {}

    # ---- Exp 1: SPD矩阵 (4×4网格, 4内部节点, 4 qubits) ----
    K_ii, interior, _ = make_fem_poisson(4, 4)
    n_i = K_ii.shape[0]
    f = make_rhs(n_i)
    print(f"\n[Exp1 SPD] Grid 4x4, interior nodes={n_i}")
    all_results['spd_4x4'] = run_experiment('spd_4x4', K_ii, f, alpha=2.0, n_layers=3)

    # ---- Exp 2: SPD矩阵 (5×5网格, 9内部节点, 9 qubits) ----
    K_ii2, interior2, _ = make_fem_poisson(5, 5)
    n_i2 = K_ii2.shape[0]
    f2 = make_rhs(n_i2)
    print(f"\n[Exp2 SPD] Grid 5x5, interior nodes={n_i2}")
    all_results['spd_5x5'] = run_experiment('spd_5x5', K_ii2, f2, alpha=2.0, n_layers=4)

    # ---- Exp 3: 不定矩阵 (4×4网格) ----
    K_ii3, interior3, _ = make_fem_poisson(4, 4)
    A_indef, alpha_shift = make_indefinite_matrix(K_ii3, alpha_frac=0.3)
    f3 = make_rhs(K_ii3.shape[0], seed=123)
    print(f"\n[Exp3 Indefinite] Grid 4x4, kappa={np.linalg.cond(A_indef):.1f}")

    # 对于不定矩阵，需要更大的alpha因为解可能更大
    all_results['indef_4x4'] = run_experiment('indef_4x4', A_indef, f3, alpha=5.0, n_layers=4)

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for name, res in all_results.items():
        print(f"  {name}: κ={res['kappa']:.1f}, RMSE={res['rmse']:.4f}, "
              f"rel_err={res['rel_error']:.4f}, time={res['total_time']:.1f}s")

    # 保存汇总
    summary = {name: {'kappa': r['kappa'], 'rmse': r['rmse'],
                       'rel_error': r['rel_error'], 'time': r['total_time'],
                       'energy_error': r['energy_error']}
               for name, r in all_results.items()}
    summary_path = os.path.join(OUTPUT_DIR, 'vqe_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    return all_results


if __name__ == '__main__':
    main()
