# -*- coding: utf-8 -*-
"""
SA实验：PDE线性系统的模拟退火求解
用于论文跨平台验证 —— SA vs Kaiwu CIM vs WuYue VQE
"""
import sys
sys.path.insert(0, r"D:\QPDE\模拟退火")

import numpy as np
import json
import os
import time
from sa_utils import QUBOEnergy, simulated_annealing

OUTPUT_DIR = r"C:\Users\username\Desktop\琶洲算法大赛\队伍2\sa_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_fem_poisson(nx, ny, Lx=3.0, Ly=2.0):
    """5点差分构造2D泊松方程刚度矩阵（与VQE实验完全一致）"""
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
    """构造不定矩阵（与VQE实验完全一致）"""
    eigs = np.linalg.eigvalsh(K_ii)
    lam_min, lam_max = eigs[0], eigs[-1]
    alpha = lam_min + alpha_frac * (lam_max - lam_min)
    A = K_ii - alpha * np.eye(len(K_ii))
    return A, alpha


def make_rhs(n_i, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n_i)


def run_sa_experiment(A, b_vec, p, bits, label, n_steps=300000):
    """运行单次SA实验"""
    n = A.shape[0]
    kappa = np.linalg.cond(A)

    print(f"\n  [{label}] n={n}, p={p}, bits={bits}, κ(A)={kappa:.1f}")
    t0 = time.time()

    qubo = QUBOEnergy(A, b_vec, p=p, bits=bits, lb=-2.0, ub=2.0, quantize_bits=8)
    z_opt, energy = simulated_annealing(qubo, n_steps=n_steps, cooling=0.99995, verbose=False)
    elapsed = time.time() - t0

    x_sa = qubo.decode(z_opt)
    x_exact = qubo.x_ref
    rmse = np.sqrt(np.mean((x_sa - x_exact)**2))
    b_eff = -np.log2(rmse) if rmse > 1e-16 else float(bits)
    rel_err = np.linalg.norm(x_sa - x_exact) / np.linalg.norm(x_exact)

    result = {
        'label': label,
        'p': p,
        'n': n,
        'n_qubits': qubo.N,
        'bits': bits,
        'cond_A': float(qubo.cond_A),
        'cond_C': float(qubo.cond_C),
        'rmse': float(rmse),
        'b_eff': float(b_eff),
        'rel_error': float(rel_err),
        'energy': float(energy),
        'time': elapsed,
        'n_steps': n_steps,
    }

    print(f"    κ(C_p{p})={qubo.cond_C:.1f}, RMSE={rmse:.4f}, b_eff={b_eff:.2f}, time={elapsed:.2f}s")
    return result


def main():
    print("=" * 60)
    print("SA实验 — PDE线性系统模拟退火求解")
    print("=" * 60)

    all_results = {}
    N_STEPS = 300000

    # ---- Exp 1: SPD 4x4 grid, SAC vs LS ----
    print("\n[Exp1] SPD 4x4 grid")
    K_ii = make_fem_poisson(4, 4)
    f = make_rhs(K_ii.shape[0])

    r_sac = run_sa_experiment(K_ii, f, p=1, bits=8, label='spd_4x4_SAC', n_steps=N_STEPS)
    r_ls = run_sa_experiment(K_ii, f, p=2, bits=8, label='spd_4x4_LS', n_steps=N_STEPS)
    all_results['spd_4x4'] = {'SAC': r_sac, 'LS': r_ls}

    # ---- Exp 2: SPD 5x5 grid, SAC vs LS ----
    print("\n[Exp2] SPD 5x5 grid")
    K_ii5 = make_fem_poisson(5, 5)
    f5 = make_rhs(K_ii5.shape[0])

    r_sac5 = run_sa_experiment(K_ii5, f5, p=1, bits=8, label='spd_5x5_SAC', n_steps=N_STEPS)
    r_ls5 = run_sa_experiment(K_ii5, f5, p=2, bits=8, label='spd_5x5_LS', n_steps=N_STEPS)
    all_results['spd_5x5'] = {'SAC': r_sac5, 'LS': r_ls5}

    # ---- Exp 3: Indefinite 4x4 grid ----
    print("\n[Exp3] Indefinite 4x4 grid")
    K_ii_indef = make_fem_poisson(4, 4)
    A_indef, alpha_shift = make_indefinite_matrix(K_ii_indef, alpha_frac=0.3)
    f_indef = make_rhs(K_ii_indef.shape[0], seed=123)

    r_indef_sac = run_sa_experiment(A_indef, f_indef, p=1, bits=8, label='indef_4x4_SAC', n_steps=N_STEPS)
    r_indef_ls = run_sa_experiment(A_indef, f_indef, p=2, bits=8, label='indef_4x4_LS', n_steps=N_STEPS)
    all_results['indef_4x4'] = {'SAC': r_indef_sac, 'LS': r_indef_ls}

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SA Results Summary")
    print(f"{'='*60}")
    for exp_name, methods in all_results.items():
        for method_name, r in methods.items():
            print(f"  {exp_name} {method_name}: κ(A)={r['cond_A']:.1f}, κ(C)={r['cond_C']:.1f}, "
                  f"RMSE={r['rmse']:.4f}, b_eff={r['b_eff']:.2f}, time={r['time']:.2f}s")

    # Save results
    out_path = os.path.join(OUTPUT_DIR, 'sa_results.json')
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(all_results, fp, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")

    return all_results


if __name__ == '__main__':
    main()
