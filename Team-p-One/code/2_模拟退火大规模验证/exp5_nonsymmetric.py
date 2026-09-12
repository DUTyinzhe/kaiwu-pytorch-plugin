"""
实验 5：非对称矩阵推广 — Normal equations 的 PMIC 降幂验证（穷举 + 量化对照）

对于非对称矩阵 A，求解 Ax = b：
  标准路线: normal equations → LS-QUBO → Hessian = (A^T A)² → p=4
  PMIC 路线: normal equations → 直接 C = A^T A     → p=2

理论: RMSE(LS) / RMSE(PMIC) ≈ κ(A)²

量化对照: float64（无量化）vs int8（模拟 Kaiwu CIM [-128,127]）
"""
import numpy as np
import json
from pathlib import Path
from sa_utils import QUBOEnergy, OUTPUT

OUTPUT_EXP = OUTPUT / "exp5_nonsymmetric"
OUTPUT_EXP.mkdir(parents=True, exist_ok=True)


def make_nonsymmetric(n, kappa, seed):
    """生成非对称矩阵 A = U Σ V^T, κ(A)=kappa"""
    rs = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rs.standard_normal((n, n)))
    V, _ = np.linalg.qr(rs.standard_normal((n, n)))
    sv = np.geomspace(kappa, 1.0, n)
    A = U @ np.diag(sv) @ V.T
    return A


def exhaustive_solve(qubo):
    """穷举 QUBO 基态，返回 (RMSE, b_eff)"""
    N = qubo.N
    E_best = np.inf
    z_best = None
    total = 1 << N
    for k in range(total):
        z = np.array([(k >> j) & 1 for j in range(N)], dtype=np.float64)
        E = qubo.energy(z)
        if E < E_best:
            E_best = E
            z_best = z.copy()
    rmse = qubo.compute_rmse(z_best)
    b_eff = -np.log2(max(rmse, 1e-16))
    return rmse, b_eff


# ============================================================
# 主实验
# ============================================================
N = 4
KAPPAS = [4, 8, 16]
BIT_DEPTHS = [3,5 ]          # b=6 对 n=4 需 16.8M 状态，太慢且效果差
QUANTIZE_MODES = [("float64", None), ("int8", 8)]
x_true = np.array([1.0, 0.5, -0.5, 0.3])

print("=" * 70)
print("  实验 5: 非对称矩阵 — RMSE 比值验证（穷举 + 量化对照）")
print(f"  矩阵: {N}×{N} 非对称, κ(A) = {KAPPAS}")
print(f"  比特深度: {BIT_DEPTHS}, 量化: float64 + int8")
print(f"  共计 {len(KAPPAS) * len(BIT_DEPTHS) * len(QUANTIZE_MODES) * 2} 次穷举")
print(f"  理论预测: RMSE(LS)/RMSE(PMIC) ≈ κ(A)²")
print("=" * 70)

all_results = []

for kappa in KAPPAS:
    A = make_nonsymmetric(N, kappa, seed=100 + kappa)
    b_vec = A @ x_true
    cond_A = np.linalg.cond(A)
    log2k = np.log2(cond_A)

    # Normal equations
    M = A.T @ A
    c = A.T @ b_vec
    cond_M = np.linalg.cond(M)

    print(f"\n{'─'*60}")
    print(f"  κ(A) = {cond_A:.1f}  (log₂κ = {log2k:.2f})")
    print(f"  κ(A^T A) = {cond_M:.1f} ≈ κ²,  κ((A^T A)²) = {cond_M**2:.1f} ≈ κ⁴")
    print(f"  预测: RMSE(LS)/RMSE(PMIC) ≈ κ² = {kappa**2}")

    for q_name, q_bits in QUANTIZE_MODES:
        print(f"\n  --- 量化模式: {q_name} ---")
        for bits in BIT_DEPTHS:
            N_qubits = N * bits
            total = 1 << N_qubits

            # PMIC: C = M (M 已是 SPD, p=1 on M → effective p=2 on A's singular values)
            qubo_pmic = QUBOEnergy(M, c, p=1, bits=bits, quantize_bits=q_bits)
            print(f"    b={bits} PMIC (2^{N_qubits}={total:>11,})...", end=" ", flush=True)
            rmse_pmic, b_eff_pmic = exhaustive_solve(qubo_pmic)

            # LS: C = M² (p=2 on M → effective p=4 on A's singular values)
            qubo_ls = QUBOEnergy(M, c, p=2, bits=bits, quantize_bits=q_bits)
            print(f"LS...", end=" ", flush=True)
            rmse_ls, b_eff_ls = exhaustive_solve(qubo_ls)

            ratio = rmse_ls / rmse_pmic if rmse_pmic > 1e-16 else float("inf")

            result = {
                "n": N, "kappa_A": float(cond_A), "log2_kappa": float(log2k),
                "kappa_M": float(cond_M),
                "bits": bits, "n_qubits": N_qubits, "total_states": total,
                "quantize": q_name,
                "pmic": {"p_on_A_sv": 2, "cond_C": float(qubo_pmic.cond_C),
                         "rmse": float(rmse_pmic), "b_eff": float(b_eff_pmic)},
                "ls": {"p_on_A_sv": 4, "cond_C": float(qubo_ls.cond_C),
                       "rmse": float(rmse_ls), "b_eff": float(b_eff_ls)},
                "rmse_ratio": float(ratio),
                "theoretical_ratio": float(kappa**2),
            }
            all_results.append(result)
            print(f"PMIC b_eff={b_eff_pmic:.2f}, LS b_eff={b_eff_ls:.2f}, 比值={ratio:.2f} (理论={kappa**2})")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*70}")
print(f"  汇总: RMSE(LS) / RMSE(PMIC) vs κ(A)²")
print(f"{'='*70}")

for q_name, _ in QUANTIZE_MODES:
    print(f"\n  --- 量化: {q_name} ---")
    print(f"  {'κ(A)':>8} {'κ(A)²':>8}", end="")
    for bits in BIT_DEPTHS:
        print(f" {'b=' + str(bits):>10}", end="")
    print()
    print(f"  {'─'*(16 + 10*len(BIT_DEPTHS))}")

    for kappa in KAPPAS:
        cond_A = next((r["kappa_A"] for r in all_results
                       if abs(r["kappa_A"] - kappa) < 1 and r["quantize"] == q_name), None)
        if cond_A is None:
            continue
        print(f"  {kappa:>8.1f} {kappa**2:>8}", end="")
        for bits in BIT_DEPTHS:
            r = next((x for x in all_results
                      if abs(x["kappa_A"] - kappa) < 1
                      and x["bits"] == bits and x["quantize"] == q_name), None)
            if r:
                print(f" {r['rmse_ratio']:>10.2f}", end="")
        print()

# ============================================================
# 关键对照: float64 vs int8 下 RMSE 比值对比
#   — float64 比值来自 Hessian 条件数差异（κ² vs κ⁴）
#   — int8 比值在此基础上叠加量化效应
# ============================================================
print(f"\n{'='*70}")
print(f"  关键对照: RMSE(LS) / RMSE(PMIC) — float64 vs int8")
print(f"  float64: Hessian 条件数效应  |  int8: 叠加上量化放大")
print(f"{'='*70}")
print(f"  {'κ':>3}  {'κ²理论':>8}  {'b':>3}  {'float64比':>10}  {'int8比':>10}  {'量化放大':>10}")
print(f"  {'─'*52}")
for kappa in KAPPAS:
    for bits in BIT_DEPTHS:
        pmic_f = next((r for r in all_results if abs(r["kappa_A"] - kappa) < 1
                       and r["bits"] == bits and r["quantize"] == "float64"), None)
        ls_f   = next((r for r in all_results if abs(r["kappa_A"] - kappa) < 1
                       and r["bits"] == bits and r["quantize"] == "float64"), None)
        pmic_i = next((r for r in all_results if abs(r["kappa_A"] - kappa) < 1
                       and r["bits"] == bits and r["quantize"] == "int8"), None)
        ls_i   = next((r for r in all_results if abs(r["kappa_A"] - kappa) < 1
                       and r["bits"] == bits and r["quantize"] == "int8"), None)
        ratio_f = ls_f["ls"]["rmse"] / pmic_f["pmic"]["rmse"] if pmic_f and ls_f and pmic_f["pmic"]["rmse"] > 1e-16 else float("inf")
        ratio_i = ls_i["ls"]["rmse"] / pmic_i["pmic"]["rmse"] if pmic_i and ls_i and pmic_i["pmic"]["rmse"] > 1e-16 else float("inf")
        amp = ratio_i / ratio_f if ratio_f > 1e-16 else float("inf")
        print(f"  {kappa:>3}  {kappa**2:>8}  {bits:>3}  {ratio_f:>10.2f}  {ratio_i:>10.2f}  {amp:>10.1f}x")

with open(OUTPUT_EXP / "nonsymmetric_results.json", "w") as f:
    json.dump({"details": all_results,
               "config": {"n": N, "kappas": KAPPAS, "bit_depths": BIT_DEPTHS,
                          "quantize_modes": [q for q, _ in QUANTIZE_MODES]}},
              f, indent=2, default=float)

print(f"\n详细结果: {OUTPUT_EXP / 'nonsymmetric_results.json'}")
print("实验 5 完成。")
