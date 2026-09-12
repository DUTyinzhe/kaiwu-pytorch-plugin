"""
实验 4：定量标度律验证 — RMSE 比值 vs κ(A)（穷举 + QUBO 量化对照）

理论预测: b_eff = b - p·log₂κ
          RMSE ≈ 2^{-b_eff} = 2^{-b} · κ^p
          RMSE(LS) / RMSE(SAC) ≈ κ

量化对照: float64（无量化）vs int8（模拟 Kaiwu CIM [-128,127]）
"""
import numpy as np
import json
from pathlib import Path
from sa_utils import QUBOEnergy, OUTPUT

OUTPUT_EXP = OUTPUT / "exp4_scaling_law"
OUTPUT_EXP.mkdir(parents=True, exist_ok=True)


def make_matrix_with_kappa(n, kappa, seed=42):
    """生成 n×n SPD 矩阵，条件数精确等于 kappa"""
    rs = np.random.default_rng(seed)
    if kappa == 1.0:
        eigvals = np.ones(n)
    else:
        eigvals = np.geomspace(kappa, 1.0, n)
    D = np.diag(eigvals)
    Q, _ = np.linalg.qr(rs.standard_normal((n, n)))
    A = Q @ D @ Q.T
    return A, eigvals


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
N = 3
KAPPAS = [2, 4, 8, 16, 32, 64]
BIT_DEPTHS = [4, 5, 6]
QUANTIZE_MODES = [("float64", None), ("int8", 8)]
x_true = np.array([1.0, 0.5, -0.5])

print("=" * 70)
print("  实验 4: 定量标度律验证 — RMSE 比值 vs κ(A)（穷举）")
print(f"  矩阵: {N}×{N} SPD, κ = {KAPPAS}")
print(f"  比特深度: {BIT_DEPTHS}, 量化: float64 + int8")
print(f"  共计 {len(KAPPAS) * len(BIT_DEPTHS) * len(QUANTIZE_MODES) * 2} 次穷举")
print("=" * 70)

all_results = []

for kappa in KAPPAS:
    A, eigvals = make_matrix_with_kappa(N, kappa, seed=42)
    b_vec = A @ x_true
    log2k = np.log2(kappa)
    cond_A = np.linalg.cond(A)

    print(f"\n{'─'*60}")
    print(f"  κ = {kappa}  (log₂κ = {log2k:.2f}), 实测 κ(A) = {cond_A:.1f}")
    print(f"  理论: RMSE(LS)/RMSE(SAC) ≈ κ = {kappa}")

    for q_name, q_bits in QUANTIZE_MODES:
        print(f"\n  --- 量化模式: {q_name} ---")
        for bits in BIT_DEPTHS:
            N_qubits = N * bits
            total = 1 << N_qubits

            for p, name in [(1, "SAC"), (2, "LS")]:
                qubo = QUBOEnergy(A, b_vec, p=p, bits=bits, quantize_bits=q_bits)
                print(f"    b={bits} {name} (2^{N_qubits}={total:>10,})...", end=" ", flush=True)
                rmse, b_eff = exhaustive_solve(qubo)

                result = {
                    "kappa": float(kappa), "log2_kappa": float(log2k),
                    "cond_A": float(cond_A), "cond_C": float(qubo.cond_C),
                    "bits": bits, "p": p, "method": name,
                    "quantize": q_name,
                    "rmse": float(rmse), "b_eff": float(b_eff),
                    "n_qubits": N_qubits, "total_states": total,
                }
                all_results.append(result)
                print(f"RMSE={rmse:.6f}  b_eff={b_eff:.2f}")

        # 计算比值（每个 b 下）
        for bits in BIT_DEPTHS:
            sac = next(r for r in all_results
                       if r["kappa"] == kappa and r["p"] == 1
                       and r["bits"] == bits and r["quantize"] == q_name)
            ls  = next(r for r in all_results
                       if r["kappa"] == kappa and r["p"] == 2
                       and r["bits"] == bits and r["quantize"] == q_name)
            ratio = ls["rmse"] / sac["rmse"] if sac["rmse"] > 1e-16 else float("inf")
            print(f"    b={bits}  RMSE 比值 LS/SAC = {ratio:.2f}  (理论 ≈ {kappa})")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*70}")
print(f"  汇总: b_eff 差值 & RMSE 比值 vs κ(A)")
print(f"{'='*70}")

for q_name, _ in QUANTIZE_MODES:
    print(f"\n  --- 量化: {q_name}, 最高比特深度 (b={max(BIT_DEPTHS)}) ---")
    print(f"  {'κ':>6} {'log₂κ':>7} {'b_eff SAC':>11} {'b_eff LS':>11} "
          f"{'Δb_eff':>9} {'Δb 理论':>9} {'RMSE比':>9} {'比值理论':>9}")
    print(f"  {'─'*78}")
    summary = []
    for kappa in KAPPAS:
        log2k = np.log2(kappa)
        bits = max(BIT_DEPTHS)
        sac = next((r for r in all_results if r["kappa"] == kappa and r["p"] == 1
                    and r["bits"] == bits and r["quantize"] == q_name), None)
        ls  = next((r for r in all_results if r["kappa"] == kappa and r["p"] == 2
                    and r["bits"] == bits and r["quantize"] == q_name), None)
        if sac is None or ls is None:
            continue
        delta_beff = sac["b_eff"] - ls["b_eff"]
        rmse_ratio = ls["rmse"] / sac["rmse"] if sac["rmse"] > 1e-16 else float("inf")
        summary.append({
            "kappa": float(kappa), "log2_kappa": float(log2k),
            "b_eff_SAC": sac["b_eff"], "b_eff_LS": ls["b_eff"],
            "delta_b_eff": float(delta_beff), "theoretical_delta": float(log2k),
            "rmse_ratio": float(rmse_ratio), "theoretical_ratio": float(kappa),
            "rmse_SAC": sac["rmse"], "rmse_LS": ls["rmse"],
        })
        print(f"  {kappa:>6} {log2k:>7.2f} {sac['b_eff']:>11.2f} {ls['b_eff']:>11.2f} "
              f"{delta_beff:>9.2f} {log2k:>9.2f} {rmse_ratio:>9.2f} {kappa:>9}")

    # RMSE 比值 vs κ 线性拟合
    if len(summary) >= 3:
        kappa_arr = np.array([s["kappa"] for s in summary])
        ratio_arr = np.array([s["rmse_ratio"] for s in summary])
        valid = ratio_arr < 1000
        if valid.sum() >= 3:
            slope, intercept = np.polyfit(kappa_arr[valid], ratio_arr[valid], 1)
            print(f"\n  RMSE 比值拟合: ratio = {slope:.3f} · κ + {intercept:.3f}  (理论: ratio = 1.000 · κ)")

        delta_arr = np.array([s["delta_b_eff"] for s in summary])
        log2k_arr = np.array([s["log2_kappa"] for s in summary])
        slope2, intercept2 = np.polyfit(log2k_arr[valid], delta_arr[valid], 1)
        print(f"  Δb_eff 拟合:    Δb   = {slope2:.3f} · log₂κ + {intercept2:.3f}  (理论: Δb = 1.000 · log₂κ)")

# ============================================================
# 关键对照: float64 下 SAC vs LS 的 RMSE 是否相同
# ============================================================
print(f"\n{'='*70}")
print(f"  关键对照: float64 下 RMSE(SAC) vs RMSE(LS)")
print(f"  （无量化时穷举应得相同 RMSE，差异仅来自量化）")
print(f"{'='*70}")
for kappa in KAPPAS:
    for bits in BIT_DEPTHS:
        sac_f = next((r for r in all_results if r["kappa"] == kappa and r["p"] == 1
                      and r["bits"] == bits and r["quantize"] == "float64"), None)
        ls_f  = next((r for r in all_results if r["kappa"] == kappa and r["p"] == 2
                      and r["bits"] == bits and r["quantize"] == "float64"), None)
        sac_i = next((r for r in all_results if r["kappa"] == kappa and r["p"] == 1
                      and r["bits"] == bits and r["quantize"] == "int8"), None)
        ls_i  = next((r for r in all_results if r["kappa"] == kappa and r["p"] == 2
                      and r["bits"] == bits and r["quantize"] == "int8"), None)
        same_f = "✓" if sac_f and ls_f and abs(sac_f["rmse"] - ls_f["rmse"]) < 1e-10 else "✗"
        ratio_i = ls_i["rmse"] / sac_i["rmse"] if sac_i and ls_i and sac_i["rmse"] > 1e-16 else float("inf")
        print(f"  κ={kappa:>3} b={bits}: float64 SAC=LS? {same_f}  |  int8 RMSE比={ratio_i:.2f} (理论={kappa})")

with open(OUTPUT_EXP / "scaling_law_results.json", "w") as f:
    json.dump({"details": all_results,
               "config": {"n": N, "bit_depths": BIT_DEPTHS, "kappas": KAPPAS,
                          "quantize_modes": [q for q, _ in QUANTIZE_MODES]}},
              f, indent=2, default=float)

print(f"\n详细结果: {OUTPUT_EXP / 'scaling_law_results.json'}")
print("实验 4 完成。")
