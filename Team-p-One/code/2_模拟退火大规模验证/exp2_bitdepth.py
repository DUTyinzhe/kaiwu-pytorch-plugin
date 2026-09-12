"""
实验 2：位深扫描 — 穷举验证 b_eff = b - p·log₂κ(A)
小矩阵 n=3, b=4..7, 穷举 QUBO 基态保证全局最优
2 种方法(p=1, p=2) × 4 种位深 = 8 次穷举
"""
import numpy as np
import json
from pathlib import Path
from itertools import product
from sa_utils import QUBOEnergy, save_results, OUTPUT

OUTPUT_EXP = OUTPUT / "exp2_bitdepth"
OUTPUT_EXP.mkdir(parents=True, exist_ok=True)

# 构造 3×3 SPD 矩阵，条件数可控
A = np.array([
    [4.0, -1.0,  0.0],
    [-1.0, 4.0, -1.0],
    [0.0, -1.0,  4.0],
])
x_true = np.array([1.0, 0.5, -0.5])
b_vec = A @ x_true

BIT_DEPTHS = [4, 5, 6, 7]
METHODS = [(1, "SAC"), (2, "LS")]

print("=" * 70)
print("  实验 2: 位深扫描 — 穷举验证编码正确性（float64，无量化）")
print(f"  矩阵: 3×3 SPD, κ(A)={np.linalg.cond(A):.1f}")
print(f"  位深: {BIT_DEPTHS}, 方法: p=1, p=2")
print(f"  共计 {len(BIT_DEPTHS) * len(METHODS)} 次穷举")
print("=" * 70)

cond_A = np.linalg.cond(A)
log2_kappa = np.log2(cond_A)
print(f"\nlog₂κ = {log2_kappa:.3f}")
print(f"理论: b_eff(p=1) = b - {log2_kappa:.2f}")
print(f"理论: b_eff(p=2) = b - {2*log2_kappa:.2f}")
print(f"理论: 差值 = {log2_kappa:.2f}")

all_results = []
all_info = []

for bits in BIT_DEPTHS:
    for p, method_name in METHODS:
        label = f"exhaustive_b{bits}_{method_name}"
        print(f"\n  > {label}")

        qubo = QUBOEnergy(A, b_vec, p=p, bits=bits, quantize_bits=None)
        N = qubo.N  # n * bits

        # 穷举所有 2^N 个状态
        E_best = np.inf
        z_best = None
        total = 1 << N
        print(f"    穷举 2^{N} = {total:,} 个状态...", end=" ", flush=True)

        for k in range(total):
            z = np.array([(k >> j) & 1 for j in range(N)], dtype=np.float64)
            E = qubo.energy(z)
            if E < E_best:
                E_best = E
                z_best = z.copy()

        print(f"完成")

        rmse = qubo.compute_rmse(z_best)
        b_eff = qubo.compute_b_eff(z_best)
        x_opt = qubo.decode(z_best)

        predicted = bits - p * log2_kappa

        result = {
            "label": label,
            "p": p,
            "bits": bits,
            "n_nodes": qubo.n,
            "n_qubits": qubo.N,
            "cond_A": qubo.cond_A,
            "cond_C": qubo.cond_C,
            "rmse": float(rmse),
            "b_eff": float(b_eff),
            "energy": float(E_best),
            "method": "exhaustive",
        }
        all_results.append(result)

        info = {
            "bits": bits,
            "p": p,
            "method": method_name,
            "cond_A": result["cond_A"],
            "cond_C": result["cond_C"],
            "rmse": result["rmse"],
            "b_eff": result["b_eff"],
            "predicted_b_eff": predicted,
            "delta": result["b_eff"] - predicted,
        }
        all_info.append(info)

        print(f"    x_opt={x_opt}, RMSE={rmse:.6f}, b_eff={b_eff:.2f}")
        print(f"    预测 b_eff={predicted:.2f}, Δ={info['delta']:.4f}")

        np.savez(OUTPUT_EXP / f"result_{label}.npz",
                 x_opt=x_opt, x_true=x_true, z_best=z_best)

save_results(all_results, "exp2_bitdepth.json")

# 汇总
print(f"\n{'='*70}")
print(f"  实验 2 汇总（穷举）")
print(f"{'='*70}")

for p, method_name in METHODS:
    print(f"\n  --- {method_name} (p={p}) ---")
    print(f"  {'b':>4} {'RMSE':>12} {'b_eff':>8} {'预测':>8} {'Δ':>8}")
    print(f"  {'─'*42}")
    for info in all_info:
        if info["p"] == p:
            print(f"  {info['bits']:>4} {info['rmse']:>12.6f} {info['b_eff']:>8.2f} "
                  f"{info['predicted_b_eff']:>8.2f} {info['delta']:>8.4f}")

print(f"\n  --- b_eff 差值 (p=1 - p=2) ---")
print(f"  {'b':>4} {'b_eff(SAC)':>12} {'b_eff(LS)':>12} {'差值':>8} {'log₂κ':>10}")
print(f"  {'─'*50}")
for bits in BIT_DEPTHS:
    sac = next(x for x in all_info if x["bits"] == bits and x["p"] == 1)
    ls = next(x for x in all_info if x["bits"] == bits and x["p"] == 2)
    diff = sac["b_eff"] - ls["b_eff"]
    print(f"  {bits:>4} {sac['b_eff']:>12.2f} {ls['b_eff']:>12.2f} {diff:>8.2f} {log2_kappa:>10.2f}")

with open(OUTPUT_EXP / "summary_table.json", "w") as f:
    json.dump(all_info, f, indent=2)

print(f"\n详细结果: {OUTPUT / 'exp2_bitdepth.json'}")
print("实验 2 完成。")
