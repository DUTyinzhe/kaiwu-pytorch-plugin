"""
实验 1：大规模网格扫描 — 验证 b_eff = b - p·log₂κ(A)
5 种网格 × 2 种位深 × 2 种方法(p=1, p=2) = 20 次 SA
"""
import numpy as np
import json
from pathlib import Path
from sa_utils import make_fem_grid, static_condensation, QUBOEnergy
from sa_utils import simulated_annealing, run_single, save_results, OUTPUT

OUTPUT_EXP = OUTPUT / "exp1_large_scale"
OUTPUT_EXP.mkdir(parents=True, exist_ok=True)

GRIDS = [
    (20, 15),
    (25, 18),
    (30, 20),
    (35, 25),
    (40, 30),
]
BIT_DEPTHS = [8, 10]
METHODS = [(1, "SAC"), (2, "LS")]

N_STEPS = 300000
COOLING = 0.99995

print("=" * 70)
print("  实验 1: 大规模网格扫描 — 精度标度律验证")
print(f"  网格: {len(GRIDS)} 种, 位深: {BIT_DEPTHS}, 方法: p=1, p=2")
print(f"  共计 {len(GRIDS) * len(BIT_DEPTHS) * len(METHODS)} 次 SA 运行")
print("=" * 70)

all_results = []
all_info = []

for nx, ny in GRIDS:
    n_internal = (nx - 1) * (ny - 1)
    print(f"\n{'─'*60}")
    print(f"  网格 {nx}×{ny}, 内部节点 {n_internal}")
    print(f"{'─'*60}")

    # 按 exp3 模式：生成 → 静凝聚 → 显式转 dense
    nodes, K_full, boundary = make_fem_grid(nx, ny)
    K_ii_sparse, internal = static_condensation(K_full, boundary)
    K_ii = K_ii_sparse.toarray()  # ← 关键：显式 dense
    n = K_ii.shape[0]

    x_coords = nodes[internal, 0]
    y_coords = nodes[internal, 1]
    x_true = np.sin(np.pi * x_coords / 3.0) * np.sin(np.pi * y_coords / 2.0)
    b_vec = K_ii @ x_true

    for bits in BIT_DEPTHS:
        for p, method_name in METHODS:
            label = f"grid_{nx}x{ny}_b{bits}_{method_name}"
            print(f"\n  > {label}")

            result, x_ref = run_single(
                K_ii, b_vec, p=p, bits=bits, label=label,
                n_steps=N_STEPS, cooling=COOLING
            )

            result["nx"] = nx
            result["ny"] = ny
            result["n_internal"] = int(n)

            all_results.append(result)

            np.savez(OUTPUT_EXP / f"ref_{label}.npz",
                     x_ref=x_ref, x_coords=x_coords, y_coords=y_coords)

            predicted = bits - p * np.log2(result["cond_A"])
            info = {
                "grid": f"{nx}x{ny}",
                "n_internal": int(n),
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
            print(f"    cond_A={result['cond_A']:.1f}, b_eff={result['b_eff']:.2f}, "
                  f"预测 b_eff={predicted:.2f}, RMSE={result['rmse']:.4f}")

save_results(all_results, "exp1_large_scale.json")

print(f"\n{'='*80}")
print(f"  实验 1 汇总")
print(f"{'='*80}")
print(f"{'网格':<12} {'n':>5} {'b':>3} {'p':>2} {'cond_A':>10} {'cond_C':>10} {'RMSE':>10} {'b_eff':>8} {'预测':>8} {'Δ':>8}")
print(f"{'─'*80}")
for info in all_info:
    print(f"{info['grid']:<12} {info['n_internal']:>5} {info['bits']:>3} {info['p']:>2} "
          f"{info['cond_A']:>10.1f} {info['cond_C']:>10.1f} {info['rmse']:>10.4f} "
          f"{info['b_eff']:>8.2f} {info['predicted_b_eff']:>8.2f} {info['delta']:>8.3f}")

with open(OUTPUT_EXP / "summary_table.json", "w") as f:
    json.dump(all_info, f, indent=2)

print(f"\n详细结果: {OUTPUT / 'exp1_large_scale.json'}")
print("实验 1 完成。")
