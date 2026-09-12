"""
实验 3：不定矩阵 + 负定矩阵 — 验证 SAC 对非 SPD 矩阵的普适性
2 种矩阵 × 2 种方法(p=1, p=2) = 4 次 SA

不定: A = K - αI, α = λ_min + 0.3(λ_max - λ_min) → λ_min < 0, λ_max > 0
负定: A = -K → λ < 0
"""
import numpy as np
import json
from pathlib import Path
from scipy.sparse import linalg as spla
from sa_utils import make_fem_grid, static_condensation, QUBOEnergy
from sa_utils import simulated_annealing, run_single, save_results, OUTPUT

OUTPUT_EXP = OUTPUT / "exp3_indef_negdef"
OUTPUT_EXP.mkdir(parents=True, exist_ok=True)

NX, NY = 10, 8
BITS = 8
METHODS = [(1, "SAC"), (2, "LS")]
N_STEPS = 200000
COOLING = 0.99995

print("=" * 70)
print("  实验 3: 不定 / 负定矩阵 — SAC 普适性验证")
print(f"  网格: {NX}×{NY}, 位深: {BITS}, 方法: p=1, p=2")
print(f"  共计 4 次 SA 运行")
print("=" * 70)

nodes, K_full, boundary = make_fem_grid(NX, NY)
K_ii, internal = static_condensation(K_full, boundary)
n = K_ii.shape[0]

x_coords = nodes[internal, 0]
y_coords = nodes[internal, 1]
x_true = np.sin(np.pi * x_coords / 3.0) * np.sin(np.pi * y_coords / 2.0)

K_dense = K_ii.toarray()
eigvals = np.linalg.eigvalsh(K_dense)
l_min, l_max = eigvals[0], eigvals[-1]
print(f"\nK_ii 特征值范围: [{l_min:.4f}, {l_max:.4f}]")

# ---- 不定矩阵 A_indef = K - αI ----
alpha = l_min + 0.3 * (l_max - l_min)
A_indef = K_dense - alpha * np.eye(n)
eigvals_indef = np.linalg.eigvalsh(A_indef)
print(f"不定矩阵 A_indef: α={alpha:.4f}, λ ∈ [{eigvals_indef[0]:.4f}, {eigvals_indef[-1]:.4f}]")
# 确保确实不定
assert eigvals_indef[0] < 0 < eigvals_indef[-1], "A_indef 不是不定矩阵！"

b_indef = A_indef @ x_true

# ---- 负定矩阵 A_negdef = -K ----
A_negdef = -K_dense
eigvals_negdef = np.linalg.eigvalsh(A_negdef)
print(f"负定矩阵 A_negdef: λ ∈ [{eigvals_negdef[0]:.4f}, {eigvals_negdef[-1]:.4f}]")
assert eigvals_negdef[-1] < 0, "A_negdef 不是负定矩阵！"

b_negdef = A_negdef @ x_true

all_results = []
all_info = []

configs = [
    (A_indef, b_indef, "indefinite", "不定"),
    (A_negdef, b_negdef, "negative_def", "负定"),
]

for A_mat, b_vec, short_name, chinese_name in configs:
    cond_A = np.linalg.cond(A_mat)
    log2_kappa = np.log2(cond_A)
    print(f"\n{'─'*60}")
    print(f"  {chinese_name}矩阵: κ(A) = {cond_A:.1f}, log₂κ = {log2_kappa:.2f}")
    print(f"{'─'*60}")

    for p, method_name in METHODS:
        label = f"{short_name}_b{BITS}_{method_name}"
        print(f"\n  > {label}")

        result, x_ref = run_single(
            A_mat, b_vec, p=p, bits=BITS, label=label,
            n_steps=N_STEPS, cooling=COOLING
        )

        result["nx"] = NX
        result["ny"] = NY
        result["n_internal"] = int(n)
        result["matrix_type"] = short_name

        all_results.append(result)

        np.savez(OUTPUT_EXP / f"ref_{label}.npz",
                 x_ref=x_ref, x_coords=x_coords, y_coords=y_coords)

        predicted = BITS - p * log2_kappa
        info = {
            "matrix": chinese_name,
            "matrix_type": short_name,
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
        print(f"    b_eff={result['b_eff']:.2f}, 预测={predicted:.2f}, Δ={info['delta']:.3f}")

save_results(all_results, "exp3_indef_negdef.json")

# 汇总
print(f"\n{'='*70}")
print(f"  实验 3 汇总")
print(f"{'='*70}")
print(f"  {'矩阵':<6} {'p':>2} {'cond_A':>10} {'cond_C':>10} {'RMSE':>10} {'b_eff':>8} {'预测':>8} {'Δ':>8}")
print(f"  {'─'*70}")
for info in all_info:
    print(f"  {info['matrix']:<6} {info['p']:>2} {info['cond_A']:>10.1f} {info['cond_C']:>10.1f} "
          f"{info['rmse']:>10.4f} {info['b_eff']:>8.2f} "
          f"{info['predicted_b_eff']:>8.2f} {info['delta']:>8.3f}")

# 关键对比: p=1 vs p=2 在不定矩阵上的差距
print(f"\n  关键发现:")
for short_name, chinese_name in [("indefinite", "不定"), ("negative_def", "负定")]:
    sac = next(x for x in all_info if x["matrix_type"] == short_name and x["p"] == 1)
    ls = next(x for x in all_info if x["matrix_type"] == short_name and x["p"] == 2)
    diff = sac["b_eff"] - ls["b_eff"]
    log2k = np.log2(sac["cond_A"])
    print(f"    {chinese_name}: b_eff(SAC)={sac['b_eff']:.2f}, b_eff(LS)={ls['b_eff']:.2f}, "
          f"差值={diff:.2f} (≈ log₂κ = {log2k:.2f})")

with open(OUTPUT_EXP / "summary_table.json", "w") as f:
    json.dump(all_info, f, indent=2)

print(f"\n详细结果: {OUTPUT / 'exp3_indef_negdef.json'}")
print("实验 3 完成。")
