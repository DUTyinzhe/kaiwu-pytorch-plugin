"""
共享工具箱：FEM 网格 / QUBO 构造 / 模拟退火求解器 / 可视化
"""
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
import json
from pathlib import Path

OUTPUT = Path("D:/QPDE/模拟退火/outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. FEM 网格生成（5 点差分，矩形域 Poisson）
# ============================================================
def make_fem_grid(nx, ny, Lx=3.0, Ly=2.0):
    """
    nx, ny: 每个方向的网格划分数
    返回: nodes (n_total, 2), K_full (n_total, n_total) sparse
    """
    hx, hy = Lx / nx, Ly / ny
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    n_total = len(nodes)

    # 5 点差分刚度矩阵
    diag = np.ones(n_total) * (2.0 / hx**2 + 2.0 / hy**2)
    off_x = -np.ones(n_total) / hx**2
    off_y = -np.ones(n_total) / hy**2

    K = sparse.diags([diag, off_x, off_x, off_y, off_y],
                     [0, -1, 1, -(nx + 1), nx + 1],
                     shape=(n_total, n_total), format="lil")

    # Dirichlet 边界：对角置 1，非对角清零
    boundary = np.zeros(n_total, dtype=bool)
    boundary[0:nx + 1] = True
    boundary[-(nx + 1):] = True
    boundary[::nx + 1] = True
    boundary[nx::nx + 1] = True

    K[boundary, :] = 0
    K[:, boundary] = 0
    K[boundary, boundary] = 1.0

    return nodes, K.tocsr(), boundary


def static_condensation(K_full, boundary):
    """静凝聚：提取内部节点的 K_ii"""
    internal = ~boundary
    K_ii = K_full[internal, :][:, internal]
    return K_ii, internal


# ============================================================
# 2. 二进制编码 / QUBO 能量
# ============================================================
class QUBOEnergy:
    """
    p=1 (SAC): C = |A|, r = sign(A) b    （SPD 时退化为 C=A, r=b）
    p=2 (LS):  C = A²,  r = A b

    quantize_bits: 若不指定则默认 8（模拟 Kaiwu CIM [-128,127] 整数限制）；
                   设为 None 可禁用量化（回退到 float64 精度）。
    """
    def __init__(self, A, b_vec, p=1, bits=8, lb=-2.0, ub=2.0, quantize_bits=8):
        self.n = A.shape[0]
        self.bits = bits
        self.lb, self.ub = lb, ub
        self.alpha = (ub - lb) / (2**bits - 1)
        self.s = self.alpha * (2.0 ** np.arange(bits))  # [α, 2α, 4α, ...]

        if p == 1:
            # SAC
            if sparse.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = A
            eigvals, eigvecs = np.linalg.eigh(A_dense)
            abs_eigvals = np.abs(eigvals)
            self.C = (eigvecs * abs_eigvals) @ eigvecs.T  # |A|
            self.r = (eigvecs * (abs_eigvals / eigvals)) @ eigvecs.T @ b_vec  # sign(A)b
        elif p == 2:
            # LS
            if sparse.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = A
            self.C = A_dense @ A_dense
            self.r = A_dense @ b_vec
        else:
            raise ValueError(f"p must be 1 or 2, got {p}")

        self.cond_C = np.linalg.cond(self.C)
        self.cond_A = np.linalg.cond(A.toarray() if sparse.issparse(A) else A)

        # 参考解
        self.x_ref = np.linalg.solve(
            A.toarray() if sparse.issparse(A) else A, b_vec)

        # QUBO 变量信息
        self.N = self.n * bits  # 总二进制变量数

        # ----- 构造 QUBO 矩阵 Q = diag(h) + S/2 -----
        # E(z) = const + z^T Q z, z ∈ {0,1}^N
        # S = M^T C M = C ⊗ (s s^T), h = (lb·C·1 - r) ⊗ s
        ssT = np.outer(self.s, self.s)
        S_half = np.kron(self.C, ssT) / 2.0          # S/2
        v = self.lb * self.C.sum(axis=1) - self.r    # lb·C·1 - r
        h = np.kron(v, self.s)                       # h = v ⊗ s
        Q = S_half + np.diag(h)

        # ----- 量化 QUBO 条目到整数 -----
        self.quantize_bits = quantize_bits
        if quantize_bits is not None:
            q_max = np.max(np.abs(Q))
            if q_max > 1e-16:
                scale = (2**(quantize_bits - 1) - 1) / q_max  # 127 for 8-bit
                Q_int = np.round(Q * scale)
                Q = Q_int / scale
            self.Q = Q
        else:
            self.Q = Q

        # 常数项（符号不影响基态搜索，仅能量值报告）
        self._const = 0.5 * self.lb**2 * np.sum(self.C) - self.lb * np.sum(self.r)

    def decode(self, z):
        """z (N,) binary → x (n,) continuous"""
        z = z.reshape(self.n, self.bits)
        x = self.lb + np.dot(z, self.s)
        return x

    def energy(self, z):
        """E(z) = z^T Q z + const（量化 QUBO 矩阵版本）"""
        z = np.asarray(z, dtype=np.float64)
        return self._const + np.dot(z, np.dot(self.Q, z))

    def compute_rmse(self, z):
        x = self.decode(z)
        return np.sqrt(np.mean((x - self.x_ref)**2))

    def compute_b_eff(self, z):
        """有效精度 b_eff = -log2(RMSE)"""
        rmse = self.compute_rmse(z)
        if rmse < 1e-16:
            return float(self.bits)
        return -np.log2(rmse)


# ============================================================
# 3. 模拟退火求解器
# ============================================================
def simulated_annealing(qubo: QUBOEnergy, n_steps=200000,
                        T0=None, cooling=0.99995, verbose=True):
    """
    模拟退火搜索 QUBO 基态。
    使用 ΔE = ±s_k * grad_i + 1/2 * s_k^2 * C_{ii} 进行 O(1) 能量差分。
    """
    n, bits, N = qubo.n, qubo.bits, qubo.N
    C, r, s = qubo.C, qubo.r, qubo.s

    # 随机初始状态
    z = np.random.randint(0, 2, N).astype(np.float64)
    x = qubo.decode(z)
    grad = np.dot(C, x) - r
    E = 0.5 * np.dot(x, np.dot(C, x)) - np.dot(x, r)

    z_best = z.copy()
    x_best = x.copy()
    E_best = E

    # 自调初始温度：使 ~80% 坏移动被接受
    if T0 is None:
        deltas = []
        for _ in range(200):
            idx = np.random.randint(N)
            i, k = divmod(idx, bits)
            delta = (1.0 if z[idx] == 0 else -1.0) * s[k] * grad[i] + 0.5 * s[k]**2 * C[i, i]
            if delta > 0:
                deltas.append(delta)
        T0 = np.mean(deltas) / 1.4 if deltas else 1.0  # log(0.8) ≈ -0.22

    T = T0
    accepted = 0

    for step in range(n_steps):
        idx = np.random.randint(N)
        i, k = divmod(idx, bits)

        # ΔE 计算
        sign = 1.0 if z[idx] == 0 else -1.0
        ds = sign * s[k]
        dE = ds * grad[i] + 0.5 * s[k]**2 * C[i, i]

        if dE < 0 or np.random.random() < np.exp(-dE / T):
            # 接受移动
            z[idx] = 1.0 - z[idx]
            x[i] += ds
            grad += C[:, i] * ds
            E += dE
            accepted += 1

            if E < E_best:
                z_best = z.copy()
                x_best = x.copy()
                E_best = E

        T *= cooling

        if verbose and (step + 1) % (n_steps // 5) == 0:
            acc_rate = accepted / (step + 1)
            print(f"  step {step+1:>7d}/{n_steps}  T={T:.4f}  acc={acc_rate:.3f}  "
                  f"E_best={E_best:.2f}  RMSE={qubo.compute_rmse(z_best):.4f}")

    return z_best, E_best


# ============================================================
# 4. 运行单次实验
# ============================================================
def run_single(A, b_vec, p, bits, label, n_steps=200000, T0=None, cooling=0.99995):
    """跑一次 SA，返回结果 dict"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  p={p}, b={bits}, κ(A)={np.linalg.cond(A.toarray() if sparse.issparse(A) else A):.1f}")
    print(f"{'='*60}")

    qubo = QUBOEnergy(A, b_vec, p=p, bits=bits)
    print(f"  κ(C_p{p}) = {qubo.cond_C:.1f},  N_qubits = {qubo.N}")

    z_opt, energy = simulated_annealing(qubo, n_steps=n_steps,
                                        T0=T0, cooling=cooling, verbose=True)
    rmse = qubo.compute_rmse(z_opt)
    b_eff = qubo.compute_b_eff(z_opt)

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
        "energy": float(energy),
        "n_steps": n_steps,
    }
    print(f"  => RMSE={rmse:.4f}, b_eff={b_eff:.2f}, cond_C={qubo.cond_C:.1f}")
    return result, qubo.x_ref


def save_results(results, filename):
    """保存结果到 JSON"""
    with open(OUTPUT / filename, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n结果已保存至 {OUTPUT / filename}")
