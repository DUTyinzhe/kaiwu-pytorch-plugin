# 支撑材料：源代码与实验数据

琶洲算法大赛 · 量子计算赛题 · 队伍2

## 目录结构

```
code/
├── README.md                              # 本文件
├── 0_原始数据集/                           # 四个领域的原始数据集
│   ├── breast-cancer-wisconsin.data       # Breast Cancer Wisconsin（生物医药）
│   ├── Data_for_UCI_named.csv             # Electrical Grid Stability（电力系统）
│   ├── HTRU2.zip                          # HTRU2 Pulsar（射电天文）
│   └── PBE_data.csv                       # Halide Perovskite（材料设计）
├── 1_QUBO编码与CIM真机实验/                # SAC/LS QUBO构造 + Kaiwu CIM真机
├── 2_模拟退火大规模验证/                    # SA精度缩放律验证
├── 3_量子实时约束深度网络/                  # 四个领域的DL量子约束实验
│   ├── 生物医药-BreastCancer-PINN/
│   ├── 电力系统-GridStability-KAN/
│   ├── 材料设计-Perovskite-GCN/
│   └── 射电天文-HTRU2-KAN/
├── 4_DD-QUBO区域分解/                      # 区域分解任意规模PDE求解
└── 5_WuYue_VQE跨平台验证/                  # 通用量子门线路VQE实验
```

---
## 0. 原始数据集

四个竞赛指定领域的数据集，实验中通过URL加载。

| 文件 | 来源 | 大小 | 对应实验 |
|------|------|------|----------|
| `breast-cancer-wisconsin.data` | UCI Breast Cancer Wisconsin | 20KB | 3.1 生物医药-PINN |
| `Data_for_UCI_named.csv` | UCI Electrical Grid Stability | 750KB | 3.2 电力系统-KAN |
| `HTRU2.zip` | UCI HTRU2 Pulsar | 819KB | 3.4 射电天文-KAN |
| `PBE_data.csv` | GitHub halide_perovs_design | 188KB | 3.3 材料设计-GCN |

**注意：** 模拟退火实验（第2节）使用FEM自动生成的合成网格数据，DD-QUBO（第4节）使用FEM泊松方程数据，VQE（第5节）使用程序生成的泊松方程矩阵，均无需外部数据集。

---

## 1. QUBO编码与CIM真机实验

SAC（符号感知凸化）与LS（最小二乘）两种QUBO编码的构造与真机验证。

| 文件 | 说明 |
|------|------|
| `pde_energy_qubo.ipynb` | SPD矩阵SAC-QUBO（能量泛函编码） |
| `pde_ls_qubo.ipynb` | SPD矩阵LS-QUBO（残差最小化编码） |
| `pde_energy_single.ipynb` | 单次SAC-QUBO求解 |
| `pde_ls_single.ipynb` | 单次LS-QUBO求解 |
| `energy_indef.ipynb` | 不定矩阵SAC-QUBO |
| `ls_indef.ipynb` | 不定矩阵LS-QUBO |
| `energy_negdef.ipynb` | 负定矩阵SAC-QUBO |
| `ls_negdef.ipynb` | 负定矩阵LS-QUBO |
| `cond_measure.ipynb` | 条件数与精度关系测量 |
| `generate_all.py` | 批量生成QUBO并提交 |
| `output_*/` | CIM真机输出数据 |

**运行方式：** 在Jupyter Notebook中按顺序执行各cell，Kaiwu SDK的quota模式提交至550量子比特相干光量子计算机。

---

## 2. 模拟退火大规模验证

使用经典模拟退火（Metropolis准则）验证精度缩放律 $b_{\text{eff}} = b - p\log_2\kappa(A)$。

| 文件 | 说明 |
|------|------|
| `sa_utils.py` | **核心工具箱**：FEM网格生成、QUBOEnergy类（p=1 SAC, p=2 LS，8位量化）、SA求解器（O(1)单比特翻转能量更新） |
| `exp1_large_scale.py` | **实验1**：大规模网格扫描（5种网格×2位深×2方法=20次SA），验证精度缩放律 |
| `exp2_bitdepth.py` | **实验2**：小矩阵穷举枚举（3×3 SPD），验证不同位深下b_eff |
| `exp3_indef_negdef.py` | **实验3**：不定矩阵+负定矩阵SAC验证 |
| `exp4_scaling_law_verify.py` | **实验4**：控制条件数扫描（κ=2,4,8,16,32,64），量化壁垒验证 |
| `exp5_nonsymmetric.py` | **实验5**：非对称矩阵PMIC验证 |
| `exp1_large_scale.json` | 实验1完整结果数据 |
| `exp3_indef_negdef.json` | 实验3完整结果数据 |
| `plot_*.py` | 对应实验的绘图脚本 |

**运行方式：**
```bash
cd code/2_模拟退火大规模验证
python exp1_large_scale.py    # 约需30分钟（5×2×2=20次SA，每次30万步）
python exp3_indef_negdef.py   # 约需15分钟
python exp4_scaling_law_verify.py  # 约需10分钟
```

---

## 3. 量子实时约束深度网络

将CIM作为神经网络训练的物理正则化协处理器，覆盖四个竞赛指定领域。

### 3.1 生物医药 — Breast Cancer Wisconsin + PINN

| 文件 | 说明 |
|------|------|
| `bc_energy_qubo.ipynb` | SAC-QUBO预设构造 |
| `bc_ls_qubo.ipynb` | LS-QUBO预设构造 |
| `bc_pinn_energy.py` | SAC预设PINN训练（1500 epoch） |
| `bc_pinn_classical.py` | 纯经典PINN对照 |
| `bc_pinn_ls.py` | LS预设PINN训练 |
| `result_pinn_*.json` | 训练结果（L2/FEM误差） |

### 3.2 电力系统 — Electrical Grid Stability + KAN

| 文件 | 说明 |
|------|------|
| `gs_energy_qubo.ipynb` | SAC-QUBO预设构造 |
| `gs_ls_qubo.ipynb` | LS-QUBO预设构造 |
| `gs_kan_energy.py` | SAC预设KAN训练（RBF-KAN, [2,64,64,1]） |
| `gs_kan_classical.py` | 纯经典KAN对照 |
| `gs_kan_ls.py` | LS预设KAN训练 |
| `result_kan_*.json` | 训练结果 |

### 3.3 材料设计 — Halide Perovskite + GCN

| 文件 | 说明 |
|------|------|
| `perov_energy_qubo.ipynb` | SAC-QUBO图拉普拉斯正则化 |
| `perov_ls_qubo.ipynb` | LS-QUBO图拉普拉斯正则化 |
| `perov_gnn_train.py` | 2层GCN半监督训练（10标签节点） |
| `result_gnn.json` | RMSE对比结果 |

### 3.4 射电天文 — HTRU2 + KAN实时刷新

| 文件 | 说明 |
|------|------|
| `htru2_energy_qubo.ipynb` | SAC-QUBO预设构造 |
| `htru2_kan_quantum.py` | **核心实验**：KAN+量子实时刷新（每50 epoch CIM提交，共10次） |
| `htru2_kan_standalone.py` | 纯KAN对照 |
| `result_kan_quantum_realtime.json` | 实时刷新结果（含loss/accuracy/cim_rmse完整历史） |

**运行方式：**
```bash
# 先运行对应notebook构造QUBO预设
jupyter notebook bc_energy_qubo.ipynb

# 再运行训练脚本
python bc_pinn_energy.py
```

---

## 4. DD-QUBO区域分解

通过静态凝聚将全局PDE分解为固定规模子域QUBO，实现任意规模PDE的量子求解。

| 文件 | 说明 |
|------|------|
| `cond_analysis.py` | 条件数解耦分析（验证子域κ与全局网格无关） |
| `plot_framework.py` | 框架示意图生成 |
| `scaling_ls_2x2.ipynb` | 2×2子域LS-DD求解 |
| `scaling_ls_4x4.ipynb` | 4×4子域LS-DD求解 |
| `cond_results.json` | 完整条件数解耦数据（全局 vs 子域） |
| `result_2x2.json` | 2×2 DD求解精度结果 |
| `result_4x4.json` | 4×4 DD求解精度结果 |
| `text1(1).py` | 有限元子网格高精度求解辅助脚本 |

**运行方式：**
```bash
python cond_analysis.py    # 条件数分析
jupyter notebook scaling_ls_2x2.ipynb  # 2×2子域求解
```

---

## 5. WuYue VQE跨平台验证

基于WuYue SDK的变分量子本征求解器（VQE），实现通用量子门线路对PDE线性系统的求解，与Kaiwu CIM形成跨技术路线对比。

| 文件 | 说明 |
|------|------|
| `wuyue_vqe_pde.py` | **单比特编码版**：1 qubit/变量，Ising哈密顿量编码 |
| `wuyue_vqe_multibit.py` | **多比特编码版（改进）**：q bits/变量二进制展开，RMSE改善6× |
| `wuyue_outputs/vqe_spd_4x4.json` | 单比特SPD 4×4结果 |
| `wuyue_outputs/vqe_spd_5x5.json` | 单比特SPD 5×5结果 |
| `wuyue_outputs/vqe_indef_4x4.json` | 单比特不定矩阵结果 |
| `wuyue_outputs/vqe_spd_4x4_3bit_multibit.json` | **多比特SPD 4×4 3-bit结果**（RMSE=0.30，vs单比特1.81） |
| `wuyue_outputs/vqe_summary.json` | 单比特实验汇总 |
| `wuyue_outputs/vqe_multibit_summary.json` | 多比特实验汇总 |

**VQE线路设计：** 硬件高效拟设（RY层 + 环形CNOT纠缠 + 最终RY层），L-BFGS-B优化 + 5次随机重启。

**多比特编码：** $x_i = lb + \alpha\sum_k 2^k z_{i,k}$，将每变量映射为$q$个qubit，$2^q$级离散精度。

**运行方式：**
```bash
# 单比特编码（基础版）
E:\xuexi\envs\wuyue\python.exe wuyue_vqe_pde.py

# 多比特编码（改进版，RMSE改善6×）
E:\xuexi\envs\wuyue\python.exe wuyue_vqe_multibit.py
```

**依赖：** WuYue SDK（通用量子计算SDK），已fork并star：https://gitee.com/OpenWuYue/WuYueSDK

---

## 运行环境

- Python 3.11+
- numpy, scipy, matplotlib
- Jupyter Notebook
- Kaiwu SDK（相干光量子计算机，quota真机模式）
- WuYue SDK（通用量子计算机，虚拟环境 `E:\xuexi\envs\wuyue\`）

## 关键引用

- Kaiwu CIM 量子云平台：https://ecloud.10086.cn/portal/product/WYQCLOUD
- WuYue SDK：https://gitee.com/OpenWuYue/WuYueSDK（已fork并star）
- Kaiwu-PyTorch-Plugin：https://github.com/qboson/kaiwu-pytorch-plugin（已fork并star）
- 移动云开发者社区：https://ecloud.10086.cn/api/query/developer/user/home.html
