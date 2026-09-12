"""
Halide Perovskite — GCN + 量子预设监督学习
图卷积网络学习: 元素描述符 → PBE 带隙
量子预设提供全局监督信号，弥补少量 DFT 标签的不足
"""
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd, warnings, json, os, time
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'D:/QPDE/photo+kan/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1. GCN (纯 PyTorch, 无外部依赖) ==========
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, adj_norm):
        # adj_norm: 归一化邻接矩阵 (含自环) [N, N]
        x = adj_norm @ x @ self.weight + self.bias
        return x

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        self.conv1 = GCNLayer(in_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        x = self.conv1(x, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj_norm)
        return x.squeeze(-1)

# ========== 2. 加载 Halide Perovskite 数据 ==========
url = 'https://raw.githubusercontent.com/mannodiarun/halide_perovs_design/main/PBE_data.csv'
df = pd.read_csv(url, header=None)
target_all = df.iloc[:, 3].values.astype(float)
features_all = df.iloc[:, 22:58].values.astype(float)

scaler = StandardScaler().fit(features_all)
features_all = scaler.transform(features_all)

# ========== 3. 加载量子预设 ==========
idx_energy = np.load(f'{OUTPUT_DIR}/perov_nodes_energy.npy')
x_quantum_energy = np.load(f'{OUTPUT_DIR}/perov_values_energy.npy')
W_energy = np.load(f'{OUTPUT_DIR}/perov_W_energy.npy')
labeled_mask = np.load(f'{OUTPUT_DIR}/perov_labeled_mask_energy.npy')

idx_ls = np.load(f'{OUTPUT_DIR}/perov_nodes_ls.npy')
x_quantum_ls = np.load(f'{OUTPUT_DIR}/perov_values_ls.npy')
W_ls = np.load(f'{OUTPUT_DIR}/perov_W_ls.npy')

N = len(idx_energy)
feat_sub = features_all[idx_energy]
target_sub = target_all[idx_energy]
labeled_idx = np.where(labeled_mask)[0]
unlabeled_idx = np.where(~labeled_mask)[0]

print(f'节点: {N}, 标记: {len(labeled_idx)}, 未知: {len(unlabeled_idx)}')

# ========== 4. 构建归一化邻接矩阵 ==========
def build_norm_adj(W):
    W_loop = W + np.eye(N)  # 加自环
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W_loop.sum(axis=1)))
    return D_inv_sqrt @ W_loop @ D_inv_sqrt

adj_norm = torch.tensor(build_norm_adj(W_energy), dtype=torch.float32, device=DEVICE)
feat_t = torch.tensor(feat_sub, dtype=torch.float32, device=DEVICE)
target_t = torch.tensor(target_sub, dtype=torch.float32, device=DEVICE)
labeled_t = torch.tensor(labeled_idx, dtype=torch.long, device=DEVICE)
unlabeled_t = torch.tensor(unlabeled_idx, dtype=torch.long, device=DEVICE)

# ========== 5. 训练 GCN + 能量泛函预设 ==========
print('\n=== GCN + 能量泛函预设 ===')
model_e = GCN(36, 64, 1).to(DEVICE)
opt = torch.optim.Adam(model_e.parameters(), lr=1e-2, weight_decay=5e-4)
preset_t_e = torch.tensor(x_quantum_energy, dtype=torch.float32, device=DEVICE)
w_preset = 0.5
N_epoch = 300

t0 = time.time()
for epoch in range(N_epoch):
    model_e.train()
    pred = model_e(feat_t, adj_norm)
    loss_labeled = F.mse_loss(pred[labeled_idx], target_t[labeled_idx])
    loss_preset = F.mse_loss(pred, preset_t_e)
    loss = loss_labeled + w_preset * loss_preset
    opt.zero_grad(); loss.backward(); opt.step()

    if epoch % 100 == 0:
        model_e.eval()
        with torch.no_grad():
            pred_all = model_e(feat_t, adj_norm)
            rmse_labeled = (pred_all[labeled_idx] - target_t[labeled_idx]).pow(2).mean().sqrt().item()
            rmse_unlabeled = (pred_all[unlabeled_idx] - target_t[unlabeled_idx]).pow(2).mean().sqrt().item()
        print(f'epoch {epoch:3d}: loss={loss.item():.4e}, RMSE_labeled={rmse_labeled:.4e}, RMSE_unlabeled={rmse_unlabeled:.4e}')

elapsed = time.time() - t0

model_e.eval()
with torch.no_grad():
    pred_e = model_e(feat_t, adj_norm).cpu().numpy()
rmse_e_all = np.sqrt(np.mean((pred_e - target_sub)**2))
rmse_e_unlabeled = np.sqrt(np.mean((pred_e[unlabeled_idx] - target_sub[unlabeled_idx])**2))
print(f'GCN+能量预设: RMSE_all={rmse_e_all:.4e}, RMSE_unlabeled={rmse_e_unlabeled:.4e}, 耗时={elapsed:.1f}s')

# ========== 6. 训练 GCN + LS 预设 ==========
print('\n=== GCN + LS 预设 ===')
model_ls = GCN(36, 64, 1).to(DEVICE)
opt_ls = torch.optim.Adam(model_ls.parameters(), lr=1e-2, weight_decay=5e-4)
preset_t_ls = torch.tensor(x_quantum_ls, dtype=torch.float32, device=DEVICE)

t0 = time.time()
for epoch in range(N_epoch):
    model_ls.train()
    pred = model_ls(feat_t, adj_norm)
    loss_labeled = F.mse_loss(pred[labeled_idx], target_t[labeled_idx])
    loss_preset = F.mse_loss(pred, preset_t_ls)
    loss = loss_labeled + w_preset * loss_preset
    opt_ls.zero_grad(); loss.backward(); opt_ls.step()

    if epoch % 100 == 0:
        model_ls.eval()
        with torch.no_grad():
            pred_all = model_ls(feat_t, adj_norm)
            rmse_u = (pred_all[unlabeled_idx] - target_t[unlabeled_idx]).pow(2).mean().sqrt().item()
        print(f'epoch {epoch:3d}: loss={loss.item():.4e}, RMSE_unlabeled={rmse_u:.4e}')

elapsed_ls = time.time() - t0

model_ls.eval()
with torch.no_grad():
    pred_ls = model_ls(feat_t, adj_norm).cpu().numpy()
rmse_ls_all = np.sqrt(np.mean((pred_ls - target_sub)**2))
rmse_ls_unlabeled = np.sqrt(np.mean((pred_ls[unlabeled_idx] - target_sub[unlabeled_idx])**2))
print(f'GCN+LS预设: RMSE_all={rmse_ls_all:.4e}, RMSE_unlabeled={rmse_ls_unlabeled:.4e}, 耗时={elapsed_ls:.1f}s')

# ========== 7. 纯监督基线 (仅 10 个标签) ==========
print('\n=== 纯监督基线 (仅标签) ===')
model_b = GCN(36, 64, 1).to(DEVICE)
opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-2, weight_decay=5e-4)

t0 = time.time()
for epoch in range(N_epoch):
    model_b.train()
    pred = model_b(feat_t, adj_norm)
    loss = F.mse_loss(pred[labeled_idx], target_t[labeled_idx])
    opt_b.zero_grad(); loss.backward(); opt_b.step()

    if epoch % 100 == 0:
        model_b.eval()
        with torch.no_grad():
            pred_all = model_b(feat_t, adj_norm)
            rmse_u = (pred_all[unlabeled_idx] - target_t[unlabeled_idx]).pow(2).mean().sqrt().item()
        print(f'epoch {epoch:3d}: loss={loss.item():.4e}, RMSE_unlabeled={rmse_u:.4e}')

elapsed_b = time.time() - t0

model_b.eval()
with torch.no_grad():
    pred_b = model_b(feat_t, adj_norm).cpu().numpy()
rmse_b_all = np.sqrt(np.mean((pred_b - target_sub)**2))
rmse_b_unlabeled = np.sqrt(np.mean((pred_b[unlabeled_idx] - target_sub[unlabeled_idx])**2))
print(f'纯监督基线: RMSE_all={rmse_b_all:.4e}, RMSE_unlabeled={rmse_b_unlabeled:.4e}, 耗时={elapsed_b:.1f}s')

# ========== 8. 结果汇总 ==========
print('\n========== 结果汇总 ==========')
print(f'量子能量解 RMSE vs DFT: {np.sqrt(np.mean((x_quantum_energy - target_sub)**2)):.4e}')
print(f'量子LS解   RMSE vs DFT: {np.sqrt(np.mean((x_quantum_ls - target_sub)**2)):.4e}')
print(f'GCN+能量预设 RMSE_unlabeled: {rmse_e_unlabeled:.4e}')
print(f'GCN+LS预设   RMSE_unlabeled: {rmse_ls_unlabeled:.4e}')
print(f'纯监督基线   RMSE_unlabeled: {rmse_b_unlabeled:.4e}')

results = {
    'quantum_energy_rmse': float(np.sqrt(np.mean((x_quantum_energy - target_sub)**2))),
    'quantum_ls_rmse': float(np.sqrt(np.mean((x_quantum_ls - target_sub)**2))),
    'gnn_energy_rmse_unlabeled': float(rmse_e_unlabeled),
    'gnn_ls_rmse_unlabeled': float(rmse_ls_unlabeled),
    'gnn_baseline_rmse_unlabeled': float(rmse_b_unlabeled),
    'n_labeled': int(len(labeled_idx)),
    'n_unlabeled': int(len(unlabeled_idx)),
    'epochs': N_epoch,
    'time_energy_sec': float(elapsed),
    'time_ls_sec': float(elapsed_ls),
    'time_baseline_sec': float(elapsed_b),
}
with open(f'{OUTPUT_DIR}/result_gnn.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\n结果已保存')
