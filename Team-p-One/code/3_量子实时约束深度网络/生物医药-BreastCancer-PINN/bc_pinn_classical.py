"""
Breast Cancer — PINN + 经典 FEM 对比
PDE 约束: -∇²u = f (Poisson), BC: u=0 on boundary
经典 FEM 仅用于误差对比，不参与训练
"""
import numpy as np
import torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd, warnings, json, os, time
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'D:/QPDE/pde+pinn/outputs_bc'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1. 数据加载 & PCA ==========
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
cols = ['id','clump_thickness','cell_size','cell_shape','marginal_adhesion','epithelial_size',
        'bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
df = pd.read_csv(url, names=cols, na_values='?').dropna().drop(columns=['id'])
X = df.drop(columns=['class']).values.astype(float)
y = np.where(df['class'].values == 4, 1, -1)

X_2d = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
for j in range(2):
    lo, hi = X_2d[:,j].min(), X_2d[:,j].max()
    X_2d[:,j] = (X_2d[:,j] - lo) / (hi - lo) * 0.7 + 0.15

# ========== 2. FEM: 7x7 网格 ==========
N = 7; h = 1.0/(N-1)
GX, GY = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N), indexing='ij')
nodes = np.column_stack([GX.ravel(), GY.ravel()])
boundary = (GX.ravel()==0)|(GX.ravel()==1)|(GY.ravel()==0)|(GY.ravel()==1)
internal = ~boundary; n_i = internal.sum()

sigma = 0.12
f = np.zeros(N*N)
for idx in range(len(y)):
    f += y[idx] * np.exp(-np.sum((nodes - X_2d[idx])**2, axis=1) / (2*sigma**2))

idx_map = -np.ones(N*N, dtype=int); idx_map[internal] = np.arange(n_i)
K_ii = np.zeros((n_i, n_i))
for i in range(1,N-1):
    for j in range(1,N-1):
        ki = idx_map[i*N+j]; K_ii[ki,ki] = 4.0
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            nk = (i+di)*N + (j+dj)
            if internal[nk]: K_ii[ki, idx_map[nk]] = -1.0
rhs = h**2 * f[internal]

# 经典 FEM 解 (仅用于对比)
u_classical_i = np.linalg.solve(K_ii, rhs)
u_classical = np.zeros(N*N); u_classical[internal] = u_classical_i

# 源项转为 tensor 用于 PDE 残差 (在 collocation 点用高斯核计算)
X_2d_t = torch.tensor(X_2d, dtype=torch.float32, device=DEVICE)
y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
sigma_t = torch.tensor(sigma, dtype=torch.float32, device=DEVICE)

def source_fn(xy):
    """高斯核源项 f(x,y) = Σ y_k * exp(-||xy - pk||² / (2σ²))"""
    diff = xy.unsqueeze(1) - X_2d_t.unsqueeze(0)  # [B, M, 2]
    dist2 = (diff**2).sum(dim=-1)
    return (y_t.unsqueeze(0) * torch.exp(-dist2 / (2*sigma_t**2))).sum(dim=-1)

# ========== 3. PINN 网络 ==========
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(4):
            layers.extend([nn.Linear(64,64), nn.Tanh()])
        self.net = nn.Sequential(nn.Linear(2,64), nn.Tanh(), *layers, nn.Linear(64,1))
    def forward(self, x): return self.net(x)

model = PINN().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== 4. 训练 ==========
N_col, N_bc, N_epoch = 1024, 256, 1500
w_pde, w_bc = 1.0, 10.0
hist = {'loss':[], 'l2':[]}

# 固定验证点: FEM 网格节点
nodes_t = torch.tensor(nodes, dtype=torch.float32, device=DEVICE)
u_ref_t = torch.tensor(u_classical, dtype=torch.float32, device=DEVICE).view(-1,1)

t0 = time.time()
for epoch in range(N_epoch):
    # 随机采样
    col = torch.rand(N_col, 2, device=DEVICE, requires_grad=True)
    bc_pts = torch.rand(N_bc, 1, device=DEVICE)
    bc_side = torch.randint(0, 4, (N_bc,), device=DEVICE)
    bc_xy = torch.zeros(N_bc, 2, device=DEVICE)
    # 4 edges: x=0, x=1, y=0, y=1
    mask_0 = (bc_side==0); mask_1 = (bc_side==1); mask_2 = (bc_side==2); mask_3 = (bc_side==3)
    bc_xy[mask_0,0]=0; bc_xy[mask_0,1]=bc_pts[mask_0,0]
    bc_xy[mask_1,0]=1; bc_xy[mask_1,1]=bc_pts[mask_1,0]
    bc_xy[mask_2,0]=bc_pts[mask_2,0]; bc_xy[mask_2,1]=0
    bc_xy[mask_3,0]=bc_pts[mask_3,0]; bc_xy[mask_3,1]=1

    # PDE 残差: -∇²u - f = 0
    u_col = model(col)
    grad = torch.autograd.grad(u_col, col, torch.ones_like(u_col), create_graph=True)[0]
    u_x, u_y = grad[:,0:1], grad[:,1:2]
    u_xx = torch.autograd.grad(u_x, col, torch.ones_like(u_x), create_graph=True)[0][:,0:1]
    u_yy = torch.autograd.grad(u_y, col, torch.ones_like(u_y), create_graph=True)[0][:,1:2]
    f_val = source_fn(col)
    res = -u_xx - u_yy - f_val.view(-1,1)

    # BC
    u_bc = model(bc_xy)

    loss = w_pde * res.pow(2).mean() + w_bc * u_bc.pow(2).mean()

    opt.zero_grad(); loss.backward(); opt.step()
    hist['loss'].append(loss.item())

    if epoch % 500 == 0:
        with torch.no_grad():
            u_pred = model(nodes_t)
            l2 = (u_pred - u_ref_t).pow(2).mean().sqrt().item()
            hist['l2'].append(l2)
        print(f'epoch {epoch}: loss={loss.item():.4e}, L2_vs_FEM={l2:.4e}')

elapsed = time.time() - t0
print(f'训练完成, 耗时: {elapsed:.1f}s')

# ========== 5. 评估 ==========
with torch.no_grad():
    u_pinn = model(nodes_t).cpu().numpy().flatten()
l2_pinn = np.sqrt(np.mean((u_pinn - u_classical)**2))
rmse_internal = np.sqrt(np.mean((u_pinn[internal] - u_classical[internal])**2))

print(f'PINN vs FEM: L2={l2_pinn:.4e}, 内部RMSE={rmse_internal:.4e}')

# 保存
np.save(f'{OUTPUT_DIR}/u_pinn_classical.npy', u_pinn)
res_dict = {'method':'pinn_classical','epochs':N_epoch,'time_sec':elapsed,
            'L2_vs_FEM':float(l2_pinn),'RMSE_internal':float(rmse_internal),
            'final_loss':hist['loss'][-1],'l2_history':hist['l2']}
with open(f'{OUTPUT_DIR}/result_pinn_classical.json','w') as f: json.dump(res_dict, f, indent=2)
print(f'结果已保存到 {OUTPUT_DIR}')
