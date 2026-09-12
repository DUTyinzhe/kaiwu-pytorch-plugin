"""
Breast Cancer — PINN + 能量泛函量子预设
PDE 约束 + 量子 CIM 预设解作为监督锚点
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

# ========== 1. 数据 & 量子预设加载 ==========
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

# 加载量子预设 (由 bc_energy_qubo.ipynb 生成)
preset_nodes = np.load(f'{OUTPUT_DIR}/preset_nodes_energy.npy')
preset_values = np.load(f'{OUTPUT_DIR}/preset_values_energy.npy')
internal_mask = np.load(f'{OUTPUT_DIR}/internal_mask.npy')
K_ii = np.load(f'{OUTPUT_DIR}/K_ii.npy')
rhs = np.load(f'{OUTPUT_DIR}/rhs.npy')

# 经典 FEM 解 (仅对比)
N = 7; n_i = internal_mask.sum()
u_classical_i = np.linalg.solve(K_ii, rhs)
u_classical = np.zeros(len(preset_values)); u_classical[internal_mask] = u_classical_i

# ========== 2. 源项 & 预设 tensor ==========
X_2d_t = torch.tensor(X_2d, dtype=torch.float32, device=DEVICE)
y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
sigma = 0.12; sigma_t = torch.tensor(sigma, dtype=torch.float32, device=DEVICE)
preset_xy = torch.tensor(preset_nodes[internal_mask], dtype=torch.float32, device=DEVICE)
preset_u = torch.tensor(preset_values[internal_mask], dtype=torch.float32, device=DEVICE).view(-1,1)

def source_fn(xy):
    diff = xy.unsqueeze(1) - X_2d_t.unsqueeze(0)
    dist2 = (diff**2).sum(dim=-1)
    return (y_t.unsqueeze(0) * torch.exp(-dist2 / (2*sigma_t**2))).sum(dim=-1)

# ========== 3. PINN ==========
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(4): layers.extend([nn.Linear(64,64), nn.Tanh()])
        self.net = nn.Sequential(nn.Linear(2,64), nn.Tanh(), *layers, nn.Linear(64,1))
    def forward(self, x): return self.net(x)

model = PINN().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== 4. 训练 ==========
N_col, N_bc, N_epoch = 1024, 256, 1500
w_pde, w_bc, w_preset = 1.0, 10.0, 8.0
nodes_t = torch.tensor(preset_nodes, dtype=torch.float32, device=DEVICE)
u_ref_t = torch.tensor(u_classical, dtype=torch.float32, device=DEVICE).view(-1,1)
hist = {'loss':[], 'l2':[]}

t0 = time.time()
for epoch in range(N_epoch):
    col = torch.rand(N_col, 2, device=DEVICE, requires_grad=True)
    bp = torch.rand(N_bc, 1, device=DEVICE)
    bs = torch.randint(0, 4, (N_bc,), device=DEVICE)
    bc_xy = torch.zeros(N_bc, 2, device=DEVICE)
    bc_xy[bs==0,0]=0; bc_xy[bs==0,1]=bp[bs==0,0]
    bc_xy[bs==1,0]=1; bc_xy[bs==1,1]=bp[bs==1,0]
    bc_xy[bs==2,0]=bp[bs==2,0]; bc_xy[bs==2,1]=0
    bc_xy[bs==3,0]=bp[bs==3,0]; bc_xy[bs==3,1]=1

    u_col = model(col)
    grad = torch.autograd.grad(u_col, col, torch.ones_like(u_col), create_graph=True)[0]
    u_x, u_y = grad[:,0:1], grad[:,1:2]
    u_xx = torch.autograd.grad(u_x, col, torch.ones_like(u_x), create_graph=True)[0][:,0:1]
    u_yy = torch.autograd.grad(u_y, col, torch.ones_like(u_y), create_graph=True)[0][:,1:2]
    f_val = source_fn(col)

    loss_pde = (-u_xx - u_yy - f_val.view(-1,1)).pow(2).mean()
    loss_bc = model(bc_xy).pow(2).mean()
    loss_preset = (model(preset_xy) - preset_u).pow(2).mean()
    loss = w_pde*loss_pde + w_bc*loss_bc + w_preset*loss_preset

    opt.zero_grad(); loss.backward(); opt.step()
    hist['loss'].append(loss.item())

    if epoch % 500 == 0:
        with torch.no_grad():
            l2 = (model(nodes_t) - u_ref_t).pow(2).mean().sqrt().item()
            hist['l2'].append(l2)
        print(f'epoch {epoch}: loss={loss.item():.4e} pde={loss_pde.item():.2e} preset={loss_preset.item():.2e} L2={l2:.4e}')

elapsed = time.time() - t0
print(f'训练完成, 耗时: {elapsed:.1f}s')

# ========== 5. 评估 ==========
with torch.no_grad():
    u_pinn = model(nodes_t).cpu().numpy().flatten()
l2_pinn = np.sqrt(np.mean((u_pinn - u_classical)**2))
rmse_i = np.sqrt(np.mean((u_pinn[internal_mask] - u_classical_i)**2))
print(f'PINN+Energy vs FEM: L2={l2_pinn:.4e}, 内部RMSE={rmse_i:.4e}')

np.save(f'{OUTPUT_DIR}/u_pinn_energy.npy', u_pinn)
res = {'method':'pinn_energy_preset','epochs':N_epoch,'time_sec':elapsed,
       'L2_vs_FEM':float(l2_pinn),'RMSE_internal':float(rmse_i),'final_loss':hist['loss'][-1]}
with open(f'{OUTPUT_DIR}/result_pinn_energy.json','w') as f: json.dump(res, f, indent=2)
print('结果已保存')
