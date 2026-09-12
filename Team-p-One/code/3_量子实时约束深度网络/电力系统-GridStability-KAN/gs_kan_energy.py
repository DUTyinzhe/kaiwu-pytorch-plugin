"""
Grid Stability — KAN + PDE约束 + 能量泛函量子预设
KAN with PDE residual + quantum energy-QUBO preset supervision
"""
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd, warnings, json, os, time
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'D:/QPDE/pde+pinn/outputs_gs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 1. RBF-KAN ==========
class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=8):
        super().__init__()
        self.register_buffer('centers', torch.linspace(-1, 1, grid_size))
        self.width = nn.Parameter(torch.ones(1) * 2.0 / grid_size)
        self.coef = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.1)
        self.base = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        x_exp = x.unsqueeze(-1); c = self.centers.view(1, 1, -1)
        rbf = torch.exp(-((x_exp - c) / self.width)**2)
        return torch.einsum('big,oig->bo', rbf, self.coef) + self.base(x)

class KAN(nn.Module):
    def __init__(self, layers, grid_size=8):
        super().__init__()
        self.layers = nn.ModuleList([KANLayer(layers[i], layers[i+1], grid_size) for i in range(len(layers)-1)])
    def forward(self, x):
        for l in self.layers[:-1]: x = torch.tanh(l(x))
        return self.layers[-1](x)

# ========== 2. 数据 & 预设加载 ==========
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv'
df = pd.read_csv(url)
X = df.drop(columns=['stab', 'stabf']).values.astype(float)
y_raw = df['stabf'].values
y_all = np.where(y_raw == 'unstable', 1, -1)
rng = np.random.default_rng(42)
idx = rng.choice(len(y_all), min(2000, len(y_all)), replace=False)
X_sample, y_sample = X[idx], y_all[idx]

X_2d = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X_sample))
for j in range(2):
    lo, hi = X_2d[:,j].min(), X_2d[:,j].max()
    X_2d[:,j] = (X_2d[:,j] - lo) / (hi - lo) * 0.7 + 0.15

preset_nodes = np.load(f'{OUTPUT_DIR}/preset_nodes_energy.npy')
preset_values = np.load(f'{OUTPUT_DIR}/preset_values_energy.npy')
internal_mask = np.load(f'{OUTPUT_DIR}/internal_mask.npy')
K_ii = np.load(f'{OUTPUT_DIR}/K_ii.npy'); rhs = np.load(f'{OUTPUT_DIR}/rhs.npy')

N = 7; n_i = internal_mask.sum()
u_classical_i = np.linalg.solve(K_ii, rhs)
u_classical = np.zeros(N*N); u_classical[internal_mask] = u_classical_i

X_2d_t = torch.tensor(X_2d, dtype=torch.float32, device=DEVICE)
y_t = torch.tensor(y_sample, dtype=torch.float32, device=DEVICE)
sigma_t = torch.tensor(0.12, dtype=torch.float32, device=DEVICE)
preset_xy = torch.tensor(preset_nodes[internal_mask], dtype=torch.float32, device=DEVICE)
preset_u = torch.tensor(preset_values[internal_mask], dtype=torch.float32, device=DEVICE).view(-1,1)

def source_fn(xy):
    diff = xy.unsqueeze(1) - X_2d_t.unsqueeze(0)
    dist2 = (diff**2).sum(dim=-1)
    return (y_t.unsqueeze(0) * torch.exp(-dist2 / (2*sigma_t**2))).sum(dim=-1)

# ========== 3. KAN 训练 ==========
model = KAN([2, 64, 64, 1], grid_size=8).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
N_col, N_bc, N_epoch = 1024, 256, 1500
w_pde, w_bc, w_preset = 1.0, 10.0, 8.0
nodes_t = torch.tensor(preset_nodes, dtype=torch.float32, device=DEVICE)
u_ref_t = torch.tensor(u_classical, dtype=torch.float32, device=DEVICE).view(-1,1)

t0 = time.time()
for epoch in range(N_epoch):
    col = torch.rand(N_col, 2, device=DEVICE, requires_grad=True)
    bp = torch.rand(N_bc, 1, device=DEVICE); bs = torch.randint(0, 4, (N_bc,), device=DEVICE)
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
    if epoch % 500 == 0:
        with torch.no_grad():
            l2 = (model(nodes_t) - u_ref_t).pow(2).mean().sqrt().item()
        print(f'epoch {epoch}: loss={loss.item():.4e} pde={loss_pde.item():.2e} preset={loss_preset.item():.2e} L2={l2:.4e}')

elapsed = time.time() - t0
print(f'训练完成, 耗时: {elapsed:.1f}s')

with torch.no_grad():
    u_kan = model(nodes_t).cpu().numpy().flatten()
l2 = np.sqrt(np.mean((u_kan - u_classical)**2))
rmse_i = np.sqrt(np.mean((u_kan[internal_mask] - u_classical_i)**2))
print(f'KAN+Energy vs FEM: L2={l2:.4e}, 内部RMSE={rmse_i:.4e}')

np.save(f'{OUTPUT_DIR}/u_kan_energy.npy', u_kan)
res = {'method':'kan_energy_preset','epochs':N_epoch,'time_sec':elapsed,'L2_vs_FEM':float(l2),'RMSE_internal':float(rmse_i)}
with open(f'{OUTPUT_DIR}/result_kan_energy.json','w') as f: json.dump(res, f, indent=2)
print('结果已保存')
