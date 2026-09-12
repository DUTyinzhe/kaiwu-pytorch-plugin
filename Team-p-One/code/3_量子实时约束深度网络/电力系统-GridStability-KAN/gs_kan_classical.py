"""
Grid Stability — KAN 纯监督学习 (无 PDE 约束)
在 FEM 经典解上做监督学习: (x,y) → u_FEM(x,y)
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

# ========== 1. RBF-KAN 实现 ==========
class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, grid_size=8):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.register_buffer('centers', torch.linspace(-1, 1, grid_size))
        self.width = nn.Parameter(torch.ones(1) * 2.0 / grid_size)
        self.coef = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.1)
        self.base = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x_exp = x.unsqueeze(-1)
        c = self.centers.view(1, 1, -1)
        rbf = torch.exp(-((x_exp - c) / self.width)**2)
        out = torch.einsum('big,oig->bo', rbf, self.coef)
        return out + self.base(x)

class KAN(nn.Module):
    def __init__(self, layers, grid_size=8):
        super().__init__()
        self.layers = nn.ModuleList([KANLayer(layers[i], layers[i+1], grid_size) for i in range(len(layers)-1)])
    def forward(self, x):
        for l in self.layers[:-1]: x = torch.tanh(l(x))
        return self.layers[-1](x)

# ========== 2. 数据加载 ==========
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv'
df = pd.read_csv(url)
X = df.drop(columns=['stab', 'stabf']).values.astype(float)
y_raw = df['stabf'].values
y = np.where(y_raw == 'unstable', 1, -1)
rng = np.random.default_rng(42)
idx = rng.choice(len(y), min(2000, len(y)), replace=False)
X_sample, y_sample = X[idx], y[idx]

X_2d = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X_sample))
for j in range(2):
    lo, hi = X_2d[:,j].min(), X_2d[:,j].max()
    X_2d[:,j] = (X_2d[:,j] - lo) / (hi - lo) * 0.7 + 0.15

# ========== 3. FEM 经典解 (监督标签) ==========
N = 7; h = 1.0/(N-1)
GX, GY = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N), indexing='ij')
nodes = np.column_stack([GX.ravel(), GY.ravel()])
boundary = (GX.ravel()==0)|(GX.ravel()==1)|(GY.ravel()==0)|(GY.ravel()==1)
internal = ~boundary; n_i = internal.sum()

sigma = 0.12
f = np.zeros(N*N)
for idx_k in range(len(y_sample)):
    f += y_sample[idx_k] * np.exp(-np.sum((nodes - X_2d[idx_k])**2, axis=1) / (2*sigma**2))

idx_map = -np.ones(N*N, dtype=int); idx_map[internal] = np.arange(n_i)
K_ii = np.zeros((n_i, n_i))
for i in range(1,N-1):
    for j in range(1,N-1):
        ki = idx_map[i*N+j]; K_ii[ki,ki] = 4.0
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            nk = (i+di)*N + (j+dj)
            if internal[nk]: K_ii[ki, idx_map[nk]] = -1.0
rhs = h**2 * f[internal]
u_classical_i = np.linalg.solve(K_ii, rhs)
u_classical = np.zeros(N*N); u_classical[internal] = u_classical_i

# ========== 4. KAN 训练 (纯监督) ==========
nodes_t = torch.tensor(nodes, dtype=torch.float32, device=DEVICE)
u_fem_t = torch.tensor(u_classical, dtype=torch.float32, device=DEVICE).view(-1,1)

model = KAN([2, 64, 64, 1], grid_size=8).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
N_epoch = 1500

t0 = time.time()
for epoch in range(N_epoch):
    u_pred = model(nodes_t)
    loss = F.mse_loss(u_pred, u_fem_t)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 500 == 0:
        with torch.no_grad():
            l2 = (u_pred - u_fem_t).pow(2).mean().sqrt().item()
        print(f'epoch {epoch}: loss={loss.item():.4e}, L2_vs_FEM={l2:.4e}')

elapsed = time.time() - t0
print(f'训练完成, 耗时: {elapsed:.1f}s')

with torch.no_grad():
    u_kan = model(nodes_t).cpu().numpy().flatten()
l2 = np.sqrt(np.mean((u_kan - u_classical)**2))
rmse_i = np.sqrt(np.mean((u_kan[internal] - u_classical_i)**2))
print(f'KAN vs FEM: L2={l2:.4e}, 内部RMSE={rmse_i:.4e}')

np.save(f'{OUTPUT_DIR}/u_kan_classical.npy', u_kan)
res = {'method':'kan_classical','epochs':N_epoch,'time_sec':elapsed,'L2_vs_FEM':float(l2),'RMSE_internal':float(rmse_i)}
with open(f'{OUTPUT_DIR}/result_kan_classical.json','w') as f: json.dump(res, f, indent=2)
print('结果已保存')
