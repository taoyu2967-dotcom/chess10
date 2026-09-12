# -*- coding: utf-8 -*-
# 损失地貌计算器：两份权重快照间 1D 插值曲线 + 2D 等高线网格（Filter-Normalized）
# 用法: py loss_surface.py <快照A名> <快照B名> [输出json]
# loss 定义与 torch_ov_train.py 完全一致: total = 1.3*MSE(raw, arctanh(clamp z)) + 2.0*CE(pi)
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = 'D:/data/新建文件夹/chess_game'
SNAP = os.path.join(BASE, 'training', 'data', 'snapshots')
TEACHER = os.path.join(BASE, 'training', 'teacher')
sys.path.insert(0, HERE)
from az_model import build_net, load_weights_bin, save_weights_bin, C_IN, PCH, N_POS

SNAP_A = sys.argv[1] if len(sys.argv) > 1 else 'r97'
SNAP_B = sys.argv[2] if len(sys.argv) > 2 else 'r136'
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE, 'training', 'data', f'loss_surface_{SNAP_A}_{SNAP_B}.json')
N1D, N2D_B, SAMPLES = 41, 25, 3000

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:', torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', flush=True)

def load_snap(name):
    p = os.path.join(SNAP, name + '.bin')
    if not os.path.exists(p):
        p2 = os.path.join(SERVER_W := os.path.join(BASE, 'server'), name)
        if os.path.exists(p2):
            p = p2
        else:
            raise SystemExit(f'快照不存在: {name}')
    sd = load_weights_bin(p)
    net = build_net(os.path.join(BASE, 'server', 'weights_ov.bin'))  # 结构模板
    with torch.no_grad():
        for k, v in net.state_dict().items():
            pass
    # 逐张量写入
    def assign(name_t, arr):
        pass
    return sd, net, p

# 直接用 az_model 的加载：build_net(weights_path) 只认路径 → 写临时文件切换权重
TMP = os.path.join(HERE, '_tmp_weights.bin')

def make_net_from_flat(flat_np):
    flat_np.tofile(TMP)
    return build_net(TMP).to(dev)

# 数据（与训练同款预处理）
def load_f32(p): return np.fromfile(p, dtype=np.float32)
X = torch.from_numpy(load_f32(os.path.join(TEACHER, 'train_encs.f32')).reshape(-1, C_IN, 10, 10)[:SAMPLES]).to(dev)
Pi = torch.from_numpy(load_f32(os.path.join(TEACHER, 'train_pis.f32')).reshape(-1, PCH * N_POS)[:SAMPLES]).to(dev)
zs = load_f32(os.path.join(TEACHER, 'train_zs.f32'))[:SAMPLES]
Zt = torch.from_numpy(np.arctanh(np.clip(zs, -0.75, 0.75)).astype(np.float32)).to(dev)
print(f'data: {len(zs)} samples', flush=True)

# 权重平坦向量（state_dict 顺序拼接）
def flat_of(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).cpu()

def net_from_flat(flat, template):
    net = build_net(os.path.join(BASE, 'server', 'weights_ov.bin')).to(dev)
    off = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[off:off + n].reshape(p.shape).to(dev))
            off += n
    return net

@torch.no_grad()
def losses(net):
    net.eval()
    vL, pL, tot, nb = 0.0, 0.0, 0.0, 0
    for i in range(0, len(zs), 256):
        xb = X[i:i+256]; pb = Pi[i:i+256]; zb = Zt[i:i+256]
        pol, raw = net(xb)
        lv = F.mse_loss(raw, zb)
        lp = -(pb * F.log_softmax(pol, dim=1)).sum(1).mean()
        vL += lv.item() * len(xb); pL += lp.item() * len(xb); tot += (1.3 * lv + 2.0 * lp).item() * len(xb); nb += len(xb)
    return tot / nb, vL / nb, pL / nb

netA = build_net(os.path.join(BASE, 'server', 'weights_ov.bin')).to(dev)
fA = flat_of(netA)
netB = build_net(os.path.join(SNAP, SNAP_B + '.bin')).to(dev)
netA2 = build_net(os.path.join(SNAP, SNAP_A + '.bin')).to(dev)
fB = flat_of(netB)
fA_old = flat_of(netA2)
print(f'A={SNAP_A} ({fA_old.numel()} params)  B={SNAP_B}', flush=True)

# 方向1：B - A_old（训练走过的路）；方向2：filter-normalized 随机方向
delta1 = fB - fA_old
gen = torch.Generator().manual_seed(42)
delta2 = []
for p in netA.parameters():
    pc = p.detach().cpu()
    r = torch.randn(pc.shape, generator=gen)
    norm_p = pc.norm()
    delta2.append((r / (r.norm() + 1e-12)) * norm_p)
delta2 = torch.cat([d.reshape(-1) for d in delta2]).cpu()
n2n = delta2.norm()
print(f'|delta1|={delta1.norm():.1f} |delta2|={n2n:.1f} (before norm)', flush=True)
scale = delta1.norm() / (n2n + 1e-12)
delta2 = delta2 * scale   # 与训练位移同尺度

def eval_at(alpha, beta):
    f = fA_old + alpha * delta1 + beta * delta2
    net = net_from_flat(f, netA)
    t, v, p = losses(net)
    return t, v, p

# 1D 曲线：A_old -> B（延伸到 -0.2 / 1.4）
t0 = time.time()
curve = []
alphas = np.linspace(-0.2, 1.4, N1D)
for i, a in enumerate(alphas):
    t, v, p = eval_at(float(a), 0.0)
    curve.append({'a': round(float(a), 4), 'total': round(t, 5), 'value': round(v, 5), 'policy': round(p, 5)})
    if i % 10 == 0: print(f'1D {i}/{N1D} a={a:.2f} total={t:.4f}', flush=True)
print(f'1D done {time.time()-t0:.0f}s', flush=True)

# 2D 网格：alpha x beta
t0 = time.time()
betas = np.linspace(-0.6, 0.6, N2D_B)
grid_t = [[0.0] * N1D for _ in range(N2D_B)]
for j, b in enumerate(betas):
    for i, a in enumerate(alphas):
        t, v, p = eval_at(float(a), float(b))
        grid_t[j][i] = round(t, 5)
    print(f'2D row {j+1}/{N2D_B} beta={b:.2f} min={min(grid_t[j]):.4f} ({time.time()-t0:.0f}s)', flush=True)

out = {
    'meta': {
        'snapA': SNAP_A, 'snapB': SNAP_B, 'samples': int(len(zs)),
        'lossDef': 'total = 1.3*MSE(value, arctanh z) + 2.0*CE(policy)',
        'alphaRange': [float(alphas[0]), float(alphas[-1])],
        'betaRange': [float(betas[0]), float(betas[-1])],
        'note': 'alpha=0 -> snapA; alpha=1 -> snapB; beta=0 平面 = 训练轨迹所在截面',
    },
    'alphas': [round(float(a), 4) for a in alphas],
    'betas': [round(float(b), 3) for b in betas],
    'curve': curve,
    'gridTotal': grid_t,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as fp:
    json.dump(out, fp, ensure_ascii=False)
print('WROTE', OUT, os.path.getsize(OUT), 'bytes', flush=True)
try: os.remove(TMP)
except: pass
