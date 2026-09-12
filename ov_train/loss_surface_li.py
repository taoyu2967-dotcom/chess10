# -*- coding: utf-8 -*-
# Li et al. (2018) 规范版损失面：单一参考点 θ* + 两个独立随机方向（严格逐滤波归一化）
# 用法: py loss_surface_li.py [θ*快照名, 默认r136] [输出json]
# 产出: 41×41 total/value/policy 网格 + 切片局部特征标注(极小/极大/鞍点候选) + 锐度曲线
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = 'D:/data/新建文件夹/chess_game'
SNAP = os.path.join(BASE, 'training', 'data', 'snapshots')
TEACHER = os.path.join(BASE, 'training', 'teacher')
sys.path.insert(0, HERE)
from az_model import build_net, C_IN, PCH, N_POS

REF = sys.argv[1] if len(sys.argv) > 1 else 'r136'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, 'training', 'data', f'loss_li_{REF}.json')
NGRID, SAMPLES, SEED = 41, 3000, 20260901
RANGE = 1.0   # α,β ∈ [-1, 1]

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:', torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', flush=True)

ref_path = os.path.join(SNAP, REF + '.bin')
if not os.path.exists(ref_path):
    ref_path = os.path.join(BASE, 'server', REF if REF.endswith('.bin') else REF + '.bin')
net_ref = build_net(ref_path).to(dev)
named = list(net_ref.named_parameters())
params = [p for _, p in named]
# Li et al. 精神：归一化层（FrozenBN 的 g/b/m/v 与 LayerNorm）不参与扰动——
# 扰动 BN 统计量会灾难性破坏归一化（实测 31/1681 有限格、悬崖主导一切），不是权重几何信息
EXCLUDE = [bool(('bn' in n) or n.startswith('ln')) for n, _ in named]
print('扰动参数张量:', sum(1 for x in EXCLUDE if not x), '/', len(EXCLUDE), '（归一化层已排除）', flush=True)

def load_f32(p): return np.fromfile(p, dtype=np.float32)
X = torch.from_numpy(load_f32(os.path.join(TEACHER, 'train_encs.f32')).reshape(-1, C_IN, 10, 10)[:SAMPLES]).to(dev)
Pi = torch.from_numpy(load_f32(os.path.join(TEACHER, 'train_pis.f32')).reshape(-1, PCH * N_POS)[:SAMPLES]).to(dev)
zs = load_f32(os.path.join(TEACHER, 'train_zs.f32'))[:SAMPLES]
Zt = torch.from_numpy(np.arctanh(np.clip(zs, -0.75, 0.75)).astype(np.float32)).to(dev)
print(f'data {len(zs)} samples', flush=True)

# ---- Li et al. 逐滤波归一化随机方向 ----
# 分组：dim>=2 的张量按第 0 维(输出神经元/卷积核)逐组；1-D 张量整组
gen = torch.Generator().manual_seed(SEED)
def rand_fn_direction():
    parts = []
    for (name, p), excl in zip(named, EXCLUDE):
        if excl:
            parts.append(torch.zeros(p.shape, device=dev))
            continue
        r = torch.randn(p.shape, generator=gen).to(dev)
        with torch.no_grad():
            if p.dim() >= 2:
                pf = p.flatten(1).norm(dim=1)
                rf = r.flatten(1).norm(dim=1)
                scale = (pf / (rf + 1e-10)).view(-1, *([1] * (p.dim() - 1)))
                r = r * scale
            else:
                scale = p.norm() / (r.norm() + 1e-10)
                r = r * scale
        parts.append(r)
    return torch.cat([x.reshape(-1) for x in parts])

d1 = rand_fn_direction()
d2 = rand_fn_direction()
f0 = torch.cat([p.detach().reshape(-1) for p in params])
print(f'|θ*|={f0.norm():.1f} |d1|={d1.norm():.1f} |d2|={d2.norm():.1f}（Li 归一化后应≈|θ*|）', flush=True)

shapes = [p.shape for p in params]
def net_at(flat):
    net = build_net(os.path.join(BASE, 'server', 'weights_ov.bin')).to(dev)
    off = 0
    with torch.no_grad():
        for q, s in zip(net.parameters(), shapes):
            n = q.numel()
            q.copy_(flat[off:off + n].reshape(s))
            off += n
    return net

@torch.no_grad()
def losses(net):
    net.eval()
    vL = pL = tot = 0.0
    for i in range(0, len(zs), 256):
        pol, raw = net(X[i:i+256])
        lv = F.mse_loss(raw, Zt[i:i+256])
        lp = -(Pi[i:i+256] * F.log_softmax(pol, dim=1)).sum(1).mean()
        vL += lv.item() * len(raw); pL += lp.item() * len(raw)
        tot += (1.3 * lv + 2.0 * lp).item() * len(raw)
    n = len(zs)
    return tot / n, vL / n, pL / n

ax = np.linspace(-RANGE, RANGE, NGRID)
t0 = time.time()
Gt = [[0.0]*NGRID for _ in range(NGRID)]
Gv = [[0.0]*NGRID for _ in range(NGRID)]
Gp = [[0.0]*NGRID for _ in range(NGRID)]
for j, b in enumerate(ax):
    for i, a in enumerate(ax):
        t, v, p = losses(net_at(f0 + float(a) * d1 + float(b) * d2))
        Gt[j][i], Gv[j][i], Gp[j][i] = round(t, 5), round(v, 5), round(p, 5)
    print(f'row {j+1}/{NGRID} β={b:+.2f} min={min(Gt[j]):.4f} ({time.time()-t0:.0f}s)', flush=True)

# ---- 切片局部特征标注 ----
def nbrs(G, j, i):
    return [G[jj][ii] for jj in (j-1, j, j+1) if 0 <= jj < NGRID for ii in (i-1, i, i+1) if 0 <= ii < NGRID and not (jj == j and ii == i)]
annos = []
for j in range(1, NGRID-1):
    for i in range(1, NGRID-1):
        ns = nbrs(Gt, j, i)
        c = Gt[j][i]
        if c < min(ns):
            annos.append({'type': 'min', 'a': round(float(ax[i]),3), 'b': round(float(ax[j]),3), 'loss': c})
        elif c > max(ns):
            annos.append({'type': 'max', 'a': round(float(ax[i]),3), 'b': round(float(ax[j]),3), 'loss': c})
        else:
            # 鞍点候选：沿 α 是谷底、沿 β 是峰（或反之）
            alongA = min(Gt[j][i-1], Gt[j][i+1]); alongB = min(Gt[j-1][i], Gt[j+1][i])
            peakA = max(Gt[j][i-1], Gt[j][i+1]); peakB = max(Gt[j-1][i], Gt[j+1][i])
            if (c <= alongA and c >= peakB) or (c <= alongB and c >= peakA):
                annos.append({'type': 'saddle', 'a': round(float(ax[i]),3), 'b': round(float(ax[j]),3), 'loss': c})

# ---- 锐度曲线（从中心 θ* 沿切面的损失增量，Keskar/Li 风格）----
center = Gt[NGRID//2][NGRID//2]
sharp = []
step = 2 * RANGE / (NGRID - 1)
for r in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
    k = int(round(r / step))
    if k > NGRID // 2: k = NGRID // 2
    ring = [Gt[jj][ii] for jj in range(NGRID//2 - k, NGRID//2 + k + 1) for ii in range(NGRID//2 - k, NGRID//2 + k + 1)
            if max(abs(jj - NGRID//2), abs(ii - NGRID//2)) == k]
    sharp.append({'r': r, 'mean': round(sum(ring)/len(ring) - center, 5), 'max': round(max(ring) - center, 5)})

out = {
  'kind': 'li',
  'meta': {'ref': REF, 'samples': int(len(zs)), 'seed': SEED, 'range': [-RANGE, RANGE],
           'lossDef': 'total = 1.3*MSE(value, arctanh z) + 2.0*CE(policy)',
           'method': 'Li et al. 2018 filter-normalized random directions（dim≥2 逐输出滤波归一，1-D 整张量归一）',
           'note': '中心(0,0)=参考点 θ*。全空间驻点(全局最小/真实鞍点)不可见于 2D 切片；标注为切片局部特征。'},
  'alphas': [round(float(a),4) for a in ax],
  'betas': [round(float(b),4) for b in ax],
  'gridTotal': Gt, 'gridValue': Gv, 'gridPolicy': Gp,
  'center': center, 'annotations': annos, 'sharpness': sharp,
}
with open(OUT, 'w', encoding='utf-8') as fp: json.dump(out, fp, ensure_ascii=False)
print('WROTE', OUT, flush=True)
print('center loss =', center, '| annos:', {t: sum(1 for x in annos if x['type']==t) for t in ('min','max','saddle')}, flush=True)
print('sharpness:', json.dumps(sharp), flush=True)
