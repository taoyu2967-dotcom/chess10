# -*- coding: utf-8 -*-
# Li et al. (2018) 规范版损失面 · v2 架构适配版
# 单一参考点 θ* + 两个独立随机方向（严格逐滤波归一化）；归一化类张量不扰动。
# 用法: py loss_surface_li_v2.py [θ*快照名, 默认r140] [输出json]
# 产出: 41×41 total/value/policy 网格 + 切片局部特征标注 + 锐度曲线
#       （JSON → training/build_landscape_html_v2.js 生成三维交互页）
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('CHESS10_ROOT') or os.path.dirname(HERE)   # 路径中枢约定
SNAP = os.path.join(BASE, 'training', 'data', 'snapshots')
DATA = os.path.join(BASE, 'training', 'data')
sys.path.insert(0, HERE)
from az_model import build_net, load_weights_bin, C_IN, PCH, N_POS

REF = sys.argv[1] if len(sys.argv) > 1 else 'r140'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA, f'loss_li_v2_{REF}.json')
NGRID, SAMPLES, SEED = 41, 3000, 20260901
RANGE = 1.0   # α,β ∈ [-1, 1]

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:', torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', flush=True)

ref_path = os.path.join(SNAP, REF + '.bin')
if not os.path.exists(ref_path):
    ref_path = os.path.join(BASE, 'server', REF if REF.endswith('.bin') else REF + '.bin')
Wd = load_weights_bin(ref_path)
is_v2 = bool(Wd.get('__v2__'))
net = build_net(ref_path).to(dev)
named = list(net.named_parameters())
params = [p for _, p in named]

# Li et al. 精神：归一化层不参与扰动——FrozenBN 的 g/b/m/v（bn0/blocks.*.bn1/bn2）、
# 全部 LayerNorm 仿射（ln1/ln2 顶层 + mano.ln，后者名字含 '.ln.' 不以 ln 开头，须单独匹配）、
# GRN γ（归一化仿射，自归一化量的缩放因子）。
# 其余全部扰动（conv/linear 权重与 bias、MANO QKV/D/U、rpb），dim≥2 逐输出滤波归一，1-D 整张量归一。
def is_norm(name):
    return ('bn' in name) or name.startswith('ln') or ('.ln.' in name) or ('grn.gamma' in name)
EXCLUDE = [is_norm(n) for n, _ in named]
n_perturb = sum(1 for x in EXCLUDE if not x)
print(f'arch={"v2" if is_v2 else "v1"} snapshot={ref_path}', flush=True)
print(f'扰动参数张量: {n_perturb}/{len(EXCLUDE)}（BN/LN/GRNγ 已排除）', flush=True)

# 数据：θ* 所在轮次的训练数据，固定种子随机抽 SAMPLES
def load_f32(p): return np.fromfile(p, dtype=np.float32)
enc = load_f32(os.path.join(DATA, f'{REF}_encs.f32')).reshape(-1, C_IN, 10, 10)
pis = load_f32(os.path.join(DATA, f'{REF}_pis.f32')).reshape(-1, PCH * N_POS)
zs = load_f32(os.path.join(DATA, f'{REF}_zs.f32'))
n_all = int(len(zs))
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(n_all, min(SAMPLES, n_all), replace=False))
X = torch.from_numpy(enc[idx]).to(dev)
Pi = torch.from_numpy(pis[idx]).to(dev)
Zt = torch.from_numpy(np.arctanh(np.clip(zs[idx], -0.75, 0.75)).astype(np.float32)).to(dev)
print(f'data {REF}: {n_all} samples, 抽样 {len(idx)}', flush=True)

# ---- Li et al. 逐滤波归一化随机方向 ----
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

# 一次建网，逐点就地拷贝参数（避免每网格点重建模块/读盘）
shapes = [p.shape for p in params]
def set_flat(flat):
    off = 0
    with torch.no_grad():
        for q, s in zip(params, shapes):
            n = q.numel()
            q.copy_(flat[off:off + n].reshape(s))
            off += n

@torch.no_grad()
def losses():
    net.eval()
    vL = pL = tot = 0.0
    for i in range(0, len(idx), 1024):
        pol, raw = net(X[i:i+1024])
        lv = F.mse_loss(raw, Zt[i:i+1024])
        lp = -(Pi[i:i+1024] * F.log_softmax(pol, dim=1)).sum(1).mean()
        vL += lv.item() * len(raw); pL += lp.item() * len(raw)
        tot += (1.3 * lv + 2.0 * lp).item() * len(raw)
    n = len(idx)
    return tot / n, vL / n, pL / n

ax = np.linspace(-RANGE, RANGE, NGRID)
t0 = time.time()
Gt = [[0.0]*NGRID for _ in range(NGRID)]
Gv = [[0.0]*NGRID for _ in range(NGRID)]
Gp = [[0.0]*NGRID for _ in range(NGRID)]
for j, b in enumerate(ax):
    for i, a in enumerate(ax):
        set_flat(f0 + float(a) * d1 + float(b) * d2)
        t, v, p = losses()
        Gt[j][i], Gv[j][i], Gp[j][i] = round(t, 5), round(v, 5), round(p, 5)
    print(f'row {j+1}/{NGRID} β={b:+.2f} min={min(Gt[j]):.4f} ({time.time()-t0:.0f}s)', flush=True)
set_flat(f0)   # 恢复 θ*

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
  'kind': 'li-v2',
  'meta': {'ref': REF, 'arch': 'v2' if is_v2 else 'v1', 'snapshot': ref_path,
           'nParams': int(f0.numel()), 'perturbTensors': n_perturb,
           'samples': int(len(idx)), 'dataPool': n_all, 'seed': SEED, 'range': [-RANGE, RANGE],
           'lossDef': 'total = 1.3*MSE(value, arctanh z) + 2.0*CE(policy)',
           'method': 'Li et al. 2018 filter-normalized random directions（dim≥2 逐输出滤波归一，1-D 整张量归一；BN/LN/GRNγ 不扰动）',
           'note': '中心(0,0)=参考点 θ*（r140 训练权重）。全空间驻点(全局最小/真实鞍点)不可见于 2D 切片；标注为切片局部特征。'},
  'alphas': [round(float(a),4) for a in ax],
  'betas': [round(float(b),4) for b in ax],
  'gridTotal': Gt, 'gridValue': Gv, 'gridPolicy': Gp,
  'center': center, 'annotations': annos, 'sharpness': sharp,
}
with open(OUT, 'w', encoding='utf-8') as fp: json.dump(out, fp, ensure_ascii=False)
print('WROTE', OUT, flush=True)
print('center loss =', center, '| annos:', {t: sum(1 for x in annos if x['type']==t) for t in ('min','max','saddle')}, flush=True)
print('sharpness:', json.dumps(sharp), flush=True)
