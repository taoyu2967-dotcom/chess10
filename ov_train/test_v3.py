# -*- coding: utf-8 -*-
# v3 权重布局自检（Python 侧）
#   1) v2 权重可加载，默认 flags 下与 v2 行为一致（额外层 0 层、按级仿射冻结在恒等）
#   2) 存/读对称：v3 导出再加载，前向逐位相同
#   3) 纯增量兼容：policy 头前 100 行 == 权重文件里的旧 100 行（逐位）
#   4) 恒等：把 flags 打开（额外层 2 层零权重、按级仿射 γ=1/β=0）前向仍逐位相同
# 用法: py -3 test_v3.py [weights.bin]
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import az_model as A

BASE = os.environ.get('CHESS10_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = sys.argv[1] if len(sys.argv) > 1 else BASE + '/training/data/snapshots/r160.bin'
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tmp_v3.bin')

print(f'权重: {W}  ({os.path.getsize(W)} 字节 = {os.path.getsize(W)//4} floats)')

# ---- 1) 加载 v2 ----
raw = A.load_weights_bin(W)
print(f'版本: v2={raw["__v2__"]} v3={raw["__v3__"]}  flags={np.array2string(raw["flags"][:6], precision=0)}')
net2 = A.Net(raw).eval()
print(f'n_attn_extra={net2.n_attn_extra}  plevel_trainable={net2.mano_plevel_trainable}  norm_mode={net2.norm_mode}')
print(f'policy 头: {tuple(net2.polo.weight.shape)}  平面={A.PCH * A.N_POS}')

# ---- 3) 纯增量：前 100 行必须逐位等于文件里的旧行 ----
w_old = raw['Wp2'].reshape(A.PCH_LEGACY, 32)
w_new = net2.polo.weight.detach().numpy()[:A.PCH_LEGACY].reshape(A.PCH_LEGACY, 32)
b_old = raw['bp2']; b_new = net2.polo.bias.detach().numpy()[:A.PCH_LEGACY]
d_w = np.abs(w_new - w_old).max(); d_b = np.abs(b_new - b_old).max()
print(f'[3] policy 前 100 行 vs 文件旧行: max|ΔW|={d_w:.3e} max|Δb|={d_b:.3e} → '
      f'{"OK 逐位一致" if d_w == 0 and d_b == 0 else "FAIL"}')

# ---- 探针编码（真实局面）----
pj = os.path.join(BASE, 'training/teacher/parity.json')
if os.path.exists(pj):
    pe = np.array(json.load(open(pj))['encs'], dtype=np.float32)[:3]
else:
    pe = np.random.RandomState(0).randn(3, A.C_IN, 10, 10).astype(np.float32) * 0.1
x = torch.from_numpy(pe.reshape(-1, A.C_IN, 10, 10))


def run(net):
    with torch.no_grad():
        pol, raw_v = net(x)
        return pol.numpy(), raw_v.numpy()


pol2, val2 = run(net2)
print(f'[1] v2 加载前向: value_raw={np.array2string(val2, precision=4)}  finite={np.isfinite(pol2).all() and np.isfinite(val2).all()}')

# ---- 2) 存/读对称 ----
n = A.save_weights_bin(net2, TMP)
print(f'[2] 导出 v3: {n} floats (期望 {A.V3_FLOATS}) → {"OK" if n == A.V3_FLOATS else "FAIL"}')
net3 = A.build_net(TMP)
pol3, val3 = run(net3)
dv = np.abs(val3 - val2).max(); dp = np.abs(pol3 - pol2).max()
print(f'[2] v3 重载 vs v2 前向: max|Δvalue|={dv:.3e} max|Δpolicy|={dp:.3e} → '
      f'{"OK 逐位一致" if dv == 0 and dp == 0 else "FAIL"}')

# ---- 4) 恒等：打开全部新增项（零权重/恒等仿射）----
raw4 = A.load_weights_bin(TMP)
raw4['flags'][A.FLAG_IDX['attn_extra']] = 2.0
raw4['flags'][A.FLAG_IDX['plevel']] = 1.0
# 给额外层/按级仿射填入随机值以检验"零权重才恒等"的边界：先测零权重路径
net4 = A.Net(raw4).eval()
pol4, val4 = run(net4)
dp4 = np.abs(pol4 - pol2).max(); dv4 = np.abs(val4 - val2).max()
print(f'[4] 额外层×2 + 按级仿射(恒等初值): max|Δpolicy|={dp4:.3e} max|Δvalue|={dv4:.3e} → '
      f'{"OK 逐位一致" if dp4 == 0 and dv4 == 0 else "FAIL"}')

# 非恒等随机值必须产生差异（证明开关真的接通，而不是被忽略）
raw5 = A.load_weights_bin(TMP)
raw5['flags'][A.FLAG_IDX['attn_extra']] = 1.0
rs = np.random.RandomState(7)
for k in ('Wq', 'Wk', 'Wv', 'Wo', 'Wff1', 'Wff2'):
    raw5[f'attnX0_{k}'] = (rs.randn(*raw5[f'attnX0_{k}'].shape) * 0.05).astype(np.float32)
raw5['attnX0_Wo'] = (rs.randn(*raw5['attnX0_Wo'].shape) * 0.05).astype(np.float32)
raw5['attnX0_Wff2'] = (rs.randn(*raw5['attnX0_Wff2'].shape) * 0.05).astype(np.float32)
net5 = A.Net(raw5).eval()
pol5, val5 = run(net5)
dp5 = np.abs(pol5 - pol2).max()
print(f'[4] 额外层×1 随机权重: max|Δpolicy|={dp5:.3e} → {"OK 开关生效" if dp5 > 1e-4 else "FAIL 未生效"}')

ok = (d_w == 0 and d_b == 0 and n == A.V3_FLOATS and dv == 0 and dp == 0
      and dp4 == 0 and dv4 == 0 and dp5 > 1e-4)
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
