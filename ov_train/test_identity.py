# -*- coding: utf-8 -*-
# 恒等测试：v1 权重载入 v2 网络（MANO/GRN/rpb 零初始化）后，前向必须与 v1 网络逐位一致
# 顺带测 v2 权重 save→load 往返一致性
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = 'D:/data/新建文件夹/chess_game'
sys.path.insert(0, HERE)
import az_model
import az_model_v1

torch.manual_seed(0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE:', dev, flush=True)

W_IN = os.path.join(BASE, 'server', 'weights.bin')
# 注：v1 参照加载器（az_model_v1）只接受 v1 长度；server/weights_ov.bin 已是 v2(3,033,545)，
#     自 v2 上线起这里就必须指向 v1 文件，否则本测试必然抛 size mismatch。
# 随机输入（覆盖多局面：确定性 seed 随机张量 + 真 encode）
encs = np.fromfile(os.path.join(BASE, 'training', 'teacher', 'train_encs.f32'), dtype=np.float32).reshape(-1, 24, 10, 10)[:64]
X = torch.from_numpy(encs).to(dev)

net1 = az_model_v1.build_net(W_IN).to(dev)
net2 = az_model.build_net(W_IN).to(dev)   # v1 权重 → v2 结构（应恒等）
assert not net2.mano.Wo.weight.abs().sum().item() > 0, 'MANO Wo 应零初始化'
print('v2 loaded from v1 bin: v2 flag =', az_model.load_weights_bin(W_IN)['__v2__'], flush=True)

with torch.no_grad():
    pol1, raw1 = net1(X)
    pol2, raw2 = net2(X)
# v3：policy 平面扩宽到 160 通道；v1 参照只有旧 100 通道，故只对拍前 100*N_POS 维
# （这正是"纯增量"不变量：旧 100 通道的 logits 必须逐位不变）
LEG = az_model.PCH_LEGACY * az_model.N_POS
print(f'policy 宽度: v1={pol1.shape[1]} v3={pol2.shape[1]} 对拍范围={LEG}', flush=True)
d_pol = (pol1 - pol2[:, :LEG]).abs().max().item()
d_raw = (raw1 - raw2).abs().max().item()
print(f'恒等差: pol max|Δ|={d_pol:.3e}  raw max|Δ|={d_raw:.3e}', flush=True)

# v2 save → load 往返
TMP = os.path.join(HERE, '_tmp_v2.bin')
n = az_model.save_weights_bin(net2, TMP)
W2 = az_model.load_weights_bin(TMP)
print(f'v2 往返: {n} floats, v2 flag = {W2["__v2__"]}', flush=True)
net3 = az_model.Net(W2).to(dev)
with torch.no_grad():
    pol3, raw3 = net3(X)
d2_pol = (pol2 - pol3).abs().max().item()
d2_raw = (raw2 - raw3).abs().max().item()
print(f'v2 往返差: pol max|Δ|={d2_pol:.3e}  raw max|Δ|={d2_raw:.3e}', flush=True)

# 激活测试：把 MANO Wo 与 rpb 随机化后前向必须改变输出（证明新模块真正在计算图中）
with torch.no_grad():
    net2.mano.Wo.weight.normal_(0, 0.02)
    net2.rpb.normal_(0, 0.01)
    for blk in net2.blocks:
        blk.grn.gamma.normal_(0, 0.01)
    pol4, raw4 = net2(X)
act = max((pol2 - pol4).abs().max().item(), (raw2 - raw4).abs().max().item())
print(f'激活测试（扰动后输出应变化）: max|Δ|={act:.3e}', flush=True)

ok = d_pol == 0 and d_raw == 0 and d2_pol == 0 and d2_raw == 0 and act > 1e-4
print('RESULT:', 'PASS' if ok else 'FAIL', flush=True)
try: os.remove(TMP)
except: pass
sys.exit(0 if ok else 1)
