# -*- coding: utf-8 -*-
# 混合切分精度实测：cnn3@NPU / trunk@NPU（OV fp16）+ 剩余在 torch fp32 CPU，对拍 torch 全 fp32
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
SRV = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'ov_train')))
from az_model import build_net, C_IN, C_HID, HEADS, D_MODEL, HD, D_FF, PCH, N_POS  # noqa: E402
HD = D_MODEL // HEADS

with open(os.path.normpath(os.path.join(HERE, '..', '..', 'training', 'teacher', 'parity.json'))) as fp:
    x = torch.from_numpy(np.array(json.load(fp)['encs'], dtype=np.float32)[:4].reshape(-1, C_IN, 10, 10))

net = build_net(os.path.join(SRV, 'weights_ov.bin')).eval()
with torch.no_grad():
    pol_t, raw_t = net(x)
val_t = np.tanh(raw_t.numpy()); pol_t = pol_t.numpy()

class CNNOnly(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        f = F.relu(self.net.bn0(self.net.conv0(x)))
        for blk in self.net.blocks[:3]:
            f = blk(f)
        return f

class TrunkOnly(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        return self.net.trunk(x)

class TorchTail(nn.Module):
    """OV 输出特征之后的剩余（cnn3 → MANO+3CNN+T+双头；trunk → T+双头）"""
    def __init__(self, net, split):
        super().__init__(); self.net = net; self.split = split
    def forward(self, f):
        n = self.net
        if self.split == 'cnn3':
            f = n.mano(f)
            for blk in n.blocks[3:]:
                f = blk(f)
        t = f.flatten(2).transpose(1, 2)
        q, k, v = n.Wq(t), n.Wk(t), n.Wv(t)
        q = q.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        k = k.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        v = v.view(-1, N_POS, HEADS, HD).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (HD ** 0.5)
        scores = scores + n.rpb[:, n.idx_r, n.idx_c].unsqueeze(0)
        att = torch.softmax(scores, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(-1, N_POS, D_MODEL)
        h = n.ln1(n.Wo(o) + t)
        h2 = n.ln2(n.ff2(F.relu(n.ff1(h))) + h)
        feat = h2.transpose(1, 2).reshape(-1, C_HID, 10, 10)
        pol = n.polo(F.relu(n.polc(feat))).flatten(1)
        hv = F.relu(n.valc(feat)).flatten(1)
        raw = n.vall2(F.relu(n.vall1(hv))).squeeze(-1)
        return pol, raw

core = ov.Core()
core.set_property({'CACHE_DIR': os.path.join(SRV, 'models_ov', 'cache')})
B = 4
for split in ('cnn3', 'trunk'):
    mod = CNNOnly(net).eval() if split == 'cnn3' else TrunkOnly(net).eval()
    m = ov.convert_model(mod, example_input=torch.zeros(B, C_IN, 10, 10), input=(B, C_IN, 10, 10))
    try:
        compiled = core.compile_model(m, 'NPU')
    except Exception as e:
        print(f'{split}@NPU compile FAIL: {str(e)[:140]}', flush=True)
        continue
    res = compiled({0: x.numpy().astype(np.float32)})
    feat = None
    for k, v in res.items():
        if len(v.shape) == 4: feat = v
    tail = TorchTail(net, split).eval()
    with torch.no_grad():
        pol_o, raw_o = tail(torch.from_numpy(np.array(feat).copy()))
    val_o = np.tanh(raw_o.numpy()); pol_o = pol_o.numpy()
    dv = np.abs(val_o - val_t).max(); dp = np.abs(pol_o - pol_t).max()
    top1 = float((pol_o.argmax(1) == pol_t.argmax(1)).mean())
    print(f'{split}@NPU+torch: value maxAbs={dv:.4f}  policy maxAbs={dp:.4f}  top1={top1:.3f}', flush=True)
print('DIAG DONE', flush=True)
