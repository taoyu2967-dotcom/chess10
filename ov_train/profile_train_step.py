# -*- coding: utf-8 -*-
# 训练侧资源画像：峰值显存 / 每步耗时 / 吞吐 —— 用于回答"这模型需要什么档次的 GPU"
# 用法: py -3 profile_train_step.py [batch] [steps]
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import az_model as A

B = int(sys.argv[1]) if len(sys.argv) > 1 else 256
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
BASE = 'D:/data/新建文件夹/chess_game'
W = os.path.join(BASE, 'server', 'weights_ov.bin')

FORCE_CPU = os.environ.get('CHESS10_FORCE_CPU') == '1'
dev = torch.device('cpu' if (FORCE_CPU or not torch.cuda.is_available()) else 'cuda')
net = A.build_net(W).to(dev)
net.train()
nparam = sum(p.numel() for p in net.parameters())
opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
print(f'DEVICE: {torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"}')
print(f'参数量: {nparam/1e6:.3f} M   batch={B}   输入 {A.C_IN}x10x10   策略面 {A.PCH*A.N_POS}')

x = torch.randn(B, A.C_IN, 10, 10, device=dev)
pi = torch.zeros(B, A.PCH * A.N_POS, device=dev); pi[:, 0] = 1.0
z = torch.zeros(B, device=dev)

if dev.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
t0 = time.time()
for i in range(STEPS):
    pol, raw = net(x)
    loss = 1.3 * F.mse_loss(raw, z) - 2.0 * (pi * F.log_softmax(pol, dim=1)).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
if dev.type == 'cuda':
    torch.cuda.synchronize()
dt = (time.time() - t0) / STEPS

print(f'每步耗时: {dt*1000:.1f} ms   （{B/dt:.0f} 样本/秒）')
if dev.type == 'cuda':
    peak = torch.cuda.max_memory_allocated() / 1024**2
    resv = torch.cuda.max_memory_reserved() / 1024**2
    print(f'峰值显存: allocated {peak:.0f} MB / reserved {resv:.0f} MB')
print(f'单轮 14000 局面 × 1 epoch ≈ {dt*14000/B:.1f} s（batch {B}）')
