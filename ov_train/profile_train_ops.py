# -*- coding: utf-8 -*-
# 训练步算子级热点剖析（torch.profiler）：为手写 CUDA 算子立项提供数据
# 用法: py -3 profile_train_ops.py [batch=256] [steps=12] [权重路径]
import os, sys, time
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import az_model as A

B = int(sys.argv[1]) if len(sys.argv) > 1 else 256
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
W = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', 'BJ1_r208_v3.bin')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = A.build_net(W).to(dev)
net.train()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
print(f'DEVICE={torch.cuda.get_device_name(0) if dev.type=="cuda" else "CPU"} B={B} arch_flags={getattr(net, "flags", None) is not None}')

x = torch.randn(B, A.C_IN, 10, 10, device=dev)
pi = torch.zeros(B, A.PCH * A.N_POS, device=dev); pi[:, 0] = 1.0
z = torch.zeros(B, device=dev)

def step():
    pol, raw = net(x)
    loss = 1.3 * F.mse_loss(raw, z) - 2.0 * (pi * F.log_softmax(pol, dim=1)).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# 预热
for _ in range(5):
    step()
torch.cuda.synchronize()

# 基线墙钟
t0 = time.time()
for _ in range(STEPS):
    step()
torch.cuda.synchronize()
wall = (time.time() - t0) / STEPS * 1000
print(f'每步墙钟: {wall:.1f} ms  (B={B})')

# 算子级剖析
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()

print('\n===== CUDA kernel Top 20（按 GPU 总耗时）=====')
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20, max_name_column_width=60))

# 前向/反向拆分粗估
ka = prof.key_averages()
fwd_kernels = [k for k in ka if 'backward' not in k.key.lower() and k.device_time_total > 0]
bwd_kernels = [k for k in ka if 'backward' in k.key.lower() and k.device_time_total > 0]
tf = sum(k.device_time_total for k in fwd_kernels) / STEPS / 1000
tb = sum(k.device_time_total for k in bwd_kernels) / STEPS / 1000
print(f'kernel 时间粗分: 前向≈{tf:.1f}ms/步  反向≈{tb:.1f}ms/步')
