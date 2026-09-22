# -*- coding: utf-8 -*-
# 训练步模块级热点（hook + cuda event）：手写 CUDA 算子的立项依据
# 用法: py -3 profile_train_modules.py [batch=256]
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import az_model as A

B = int(sys.argv[1]) if len(sys.argv) > 1 else 256
STEPS = 30
W = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', 'BJ1_r208_v3.bin')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = A.build_net(W).to(dev)
net.train()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4)

x = torch.randn(B, A.C_IN, 10, 10, device=dev)
pi = torch.zeros(B, A.PCH * A.N_POS, device=dev); pi[:, 0] = 1.0
z = torch.zeros(B, device=dev)

def step():
    pol, raw = net(x)
    loss = 1.3 * F.mse_loss(raw, z) - 2.0 * (pi * F.log_softmax(pol, dim=1)).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()

for _ in range(5):
    step()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(STEPS):
    step()
torch.cuda.synchronize()
print(f'每步墙钟: {(time.time()-t0)/STEPS*1000:.1f} ms (B={B})')

# ---- 前向 only ----
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(STEPS):
        pol, raw = net(x)
torch.cuda.synchronize()
fwd = (time.time()-t0)/STEPS*1000
print(f'前向: {fwd:.1f} ms  → 反向+优化器 ≈ {(time.time()-t0)/STEPS*1000:.1f} 之外的 {( (time.time()-0)/0 if False else 0):.0f}')

# ---- 模块级前向计时（cuda event, 前 3 段结构）----
groups = {}
def mk(name, mod):
    def h(m, inp, out):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); 
        yield
    return h

# 更直接：按顺序手动分段重放前向
with torch.no_grad():
    n = net
    # conv0+bn0
    segs = {}
    def timeit(fn, reps=STEPS):
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(reps):
            out = fn()
        torch.cuda.synchronize()
        return (time.time()-t0)/reps*1000
    f0 = lambda: F.relu(n.bn0(n.conv0(x)))
    segs['conv0+bn0+relu'] = timeit(f0)
    f1 = lambda: [blk for blk in [n.blocks[0], n.blocks[1], n.blocks[2]]][-1](n.blocks[1](n.blocks[0](f0())))
    segs['resblocks0-2(conv+bn+grn)'] = timeit(lambda: n.blocks[2](n.blocks[1](n.blocks[0](f0()))))
    m3 = n.blocks[3](n.blocks[2](n.blocks[1](n.blocks[0](f0()))))
    segs['block3(MANO)'] = timeit(lambda: n.blocks[3](n.blocks[2](n.blocks[1](n.blocks[0](f0())))))
    base4 = n.blocks[3](n.blocks[2](n.blocks[1](n.blocks[0](f0()))))
    extra = getattr(n, 'n_attn_extra', 0)
    cur = base4
    for i in range(extra):
        cur = getattr(n, f'attnX{i}')(cur)
    if extra:
        segs['attnX(额外注意力)'] = timeit(lambda: [ (cur := getattr(n, f'attnX{i}')(base4)) for i in range(extra)][-1])
    # transformer（Net.forward 内联段）
    def trans():
        t = base4.flatten(2).transpose(1, 2)
        q, k, v = n.Wq(t), n.Wk(t), n.Wv(t)
        q = q.view(-1, A.N_POS, A.HEADS, A.HD).transpose(1, 2)
        k = k.view(-1, A.N_POS, A.HEADS, A.HD).transpose(1, 2)
        v = v.view(-1, A.N_POS, A.HEADS, A.HD).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (A.HD ** 0.5)
        scores = scores + n.rpb[:, n.idx_r, n.idx_c].unsqueeze(0)
        att = torch.softmax(scores, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(-1, A.N_POS, A.D_MODEL)
        h = n.ln1(n.Wo(o) + t)
        return n.ln2(n.ff2(F.relu(n.ff1(h))) + h)
    segs['transformer(内联)'] = timeit(trans)
    h2 = trans()
    feat = h2.transpose(1, 2).reshape(-1, A.C_HID, 10, 10)
    segs['heads(pol+val卷积头)'] = timeit(lambda: (
        n.polo(F.relu(n.polc(feat))).flatten(1),
        n.vall2(F.relu(n.vall1(F.relu(n.valc(feat)).flatten(1)))).squeeze(-1)))

for k, v in segs.items():
    print(f'  {k:28s} {v:6.2f} ms (B={B})')

# ---- torch.compile 对照（判断手写 CUDA 有多少空间被编译器吃掉）----
try:
    cnet = torch.compile(net)
    def cstep():
        pol, raw = cnet(x)
        loss = 1.3 * F.mse_loss(raw, z) - 2.0 * (pi * F.log_softmax(pol, dim=1)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(8):
        cstep()   # 编译预热
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(STEPS):
        cstep()
    torch.cuda.synchronize()
    print(f'torch.compile 每步: {(time.time()-t0)/STEPS*1000:.1f} ms')
except Exception as e:
    print('torch.compile 失败:', str(e)[:200])
