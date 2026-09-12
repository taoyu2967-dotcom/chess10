# -*- coding: utf-8 -*-
# v2.3 诊断：NPU 算子剖析 + 原生批触发（哑输出破条件）+ BN 折叠 + Arc iGPU 对照
import os, sys, time

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
import torch.nn as nn
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
SRV = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'ov_train')))
from az_model import build_net, C_IN, C_HID  # noqa: E402

net = build_net(os.path.join(SRV, 'weights_ov.bin')).eval()

class TrunkOnly(nn.Module):
    def __init__(self, net_):
        super().__init__(); self.net = net_
    def forward(self, x):
        return self.net.trunk(x)

class TrunkDummyOut(TrunkOnly):
    """第二输出对 batch 维求均值（无 batch 轴）→ 打破'输出同批'条件 → 触发原生 batch 编译"""
    def forward(self, x):
        f = self.net.trunk(x)
        return f, f.mean(dim=0)

def fold_bn(net_):
    """FrozenBN 折入前置卷积（权重数学等价）：y=BN(conv(x)) → conv'(x)"""
    import copy
    n = copy.deepcopy(net_)
    def fold(conv, bn):
        s = bn.g / torch.sqrt(bn.v + 1e-5)
        conv.weight.data *= s.view(-1, 1, 1, 1)
        conv.bias.data = conv.bias.data * s + (bn.b - bn.m * s)
    fold(n.conv0, n.bn0)
    for blk in n.blocks:
        fold(blk.conv1, blk.bn1)
        fold(blk.conv2, blk.bn2)
    return n

net_folded_src = fold_bn(net)

class TrunkOnlyFolded(nn.Module):
    def __init__(self, net_):
        super().__init__(); self.net = net_
    def forward(self, x):
        f = F.relu(self.net.bn0(self.net.conv0(x)))   # bn 已折入 conv，模块仍会被调用？→ 不行，改走原始块
        return f

core = ov.Core()
core.set_property({'CACHE_DIR': os.path.join(SRV, 'models_ov', 'cache')})

import torch.nn.functional as F  # noqa: E402

class TrunkFolded(nn.Module):
    """BN 折叠版主干：conv 权重已折，forward 不再过 BN 模块"""
    def __init__(self, net_):
        super().__init__(); self.net = net_
        # 折叠写回：conv0
        s = net_.bn0.g.view(-1) / torch.sqrt(net_.bn0.v.view(-1) + 1e-5)
        net_.conv0.weight.data *= s.view(-1, 1, 1, 1)
        net_.conv0.bias.data = net_.conv0.bias.data * s + (net_.bn0.b.view(-1) - net_.bn0.m.view(-1) * s)
        for blk in net_.blocks:
            for conv, bn in ((blk.conv1, blk.bn1), (blk.conv2, blk.bn2)):
                s2 = bn.g.view(-1) / torch.sqrt(bn.v.view(-1) + 1e-5)
                conv.weight.data *= s2.view(-1, 1, 1, 1)
                conv.bias.data = conv.bias.data * s2 + (bn.b.view(-1) - bn.m.view(-1) * s2)
        # 折叠后 FrozenBN 置恒等（g=1,b=0,m=0,v=1）
        for mod in (net_.bn0,) + tuple(b.bn1 for b in net_.blocks) + tuple(b.bn2 for b in net_.blocks):
            mod.g.data.fill_(1); mod.b.data.fill_(0); mod.m.data.fill_(0); mod.v.data.fill_(1)
    def forward(self, x):
        n = self.net
        f = F.relu(n.bn0(n.conv0(x)))
        for blk in n.blocks[:3]:
            f = blk(f)
        f = n.mano(f)
        for blk in n.blocks[3:]:
            f = blk(f)
        return f

def convert(module, b):
    return ov.convert_model(module.eval(), example_input=torch.zeros(b, C_IN, 10, 10),
                            input=(b, C_IN, 10, 10))

def bench(comp, b, R=20, profile=False, tag=''):
    ireq = comp.create_infer_request()
    xb = np.zeros((b, C_IN, 10, 10), dtype=np.float32)
    it = ireq.get_input_tensor(); it.data[:] = xb
    ireq.infer()
    if profile:
        try:
            pis = ireq.get_profiling_info()
            tops = sorted(pis, key=lambda p: p.real_time, reverse=True)[:8]
            tot_us = sum(p.real_time for p in pis)
            print(f'  [{tag}] profiling: {len(pis)} nodes, total={tot_us/1000:.2f}ms', flush=True)
            for pi in tops:
                print(f'    {pi.node_name[:44]:44s} {pi.real_time/1000:.2f}ms  {pi.exec_type}', flush=True)
        except Exception as e:
            print(f'  [{tag}] profiling unsupported: {str(e)[:90]}', flush=True)
    t1 = time.time()
    for _ in range(R):
        ireq.infer()
    ms = (time.time() - t1) / R * 1000
    print(f'  [{tag}] b{b}: {ms:.2f}ms ({ms/b:.3f}ms/盘)', flush=True)
    return ms

# ---- 1) 基线（现行 fp16 trunk b32） ----
# 官方结论先行：NPU 插件不支持 ENABLE_PROFILING（编译即 NOT_FOUND），算子级剖析不可用
print('=== 1) baseline fp16 b32 ===', flush=True)
print('  [profiling] NPU plugin rejects ENABLE_PROFILING (NOT_FOUND) — per-op profiling unavailable', flush=True)
comp = core.compile_model(convert(TrunkOnly(net), 32), 'NPU', {'PERFORMANCE_HINT': 'LATENCY'})
bench(comp, 32)

# ---- 2) 原生批触发：哑第二输出 ----
print('=== 2) native-batch trigger (dummy 2nd output) ===', flush=True)
for b in (32, 64, 128):
    try:
        comp_n = core.compile_model(convert(TrunkDummyOut(net), b), 'NPU', {'PERFORMANCE_HINT': 'THROUGHPUT'})
        bench(comp_n, b, R=12, tag=f'native b{b}')
    except Exception as e:
        print(f'  native b{b}: FAIL {str(e)[:120]}', flush=True)

# ---- 3) Arc iGPU 对照（在 BN 折叠污染 net 之前测） ----
print('=== 3) Arc iGPU (GPU.0) ===', flush=True)
try:
    t0 = time.time()
    comp_g = core.compile_model(convert(TrunkOnly(net), 32), 'GPU.0')
    print(f'  compile GPU.0 b32: {time.time()-t0:.1f}s', flush=True)
    bench(comp_g, 32, tag='arc fp32 b32')
except Exception as e:
    print(f'  Arc FAIL: {str(e)[:140]}', flush=True)

# ---- 4) BN 折叠（会原地改 net 权重，必须最后跑） ----
print('=== 4) BN-folded trunk ===', flush=True)
try:
    tf = TrunkFolded(net)
    with torch.no_grad():
        ref = net.trunk(torch.zeros(2, C_IN, 10, 10)).numpy()
        got = tf(torch.zeros(2, C_IN, 10, 10)).numpy()
    print(f'  BN-fold 数学等价性 vs torch: maxAbs={np.abs(ref-got).max():.2e}', flush=True)
    comp_f = core.compile_model(convert(tf, 32), 'NPU', {'PERFORMANCE_HINT': 'LATENCY'})
    bench(comp_f, 32, tag='bnfold b32')
except Exception as e:
    print(f'  BN-fold FAIL: {str(e)[:140]}', flush=True)

print('DIAG2 DONE', flush=True)
