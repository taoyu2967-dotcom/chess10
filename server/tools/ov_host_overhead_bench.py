# -*- coding: utf-8 -*-
# 主机开销解剖：fresh 数组 vs 原地张量（免 shadow copy）vs 钉核 —— 单步延迟里拷贝/调度占多深
import os, sys, time

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
import torch.nn as nn
import openvino as ov

HERE = os.path.dirname(os.path.abspath(__file__))
SRV = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'ov_train')))
from az_model import build_net, C_IN  # noqa: E402

net = build_net(os.path.join(SRV, 'weights_ov.bin')).eval()

class TrunkOnly(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net
    def forward(self, x):
        return self.net.trunk(x)

core = ov.Core()
core.set_property({'CACHE_DIR': os.path.join(SRV, 'models_ov', 'cache')})

def make_comp(pin):
    m = ov.convert_model(TrunkOnly(net).eval(),
                         example_input=torch.zeros(32, C_IN, 10, 10), input=(32, C_IN, 10, 10))
    props = {'PERFORMANCE_HINT': 'LATENCY'}
    if pin:
        props['ENABLE_CPU_PINNING'] = 'YES'
    return core.compile_model(m, 'NPU', props)

def bench_fresh(comp, b, R=60):
    ireq = comp.create_infer_request()
    xb = np.zeros((b, C_IN, 10, 10), dtype=np.float32)
    ireq.infer({0: xb})
    t1 = time.time()
    for _ in range(R):
        ireq.infer({0: xb})   # 每次传 fresh 字典（桥的现行路径）
    return (time.time() - t1) / R * 1000

def bench_inplace(comp, b, R=60):
    """原地张量：把插件自管的输入缓冲直接写满（官方免 shadow-copy 路径）+ 输出零拷贝读"""
    ireq = comp.create_infer_request()
    it = ireq.get_input_tensor()
    xb = np.zeros((b, C_IN, 10, 10), dtype=np.float32)
    it.data[:] = xb
    ot = ireq.get_output_tensor(0)
    ireq.infer()
    t1 = time.time()
    for _ in range(R):
        it.data[:] = xb   # 直接写设备映射缓冲
        ireq.infer()      # 无字典绑定 → 无 shadow copy
        _ = ot.data[:2]   # 零拷贝读（不 copy 全量）
    return (time.time() - t1) / R * 1000

for pin in (False, True):
    tag = 'pinning=ON ' if pin else 'pinning=OFF'
    try:
        comp = make_comp(pin)
    except Exception as e:
        print(f'{tag}: compile FAIL {str(e)[:100]}', flush=True)
        continue
    for b in (1, 8, 32):
        try:
            f = bench_fresh(comp, b)
            i = bench_inplace(comp, b)
            gain = (f - i) / f * 100
            print(f'{tag} b{b}: fresh={f:.3f}ms  inplace={i:.3f}ms  主机开销占比={gain:.1f}%', flush=True)
        except Exception as e:
            print(f'{tag} b{b}: FAIL {str(e)[:120]}', flush=True)

# ---- 端到端单轮影响估算：MCTS 一步的 flush 延迟 ----
print('=== 单轮(一步思考)延迟影响估算 ===', flush=True)
try:
    comp = make_comp(True)
    ireq = comp.create_infer_request()
    it = ireq.get_input_tensor()
    it.data[:] = np.zeros((32, C_IN, 10, 10), dtype=np.float32)
    ireq.infer()
    t1 = time.time()
    for _ in range(60):
        it.data[:] = np.zeros((32, C_IN, 10, 10), dtype=np.float32)
        ireq.infer()
    per32 = (time.time() - t1) / 60 * 1000
    n_flush = 12   # e2e 实测：3000 节点 ≈ 12 次批量评估
    print(f'优化后单次 flush(b32)≈{per32:.1f}ms → 一步 {n_flush} flush ≈ {per32*n_flush:.0f}ms（现行路径≈{44*n_flush:.0f}ms）', flush=True)
except Exception as e:
    print(f'估算 FAIL: {str(e)[:120]}', flush=True)
print('BENCH DONE', flush=True)
