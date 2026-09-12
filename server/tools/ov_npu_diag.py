# -*- coding: utf-8 -*-
# NPU 利用率诊断：属性查询 + 批量扩展曲线 + AsyncInferQueue 并发吞吐 + TURBO 对照
# 目的：判定"并发回退批处理"（b1 串行命令列表）假设，并找最优并发参数
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

XML = os.path.join(SRV, 'models_ov', 'trunk_v2_b64.xml')
core = ov.Core()
core.set_property({'CACHE_DIR': os.path.join(SRV, 'models_ov', 'cache')})

# ---- 属性查询 ----
print('=== device properties ===', flush=True)
for name in ('NPU_MAX_TILES', 'NPU_DRIVER_VERSION', 'NPU_COMPILER_TYPE', 'NPU_PLATFORM',
             'OPTIMAL_NUMBER_OF_INFER_REQUESTS', 'NPU_ARCHITECTURE'):
    try:
        v = core.get_property('NPU', name)
        print(f'{name} = {v}', flush=True)
    except Exception as e:
        print(f'{name} read FAIL: {str(e)[:70]}', flush=True)

m = core.read_model(XML)
t0 = time.time()
comp_lat = core.compile_model(m, 'NPU', {'PERFORMANCE_HINT': 'LATENCY'})
print(f'compile LATENCY b64: {time.time()-t0:.1f}s', flush=True)
try:
    print('LATENCY optimal_nireq =', comp_lat.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS'), flush=True)
except Exception as e:
    print('LATENCY optimal_nireq read FAIL:', str(e)[:80], flush=True)
try:
    print('LATENCY num_streams =', comp_lat.get_property('NUM_STREAMS'), flush=True)
except Exception as e:
    print('num_streams read FAIL:', str(e)[:60], flush=True)

# ---- 扩展曲线（LATENCY hint，同步） ----
print('=== scaling curve (LATENCY, sync) ===', flush=True)
for b in (1, 4, 8, 16, 32, 64):
    if b == 64:
        comp_b, ireq_b = comp_lat, comp_lat.create_infer_request()
    else:
        try:
            mb = ov.convert_model(TrunkOnly(net).eval(), example_input=torch.zeros(b, C_IN, 10, 10),
                                  input=(b, C_IN, 10, 10))
            comp_b = core.compile_model(mb, 'NPU', {'PERFORMANCE_HINT': 'LATENCY'})
            ireq_b = comp_b.create_infer_request()
        except Exception as e:
            print(f'b{b}: compile FAIL {str(e)[:80]}', flush=True)
            continue
    xb = np.zeros((b, C_IN, 10, 10), dtype=np.float32)
    ireq_b.infer({0: xb})
    t1 = time.time()
    R = 20
    for _ in range(R):
        ireq_b.infer({0: xb})
    ms = (time.time() - t1) / R * 1000
    print(f'b{b}: {ms:.2f}ms ({ms/b:.3f}ms/盘)', flush=True)

# ---- THROUGHPUT hint b64 + AsyncInferQueue 并发 ----
print('=== async concurrency (THROUGHPUT) ===', flush=True)
t0 = time.time()
comp_thr = core.compile_model(m, 'NPU', {'PERFORMANCE_HINT': 'THROUGHPUT'})
print(f'compile THROUGHPUT b64: {time.time()-t0:.1f}s', flush=True)
try:
    print('THROUGHPUT optimal_nireq =', comp_thr.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS'), flush=True)
except Exception as e:
    print('optimal_nireq read FAIL:', str(e)[:80], flush=True)
ireq_thr = comp_thr.create_infer_request()
xb64 = np.zeros((64, C_IN, 10, 10), dtype=np.float32)
ireq_thr.infer({0: xb64})
t1 = time.time()
for _ in range(20):
    ireq_thr.infer({0: xb64})
thr_sync = (time.time() - t1) / 20 * 1000
print(f'THROUGHPUT sync b64: {thr_sync:.2f}ms ({thr_sync/64:.3f}ms/盘)', flush=True)

chunks = [np.zeros((64, C_IN, 10, 10), dtype=np.float32) for _ in range(8)]
for nreq in (2, 4, 8):
    try:
        q = ov.AsyncInferQueue(comp_thr, nreq)
        state = {'done': 0}

        def cb(infer_request, userdata):
            state['done'] += 1

        q.set_callback(cb)
        for i in range(8):
            q.start_async({0: chunks[i]}, i)
        q.wait_all()
        assert state['done'] == 8
        t1 = time.time()
        for _ in range(3):
            for i in range(8):
                q.start_async({0: chunks[i]}, i)
            q.wait_all()
        dt = (time.time() - t1) / 3
        print(f'async nireq={nreq}: 8×b64 = {dt*1000:.1f}ms → {8*64/dt:.0f} 盘/s', flush=True)
        del q
    except Exception as e:
        print(f'async nireq={nreq}: FAIL {str(e)[:140]}', flush=True)

# ---- TURBO 对照 ----
try:
    t0 = time.time()
    comp_turbo = core.compile_model(m, 'NPU', {'PERFORMANCE_HINT': 'THROUGHPUT', 'NPU_TURBO': 'YES'})
    ireq_t = comp_turbo.create_infer_request()
    ireq_t.infer({0: xb64})
    t1 = time.time()
    for _ in range(20):
        ireq_t.infer({0: xb64})
    tb = (time.time() - t1) / 20 * 1000
    print(f'TURBO sync b64: {tb:.2f}ms ({tb/64:.3f}ms/盘)', flush=True)
    q = ov.AsyncInferQueue(comp_turbo, 4)
    state2 = {'done': 0}

    def cb2(infer_request, userdata):
        state2['done'] += 1

    q.set_callback(cb2)
    for i in range(8):
        q.start_async({0: chunks[i]}, i)
    q.wait_all()
    t1 = time.time()
    for _ in range(3):
        for i in range(8):
            q.start_async({0: chunks[i]}, i)
        q.wait_all()
    dt = (time.time() - t1) / 3
    print(f'TURBO async nireq=4: 8×b64 = {dt*1000:.1f}ms → {8*64/dt:.0f} 盘/s', flush=True)
except Exception as e:
    print(f'TURBO FAIL: {str(e)[:140]}', flush=True)

print('DIAG DONE', flush=True)
