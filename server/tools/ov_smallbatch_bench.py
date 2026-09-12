# -*- coding: utf-8 -*-
# 小批量(b1/b8)主机开销解剖：各 batch 独立编译，fresh vs 原地张量 vs 钉核
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

def make_comp(b, pin):
    m = ov.convert_model(TrunkOnly(net).eval(),
                         example_input=torch.zeros(b, C_IN, 10, 10), input=(b, C_IN, 10, 10))
    props = {'PERFORMANCE_HINT': 'LATENCY'}
    if pin:
        props['ENABLE_CPU_PINNING'] = 'YES'
    return core.compile_model(m, 'NPU', props)

for b in (1, 8):
    for pin in (False, True):
        tag = f'b{b} pin={"ON " if pin else "OFF"}'
        try:
            comp = make_comp(b, pin)
            ireq = comp.create_infer_request()
            xb = np.zeros((b, C_IN, 10, 10), dtype=np.float32)
            # fresh 路径（桥现行）
            ireq.infer({0: xb})
            t1 = time.time()
            R = 100
            for _ in range(R):
                ireq.infer({0: xb})
            f = (time.time() - t1) / R * 1000
            # 原地路径（免 shadow copy）
            it = ireq.get_input_tensor()
            ot = ireq.get_output_tensor(0)
            it.data[:] = xb
            ireq.infer()
            t1 = time.time()
            for _ in range(R):
                it.data[:] = xb
                ireq.infer()
                _ = ot.data[:1]
            i = (time.time() - t1) / R * 1000
            gain = (f - i) / f * 100
            print(f'{tag}: fresh={f:.3f}ms  inplace={i:.3f}ms  主机开销占比={gain:.1f}%', flush=True)
        except Exception as e:
            print(f'{tag}: FAIL {str(e)[:130]}', flush=True)
print('BENCH DONE', flush=True)
